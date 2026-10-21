# Copyright 2026 The RPent Authors.
"""Durable public RGB or RGB-D observations, indexed by simulator control step.

The writer owns an append-only index and immutable sensor files. It never reads
task predicates or simulator object state. Environment adapters supply aligned
sensor captures and explicitly public proprioception at each control boundary.
"""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np

from rpent.tools.correlation import current_root_tool_call_id

TRACE_MODE = "full_rgbd"
TRACE_MODES = frozenset({"full_rgb", TRACE_MODE})
TRACE_INDEX = "public/trajectory.jsonl"
TRACE_META = "public/trajectory_meta.json"
PUBLIC_ROBOT_STATE = {
    "eef_position": 3,
    "eef_quaternion": 4,
    "gripper_qpos": 2,
    "base_position": 3,
    "base_quaternion": 4,
}
# Optional sensor channels are explicitly named; arbitrary observation keys are
# never copied. A mobile robot's joint vector may mix rotary and slide joints.
OPTIONAL_ROBOT_STATE = {
    "joint_position": (None, "radian (revolute), meter (prismatic)", "joint"),
    "joint_velocity": (None, "radian/second (revolute), meter/second (prismatic)", "joint"),
    "joint_acceleration": (None, "radian/second^2 (revolute), meter/second^2 (prismatic)", "joint"),
    "joint_position_cos": (None, "dimensionless", "joint"),
    "joint_position_sin": (None, "dimensionless", "joint"),
    "gripper_qvel": (2, "meter/second", "finger_slide_joint"),
    "base_linear_velocity": (3, "meter/second", "environment_observation_untransformed"),
    "base_angular_velocity": (3, "radian/second", "environment_observation_untransformed"),
    "eef_linear_velocity": (3, "meter/second", "environment_observation_untransformed"),
    "eef_angular_velocity": (3, "radian/second", "environment_observation_untransformed"),
    "base_to_eef_position": (3, "meter", "robot_base"),
    "base_to_eef_quaternion": (4, "xyzw", "robot_base"),
    "eef_site_quaternion": (4, "xyzw", "world_eef_site"),
    "base_to_eef_site_quaternion": (4, "xyzw", "robot_base_eef_site"),
}
_REQUIRED_STATE_METADATA = {
    "eef_position": {"unit": "meter", "frame": "world_eef_site"},
    "eef_quaternion": {"unit": "xyzw", "frame": "world_eef_body"},
    "gripper_qpos": {"unit": "meter", "frame": "finger_slide_joint"},
    "base_position": {"unit": "meter", "frame": "world"},
    "base_quaternion": {"unit": "xyzw", "frame": "world"},
}
_CAPABILITY: ContextVar[tuple[str | None, int | None]] = ContextVar(
    "rpent_trace_capability", default=(None, None)
)


class TraceError(RuntimeError):
    """A required sensor trace could not be durably recorded or verified."""


@contextmanager
def trace_capability_scope(capability_id: str, sequence: int):
    token = _CAPABILITY.set((capability_id, sequence))
    try:
        yield
    finally:
        _CAPABILITY.reset(token)


def trace_mode() -> str:
    mode = os.environ.get("RPENT_TRACE_MODE", "off")
    if mode not in {"off", *TRACE_MODES}:
        raise ValueError("RPENT_TRACE_MODE must be off, full_rgb or full_rgbd")
    return mode


def current_trace_context() -> dict[str, Any]:
    """Carry action provenance across an environment RPC boundary."""
    capability_id, sequence = _CAPABILITY.get()
    context = {}
    try:
        from rpent.planner.recording import current_decision_context
    except ImportError:
        pass
    else:
        context = current_decision_context()
    return {
        "root_tool_call_id": current_root_tool_call_id(),
        "capability_id": capability_id,
        "capability_sequence": sequence,
        **context,
    }


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise TraceError(f"{name} must have finite shape {shape}")
    return array


def public_robot_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Select the public sensor fields; never serialize arbitrary observations."""
    public = {
        name: _finite(state.get(name), (size,), name).tolist()
        for name, size in PUBLIC_ROBOT_STATE.items()
    }
    for name, (size, _unit, _frame) in OPTIONAL_ROBOT_STATE.items():
        if name not in state:
            continue
        value = np.asarray(state[name], dtype=np.float64)
        if value.ndim != 1 or not 0 < len(value) <= 256:
            raise TraceError(f"{name} must be a nonempty finite sensor vector")
        public[name] = _finite(value, (len(value) if size is None else size,), name).tolist()
    return public


def _robot_state_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(_REQUIRED_STATE_METADATA)
    metadata.update({
        name: {"unit": unit, "frame": frame}
        for name, (_size, unit, frame) in OPTIONAL_ROBOT_STATE.items() if name in state
    })
    return metadata


def camera_calibration(metadata: Mapping[str, Any], state: Mapping[str, Any]):
    """Convert fresh robosuite optical-camera metadata to public calibration.

    robosuite's intrinsics/extrinsics use top-left image coordinates and the
    optical frame (x right, y down, z forward). The environment adapter supplies
    RGB and metric depth flipped vertically from the native OpenGL render.
    """
    intrinsic = _finite(metadata.get("intrinsic"), (3, 3), "intrinsic")
    world_from_camera = _finite(
        metadata.get("extrinsic_cam2world"), (4, 4), "extrinsic_cam2world"
    )
    position = _finite(state.get("base_position"), (3,), "base_position")
    quaternion = _finite(state.get("base_quaternion"), (4,), "base_quaternion")
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        raise TraceError("base quaternion must have nonzero norm")
    x, y, z, w = quaternion / norm
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    world_from_base[:3, 3] = position
    calibration = {
        "intrinsic": intrinsic.tolist(),
        "extrinsic_camera_to_world": world_from_camera.tolist(),
        "extrinsic_camera_to_base": (
            np.linalg.inv(world_from_base) @ world_from_camera
        ).tolist(),
        "image_origin": "top_left",
        "camera_axes": "x_right_y_down_z_forward",
        "extrinsic_convention": "column vectors",
        "depth_semantics": "optical_axis_z",
        "camera_frame": str(metadata["camera_name"]),
        "target_frame": "robot_base",
        "depth_near_m": float(metadata["depth_near"]),
        "depth_far_m": float(metadata["depth_far"]),
    }
    _validate_calibration(calibration)
    return calibration


def _validate_calibration(calibration: Mapping[str, Any]) -> None:
    intrinsic = _finite(calibration.get("intrinsic"), (3, 3), "intrinsic")
    if abs(np.linalg.det(intrinsic)) <= 1e-12:
        raise TraceError("camera intrinsic must be invertible")
    for name in ("extrinsic_camera_to_world", "extrinsic_camera_to_base"):
        transform = _finite(calibration.get(name), (4, 4), name)
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise TraceError(f"{name} must be a homogeneous transform")
    if calibration.get("image_origin") != "top_left" or calibration.get("camera_axes") != "x_right_y_down_z_forward":
        raise TraceError("camera calibration orientation does not match archived pixels")
    for name, expected in (("depth_semantics", "optical_axis_z"),
                           ("extrinsic_convention", "column vectors"),
                           ("target_frame", "robot_base")):
        if calibration.get(name) != expected:
            raise TraceError(f"camera calibration {name} must be {expected!r}")
    near, far = float(calibration["depth_near_m"]), float(calibration["depth_far_m"])
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise TraceError("camera clipping distances must be finite positive metres")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish a completed file atomically without per-sensor fsync."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


class RGBDTrajectoryWriter:
    """Snapshot synchronously; encode and write on one bounded worker.

    ``append`` has a single producer (the simulator control thread). Its memory
    budget includes queued and currently encoding pixel arrays. Encoding needs
    at most one additional frame's compressed payload. ``drain`` publishes all
    accepted rows; ``finish`` additionally seals the archive. Neither recording
    nor final inspection computes file hashes or rereads archived pixels.
    """

    _STOP = object()

    def __init__(
        self,
        output_dir: str | Path,
        *,
        cameras: tuple[str, ...] = ("agentview", "navview", "wrist"),
        queue_size: int = 4,
        max_pending_bytes: int = 64 * 1024**2,
        flush_every: int = 8,
        mode: str = TRACE_MODE,
        initialization: Mapping[str, Any] | None = None,
    ):
        self.root = Path(output_dir).resolve()
        if mode not in TRACE_MODES:
            raise ValueError(f"unsupported trajectory mode: {mode!r}")
        self.mode = mode
        self.cameras = cameras
        if not cameras or len(set(cameras)) != len(cameras) or not all(
            isinstance(camera, str) and camera.isidentifier() for camera in cameras
        ):
            raise ValueError("trace cameras must be unique simple camera identifiers")
        if any(type(value) is not int or value <= 0 for value in (
            queue_size, max_pending_bytes, flush_every
        )):
            raise ValueError("trace queue, memory budget and flush batch must be positive integers")
        self.last_env_step: int | None = None
        self.last_written_env_step: int | None = None
        self.error: str | None = None
        self.max_pending_bytes = max_pending_bytes
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.backpressure_seconds = 0.0
        self.snapshot_seconds = 0.0
        self.encoding_seconds = 0.0
        self.writing_seconds = 0.0
        self._flush_every = flush_every
        self._unflushed_rows = 0
        self._closed = False
        self._accepting = True
        self._condition = threading.Condition(threading.RLock())
        self._io_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        public = self.root / "public"
        public.mkdir(parents=True, exist_ok=True)
        self.index = self.root / TRACE_INDEX
        self.meta_path = self.root / TRACE_META
        if self.index.exists() or self.meta_path.exists() or (public / "trajectory").exists():
            raise TraceError("trajectory already exists; use a fresh episode output directory")
        self._meta = {
            "schema_version": 1,
            "mode": self.mode,
            "required": True,
            "cameras": list(cameras),
            "index": TRACE_INDEX,
            "sensors": ["rgb", "depth"] if self.mode == TRACE_MODE else ["rgb"],
            "initial_state": True,
            "stride": 1,
            "writer_mode": "bounded_async",
            "queue_size": queue_size,
            "max_pending_bytes": max_pending_bytes,
            "flush_every": flush_every,
            "trace_complete": False,
            "complete": False,
            "error": None,
        }
        if self.mode == TRACE_MODE:
            self._meta.update(depth_encoding="npz_compressed_float32",
                              depth_key="depth_m", depth_unit="meter")
        if initialization is not None:
            self._meta["initialization"] = dict(initialization)
        _atomic_bytes(self.meta_path, _json_bytes(self._meta))
        (public / "trajectory").mkdir()
        self._index_handle = self.index.open("xb")
        self._worker = threading.Thread(
            target=self._run, name=f"rgbd-writer-{self.root.name}", daemon=True
        )
        self._worker.start()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self.error is None:
                self.error = f"{type(error).__name__}: {error}"
            self._meta.update(error=self.error, trace_complete=False, complete=False)
            try:
                _atomic_bytes(self.meta_path, _json_bytes(self._meta))
            except OSError:
                # The original exception remains visible even when its marker
                # cannot be written. No successful finish marker exists.
                pass
            self._condition.notify_all()

    def preflight(self) -> None:
        if self.error is not None:
            raise TraceError(f"trajectory recording previously failed: {self.error}")
        if not self._accepting:
            raise TraceError("trajectory recording has already finished")

    def append(
        self,
        *,
        env_step: int,
        action: Mapping[str, Any] | None,
        robot_state: Mapping[str, Any],
        captures: Mapping[str, Mapping[str, Any]],
        correlation: Mapping[str, Any] | None = None,
    ) -> None:
        """Own the exact boundary's arrays and context before returning."""
        reserved = 0
        snapshot_started = time.monotonic()
        waiting_seconds = 0.0
        try:
            self.preflight()
            expected = 0 if self.last_env_step is None else self.last_env_step + 1
            if type(env_step) is not int or env_step != expected or (env_step == 0) != (action is None):
                raise TraceError(f"noncontiguous trajectory: expected env_step {expected}")
            if set(captures) != set(self.cameras):
                raise TraceError("trajectory requires every configured camera")
            state = public_robot_state(robot_state)
            action_record = None
            if action is not None:
                values = np.asarray(action["env_action"], dtype=np.float64)
                if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                    raise TraceError("executed action must be a finite vector")
                if (action["env_steps_before"], action["env_steps_after"]) != (env_step-1, env_step):
                    raise TraceError("executed action does not align with observation")
                action_record = {
                    "env_steps_before": env_step-1,
                    "env_steps_after": env_step,
                    "env_action": values.tolist(),
                }
            sensors = {}
            total_bytes = 0
            for camera in self.cameras:
                capture = captures[camera]
                rgb = np.asarray(capture["rgb"])
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise TraceError("RGB must be uint8 HxWx3")
                sensors[camera] = {"rgb": rgb, "image_origin": "top_left"}
                total_bytes += rgb.nbytes
                if self.mode == TRACE_MODE:
                    depth = np.asarray(capture["depth_m"])
                    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or np.any(depth < 0):
                        raise TraceError("depth must be aligned, finite, nonnegative HxW metres")
                    metadata = capture["metadata"]
                    if (metadata.get("height"), metadata.get("width")) != rgb.shape[:2]:
                        raise TraceError("camera calibration resolution does not match sensor pixels")
                    sensors[camera].update(depth_m=depth,
                                          calibration=camera_calibration(metadata, state))
                    total_bytes += depth.size * np.dtype(np.float32).itemsize
            if total_bytes > self.max_pending_bytes:
                raise TraceError(
                    f"one {self.mode} snapshot needs {total_bytes} bytes, exceeds writer budget {self.max_pending_bytes}"
                )
            started = time.monotonic()
            with self._condition:
                while self.pending_bytes + total_bytes > self.max_pending_bytes:
                    self.preflight()
                    self._condition.wait()
                self.preflight()
                self.pending_bytes += total_bytes
                reserved = total_bytes
                self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
            waiting_seconds += time.monotonic() - started
            # Copy before enqueuing; neither a simulator observation buffer nor
            # an RPC-owned array may be reused by the background encoder.
            for capture in sensors.values():
                capture["rgb"] = np.array(capture["rgb"], copy=True, order="C")
                if self.mode == TRACE_MODE:
                    capture["depth_m"] = np.array(capture["depth_m"], dtype=np.float32, copy=True, order="C")
                    if not np.isfinite(capture["depth_m"]).all():
                        raise TraceError("metric depth overflows float32 archival encoding")
            context = dict(correlation) if correlation is not None else current_trace_context()
            snapshot = {
                "schema_version": 1, "env_step": env_step,
                "observation_id": f"env-{env_step:06d}",
                "episode_id": context.get("episode_id", self.root.name),
                "decision_id": context.get("decision_id"),
                "actor": context.get("actor", "planner") if action is not None else None,
                "action": action_record, "robot_state": state,
                "robot_state_metadata": _robot_state_metadata(state),
                "correlation": context,
                "cameras": sensors,
            }
            started = time.monotonic()
            self._queue.put((snapshot, total_bytes))  # bounded backpressure; never drop
            waiting_seconds += time.monotonic() - started
            reserved = 0  # The worker now owns and releases this reservation.
            self.backpressure_seconds += waiting_seconds
            self.snapshot_seconds += time.monotonic() - snapshot_started - waiting_seconds
            self.last_env_step = env_step
            self.preflight()  # Surface a concurrent encoder failure promptly.
        except Exception as exc:
            if reserved:
                with self._condition:
                    self.pending_bytes -= reserved
                    self._condition.notify_all()
            self.fail(exc)
            raise TraceError(f"failed to record env_step {env_step}: {exc}") from exc

    def _run(self) -> None:
        write_failed = False
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                snapshot, size = item
                if not write_failed:
                    try:
                        with self._io_lock:
                            self._write_snapshot(snapshot)
                    except Exception as exc:
                        write_failed = True
                        self.fail(exc)
                with self._condition:
                    self.pending_bytes -= size
                    self._condition.notify_all()
            finally:
                self._queue.task_done()

    def _write_snapshot(self, row: dict[str, Any]) -> None:
        write_started = time.monotonic()
        previous_encoding_seconds = self.encoding_seconds
        env_step = row["env_step"]
        directory = self.root / "public" / "trajectory" / f"{env_step:06d}"
        directory.mkdir()
        camera_records = {}
        for camera, capture in row["cameras"].items():
            started = time.monotonic()
            rgb = capture["rgb"]
            rgb_payload = imageio.imwrite("<bytes>", rgb, format="png")
            payloads = [("rgb", "png", rgb_payload)]
            camera_record = {
                "height": int(rgb.shape[0]), "width": int(rgb.shape[1]),
                "capture_id": f"{row['observation_id']}/{camera}",
                "image_origin": "top_left",
            }
            if self.mode == TRACE_MODE:
                buffer = io.BytesIO()
                np.savez_compressed(buffer, depth_m=capture["depth_m"])
                payloads.append(("depth", "npz", buffer.getvalue()))
                camera_record.update(depth_unit="meter", depth_dtype="float32",
                                     depth_key="depth_m", calibration=capture["calibration"])
            self.encoding_seconds += time.monotonic() - started
            for kind, suffix, payload in payloads:
                destination = directory / f"{camera}_{kind}.{suffix}"
                _atomic_bytes(destination, payload)
                camera_record[kind] = destination.relative_to(self.root).as_posix()
                camera_record[f"{kind}_bytes"] = len(payload)
            camera_records[camera] = camera_record
        row["cameras"] = camera_records
        self._index_handle.write(_json_bytes(row))
        self._unflushed_rows += 1
        if self._unflushed_rows >= self._flush_every:
            self._flush_index()
        self.last_written_env_step = env_step
        self.writing_seconds += (
            time.monotonic() - write_started - (self.encoding_seconds - previous_encoding_seconds)
        )

    def _flush_index(self) -> None:
        self._index_handle.flush()
        if self._unflushed_rows:
            os.fsync(self._index_handle.fileno())
            self._unflushed_rows = 0

    def drain(self) -> None:
        """Wait for all accepted snapshots and flush their completed index rows."""
        drain_started = time.perf_counter()
        self._queue.join()
        from rpent.utils.execution import timing
        timing("writer_drain", drain_started)
        if not self._closed:
            try:
                with self._io_lock:
                    started = time.monotonic()
                    self._flush_index()
                    self.writing_seconds += time.monotonic() - started
            except Exception as exc:
                self.fail(exc)
        if self.error is not None:
            raise TraceError(f"trajectory recording previously failed: {self.error}")

    def finish(self, expected_env_steps: int | None = None) -> dict[str, Any]:
        """Seal only after drain; failures stay incomplete and raise to caller."""
        if not self._closed:
            self._accepting = False
            try:
                self.drain()
            except TraceError:
                pass  # Always stop/join the worker and publish its failure state.
            self._queue.put(self._STOP)
            self._worker.join()
            try:
                self._index_handle.close()
            except OSError as exc:
                self.fail(exc)
            self._closed = True
            expected = self.last_env_step if expected_env_steps is None else expected_env_steps
            if expected is None or self.last_written_env_step != expected:
                self.fail(TraceError(
                    f"expected final env_step {expected}, recorded {self.last_written_env_step}"
                ))
            self._meta.update(
                complete=self.error is None, trace_complete=self.error is None,
                env_steps=expected,
                observation_count=0 if self.last_written_env_step is None else self.last_written_env_step + 1,
                accepted_observation_count=0 if self.last_env_step is None else self.last_env_step + 1,
                peak_pending_bytes=self.peak_pending_bytes,
                backpressure_seconds=self.backpressure_seconds,
                snapshot_seconds=self.snapshot_seconds,
                encoding_seconds=self.encoding_seconds,
                writing_seconds=self.writing_seconds,
            )
            try:
                _atomic_bytes(self.meta_path, _json_bytes(self._meta))
            except OSError as exc:
                self.fail(exc)
        if self.error is not None:
            raise TraceError(f"trajectory recording previously failed: {self.error}")
        coverage = inspect_trajectory(
            self.root, self.last_env_step if expected_env_steps is None else expected_env_steps
        )
        if coverage is None or not coverage["complete"]:
            error = TraceError("; ".join(coverage["errors"]) if coverage else "trajectory is unavailable")
            self.fail(error)
            raise error
        return coverage

    close = finish


def inspect_trajectory(output_dir: str | Path, expected_env_steps: int | None) -> dict[str, Any] | None:
    """Check completion, contiguous index and file metadata without reading pixels.

    Pixel decoding is deferred to selected-frame readers. This is an integrity
    check of recorded coverage, not a checksum audit of the entire archive.
    """
    root = Path(output_dir).resolve()
    meta_path = root / TRACE_META
    index = root / TRACE_INDEX
    required = (
        trace_mode() in TRACE_MODES or meta_path.exists() or index.exists()
        or (root / "public/trajectory").exists()
    )
    if not required:
        return None
    errors: list[str] = []
    count = 0
    cameras: list[str] = []
    mode = trace_mode()
    initialization = None
    try:
        meta = json.loads(meta_path.read_text())
        mode = meta.get("mode", TRACE_MODE)
        if mode not in TRACE_MODES:
            raise TraceError(f"unsupported trajectory mode: {mode!r}")
        initialization = meta.get("initialization")
        required = required or bool(meta.get("required"))
        cameras = meta["cameras"]
        if meta.get("schema_version") != 1 or not cameras or meta.get("stride") != 1:
            raise TraceError("invalid trajectory metadata")
        if meta.get("error"):
            errors.append(str(meta["error"]))
        if meta.get("writer_mode") == "bounded_async" and meta.get("trace_complete") is not True:
            errors.append("trajectory writer has not completed drain and finish")
        with index.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("schema_version") != 1 or row.get("env_step") != count:
                    raise TraceError(f"missing or unordered env_step {count}")
                if meta.get("platform") == "robotwin":
                    from robots.robotwin.trace import public_robot_state as robotwin_public_state
                    robotwin_public_state(row["robot_state"])
                else:
                    public_robot_state(row["robot_state"])
                action = row.get("action")
                if count == 0:
                    if action is not None:
                        raise TraceError("initial trajectory state must not have an action")
                elif not isinstance(action, dict) or (action.get("env_steps_before"), action.get("env_steps_after")) != (count-1, count):
                    raise TraceError(f"invalid action alignment at env_step {count}")
                else:
                    values = np.asarray(action.get("env_action"), dtype=np.float64)
                    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                        raise TraceError(f"invalid action vector at env_step {count}")
                if set(row["cameras"]) != set(cameras):
                    raise TraceError(f"missing camera at env_step {count}")
                for camera in cameras:
                    capture = row["cameras"][camera]
                    if mode == TRACE_MODE:
                        _validate_calibration(capture["calibration"])
                        if capture.get("depth_unit") != "meter" or capture.get("depth_dtype") != "float32" or capture.get("depth_key") != "depth_m":
                            raise TraceError(f"invalid depth encoding at env_step {count}")
                    elif capture.get("image_origin") != "top_left":
                        raise TraceError(f"invalid RGB image origin at env_step {count}")
                    if not all(isinstance(capture.get(name), int) and capture[name] > 0 for name in ("height", "width")):
                        raise TraceError(f"invalid image dimensions at env_step {count}")
                    for kind in ("rgb", "depth") if mode == TRACE_MODE else ("rgb",):
                        path = (root / capture[kind]).resolve()
                        if not path.is_relative_to(root / "public" / "trajectory"):
                            raise TraceError("trajectory sensor path escapes the public archive")
                        if not path.is_file() or path.stat().st_size != capture[f"{kind}_bytes"]:
                            raise TraceError(f"missing or truncated {kind} for {camera} at env_step {count}")
                count += 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError, TraceError) as exc:
        errors.append(str(exc))
    if expected_env_steps is None:
        errors.append("final environment step count is unavailable")
    elif count != expected_env_steps + 1:
        errors.append(f"expected {expected_env_steps + 1} observations, verified {count}")
    result = {
        "schema_version": 1, "mode": mode, "required": required,
        "complete": not errors, "trace_complete": not errors, "index": TRACE_INDEX,
        "metadata": TRACE_META, "cameras": cameras,
        "observation_count": count,
        "env_steps": expected_env_steps, "initial_state": True, "stride": 1,
        "errors": errors,
    }
    if mode == TRACE_MODE:
        result.update(depth_encoding="npz_compressed_float32", depth_key="depth_m", depth_unit="meter")
    if initialization is not None:
        result["initialization"] = initialization
    return result
