# Copyright 2026 The RPent Authors.
"""Small public-episode reader and historical RGB / RGB-D perception tools.

Read pool episode directories directly, without the retired proposal/manifest
adapters. Depth and calibration come from the selected control step. Optional
SAM diagnostics write to the supplied analysis workspace.
"""

from __future__ import annotations

import base64
import bisect
import io
import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any

from meta_harness._files import (
    atomic_json,
    file_sha256,
    read_json,
    require_contained_path,
)

DEFAULT_CAMERAS = ("agentview", "navview", "wrist")
CAMERAS = (*DEFAULT_CAMERAS, "head", "left_wrist", "right_wrist")
MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 4 * 1024 * 1024


def _sanitize(value: Any, *, max_string: int = 800, max_items: int = 40) -> Any:
    """Bound model-facing output and remove embedded image byte payloads."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            if key.startswith("_image") or key in {"image_bytes", "pixels"}:
                cleaned[key] = "<omitted binary image payload>"
            else:
                cleaned[key] = _sanitize(
                    child, max_string=max_string, max_items=max_items
                )
        return cleaned
    if isinstance(value, list):
        cleaned = [
            _sanitize(child, max_string=max_string, max_items=max_items)
            for child in value[:max_items]
        ]
        if len(value) > max_items:
            cleaned.append(f"<omitted {len(value) - max_items} items>")
        return cleaned
    if isinstance(value, str) and len(value) > max_string:
        return value[:max_string] + f"… <omitted {len(value) - max_string} chars>"
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any) -> dict[str, Any]:
    return {"type": "text", "text": json.dumps(value, ensure_ascii=False, allow_nan=False)}


def _image(payload: bytes, label: str) -> list[dict[str, Any]]:
    if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n") or payload[12:16] != b"IHDR":
        raise ValueError("historical image must be PNG")
    width, height = struct.unpack(">II", payload[16:24])
    if not width or not height or width * height > MAX_PIXELS:
        raise ValueError("historical image exceeds the image size limit")
    return [{"type": "text", "text": label},
            {"type": "image", "mimeType": "image/png",
             "data": base64.b64encode(payload).decode("ascii")}]


def _png(array) -> bytes:
    import imageio.v2 as imageio
    stream = io.BytesIO()
    imageio.imwrite(stream, array, format="png")
    return stream.getvalue()


def _thumbnail(payload: bytes) -> bytes:
    from PIL import Image
    with Image.open(io.BytesIO(payload)) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError("RGB image exceeds the historical image size limit")
        source.thumbnail((360, 270))
        stream = io.BytesIO()
        source.save(stream, format="PNG")
        return stream.getvalue()


class TrajectoryReader:
    """Index offsets for one episode, then seek only the requested control steps."""


    @classmethod
    def from_episode(cls, episode: Path, *, workspace: Path | None = None, sam_endpoint: str | None = None):
        """Read the same trace format from a schema-2 pool episode directly."""
        reader = cls.__new__(cls)
        reader.episode = Path(episode).resolve()
        reader.workspace = Path(workspace).resolve() if workspace is not None else reader.episode
        reader.context = {"history": {"sam3_endpoint": sam_endpoint}}
        reader.public = reader.episode / "public"
        reader.path = reader.public / "trajectory.jsonl"
        result = read_json(reader.episode / "result.json")
        reader.identity = {"source": "pool", "episode": reader.episode.parent.name,
                           "task": result.get("task", {}), "harness": result.get("harness", {}),
                           "agent": result.get("agent", {}),
                           "scenario": result.get("task", {}).get("seed"),
                           "repeat_index": result.get("task", {}).get("repeat_index")}
        from meta_harness.history import identity
        recorded = identity(reader.episode, result)
        reader.identity.update(recorded, episode=recorded["episode_id"],
                               scenario=recorded["seed"], repeat_index=recorded["repeat"])
        reader._index_result(result)
        return reader

    def _index_result(self, result):
        self.result = result
        self.expected_end = result.get("outcome", {}).get("env_steps")
        if type(self.expected_end) is not int:
            self.expected_end = result.get("metrics", {}).get("env_steps")
        if type(self.expected_end) is not int:
            self.expected_end = None
        self.metadata = result.get("public", {}).get("trajectory_metadata")
        if not isinstance(self.metadata, dict):
            metadata_path = require_contained_path(self.public / "trajectory_meta.json", anchor=self.public)
            self.metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        self.mode = self.metadata.get("mode", "full_rgbd")
        if self.mode not in {"full_rgb", "full_rgbd"}:
            raise ValueError(f"unsupported trajectory mode: {self.mode!r}")
        self.cameras = tuple(self.metadata.get("cameras", DEFAULT_CAMERAS))
        if (not self.cameras or len(set(self.cameras)) != len(self.cameras)
                or any(camera not in CAMERAS for camera in self.cameras)):
            raise ValueError("invalid recorded trajectory cameras")
        self.steps: list[int] = []
        self.offsets: list[int] = []
        self.warnings: list[str] = []
        if self.path.is_file():
            with self.path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        if not line.endswith(b"\n") and not stream.read(1):
                            self.warnings.append("Interrupted final trajectory row is unavailable.")
                            break
                        raise ValueError("malformed public trajectory JSONL") from None
                    if not isinstance(row, dict) or row.get("schema_version") != 1:
                        raise ValueError("unsupported public trajectory schema")
                    step = _integer(row.get("env_step"), "trajectory env_step")
                    if self.steps and step <= self.steps[-1]:
                        raise ValueError("trajectory env_steps must be unique and increasing")
                    self.steps.append(step)
                    self.offsets.append(offset)
        else:
            self.warnings.append("No control-step trajectory was recorded.")
        if self.metadata.get("complete") is False:
            self.warnings.append("Recorder reports an incomplete trajectory: " + str(self.metadata.get("errors", [])))
        if self.metadata.get("error"):
            self.warnings.append("Recorder reports a trajectory error: " + str(self.metadata["error"]))


    def coverage(self, *, selected_complete: bool | None = None, all_assets_checked: bool = False) -> dict[str, Any]:
        contiguous = bool(self.steps) and self.steps == list(range(self.steps[-1] + 1))
        complete = contiguous and self.expected_end is not None and self.steps[-1] == self.expected_end
        return {
            "kind": "control_steps",
            "mode": self.mode,
            "required_sensors": ["rgb", "depth"] if self.mode == "full_rgbd" else ["rgb"],
            "complete": complete and all_assets_checked and selected_complete is True and not self.warnings,
            "control_step_coverage_complete": complete,
            "recorded_complete": self.metadata.get("complete"),
            "verified_selected_frame_complete": selected_complete,
            "all_recorded_assets_checked": all_assets_checked,
            "recorded_frames": len(self.steps),
            "first_env_step": self.steps[0] if self.steps else None,
            "last_env_step": self.steps[-1] if self.steps else None,
            "expected_final_env_step": self.expected_end,
            "contiguous_from_initial": contiguous,
            "asset_validation_scope": "selected frames; paths and decoding checked; legacy hashes verified when present",
            "warnings": self.warnings,
            "recorder_metadata": self.metadata,
        }

    def _warn(self, message: str) -> None:
        if message not in self.warnings and len(self.warnings) < 40:
            self.warnings.append(message)


    def frame_identity(self, row: dict[str, Any]) -> dict[str, Any]:
        identity = {**self.identity, "env_step": row.get("env_step"),
                    "observation_step": row.get("observation_step")}
        return identity

    def row(self, *, env_step: int | None = None, observation_step: int | None = None) -> dict[str, Any]:
        if (env_step is None) == (observation_step is None):
            raise ValueError("select exactly one env_step")
        if observation_step is not None:
            raise ValueError("observation_step is not a control-step selector; use env_step")
        _integer(env_step, "env_step")
        index = bisect.bisect_left(self.steps, env_step)
        if index == len(self.steps) or self.steps[index] != env_step:
            raise ValueError("env_step is not recorded; use history_timeline to inspect coverage")
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())

    def asset(self, camera: dict[str, Any], kind: str) -> Path | None:
        reference = camera.get(kind)
        if reference is None:
            if self.path.is_file() and not (kind == "depth" and self.mode == "full_rgb"):
                self._warn(f"Selected frame has no recorded {kind} reference.")
            return None
        if not isinstance(reference, str):
            raise ValueError(f"invalid {kind} path")
        relative = Path(reference)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "public" or ".." in relative.parts:
            raise ValueError(f"{kind} must reference only this episode's public tree")
        path = require_contained_path(
            self.public.joinpath(*relative.parts[1:]), anchor=self.public, label=kind)
        if not path.is_file():
            self._warn(f"Selected frame {kind} file is missing: {reference}")
            return None
        if path.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError(f"{kind} exceeds the historical asset size limit")
        expected = camera.get(f"{kind}_sha256")
        if expected is not None and file_sha256(path) != expected:
            raise ValueError(f"{kind} differs from the recorded trajectory hash")
        return path

    def camera(self, row: dict[str, Any], camera: str) -> dict[str, Any]:
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {CAMERAS}")
        value = row.get("cameras", {}).get(camera)
        if not isinstance(value, dict):
            raise ValueError(f"camera {camera} is unavailable at this exact frame")
        if self.mode == "full_rgbd" and self.path.is_file() and not _calibration_complete(value.get("calibration")):
            self._warn(f"Selected frame camera {camera} has missing or invalid recorded calibration.")
        return value

    def context_for(self, row: dict[str, Any], *, max_chars: int = 12000) -> dict[str, Any]:
        correlation = row.get("correlation") or {}
        tool_id = correlation.get("root_tool_call_id")
        sequence = correlation.get("capability_sequence")
        capability_id = correlation.get("capability_id")
        capabilities = []
        path = require_contained_path(self.public / "capability_calls.jsonl", anchor=self.public)
        if path.is_file():
            with path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    call = json.loads(line)
                    if tool_id:
                        matched = (call.get("root_tool_call_id") == tool_id
                                   and (sequence is None or call.get("sequence") == sequence)
                                   and (capability_id is None or call.get("capability_id") == capability_id))
                    else:
                        # Direct live calls have no online Agent tool ID. The
                        # recorder's episode-local sequence + capability ID
                        # still identifies the exact executed capability.
                        matched = (type(sequence) is int and isinstance(capability_id, str)
                                   and bool(capability_id) and call.get("sequence") == sequence
                                   and call.get("capability_id") == capability_id)
                    if matched:
                        capabilities.append({"line": line_number, "record": call})
        message = None
        path = require_contained_path(self.public / "transcript.json", anchor=self.public)
        if tool_id and path.is_file():
            turn = 0
            for candidate in read_json(path).get("messages", []):
                if candidate.get("role") != "assistant":
                    continue
                turn += 1
                content = candidate.get("content", [])
                if isinstance(content, list) and any(
                    isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == tool_id
                    for block in content
                ):
                    message = {"agent_turn": turn, "message": candidate}
                    break
        value = {"correlation": correlation, "agent": message, "capabilities": capabilities,
                 "live_command": None,
                 "exact_public_context": {"transcript": str(self.public / "transcript.json"),
                    "capability_calls": str(self.public / "capability_calls.jsonl"),
                    "commands": None}}
        decision_id = row.get("decision_id") or correlation.get("decision_id")
        value["history_decision"] = ({"episode_id": self.identity["episode"], "decision_id": decision_id}
                                     if decision_id else None)
        value["exact_public_context"]["planner_events"] = str(self.episode / "planner/events.jsonl")
        if len(json.dumps(value, ensure_ascii=False)) > max_chars:
            return {"correlation": correlation, "agent": _sanitize(message, max_string=1200),
                    "capabilities": [{"line": item["line"], "record": {
                        key: item["record"].get(key) for key in (
                            "sequence", "capability_id", "root_tool_call_id", "env_steps_before", "env_steps_after"
                        )}} for item in capabilities[:40]], "truncated": True,
                    "exact_public_context": value["exact_public_context"],
                    "live_command": _sanitize(value["live_command"]),
                    "history_decision": value["history_decision"],
                    "exact_context_selector": {"source": self.identity["source"],
                        "episode": self.identity["episode"], "env_step": row.get("env_step"),
                        "context_max_chars": 1000000}}
        return value


def _depth(reader: TrajectoryReader, camera: dict[str, Any]):
    import zipfile

    import numpy as np
    if reader.mode == "full_rgb":
        raise ValueError("depth was not archived in full_rgb mode; historical 3D back-projection is unavailable")
    path = reader.asset(camera, "depth")
    if path is None:
        raise ValueError("historical depth is missing; no depth or calibration is inferred")
    if (camera.get("depth_unit") != "meter" or camera.get("depth_key") != "depth_m"
            or camera.get("depth_dtype") != "float32"):
        raise ValueError("historical depth requires explicit meter units and float32 depth_m encoding")
    if path.suffix != ".npz":
        raise ValueError("unsupported historical depth encoding")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_ASSET_BYTES:
            raise ValueError("decompressed depth exceeds the historical asset size limit")
    with np.load(path, allow_pickle=False) as archive:
        depth = np.asarray(archive["depth_m"])
    if depth.ndim != 2 or depth.size > MAX_PIXELS or depth.dtype != np.dtype("float32"):
        raise ValueError("invalid historical metric depth array")
    if depth.shape != (camera.get("height"), camera.get("width")):
        raise ValueError("depth shape does not match recorded camera dimensions")
    return depth


def _depth_summary(depth) -> dict[str, Any]:
    import numpy as np
    valid = np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    return {"shape": list(depth.shape), "unit": "meter", "valid_pixels": int(valid.sum()),
            "minimum_m": float(values.min()) if values.size else None,
            "maximum_m": float(values.max()) if values.size else None,
            "median_m": float(np.median(values)) if values.size else None}


def _selected_camera(reader: TrajectoryReader, camera: str | None) -> str:
    if camera is None:
        return next((name for name in ("agentview", "head") if name in reader.cameras), reader.cameras[0])
    if camera != "all" and camera not in reader.cameras:
        raise ValueError(f"camera must be one of {reader.cameras} or all")
    return camera


def history_timeline(reader: TrajectoryReader, *, camera: str | None = None, max_frames: int = 9,
                     start_env_step: int | None = None, end_env_step: int | None = None):
    _integer(max_frames, "max_frames", minimum=1)
    if max_frames > 24:
        raise ValueError("max_frames must be <= 24 timepoints")
    camera = _selected_camera(reader, camera)
    for name, value in (("start_env_step", start_env_step), ("end_env_step", end_env_step)):
        if value is not None:
            _integer(value, name)
    if start_env_step is not None and end_env_step is not None and start_env_step > end_env_step:
        raise ValueError("start_env_step must not exceed end_env_step")
    available = reader.steps
    candidates = [step for step in available if (start_env_step is None or step >= start_env_step)
                  and (end_env_step is None or step <= end_env_step)]
    # Inclusive endpoints ensure every overview spans the whole requested interval.
    count = min(max_frames, len(candidates))
    selected = ([candidates[round(i * (len(candidates) - 1) / (count - 1))] for i in range(count)]
                if count > 1 else candidates[:1])
    rows, images = [], []
    for step in selected:
        row = reader.row(env_step=step)
        summary = {key: row.get(key) for key in ("env_step", "observation_step", "correlation")}
        summary["cameras"] = {}
        for name in reader.cameras if camera == "all" else (camera,):
            entry = row.get("cameras", {}).get(name, {})
            calibration_complete = _calibration_complete(entry.get("calibration"))
            if reader.mode == "full_rgbd" and reader.path.is_file() and not calibration_complete:
                reader._warn(f"Selected frame camera {name} has missing or invalid recorded calibration.")
            rgb = reader.asset(entry, "rgb")
            depth = reader.asset(entry, "depth")
            summary["cameras"][name] = {"rgb_available": rgb is not None, "depth_available": depth is not None,
                                         "calibration_available": calibration_complete}
            if rgb:
                images += _image(_thumbnail(rgb.read_bytes()),
                                 f"{reader.identity['episode']} env_step={step}, {name} (thumbnail)")
        rows.append(summary)
    selected_complete = all(all(entry["rgb_available"] and (reader.mode == "full_rgb" or (entry["depth_available"] and entry["calibration_available"]))
                                for entry in row["cameras"].values()) for row in rows) if rows else None
    value = {"status": "completed", "identity": reader.identity, "coverage": reader.coverage(
                selected_complete=selected_complete,
                all_assets_checked=camera == "all" and len(selected) == len(reader.steps)),
             "sampling": "uniform across recorded frames, including interval endpoints",
             "requested_interval": [start_env_step, end_env_step], "available_in_interval": len(candidates),
             "selected_frames": rows, "dense": len(selected) == len(candidates),
             "hint": "Select a smaller env_step interval for dense frames; history_frame returns exact recorded sensors and context."}
    return value, [_text(value), *images]


def history_frame(reader: TrajectoryReader, *, camera: str | None = None, env_step: int | None = None,
                  observation_step: int | None = None, context_max_chars: int = 12000):
    _integer(context_max_chars, "context_max_chars", minimum=1000)
    if context_max_chars > 1000000:
        raise ValueError("context_max_chars must be <= 1000000")
    camera = _selected_camera(reader, camera)
    row = reader.row(env_step=env_step, observation_step=observation_step)
    names = reader.cameras if camera == "all" else (camera,)
    images, camera_values = [], {}
    for name in names:
        if camera == "all" and name not in row.get("cameras", {}):
            reader._warn(f"Selected frame camera {name} was not recorded.")
            camera_values[name] = {"rgb_available": False, "depth_available": False,
                                   "reason": "Camera was not recorded at this exact frame."}
            continue
        entry = reader.camera(row, name)
        rgb = reader.asset(entry, "rgb")
        depth_path = reader.asset(entry, "depth")
        metadata = {**entry, "rgb_available": rgb is not None, "depth_available": depth_path is not None}
        if rgb:
            images += _image(rgb.read_bytes(), f"Exact historical RGB {name}, env_step={row.get('env_step')}, observation_step={row.get('observation_step')}")
        if depth_path:
            import numpy as np
            depth = _depth(reader, entry)
            metadata["depth_summary"] = _depth_summary(depth)
            valid = np.isfinite(depth) & (depth > 0)
            preview = np.zeros(depth.shape, dtype=np.uint8)
            if valid.any():
                lo, hi = float(depth[valid].min()), float(depth[valid].max())
                preview[valid] = 255 if hi == lo else (1 + 254 * (depth[valid] - lo) / (hi - lo)).astype(np.uint8)
                metadata["depth_preview"] = {"black": "invalid", "gray_1_m": lo, "white_m": hi}
            images += _image(_png(preview), f"Recorded optical-axis depth {name}; linear metric-depth preview, exact NPZ referenced in metadata")
        camera_values[name] = metadata
    selected_complete = all(entry["rgb_available"] and (reader.mode == "full_rgb" or (entry["depth_available"] and _calibration_complete(entry.get("calibration"))))
                            for entry in camera_values.values())
    value = {"status": "completed", "identity": reader.frame_identity(row), "coverage": reader.coverage(
                selected_complete=selected_complete,
                all_assets_checked=camera == "all" and len(reader.steps) == 1),
             "action": row.get("action"), "robot_state": row.get("robot_state"),
             "robot_state_metadata": row.get("robot_state_metadata"),
             "cameras": camera_values, "agent_context": reader.context_for(row, max_chars=context_max_chars)}
    return value, [_text(value), *images]


def _pixel(value: Any, shape) -> list[int]:
    if not isinstance(value, list) or len(value) != 2 or any(type(item) is not int for item in value):
        raise ValueError("pixel/point must be integer [u,v] (column,row)")
    u, v = value
    if not (0 <= u < shape[1] and 0 <= v < shape[0]):
        raise ValueError("pixel/point is outside this historical image")
    return value


def _calibration_matrices(calibration: dict[str, Any]):
    import numpy as np
    if not isinstance(calibration, dict):
        raise ValueError("recorded calibration is required")
    if (calibration.get("image_origin") != "top_left"
            or calibration.get("camera_axes") != "x_right_y_down_z_forward"
            or calibration.get("depth_semantics") != "optical_axis_z"
            or calibration.get("extrinsic_convention") != "column vectors"
            or calibration.get("target_frame") != "robot_base"):
        raise ValueError("missing or unsupported recorded camera/depth conventions")
    intrinsic = np.asarray(calibration.get("intrinsic"), dtype=float)
    transform = np.asarray(calibration.get("extrinsic_camera_to_base"), dtype=float)
    if (intrinsic.shape != (3, 3) or transform.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(transform).all()
            or abs(np.linalg.det(intrinsic)) < 1e-12
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("recorded intrinsics and camera-to-base transform are required")
    return intrinsic, transform


def _calibration_complete(calibration: Any) -> bool:
    try:
        _calibration_matrices(calibration)
    except (ValueError, TypeError):
        return False
    return True


def _backproject(depth, calibration: dict[str, Any], mask):
    import numpy as np
    intrinsic, transform = _calibration_matrices(calibration)
    valid = mask & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    if not len(u):
        return {"valid_pixels": 0, "reason": "selected pixels have no valid recorded metric depth"}
    depths = depth[v, u]
    camera = np.linalg.solve(intrinsic, np.stack([u, v, np.ones_like(u)])) * depths
    base = transform[:3, :3] @ camera + transform[:3, 3:4]
    result = {"valid_pixels": len(u), "depth_m": {"minimum": float(depths.min()),
                "median": float(np.median(depths)), "maximum": float(depths.max())},
            "camera_point_mean_m": camera.mean(axis=1).tolist(), "base_point_mean_m": base.mean(axis=1).tolist(),
            "base_bounds_m": [base.min(axis=1).tolist(), base.max(axis=1).tolist()],
            "camera_axes": "x_right_y_down_z_forward", "target_frame": calibration.get("target_frame", "robot_base")}
    world_transform = calibration.get("extrinsic_camera_to_world")
    if world_transform is None:
        result["world_unavailable"] = "This frame has no recorded camera-to-world transform."
    else:
        world_transform = np.asarray(world_transform, dtype=float)
        if (world_transform.shape != (4, 4) or not np.isfinite(world_transform).all()
                or not np.allclose(world_transform[3], [0, 0, 0, 1])):
            raise ValueError("recorded camera-to-world transform must be finite homogeneous 4x4")
        world = world_transform[:3, :3] @ camera + world_transform[:3, 3:4]
        result.update(world_point_mean_m=world.mean(axis=1).tolist(),
                      world_bounds_m=[world.min(axis=1).tolist(), world.max(axis=1).tolist()],
                      world_frame="recorded_world", world_source="recorded sensor extrinsic_camera_to_world")
    return result


def _frame_identity(reader, row, camera, entry):
    rgb, depth = reader.asset(entry, "rgb"), reader.asset(entry, "depth")
    observation_id = row.get("observation_id")
    if observation_id is None:
        observation_id = (f"env-{row['env_step']:06d}" if row.get("env_step") is not None
                          else f"observation-{row['observation_step']:06d}")
    return {**reader.frame_identity(row),
            "camera": camera, "observation_id": observation_id,
            "capture_id": entry.get("capture_id", f"{observation_id}/{camera}"),
            "rgb": entry.get("rgb") if rgb else None,
            "depth": entry.get("depth") if depth else None}


def _mask_frame_matches(receipt, identity, reader, row, camera, entry):
    expected = receipt.get("identity")
    if expected == identity:
        return True
    # Only existing receipts that explicitly used hashes need a compatibility
    # check. New frame/mask identities never compute an artifact digest.
    if isinstance(expected, dict) and ("rgb_sha256" in expected or "depth_sha256" in expected):
        rgb, depth = reader.asset(entry, "rgb"), reader.asset(entry, "depth")
        legacy_identity = {**reader.frame_identity(row), "camera": camera,
                           "rgb_sha256": file_sha256(rgb) if rgb else None,
                           "depth_sha256": file_sha256(depth) if depth else None}
        return expected == legacy_identity
    return False


def history_depth(reader: TrajectoryReader, *, camera: str = "agentview", env_step: int | None = None,
                  observation_step: int | None = None, pixel: list[int] | None = None, mask: str | None = None):
    import imageio.v2 as imageio
    import numpy as np
    if (pixel is None) == (mask is None):
        raise ValueError("select exactly one pixel or a mask produced by history_segment")
    row = reader.row(env_step=env_step, observation_step=observation_step)
    entry = reader.camera(row, camera)
    depth = _depth(reader, entry)
    identity = _frame_identity(reader, row, camera, entry)
    if pixel is not None:
        u, v = _pixel(pixel, depth.shape)
        selected = np.zeros(depth.shape, dtype=bool)
        selected[v, u] = True
    else:
        root = require_contained_path(reader.workspace / "analysis/trajectory", anchor=reader.workspace)
        path = require_contained_path(reader.workspace / mask, anchor=root, label="historical SAM mask")
        if path.name != "mask.png" or path.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError("mask must be a history_segment mask.png")
        receipt = read_json(require_contained_path(path.parent / "result.json", anchor=root))
        if not _mask_frame_matches(receipt, identity, reader, row, camera, entry):
            raise ValueError("mask does not match this exact historical RGB-D frame")
        if "mask_sha256" in receipt:
            if receipt["mask_sha256"] != file_sha256(path):
                raise ValueError("mask does not match this exact historical RGB-D frame")
        elif (receipt.get("artifacts", {}).get("mask") != mask
              or receipt.get("mask_id") != f"{path.parent.name}/mask"
              or receipt.get("mask_bytes") != path.stat().st_size):
            raise ValueError("mask reference does not match its recorded analysis artifact")
        selected = np.asarray(imageio.imread(path)) > 0
        if selected.shape != depth.shape:
            raise ValueError("mask does not match depth dimensions")
    value = {"status": "completed", "identity": identity, "pixel": pixel,
             "mask": mask, "pixel_convention": "[u,v] = [column,row], top-left origin",
             "selected_pixels": int(selected.sum()),
             "measurement": _backproject(depth, entry.get("calibration", {}), selected)}
    return value, [_text(value)]


def history_segment(reader: TrajectoryReader, *, camera: str = "agentview", env_step: int | None = None,
                    observation_step: int | None = None, text_prompt: str | None = None,
                    point: list[int] | None = None, min_score: float = 0.2):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    if (text_prompt is None) == (point is None):
        raise ValueError("provide exactly one text_prompt or point=[u,v]")
    if text_prompt is not None and (not isinstance(text_prompt, str) or not text_prompt.strip() or len(text_prompt) > 200):
        raise ValueError("text_prompt must contain 1-200 characters")
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be finite and between 0 and 1")
    row = reader.row(env_step=env_step, observation_step=observation_step)
    entry = reader.camera(row, camera)
    rgb = reader.asset(entry, "rgb")
    if rgb is None:
        raise ValueError("RGB is unavailable at this historical frame")
    with Image.open(rgb) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("historical RGB exceeds the image size limit")
        source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] not in (3, 4) or source.shape[0] * source.shape[1] > MAX_PIXELS:
        raise ValueError("invalid historical RGB image")
    source = source[..., :3]
    if point is not None:
        _pixel(point, source.shape)
    endpoint = reader.context.get("history", {}).get("sam3_endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("historical SAM3 is unavailable: no operator-configured history.sam3_endpoint")
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.utils.rpc import make_rpc_client
    try:
        segmented = Sam3Client(make_rpc_client(endpoint)).segment(
            rgb.read_bytes(), text_prompt=text_prompt,
            point=[point[1], point[0]] if point is not None else None, min_score=min_score)
    except Exception as exc:
        raise ValueError(f"historical SAM3 service failed ({type(exc).__name__}); no environment was started") from exc
    identity = _frame_identity(reader, row, camera, entry)
    value = {"status": "completed", "identity": identity, "found": bool(segmented.found),
             "score": segmented.score, "box": segmented.box, "reason": segmented.reason,
             "text_prompt": text_prompt, "point": point, "pixel_convention": "[u,v] = [column,row]",
             "artifacts": {}}
    images = []
    root = require_contained_path(reader.workspace / "analysis/trajectory", anchor=reader.workspace)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sam-", dir=root))
    if segmented.found:
        mask = np.asarray(segmented.mask, dtype=bool)
        if mask.shape != source.shape[:2]:
            raise ValueError("SAM3 returned a mask with different historical image dimensions")
        overlay = source.copy()
        overlay[mask] = (source[mask].astype(float) * .55 + np.array([255, 40, 120]) * .45).astype(np.uint8)
        for name, array in (("mask", mask.astype(np.uint8) * 255), ("overlay", overlay)):
            path = output / f"{name}.png"
            imageio.imwrite(path, array)
            value["artifacts"][name] = path.relative_to(reader.workspace).as_posix()
            images += _image(path.read_bytes(), f"Historical {camera} env_step={row.get('env_step')} SAM3 {name}")
        value["mask_id"] = f"{output.name}/mask"
        value["mask_bytes"] = (output / "mask.png").stat().st_size
        if reader.asset(entry, "depth") is not None:
            try:
                depth = _depth(reader, entry)
                if depth.shape != mask.shape:
                    raise ValueError("RGB mask and recorded depth shapes differ")
                value["measurement"] = _backproject(depth, entry.get("calibration", {}), mask)
            except ValueError as exc:
                value["depth_unavailable"] = str(exc)
        else:
            value["depth_unavailable"] = "Metric depth was not recorded at this historical frame."
    value["artifacts"]["result"] = (output / "result.json").relative_to(reader.workspace).as_posix()
    atomic_json(output / "result.json", value)
    return value, [_text(value), *images]
