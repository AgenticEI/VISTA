# Copyright 2026 The RPent Authors.
"""LIBERO's versioned sensor/control seam for ordinary Harness packages."""

from __future__ import annotations

import copy
import json
import threading
import time
from typing import Mapping

import numpy as np
from jsonschema import Draft202012Validator

from rpent.harness import CapabilityBinding, CapabilitySpec
from rpent.result import EpisodeOutcome
from rpent.tools.correlation import current_root_tool_call_id
from rpent.trace import trace_capability_scope

OBSERVE = "observation.public@1"
ROBOT_STATE = "observation.libero.state@1"
RGBD = "observation.libero.rgbd@1"
BACK_PROJECT = "geometry.libero.back_project@1"
SAM3_SEGMENT = "perception.sam3.segment@1"
OSC_STEP = "control.libero.osc_step@1"
EEF_SERVO = "control.libero.eef_servo_world@1"
GRIPPER_SET = "control.libero.gripper.set@1"
PI05_CHUNK = "policy.libero.pi05.chunk@1"
ARTIFACT_WRITE = "artifact.public.write@1"


def _object(properties=None, required=()):
    return {"type": "object", "properties": properties or {},
            "required": list(required), "additionalProperties": False}


def _output_schema(capability_id):
    """Describe the in-process values available to capability authors."""
    def number_vector(size):
        return {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}
    state = {"type": "object", "properties": {
        "eef_position": number_vector(3), "eef_quaternion": number_vector(4),
        "gripper_qpos": number_vector(2), "base_position": number_vector(3),
        "base_quaternion": number_vector(4), "env_steps": {"type": "integer"},
        "max_episode_steps": {"type": "integer"}, "episode_active": {"type": "boolean"},
    }}
    observation = {"type": "object", "properties": {
        "task_language": {"type": "string"}, "state": state,
        "episode_active": {"type": "boolean"}, "images": {"type": "array"},
        "_image_cam_bytes": {"x-python-type": "bytes", "description": "Agentview PNG for multimodal feedback."},
        "_image_wrist_bytes": {"x-python-type": "bytes", "description": "Wrist PNG for multimodal feedback."},
    }}
    if capability_id == OBSERVE:
        return observation
    if capability_id == ROBOT_STATE:
        return state
    if capability_id == RGBD:
        return {"type": "object", "properties": {
            "capture_id": {"type": "string"}, "camera": {"type": "string"},
            "step": {"type": "integer"}, "env_steps": {"type": "integer"},
            "depth_unit": {"const": "meter"}, "image_origin": {"const": "top-left"},
            "width": {"type": "integer"}, "height": {"type": "integer"},
            "calibration": {"type": "object", "description": "intrinsic_K and extrinsic_cam2world in the aligned image frame."},
            "artifacts": {"type": "object", "description": "RGB and pseudocolor depth PNG references; use back_project for metric geometry."},
            "depth_statistics_m": {"type": "object"},
            "depth_summary": {"type": "string"},
            "_image_cam_bytes": {"x-python-type": "bytes"},
            "_image_depth_bytes": {"x-python-type": "bytes"},
        }}
    if capability_id in {OSC_STEP, EEF_SERVO, GRIPPER_SET, PI05_CHUNK}:
        return {"type": "object", "properties": {
            "operation": {"type": "object", "properties": {"name": {"type": "string"}, "env_steps": {"type": "integer"}}},
            "observation": observation,
            "_image_cam_bytes": {"x-python-type": "bytes"}, "_image_wrist_bytes": {"x-python-type": "bytes"},
        }}
    if capability_id == ARTIFACT_WRITE:
        return {"type": "object", "properties": {"kind": {"type": "string"},
                "artifact": {"type": "string"}, "step": {"type": "integer"}}}
    return {"type": "object", "description": "Unmodified native sensor-localization result, including source camera/step and world_xyz when found."}


def capability_specs():
    """The catalog and runtime use these same native argument contracts."""
    from robots.libero.tools import TOOLS_SPEC

    native = {item["name"]: item for item in TOOLS_SPEC}
    empty = _object()
    definitions = [
        (OBSERVE, "Read current task language, public robot state and both camera images; no official outcome.", empty),
        (ROBOT_STATE, "Read real world EEF-site position, EEF-body quaternion xyzw, gripper and fixed base pose.", empty),
        (RGBD, "Capture aligned top-left RGB and pseudocolor depth PNG with min/median/max in metres. Use back_project for metric geometry; no dense arrays are returned.",
         _object({"camera": {"type": "string", "enum": ["agentview", "wrist"], "default": "agentview"}})),
        (BACK_PROJECT, "Localize pixels or a region using the sensor-derived world map of a recorded observation.", native["back_project"]["input_schema"]),
        (SAM3_SEGMENT, "Segment a recorded camera view with SAM3 and locate its mask in the sensor-derived world map.", native["segment"]["input_schema"]),
        (OSC_STEP, "Execute normalized OSC delta position/rotation and gripper commands; +1 closes, -1 opens.",
         _object({"action": {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1},
                              "minItems": 7, "maxItems": 7},
                  "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 1}}, ("action",))),
        (EEF_SERVO, "Use the native closed-loop move_to controller in MuJoCo world metres with its original defaults.", native["move_to"]["input_schema"]),
        (GRIPPER_SET, "Hold the EEF pose and set the native gripper command; +1 closes, -1 opens.",
         _object({"gripper": {"type": "number", "minimum": -1, "maximum": 1, "default": -1},
                  "steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5}})),
        (PI05_CHUNK, "Predict and execute exactly one raw Pi0.5 action batch using the supplied instruction and current sensor observation.",
         _object({"instruction": {"type": "string", "minLength": 1, "maxLength": 4000}}, ("instruction",))),
        (ARTIFACT_WRITE, "Save a package-generated image, mask, numeric array, JSON or text into this episode's current observation record.",
         _object({"name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"},
                  "kind": {"type": "string", "enum": ["image", "mask", "array", "json", "text"]},
                  "data": {}}, ("name", "kind", "data"))),
    ]
    return [CapabilitySpec(id=key, description=description,
                           input_schema=copy.deepcopy(schema), output_schema=_output_schema(key))
            for key, description, schema in definitions]


class LiberoHarnessAdapter:
    def __init__(self, toolkit):
        self._toolkit = toolkit
        self._primitives = toolkit.harness_primitives
        self._calls = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._artifact_sequence = 0
        self._capture_sequence = 0
        self.live_depth = False
        methods = {
            OBSERVE: self._observe, ROBOT_STATE: self._robot_state, RGBD: self._rgbd,
            BACK_PROJECT: lambda args: self._native("back_project", args),
            SAM3_SEGMENT: lambda args: self._native("segment", args),
            OSC_STEP: self._osc_step,
            EEF_SERVO: lambda args: self._native_action("move_to", args),
            GRIPPER_SET: lambda args: self._native_action("set_gripper", args),
            PI05_CHUNK: self._pi05_chunk, ARTIFACT_WRITE: self._artifact_write,
        }
        self._bindings = {}
        for spec in capability_specs():
            def invoke(args, spec=spec, method=methods[spec.id]):
                return self._record(spec, args, method)
            self._bindings[spec.id] = CapabilityBinding(spec=spec, invoke=invoke)

    def allowed_native_tools(self):
        return frozenset(spec["name"] for spec in self._toolkit.get_tools_spec())

    def capabilities(self):
        return dict(self._bindings)

    def capability_calls(self):
        with self._lock:
            return copy.deepcopy(self._calls)

    def episode_active(self):
        env = self._primitives.env
        return not (env.terminated or env.truncated)

    def outcome(self):
        try:
            self._primitives.env.finish_trajectory()
            status = self._primitives.env.get_status()
            success = bool(status["official_success"])
            done, truncated = bool(status["done"]), bool(status["truncated"])
            steps = int(status["episode_steps"])
            if steps < 0:
                raise ValueError("negative physical step count")
            return EpisodeOutcome(
                validity="valid", official_success=success,
                terminated=done and not truncated, truncated=truncated, env_steps=steps,
                terminal_reason=("official_success" if success else "horizon" if truncated
                                 else "environment_terminated" if done else "agent_stopped"),
            )
        except Exception as exc:
            return EpisodeOutcome(validity="invalid", official_success=None,
                                  terminated=False, truncated=False, env_steps=None,
                                  terminal_reason="outcome_unavailable", error=str(exc))

    def _steps(self):
        return int(self._primitives.env.env_steps)

    def _robot_state(self, _args):
        public = self._primitives.env.public_robot_state()
        # Deliberate allowlist: simulator status and object poses never leak.
        keys = ("eef_position", "eef_quaternion", "gripper_qpos", "base_position", "base_quaternion")
        state = {key: public[key] for key in keys}
        horizon = self._primitives.env.get_status()["max_episode_steps"]
        return {**state, "env_steps": self._steps(),
                "max_episode_steps": int(horizon),
                "episode_active": self.episode_active()}

    def _observe(self, _args):
        state = self._robot_state({})
        result = {"task_language": self._primitives.env.get_task_language(),
                  "state": state, "episode_active": self.episode_active(), "images": []}
        record = self._toolkit.state.latest_record()
        if record is None:
            raise RuntimeError("observation requires an initialized scene")
        for camera, image_key in (("agentview", "_image_cam_bytes"), ("wrist", "_image_wrist_bytes")):
            artifact = next((name for name in (f"{camera}_high.png", f"{camera}.png")
                             if name in record.artifacts), None)
            if artifact is None:
                raise RuntimeError(f"missing {camera} image for observation")
            result[image_key] = self._toolkit.state.load_bytes(artifact, step=record.step_idx)
            result["images"].append({"camera": camera, "artifact": artifact, "step": record.step_idx})
        if self.live_depth:
            result["depth"] = []
            for camera in ("agentview", "wrist"):
                capture = self._rgbd({"camera": camera})
                result[f"_image_depth_{camera}_bytes"] = capture.pop("_image_depth_bytes")
                rgb_key = "_image_cam_bytes" if camera == "agentview" else "_image_wrist_bytes"
                result[rgb_key] = capture.pop("_image_cam_bytes")
                result["images"] = [item for item in result["images"] if item["camera"] != camera]
                result["images"].append({"camera": camera, **capture["artifacts"]["rgb"]})
                result["depth"].append(capture)
        return result

    def _rgbd(self, args):
        from robots.libero.tools import _metric_depth

        camera = args.get("camera", "agentview")
        native = "agentview" if camera == "agentview" else "robot0_eye_in_hand"
        raw = self._primitives.env.raw_obs()
        rgb = np.ascontiguousarray(np.asarray(raw[native + "_image"], dtype=np.uint8)[::-1])
        height, width = rgb.shape[:2]
        calibration = self._primitives.env.get_camera_meta(native, height, width)
        if not calibration:
            raise RuntimeError(f"camera calibration unavailable: {camera}")
        depth = np.ascontiguousarray(_metric_depth(raw[native + "_depth"], calibration)[::-1])
        preview, statistics, summary = depth_preview(depth)
        step = self._toolkit.state.latest_record().step_idx
        self._capture_sequence += 1
        capture = f"libero_rgbd_{self._capture_sequence:04d}_{camera}"
        artifacts = {}
        for key, filename, value in (("rgb", capture + ".png", rgb),
                                      ("depth_preview", capture + "_depth.png", preview)):
            if self._toolkit.state.save(filename, value, step=step) is None:
                raise RuntimeError(f"failed to write RGB-D artifact {filename}")
            artifacts[key] = {"artifact": filename, "step": step}
        return {"capture_id": capture, "camera": camera, "step": step,
                "env_steps": self._steps(), "width": width, "height": height,
                "depth_unit": "meter", "image_origin": "top-left", "calibration": calibration,
                "artifacts": artifacts, "depth_statistics_m": statistics, "depth_summary": summary,
                "metric_lookup": {"tool": "back_project", "camera": camera, "step": step, "resolution": "low"},
                "_image_depth_bytes": self._toolkit.state.load_bytes(capture + "_depth.png", step=step),
                "_image_cam_bytes": self._toolkit.state.load_bytes(capture + ".png", step=step)}

    def _native(self, name, args):
        result = self._toolkit.execute_tool(name, dict(args)).result
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def _native_action(self, name, args):
        if not self.episode_active():
            raise RuntimeError("episode no longer accepts actions")
        before = self._steps()
        self._native(name, args)
        return self._action_response(name, before)

    def _action_response(self, name, before):
        observation = self._observe({})
        response = {"operation": {"name": name, "env_steps": self._steps() - before},
                    "observation": observation}
        for key in tuple(observation):
            if key.startswith("_image_"):
                response[key] = observation.pop(key)
        return response

    def _capture_action(self, name, arguments, before, started):
        self._toolkit.get_env_state(command={"action": name, **dict(arguments)},
                                    result={"executed_steps": self._steps() - before},
                                    elapsed_s=time.perf_counter() - started)
        self._toolkit._publish_step(self._toolkit.state.latest_record())
        return self._action_response(name, before)

    def _osc_step(self, args):
        action = np.asarray(args["action"], dtype=np.float32)
        if not np.isfinite(action).all():
            raise ValueError("OSC action must be finite")
        if not self.episode_active():
            raise RuntimeError("episode no longer accepts actions")
        before, started = self._steps(), time.perf_counter()
        for _ in range(args.get("steps", 1)):
            self._primitives._step_env(action)
            if not self.episode_active():
                break
        return self._capture_action("osc_step", args, before, started)

    def _pi05_chunk(self, args):
        if not self.episode_active():
            raise RuntimeError("episode no longer accepts actions")
        before, started = self._steps(), time.perf_counter()
        self._primitives._vlm_chunk(args["instruction"])
        return self._capture_action("pi05_chunk", args, before, started)

    def _artifact_write(self, args):
        kind, value = args["kind"], args["data"]
        suffix = {"image": ".png", "mask": ".png", "array": ".npz", "json": ".json", "text": ".txt"}[kind]
        if kind in {"image", "mask", "array"}:
            value = np.asarray(value)
            if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
                raise ValueError("artifact array must have a numeric or boolean dtype")
            if value.nbytes > 32 * 1024 * 1024:
                raise ValueError("artifact exceeds the 32 MiB limit")
            if kind == "mask":
                if value.ndim != 2:
                    raise ValueError("mask must be HxW")
                value = (value != 0).astype(np.uint8) * 255
            elif kind == "image":
                if value.ndim not in {2, 3} or value.ndim == 3 and value.shape[-1] not in {3, 4}:
                    raise ValueError("image must be HxW, HxWx3 or HxWx4")
                value = np.clip(value, 0, 255).astype(np.uint8)
        elif kind == "text":
            if not isinstance(value, str):
                raise ValueError("text artifact data must be a string")
        else:
            json.dumps(value, allow_nan=False)
        self._artifact_sequence += 1
        filename = f"harness_artifact_{self._artifact_sequence:04d}_{args['name']}{suffix}"
        step = self._toolkit.state.latest_record().step_idx
        if self._toolkit.state.save(filename, value, step=step) is None:
            raise RuntimeError("failed to persist package artifact")
        result = {"kind": kind, "artifact": filename, "step": step}
        if kind in {"image", "mask"}:
            result["_image_cam_bytes"] = self._toolkit.state.load_bytes(filename, step=step)
        return result

    def _record(self, spec, args, operation):
        Draft202012Validator(spec.input_schema).validate(args)
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        before, started = self._steps(), time.perf_counter()
        result, error = None, None
        try:
            with trace_capability_scope(spec.id, sequence):
                result = operation(args)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            record = {"sequence": sequence, "capability_id": spec.id,
                      "arguments": _audit_value(args), "result_summary": _audit_value(result),
                      "env_steps_before": before, "env_steps_after": self._steps(),
                      "elapsed_s": time.perf_counter() - started, "error": error,
                      "root_tool_call_id": current_root_tool_call_id()}
            with self._lock:
                self._calls.append(record)


def _audit_value(value):
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): _audit_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_harness_adapter(toolkit):
    return LiberoHarnessAdapter(toolkit)


def depth_preview(depth):
    """A finite-only metric summary and blue/cyan/yellow/red preview; black invalid."""
    depth = np.asarray(depth).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not values.size:
        return image, {"min": None, "median": None, "max": None}, "Depth (m): min/median/max unavailable; black = invalid."
    lo, median, hi = map(float, (values.min(), np.median(values), values.max()))
    t = np.zeros_like(depth, dtype=np.float32)
    t[valid] = (depth[valid] - lo) / max(hi - lo, 1e-6)
    # Piecewise RGB colormap without a plotting/runtime dependency.
    anchors = np.array([[0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0]])
    for channel in range(3):
        image[..., channel][valid] = np.interp(t[valid], [0, 1/3, 2/3, 1], anchors[:, channel]).astype(np.uint8)
    return image, {"min": lo, "median": median, "max": hi}, (
        f"Depth (m): min={lo:.4f}, median={median:.4f}, max={hi:.4f}; "
        "blue=near, red=far, black=invalid; use back_project for metric coordinates.")
