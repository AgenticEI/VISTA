# Copyright 2026 The RPent Authors.
"""Pool RGB-D history retains exact frames, metric depth and recorded boundaries."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import pytest

from meta_harness import trajectory
from meta_harness._files import atomic_json, file_sha256
from meta_harness.meta_mcp import invoke
from meta_harness.trajectory import TrajectoryReader, history_frame


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.evidence = self.workspace / "evidence"
        self.episode = self.evidence / "episodes/repeat-00"
        self.public = self.episode / "public"
        self.public.mkdir(parents=True)
        self.context = self.workspace / "context.json"
        atomic_json(self.context, {"schema_version": 1, "evidence": str(self.evidence),
                                  "history": {"sam3_endpoint": "http://127.0.0.1:9876"}})
        atomic_json(self.episode / "result.json", {
            "schema_version": 1, "validity": "valid",
            "task": {"task_name": "Drawer", "seed": 40, "repeat_index": 0},
            "agent": {"model": "test-model"}, "harness": {"id": "h0", "commit": "a" * 40},
            "metrics": {"env_steps": 10}, "outcome": {"env_steps": 10, "official_success": False}})
        atomic_json(self.public / "transcript.json", {"messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "Measure before moving."},
            {"type": "tool_use", "id": "tool-1", "name": "move", "input": {"delta": [1, 0, 0]}}]}]})
        (self.public / "capability_calls.jsonl").write_text(json.dumps({
            "sequence": 1, "root_tool_call_id": "tool-1", "capability_id": "control.osc@1",
            "env_steps_before": 0, "env_steps_after": 10, "arguments": {"repeat": 10}}) + "\n")
        rows = []
        self.original_rgb = {}
        for step in range(11):
            cameras = {}
            for index, camera in enumerate(("agentview", "navview", "wrist")):
                directory = self.public / "trajectory" / f"{step:06d}"
                directory.mkdir(parents=True, exist_ok=True)
                rgb = directory / f"{camera}_rgb.png"
                depth = directory / f"{camera}_depth.npz"
                imageio.imwrite(rgb, np.full((3, 4, 3), step * 10 + index, dtype=np.uint8))
                self.original_rgb[(step, camera)] = rgb.read_bytes()
                np.savez_compressed(depth, depth_m=np.full((3, 4), 2 + step, dtype=np.float32))
                transform = np.eye(4)
                transform[:3, 3] = [step, 10, 20]
                world_transform = np.array([[0, -1, 0, 100], [1, 0, 0, 200],
                                            [0, 0, 1, 300], [0, 0, 0, 1]], dtype=float)
                cameras[camera] = {"rgb": rgb.relative_to(self.episode).as_posix(),
                    "depth": depth.relative_to(self.episode).as_posix(), "height": 3, "width": 4,
                    "rgb_sha256": file_sha256(rgb), "depth_sha256": file_sha256(depth),
                    "depth_unit": "meter", "depth_key": "depth_m", "depth_dtype": "float32",
                    "calibration": {"intrinsic": [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
                        "extrinsic_camera_to_base": transform.tolist(), "image_origin": "top_left",
                        "extrinsic_camera_to_world": world_transform.tolist(),
                        "camera_axes": "x_right_y_down_z_forward", "depth_semantics": "optical_axis_z",
                        "extrinsic_convention": "column vectors", "target_frame": "robot_base"}}
            rows.append({"schema_version": 1, "env_step": step,
                "action": None if step == 0 else {"env_steps_before": step - 1, "env_steps_after": step, "env_action": [step]},
                "robot_state": {"eef_position": [step, 0, 0]},
                "correlation": {"root_tool_call_id": "tool-1", "capability_id": "control.osc@1", "capability_sequence": 1},
                "cameras": cameras})
        self.rows = rows
        self.index = self.public / "trajectory.jsonl"
        self.index.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def call(self, name, **kwargs):
        context = json.loads(self.context.read_text())
        reader = TrajectoryReader.from_episode(
            self.episode, workspace=self.workspace,
            sam_endpoint=context.get("history", {}).get("sam3_endpoint"),
        )
        return getattr(trajectory, "history_" + name)(reader, **kwargs)

    def test_uniform_overview_dense_interval_three_views_and_exact_frame(self):
        value, blocks = self.call("timeline", camera="all", max_frames=3)
        self.assertEqual([row["env_step"] for row in value["selected_frames"]], [0, 5, 10])
        self.assertFalse(value["coverage"]["complete"])
        self.assertTrue(value["coverage"]["control_step_coverage_complete"])
        self.assertTrue(value["coverage"]["verified_selected_frame_complete"])
        self.assertFalse(value["coverage"]["all_recorded_assets_checked"])
        self.assertEqual(len([block for block in blocks if block["type"] == "image"]), 9)
        dense, _ = self.call("timeline", camera="all", max_frames=24)
        self.assertTrue(dense["coverage"]["complete"])
        value, _ = self.call("timeline", start_env_step=6, end_env_step=8, max_frames=9)
        self.assertEqual([row["env_step"] for row in value["selected_frames"]], [6, 7, 8])
        self.assertTrue(value["dense"])
        value, blocks = self.call("frame", env_step=5, camera="all")
        images = [base64.b64decode(block["data"]) for block in blocks if block["type"] == "image"]
        self.assertEqual(images[::2], [self.original_rgb[(5, camera)] for camera in ("agentview", "navview", "wrist")])
        self.assertEqual(value["robot_state"]["eef_position"], [5, 0, 0])
        self.assertEqual(value["action"]["env_steps_before"], 4)
        self.assertEqual(value["agent_context"]["agent"]["agent_turn"], 1)
        self.assertEqual(value["agent_context"]["capabilities"][0]["record"]["sequence"], 1)
        self.assertEqual(value["identity"]["agent"]["model"], "test-model")

    def test_pixel_backprojection_uses_selected_frame_and_uv_order(self):
        value, _ = self.call("depth", env_step=5, pixel=[3, 1])
        measurement = value["measurement"]
        self.assertEqual(measurement["depth_m"]["median"], 7)
        self.assertEqual(measurement["camera_point_mean_m"], [7, 0, 7])
        self.assertEqual(measurement["base_point_mean_m"], [12, 10, 27])
        self.assertEqual(measurement["world_point_mean_m"], [100, 207, 307])
        self.assertEqual(measurement["world_bounds_m"], [[100, 207, 307], [100, 207, 307]])
        for pixel in ([4, 1], [3, 3], [-1, 0], [True, 0], [1.2, 2]):
            with self.subTest(pixel=pixel), self.assertRaises(ValueError):
                self.call("depth", env_step=5, pixel=pixel)

    def test_world_measurements_require_recorded_world_extrinsics(self):
        calibration = self.rows[5]["cameras"]["agentview"]["calibration"]
        calibration.pop("extrinsic_camera_to_world")
        self.index.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        value, _ = self.call("depth", env_step=5, pixel=[3, 1])
        self.assertEqual(value["measurement"]["base_point_mean_m"], [12, 10, 27])
        self.assertNotIn("world_point_mean_m", value["measurement"])
        self.assertIn("no recorded", value["measurement"]["world_unavailable"])
        calibration["extrinsic_camera_to_world"] = [[1, 0], [0, 1]]
        self.index.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        with self.assertRaisesRegex(ValueError, "camera-to-world"):
            self.call("depth", env_step=5, pixel=[3, 1])


    def test_existing_sam_client_receives_exact_rgb_and_outputs_are_frame_bound(self):
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 3] = 255
        stream = io.BytesIO()
        imageio.imwrite(stream, mask, format="png")
        test = self

        class FakeRpc:
            def call(self, method, args=(), kwargs=None, timeout_s=None):
                test.assertEqual(method, "segment")
                test.assertEqual(base64.b64decode(kwargs["image_base64"]), test.original_rgb[(5, "agentview")])
                test.assertEqual(kwargs["point"], [1, 3])  # existing client's row,col protocol
                return {"found": True, "score": .9, "mask_shape": [3, 4],
                        "mask_png_base64": base64.b64encode(stream.getvalue()).decode()}

        before = {p.relative_to(self.evidence): p.read_bytes() for p in self.evidence.rglob("*") if p.is_file()}
        with patch("rpent.utils.rpc.make_rpc_client", return_value=FakeRpc()) as factory:
            value, blocks = self.call("segment", env_step=5, point=[3, 1])
        factory.assert_called_once_with("http://127.0.0.1:9876")
        self.assertEqual({p.relative_to(self.evidence): p.read_bytes() for p in self.evidence.rglob("*") if p.is_file()}, before)
        self.assertEqual(value["measurement"]["base_point_mean_m"], [12, 10, 27])
        self.assertEqual(value["measurement"]["world_point_mean_m"], [100, 207, 307])
        self.assertEqual(len([block for block in blocks if block["type"] == "image"]), 2)
        self.assertTrue(value["artifacts"]["mask"].startswith("analysis/trajectory/"))
        measured, _ = self.call("depth", env_step=5, mask=value["artifacts"]["mask"])
        self.assertEqual(measured["measurement"]["valid_pixels"], 1)
        with self.assertRaisesRegex(ValueError, "exact historical"):
            self.call("depth", env_step=6, mask=value["artifacts"]["mask"])
        with self.assertRaisesRegex(TypeError, "endpoint"):
            self.call("segment", env_step=5, point=[3, 1], endpoint="http://other:1")

    def test_new_analysis_artifacts_use_frame_and_mask_ids_without_hashes(self):
        for row in self.rows:
            for capture in row["cameras"].values():
                capture.pop("rgb_sha256", None)
                capture.pop("depth_sha256", None)
        self.index.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 3] = 255
        stream = io.BytesIO()
        imageio.imwrite(stream, mask, format="png")

        class FakeRpc:
            def call(self, method, args=(), kwargs=None, timeout_s=None):
                return {"found": True, "score": .9, "mask_shape": [3, 4],
                        "mask_png_base64": base64.b64encode(stream.getvalue()).decode()}

        with patch("rpent.utils.rpc.make_rpc_client", return_value=FakeRpc()), patch(
            "meta_harness.trajectory.file_sha256", side_effect=AssertionError("new artifacts must not be hashed")
        ):
            segmented, _ = self.call("segment", env_step=5, point=[3, 1])
            measured, _ = self.call("depth", env_step=5, mask=segmented["artifacts"]["mask"])
        self.assertEqual(measured["measurement"]["valid_pixels"], 1)
        self.assertEqual(segmented["identity"]["observation_id"], "env-000005")
        self.assertEqual(segmented["identity"]["capture_id"], "env-000005/agentview")
        self.assertNotIn("sha256", json.dumps(segmented))
        self.assertTrue(segmented["mask_id"].endswith("/mask"))


    def test_partial_traces_hash_mismatch_and_public_path_boundaries(self):
        self.index.write_text("".join(json.dumps(row) + "\n" for row in self.rows[:5]) + '{"schema_version":')
        value, _ = self.call("timeline")
        self.assertFalse(value["coverage"]["complete"])
        self.assertTrue(value["coverage"]["warnings"])
        private = self.episode / "private"
        private.mkdir()
        (private / "secret.png").write_bytes(self.original_rgb[(0, "agentview")])
        for reference in ("../private/secret.png", "private/secret.png", "public/../private/secret.png", str(private / "secret.png")):
            row = json.loads(json.dumps(self.rows[0]))
            row["cameras"]["agentview"]["rgb"] = reference
            self.index.write_text(json.dumps(row) + "\n")
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.call("frame", env_step=0)
        self.index.write_text(json.dumps(self.rows[0]) + "\n")
        image = self.episode / self.rows[0]["cameras"]["agentview"]["rgb"]
        image.write_bytes(self.original_rgb[(1, "agentview")])
        with self.assertRaisesRegex(ValueError, "hash"):
            self.call("frame", env_step=0)
        image.unlink()
        image.symlink_to(private / "secret.png")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.call("frame", env_step=0)

    def test_raw_recorder_error_prevents_complete_coverage(self):
        error = "TraceError: insufficient trace storage before next environment action"
        atomic_json(self.public / "trajectory_meta.json", {"schema_version": 1, "error": error})
        value, _ = self.call("timeline", camera="all", max_frames=24)
        coverage = value["coverage"]
        self.assertTrue(coverage["control_step_coverage_complete"])
        self.assertTrue(coverage["verified_selected_frame_complete"])
        self.assertTrue(coverage["all_recorded_assets_checked"])
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["recorder_metadata"]["error"], error)
        self.assertTrue(any(error in warning for warning in coverage["warnings"]))
        frame, _ = self.call("frame", env_step=5)
        self.assertTrue(any(error in warning for warning in frame["coverage"]["warnings"]))

    def test_missing_depth_unknown_calibration_and_unconfigured_sam_are_explicit(self):
        atomic_json(self.public / "trajectory_meta.json", {"complete": True})
        depth = self.episode / self.rows[5]["cameras"]["agentview"]["depth"]
        depth.rename(depth.with_suffix(".unavailable"))
        value, blocks = self.call("frame", env_step=5)
        self.assertFalse(value["cameras"]["agentview"]["depth_available"])
        self.assertTrue(value["coverage"]["control_step_coverage_complete"])
        self.assertFalse(value["coverage"]["complete"])
        self.assertTrue(value["coverage"]["recorded_complete"])
        self.assertFalse(value["coverage"]["verified_selected_frame_complete"])
        self.assertIn("depth file is missing", value["coverage"]["warnings"][0])
        self.assertEqual(len([block for block in blocks if block["type"] == "image"]), 1)
        timeline, _ = self.call("timeline", start_env_step=5, end_env_step=5)
        self.assertFalse(timeline["coverage"]["complete"])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.call("depth", env_step=5, pixel=[1, 1])
        self.rows[6]["cameras"]["agentview"]["calibration"].pop("camera_axes")
        self.index.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        frame, _ = self.call("frame", env_step=6)
        self.assertFalse(frame["coverage"]["verified_selected_frame_complete"])
        with self.assertRaisesRegex(ValueError, "conventions"):
            self.call("depth", env_step=6, pixel=[1, 1])
        context = json.loads(self.context.read_text())
        context.pop("history")
        atomic_json(self.context, context)
        with self.assertRaisesRegex(ValueError, "operator-configured"):
            self.call("segment", env_step=0, text_prompt="drawer")


    def test_meta_frame_uses_pool_episode_and_returns_exact_images(self):
        requested = []

        def history_episode(episode_id):
            requested.append(episode_id)
            return {"rollout": str(self.episode)}

        attempt = SimpleNamespace(history_episode=history_episode)
        value, blocks = invoke(
            attempt, "history_frame", {"episode_id": "recorded", "env_step": 5}
        )
        assert requested == ["recorded"]
        images = [b for b in blocks if b["type"] == "image"]
        assert base64.b64decode(images[0]["data"]) == self.original_rgb[5, "agentview"]
        assert value["identity"]["env_step"] == 5
        assert value["action"]["env_steps_after"] == 5
        assert value["agent_context"]["capabilities"][0]["record"]["sequence"] == 1

    def test_future_assets_do_not_change_index_and_selected_corruption_is_detected(self):
        reader = TrajectoryReader.from_episode(self.episode)
        future = self.public / "trajectory/000011"
        future.mkdir()
        (future / "future.npz").write_bytes(b"not in the recorded index")
        assert reader.steps == list(range(11))
        target = self.public / "trajectory/000005/agentview_depth.npz"
        target.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="hash"):
            history_frame(reader, env_step=5)

    def test_long_recorded_context_is_bounded(self):
        path = self.public / "transcript.json"
        record = json.loads(path.read_text())
        record["messages"][0]["content"].insert(0, {"type": "text", "text": "x" * 20000})
        path.write_text(json.dumps(record))
        value, _ = history_frame(
            TrajectoryReader.from_episode(self.episode), env_step=5, context_max_chars=1000
        )
        context = value["agent_context"]
        assert context["truncated"] is True
        assert "omitted" in context["agent"]["message"]["content"][0]["text"]
        assert context["exact_context_selector"]["env_step"] == 5


if __name__ == "__main__":
    unittest.main()
