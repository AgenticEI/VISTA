"""Regression coverage for the LIBERO package/host seam, without GPU models."""

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.harness_adapter import (
    ARTIFACT_WRITE, OSC_STEP, RGBD, LiberoHarnessAdapter, capability_specs,
)
from robots.libero.robot_spec import _add_cli_args, _evolution_launch, _parse_config, get_robot_spec
from robots.libero.toolkit import LiberoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.harness import bind_harness, inspect_harness, load_harness
from rpent.memory import MemoryManager
from rpent.tools.state import EnvState

ROOT = Path(__file__).resolve().parents[1]




def test_package_memory_aliases_read_candidate_and_reject_writes(tmp_path, monkeypatch):
    memory = tmp_path / "candidate" / "memory"
    (memory / "global").mkdir(parents=True)
    (memory / "MEMORY.md").write_text("candidate-only index")
    (memory / "global" / "strategy.md").write_text("candidate strategy")
    monkeypatch.setattr(LiberoToolkit, "init_primitives", lambda self, **kw: None)
    monkeypatch.setattr(LiberoToolkit, "_register_libero_tools", lambda self: None)
    toolkit = LiberoToolkit(primitives_kwargs={}, dashboard_events=NullDashboardEventSink(),
                            state_output_dir=tmp_path / "episode",
                            memory=MemoryManager(memory, package_memory=True))
    for alias in ("resources/libero/MEMORY.md", "resources/libero/memory/MEMORY.md", "MEMORY.md"):
        result = toolkit.execute_tool("read_text_file", {"path": alias}).result
        assert result["content"] == "candidate-only index"
    result = toolkit.execute_tool("read_text_file", {"path": "global/strategy.md"}).result
    assert result["content"] == "candidate strategy"
    result = toolkit.execute_tool("write_text_file", {"path": "resources/libero/MEMORY.md", "content": "mutated"}).result
    assert "writing to memory is denied" in result["error"]
    assert (memory / "MEMORY.md").read_text() == "candidate-only index"


def test_harness_parse_needs_no_external_memory(tmp_path):
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, False)
    args = parser.parse_args(["--suite", "libero_spatial_task", "--task", "0", "--env-python", "/other/python"])
    args.explore = False
    args.memory_profile = "local"
    args.memory_dir = str(tmp_path / "missing")
    args.harness_dir = str(tmp_path / "candidate")
    args.output_dir = str(tmp_path / "episode")
    assert _parse_config(args).task_desc["suite"] == "libero_spatial_task"
    assert args.env_python == "/other/python"


def test_evolution_launch_uses_real_pro_selector_and_gpu():
    launch = _evolution_launch({"id": "spatial-t00", "task_index": 0, "backend_suite": "libero_spatial_task"},
                               1000, {"env_gpu": 5, "vla_endpoint": "http://vla", "sam_endpoint": "http://sam"})
    args = launch["args"]
    assert args[args.index("--libero-type") + 1] == "pro"
    assert args[args.index("--cuda-device") + 1] == "5"
    assert "--task-name" not in args and launch["env"]["LIBERO_TYPE"] == "pro"


class FakeEnv:
    def __init__(self):
        self.env_steps = 0
        self.terminated = False
        self.truncated = False
        self.drained = 0

    def get_status(self):
        return {"official_success": self.terminated, "done": self.terminated or self.truncated,
                "truncated": self.truncated, "episode_steps": self.env_steps, "max_episode_steps": 2}

    def finish_trajectory(self):
        self.drained += 1

    def public_robot_state(self):
        return {"eef_position": [0, 0, 1], "eef_quaternion": [0, 0, 0, 1],
                "gripper_qpos": [0.04, -0.04], "base_position": [-0.66, 0, 0.912],
                "base_quaternion": [0, 0, 0, 1], "private_object_pose": [9, 9, 9]}

    def get_task_language(self):
        return "move the cup"

    def raw_obs(self):
        return {"agentview_image": np.array([[[0, 0, 0], [0, 0, 0]], [[255, 0, 0], [255, 0, 0]]], dtype=np.uint8),
                "agentview_depth": np.full((2, 2), 2, dtype=np.float32)}

    def get_camera_meta(self, *args):
        return {"intrinsic_K": np.eye(3).tolist(), "extrinsic_cam2world": np.eye(4).tolist()}


def make_adapter(tmp_path):
    env = FakeEnv()
    state = EnvState(tmp_path)
    with state.record_step(state={}, terminated=False, truncated=False) as step:
        state.save("agentview.png", np.zeros((2, 2, 3), dtype=np.uint8), step=step)
        state.save("wrist.png", np.zeros((2, 2, 3), dtype=np.uint8), step=step)
    primitive = SimpleNamespace(env=env)
    toolkit = SimpleNamespace(harness_primitives=primitive, state=state)
    return LiberoHarnessAdapter(toolkit), env, primitive


def test_official_outcome_and_horizon_remain_host_owned(tmp_path):
    adapter, env, _ = make_adapter(tmp_path)
    env.env_steps, env.truncated = 2, True
    outcome = adapter.outcome()
    assert outcome.official_success is False and outcome.terminal_reason == "horizon"
    assert outcome.env_steps == 2 and env.drained == 1
    env.truncated, env.terminated = False, True
    assert adapter.outcome().official_success is True
    assert not adapter.episode_active()
    assert "private_object_pose" not in adapter._robot_state({})


def test_live_rgbd_alignment_geometry_and_audit_payloads(tmp_path):
    adapter, _, _ = make_adapter(tmp_path)
    capture = adapter.capabilities()[RGBD].invoke({"camera": "agentview"})
    import io
    from PIL import Image
    from rpent.planner.recording import PlannerRecorder
    from meta_harness.meta_mcp import live_image_content
    rgb = np.asarray(Image.open(io.BytesIO(capture["_image_cam_bytes"])))
    assert rgb[0, 0, 0] == 255  # Top-left alignment retained.
    assert capture["depth_statistics_m"] == {"min": 2, "median": 2, "max": 2}
    assert capture["metric_lookup"] == {"tool": "back_project", "camera": "agentview", "step": 0, "resolution": "low"}
    assert capture["image_origin"] == "top-left" and capture["depth_unit"] == "meter"
    assert not any(isinstance(value, np.ndarray) for value in capture.values())
    assert not any(key in capture for key in ("_rgb_array", "_depth_m_array", "_world_xyz_array"))
    recorder = PlannerRecorder(tmp_path / "episode", episode_id="test")
    metadata = recorder.serializable(capture)
    assert len(json.dumps(metadata)) < 6000
    _, blocks = live_image_content(metadata, tmp_path)
    assert len([b for b in blocks if b["type"] == "image"]) == 2
    assert all(not name.endswith(".npz") for name in adapter._toolkit.state.latest_record().artifacts)
    json.dumps(adapter.capability_calls())


def test_live_planner_and_meta_both_receive_rgb_and_depth_images(tmp_path):
    from meta_harness.live import LiveBridge
    from meta_harness.meta_mcp import live_image_content
    from rpent.planner.recording import PlannerRecorder
    from rpent.tools.toolkit import ToolResult

    adapter, env, _ = make_adapter(tmp_path)
    original = env.raw_obs()
    env.raw_obs = lambda: {**original, "robot0_eye_in_hand_image": original["agentview_image"],
                          "robot0_eye_in_hand_depth": original["agentview_depth"]}
    recorder = PlannerRecorder(tmp_path / "episode", episode_id="live-images")
    planner = SimpleNamespace(_recorder=recorder, _timeout_s=1)
    bridge = LiveBridge(tmp_path / "mailbox", planner=planner, adapter=adapter, bound=None)
    assert planner._timeout_s is None
    bridge.controller = SimpleNamespace(wait_for_pause=lambda timeout: {"decision_id": "d1"})
    response = bridge.call("observe", {})
    _, blocks = live_image_content(response, tmp_path)
    assert sum(block["type"] == "image" for block in blocks) == 4
    assert len(json.dumps(response)) < 12000
    planner_feedback = ToolResult("observe", adapter._observe({})).content_blocks
    assert sum(block["type"] == "image" for block in planner_feedback) == 4
    assert "_image_depth" not in planner_feedback[0]["text"]
    adapter.live_depth = False
    formal_feedback = ToolResult("observe", adapter._observe({})).content_blocks
    assert sum(block["type"] == "image" for block in formal_feedback) == 2


def test_osc_stops_at_horizon_and_rejects_later_actions(tmp_path):
    adapter, env, primitive = make_adapter(tmp_path)
    def step(action):
        env.env_steps += 1
        env.truncated = env.env_steps == 2
    primitive._step_env = step
    adapter._capture_action = lambda name, args, before, started: {"steps": env.env_steps - before}
    result = adapter.capabilities()[OSC_STEP].invoke({"action": [0] * 7, "steps": 8})
    assert result["steps"] == 2
    with pytest.raises(RuntimeError, match="no longer accepts"):
        adapter.capabilities()[OSC_STEP].invoke({"action": [0] * 7})


def test_artifact_names_cannot_escape_episode(tmp_path):
    adapter, _, _ = make_adapter(tmp_path)
    with pytest.raises(Exception):
        adapter.capabilities()[ARTIFACT_WRITE].invoke({"name": "../outside", "kind": "text", "data": "x"})
    result = adapter.capabilities()[ARTIFACT_WRITE].invoke({"name": "note", "kind": "text", "data": "x"})
    assert result["artifact"].endswith("note.txt")




def test_h0_min_policy_omission_is_prefix_stable_across_turns():
    """Once an image becomes the placeholder it never changes again.

    Provider prompt caching needs a byte-identical prefix. The policy may only
    move its omission watermark forward; on turns where the watermark stays,
    every previously rendered message must be reproduced byte-for-byte.
    """
    path = ROOT / "harnesses/libero_pro_h0_min/policy.py"
    spec = importlib.util.spec_from_file_location("libero_h0_min_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.Policy()

    history = []
    previous_render = None
    advances = 0
    # 100 KB images: the 4 MiB budget holds ~40 images (~20 turns), matching
    # production frame sizes; advances should be absent early, rare later.
    for turn in range(30):
        history.append({"parts": [{"part_kind": "user-prompt", "content": [
            {"kind": "binary", "media_type": "image/png",
             "data": bytes([turn % 256]) * (100 * 1024)}
            for _ in range(2)]}]})
        before = policy._omit_before
        out = policy.prepare_request(SimpleNamespace(messages=history), {})
        render = [[("placeholder" if isinstance(item, str) else item["data"])
                   for item in message["parts"][0]["content"]]
                  for message in out.messages]
        if previous_render is not None:
            if policy._omit_before == before:
                # No advance: the entire previous rendering is a stable prefix.
                assert render[:len(previous_render)] == previous_render
            else:
                # An advance only flips images to the placeholder, permanently.
                advances += 1
                for old, new in zip(previous_render, render):
                    for old_item, new_item in zip(old, new):
                        if old_item != new_item:
                            assert old_item != "placeholder"
                            assert new_item == "placeholder"
        previous_render = render

        # The newest images always survive so the model keeps its current view.
        tail = [item for message in out.messages[-1:]
                for item in message["parts"][0]["content"]]
        assert all(isinstance(item, dict) for item in tail)
        # Budget respected once more than two images exist.
        retained = sum(len(item["data"])
                       for message in out.messages
                       for item in message["parts"][0]["content"]
                       if isinstance(item, dict))
        assert retained <= policy._MAX_HISTORY_IMAGE_BYTES

    assert advances > 0  # the budget did engage by the end of the episode


def test_toolkit_handoff_does_not_close_shared_model_client():
    """Multiple ordinary exploration sessions reuse the CLI-owned model."""
    closed = []
    toolkit = LiberoToolkit.__new__(LiberoToolkit)
    toolkit._primitives = SimpleNamespace(
        env=SimpleNamespace(finish_trajectory=lambda: None),
        model=SimpleNamespace(close=lambda: closed.append(True)),
        stop_recording=lambda: [],
    )
    toolkit.close()
    assert closed == []
