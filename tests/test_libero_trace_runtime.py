"""LIBERO control boundaries and RGB evidence without models or a simulator."""

from __future__ import annotations

import json
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from meta_harness.trajectory import (
    TrajectoryReader,
    history_depth,
    history_frame,
    history_timeline,
)
from robots.libero.env_client import LiberoEnvClient
from robots.libero.env_server import LiberoEnvFacade, _SensorEnv, build_env_cfg
from rpent.planner.recording import decision_scope
from rpent.tools.correlation import root_tool_call_scope
from rpent.trace import (
    TRACE_INDEX,
    RGBDTrajectoryWriter,
    TraceError,
    inspect_trajectory,
    trace_capability_scope,
)


class FakeRLinf:
    """Reuse raw image buffers, as a simulator is allowed to do."""

    def __init__(self, *, limit=6, success_at=None, lose_camera_at=None):
        self.limit = limit
        self.success_at = success_at
        self.lose_camera_at = lose_camera_at
        self.elapsed_steps = np.array([0])
        self.reset_calls = 0
        self.actions = []
        self.closed = False
        self.base_reads = 0
        self.task_descriptions = ["put the bowl on the plate"]
        self.current_raw_obs = [
            {
                "agentview_image": np.zeros((3, 4, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((3, 4, 3), dtype=np.uint8),
                "agentview_depth": np.full((3, 4, 1), 0.25, dtype=np.float32),
                "robot0_eye_in_hand_depth": np.full((3, 4, 1), 0.5, dtype=np.float32),
                "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.array([0.04, -0.04]),
                "robot0_joint_pos": np.arange(7, dtype=float),
                "private_object_position": [90, 91, 92],
            }
        ]
        worker = SimpleNamespace(
            env_call=self.robot_metadata, check_success=lambda: self.success_at == 0
        )
        self.env = SimpleNamespace(workers=[worker], close=self.close)

    def robot_metadata(self, method, *, target):
        assert (method, target) == ("rpent_robot_metadata", "self")
        self.base_reads += 1
        return {"base_position": [0.35, -0.42, 0.1], "base_quaternion": [0, 0, 0, 1]}

    def observation(self):
        raw = self.current_raw_obs[0]
        for index, name in enumerate(("agentview_image", "robot0_eye_in_hand_image")):
            if name in raw:
                raw[name][..., 0] = np.arange(3)[:, None]
                raw[name][..., 1] = int(self.elapsed_steps[0])
                raw[name][..., 2] = 10 + index
        return {
            "main_images": raw["agentview_image"][None],
            "wrist_images": raw.get(
                "robot0_eye_in_hand_image", np.zeros((3, 4, 3), dtype=np.uint8)
            )[None],
            "states": np.zeros((1, 8), dtype=np.float32),
            "task_descriptions": self.task_descriptions,
        }

    def reset(self):
        self.reset_calls += 1
        self.elapsed_steps[:] = 0
        return self.observation(), {}

    def step(self, actions, *, auto_reset):
        assert auto_reset is False
        self.actions.append(np.array(actions[0], copy=True))
        self.elapsed_steps += 1
        step = int(self.elapsed_steps[0])
        if step == self.lose_camera_at:
            self.current_raw_obs[0].pop("robot0_eye_in_hand_image")
        return (
            self.observation(),
            np.array([0.0]),
            np.array([step == self.success_at]),
            np.array([step >= self.limit]),
            {"episode": {"episode_len": [step]}},
        )

    def get_camera_meta(self, *, camera_name, height, width):
        transform = np.eye(4)
        transform[0, 3] = float(self.elapsed_steps[0])
        return {
            "camera_name": camera_name,
            "height": height,
            "width": width,
            "intrinsic_K": np.eye(3),
            "extrinsic_cam2world": transform,
            "depth_near": 0.1,
            "depth_far": 10.0,
        }

    def close(self):
        self.closed = True


class LocalRPC:
    def __init__(self, facade):
        self.facade = facade
        self.calls = []

    def call(self, method, *, args=(), kwargs=None, timeout_s=None):
        self.calls.append(method)
        return self.facade._dispatch(method, args, kwargs or {})


def make_client(root, **kwargs):
    env = FakeRLinf(**kwargs)
    meta = {
        "suite": "libero_spatial_task",
        "task": 0,
        "seed": 1000,
        "max_episode_steps": env.limit,
    }
    facade = LiberoEnvFacade(env, meta=meta)
    rpc = LocalRPC(facade)
    client = LiberoEnvClient(rpc, expected_meta=meta)
    client.enable_full_trace(root, mode="full_rgb")
    client.reset()
    return client, env, rpc


def rows(root):
    return [json.loads(line) for line in (root / TRACE_INDEX).read_text().splitlines()]


def test_chunk_stops_at_first_success_and_records_each_executed_boundary(tmp_path):
    client, env, rpc = make_client(tmp_path, success_at=2)
    actions = np.arange(35, dtype=np.float32).reshape(5, 7) / 35
    context = {"episode_id": "episode-7", "decision_id": "decision-2", "actor": "meta"}
    with (
        decision_scope(None, context),
        root_tool_call_scope("tool-3"),
        trace_capability_scope("libero.vla", 4),
    ):
        obs, rewards, term, trunc, infos = client.chunk_step(
            actions, return_all_frames=True
        )
    coverage = client.finish_trajectory()
    assert rpc.calls.count("env.chunk_step") == 1
    assert rpc.calls.count("env.step") == 0
    assert len(obs) == len(rewards) == len(infos) == 2
    assert term.tolist() == [False, True] and not trunc.any()
    assert client.get_status()["official_success"] is True
    assert client.env_steps == len(env.actions) == 2
    assert coverage["complete"] and coverage["mode"] == "full_rgb"
    assert coverage["initialization"]["settling_recorded"] is False
    assert coverage["initialization"]["settling_control_steps"] == 15
    recorded = rows(tmp_path)
    assert [row["env_step"] for row in recorded] == [0, 1, 2]
    assert recorded[0]["action"] is None
    assert env.base_reads == 1
    for step, row in enumerate(recorded):
        assert set(row["cameras"]) == {"agentview", "wrist"}
        assert row["robot_state"]["base_position"] == [0.35, -0.42, 0.1]
        assert "private_object_position" not in row["robot_state"]
        for camera, sensor in row["cameras"].items():
            pixels = imageio.imread(tmp_path / sensor["rgb"])
            assert pixels[0, 0].tolist() == [
                2,
                step,
                10 if camera == "agentview" else 11,
            ]
            assert "depth" not in sensor and "calibration" not in sensor
        if step:
            assert row["action"]["env_steps_before"] == step - 1
            assert (
                row["action"]["env_action"] == actions[step - 1].astype(float).tolist()
            )
            assert row["decision_id"] == "decision-2" and row["actor"] == "meta"
            assert row["correlation"]["root_tool_call_id"] == "tool-3"
            assert row["correlation"]["capability_sequence"] == 4
    assert not list((tmp_path / "public/trajectory").rglob("*.npz"))
    with pytest.raises(RuntimeError, match="episode ended"):
        rpc.call("env.step", args=(actions[0],))
    assert len(env.actions) == 2
    client.close()
    assert env.closed


def test_horizon_prevents_a_later_chunk_action_from_becoming_success(tmp_path):
    client, env, rpc = make_client(tmp_path, limit=2, success_at=3)
    _, _, term, trunc, _ = client.chunk_step(np.zeros((5, 7)))
    client.finish_trajectory()
    status = client.get_status()
    assert term.tolist() == [False, False] and trunc.tolist() == [False, True]
    assert (
        status["terminal_reason"] == "horizon" and status["official_success"] is False
    )
    assert status["episode_steps"] == len(env.actions) == 2
    with pytest.raises(RuntimeError, match="episode ended"):
        rpc.call("env.chunk_step", args=(np.zeros((5, 7)),))
    client.close()


def test_invalid_chunk_is_rejected_before_any_action(tmp_path):
    client, env, _ = make_client(tmp_path)
    actions = np.zeros((3, 7))
    actions[2, 0] = np.nan
    with pytest.raises(ValueError, match="nonempty finite"):
        client.chunk_step(actions)
    assert not env.actions and client.env_steps == 0
    client.close()


def test_already_successful_initial_state_uses_the_official_predicate(tmp_path):
    client, env, _ = make_client(tmp_path, success_at=0)
    assert client.terminated and client.get_status()["official_success"]
    assert client.get_status()["episode_steps"] == 0
    assert client.finish_trajectory()["observation_count"] == 1
    assert not env.actions
    client.close()


def test_missing_sensor_fails_the_trace_without_claiming_complete(tmp_path):
    client, env, _ = make_client(tmp_path, lose_camera_at=1)
    with pytest.raises(TraceError, match="capture failed"):
        client.step(np.zeros(7))
    with pytest.raises(TraceError):
        client.finish_trajectory()
    assert len(env.actions) == 1
    coverage = inspect_trajectory(tmp_path, 1)
    assert not coverage["complete"]
    with pytest.raises(TraceError):
        client.close()
    assert env.closed


def test_rgb_history_is_complete_without_depth_or_navview(tmp_path):
    client, _, _ = make_client(tmp_path)
    client.step(np.zeros(7))
    coverage = client.finish_trajectory()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "task": {"robot": "libero"},
                "outcome": {"env_steps": 1},
                "public": {"trajectory_metadata": coverage},
            }
        )
    )
    reader = TrajectoryReader.from_episode(tmp_path)
    frame, _ = history_frame(reader, camera="all", env_step=1)
    timeline, _ = history_timeline(reader, camera="all", max_frames=2)
    assert set(frame["cameras"]) == {"agentview", "wrist"}
    assert frame["coverage"]["verified_selected_frame_complete"]
    assert timeline["coverage"]["complete"]
    assert not reader.warnings
    with pytest.raises(ValueError, match="not archived in full_rgb"):
        history_depth(reader, camera="wrist", env_step=1, pixel=[0, 0])
    sensor_path = tmp_path / rows(tmp_path)[1]["cameras"]["wrist"]["rgb"]
    sensor_path.unlink()
    assert not inspect_trajectory(tmp_path, 1)["complete"]


@pytest.mark.parametrize("mode", ["full_rgb", "full_rgbd"])
def test_writer_preserves_rgbd_compatibility_and_mode_specific_validation(
    tmp_path, mode
):
    writer = RGBDTrajectoryWriter(tmp_path, cameras=("agentview",), mode=mode)
    state = {
        "eef_position": [0, 0, 1],
        "eef_quaternion": [0, 0, 0, 1],
        "gripper_qpos": [0.04, -0.04],
        "base_position": [0.3, 0.2, 0],
        "base_quaternion": [0, 0, 0, 1],
    }
    capture = {"rgb": np.zeros((3, 4, 3), dtype=np.uint8)}
    if mode == "full_rgbd":
        capture.update(
            depth_m=np.ones((3, 4)),
            metadata={
                "height": 3,
                "width": 4,
                "camera_name": "agentview",
                "intrinsic": np.eye(3),
                "extrinsic_cam2world": np.eye(4),
                "depth_near": 0.1,
                "depth_far": 10.0,
            },
        )
    writer.append(
        env_step=0, action=None, robot_state=state, captures={"agentview": capture}
    )
    coverage = writer.finish(0)
    assert coverage["complete"] and coverage["mode"] == mode
    camera = rows(tmp_path)[0]["cameras"]["agentview"]
    if mode == "full_rgbd":
        with np.load(tmp_path / camera["depth"]) as stored:
            assert np.array_equal(stored["depth_m"], np.ones((3, 4)))
        (tmp_path / camera["depth"]).unlink()
        assert not inspect_trajectory(tmp_path, 0)["complete"]
    else:
        assert "depth" not in camera and "depth_encoding" not in coverage


def test_base_transform_comes_from_simulator_and_converts_wxyz():
    data = SimpleNamespace(
        get_body_xpos=lambda name: [1.0, 2.0, 3.0],
        get_body_xquat=lambda name: [0.5, 0.5, -0.5, 0.5],
    )
    rob = SimpleNamespace(
        sim=SimpleNamespace(data=data),
        robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"))],
    )
    metadata = _SensorEnv(SimpleNamespace(env=rob)).rpent_robot_metadata()
    assert metadata["base_position"] == [1.0, 2.0, 3.0]
    assert metadata["base_quaternion"] == [0.5, -0.5, 0.5, 0.5]


def test_policy_horizon_does_not_include_reset_settling():
    config = build_env_cfg(max_episode_steps=500)
    assert config.max_episode_steps == 500
    assert config.init_params.ignore_done is True
    assert config.init_params.camera_depths is True
