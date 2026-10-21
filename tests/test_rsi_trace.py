"""Asynchronous RGB-D coverage and separate RSI episode facts."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from rpent.result import EpisodeOutcome, write_episode_result
from rpent.trace import RGBDTrajectoryWriter, TRACE_INDEX, inspect_trajectory


def snapshot(step=0):
    state = {
        "eef_position": [step, 0, 1], "eef_quaternion": [0, 0, 0, 1],
        "base_position": [step, 0, 0], "base_quaternion": [0, 0, 0, 1],
        "gripper_qpos": [0.03, -0.03],
    }
    captures = {}
    for camera in ("agentview", "navview", "wrist"):
        transform = np.eye(4)
        transform[0, 3] = step
        captures[camera] = {
            "rgb": np.full((3, 4, 3), step, dtype=np.uint8),
            "depth_m": np.full((3, 4), step + 1, dtype=np.float64),
            "metadata": {
                "camera_name": camera, "height": 3, "width": 4,
                "intrinsic": np.eye(3), "extrinsic_cam2world": transform,
                "depth_near": 0.01, "depth_far": 100,
            },
        }
    return dict(env_step=step, robot_state=state, captures=captures,
                action=None if not step else {
                    "env_steps_before": step - 1, "env_steps_after": step,
                    "env_action": np.array([step, 0, 0]),
                })


def test_snapshot_owns_pixels_and_calibration_before_async_encode(tmp_path, monkeypatch):
    writer = RGBDTrajectoryWriter(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = writer._write_snapshot

    def hold(row):
        entered.set()
        assert release.wait(5)
        original(row)

    monkeypatch.setattr(writer, "_write_snapshot", hold)
    capture = snapshot()
    writer.append(**capture)
    assert entered.wait(5)
    for sensor in capture["captures"].values():
        sensor["rgb"][:] = 99
        sensor["depth_m"][:] = 99
        sensor["metadata"]["extrinsic_cam2world"][:] = 99
    capture["robot_state"]["eef_position"][0] = 99
    assert not inspect_trajectory(tmp_path, 0)["complete"]
    release.set()
    writer.finish(0)
    row = json.loads((tmp_path / TRACE_INDEX).read_text())
    assert row["observation_id"] == "env-000000"
    assert row["robot_state"]["eef_position"][0] == 0
    for camera, sensor in row["cameras"].items():
        assert np.all(imageio.imread(tmp_path / sensor["rgb"]) == 0)
        with np.load(tmp_path / sensor["depth"]) as arrays:
            assert np.all(arrays["depth_m"] == 1)
        assert sensor["calibration"]["extrinsic_camera_to_world"][0][3] == 0
        assert sensor["capture_id"] == f"env-000000/{camera}"
        assert not any("sha256" in key for key in sensor)


def test_bounded_queue_applies_backpressure_without_dropping_frames(tmp_path, monkeypatch):
    frame_bytes = 3 * (3 * 4 * 3 + 3 * 4 * 4)
    writer = RGBDTrajectoryWriter(tmp_path, queue_size=1, max_pending_bytes=frame_bytes * 2)
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original = writer._write_snapshot

    def hold_first(row):
        if row["env_step"] == 0:
            entered.set()
            assert release.wait(5)
        original(row)

    monkeypatch.setattr(writer, "_write_snapshot", hold_first)
    writer.append(**snapshot(0))
    assert entered.wait(5)
    writer.append(**snapshot(1))
    errors = []

    def append_third():
        try:
            writer.append(**snapshot(2))
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    producer = threading.Thread(target=append_third, daemon=True)
    producer.start()
    assert not returned.wait(0.05)
    assert writer.pending_bytes == frame_bytes * 2
    release.set()
    producer.join(5)
    assert returned.is_set() and not errors
    assert writer.finish(2)["complete"]
    assert writer.peak_pending_bytes <= frame_bytes * 2
    rows = [json.loads(line) for line in (tmp_path / TRACE_INDEX).read_text().splitlines()]
    assert [row["env_step"] for row in rows] == [0, 1, 2]
    assert all(set(row["cameras"]) == {"agentview", "navview", "wrist"} for row in rows)


def test_async_writer_keeps_the_calling_decisions_actor_context(tmp_path):
    from rpent.planner.recording import decision_scope

    writer = RGBDTrajectoryWriter(tmp_path)
    writer.append(**snapshot())
    context = {"episode_id": "ep007", "decision_id": "d03", "actor": "meta"}
    with decision_scope(None, context):
        writer.append(**snapshot(1))
    writer.finish(1)
    row = json.loads((tmp_path / TRACE_INDEX).read_text().splitlines()[1])
    assert {key: row[key] for key in context} == context
    assert {key: row["correlation"][key] for key in context} == context


def test_background_failure_is_visible_and_cannot_publish_completion(tmp_path, monkeypatch):
    writer = RGBDTrajectoryWriter(tmp_path)
    writer.append(**snapshot(0))
    writer.drain()
    assert not inspect_trajectory(tmp_path, 0)["complete"]

    def fail(_row):
        raise OSError("test encoder failed")

    monkeypatch.setattr(writer, "_write_snapshot", fail)
    writer.append(**snapshot(1))
    with pytest.raises(RuntimeError, match="test encoder failed"):
        writer.finish(1)
    assert not writer._worker.is_alive()
    coverage = inspect_trajectory(tmp_path, 1)
    assert not coverage["complete"]
    assert coverage["observation_count"] == 1
    with pytest.raises(RuntimeError, match="previously failed"):
        writer.preflight()


def test_finish_checks_index_without_rereading_any_sensor_bytes(tmp_path, monkeypatch):
    writer = RGBDTrajectoryWriter(tmp_path, flush_every=4)
    for step in range(5):
        writer.append(**snapshot(step))
    writer.drain()

    def cannot_read_bytes(_path):
        raise AssertionError("archive pixels must not be read at finalization")

    monkeypatch.setattr(Path, "read_bytes", cannot_read_bytes)
    assert writer.finish(4)["complete"]
    assert inspect_trajectory(tmp_path, 4)["trace_complete"]


def test_oversized_single_frame_fails_explicitly_instead_of_exceeding_budget(tmp_path):
    writer = RGBDTrajectoryWriter(tmp_path, max_pending_bytes=1)
    with pytest.raises(RuntimeError, match="exceeds writer budget"):
        writer.append(**snapshot())
    with pytest.raises(RuntimeError, match="exceeds writer budget"):
        writer.finish()
    assert writer.peak_pending_bytes == 0
    assert not writer._worker.is_alive()


def test_result_keeps_git_config_claim_and_manual_controller_separate(tmp_path):
    writer = RGBDTrajectoryWriter(tmp_path)
    writer.append(**snapshot())
    writer.finish(0)
    result = json.loads(write_episode_result(
        tmp_path, task={"id": "open"}, harness={"id": "h0", "commit": "a" * 40},
        agent={"model": "qwen3.8-max"}, config_ref="runs/test/config.json",
        controller="planner", actors=["planner", "meta", "planner"],
        agent_claimed_success=True, metrics={},
        outcome=EpisodeOutcome(validity="valid", official_success=False, terminated=False,
                               truncated=False, env_steps=0, terminal_reason="agent_stopped"),
    ).read_text())
    assert result["harness_commit"] == "a" * 40
    assert result["config_ref"] == "runs/test/config.json"
    assert result["controller"] == "mixed"
    assert result["actors"] == ["planner", "meta"]
    assert result["official_success"] is False
    assert result["agent_claimed_success"] is True
    assert result["trace_complete"] is True
    assert "experiment_sha256" not in result


def test_result_before_finish_is_invalid_even_if_all_frames_were_flushed(tmp_path):
    writer = RGBDTrajectoryWriter(tmp_path)
    writer.append(**snapshot())
    writer.drain()
    result = json.loads(write_episode_result(
        tmp_path, task={}, harness={}, agent={}, metrics={},
        outcome=EpisodeOutcome(validity="valid", official_success=True, terminated=True,
                               truncated=False, env_steps=0, terminal_reason="official_success"),
    ).read_text())
    assert result["validity"] == "invalid"
    assert result["trace_complete"] is False
    assert result["official_success"] is None
    assert result["outcome"]["official_success"] is None
    writer.finish(0)


@pytest.mark.parametrize("failed", [False, True])
def test_libero_toolkit_close_flushes_trace_and_exposes_writer_failure(tmp_path, failed):
    from types import SimpleNamespace
    from robots.libero.toolkit import LiberoToolkit

    writer = RGBDTrajectoryWriter(tmp_path)
    writer.append(**snapshot())
    if failed:
        writer.fail(OSError("write failed during episode"))
    stopped = []
    toolkit = LiberoToolkit.__new__(LiberoToolkit)
    toolkit._primitives = SimpleNamespace(
        stop_recording=lambda: stopped.append(writer._closed) or [],
        env=SimpleNamespace(finish_trajectory=lambda: writer.finish(0)),
    )
    if failed:
        with pytest.raises(RuntimeError, match="write failed during episode"):
            toolkit.close()
    else:
        toolkit.close()
    assert stopped == ([] if failed else [True])
    assert not writer._worker.is_alive()
    result = json.loads(write_episode_result(
        tmp_path, task={}, harness={}, agent={}, metrics={},
        outcome=EpisodeOutcome(validity="valid", official_success=True, terminated=True,
                               truncated=False, env_steps=0, terminal_reason="official_success"),
    ).read_text())
    assert result["validity"] == ("invalid" if failed else "valid")
    assert result["trace_complete"] is (not failed)
    assert result["official_success"] is (None if failed else True)
