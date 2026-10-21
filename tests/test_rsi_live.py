# Copyright 2026 The RPent Authors.
"""Mailbox interventions retain their actor and leave extra diagnostics separate."""

import asyncio
import threading
from types import SimpleNamespace

from meta_harness.live import LiveBridge
from rpent.harness import PlannerRequest
from rpent.planner.recording import PlannerRecorder


def test_extra_observation_is_serializable_meta_evidence_and_reply_uses_real_menu(
    tmp_path,
):
    recorder = PlannerRecorder(tmp_path / "episode")
    planner = SimpleNamespace(_recorder=recorder)
    binding = SimpleNamespace(invoke=lambda args: {"rgb": b"bytes", "value": 3})
    from robots.libero.harness_adapter import OBSERVE

    adapter = SimpleNamespace(capabilities=lambda: {OBSERVE: binding})
    bridge = LiveBridge(
        tmp_path / "mailbox", planner=planner, adapter=adapter, bound=None
    )
    request = PlannerRequest(
        messages=[],
        instructions="actual instruction",
        model_settings={},
        tools=[
            {
                "name": "finish",
                "description": "done",
                "input_schema": {
                    "type": "object",
                    "properties": {"success": {"type": "boolean"}},
                    "required": ["success"],
                },
            }
        ],
    )
    received = []

    def pause():
        received.append(
            asyncio.run(
                bridge.controller.at_boundary(
                    request, {"episode_id": "episode", "decision_id": "decision-1"}
                )
            )
        )

    thread = threading.Thread(target=pause)
    thread.start()
    assert bridge.controller.wait_for_pause(1)
    observation = bridge.call("observe", {})
    assert observation["planner_feedback"] is False
    assert recorder.stats["actors"] == ["meta"] and not recorder.stats["autonomous"]
    import pytest

    with pytest.raises(ValueError):
        bridge.call(
            "reply",
            {"decision_id": "stale", "tool_name": "finish", "args": {"success": False}},
        )
    bridge.call(
        "reply",
        {
            "decision_id": "decision-1",
            "tool_name": "finish",
            "args": {"success": False},
        },
    )
    thread.join(1)
    assert received[0][1].tool_name == "finish"
    assert recorder.inspect_decision("decision-1")[0]["event"] == "meta_diagnostic"


def test_service_restart_never_reuses_lost_legacy_state(monkeypatch):
    from rpent.utils.rpc import RpcError
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    client = HttpRpcClient("http://127.0.0.1:9")
    client._service_owner = {
        "pid": 1,
        "start_ticks": "old",
        "pgid": 1,
        "boot_id": "old",
    }
    monkeypatch.setattr("rpent.utils.execution.owns_process", lambda owner: False)
    import pytest

    with pytest.raises(RpcError, match="service_state_lost"):
        client.call("vla.predict", args=({}, {}))


def test_closed_live_requests_return_terminal_state_instead_of_waiting_forever(tmp_path):
    import time

    from meta_harness.live import exchange
    from meta_harness.storage import RunStore

    store = RunStore(tmp_path)
    store.write_once("pool/live/old/request.json", {"deadline": time.time() + 20})
    store.write_once("pool/live/old/close.json", {"reason": "closed"})
    result = exchange(store, "old", "new-reply", "reply", {"resume": True})
    assert result["state"] == "closed" and result["new_session_required"]
    assert not store.path("pool/live/old/requests/new-reply.json").exists()
