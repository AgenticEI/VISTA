# Copyright 2026 The RPent Authors.
"""Host mailbox bridge at the real RPent model boundary (no second Planner)."""

from __future__ import annotations

import threading
import time

from meta_harness.storage import RunStore
from rpent.harness import ToolDecision, inspect_harness
from rpent.planner.boundary import DecisionController
from rpent.planner.recording import decision_scope


class LiveBridge:
    def __init__(self, mailbox, *, planner, adapter, bound):
        self.records = RunStore(mailbox)
        self.controller = DecisionController(pause_next=True)
        planner._decision_controller = self.controller
        self.planner, self.adapter, self.bound = planner, adapter, bound
        if hasattr(adapter, "live_depth"):
            adapter.live_depth = True
        # Paused human/Meta inspection is not charged to Planner solve time.
        planner._timeout_s = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1)

    def call(self, operation, arguments):
        paused = self.controller.wait_for_pause(timeout=0)
        recorder = self.planner._recorder
        if operation == "inspect_harness":
            return inspect_harness(self.bound)
        if operation == "decision":
            if paused is None:
                self.controller.pause_next()
                return {"state": "waiting_for_boundary"}
            return {"state": "paused", **recorder.serializable(paused)}
        if paused is None:
            raise ValueError("wait for a live decision before interacting")
        if operation == "reply":
            if arguments.get("decision_id") != paused["decision_id"]:
                raise ValueError("reply must identify the current paused decision")
            if arguments.get("resume"):
                self.controller.resume()
            else:
                self.controller.pause_next()
                self.controller.replace(
                    ToolDecision(arguments["tool_name"], arguments.get("args", {}))
                )
            return {"state": "released", "decision_id": paused["decision_id"]}
        if operation in {"capability", "observe"}:
            capability = "observation.public@1" if operation == "observe" else arguments["capability"]
            binding = self.adapter.capabilities()[capability]
            # This is a separate Meta experiment, not model-visible feedback.
            with decision_scope(
                recorder,
                {
                    "episode_id": recorder.episode_id,
                    "decision_id": paused["decision_id"],
                    "actor": "meta",
                },
            ):
                recorder.mark_actor("meta")
                result = binding.invoke(arguments.get("args", {}))
                recorder.write(
                    "meta_diagnostic",
                    result,
                    capability=capability,
                    planner_feedback=False,
                )
            return {"result": recorder.serializable(result), "planner_feedback": False}
        raise ValueError(f"unknown live operation: {operation}")

    def serve(self):
        self.records.write_once("ready.json", {"ready": True})
        while not self.stopped.is_set():
            for path in sorted(self.records.path("requests").glob("*.json")):
                response_ref = f"responses/{path.name}"
                if self.records.path(response_ref).exists():
                    continue
                request = self.records.read(
                    path.relative_to(self.records.root).as_posix()
                )
                try:
                    result = self.call(
                        request["operation"], request.get("arguments", {})
                    )
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.records.write_once(response_ref, response)
            self.stopped.wait(0.025)


def terminal_status(store, session_id):
    root = f"pool/live/{session_id}"
    request = store.read(f"{root}/request.json")
    state = None
    if store.path(f"{root}/close.json").exists():
        state = "closed"
    elif store.path(f"episodes/live-{session_id}/result.json").exists():
        state = "finished"
    elif (request["deadline"] is not None and time.time() >= request["deadline"]):
        state = "expired"
    return ({"state": state, "session_id": session_id, "episode_id": f"live-{session_id}",
             "new_session_required": True,
             "message": "This scene has ended. Read its history or live_start with a new unique session_id."}
            if state else None)


def exchange(store, session_id, request_id, operation, arguments=None):
    from meta_harness.evaluation import _record_id

    for value in (session_id, request_id):
        _record_id(value, "live ID")
    root = f"pool/live/{session_id}"
    if terminal := terminal_status(store, session_id):
        return terminal
    store.write_once(
        f"{root}/requests/{request_id}.json",
        {"operation": operation, "arguments": arguments or {}},
    )
    try:
        return store.read(f"{root}/responses/{request_id}.json")
    except FileNotFoundError:
        return {"state": "pending", "request_id": request_id}
