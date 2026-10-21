"""Read-only navigation over the existing episode, Planner and physical journals."""
from __future__ import annotations

import json
from pathlib import Path


def journal(path):
    """Yield complete records; tolerate only an interrupted final JSONL line."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                if not line.endswith("\n") and not stream.read(1):
                    return
                raise ValueError(f"Malformed journal record: {path}:{number}") from None
            yield number, value


def identity(rollout, result=None):
    rollout = Path(rollout)
    result = result or {}
    outer = rollout.parent / "episode.json"
    episode = json.loads(outer.read_text()) if rollout.name == "rollout" and outer.is_file() else {}
    task = result.get("task", {})
    return {
        "episode_id": episode.get("episode_id", rollout.parent.name if rollout.name == "rollout" else rollout.name),
        "task_id": episode.get("task_id"), "task": task,
        "seed": episode.get("seed", task.get("seed")),
        "repeat": episode.get("repeat", task.get("repeat_index")),
        "policy_seed": episode.get("policy_seed"),
        "execution_attempt": episode.get("execution_attempt"),
        "evaluation_id": episode.get("evaluation_id"),
        "harness_commit": episode.get("harness_commit", result.get("harness_commit", result.get("harness", {}).get("commit"))),
        "controller": episode.get("controller", result.get("controller")),
        "purpose": episode.get("purpose"),
    }


def decisions(rollout):
    """A small decision directory, with exact links into control-step evidence."""
    rollout = Path(rollout)
    groups = {}

    def entry(did):
        return groups.setdefault(did, {"decision_id": did, "actors": [], "events": [],
            "tool_calls": [], "first_env_step": None, "last_env_step": None,
            "physical_step_count": 0, "physical_spans": [], "root_tool_call_ids": []})

    for number, row in journal(rollout / "planner/events.jsonl"):
        did = row.get("decision_id")
        if not did:
            continue
        item = entry(did)
        actor = row.get("actor")
        if actor and actor not in item["actors"]:
            item["actors"].append(actor)
        event = row.get("event")
        item["events"].append({"line": number, "seq": row.get("seq"), "event": event})
        if event in {"tool_call", "tool_rejected"}:
            payload = row.get("payload", {})
            call = payload.get("call", payload)
            item["tool_calls"].append({k: call.get(k) for k in ("tool_name", "tool_call_id", "args")})
    for _, row in journal(rollout / "public/trajectory.jsonl"):
        step = row.get("env_step")
        if not isinstance(step, int) or step == 0:
            continue
        correlation = row.get("correlation") or {}
        did = row.get("decision_id") or correlation.get("decision_id")
        if not did:
            continue
        item = entry(did)
        item["physical_step_count"] += 1
        item["first_env_step"] = step if item["first_env_step"] is None else min(step, item["first_env_step"])
        item["last_env_step"] = step if item["last_env_step"] is None else max(step, item["last_env_step"])
        tid = correlation.get("root_tool_call_id")
        if tid and tid not in item["root_tool_call_ids"]:
            item["root_tool_call_ids"].append(tid)
        actor = row.get("actor") or correlation.get("actor")
        if actor and actor not in item["actors"]:
            item["actors"].append(actor)
        span_identity = {"actor": actor, "root_tool_call_id": tid,
                         "capability_id": correlation.get("capability_id"),
                         "capability_sequence": correlation.get("capability_sequence")}
        spans = item["physical_spans"]
        if (spans and spans[-1]["last_env_step"] + 1 == step
                and all(spans[-1][k] == v for k, v in span_identity.items())):
            spans[-1]["last_env_step"] = step
        else:
            spans.append({**span_identity, "first_env_step": step, "last_env_step": step,
                          "before_env_step": step - 1})
    for item in groups.values():
        if item["first_env_step"] is not None:
            item["before_env_step"] = item["first_env_step"] - 1
            item["after_env_step"] = item["last_env_step"]
    return list(groups.values())


def decision_detail(rollout, decision_id):
    directory = decisions(rollout)
    selected = next((x for x in directory if x["decision_id"] == decision_id), None)
    if selected is None:
        raise ValueError("decision_id is not recorded; use history_decisions")
    rows = [row for _, row in journal(Path(rollout) / "planner/events.jsonl") if row.get("decision_id") == decision_id]
    return {"decision": selected, "records": rows,
            "evidence_files": {"planner": str(Path(rollout) / "planner/events.jsonl"),
                               "trajectory": str(Path(rollout) / "public/trajectory.jsonl")},
            "note": "SDK request, actual HTTP attempts, raw tool returns and Planner feedback retain separate event types. Frame ranges are inclusive executed steps; use before_env_step for the preceding observation."}
