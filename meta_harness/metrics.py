# Copyright 2026 The RPent Authors.
"""Small-record timing and cost rollup; never re-read RGB/depth artifacts."""

import json
from collections import defaultdict
from datetime import datetime

from meta_harness.pool import read_optional
from meta_harness.storage import RunStore


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "seconds_sum": sum(values),
        "p50_s": ordered[(len(ordered) - 1) // 2] if ordered else None,
        "p95_s": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        if ordered
        else None,
    }


def report(root, evaluation_id=None):
    store = RunStore(root)
    groups = defaultdict(list)
    episodes = []
    cost = defaultdict(lambda: {"attempts": 0, "seconds": 0, "env_steps": 0})
    starts = []
    ends = []
    events = []
    for path in sorted(store.path("pool/jobs").glob("*/spec.json")):
        spec = store.read(path.relative_to(store.root).as_posix())
        episode = spec["episode"]
        eid = episode["episode_id"]
        if evaluation_id and episode["evaluation_id"] != evaluation_id:
            continue
        result = read_optional(store, f"episodes/{eid}/result.json")
        if not result:
            continue
        end = datetime.fromisoformat(result["completed_at"]).timestamp()
        start = spec["allocated_at"]
        starts.append(start)
        ends.append(end)
        events += [(start, 1), (end, -1)]
        cost[episode["purpose"]]["attempts"] += 1
        cost[episode["purpose"]]["seconds"] += result["elapsed_seconds"]
        cost[episode["purpose"]]["env_steps"] += result["env_steps"]
        launch = read_optional(store, f"pool/jobs/{eid}/launch.json")
        if launch:
            groups["prepare"].append(launch["prepare_seconds"])
        queued = spec.get("queued_at")
        if queued:
            groups["pool_queue"].append(start - queued)
        for timing in store.path(f"episodes/{eid}/timing").glob("*.jsonl"):
            for line in timing.read_text().splitlines():
                value = json.loads(line)
                groups[value["stage"]].append(value["seconds"])
        meta = read_optional(
            store, f"episodes/{eid}/rollout/public/trajectory_meta.json", {}
        )
        for key in (
            "snapshot_seconds",
            "encoding_seconds",
            "writing_seconds",
            "backpressure_seconds",
        ):
            if key in meta:
                groups["trace." + key].append(meta[key])
        episodes.append(
            {
                "episode_id": eid,
                "controller": result["controller"],
                "validity": result["validity"],
                "steps": result["env_steps"],
                "seconds": result["elapsed_seconds"],
                "official_success": result["official_success"],
                "start": start,
                "end": end,
            }
        )
    peak = active = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(active, peak)
    wall = max(ends) - min(starts) if ends else 0
    valid = sum(
        e["validity"] == "valid" and e["controller"] == "planner" for e in episodes
    )
    autonomous_steps = sum(
        e["steps"]
        for e in episodes
        if e["controller"] == "planner" and e["validity"] == "valid"
    )
    return {
        "episodes": episodes,
        "peak_overlap": peak,
        "wall_seconds": wall,
        "valid_autonomous_episodes": valid,
        "valid_episodes_per_hour": valid * 3600 / wall if wall else None,
        "autonomous_env_steps_per_second": autonomous_steps / wall if wall else None,
        "invalid_attempts": sum(e["validity"] == "invalid" for e in episodes),
        "cost": dict(cost),
        "timings": {k: distribution(v) for k, v in sorted(groups.items())},
        "timing_limits": [
            "RPC timings include server queue and execution; legacy services do not expose that split.",
            "Overlapping stages are not additive wall time.",
        ],
    }
