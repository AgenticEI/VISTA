# Copyright 2026 The RPent Authors.
"""Transfer a suite's evidence, including results arriving from draining workers."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from meta_harness.storage import RunStore, _json_bytes, _publish
from rpent.utils.execution import is_locked


def copy_tree(source, target, *, exclude=(), immutable=False):
    """Copy mutable records; link finalized evidence without duplicating large media."""
    source, target = Path(source), Path(target)
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in exclude:
            continue
        destination = target / child.name
        if child.is_symlink():
            raise ValueError(f"cannot export a source symlink: {child}")
        if child.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            flags = ["-a", "--link", "--remove-destination"] if immutable else ["-a", "--reflink=auto"]
            subprocess.run(["cp", *flags, "--", str(child) + "/.",
                            str(destination)], check=True, capture_output=True)
        else:
            subprocess.run(["cp", "--reflink=auto", "--preserve=mode,timestamps", "--",
                            str(child), str(destination)], check=True, capture_output=True)


def copy_episode(source_root, target_root, episode_id):
    source_root, target_root = Path(source_root), Path(target_root)
    source = source_root / "episodes" / episode_id
    target = target_root / "episodes" / episode_id
    # Publish the authoritative outer result last, after images and journals.
    copy_tree(source, target, exclude={"result.json", "rollout"})
    if (source / "rollout").exists():
        (target / "rollout").mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", "--link", "--remove-destination", "--",
                        str(source / "rollout") + "/.", str(target / "rollout")],
                       check=True, capture_output=True)
    if (source / "result.json").exists():
        shutil.copy2(source / "result.json", target / "result.json")


def transfer_state(store):
    path = store.path("transfer/state.json")
    return store.read("transfer/state.json") if path.exists() else None


def source_reservations(store):
    """Read only resource leases; other suites never enter target score/history tables."""
    state = transfer_state(store)
    if not state:
        return {}
    source = Path(state["source_root"])
    active = {}
    for eid in state["draining_episode_ids"]:
        lease = source / f"pool/jobs/{eid}/lease.lock"
        if lease.exists() and is_locked(lease):
            active[eid] = json.loads((source / f"pool/jobs/{eid}/spec.json").read_text())
    return active


def waiting_for_import(store, evaluation_id, cell_id):
    state = transfer_state(store)
    return bool(state and any(item["evaluation_id"] == evaluation_id and item["cell_id"] == cell_id
                              for item in state["pending_episodes"].values()))


def receive_results(store):
    """Called under the target pool lock before admission; no source records change."""
    state = transfer_state(store)
    if not state:
        return
    source = Path(state["source_root"])
    changed = False
    for eid in list(state["pending_episodes"]):
        result = source / f"episodes/{eid}/result.json"
        lease = source / f"pool/jobs/{eid}/lease.lock"
        alive = lease.exists() and is_locked(lease)
        if not result.exists() and alive:
            continue
        copy_episode(source, store.root, eid)
        if not result.exists():
            from meta_harness.evaluation import record_episode_result
            from meta_harness.pool import invalid_result
            episode = store.read(f"episodes/{eid}/episode.json")
            record_episode_result(store, eid, invalid_result(episode, "source_worker_interrupted"))
        store.write_once(f"transfer/received/{eid}.json", {
            "source_result": str(result), "source_result_available": result.exists(),
        })
        state["pending_episodes"].pop(eid)
        changed = True
    active = source_reservations(store)
    remaining = [eid for eid in state["draining_episode_ids"] if eid in active]
    if remaining != state["draining_episode_ids"]:
        state["draining_episode_ids"] = remaining
        changed = True
    if changed:
        _publish(store.path("transfer/state.json"), _json_bytes(state), replace=True)
