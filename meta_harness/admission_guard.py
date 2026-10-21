"""Optional batch-local launch admission; never stops an active worker."""
import json
import shutil
import time
from pathlib import Path

GIB = 2**30

def control(store):
    p = store.path("admission-control.json")
    return json.loads(p.read_text()) if p.exists() else None

def scope_allowed(store, scope_id, purpose=None):
    c = control(store)
    if c is None:
        return True
    if c.get("stop_launches", True):
        return False
    allowed = c.get("allowed_task_ids")
    if allowed is not None and scope_id.split("/task-", 1)[-1] not in allowed:
        return False
    return not c.get("baseline_only", False) or purpose == "baseline"

def storage_allowed(store, free_bytes=None):
    c = control(store)
    if c is None:
        return True
    from meta_harness.pool import replace_record
    free = shutil.disk_usage(store.root).free if free_bytes is None else free_bytes
    stop = (c["episode_reserve_gib"] * c["max_active_episodes"]
            + c["meta_reserve_gib"] * c["max_active_meta"]
            + c["shared_floor_gib"]) * GIB
    status_path = store.path("admission-status.json")
    last = json.loads(status_path.read_text()) if status_path.exists() else {}
    # Hysteresis prevents repeated admissions at the low-space boundary.
    threshold = stop + (c["resume_margin_gib"] * GIB if last.get("space_paused") else 0)
    paused = free < threshold
    status = {"checked_at": time.time(), "free_bytes": free,
              "stop_bytes": stop, "required_bytes": threshold,
              "space_paused": paused, "stop_launches": c.get("stop_launches", True)}
    if paused != last.get("space_paused") or time.time() - last.get("checked_at", 0) >= 10:
        replace_record(store, "admission-status.json", status)
    return not c.get("stop_launches", True) and not paused

def episode_allowed(store, active_count):
    c = control(store)
    if c is None:
        return True
    return active_count < c["max_active_episodes"] and storage_allowed(store)

def meta_allowed(store, scope_id, active_count):
    c = control(store)
    if c is None:
        return True
    return (active_count < c["max_active_meta"]
            and scope_allowed(store, scope_id, "development") and storage_allowed(store))
