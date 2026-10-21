"""Host-owned campaign overrides and shared admission, preserving run facts."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import time

from rpent.utils.execution import file_lock, is_locked, owns_process


def control(store):
    if store is None:
        return {}
    path = store.path("campaign-runtime.json")
    return json.loads(path.read_text()) if path.exists() else {}


def effective_config(store, config):
    spec = control(store)
    if not spec:
        return config
    meta = spec["meta"]
    endpoint = meta["provider"]["base_url"].rstrip("/")
    if endpoint != "http://127.0.0.1:18445/v1":
        raise ValueError("campaign requests must use the approved shared budget relay")
    command, index = [], 0
    while index < len(config.episode.command):
        token = config.episode.command[index]
        if token == "--base-url":
            index += 2
            continue
        if not token.startswith("--base-url="):
            command.append(token)
        index += 1
    command.extend(("--base-url", endpoint))
    return replace(config,
                   meta=replace(config.meta, model=meta["model"],
                                reasoning_effort=meta["reasoning_effort"],
                                provider=meta["provider"]),
                   episode=replace(config.episode, command=tuple(command)))


def shared_spec(store):
    return control(store).get("shared_pool")


def admission_lock(store):
    spec = shared_spec(store)
    return file_lock(Path(spec["root"]) / "resource-admission.lock") if spec else nullcontext()


def shared_active(store, local):
    spec = shared_spec(store)
    if not spec:
        return local
    result = {str(store.root) + "/" + key: value for key, value in local.items()}
    for root_value in spec["runs"]:
        root = Path(root_value)
        if root.resolve() == store.root.resolve():
            continue
        for path in (root / "pool/active").glob("*.json"):
            job = json.loads(path.read_text())
            episode_id = job["episode"]["episode_id"]
            folder = root / "pool/jobs" / episode_id
            owner_path = folder / "owner.json"
            owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
            if is_locked(folder / "lease.lock") or owner and owns_process(owner):
                result[str(root) + "/" + episode_id] = job
    return result


def ready_endpoints(store):
    spec = shared_spec(store)
    if not spec:
        return None
    owner = Path(spec["service_owner"])
    endpoints = set()
    for path in (owner / "services").glob("*/state.json"):
        value = json.loads(path.read_text())
        identity_path = path.parent / "owner.json"
        identity = (json.loads(identity_path.read_text())
                    if value.get("managed") and identity_path.exists() else value.get("owner"))
        if value.get("state") == "ready" and identity and owns_process(identity):
            endpoints.add(value["endpoint"])
    return endpoints


def admissible_env_devices(store, active, devices):
    """Keep 64 logical leases, but wait for real VRAM and serialized startup."""
    spec = shared_spec(store)
    if not spec or not spec.get("startup_memory_mib"):
        return list(devices)
    path = Path(spec["root"]) / "gpu-memory.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if time.time() - previous.get("checked_at", 0) >= 2:
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                text=True, timeout=3)
            previous = {"checked_at": time.time(), "free_mib": {
                left.strip(): int(right.strip()) for left, right in
                (line.split(",") for line in output.splitlines())}}
            from meta_harness.storage import _publish, _json_bytes
            _publish(path, _json_bytes(previous), replace=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            return []  # Pause admissions; keep all existing scenes alive.
    initializing = {}
    for key, job in active.items():
        root_value, episode_id = key.rsplit("/", 1)
        marker = Path(root_value) / "episodes" / episode_id / "rollout/public/trajectory.jsonl"
        if not marker.exists() or marker.stat().st_size == 0:
            device=job["episode"]["resources"]["env_gpu"]
            initializing[device]=initializing.get(device,0)+1
    startup=spec["startup_memory_mib"]
    reserve=spec.get("inference_reserve_mib",0)
    parallel=spec.get("max_parallel_env_startups_per_gpu",1)
    # Reserve the full measured startup footprint even when a pending worker
    # has not allocated CUDA memory yet. Higher concurrency never spends the
    # same free VRAM twice or removes the inference reserve.
    return [device for device in devices if initializing.get(device,0)<parallel
            and previous["free_mib"].get(str(device),0)
            >= startup*(1+initializing.get(device,0))+reserve]
