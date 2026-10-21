# Copyright 2026 The RPent Authors.
"""Persistent model services owned by the shared pool, surviving host restart."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from meta_harness.environment import lower_environment
from meta_harness.storage import RunStore, _json_bytes, _publish
from rpent.utils.execution import (
    file_lock,
    is_locked,
    local_service_owner,
    owns_process,
    signal_owned,
)
from rpent.utils.rpc.http_rpc import HttpRpcClient


def resolve_service_command(command, root: Path, repo: Path):
    values = {**os.environ, "RUN_DIR": str(root), "REPO_ROOT": str(repo)}

    def substitute(match):
        name = match.group(1)
        if not values.get(name):
            raise ValueError(f"model service needs environment variable {name}; source configs/env.sh")
        return values[name]

    return [re.sub(r"@([A-Z][A-Z0-9_]*)@", substitute, item) for item in command]


class ServiceManager:
    def __init__(self, root, config):
        self.store = RunStore(root)
        self.config = config
        self.children = {}
        self.next_health = {}

    def _read(self, ref):
        try:
            return self.store.read(ref)
        except FileNotFoundError:
            return None

    def _write(self, ref, record):
        _publish(self.store.path(ref), _json_bytes(record), replace=True)
        return record

    def tick(self):
        ready = set()
        states = {}
        with file_lock(self.store.path("services/manager.lock")):
            for sid, child in list(self.children.items()):
                if child.poll() is not None:
                    child.wait()
                    del self.children[sid]
            for kind in ("vla", "sam"):
                for index, replica in enumerate(self.config.pool.profile[kind]):
                    sid = f"{kind}-{index}"
                    record = self._tick_one(sid, replica)
                    states[sid] = record
                    if record["state"] == "ready":
                        ready.add(replica["endpoint"])
        return ready, states

    def _tick_one(self, sid, replica):
        prefix = f"services/{sid}"
        ref = f"{prefix}/state.json"
        previous = self._read(ref) or {}
        launch = replica.get("launch")
        owner = self._read(f"{prefix}/owner.json") if previous.get("managed") else previous.get("owner")
        alive = owner is None or owns_process(owner)
        now = time.time()
        if previous.get("state") == "ready" and alive and now < self.next_health.get(sid, 0):
            return previous
        healthy = False
        try:
            if replica.get("session_protocol") == "lingbot":
                # Protocol metadata contains ordinary MessagePack values only.
                # The host environment deliberately doesn't install the GPU
                # deploy package; use its lightweight wire decoder here.
                from msgpack import unpackb
                from websockets.sync.client import connect
                from robots.robotwin.robot_spec import vla_runtime_contract

                with connect(replica["endpoint"], open_timeout=2, close_timeout=1, proxy=None) as channel:
                    metadata = unpackb(channel.recv(timeout=2), raw=False)
                    if metadata != vla_runtime_contract():
                        raise RuntimeError("LingBot service metadata mismatch")
            else:
                client = HttpRpcClient(replica["endpoint"])
                client.call("healthz", timeout_s=0.3)
            if replica.get("session_protocol") == "pi05":
                protocol = client.call("vla.protocol", timeout_s=0.3)
                if protocol != {"protocol": "pi05", "version": 1}:
                    raise RuntimeError("service does not implement the Pi0.5 session protocol")
            healthy = True
        except Exception:
            pass
        if healthy:
            previous_owner = owner
            if (owner is None or not alive) and urlparse(replica["endpoint"]).hostname in {"localhost", "127.0.0.1", "::1"}:
                owner = local_service_owner(replica["endpoint"])
            self.next_health[sid] = now + 10
            if previous.get("state") == "ready" and alive and owner == previous_owner:
                return previous
            return self._write(ref, {**previous, "state": "ready", "endpoint": replica["endpoint"],
                "managed": previous.get("managed", False) and owner == previous_owner,
                "attempt": 0, "owner": owner, "ready_at": now,
                "startup_seconds": now - previous.get("started_at", now)})
        if owner is not None and alive and previous.get("ready_at"):
            # A busy inference process can miss a short health deadline. Never
            # launch a duplicate on its occupied port or kill a borrowed service.
            self.next_health[sid] = now + 5
            return self._write(ref, {**previous, "state": "ready", "health_warning": "health probe timed out while owner is alive"})
        if previous.get("managed") and is_locked(self.store.path(f"{prefix}/lease.lock")):
            if now > previous["deadline"]:
                if owner:
                    signal_owned(owner, 9)
                self._write(ref, {**previous, "state": "failed", "failed_at": now,
                                  "error": "startup/health timeout"})
                raise RuntimeError(f"model service {sid} startup/health timeout; inspect {self.store.path(prefix)}")
            return self._write(ref, {**previous, "state": "starting"})
        if launch is None:
            raise RuntimeError(f"model service unavailable at {replica['endpoint']}; no launch command configured")
        if urlparse(replica["endpoint"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("managed model services require a local endpoint")
        attempt = previous.get("attempt", 0) + 1
        if attempt > 1 + (self.config.evaluation_retries or 0):
            self._write(ref, {**previous, "state": "failed", "failed_at": now,
                              "error": "exhausted startup retries"})
            raise RuntimeError(f"model service {sid} exhausted startup retries; inspect {self.store.path(prefix)}")
        command = resolve_service_command(launch["command"], self.store.root, self.config.repo_root)
        env = lower_environment(extra_names=tuple(launch["env_passthrough"]))
        if replica.get("session_protocol") == "lingbot" and "gpu_device" in replica:
            env["CUDA_VISIBLE_DEVICES"] = str(replica["gpu_device"])
        with file_lock(self.store.path(f"{prefix}/lease.lock"), blocking=False) as fd:
            with self.store.path(f"{prefix}/server.log").open("a") as log:
                child = subprocess.Popen([
                    sys.executable, "-m", "rpent.utils.execution",
                    str(self.store.path(f"{prefix}/owner.json")), str(fd), *command,
                ], cwd=self.config.repo_root, env=env, pass_fds=(fd,), start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            self.children[sid] = child
            # The child publishes its identity before exec and inherits the
            # lease, so a host crash never launches a duplicate live service.
            return self._write(ref, {"state": "starting", "endpoint": replica["endpoint"],
                "managed": True, "attempt": attempt, "started_at": now,
                "deadline": now + launch["timeout_s"], "command": command})

    def close(self):
        """Call only after all episode workers finish; never stop attached services."""
        from meta_harness.campaign_runtime import shared_active, shared_spec
        shared = shared_spec(self.store)
        if shared:
            if shared_active(self.store, {}):
                return
            for root in shared["runs"]:
                state_path = Path(root) / "state.json"
                if not state_path.exists():
                    return
                import json
                if json.loads(state_path.read_text()).get("status") == "running":
                    return
        owners = []
        for path in self.store.path("services").glob("*/state.json"):
            record = self.store.read(path.relative_to(self.store.root).as_posix())
            if record.get("managed"):
                owner = self._read(str(path.parent.relative_to(self.store.root) / "owner.json"))
                if owner:
                    signal_owned(owner)
                    owners.append(owner)
        end = time.monotonic() + 3
        while any(owns_process(owner) for owner in owners) and time.monotonic() < end:
            for child in self.children.values():
                child.poll()
            time.sleep(0.05)
        for owner in owners:
            signal_owned(owner, 9)
        for child in self.children.values():
            child.wait(timeout=3)
