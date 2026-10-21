# Copyright 2026 The RPent Authors.
"""One durable, fair execution queue for evaluations and live episodes.

The scheduler holds a short metadata lock only. A worker inherits its lease FD
before Popen returns, so a scheduler crash cannot make a live process disappear
from resource accounting. No image scans or GPU admission probes occur here.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from meta_harness.config import EvolutionConfig
from meta_harness.evaluation import (
    collect_evaluation,
    pending_evaluation_cells,
    plan_evaluation,
    record_episode,
    record_episode_result,
)
from meta_harness.storage import HarnessStore, RecordConflict, RunStore, _json_bytes, _publish
from rpent.utils.execution import file_lock, is_locked, owns_process, signal_owned


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_optional(store: RunStore, ref: str, default=None):
    try:
        return store.read(ref)
    except FileNotFoundError:
        return default


def replace_record(store: RunStore, ref: str, value: dict):
    _publish(store.path(ref), _json_bytes(value), replace=True)


def invalid_result(episode: dict, reason: str, elapsed: float = 0) -> dict:
    return {
        **{
            key: episode[key]
            for key in (
                "schema_version",
                "episode_id",
                "evaluation_id",
                "cell_id",
                "execution_attempt",
                "controller",
            )
        },
        "validity": "invalid",
        "official_success": None,
        "agent_claimed_success": None,
        "terminal_reason": reason,
        "env_steps": 0,
        "trace_complete": False,
        "elapsed_seconds": max(0, elapsed),
    }


class ExecutionPool:
    def __init__(self, root: str | Path):
        self.store = RunStore(root)
        self.config = EvolutionConfig.parse(self.store.read("config.json"))
        from meta_harness.resource_profile import effective_config
        self.config, self.resource_profile_revision = effective_config(self.store, self.config)
        from meta_harness.campaign_runtime import effective_config as campaign_config
        self.config = campaign_config(self.store, self.config)
        self.profile = self.config.pool.profile
        if self.profile is None:
            raise ValueError(
                "pool.profile must explicitly assign services and capacity"
            )
        self.harnesses = HarnessStore(root)
        self._children: dict[str, subprocess.Popen] = {}
        self.services = None
        self._ready_services = None
        if any("launch" in replica for kind in ("vla", "sam") for replica in self.profile[kind]):
            from meta_harness.services import ServiceManager

            self.services = ServiceManager(root, self.config)
        # Import only the old stage-3 prototype layout once. Thereafter active
        # references are a small durable outbox, independent of run length.
        with file_lock(self.store.path("pool/scheduler.lock")):
            if not self.store.path("pool/active-index.json").exists():
                for path in self.store.path("pool/jobs").glob("*/spec.json"):
                    eid = path.parent.name
                    if (
                        is_locked(path.parent / "lease.lock")
                        or not self.store.path(f"episodes/{eid}/result.json").exists()
                    ):
                        self.store.write_once(
                            f"pool/active/{eid}.json",
                            self.store.read(
                                path.relative_to(self.store.root).as_posix()
                            ),
                        )
                self.store.write_once("pool/active-index.json", {"schema_version": 1})

    def submit(self, **kwargs) -> str:
        if self.config.robot == "robotwin":
            for cell in kwargs["cells"]:
                self._require_development_scene(cell["task_id"], cell["seed"])
                if cell["repeat"] >= self.config.parallel.repeats:
                    raise ValueError("RoboTwin policy repeat outside the declared development matrix")
        # The immutable request itself is the queue entry; no second publication
        # is required for dispatch after a caller dies just after persistence.
        request = plan_evaluation(self.store, **kwargs)
        return request["evaluation_id"]

    def _require_development_scene(self, task_id, seed):
        if self.config.robot == "robotwin":
            task = self.task(task_id)
            if task.seeds is None or seed not in task.seeds:
                raise ValueError("RoboTwin scene is outside the declared development split")

    def evaluation_status(self, evaluation_id: str) -> dict:
        summary = collect_evaluation(self.store, evaluation_id)
        from meta_harness.policy_rejection import rejection
        rejected=rejection(self.store,evaluation_id)
        if rejected:
            summary.update(state='rejected',rejected=True,rejection=rejected,
                comparable=False,exhausted=True,queue={k:0 for k in ('ready','active','backoff','exhausted')})
            return summary
        active = self.active()
        pending = self.pending_cells(summary, active_episode_ids=set(active))
        summary["queue"] = {key: len(value) for key, value in pending.items()}
        summary["exhausted"] = bool(pending["exhausted"])
        return summary

    def pending_cells(self, summary, *, active_episode_ids):
        """One admission policy shared by dispatch and host exhaustion checks."""
        from meta_harness.resource_profile import current_revision

        revision = current_revision(self.store)
        unlimited = bool(revision and (
            self.config.robot == "libero" and "geometry-memory" in revision
            or self.config.robot == "robotwin" and "per-task-unlimited" in revision
        ))
        return pending_evaluation_cells(
            summary, max_retries=None if unlimited else self.config.evaluation_retries,
            active_episode_ids=active_episode_ids,
        )

    def live_start(
        self,
        *,
        session_id: str,
        scope_id: str,
        commit: str,
        task_id: str,
        seed: int,
        deadline: float,
        mode: str = "planner",
    ) -> dict:
        if not self.config.development.live_enabled:
            raise PermissionError("Live tools are disabled for this run's ablation protocol")
        from meta_harness.evaluation import (
            _record_commit,
            _record_id,
            _record_reference,
        )

        _record_id(session_id, "session_id")
        _record_reference(scope_id, "scope_id")
        _record_commit(commit)
        if mode not in {"planner", "capability"}:
            raise ValueError("live mode must be planner or capability")
        self.task(task_id)
        self._require_development_scene(task_id, seed)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be nonnegative integer")
        request = {
            "session_id": session_id,
            "scope_id": scope_id,
            "harness_commit": commit,
            "task_id": task_id,
            "seed": seed,
            "repeat": 0,
            "deadline": deadline,
            "mode": mode,
        }
        self.store.write_once(f"pool/live/{session_id}/request.json", request)
        return {"session_id": session_id, "state": "queued"}

    def task(self, task_id: str):
        return next(
            task
            for suite in self.config.suites
            for task in suite.tasks
            if task.id == task_id
        )

    def active(self) -> dict[str, dict]:
        active = {}
        for path in self.store.path("pool/active").glob("*.json"):
            episode_id = path.stem
            if is_locked(self.store.path(f"pool/jobs/{episode_id}/lease.lock")):
                active[episode_id] = self.store.read(
                    path.relative_to(self.store.root).as_posix()
                )
        from meta_harness.suite_transfer import source_reservations
        active.update(source_reservations(self.store))
        return active

    def _reap(self):
        for episode_id, process in list(self._children.items()):
            if process.poll() is not None:
                process.wait()
                del self._children[episode_id]
        active = self.active()
        for path in self.store.path("pool/active").glob("*.json"):
            episode_id = path.stem
            spec = self.store.read(path.relative_to(self.store.root).as_posix())
            result = read_optional(self.store, f"episodes/{episode_id}/result.json")
            if episode_id in active:
                owner = read_optional(self.store, f"pool/jobs/{episode_id}/owner.json")
                orphan = owner is not None and not owns_process(owner)
                if (spec["deadline"] is not None and time.time() >= spec["deadline"]) or orphan:
                    cleanup_path = f"pool/jobs/{episode_id}/cleanup.json"
                    cleanup = read_optional(self.store, cleanup_path)
                    if cleanup is None:
                        cleanup = self.store.write_once(cleanup_path, {"started_at": time.time()})
                    sig = 15 if time.time() < cleanup["started_at"] + 5 else 9
                    if owner:
                        signal_owned(owner, sig)
                    runner = read_optional(self.store, f"pool/jobs/{episode_id}/runner-owner.json")
                    if runner:
                        signal_owned(runner, sig)
            elif result is None:
                self.store.write_once(f"pool/jobs/{episode_id}/spec.json", spec)
                record_episode(self.store, spec["episode"])  # finish a torn allocation
                record_episode_result(
                    self.store,
                    episode_id,
                    invalid_result(
                        spec["episode"],
                        "worker_interrupted",
                        time.time() - spec["allocated_at"],
                    ),
                )
            if episode_id not in active:
                path.unlink(missing_ok=True)
        return active

    def _resources(self, active: dict) -> dict | None:
        from meta_harness.campaign_runtime import shared_active, shared_spec, ready_endpoints
        local_count = len(active)
        store = getattr(self, "store", None)
        from meta_harness.admission_guard import episode_allowed
        if store is not None and not episode_allowed(store, local_count):
            return None
        if local_count >= self.profile["max_episodes"]:
            return None
        active = shared_active(store, active)
        shared = shared_spec(store)
        if shared and len(active) >= shared["max_episodes"]:
            return None
        counts = defaultdict(int)
        for job in active.values():
            res = job["episode"]["resources"]
            for kind in ("vla", "sam"):
                if res.get(f"{kind}_endpoint"):
                    counts[kind, res[f"{kind}_endpoint"]] += 1
            counts["env", res["env_gpu"]] += 1
        chosen = {}
        shared_ready = ready_endpoints(store)
        for kind in ("vla", "sam"):
            if kind == "sam" and not self.profile[kind]:
                chosen[kind] = {"endpoint": None}
                continue
            replicas = [
                replica
                for replica in self.profile[kind]
                if counts[kind, replica["endpoint"]] < replica["capacity"]
                and (self._ready_services is None or replica["endpoint"] in self._ready_services)
                and (shared_ready is None or replica["endpoint"] in shared_ready)
            ]
            if not replicas:
                return None
            chosen[kind] = min(replicas, key=lambda r: counts[kind, r["endpoint"]])
        devices = [device for device in self.profile["env_devices"]
                   if not shared or counts["env", device] < shared["max_episodes_per_gpu"]]
        from meta_harness.campaign_runtime import admissible_env_devices
        devices = admissible_env_devices(store, active, devices)
        if not devices:
            return None
        gpu = min(devices, key=lambda device: counts["env", device])
        return {
            "env_gpu": gpu,
            "vla_endpoint": chosen["vla"]["endpoint"],
            "sam_endpoint": chosen["sam"]["endpoint"],
            "session_protocol": chosen["vla"]["session_protocol"],
        }

    def request_admitted(self, request, state=None):
        """Defer later suites without cancelling their already-running workers."""
        from meta_harness.admission_guard import scope_allowed
        store = getattr(self, "store", None)
        if store is not None and not scope_allowed(store, request["scope_id"], request.get("purpose", "diagnostic")):
            return False
        if self.config.scheduling != "pipeline":
            return True
        control = read_optional(self.store, "suite-control.json", {})
        if state is None:
            state = read_optional(self.store, "state.json", {})
        if "suite_index" not in state and not control:  # Standalone diagnostic pool.
            return True
        suite_id = control.get("active_suite", self.config.suites[0].id)
        return request["scope_id"].split("/", 1)[0] == suite_id

    def tick(self) -> dict:
        from meta_harness.campaign_runtime import admission_lock
        with admission_lock(self.store):
            return self._tick()

    def _tick(self) -> dict:
        if self.services is not None:
            self._ready_services, _ = self.services.tick()
        with file_lock(self.store.path("pool/scheduler.lock")):
            from meta_harness.suite_transfer import receive_results, waiting_for_import
            receive_results(self.store)
            active = self._reap()
            groups = defaultdict(deque)
            summaries = {}
            admission_state = read_optional(self.store, "state.json", {})
            for path in sorted(self.store.path("evaluations").glob("*/request.json")):
                eid = path.parent.name
                if self.store.path(f"evaluations/{eid}/complete.json").exists():
                    summaries[eid] = {"state": "complete", "exhausted": False}
                    continue
                request = self.store.read(path.relative_to(self.store.root).as_posix())
                if not self.request_admitted(request, admission_state):
                    continue
                if read_optional(self.store, f"evaluations/{eid}/cancel.json"):
                    continue
                from meta_harness.policy_rejection import rejection
                if rejection(self.store,eid):
                    continue
                summary = collect_evaluation(self.store, eid)
                if summary["state"] == "complete":
                    self.store.write_once(
                        f"evaluations/{eid}/complete.json",
                        {"selected_episodes": summary["selected_episodes"]},
                    )
                pending = self.pending_cells(summary, active_episode_ids=set(active))
                # Ephemeral scheduling metadata; immutable request/seed records
                # remain unchanged. Finishing a nearly complete H0 unlocks Meta.
                if request.get("purpose") == "baseline":
                    request["_baseline_remaining"] = len(request["cells"]) - summary["valid"]
                summaries[eid] = {
                    "state": summary["state"],
                    "exhausted": bool(pending["exhausted"]),
                }
                for cell in pending["ready"]:
                    if waiting_for_import(self.store, eid, cell["cell_id"]):
                        continue
                    groups[request["scope_id"]].append(("evaluation", request, cell))
            for path in sorted(self.store.path("pool/live").glob("*/request.json")):
                req = self.store.read(path.relative_to(self.store.root).as_posix())
                sid = req["session_id"]
                if (
                    self.request_admitted(req, admission_state)
                    and not self.store.path(f"pool/jobs/live-{sid}/spec.json").exists()
                    and not self.store.path(f"pool/live/{sid}/close.json").exists()
                    and (req["deadline"] is None or time.time() < req["deadline"])
                ):
                    groups[req["scope_id"]].appendleft(("live", req, None))
            if "development_slots" in self.profile:
                launched = self._admit_pipeline(groups, active)
                return {"active": len(active), "launched": launched, "evaluations": summaries}
            cursor = read_optional(self.store, "pool/cursor.json", {"scope": ""})[
                "scope"
            ]
            scopes = sorted(groups)
            scopes = [s for s in scopes if s > cursor] + [
                s for s in scopes if s <= cursor
            ]
            queue = deque(scopes)
            priority = read_optional(self.store, "pool/scheduling-priority.json", {})
            preferred = priority.get("scope_id")
            scope_state = read_optional(self.store, "state.json", {}).get("scopes", {}).get(preferred, {})
            if scope_state.get("rounds_completed", 0) >= priority.get("until_round", 1):
                preferred = None
            if preferred in queue:
                queue.remove(preferred)
                queue.appendleft(preferred)
            launched = []
            while queue:
                resources = self._resources(active)
                if resources is None:
                    break
                scope = queue.popleft()
                kind, request, cell = groups[scope].popleft()
                spec = self._allocate(kind, request, cell, resources)
                active[spec["episode"]["episode_id"]] = spec
                launched.append(spec["episode"]["episode_id"])
                replace_record(self.store, "pool/cursor.json", {"scope": scope})
                if groups[scope]:
                    if scope == preferred:
                        queue.appendleft(scope)
                    else:
                        queue.append(scope)
            return {
                "active": len(active),
                "launched": launched,
                "evaluations": summaries,
            }

    def _admit_pipeline(self, groups, active):
        """Separate baseline/candidate lanes prevent initial matrices starving iteration.

        Each lane fills its oldest matrix first. Reservations are soft and
        borrowed slots return on episode exit; running work is never preempted.
        """
        queues = {"baseline": [], "formal": [], "development": []}
        def category(kind, request):
            if kind == "live" or request.get("purpose") == "development":
                return "development"
            return "baseline" if request.get("purpose") == "baseline" else "formal"
        for items in groups.values():
            for kind, request, cell in items:
                ref = (f"evaluations/{request['evaluation_id']}/request.json" if kind == "evaluation"
                       else f"pool/live/{request['session_id']}/request.json")
                queues[category(kind, request)].append((self.store.path(ref).stat().st_mtime_ns, ref, kind, request, cell))
        for key in queues:
            queues[key] = deque(sorted(queues[key], key=lambda item: (item[0], item[1])))
        from meta_harness.campaign_runtime import control
        rotate_baselines = control(self.store).get("baseline_task_round_robin", False)
        if rotate_baselines:
            by_scope = defaultdict(deque)
            for item in queues["baseline"]:
                by_scope[item[3]["scope_id"]].append(item)
            cursor = read_optional(self.store, "pool/baseline-task-cursor.json", {}).get("scope", "")
            scopes = sorted(by_scope)
            scopes = [scope for scope in scopes if scope > cursor] + [
                scope for scope in scopes if scope <= cursor]
            queues["baseline"] = deque()
            finish_ready = control(self.store).get("baseline_finish_ready_first", False)
            priorities = sorted({by_scope[scope][0][3].get("_baseline_remaining", 0)
                                 for scope in scopes}) if finish_ready else [None]
            for priority in priorities:
                cohort = [scope for scope in scopes if by_scope[scope] and (priority is None or
                    by_scope[scope][0][3].get("_baseline_remaining", 0) == priority)]
                while cohort:
                    for scope in cohort:
                        queues["baseline"].append(by_scope[scope].popleft())
                    cohort = [scope for scope in cohort if by_scope[scope]]
        reserve = self.profile["development_slots"]
        evaluation_slots = self.profile["max_episodes"] - reserve
        limits = {"development": reserve, "baseline": evaluation_slots // 2,
                  "formal": evaluation_slots - evaluation_slots // 2}
        # If one evaluation lane has no queued work, its sibling can use all
        # evaluation slots before borrowing the development reservation.
        if not queues["baseline"]:
            limits["formal"] = evaluation_slots
        if not queues["formal"]:
            limits["baseline"] = evaluation_slots
        launched = []
        while any(queues.values()):
            resources = self._resources(active)
            if resources is None:
                break
            counts = {key: 0 for key in queues}
            for spec in active.values():
                counts[category(spec["kind"], spec["episode"]) ] += 1
            eligible = [key for key in queues if queues[key] and counts[key] < limits[key]]
            if not eligible:
                eligible = [key for key in queues if queues[key]]
            # Shared-host capacity can be below this run's nominal lane sizes.
            # Age alone otherwise lets old H0 requests monopolize every slot
            # before their lane reaches its nominal limit, starving candidates.
            chosen = min(eligible, key=lambda key: (
                counts[key] / max(1, limits[key]), queues[key][0][:2]))
            _, _, kind, request, cell = queues[chosen].popleft()
            spec = self._allocate(kind, request, cell, resources)
            eid = spec["episode"]["episode_id"]
            active[eid] = spec
            launched.append(eid)
            if rotate_baselines and chosen == "baseline":
                replace_record(self.store, "pool/baseline-task-cursor.json", {
                    "scope": request["scope_id"]})
        return launched

    def _allocate(self, kind, request, cell, resources):
        if kind == "evaluation":
            eid = request["evaluation_id"]
            episode_id = f"{eid}-{cell['cell_id']}-a{cell['execution_attempt']:04d}"
            condition = cell
            controller = request["controller"]
            purpose = request["purpose"]
        else:
            eid = None
            episode_id = f"live-{request['session_id']}"
            condition = request
            controller, purpose = (
                ("mixed" if request["mode"] == "planner" else "meta"),
                "diagnostic",
            )
        allocated = time.time()
        protocol = resources.pop("session_protocol")
        resources["session_id"] = episode_id
        episode = {
            "schema_version": 1,
            "episode_id": episode_id,
            "evaluation_id": eid,
            "cell_id": cell["cell_id"] if cell else None,
            "execution_attempt": cell["execution_attempt"] if cell else 1,
            **{key: condition[key] for key in ("task_id", "seed", "repeat")},
            "policy_seed": self.config.episode.policy_seed
            + condition["seed"] * 10000
            + condition["repeat"],
            "harness_commit": request["harness_commit"],
            "config_ref": "config.json",
            "purpose": purpose,
            "controller": controller,
            "resources": resources,
            "started_at": now(),
        }
        queue_ref = (
            f"evaluations/{eid}/request.json"
            if eid
            else f"pool/live/{request['session_id']}/request.json"
        )
        spec = {
            "episode": episode,
            "resource_profile_revision": self.resource_profile_revision,
            "runtime_revision": read_optional(self.store, "runtime-current.json", {}).get("revision"),
            **({"planner_timeout_s": int(self.config.episode.timeout_s)}
               if self.config.scheduling == "pipeline" else {}),
            "kind": kind,
            "scope_id": request["scope_id"],
            "queued_at": self.store.path(queue_ref).stat().st_mtime,
            "session_protocol": protocol,
            "allocated_at": allocated,
            # An outage must not kill and reset the simulator while its model
            # request waits. Physical action/turn limits remain unchanged.
            "deadline": None,
            "mode": request.get("mode"),
            "live_session_id": request.get("session_id"),
        }
        job = f"pool/jobs/{episode_id}"
        with file_lock(self.store.path(f"{job}/lease.lock"), blocking=False) as fd:
            # Publish spec before the allocation, allowing recovery to finish it.
            self.store.write_once(f"pool/active/{episode_id}.json", spec)
            self.store.write_once(f"{job}/spec.json", spec)
            record_episode(self.store, episode)
            with self.store.path(f"{job}/worker.log").open("a") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "meta_harness.pool_worker",
                        str(self.store.root),
                        episode_id,
                        str(fd),
                    ],
                    cwd=self.config.repo_root,
                    pass_fds=(fd,),
                    start_new_session=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self._children[episode_id] = process
        # Do not LOCK_UN the inherited descriptor: file_lock closes only the
        # parent's copy; the child's open file description keeps the lease.
        return spec

    def cancel_scope(self, scope_id: str, *, evaluations: bool = True):
        with file_lock(self.store.path("pool/scheduler.lock")):
            for path in self.store.path("evaluations").glob("*/request.json"):
                req = self.store.read(path.relative_to(self.store.root).as_posix())
                if evaluations and req["scope_id"] == scope_id and req["purpose"] == "development":
                    ref = f"evaluations/{req['evaluation_id']}/cancel.json"
                    # A prior cancellation may carry immutable rejection proof.
                    # Scope cleanup must preserve it, including on crash replay.
                    if read_optional(self.store, ref) is None:
                        try:
                            self.store.write_once(ref, {"reason": "development_finished"})
                        except RecordConflict:
                            if read_optional(self.store, ref) is None:
                                raise
            for path in self.store.path("pool/live").glob("*/request.json"):
                req = self.store.read(path.relative_to(self.store.root).as_posix())
                if req["scope_id"] == scope_id:
                    self.store.write_once(
                        f"pool/live/{req['session_id']}/close.json",
                        {"reason": "closed"},
                    )
            for eid, spec in self.active().items():
                if spec["scope_id"] == scope_id and (
                    spec["episode"]["purpose"] == "diagnostic"
                    or evaluations and spec["episode"]["purpose"] == "development"
                ):
                    owner = read_optional(self.store, f"pool/jobs/{eid}/owner.json")
                    if owner:
                        signal_owned(owner)
