# Copyright 2026 The RPent Authors.
"""One durable development attempt: workspace, deadline, requests and submission."""

from __future__ import annotations

import json
import time
from pathlib import Path

from meta_harness.inspection import inspect_package, validate_package
from meta_harness.pool import ExecutionPool, read_optional, replace_record
from meta_harness.storage import HarnessStore, RunStore
from rpent.utils.execution import file_lock

RUNTIME = (Path(__file__).parent / "assets/skills/evolve-harness/SKILL.md").read_text(encoding="utf-8")


def runtime_description(config):
    from rpent.robots import get_robot_spec
    platform_runtime = get_robot_spec(config.robot).harness_runtime
    if config.robot == 'robotwin' and '--robotwin-feedback-profile' in config.episode.command:
        command = config.episode.command
        if command[command.index('--robotwin-feedback-profile') + 1] == 'native':
            platform_runtime = platform_runtime.replace(
                'Official task success is recorded by the host, not\nexposed as policy feedback.',
                'Native RPent task-success feedback remains visible to the Planner in every frozen package.')
            platform_runtime += ('\nThis run uses the exact executed native H0 baseline. Its package has '
                                 'memory alias failures: the host maps the guide to memory/guides/GUIDE_RPENT.md '
                                 'and task_only requests to memory/task-specific, while H0 stored the guide '
                                 'at memory/GUIDE_RPENT.md and references under memory/task_only. '
                                 'Repairing these package paths is an allowed Harness change and must be '
                                 'included in candidate evaluation; do not modify the host runtime.\n')
    if not config.development.live_enabled:
        return ((Path(__file__).parent / "assets/rollout_only_runtime.md").read_text(encoding="utf-8")
                + "\nPlatform: " + config.robot + "\n" + platform_runtime)
    return (RUNTIME + f"\nLive ablation protocol: perform {config.development.live_trials} distinct live trials per development round. "
            "A trial is one unique live_start session inspected at a real decision boundary; "
            "polling or repeated observations do not count as new trials. Zero disables live for the ablation. "
            "Close each session after collecting useful evidence. Record session IDs, hypothesis, "
            "intervention, observations and the resulting package change in package memory. "
            "Infrastructure-invalid trials are reported separately and replaced. "
            "For merge repair, use relevant donor evidence and test the integrated package.\n"
            + "Platform: " + config.robot + "\n" + platform_runtime)


def package_files(harnesses, workspace):
    # Include deletions and drafts explicitly; Git's small-file guard applies.
    tracked = harnesses._git("ls-files", "-z", cwd=workspace).split("\0")
    untracked = harnesses._git(
        "ls-files", "--others", "--exclude-standard", "-z", cwd=workspace
    ).split("\0")
    return sorted(set(filter(None, tracked + untracked)))


class MetaAttempt:
    def __init__(self, root, attempt_id):
        self.store = RunStore(root)
        self.prefix = f"meta/attempts/{attempt_id}"
        self.context = self.store.read(f"{self.prefix}/context.json")
        self.attempt_id = attempt_id
        self.pool = ExecutionPool(root)
        self.config = self.pool.config
        self.harnesses = HarnessStore(root)
        self.workspace = self.store.path(self.context["workspace"])

    @classmethod
    def create(
        cls,
        root,
        *,
        attempt_id,
        scope_id,
        selected_commit,
        selected_evaluation,
        initial_base_commit,
        task_ids,
        round_index,
        kind="development",
        donors=None,
        objective=None,
    ):
        store = RunStore(root)
        harnesses = HarnessStore(root)
        workspace = harnesses.workspace(scope_id, initial_base_commit)
        prefix = f"meta/attempts/{attempt_id}"
        config = ExecutionPool(root).config
        with file_lock(store.path(f"{prefix}/lock")):
            store.write_once(
                f"{prefix}/context.json",
                {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "scope_id": scope_id,
                    "selected_commit": selected_commit,
                    "selected_evaluation": selected_evaluation,
                    "initial_base_commit": initial_base_commit,
                    "task_ids": task_ids,
                    "round_index": round_index,
                    "kind": kind,
                    "donors": donors or [],
                    "workspace": str(workspace.relative_to(store.root)),
                    "runtime": runtime_description(config),
                    "objective": objective,
                },
            )
            if not store.path(f"{prefix}/budget.json").exists():
                started = time.time()
                store.write_once(
                    f"{prefix}/budget.json",
                    {
                        "started_at": started,
                        "deadline": (started + config.development.timeout_s
                                     if config.development.time_limit_enabled else None),
                        "stop_developing_at": (started + config.development.timeout_s
                                               - config.development.finish_reserve_s
                                               if config.development.time_limit_enabled else None),
                        "max_evaluation_requests": config.development.max_evaluation_requests,
                    },
                )
        return cls(root, attempt_id)

    def _round_evaluation_budget(self):
        """Count development trials across every recovery of this task and round."""
        if not getattr(self, "context", None) or self.context["kind"] != "development":
            return None
        control = read_optional(self.store, "development-evaluation-budget-current.json")
        from meta_harness.evaluation import _record_id
        revision = None
        request_limit = self.store.read(f"{self.prefix}/budget.json")["max_evaluation_requests"]
        cell_limit = None
        if control is not None:
            revision = _record_id(control["revision"], "development evaluation budget revision")
            policy = self.store.read(f"development-evaluation-budgets/{revision}.json")
            request_limit = policy["max_requests_per_round"]
            cell_limit = policy["max_cells_per_round"]
        if (type(request_limit) is not int or request_limit < 1
                or cell_limit is not None and (type(cell_limit) is not int or cell_limit < 1)):
            raise ValueError("development evaluation limits must be positive integers")
        scope_key = self.context["scope_id"].replace("/", "-")
        round_key = f"{scope_key}-development-r{self.context['round_index']:02d}"
        requests = cells = 0
        paths = set(self.store.path("meta/attempts").glob(
            f"{round_key}-d*/evaluations/*/request.json"))
        # Public MetaAttempt callers can use non-scheduler attempt IDs. Keep
        # their own durable requests in the same ledger as canonical recoveries.
        paths.update(self.store.path(f"{self.prefix}/evaluations").glob("*/request.json"))
        for path in paths:
            saved = self.store.read(path.relative_to(self.store.root).as_posix())
            requests += 1
            cells += len(saved["cells"])
        return request_limit, cell_limit, requests, cells, revision, round_key

    def budget(self):
        grant = self.store.read(f"{self.prefix}/budget.json")
        requests = len(
            list(self.store.path(f"{self.prefix}/evaluations").glob("*/request.json"))
        )
        round_budget = self._round_evaluation_budget()
        request_limit = grant["max_evaluation_requests"]
        if round_budget is not None:
            request_limit = min(request_limit, round_budget[0])
            requests = round_budget[2]
        return {
            **grant,
            "max_evaluation_requests": request_limit,
            "remaining_seconds": (max(0, grant["deadline"] - time.time()) if grant["deadline"] is not None else None),
            "development_seconds_left": (max(0, grant["stop_developing_at"] - time.time())
                                         if grant["stop_developing_at"] is not None else None),
            "evaluation_requests_used": requests,
            "evaluation_requests_remaining": max(
                0, request_limit - requests
            ),
            **({"evaluation_cells_used": round_budget[3],
                "evaluation_cells_remaining": (max(0, round_budget[1] - round_budget[3])
                                               if round_budget[1] is not None else None),
                "development_evaluation_budget_revision": round_budget[4]}
               if round_budget is not None else {}),
            "submission": read_optional(self.store, f"{self.prefix}/submission.json"),
        }

    def _editable(self):
        if self.store.path(f"{self.prefix}/finished.json").exists():
            raise ValueError("attempt is finished")
        if self.budget()["development_seconds_left"] is not None and self.budget()["development_seconds_left"] <= 0:
            raise ValueError(
                "development deadline reached; host is finishing the attempt"
            )
        if self.budget()["submission"]:
            raise ValueError("attempt already submitted")

    def _freeze(self, message):
        started = time.monotonic()
        from rpent.harness import load_harness

        load_harness(
            self.workspace
        )  # Fast syntax/manifest check, without executing code.
        files = package_files(self.harnesses, self.workspace)
        # The store validates resources before staging/committing, including a
        # resolved MERGE_HEAD. A trial freeze is not a final Meta submission.
        commit = self.harnesses.freeze(self.workspace, files=files, message=message)
        validate_package(
            self.harnesses.materialize(commit),
            timeout_s=min(10, max(0.1, self.budget()["remaining_seconds"] if self.budget()["remaining_seconds"] is not None else 10)),
            robot=self.config.robot,
        )
        return commit, time.monotonic() - started

    def evaluate(self, request_id, cells):
        from meta_harness.evaluation import _record_id, validate_evaluation_request

        _record_id(request_id, "request_id")
        validate_evaluation_request(
            {
                "schema_version": 1,
                "evaluation_id": request_id,
                "purpose": "development",
                "scope_id": self.context["scope_id"],
                "harness_commit": self.context["selected_commit"],
                "config_ref": "config.json",
                "controller": "planner",
                "cells": cells,
            }
        )
        prefix = f"{self.prefix}/evaluations/{request_id}"
        scope_key = self.context["scope_id"].replace("/", "-")
        round_key = f"{scope_key}-development-r{self.context['round_index']:02d}"
        # The shared lock prevents two recovered attempts from spending the
        # same last trial at once. Existing request IDs remain idempotent.
        with (file_lock(self.store.path(f"meta/development-evaluation-locks/{round_key}.lock")),
              file_lock(self.store.path(f"{self.prefix}/lock"))):
            saved = read_optional(self.store, f"{prefix}/request.json")
            if saved is not None:
                if saved["cells"] != cells:
                    raise ValueError(
                        "evaluation request ID reused with different cells"
                    )
            else:
                from meta_harness.round_cap import phase_for
                if phase_for(self) != "development":
                    raise ValueError("Meta generation cap reached; finalize and submit without a new evaluation")
                self._editable()
                budget = self.budget()
                if not budget["evaluation_requests_remaining"]:
                    raise ValueError(
                        "autonomous evaluation request quota exhausted; submit remains available"
                    )
                remaining_cells = budget.get("evaluation_cells_remaining")
                if remaining_cells is not None and len(cells) > remaining_cells:
                    raise ValueError(
                        "autonomous evaluation scene quota exhausted; submit remains available"
                    )
                if any(
                    cell["task_id"] not in self.context["task_ids"] for cell in cells
                ):
                    raise ValueError("cell task outside this scope")
                commit, elapsed = self._freeze(f"Development evaluation {request_id}")
                saved = {"cells": cells, "commit": commit, "freeze_seconds": elapsed}
                self.store.write_once(f"{prefix}/request.json", saved)
            eid = f"dev-{self.attempt_id}-{request_id}"
            return {
                "evaluation_id": self.pool.submit(
                    evaluation_id=eid,
                    harness_commit=saved["commit"],
                    scope_id=self.context["scope_id"],
                    purpose="development",
                    cells=cells,
                )
            }

    def submit(self, notes, *, origin="meta"):
        with file_lock(self.store.path(f"{self.prefix}/lock")):
            previous = read_optional(self.store, f"{self.prefix}/submission.json")
            if previous:
                return previous
            if origin == "meta":
                self._editable()
                from meta_harness.token_gate import TokenGate
                from meta_harness.round_cap import configured_limits
                if not configured_limits(self):
                    TokenGate(self).check_submit()
            commit, elapsed = self._freeze(f"{origin} submission {self.attempt_id}")
            if origin not in {"meta", "round_limit"} and commit == self.context["initial_base_commit"]:
                raise ValueError("no changed loadable candidate")
            result = {
                "commit": commit,
                "origin": origin,
                "notes": notes,
                "submitted_at": time.time(),
                "freeze_seconds": elapsed,
                "autonomously_verified": False,
            }
            self.store.write_once(f"{self.prefix}/submission.json", result)
            return result

    def live_start(self, session_id, task_id, seed, mode="planner"):
        if not self.config.development.live_enabled:
            raise PermissionError("Live tools are disabled for this run's ablation protocol")
        from meta_harness.evaluation import _record_id

        _record_id(session_id, "session_id")
        with file_lock(self.store.path(f"{self.prefix}/lock")):
            existing = read_optional(self.store, f"pool/live/{session_id}/request.json")
            if existing:
                expected = {
                    "scope_id": self.context["scope_id"],
                    "task_id": task_id,
                    "seed": seed,
                    "mode": mode,
                }
                if any(existing[k] != v for k, v in expected.items()):
                    raise ValueError("live session ID reused with different conditions")
                from meta_harness.live import terminal_status

                if terminal := terminal_status(self.store, session_id):
                    return terminal
                return {"session_id": session_id, "state": "queued_or_running"}
            self._editable()
            from meta_harness.round_cap import phase_for
            if phase_for(self) != "development":
                raise ValueError("Meta generation cap reached; no new live scene during finalization")
            if task_id not in self.context["task_ids"]:
                raise ValueError("live task outside scope")
            commit, _ = self._freeze("Freeze live experiment")
            return self.pool.live_start(
                session_id=session_id,
                task_id=task_id,
                seed=seed,
                mode=mode,
                scope_id=self.context["scope_id"],
                commit=commit,
                deadline=self.budget()["stop_developing_at"],
            )

    def select_parent(self, commit):
        with file_lock(self.store.path(f"{self.prefix}/lock")):
            self._editable()
            if self.context["kind"] == "merge":
                raise ValueError(
                    "resolve the merge in place; its parents are the selected donors"
                )
            saved = self.harnesses.select_parent(
                self.workspace,
                commit,
                files=package_files(self.harnesses, self.workspace),
            )
            number = (
                len(list(self.store.path(f"{self.prefix}/parents").glob("*.json"))) + 1
            )
            event = {"parent": commit, "saved_draft": saved}
            self.store.write_once(f"{self.prefix}/parents/p{number:04d}.json", event)
            return event

    def finish(self, *, reason):
        previous = read_optional(self.store, f"{self.prefix}/finished.json")
        if previous:
            return previous
        submission = read_optional(self.store, f"{self.prefix}/submission.json")
        error = None
        if submission is None and reason in ("deadline", "token_budget"):
            try:
                submission = self.submit(
                    "Host snapshot; not a claim of Meta verification.",
                    origin="deadline_snapshot",
                )
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
        # Submitted packages keep their already-queued/running autonomous tests.
        # Closing live diagnostics must not destroy pending evaluation evidence.
        self.pool.cancel_scope(self.context["scope_id"], evaluations=submission is None)
        grant = self.store.read(f"{self.prefix}/budget.json")
        result = {
            "state": "submitted" if submission else "no_candidate",
            "submission": submission,
            "reason": reason,
            "error": error,
            "elapsed_seconds": time.time() - grant["started_at"],
        }
        self.store.write_once(f"{self.prefix}/finished.json", result)
        return result

    def _history_task_ids(self):
        """Suite-local history for independent pipeline experiments."""
        config = getattr(self, "config", None)
        if config is None or config.scheduling != "pipeline":
            return None
        tasks = set(self.context["task_ids"])
        suites = [suite for suite in config.suites
                  if tasks and tasks <= {task.id for task in suite.tasks}]
        if len(suites) != 1:
            raise ValueError("Meta history requires one unambiguous suite")
        return {task.id for task in suites[0].tasks}

    def _history_evaluation_allowed(self, evaluation_id):
        from meta_harness.evaluation import _record_id
        _record_id(evaluation_id, "evaluation_id")
        allowed = self._history_task_ids()
        if allowed is None:
            return True
        request = self.store.read(f"evaluations/{evaluation_id}/request.json")
        tasks = {cell["task_id"] for cell in request["cells"]}
        return bool(tasks) and tasks <= allowed

    def evaluation_status(self, evaluation_id):
        if not self._history_evaluation_allowed(evaluation_id):
            raise ValueError("evaluation is outside this Meta suite")
        return self.pool.evaluation_status(evaluation_id)

    def history_summary(self):
        return [
            {
                "evaluation_id": path.parent.name,
                **{
                    key: value
                    for key, value in self.pool.evaluation_status(
                        path.parent.name
                    ).items()
                    if key
                    in {"harness_commit", "purpose", "controller", "success_rate", "state", "cost",
                           "requested", "valid", "invalid", "denominator", "by_task", "by_seed", "deployment_time"}
                },
            }
            for path in sorted(self.store.path("evaluations").glob("*/request.json"))
            if self._history_evaluation_allowed(path.parent.name)
        ]

    def history_episode(self, episode_id):
        from meta_harness.evaluation import _record_id

        _record_id(episode_id, "episode_id")
        episode = self.store.read(f"episodes/{episode_id}/episode.json")
        allowed = self._history_task_ids()
        if allowed is not None and episode["task_id"] not in allowed:
            raise ValueError("episode is outside this Meta suite")
        rollout = self.store.path(f"episodes/{episode_id}/rollout")
        return {
            "episode": episode,
            "result": read_optional(self.store, f"episodes/{episode_id}/result.json"),
            "rollout": str(rollout),
            "evidence": {name: {"path": str(rollout / relative), "available": (rollout / relative).is_file()}
                         for name, relative in {"planner_events": "planner/events.jsonl",
                             "control_steps": "public/trajectory.jsonl", "capability_calls": "public/capability_calls.jsonl",
                             "outcome": "result.json", "video": "episode.mp4"}.items()},
            "navigation": {"decisions": {"tool": "history_decisions", "episode_id": episode_id},
                           "timeline": {"tool": "history_timeline", "episode_id": episode_id, "camera": "all"}},
        }

    def history_episodes(self, task_id=None, seed=None, repeat=None, evaluation_id=None,
                         controller=None, validity=None, offset=0, limit=20):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("offset >= 0 and limit in [1, 100] are required")
        allowed = self._history_task_ids()
        if allowed is not None and task_id is not None and task_id not in allowed:
            raise ValueError("task is outside this Meta suite")
        if evaluation_id is not None and not self._history_evaluation_allowed(evaluation_id):
            raise ValueError("evaluation is outside this Meta suite")
        rows = []
        for path in sorted(self.store.path("episodes").glob("*/episode.json")):
            episode = self.store.read(path.relative_to(self.store.root).as_posix())
            if allowed is not None and episode["task_id"] not in allowed:
                continue
            if any(value is not None and episode.get(key) != value for key, value in
                   (("task_id", task_id), ("seed", seed), ("repeat", repeat),
                    ("evaluation_id", evaluation_id), ("controller", controller))):
                continue
            result = read_optional(self.store, f"episodes/{path.parent.name}/result.json")
            if validity is not None and (result or {}).get("validity") != validity:
                continue
            rows.append({**{key: episode.get(key) for key in
                ("episode_id", "task_id", "seed", "repeat", "policy_seed", "execution_attempt",
                 "evaluation_id", "harness_commit", "controller", "purpose")},
                "result": result, "history_episode": {"episode_id": path.parent.name}})
        return {"episodes": rows[offset:offset + limit], "total": len(rows),
                "next_offset": offset + limit if offset + limit < len(rows) else None}

    def _relocate_history(self, value):
        """Keep stored journals exact, but resolve their image links in this run."""
        lineage = read_optional(self.store, "suite-export.json", {})
        old_root = lineage.get("source_root")
        if not old_root:
            return value
        def visit(item):
            if isinstance(item, list):
                return [visit(child) for child in item]
            if not isinstance(item, dict):
                return item
            result = {key: visit(child) for key, child in item.items()}
            path = result.get("path")
            if isinstance(path, str) and path.startswith(old_root + "/"):
                replacement = self.store.root / path[len(old_root) + 1:]
                if replacement.is_file():
                    result.update(original_path=path, path=str(replacement))
            return result
        return visit(value)

    def history_decisions(self, episode_id):
        from meta_harness.history import decisions
        episode = self.history_episode(episode_id)
        return {"episode": episode["episode"], "decisions": decisions(episode["rollout"])}

    def history_decision(self, episode_id, decision_id):
        from meta_harness.history import decision_detail
        episode = self.history_episode(episode_id)
        return self._relocate_history({"episode": episode["episode"], **decision_detail(episode["rollout"], decision_id)})

    def live_decision(self, **args):
        from meta_harness.meta_mcp import invoke
        return invoke(self, "live_decision", args)

    def live_observe(self, **args):
        from meta_harness.meta_mcp import invoke
        return invoke(self, "live_observe", args)

    def live_capability(self, **args):
        from meta_harness.meta_mcp import invoke
        return invoke(self, "live_capability", args)

    def live_reply(self, **args):
        from meta_harness.meta_mcp import invoke
        return invoke(self, "live_reply", args)

    def live_close(self, **args):
        from meta_harness.meta_mcp import invoke
        return invoke(self, "live_close", args)

    def inspect_decision(self, episode_id, decision_id):
        from meta_harness.history import journal
        episode = self.history_episode(episode_id)
        return [row for _, row in journal(Path(episode["rollout"]) / "planner/events.jsonl")
                if row.get("decision_id") == decision_id]

    def opening_images(self):
        """Attach real before/after RGB to the initial Meta prompt, with provenance."""
        from meta_harness.trajectory import TrajectoryReader
        material = self.store.read(self.prefix + "/opening-material.json")
        example = material.get("recorded_example") or {}
        span = example.get("physical_span")
        images = []
        error = None
        if span:
            try:
                rollout = self.store.path(f"episodes/{example['episode_id']}/rollout")
                reader = TrajectoryReader.from_episode(rollout)
                for step in dict.fromkeys((span["before_env_step"], span["after_env_step"])):
                    row = reader.row(env_step=step)
                    for camera in reader.cameras:
                        path = reader.asset(reader.camera(row, camera), "rgb")
                        if path:
                            images.append({"episode_id": example["episode_id"], "env_step": step,
                                           "camera": camera, "path": str(path)})
            except (OSError, ValueError, KeyError) as exc:
                error = str(exc)
        replace_record(self.store, self.prefix + "/opening-images.json",
                       {"images": images, "error": error, "transport": "Codex initial --image attachments"})
        return tuple(Path(item["path"]) for item in images)

    def opening(self):
        preview_error = None
        inspection_source = str(self.workspace)
        try:
            inspection = inspect_package(self.workspace, robot=self.config.robot)
        except Exception as exc:
            # Broken drafts and conflicted manifests are precisely what a
            # recovery/merge Meta must repair. Keep them intact and show the
            # selected parent as a clearly labelled reference menu.
            preview_error = {"type": type(exc).__name__, "message": str(exc)}
            parent = self.harnesses.materialize(self.context["selected_commit"])
            inspection_source = str(parent)
            inspection = inspect_package(parent, robot=self.config.robot)
        example = None
        eid = self.context["selected_evaluation"]
        optimization = None
        if eid:
            summary = self.pool.evaluation_status(eid)
            from meta_harness.selection import objective
            optimization = objective(self.config.selection, summary)
            if summary["selected_episodes"]:
                episodes = list(summary["selected_episodes"].values())
                failures = [ep for ep in episodes if (read_optional(self.store, f"episodes/{ep}/result.json", {}).get("official_success") is False)]
                episode = (failures or episodes)[0]
                events = self.store.path(
                    f"episodes/{episode}/rollout/planner/events.jsonl"
                )
                if events.is_file():
                    from meta_harness.history import decisions, journal
                    rows = [row for _, row in journal(events)]
                    physical = [d for d in decisions(events.parent.parent) if d["physical_step_count"]]
                    calls = [
                        row
                        for row in rows
                        if row["event"]
                        in {"model_response", "tool_feedback", "tool_raw"}
                    ]
                    decision_id = (physical[-1 if failures else 0]["decision_id"] if physical
                                   else calls[0].get("decision_id") if calls else None)
                    selected_rows = [
                        row for row in rows if row.get("decision_id") == decision_id
                    ]
                    inputs = [
                        row for row in selected_rows if row["event"] == "sdk_request"
                    ]
                    example = {
                        "episode_id": episode,
                        "decision_id": decision_id,
                        "input": inputs[:1],
                        "call_and_feedback": [row for row in selected_rows if row["event"] in {"model_response", "tool_raw", "policy_feedback", "tool_feedback"}],
                        "http_requests": [row for row in selected_rows if row["event"] == "http_request"],
                        "selection_reason": "last physical decision of an unsuccessful episode" if failures and physical else "first physical decision" if physical else "first recorded decision; no physical decision available",
                        "history_decision": {"episode_id": episode, "decision_id": decision_id},
                        "physical_span": next((d for d in physical if d["decision_id"] == decision_id), None),
                    }
        example = self._relocate_history(example)
        material = self.store.path(self.prefix + "/opening-material.json")
        from meta_harness.meta_mcp import definitions
        benchmark = read_optional(self.store, "benchmark.json", {})
        tasks = [{key: task.get(key) for key in ("id", "backend_suite", "task_index", "language", "scenes", "episode_horizon")}
                 for task in benchmark.get("tasks", []) if task.get("id") in self.context["task_ids"]]
        replace_record(self.store, self.prefix + "/opening-material.json", {
            "inspection_source": inspection_source,
            "preview_error": preview_error,
            "inspection": inspection,
            "recorded_example": example,
            "meta_tools": definitions(live_enabled=self.config.development.live_enabled),
            "tasks": tasks,
            "optimization": optimization,
        })
        menu = [
            {"name": tool["name"], "description": tool.get("description", "")[:240]}
            for tool in inspection["tools"]
        ]
        feedback = self.store.path(f"workspaces/{self.context['scope_id']}/project/host-feedback.md")
        return (
            (self.context.get("objective") or "")
            + ("\nHost diagnostic feedback:\n" + feedback.read_text() if feedback.exists() else "")
            + "\n"
            + f"Improve the inheritable Harness for {self.context['task_ids']}; "
            f"round {self.context['round_index']}, kind {self.context['kind']}.\n"
            f"Selected: {self.context['selected_commit']}; starting base: {self.context['initial_base_commit']}.\n"
            f"Candidate worktree: {self.workspace}. Frozen context: {self.store.path(self.prefix + '/context.json')}.\n"
            + self.context["runtime"]
            + "\nCurrent selection objective and timing rules: " + json.dumps(optimization, ensure_ascii=False)
            + "\nDraft preview error (repair the existing worktree when non-null): "
            + json.dumps(preview_error, ensure_ascii=False)
            + "\nInspection source: " + inspection_source
            + "\nCurrent bound menu (brief; exact schemas in opening-material.json): "
            + json.dumps(menu, ensure_ascii=False)
            + f"\nFull bound menu, backend capability catalog, and recorded real request/call/feedback: {material}. "
            "Read the relevant entries before editing or calling tools; the full evidence is not truncated. "
            + "\nMeta tools are exposed by the rsi MCP server. If your client does not expose it, "
            "use this same validated host API from Python, not a new runner:\n"
            "from meta_harness.meta import MetaAttempt\n"
            "from meta_harness.meta_mcp import invoke\n"
            f"attempt = MetaAttempt({str(self.store.root)!r}, {self.attempt_id!r})\n"
            "result = invoke(attempt, TOOL_NAME, ARGUMENT_DICT)\n"
            "Image tools return (metadata, content_blocks); use the MCP image output or open the referenced PNG "
            "with your image-view tool. Printing image paths/array shapes is not visual inspection. "
            + "\nRecorded example identity: "
            + json.dumps({k: example[k] for k in ("episode_id", "decision_id")} if example else None)
            + "\nBudget: "
            + json.dumps(self.budget())
            + "\nResolve all Git merge conflicts before submitting when kind=merge. "
            "Use the RSI MCP tools; do not modify host records or config. "
            "Write a short notes.md in the scope project for the next session."
        )
