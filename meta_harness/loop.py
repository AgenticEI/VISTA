# Copyright 2026 The RPent Authors.
"""Single schema-2 evolution state machine; immutable facts drive recovery."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from meta_harness.compose import compose_selected
from meta_harness.meta import MetaAttempt
from meta_harness.pool import ExecutionPool, read_optional
from meta_harness.storage import HarnessStore, RunStore
from rpent.utils.execution import file_lock, is_locked, owns_process, signal_owned


class EvolutionLoop:
    def __init__(self, root, *, pool=None, meta_factory=MetaAttempt.create):
        self.store = RunStore(root)
        self.pool = pool or ExecutionPool(root)
        self.config = self.pool.config
        self.harnesses = HarnessStore(root)
        self.meta_factory = meta_factory
        self._children = {}

    def _initial_state(self):
        return {
            "schema_version": 2,
            "status": "running",
            "suite_index": 0,
            "phase": "parallel",
            "base_commit": self.store.read("initial.json")["commit"],
            "scopes": {},
            "stages": [],
        }

    def status(self):
        return read_optional(self.store, "state.json", self._initial_state())

    def _scope(self, state, scope_id, task_ids, phase, base):
        return state["scopes"].setdefault(
            scope_id,
            {
                "scope_id": scope_id,
                "task_ids": task_ids,
                "phase": phase,
                "base_commit": base,
                "selected_commit": base,
                "selected_evaluation": None,
                "rounds_completed": 0,
                "step": "baseline",
                "recoveries": 0,
            },
        )

    def _matrix(self, scope, suite):
        return [
            asdict(c)
            for c in self.config.matrix(
                suite.id,
                scope["phase"],
                task_id=scope["task_ids"][0] if scope["phase"] == "parallel" else None,
            )
        ]

    @staticmethod
    def _scope_key(scope):
        return scope["scope_id"].replace("/", "-") + ("-" + scope["evaluation_epoch"] if scope.get("evaluation_epoch") else "")

    def _evaluation(self, scope, suite, purpose, commit, suffix):
        eid = self._scope_key(scope) + "-" + suffix
        self.pool.submit(
            evaluation_id=eid,
            harness_commit=commit,
            scope_id=scope["scope_id"],
            purpose=purpose,
            cells=self._matrix(scope, suite),
        )
        return eid, self.pool.evaluation_status(eid)

    def _meta_active(self):
        for aid, child in list(self._children.items()):
            if child.poll() is not None:
                child.wait()
                del self._children[aid]
        active = []
        for path in self.store.path("meta/attempts").glob("*/budget.json"):
            leased = is_locked(path.parent / "lease.lock")
            provider = read_optional(
                self.store, f"meta/attempts/{path.parent.name}/provider-owner.json"
            )
            orphan_provider = not leased and provider and owns_process(provider)
            if orphan_provider:
                signal_owned(provider)
            if leased or orphan_provider:
                active.append(path.parent.name)
                grant = self.store.read(path.relative_to(self.store.root).as_posix())
                if (grant["deadline"] is not None and time.time() >= grant["deadline"]):
                    for owner_name in ("provider-owner", "owner"):
                        owner = read_optional(
                            self.store,
                            f"meta/attempts/{path.parent.name}/{owner_name}.json",
                        )
                        if owner:
                            signal_owned(
                                owner, 15 if time.time() < grant["deadline"] + 5 else 9
                            )
        return active

    def _develop(self, scope, *, kind="development", donors=None):
        attempt_index = (
            scope.get("merge_attempt_index", scope["recoveries"])
            if kind == "merge"
            else scope["recoveries"]
        )
        aid = (
            self._scope_key(scope)
            + f"-{kind}-r{scope['rounds_completed'] + 1:02d}-d{attempt_index}"
        )
        prefix = f"meta/attempts/{aid}"
        finished = read_optional(self.store, f"{prefix}/finished.json")
        active = self._meta_active()
        if aid in active:
            return None
        if finished:
            return finished
        from meta_harness.admission_guard import meta_allowed
        if not meta_allowed(self.store, scope["scope_id"], len(active)):
            return None
        if len(active) >= self.config.pool.profile["max_meta"]:
            return None
        # Existing drafts are preserved on reopening. Each new round starts
        # from the selected incumbent unless Meta explicitly selects a parent.
        previous = read_optional(self.store, f"{prefix}/context.json")
        base = previous["initial_base_commit"] if previous else scope["selected_commit"]
        if previous is None and kind != "merge" and scope["recoveries"] == 0:
            workspace = self.harnesses.workspace(scope["scope_id"], base)
            from meta_harness.meta import package_files

            self.harnesses.select_parent(
                workspace, base, files=package_files(self.harnesses, workspace)
            )
        attempt = self.meta_factory(
            self.store.root,
            attempt_id=aid,
            scope_id=scope["scope_id"],
            selected_commit=scope["selected_commit"],
            selected_evaluation=scope["selected_evaluation"],
            initial_base_commit=base,
            task_ids=scope["task_ids"],
            round_index=scope["rounds_completed"] + 1,
            kind=kind,
            donors=donors,
        )
        if attempt.budget()["development_seconds_left"] is not None and attempt.budget()["development_seconds_left"] <= 0:
            return attempt.finish(reason="deadline")
        with file_lock(self.store.path(f"{prefix}/lease.lock"), blocking=False) as fd:
            with self.store.path(f"{prefix}/worker.log").open("a") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "meta_harness.meta_worker",
                        str(self.store.root),
                        aid,
                        str(fd),
                    ],
                    cwd=self.config.repo_root,
                    pass_fds=(fd,),
                    start_new_session=True,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self._children[aid] = process
        return None

    def _development_budget(self, scope):
        control = read_optional(self.store, "development-budget-current.json")
        if control is None or scope["phase"] != "parallel":
            return None
        from meta_harness.evaluation import _record_id
        revision = _record_id(control["revision"], "development budget revision")
        record = self.store.read(f"development-budgets/{revision}.json")
        limit = record["max_candidate_rounds"]
        if type(limit) is not int or limit < 1:
            raise ValueError("max_candidate_rounds must be a positive integer")
        records = [self.store.read(f"selections/{scope['scope_id']}/r{n:02d}.json")
                   for n in range(1, scope["rounds_completed"] + 1)]
        count = sum(not item["confirmation"] for item in records)
        return limit, count, revision

    def _stop_at_development_budget(self, scope):
        budget = self._development_budget(scope)
        if budget is None or budget[1] < budget[0] or scope["step"] != "develop":
            return False
        aid = (self._scope_key(scope) +
               f"-development-r{scope['rounds_completed'] + 1:02d}-d{scope['recoveries']}")
        prefix = f"meta/attempts/{aid}"
        provider = read_optional(self.store, f"{prefix}/provider-owner.json")
        # Drain an already running development attempt; never launch or formally
        # evaluate an over-budget candidate. Preserve all drafts and results.
        active = is_locked(self.store.path(f"{prefix}/lease.lock")) or (
            provider is not None and owns_process(provider))
        scope["development_budget_revision"] = budget[2]
        if active:
            scope["budget_stop_pending"] = True
            return True
        scope.pop("budget_stop_pending", None)
        scope.update(step="complete", convergence_reason="development_budget_exhausted")
        return True

    def _advance_scope(self, scope, suite):
        from meta_harness.admission_guard import scope_allowed
        if not scope_allowed(self.store, scope["scope_id"], "baseline" if scope["step"] == "baseline" else "development"):
            return
        if scope["step"] in {"complete", "incomplete"}:
            return
        if self._stop_at_development_budget(scope):
            return
        if scope["step"] == "baseline":
            eid, result = self._evaluation(
                scope, suite, "baseline", scope["selected_commit"], "baseline"
            )
            if result["comparable"]:
                perfect = result["denominator"] > 0 and result["successes"] == result["denominator"]
                scope.update(selected_evaluation=eid, step="develop", perfect_streak=int(perfect))
                if perfect and self.config.selection.perfect_rounds:
                    scope.update(candidate_commit=scope["selected_commit"], step="confirm")
            elif result.get("exhausted"):
                scope.update(
                    step="incomplete", error="baseline infrastructure retries exhausted"
                )
        elif scope["step"] == "develop":
            result = self._develop(scope)
            if result:
                if result["state"] == "submitted":
                    scope.update(
                        candidate_commit=result["submission"]["commit"], step="evaluate"
                    )
                elif (
                    scope["recoveries"]
                    < self.config.development.max_no_candidate_recoveries
                ):
                    scope["recoveries"] += 1
                else:
                    scope.update(
                        step="incomplete", error="no loadable candidate after recovery"
                    )
        elif scope["step"] in {"evaluate", "confirm"}:
            confirming = scope["step"] == "confirm"
            number = scope["rounds_completed"] + 1
            eid, candidate = self._evaluation(
                scope, suite, "formal", scope["candidate_commit"], f"r{number:02d}"
            )
            if candidate.get('rejected'):
                from meta_harness.policy_rejection import reject_candidate
                reject_candidate(self,scope,eid,candidate,candidate['rejection'])
                return
            if candidate.get("exhausted"):
                scope.update(
                    step="incomplete",
                    error="candidate infrastructure retries exhausted",
                )
                return
            if not candidate["comparable"]:
                return
            selected = self.pool.evaluation_status(scope["selected_evaluation"])
            if (
                not selected["comparable"]
                or candidate["denominator"] != selected["denominator"]
            ):
                raise ValueError("selection requires complete comparable matrices")
            from meta_harness.selection import compare
            comparison = compare(self.config.selection, selected, candidate)
            accepted = comparison["accepted"]
            perfect = candidate["denominator"] > 0 and candidate["successes"] == candidate["denominator"]
            same_package = scope["candidate_commit"] == scope["selected_commit"]
            streak = (scope.get("perfect_streak", 0) + 1 if same_package else 1) if perfect else 0
            # A rejected different package cannot certify the incumbent.
            if not same_package and not accepted:
                streak = 0
            converged = bool(self.config.selection.perfect_rounds and streak >= self.config.selection.perfect_rounds)
            # Publication precedes mutable state; replay after a crash verifies
            # the same selection and never increments twice.
            record = {
                "round": number,
                "confirmation": confirming,
                "perfect_streak": streak,
                "converged": converged,
                "incumbent_commit": scope["selected_commit"],
                "incumbent_evaluation": scope["selected_evaluation"],
                "candidate_commit": scope["candidate_commit"],
                "candidate_evaluation": eid,
                "accepted": accepted,
                "selected_commit": scope["candidate_commit"]
                if accepted
                else scope["selected_commit"],
                "selected_evaluation": eid
                if accepted or confirming
                else scope["selected_evaluation"],
                "incumbent_successes": selected["successes"],
                "candidate_successes": candidate["successes"],
                "denominator": candidate["denominator"],
                "comparison": comparison,
            }
            self.store.write_once(
                f"selections/{scope['scope_id']}/r{number:02d}.json", record
            )
            self.harnesses.retain(
                scope["candidate_commit"], f"candidates/{scope['candidate_commit']}"
            )
            scope.update(
                selected_commit=record["selected_commit"],
                selected_evaluation=record["selected_evaluation"],
                rounds_completed=number,
                recoveries=0,
                perfect_streak=streak,
                convergence_reason="two_perfect_evaluations" if converged else None,
                step="complete"
                if converged or number >= self.config.phase(scope["phase"]).rounds
                else "confirm" if streak and self.config.selection.perfect_rounds else "develop",
            )
            budget = self._development_budget(scope)
            if budget is not None:
                exhausted = budget[1] >= budget[0]
                scope["development_budget_revision"] = budget[2]
                # Confirmation is separate from the candidate-development cap.
                scope["step"] = ("complete" if converged else "confirm" if streak
                                 and self.config.selection.perfect_rounds else
                                 "complete" if exhausted else "develop")
                scope["convergence_reason"] = ("two_perfect_evaluations" if converged
                    else "development_budget_exhausted" if exhausted and scope["step"] == "complete"
                    else None)
            if scope["step"] == "confirm":
                scope["candidate_commit"] = scope["selected_commit"]

    def _advance_merge(self, state, suite):
        scope_id = f"{suite.id}/joint"
        control_id = f"{suite.id}/merge-control"
        merge_scope = state["scopes"].get(control_id)
        if merge_scope is not None and merge_scope.get("step") == "incomplete":
            return
        intent = state.get("merge_intent", {})
        if intent.get("kind") == "resolution":
            self._apply_merge_resolution(state, merge_scope, intent)
        elif merge_scope is not None and merge_scope.get("awaiting_submission", True):
            merge_scope.setdefault("merge_attempt_index", merge_scope["recoveries"])
            result = self._develop(merge_scope, kind="merge", donors=state["donors"])
            if result is None:
                return
            if result["state"] != "submitted":
                if (
                    merge_scope["recoveries"]
                    < self.config.development.max_no_candidate_recoveries
                ):
                    merge_scope["recoveries"] += 1
                    merge_scope["merge_attempt_index"] += 1
                else:
                    merge_scope.update(
                        step="incomplete", error="merge has no loadable resolution"
                    )
                return
            # Only the finished attempt's submission may seed the next merge,
            # never an intermediate evaluate/live freeze in the same worktree.
            commit = result["submission"]["commit"]
            intent = {
                "kind": "resolution",
                "scope_id": scope_id,
                "commit": commit,
                "next_attempt_index": merge_scope["merge_attempt_index"] + 1,
            }
            state["merge_intent"] = intent
            self._persist(state)
            self._apply_merge_resolution(state, merge_scope, intent)

        def before_merge(intent):
            state["merge_intent"] = intent
            self._persist(state)

        result = compose_selected(
            self.harnesses,
            scope_id,
            state["base_commit"],
            state["donors"],
            before_merge=before_merge,
        )
        state.pop("merge_intent", None)
        if result["state"] == "complete":
            self._scope(
                state, scope_id, [t.id for t in suite.tasks], "joint", result["commit"]
            )
            state.update(phase="joint")
            return
        merge_scope = self._scope(
            state,
            control_id,
            [t.id for t in suite.tasks],
            "joint",
            state["base_commit"],
        )
        merge_scope.update(scope_id=scope_id, awaiting_submission=True)
        # Persist this ownership before a later tick starts the repair attempt.

    def _apply_merge_resolution(self, state, merge_scope, intent):
        """Finish the recorded parent switch before starting another donor.

        A legacy checkpoint may still point at a finished resolution while Git
        already contains the next donor conflict. Preserve that conflict when
        the submitted resolution is an ancestor of its current HEAD.
        """
        from meta_harness.meta import package_files

        commit = intent["commit"]
        workspace = self.harnesses.workspace(intent["scope_id"], commit)
        with self.harnesses._locked():
            merge_path = self.harnesses._git(
                "rev-parse", "--git-path", "MERGE_HEAD", cwd=workspace
            )
            merge_path = Path(merge_path)
            if not merge_path.is_absolute():
                merge_path = workspace / merge_path

            def is_ancestor(ancestor, descendant):
                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(workspace),
                        "merge-base",
                        "--is-ancestor",
                        ancestor,
                        descendant,
                    ],
                    capture_output=True,
                )
                if result.returncode not in {0, 1}:
                    raise RuntimeError(
                        f"cannot inspect merge recovery ancestry: {result.stderr.decode().strip()}"
                    )
                return result.returncode == 0

            composition_advanced = False
            if merge_path.exists():
                pending_donors = merge_path.read_text().splitlines()
                composition_advanced = is_ancestor(commit, "HEAD") and all(
                    donor in state["donors"] for donor in pending_donors
                )
            elif is_ancestor(commit, "HEAD"):
                # Also recover an old checkpoint after a clean donor merge.
                # A later Meta trial freeze alone is not enough: Git must show
                # a selected donor newly integrated after the submission.
                composition_advanced = any(
                    is_ancestor(donor, "HEAD") and not is_ancestor(donor, commit)
                    for donor in state["donors"]
                )
            if not composition_advanced:
                self.harnesses.select_parent(
                    workspace, commit, files=package_files(self.harnesses, workspace)
                )
        # Publish completion before compose_selected can modify the worktree.
        # Replaying the intent assigns the same index instead of incrementing.
        merge_scope["merge_attempt_index"] = intent["next_attempt_index"]
        merge_scope["recoveries"] = 0
        merge_scope["awaiting_submission"] = False
        state.pop("merge_intent", None)
        self._persist(state)

    def _pool_pending(self, snapshot):
        """Do not return while a persisted request or leased worker needs ticks."""
        if snapshot.get("active"):
            return True
        for evaluation_id, summary in snapshot.get("evaluations", {}).items():
            if summary["state"] == "complete":
                continue
            queue = self.pool.evaluation_status(evaluation_id).get("queue", {})
            # An exhausted cell can coexist with ready cells in the same
            # matrix. Those remaining cells still need resource accounting.
            # Cells backing off after infrastructure failures still owe work.
            if queue.get("ready") or queue.get("active") or queue.get("backoff"):
                return True
        for path in self.store.path("pool/live").glob("*/request.json"):
            request = self.store.read(path.relative_to(self.store.root).as_posix())
            sid = request["session_id"]
            if (
                self.pool.request_admitted(request)
                and (request["deadline"] is None or time.time() < request["deadline"])
                and not self.store.path(f"pool/live/{sid}/close.json").exists()
                and not self.store.path(f"pool/jobs/live-{sid}/spec.json").exists()
            ):
                return True
        return False

    def _retire_completed_development(self, state):
        """Completed task searches must not keep charging for orphaned tests."""
        for scope_id, scope in state.get("scopes", {}).items():
            if (scope.get("step") == "complete" and scope.get("phase") == "parallel"
                    and not scope.get("development_cleanup")):
                self.pool.cancel_scope(scope_id, evaluations=True)
                scope["development_cleanup"] = {"reason": "task_search_complete", "at": time.time()}

    def tick(self):
        with self.store.scheduler_lock():
            state = self.status()
            if state["status"] not in {"running", "incomplete", "complete", "paused"}:
                return state
            # Old versions could persist incomplete while independent scopes
            # and pool jobs were unfinished. Resume their work without granting
            # extra recovery attempts to an already exhausted scope.
            state["status"] = "running"
            if self.config.scheduling == "pipeline":
                self._pipeline_root = state
                try:
                    return self._tick_pipeline(state)
                finally:
                    self._pipeline_root = None
            terminal = (
                "complete" if state["suite_index"] == len(self.config.suites) else None
            )
            suite = None if terminal else self.config.suites[state["suite_index"]]
            if terminal is not None:
                pass
            elif state["phase"] == "parallel":
                scopes = [
                    self._scope(
                        state,
                        f"{suite.id}/task-{task.id}",
                        [task.id],
                        "parallel",
                        state["base_commit"],
                    )
                    for task in suite.tasks
                ]
                for scope in scopes:
                    scope.setdefault("waiting_since", time.time())
                for scope in sorted(scopes, key=lambda s: s["waiting_since"]):
                    previous_step = scope["step"]
                    self._advance_scope(scope, suite)
                    if scope["step"] != previous_step:
                        scope["waiting_since"] = time.time()
                if all(s["step"] == "complete" for s in scopes):
                    task_selected = {
                        s["task_ids"][0]: s["selected_commit"] for s in scopes
                    }
                    self.store.write_once(
                        f"stages/{suite.id}/parallel-selected.json",
                        task_selected,
                    )
                    if self.config.joint.rounds == 0:
                        # Speed track: keep per-task selecteds; do not merge/joint.
                        selected = {
                            "suite_id": suite.id,
                            "selected_commit": state["base_commit"],
                            "parallel_only": True,
                            "task_selected": task_selected,
                        }
                        self.store.write_once(
                            f"stages/{suite.id}/selected.json", selected
                        )
                        state["stages"].append(selected)
                        state.update(
                            suite_index=state["suite_index"] + 1,
                            phase="parallel",
                        )
                        if state["suite_index"] == len(self.config.suites):
                            terminal = "complete"
                    else:
                        state.update(
                            phase="merge",
                            donors=[s["selected_commit"] for s in scopes],
                        )
                elif all(s["step"] in {"complete", "incomplete"} for s in scopes):
                    terminal = "incomplete"
            elif state["phase"] == "merge":
                self._advance_merge(state, suite)
                control = state["scopes"].get(f"{suite.id}/merge-control", {})
                if control.get("step") == "incomplete":
                    terminal = "incomplete"
            else:
                scope = state["scopes"][f"{suite.id}/joint"]
                self._advance_scope(scope, suite)
                if scope["step"] == "incomplete":
                    terminal = "incomplete"
                elif scope["step"] == "complete":
                    selected = {
                        "suite_id": suite.id,
                        "selected_commit": scope["selected_commit"],
                        "selected_evaluation": scope["selected_evaluation"],
                    }
                    self.store.write_once(f"stages/{suite.id}/selected.json", selected)
                    state["stages"].append(selected)
                    state.update(
                        suite_index=state["suite_index"] + 1,
                        base_commit=scope["selected_commit"],
                        phase="parallel",
                    )
                    if state["suite_index"] == len(self.config.suites):
                        terminal = "complete"
            # Admission, reaping and Meta cleanup must happen before publishing
            # a terminal scheduler state, including when reopening an old run.
            self._retire_completed_development(state)
            pool_snapshot = self.pool.tick()
            meta_active = self._meta_active()
            if (
                terminal is not None
                and not meta_active
                and not self._pool_pending(pool_snapshot)
            ):
                state["status"] = terminal
            self._persist(state)
        return state

    def _persist(self, state):
        self.store.write_state(getattr(self, "_pipeline_root", None) or state)

    def _tick_pipeline(self, state):
        """Evolve one explicitly focused suite; all suites have independent H0s."""
        control = read_optional(self.store, "suite-control.json", {})
        if control.get("mode", "independent") != "independent":
            raise ValueError("pipeline suite control must use independent evolution")
        focused = self.config.suite(control.get("active_suite", self.config.suites[0].id))
        focused_suites = (focused,)
        state.update(active_suite=focused.id,
                     suite_index=list(self.config.suites).index(focused),
                     suite_evolution="independent")
        pipelines = state.setdefault("pipelines", {})
        work = []
        for suite in focused_suites:
            pipeline = pipelines.setdefault(suite.id, {
                "phase": "parallel", "base_commit": state["base_commit"], "scopes": {},
            })
            # Keep the public scope map canonical, including merge recovery state.
            for key, scope in pipeline["scopes"].items():
                state["scopes"].setdefault(key, scope)
            pipeline["scopes"] = {key: scope for key, scope in state["scopes"].items()
                                  if key.startswith(suite.id + "/")}
            if pipeline["phase"] == "parallel":
                scopes = [self._scope(pipeline, f"{suite.id}/task-{task.id}", [task.id],
                                      "parallel", pipeline["base_commit"]) for task in suite.tasks]
                work.extend((suite, scope) for scope in scopes)
            elif pipeline["phase"] == "joint":
                work.append((suite, pipeline["scopes"][f"{suite.id}/joint"]))
            state["scopes"].update(pipeline["scopes"])
        for _, scope in work:
            scope.setdefault("waiting_since", time.time())
        for suite, scope in sorted(work, key=lambda item: item[1]["waiting_since"]):
            previous = scope["step"]
            self._advance_scope(scope, suite)
            if previous != scope["step"]:
                scope["waiting_since"] = time.time()
        for suite in focused_suites:
            pipeline = pipelines[suite.id]
            if pipeline["phase"] == "parallel":
                scopes = list(pipeline["scopes"].values())
                if all(s["step"] == "complete" for s in scopes):
                    selected = {s["task_ids"][0]: s["selected_commit"] for s in scopes}
                    self.store.write_once(f"stages/{suite.id}/parallel-selected.json", selected)
                    pipeline["task_selected"] = selected
                    pipeline["donors"] = list(selected.values())
                    pipeline["phase"] = (
                        "parallel_complete" if control.get("pause_after_parallel", False)
                        else "merge" if self.config.joint.rounds else "complete"
                    )
                elif all(s["step"] in {"complete", "incomplete"} for s in scopes):
                    pipeline["phase"] = "incomplete"
            if pipeline["phase"] == "parallel_complete" and not control.get("pause_after_parallel", False):
                pipeline["phase"] = "merge" if self.config.joint.rounds else "complete"
            if pipeline["phase"] == "merge":
                self._advance_merge(pipeline, suite)
                if pipeline["scopes"].get(f"{suite.id}/merge-control", {}).get("step") == "incomplete":
                    pipeline["phase"] = "incomplete"
            if pipeline["phase"] == "joint":
                scope = pipeline["scopes"][f"{suite.id}/joint"]
                if scope["step"] in {"complete", "incomplete"}:
                    pipeline["phase"] = scope["step"]
            state["scopes"].update(pipeline["scopes"])
            if pipeline["phase"] == "complete":
                joint = pipeline["scopes"].get(f"{suite.id}/joint")
                selected = ({"suite_id": suite.id, "selected_commit": joint["selected_commit"],
                             "selected_evaluation": joint["selected_evaluation"]} if joint else
                            {"suite_id": suite.id, "parallel_only": True,
                             "selected_commit": pipeline["base_commit"], "task_selected": pipeline["task_selected"]})
                self.store.write_once(f"stages/{suite.id}/selected.json", selected)
                if selected not in state["stages"]:
                    state["stages"].append(selected)
        self._retire_completed_development(state)
        snapshot = self.pool.tick()
        active = self._meta_active()
        state["phase"] = pipelines[focused.id]["phase"]
        if state["phase"] == "parallel_complete" and not active and not self._pool_pending(snapshot):
            state["status"] = "paused"
            state["pause_reason"] = "parallel_complete_waiting_for_operator"
        else:
            state.pop("pause_reason", None)
        if state["phase"] in {"complete", "incomplete"} and not active and not self._pool_pending(snapshot):
            state["status"] = state["phase"]
            state["completion_scope"] = focused.id
        self._persist(state)
        return state

    def run(self):
        while True:
            state = self.tick()
            if state["status"] != "running":
                return state
            time.sleep(0.2)
