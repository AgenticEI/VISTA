# Copyright 2026 The RPent Team. All rights reserved.
"""New run records: failure recovery and scoring without executing a robot."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meta_harness.evaluation import (
    collect_evaluation,
    pending_evaluation_cells,
    plan_evaluation,
    record_episode,
    record_episode_result,
)
from meta_harness.storage import RunStore


class EvaluationRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = RunStore(Path(self.temporary.name))
        self.store.write_once("config.json", {"episode": {"policy_seed": 123}})
        # Request identities are supplied by the Git store, never file hashes.
        self.commit = "a" * 40
        self.cells = [
            {"cell_id": "c01", "task_id": "open-cabinet", "seed": 1000, "repeat": 0},
            {"cell_id": "c02", "task_id": "open-cabinet", "seed": 1001, "repeat": 0},
        ]
        self.request()

    def request(self, evaluation_id="e001", **changes):
        arguments = dict(
            evaluation_id=evaluation_id, harness_commit=self.commit,
            scope_id="atomic-seen/open-cabinet", purpose="formal", cells=self.cells,
        )
        arguments.update(changes)
        return plan_evaluation(self.store, **arguments)

    def episode(self, episode_id, cell=0, attempt=1, evaluation_id="e001", **changes):
        request = self.store.read(f"evaluations/{evaluation_id}/request.json")
        condition = request["cells"][cell]
        record = {
            "schema_version": 1, "episode_id": episode_id,
            "evaluation_id": evaluation_id, "execution_attempt": attempt,
            **condition, "policy_seed": 123 + condition["seed"] * 10000 + condition["repeat"],
            "harness_commit": request["harness_commit"], "config_ref": "config.json",
            "purpose": request["purpose"], "controller": request["controller"],
            "started_at": "2026-09-21T00:00:00Z",
            "resources": {"env_gpu": 0, "vla_endpoint": "http://localhost:8525",
                          "sam_endpoint": "http://localhost:8516", "session_id": episode_id},
        }
        record.update(changes)
        record_episode(self.store, record)
        return record

    def result(self, episode_id, **changes):
        episode = self.store.read(f"episodes/{episode_id}/episode.json")
        result = {field: episode[field] for field in (
            "schema_version", "episode_id", "evaluation_id", "cell_id",
            "execution_attempt", "controller",
        )}
        result.update(validity="valid", official_success=False, agent_claimed_success=True,
                      terminal_reason="agent_stopped", env_steps=500, trace_complete=True,
                      elapsed_seconds=12.5)
        result.update(changes)
        record_episode_result(self.store, episode_id, result)
        return result

    def test_recovery_uses_first_valid_attempt_not_best_outcome(self):
        self.episode("ep001")
        self.result("ep001", validity="invalid", official_success=None, trace_complete=False)
        self.episode("ep002", attempt=2)
        self.result("ep002")  # Valid physical failure must not be replaced by a success.
        self.episode("ep003", attempt=3)
        self.result("ep003", official_success=True)
        self.episode("ep004", cell=1)
        self.result("ep004", official_success=True)
        summary = collect_evaluation(self.store, "e001")
        self.assertEqual(summary["selected_episodes"], {"c01": "ep002", "c02": "ep004"})
        self.assertEqual((summary["successes"], summary["denominator"]), (1, 2))
        self.assertTrue(summary["comparable"])
        self.assertEqual(summary["cost"]["episode_attempts"], 4)
        self.assertEqual(summary["cost"]["retry_attempts"], 2)
        self.assertEqual(summary["cost"]["episode_elapsed_seconds_sum"], 50.0)
        self.assertEqual(pending_evaluation_cells(summary, max_retries=3, active_episode_ids=set()),
                         {"ready": [], "active": [], "exhausted": [], "backoff": []})

    def test_partial_result_never_claims_complete_and_live_lease_prevents_duplicate(self):
        self.episode("done")
        self.result("done", official_success=True)
        self.episode("running", cell=1)
        summary = collect_evaluation(self.store, "e001")
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["denominator"], 1)
        self.assertEqual(summary["state"], "incomplete")
        self.assertFalse(summary["comparable"])
        pending = pending_evaluation_cells(summary, max_retries=3, active_episode_ids={"running"})
        self.assertEqual([cell["cell_id"] for cell in pending["active"]], ["c02"])
        self.assertEqual(pending["ready"], [])
        pending = pending_evaluation_cells(summary, max_retries=3, active_episode_ids=set())
        self.assertEqual(pending["ready"][0]["execution_attempt"], 2)

    def test_allocation_survives_interruption_before_episode_json(self):
        original = self.store.write_once

        def interrupted(path, record):
            if path == "episodes/interrupted/episode.json":
                raise OSError("simulated interruption")
            return original(path, record)

        with patch.object(self.store, "write_once", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                self.episode("interrupted")
        with self.assertRaises(ValueError):
            self.episode("interrupted", cell=1)
        self.request("other-evaluation")
        with self.assertRaises(ValueError):
            self.episode("interrupted", evaluation_id="other-evaluation")
        summary = collect_evaluation(self.store, "e001")
        self.assertEqual(summary["completed"], 0)
        pending = pending_evaluation_cells(summary, max_retries=1, active_episode_ids=set())
        self.assertEqual(pending["ready"][0]["execution_attempt"], 2)
        self.episode("retry", attempt=2)
        self.result("retry", validity="invalid", official_success=None)
        pending = pending_evaluation_cells(collect_evaluation(self.store, "e001"),
                                           max_retries=1, active_episode_ids=set())
        self.assertEqual(pending["backoff"], [])
        self.assertEqual([cell["cell_id"] for cell in pending["exhausted"]], ["c01"])
        import time as _time
        from meta_harness.evaluation import INFRA_RETRY_BACKOFF_S
        later = _time.time() + INFRA_RETRY_BACKOFF_S + 1
        pending = pending_evaluation_cells(collect_evaluation(self.store, "e001"),
                                           max_retries=1, active_episode_ids=set(), now=later)
        self.assertNotIn("c01", [cell["cell_id"] for cell in pending["ready"]])
        self.assertEqual([cell["cell_id"] for cell in pending["exhausted"]], ["c01"])

    def test_infra_failures_exhaust_without_becoming_task_failures(self):
        for attempt in range(1, 5):
            self.episode(f"burned-{attempt}", attempt=attempt)
            self.result(f"burned-{attempt}", validity="invalid", official_success=None,
                        terminal_reason="runner_error", trace_complete=False)
        pending = pending_evaluation_cells(collect_evaluation(self.store, "e001"),
                                           max_retries=3, active_episode_ids=set())
        self.assertEqual([c["cell_id"] for c in pending["exhausted"]], ["c01"])
        self.assertEqual(pending["backoff"], [])
        import time as _time
        from meta_harness.evaluation import INFRA_RETRY_BACKOFF_S
        later = _time.time() + INFRA_RETRY_BACKOFF_S + 1
        pending = pending_evaluation_cells(collect_evaluation(self.store, "e001"),
                                           max_retries=3, active_episode_ids=set(), now=later)
        ready = {cell["cell_id"]: cell["execution_attempt"] for cell in pending["ready"]}
        self.assertNotIn("c01", ready)
        self.assertEqual([c["cell_id"] for c in pending["exhausted"]], ["c01"])

    def test_infra_retry_backoff_holds_recent_failures(self):
        # A cell whose latest attempt just failed on infrastructure backs off
        # instead of burning the next retry inside the same outage window.
        from datetime import datetime, timedelta, timezone
        from meta_harness.evaluation import INFRA_RETRY_BACKOFF_S, pending_evaluation_cells
        now = datetime.now(timezone.utc)
        old = (now - timedelta(seconds=INFRA_RETRY_BACKOFF_S + 60)).isoformat()
        recent = now.isoformat()

        def cell(cell_id, completed_at):
            return {"cell_id": cell_id, "task_id": "t", "seed": 1000, "repeat": 0,
                    "attempts": [{"episode_id": f"{cell_id}-a1", "execution_attempt": 1,
                                  "current_profile": True, "current_execution": True,
                                  "complete": True, "validity": "invalid",
                                  "completed_at": completed_at}]}

        summary = {"pending_cells": ["c_old", "c_recent"],
                   "cells": {"c_old": cell("c_old", old), "c_recent": cell("c_recent", recent)}}
        pending = pending_evaluation_cells(summary, max_retries=3, active_episode_ids=set())
        self.assertEqual([c["cell_id"] for c in pending["ready"]], ["c_old"])
        self.assertEqual([c["cell_id"] for c in pending["backoff"]], ["c_recent"])
        self.assertEqual(pending["exhausted"], [])

    def test_late_earlier_attempt_cannot_replace_the_first_valid_completion(self):
        self.episode("old-worker", attempt=1)
        self.episode("retry-worker", attempt=2)
        self.result("retry-worker", official_success=True)
        first = collect_evaluation(self.store, "e001")
        self.assertEqual(first["selected_episodes"], {"c01": "retry-worker"})
        self.result("old-worker", official_success=False)
        # A new store has no in-memory selection cache after restart.
        restarted = RunStore(self.store.root)
        recovered = collect_evaluation(restarted, "e001")
        self.assertEqual(recovered["selected_episodes"], first["selected_episodes"])
        self.assertEqual(recovered["success_rate"], 1.0)
        self.assertEqual(recovered["cost"]["finished_attempts"], 2)

    def test_collection_refreshes_a_result_acknowledged_after_its_snapshot(self):
        self.episode("just-finished")
        self.result("just-finished", official_success=True)
        collect_evaluation(self.store, "e001")
        original = self.store.read
        stale_read = True

        def read_with_stale_snapshot(path):
            nonlocal stale_read
            if path == "episodes/just-finished/result.json" and stale_read:
                stale_read = False
                raise FileNotFoundError("result was not published at snapshot time")
            return original(path)

        with patch.object(self.store, "read", side_effect=read_with_stale_snapshot):
            summary = collect_evaluation(self.store, "e001")
        self.assertEqual(summary["selected_episodes"], {"c01": "just-finished"})
        self.assertEqual(summary["cost"]["finished_attempts"], 1)

    def test_requests_and_results_are_immutable_and_identity_is_checked(self):
        self.request()  # An identical request is idempotent.
        changed = copy.deepcopy(self.cells)
        changed[0]["seed"] += 2
        with self.assertRaises(ValueError):
            self.request(cells=changed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.request("duplicate", cells=[self.cells[0], {**self.cells[0], "cell_id": "different"}])
        with self.assertRaisesRegex(ValueError, "harness_commit"):
            self.episode("wrong-version", harness_commit="b" * 40)
        with self.assertRaisesRegex(ValueError, "policy_seed"):
            self.episode("wrong-rng", policy_seed=999)
        self.episode("ep001")
        self.result("ep001")
        with self.assertRaises(ValueError):
            self.result("ep001", official_success=True)
        with self.assertRaises(ValueError):
            self.episode("ep001", cell=1)
        with self.assertRaises(ValueError):
            self.episode("duplicate-attempt")

    def test_manual_and_mixed_successes_do_not_enter_autonomous_score(self):
        self.episode("mixed")
        self.result("mixed", controller="mixed", official_success=True)
        summary = collect_evaluation(self.store, "e001")
        self.assertEqual(summary["denominator"], 0)
        self.assertEqual(summary["non_autonomous"], 1)
        self.assertIsNone(summary["success_rate"])
        self.request("manual", controller="meta", purpose="diagnostic", cells=self.cells[:1])
        self.episode("manual-success", evaluation_id="manual")
        self.result("manual-success", official_success=True)
        manual = collect_evaluation(self.store, "manual")
        self.assertEqual(manual["state"], "complete")
        self.assertFalse(manual["comparable"])
        self.assertEqual(manual["successes"], 0)
        self.assertIsNone(manual["success_rate"])
        with self.assertRaisesRegex(ValueError, "require autonomous"):
            self.request("bad-formal", controller="meta")

    def test_invalid_results_cannot_claim_success_or_complete_trace(self):
        self.episode("ep001")
        with self.assertRaisesRegex(ValueError, "invalid official_success"):
            self.result("ep001", validity="invalid", official_success=True)
        with self.assertRaisesRegex(ValueError, "complete trace"):
            self.result("ep001", trace_complete=False)
        with self.assertRaisesRegex(ValueError, "elapsed_seconds"):
            self.result("ep001", elapsed_seconds=float("nan"))
        with self.assertRaisesRegex(ValueError, "evaluation_id"):
            self.result("ep001", evaluation_id="different")

    def test_same_cell_id_in_another_evaluation_is_a_different_cell(self):
        self.episode("ep001")
        self.result("ep001", official_success=True)
        self.request("e002")
        self.assertEqual(collect_evaluation(self.store, "e002")["completed"], 0)
        # Empty directories and partial temporary result files are not facts.
        target = self.store.root / "episodes" / "ep002"
        target.mkdir()
        (target / ".result.json.tmp").write_text('{"official_success":')
        self.assertEqual(collect_evaluation(self.store, "e002")["completed"], 0)


if __name__ == "__main__":
    unittest.main()
