"""A completed infrastructure failure is still work, regardless of count."""

from meta_harness.evaluation import pending_evaluation_cells


def test_many_invalid_attempts_do_not_exhaust_the_original_cell():
    attempts = [
        {"execution_attempt": index, "episode_id": f"attempt-{index}",
         "complete": True, "validity": "invalid", "completed_at": "2026-01-01T00:00:00+00:00"}
        for index in range(1, 101)
    ]
    summary = {
        "pending_cells": ["c00001"],
        "cells": {"c00001": {
            "cell_id": "c00001", "task_id": "task", "seed": 1000,
            "repeat": 0, "attempts": attempts,
        }},
    }
    pending = pending_evaluation_cells(summary, max_retries=None, active_episode_ids=set())
    assert pending["exhausted"] == []
    assert len(pending["ready"]) == 1
    assert pending["ready"][0]["execution_attempt"] == 101

    bounded = pending_evaluation_cells(summary, max_retries=3, active_episode_ids=set())
    assert len(bounded["exhausted"]) == 1
