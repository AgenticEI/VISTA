"""A task's development budget survives Meta worker recovery."""

import pytest

from meta_harness.meta import MetaAttempt
from meta_harness.storage import RunStore


def _attempt(tmp_path, prior_cells, current_cells):
    store = RunStore(tmp_path)
    current = "demo_randomized-task-example-development-r02-d1"
    for name, sizes in (
        ("demo_randomized-task-example-development-r02-d0", prior_cells),
        (current, current_cells),
    ):
        for index, size in enumerate(sizes):
            store.write_once(
                f"meta/attempts/{name}/evaluations/c{index:02d}/request.json",
                {"cells": [
                    {"cell_id": f"s{n}", "task_id": "example", "seed": 1000 + n,
                     "repeat": 0}
                    for n in range(size)
                ]},
            )
    store.write_once(f"meta/attempts/{current}/budget.json", {
        "started_at": 0, "deadline": None, "stop_developing_at": None,
        "max_evaluation_requests": 12,
    })
    store.write_once("development-evaluation-budgets/from-libero-t.json", {
        "max_requests_per_round": 6, "max_cells_per_round": 24,
    })
    store.write_once("development-evaluation-budget-current.json", {
        "revision": "from-libero-t",
    })
    attempt = object.__new__(MetaAttempt)
    attempt.store = store
    attempt.prefix = f"meta/attempts/{current}"
    attempt.context = {
        "kind": "development",
        "scope_id": "demo_randomized/task-example",
        "round_index": 2,
        "selected_commit": "a" * 40,
        "task_ids": ["example"],
    }
    return attempt


def test_request_cap_counts_prior_recovery(tmp_path):
    attempt = _attempt(tmp_path, [4] * 5, [4])
    budget = attempt.budget()
    assert budget["evaluation_requests_used"] == 6
    assert budget["evaluation_requests_remaining"] == 0
    assert budget["evaluation_cells_used"] == 24
    with pytest.raises(ValueError, match="request quota"):
        attempt.evaluate("new", [
            {"cell_id": "x", "task_id": "example", "seed": 1000, "repeat": 0}
        ])


def test_cell_cap_counts_prior_recovery(tmp_path):
    attempt = _attempt(tmp_path, [4] * 4, [7])
    budget = attempt.budget()
    assert budget["evaluation_requests_used"] == 5
    assert budget["evaluation_requests_remaining"] == 1
    assert budget["evaluation_cells_remaining"] == 1
    with pytest.raises(ValueError, match="scene quota"):
        attempt.evaluate("new", [
            {"cell_id": "x", "task_id": "example", "seed": 1000, "repeat": 0},
            {"cell_id": "y", "task_id": "example", "seed": 1001, "repeat": 0},
        ])
