"""LIBERO release integration: real Harness Git history, controlled evaluations."""

import json
from pathlib import Path

from meta_harness import cli
from meta_harness.config import EvolutionConfig
from meta_harness.loop import EvolutionLoop
from meta_harness.meta import package_files
from robots.libero.benchmark import build_manifest
from tests.test_libero_benchmark_manifest import fixture_backend


ROOT = Path(__file__).resolve().parents[1]


class ControlledPool:
    def __init__(self, run):
        self.config = EvolutionConfig.parse(json.loads((run / "config.json").read_text()))
        self.requests = {}

    def submit(self, **request):
        previous = self.requests.setdefault(request["evaluation_id"], request)
        assert previous == request
        return request["evaluation_id"]

    def evaluation_status(self, evaluation_id):
        request = self.requests[evaluation_id]
        successes = 0 if request["purpose"] == "baseline" else {
            1: 1, 2: 0,
        }[int(evaluation_id.rsplit("r", 1)[1])]
        return {
            "comparable": True,
            "successes": successes,
            "denominator": len(request["cells"]),
            "exhausted": False,
            "selected_episodes": {},
        }

    def tick(self):
        return {}

    def cancel_scope(self, scope_id, *, evaluations=True):
        pass  # Controlled evaluations have no asynchronous workers.


def test_libero_run_initializes_and_retains_improvement_after_regression(tmp_path, monkeypatch):
    # Exercise the shipped setup example while replacing external simulator assets.
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    core, backend = fixture_backend(backend_root)
    monkeypatch.setattr(cli, "benchmark_manifest", lambda config: build_manifest(
        config.public_record(), core=core, backend=backend,
    ))
    raw = json.loads((ROOT / "configs/examples/libero.json").read_text())
    raw["initial_harness"] = str(ROOT / "harnesses/libero_pro_h0_min")
    raw["capability_catalog"] = str(ROOT / "robots/libero/harness_capabilities.json")
    raw["parallel"]["rounds"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    run = tmp_path / "run"
    initial = cli.initialize(config_path, run)
    assert initial["robot"] == "libero"

    pool = ControlledPool(run)
    loop = EvolutionLoop(run, pool=pool)
    candidates = []

    def develop(scope, **kwargs):
        workspace = loop.harnesses.workspace(scope["scope_id"], scope["selected_commit"])
        (workspace / "prompts/user.md").write_text(f"Candidate {len(candidates) + 1}")
        commit = loop.harnesses.freeze(
            workspace, files=package_files(loop.harnesses, workspace), message="candidate",
        )
        candidates.append(commit)
        return {"state": "submitted", "submission": {"commit": commit}}

    loop._develop = develop
    for _ in range(30):
        state = loop.tick()
        if state["status"] == "complete":
            break
    assert state["status"] == "complete"
    decisions = [loop.store.read(f"selections/goal/task-goal-t09/r{i:02d}.json")
                 for i in (1, 2)]
    assert [decision["accepted"] for decision in decisions] == [True, False]
    assert decisions[1]["incumbent_commit"] == candidates[0]
    assert loop.store.read("stages/goal/parallel-selected.json")["goal-t09"] == candidates[0]
    assert loop.harnesses.parents(candidates[1]) == (candidates[0],)
    assert len(pool.requests) == 3  # H0 and both frozen candidates.
    assert all(len(request["cells"]) == 1 for request in pool.requests.values())
    assert loop.store.read("benchmark.json")["task_count"] == 1
