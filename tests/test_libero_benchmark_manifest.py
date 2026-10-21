from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robots.libero.benchmark import BACKEND_SUITES, SCENE_SEEDS, build_manifest


def fixture_backend(tmp_path, *, empty=False, language="Actual BDDL instruction"):
    for name in ("bddl_files", "init_states", "assets"):
        (tmp_path / name).mkdir()
    (tmp_path / "bddl_files" / "task.bddl").write_text("(define (:language Actual BDDL instruction))")
    torch.save(np.empty((0,)) if empty else np.zeros((50, 8)), tmp_path / "init_states" / "task.init")
    task = SimpleNamespace(name="filename_is_not_language", problem_folder="", bddl_file="task.bddl", init_states_file="task.init", language=language)
    suite = SimpleNamespace(n_tasks=10, get_task=lambda index: task)
    core = SimpleNamespace(get_libero_path=lambda name: tmp_path if name == "benchmark_root" else tmp_path / name)
    backend = SimpleNamespace(__name__="liberopro.liberopro.benchmark", __file__=str(tmp_path / "benchmark.py"), get_benchmark=lambda name: lambda: suite)
    return core, backend


def configuration(mode="probe"):
    return {
        "robot": "libero", "benchmark": "libero-pro", "mode": mode,
        "seeds": [1000, 1001] if mode == "probe" else list(SCENE_SEEDS),
        "suites": [{"id": suite, "backend_suite": suite,
                    "tasks": [{"id": f"{suite}-t{index}", "task_index": index}
                              for index in (range(10) if mode == "formal" else [0])]} 
                   for suite in (BACKEND_SUITES if mode == "formal" else BACKEND_SUITES[:1])],
        "episode": {"command": ["--max-episode-steps", "10000" if mode == "formal" else "20"]},
    }


def test_formal_manifest_has_forty_tasks_and_original_seed_mapping(tmp_path):
    core, backend = fixture_backend(tmp_path)
    manifest = build_manifest(configuration("formal"), core=core, backend=backend)
    assert manifest["task_count"] == 40
    assert manifest["tasks"][0]["language"] == "Actual BDDL instruction"
    assert manifest["tasks"][0]["scenes"] == [{"seed": seed, "init_state_index": index} for index, seed in enumerate(SCENE_SEEDS)]
    assert manifest["episode_horizon"] == 10000


def test_probe_can_use_subset_and_short_physical_budget(tmp_path):
    core, backend = fixture_backend(tmp_path)
    manifest = build_manifest(configuration(), core=core, backend=backend)
    assert manifest["task_count"] == 1
    assert manifest["episode_horizon"] == 20
    assert manifest["scene_seeds"] == [1000, 1001]


def test_formal_swap_manifest_uses_all_four_position_perturbation_suites(tmp_path):
    core, backend = fixture_backend(tmp_path)
    config = configuration("formal")
    for suite in config["suites"]:
        suite["backend_suite"] = suite["backend_suite"].replace("_task", "_swap")
    manifest = build_manifest(config, core=core, backend=backend)
    assert manifest["task_count"] == 40
    assert all(task["backend_suite"].endswith("_swap") for task in manifest["tasks"])


def test_empty_init_is_a_hard_error(tmp_path):
    core, backend = fixture_backend(tmp_path, empty=True)
    with pytest.raises(ValueError, match="No valid initial states"):
        build_manifest(configuration(), core=core, backend=backend)


def test_registry_filename_language_cannot_replace_bddl(tmp_path):
    core, backend = fixture_backend(tmp_path, language="wrong instruction from filename")
    with pytest.raises(ValueError, match="differs from BDDL"):
        build_manifest(configuration(), core=core, backend=backend)


def test_formal_cannot_silently_become_probe(tmp_path):
    core, backend = fixture_backend(tmp_path)
    config = configuration()
    config["mode"] = "formal"
    with pytest.raises(ValueError, match="scene seeds"):
        build_manifest(config, core=core, backend=backend)
