"""Resolve the fixed LIBERO-PRO experiment before starting any simulation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

FAMILIES = ("spatial", "object", "goal", "10")
BACKEND_SUITES = tuple(f"libero_{family}_task" for family in FAMILIES)
SWAP_SUITES = tuple(f"libero_{family}_swap" for family in FAMILIES)
SCENE_SEEDS = tuple(range(1000, 1016))
EPISODE_HORIZON = 10000
_LANGUAGE = re.compile(r"\(:language\s+([^)]+)\)")


def load_pro_backend():
    """Import PRO directly and reject RLinf's fallback to another benchmark."""
    configured = os.environ.get("LIBERO_TYPE", "pro").lower()
    if configured != "pro":
        raise ValueError(f"LIBERO-PRO requires LIBERO_TYPE=pro, got {configured!r}")
    os.environ["LIBERO_TYPE"] = "pro"
    core = importlib.import_module("liberopro.liberopro")
    benchmark = importlib.import_module("liberopro.liberopro.benchmark")
    routing = importlib.import_module("rlinf.envs.libero.utils")
    if routing.benchmark is not benchmark:
        raise RuntimeError("RLinf selected a non-PRO benchmark; refusing fallback")
    return core, benchmark


def task_language(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.count("(") != text.count(")"):
        raise ValueError(f"Unbalanced BDDL parentheses: {path}")
    match = _LANGUAGE.search(text)
    if match is None or not match.group(1).strip():
        raise ValueError(f"BDDL has no task language: {path}")
    return match.group(1).strip()


def _horizon(config):
    command = config.get("episode", {}).get("command", [])
    values = []
    for index, value in enumerate(command):
        if value == "--max-episode-steps":
            if index + 1 >= len(command):
                raise ValueError("--max-episode-steps needs a value")
            values.append(int(command[index + 1]))
        elif str(value).startswith("--max-episode-steps="):
            values.append(int(str(value).split("=", 1)[1]))
    if len(values) != 1 or values[0] <= 0:
        raise ValueError("LIBERO-PRO command must specify one positive --max-episode-steps")
    if config.get("mode", "formal") == "formal" and values != [EPISODE_HORIZON]:
        raise ValueError("Formal LIBERO-PRO requires --max-episode-steps 10000")
    return values[0]


def _validate_config(config):
    if config.get("robot") != "libero" or config.get("benchmark") != "libero-pro":
        raise ValueError("Expected robot=libero and benchmark=libero-pro")
    formal = config.get("mode", "formal") == "formal"
    if config.get("mode", "formal") not in {"formal", "probe"}:
        raise ValueError("LIBERO-PRO mode must be formal or probe")
    seeds = config.get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Scene seeds must be distinct nonnegative integers")
    expected_seeds = SCENE_SEEDS[:8] if config.get("scheduling") == "pipeline" else SCENE_SEEDS
    if formal and seeds != list(expected_seeds):
        raise ValueError(f"Formal LIBERO-PRO requires scene seeds {expected_seeds[0]} through {expected_seeds[-1]}")
    suites = config.get("suites", [])
    selected_suites = [suite.get("backend_suite") for suite in suites]
    if not selected_suites or len(set(selected_suites)) != len(selected_suites) or not set(selected_suites) <= set(BACKEND_SUITES + SWAP_SUITES):
        raise ValueError("Expected distinct LIBERO-PRO task or swap suites")
    if formal and selected_suites not in (list(BACKEND_SUITES), list(SWAP_SUITES)):
        raise ValueError("Expected a complete task or swap set in spatial, object, goal, 10 order")
    ids = set()
    for suite in suites:
        tasks = suite.get("tasks", [])
        indices = [task.get("task_index") for task in tasks]
        if not indices or len(set(indices)) != len(indices) or any(type(index) is not int or not 0 <= index < 10 for index in indices):
            raise ValueError("Task indices must be distinct integers in [0, 10)")
        if formal and indices != list(range(10)):
            raise ValueError(f"{suite.get('id')}: tasks must enumerate indices 0 through 9")
        for task in tasks:
            identifier = task.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Task IDs must be nonempty and globally unique")
            ids.add(identifier)
    return _horizon(config)


def build_manifest(config: dict[str, Any], *, backend=None, core=None):
    """Read all 40 BDDLs and state arrays, recording each physical scene index."""
    import numpy as np
    import torch

    horizon = _validate_config(config)
    if backend is None or core is None:
        core, backend = load_pro_backend()
    roots = {
        key: str(Path(core.get_libero_path(key)).expanduser().resolve())
        for key in ("benchmark_root", "bddl_files", "init_states", "assets")
    }
    for key, value in roots.items():
        if not Path(value).is_dir():
            raise FileNotFoundError(f"Missing PRO {key}: {value}")
    tasks = []
    for selected in config["suites"]:
        suite_name = selected["backend_suite"]
        suite = backend.get_benchmark(suite_name)()
        if suite.n_tasks != 10:
            raise ValueError(f"{suite_name} contains {suite.n_tasks} tasks, expected 10")
        for task_config in selected["tasks"]:
            index = task_config["task_index"]
            task = suite.get_task(index)
            bddl = Path(roots["bddl_files"]) / task.problem_folder / task.bddl_file
            init_file = Path(roots["init_states"]) / task.problem_folder / task.init_states_file
            states = torch.load(init_file, map_location="cpu", weights_only=False)
            array = np.asarray(states)
            if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
                raise ValueError(f"No valid initial states for {suite_name} task {index}: {init_file}")
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite initial states: {init_file}")
            scene_indices = [seed % len(array) for seed in config["seeds"]]
            if len(set(scene_indices)) != len(config["seeds"]):
                raise ValueError(f"Scene seeds alias within {suite_name} task {index}")
            language = task_language(bddl)
            if task.language != language:
                raise ValueError(f"PRO registry language differs from BDDL: {bddl}")
            tasks.append({
                "id": task_config["id"], "suite_id": selected["id"],
                "backend_suite": suite_name, "task_index": index,
                "task_name": task.name, "language": language,
                "bddl_path": str(bddl), "init_states_path": str(init_file),
                "init_state_count": len(array), "state_dimension": array.shape[1],
                "scenes": [{"seed": seed, "init_state_index": state_index}
                           for seed, state_index in zip(config["seeds"], scene_indices)],
                "episode_horizon": horizon,
            })
    packages = {}
    for name in ("rpent-liberopro", "rpent-rlinf", "rpent-openpi", "rlinf-libero"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repair_receipt = Path(__file__).resolve().parents[2] / "resources" / "libero-pro-authoritative" / "repairs.json"
    return {
        "schema_version": 1, "robot": "libero", "benchmark": "libero-pro",
        "backend_module": backend.__name__, "backend_path": str(Path(backend.__file__).resolve()),
        "python_executable": sys.executable, "packages": packages,
        "resource_repairs": json.loads(repair_receipt.read_text()) if repair_receipt.is_file() else [],
        "resources": roots, "suite_order": [suite["backend_suite"] for suite in config["suites"]],
        "scene_seeds": config["seeds"], "episode_horizon": horizon,
        "mode": config.get("mode", "formal"),
        "task_count": len(tasks), "tasks": tasks,
    }


resolve_manifest = build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Validated {manifest['task_count']} LIBERO-PRO tasks: {args.output}")


if __name__ == "__main__":
    main()
