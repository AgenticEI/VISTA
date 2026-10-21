# Copyright 2026 The RPent Authors.
"""Start, resume and inspect the schema-2 experiment state machine."""

import argparse
import json
import sys
import time
import subprocess
import tempfile
from pathlib import Path

from meta_harness.config import EvolutionConfig
from meta_harness.pool import ExecutionPool
from meta_harness.storage import HarnessStore, RunStore
from rpent.harness import load_harness


def benchmark_manifest(config):
    """Resolve platform resources in the actual simulator interpreter."""
    from meta_harness.config import resolve_command_sentinels
    from meta_harness.environment import lower_environment
    from rpent.robots import get_robot_spec

    module = get_robot_spec(config.robot).benchmark_module
    if module is None:
        return None
    command = resolve_command_sentinels(list(config.episode.command))
    try:
        interpreter = command[command.index("--env-python") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("benchmark preflight requires an explicit --env-python") from exc
    with tempfile.TemporaryDirectory(prefix="rpent-benchmark-") as folder:
        source, output = Path(folder) / "config.json", Path(folder) / "benchmark.json"
        source.write_text(json.dumps(config.public_record(), ensure_ascii=False))
        completed = subprocess.run(
            [interpreter, "-m", module, "--config", str(source), "--output", str(output)],
            cwd=config.repo_root, env=lower_environment(extra_names=config.episode.env_passthrough),
            capture_output=True, text=True, timeout=120,
        )
        if completed.returncode:
            raise ValueError("benchmark preflight failed: " + completed.stderr[-4000:])
        return json.loads(output.read_text())


def initialize(config_path, root):
    config = EvolutionConfig.load(config_path)
    config.require_ready()
    manifest = benchmark_manifest(config)
    store = RunStore(root)
    store.write_once("config.json", config.public_record())
    if manifest is not None:
        store.write_once("benchmark.json", manifest)
    package = load_harness(config.initial_harness)
    commit = HarnessStore(root).initialize(
        config.initial_harness, files=sorted(package.files)
    )
    store.write_once("initial.json", {"commit": commit})
    return {"root": str(store.root), "commit": commit, "mode": config.mode, "robot": config.robot}


def run_config(config_path, root, *, seconds=0):
    """The same command initializes once, then resumes the durable state."""
    from meta_harness.loop import EvolutionLoop
    from meta_harness.runtime_source import record_runtime

    if list((Path(root) / "suite-exports").glob("*.json")):
        raise ValueError("This mixed run has exported suite data. Resume the standalone target directory recorded in suite-exports, not the source run.")
    initialize(config_path, root)  # write_once rejects changed run conditions.
    loop = EvolutionLoop(root)
    record_runtime(root, loop.config)
    started = time.monotonic()
    next_progress = started
    try:
        while True:
            state = loop.tick()
            if time.monotonic() >= next_progress:
                print(json.dumps({"status": state["status"], "phase": state.get("phase"),
                                  "suite_index": state.get("suite_index"),
                                  "active_episodes": len(loop.pool.active()),
                                  "completed_rounds": sum(s["rounds_completed"] for s in state.get("scopes", {}).values()),
                                  "run": str(loop.store.root)}, ensure_ascii=False), flush=True)
                next_progress = time.monotonic() + 15
            if state["status"] != "running":
                if not loop.pool.active() and loop.pool.services is not None:
                    loop.pool.services.close()
                return state
            if seconds and time.monotonic() - started >= seconds:
                return state
            time.sleep(0.2)
    except KeyboardInterrupt:
        # Existing workers/services own their leases; a later invocation
        # resumes them. Do not terminate valid work or reset Meta's deadline.
        return {**loop.status(), "host_interrupted": True}
    except Exception:
        if not loop.pool.active() and loop.pool.services is not None:
            loop.pool.services.close()
        raise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "run":
        parser = argparse.ArgumentParser(description="Initialize or resume one schema-2 run")
        parser.add_argument("command", choices=["run"])
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--run", dest="root", type=Path, required=True)
        parser.add_argument("--seconds", type=float, default=0,
                            help="Optional host runtime limit; preserve active workers for resume")
        args = parser.parse_args(argv)
        try:
            result = run_config(args.config, args.root, seconds=args.seconds)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") != "incomplete" else 2
    parser = argparse.ArgumentParser(description="Schema-2 evolution. Start/resume: run --config CONFIG --run DIRECTORY")
    parser.add_argument(
        "command",
        choices=[
            "evolve-init",
            "evolve-run",
            "evolve-resume",
            "evolve-status",
            "pool-submit",
            "pool-run",
            "pool-status",
        ],
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evaluation-id", default="probe")
    parser.add_argument("--scope", default="probe")
    parser.add_argument("--commit")
    parser.add_argument("--purpose", default="throughput")
    parser.add_argument("--seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if args.command == "evolve-init":
        result = initialize(args.config, args.root)
    elif args.command.startswith("evolve-"):
        from meta_harness.loop import EvolutionLoop

        loop = EvolutionLoop(args.root)
        result = loop.status() if args.command == "evolve-status" else loop.run()
    else:
        pool = ExecutionPool(args.root)
        if args.command == "pool-status":
            from meta_harness.metrics import report
            result = {"evaluation": pool.evaluation_status(args.evaluation_id),
                      "metrics": report(args.root, args.evaluation_id)}
        elif args.command == "pool-submit":
            from dataclasses import asdict

            cells = [
                asdict(cell)
                for cell in pool.config.matrix(pool.config.suites[0].id, "joint")
            ]
            result = {
                "evaluation_id": pool.submit(
                    evaluation_id=args.evaluation_id,
                    harness_commit=args.commit
                    or pool.store.read("initial.json")["commit"],
                    scope_id=args.scope,
                    purpose=args.purpose,
                    cells=cells,
                )
            }
        else:
            started = time.monotonic()
            while True:
                result = pool.tick()
                if not result["active"] and all(
                    item["state"] == "complete" or item["exhausted"]
                    for item in result["evaluations"].values()
                ):
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                time.sleep(0.1)
            result = {**result, "elapsed_seconds": time.monotonic() - started}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
