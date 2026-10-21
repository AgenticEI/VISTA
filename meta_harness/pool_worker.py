# Copyright 2026 The RPent Authors.
"""Lease-owning process wrapper around the ordinary RPent CLI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from meta_harness.config import EvolutionConfig, resolve_command_sentinels
from meta_harness.environment import lower_environment, missing_lower_variables
from meta_harness.evaluation import record_episode_result
from meta_harness.pool import invalid_result
from meta_harness.storage import HarnessStore, RunStore
from rpent.utils.execution import process_identity, signal_owned


def native_reset_repair_environment(store, episode):
    path = store.path("native-reset-repair.json")
    if not path.exists():
        return {}
    repair = store.read("native-reset-repair.json")
    if repair.get("kind") != "requested-warmup-only" or repair.get("enabled") is not True:
        return {}
    identity = {k: episode.get(k) for k in ("evaluation_id", "cell_id", "task_id", "seed")}
    if identity in repair.get("cells", []):
        return {"RPENT_ROBOTWIN_REQUESTED_WARMUP": "1"}
    return {}


def normalize(episode, raw, elapsed):
    result = invalid_result(episode, "runner_result_missing", elapsed)
    if raw is None:
        return result
    if (
        raw.get("harness_commit") != episode["harness_commit"]
        or raw.get("config_ref") != episode["config_ref"]
    ):
        result["terminal_reason"] = "runner_identity_mismatch"
        return result
    for field in (
        "controller",
        "validity",
        "official_success",
        "agent_claimed_success",
        "terminal_reason",
        "trace_complete",
    ):
        result[field] = raw[field]
    if episode["controller"] == "mixed":
        result["controller"] = "mixed"
    elif episode["controller"] == "meta" and result["controller"] != "meta":
        result["controller"] = "mixed"
    result["env_steps"] = raw.get("env_steps") or 0
    return result


def planner_command(command, spec):
    """Use the infrastructure allowance captured at allocation, not current state."""
    if "planner_timeout_s" not in spec:
        return list(command)
    timeout = spec["planner_timeout_s"]
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("planner_timeout_s must be a positive integer")
    result, index = [], 0
    while index < len(command):
        token = command[index]
        if token == "--planner-timeout-s":
            index += 2
            continue
        if not token.startswith("--planner-timeout-s="):
            result.append(token)
        index += 1
    return [*result, "--planner-timeout-s", str(timeout)]


def main():
    root, eid, fd = sys.argv[1:]
    # The FD inherited from the scheduler must remain open for this lifetime.
    lease_fd = int(fd)
    os.fstat(lease_fd)
    store = RunStore(root)
    job = f"pool/jobs/{eid}"
    spec = store.read(f"{job}/spec.json")
    episode = spec["episode"]
    store.write_once(f"{job}/owner.json", process_identity())
    started = time.time()
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raw = None
    process = None
    error = None
    try:
        config = EvolutionConfig.parse(store.read("config.json"))
        from meta_harness.resource_profile import effective_config
        config, profile_revision = effective_config(store, config, spec.get("resource_profile_revision"))
        from meta_harness.campaign_runtime import effective_config as campaign_config
        config = campaign_config(store, config)
        if missing := missing_lower_variables(config.episode.required_env):
            raise ValueError("missing required environment: " + ", ".join(missing))
        harness = HarnessStore(root).materialize(episode["harness_commit"])
        task = next(
            t
            for suite in config.suites
            for t in suite.tasks
            if t.id == episode["task_id"]
        )
        output = store.path(f"episodes/{eid}/rollout")
        resources = episode["resources"]
        from rpent.robots import get_robot_spec

        robot_spec = get_robot_spec(config.robot)
        if robot_spec.evolution_launch is None:
            raise ValueError(f"{config.robot} does not support evolution launches")
        launch = robot_spec.evolution_launch(task.launch_selector(), episode["seed"], resources)
        env = lower_environment(extra_names=config.episode.env_passthrough)
        env.update(native_reset_repair_environment(store, episode))
        env.update(
            RPENT_POLICY_RNG=str(episode["policy_seed"]),
            RPENT_EPISODE_ID=eid,
            RPENT_VLA_SESSION_ID=f"{store.root.name}-{eid}",
            RPENT_VLA_SESSION_PROTOCOL=spec["session_protocol"],
            RPENT_PLANNER_SLOTS=str(store.path("pool/planner_slots")),
            RPENT_MAX_PLANNER_REQUESTS=str(config.pool.profile["max_planner_requests"]),
            RPENT_PLANNER_RPM=str(config.pool.profile.get("planner_rpm", 0)),
            RPENT_TIMING_DIR=str(store.path(f"episodes/{eid}/timing")),
            RPENT_TRACE_MODE=config.episode.trace_mode,
            RPENT_PROFILE="1",
        )
        env.update(launch["env"])
        from meta_harness.campaign_runtime import shared_spec
        if shared := shared_spec(store):
            env["RPENT_PLANNER_SLOTS"] = str(Path(shared["root"]) / "planner_slots")
            env["RPENT_MAX_PLANNER_REQUESTS"] = str(shared["max_episodes"])
        if profile_revision is not None or config.scheduling == "pipeline":
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        if config.episode.heavy_cache_steps is not None:
            env["RLDX_KEEP_HEAVY_NPY"] = str(config.episode.heavy_cache_steps)
        flags = [
            *launch["args"],
            "--harness-dir",
            str(harness),
            "--harness-commit",
            episode["harness_commit"],
            "--config-ref",
            "config.json",
            "--result-dir",
            str(output),
            "--vla-endpoint",
            resources["vla_endpoint"],
        ]
        if resources.get("sam_endpoint"):
            flags.extend(["--sam3-endpoint", resources["sam_endpoint"]])
        command = [*planner_command(resolve_command_sentinels(list(config.episode.command)), spec), *flags]
        if spec["kind"] == "live":
            env["RPENT_LIVE_MAILBOX"] = str(
                store.path(f"pool/live/{spec['live_session_id']}")
            )
            env["RPENT_LIVE_MODE"] = spec["mode"]
        store.write_once(
            f"{job}/launch.json",
            {
                "command": command,
                "prepared_at": time.time(),
                "prepare_seconds": time.time() - started,
            },
        )
        with store.path(f"{job}/runner.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "rpent.utils.execution",
                 str(store.path(f"{job}/runner-owner.json")), str(lease_fd), *command],
                cwd=config.episode.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(lease_fd,),
                start_new_session=True,
            )
            runner_owner = process_identity(process.pid)
            while process.poll() is None:
                cancelled = (
                    spec["kind"] == "live"
                    and store.path(
                        f"pool/live/{spec['live_session_id']}/close.json"
                    ).exists()
                    or episode["evaluation_id"] is not None
                    and store.path(
                        f"evaluations/{episode['evaluation_id']}/cancel.json"
                    ).exists()
                )
                if stopped or cancelled or (spec["deadline"] is not None and time.time() >= spec["deadline"]):
                    stopped = stopped or cancelled
                    signal_owned(runner_owner)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        signal_owned(runner_owner, signal.SIGKILL)
                    break
                time.sleep(0.05)
            process.wait()
        if (output / "result.json").is_file():
            raw = json.loads((output / "result.json").read_text())
        result = normalize(episode, raw, time.time() - started)
        if stopped or (spec["deadline"] is not None and time.time() >= spec["deadline"]):
            result = invalid_result(
                episode,
                "cancelled" if stopped else "episode_deadline",
                time.time() - started,
            )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        result = invalid_result(episode, "worker_error", time.time() - started)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    if raw is not None and config.robot == "libero":
        from rpent.deployment_time import estimate
        result["deployment_time"] = estimate(
            output, store.path(f"episodes/{eid}/timing"), env_steps=result["env_steps"],
            expected_planner_calls=raw.get("metrics", {}).get("model_requests"),
        )
    record_episode_result(store, eid, result)
    store.write_once(
        f"{job}/completion.json",
        {
            "elapsed_seconds": time.time() - started,
            "error": error,
            "returncode": process.returncode if process else None,
        },
    )
    os.close(lease_fd)


if __name__ == "__main__":
    main()
