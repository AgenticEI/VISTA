# Copyright 2025 The RPent Team. All rights reserved.
"""Run RPent episodes and collect the shared pool evaluation records."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from meta_harness._files import atomic_json, read_json
from meta_harness.config import (
    TaskCase,
    resolve_env_cuda_command,
    validate_env_cuda_devices,
)
from rpent.harness.package import load_harness


@dataclass(frozen=True)
class EpisodeSummary:
    """The small public subset used by selection."""

    result_dir: Path
    validity: Literal["valid", "invalid"]
    official_success: bool | None
    model_requests: int | None
    agent_turns: int | None
    tool_calls: int | None
    env_steps: int | None
    elapsed_s: float | None
    terminal_reason: str | None

    @property
    def comparable(self) -> bool:
        return self.validity == "valid" and self.official_success is not None


class EpisodeRunner(Protocol):
    def run_episode(
        self,
        *,
        harness: Path,
        task: TaskCase,
        repeat_index: int,
        result_dir: Path,
    ) -> Path:
        """Run one fresh episode and return its Result directory."""


class SubprocessEpisodeRunner:
    """Invoke the unmodified RPent CLI once per Result directory."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        timeout_s: float,
        environment: Mapping[str, str],
        working_directory: Path,
        terminate_grace_s: float = 10.0,
        policy_seed: int | None = None,
        heavy_cache_steps: int | None = None,
        env_cuda_devices: tuple[int, ...] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if timeout_s <= 0 or terminate_grace_s <= 0:
            raise ValueError("timeouts must be positive")
        self.command = command
        self.timeout_s = timeout_s
        self.environment = dict(environment)
        self.working_directory = working_directory.resolve()
        self.terminate_grace_s = terminate_grace_s
        self.policy_seed = policy_seed
        self.heavy_cache_steps = heavy_cache_steps
        self.env_cuda_devices = validate_env_cuda_devices(env_cuda_devices)

    def episode_command(self, task: TaskCase, repeat_index: int) -> list[str]:
        return resolve_env_cuda_command(self.command, self.env_cuda_devices, task.seed, repeat_index)

    def episode_environment(self, task: TaskCase, repeat_index: int) -> dict[str, str]:
        environment = dict(self.environment)
        if self.heavy_cache_steps is not None:
            environment["RLDX_KEEP_HEAVY_NPY"] = str(self.heavy_cache_steps)
        if self.policy_seed is not None:
            # Same scene/repeat has the same RNG across Harnesses and retries;
            # distinct repetitions must not replay one fixed policy RNG.
            if not 0 <= repeat_index < 10000:
                raise ValueError("seeded evaluation requires repeat_index < 10000")
            seed = self.policy_seed + task.seed * 10000 + repeat_index
            if not 0 <= seed < 2**63:
                raise ValueError("derived policy RNG seed is outside [0, 2**63)")
            environment["RPENT_POLICY_RNG"] = str(seed)
        return environment

    def run_episode(
        self,
        *,
        harness: Path,
        task: TaskCase,
        repeat_index: int,
        result_dir: Path,
    ) -> Path:
        result_dir = result_dir.resolve()
        result_dir.mkdir(parents=True, exist_ok=False)
        episode_environment = self.episode_environment(task, repeat_index)
        command = [
            *self.episode_command(task, repeat_index),
            "--task-name",
            task.task_name,
            "--split",
            task.split,
            "--seed",
            str(task.seed),
            "--harness-dir",
            str(harness.resolve()),
            "--result-dir",
            str(result_dir),
            "--output-dir",
            str(result_dir),
        ]
        log_path = result_dir / "runner.log"
        started = time.monotonic()
        timed_out = False
        launch_error: str | None = None
        returncode: int | None = None
        process: subprocess.Popen[str] | None = None
        interruption: BaseException | None = None
        interruption_traceback = None
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.working_directory,
                    env=episode_environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = self._stop_process_group(process)
        except OSError as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
        except BaseException as exc:
            interruption = exc
            interruption_traceback = exc.__traceback__
            if process is not None:
                returncode = self._stop_process_group(process)
        finally:
            interruption_name = (
                type(interruption).__name__ if interruption is not None else None
            )
            atomic_json(
                result_dir / "meta_runner.json",
                {
                    "schema_version": 1,
                    "command": command,
                    "repeat_index": repeat_index,
                    "policy_rng": episode_environment.get("RPENT_POLICY_RNG"),
                    "heavy_cache_steps": episode_environment.get("RLDX_KEEP_HEAVY_NPY", "25"),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "launch_error": launch_error,
                    "interrupted": interruption is not None,
                    "interruption": interruption_name,
                    "elapsed_s": time.monotonic() - started,
                },
            )
            if not (result_dir / "result.json").exists():
                if interruption is not None:
                    reason = "runner_interrupted"
                    error = f"{interruption_name}: {interruption}"
                elif timed_out:
                    reason = "episode_timeout"
                    error = f"RPent exceeded the {self.timeout_s:.0f}s episode timeout"
                elif launch_error:
                    reason = "launch_error"
                    error = launch_error
                elif returncode:
                    reason = "process_exit"
                    error = f"RPent exited with status {returncode}"
                else:
                    reason = "missing_result"
                    error = "RPent exited without writing result.json"
                from rpent.result import write_invalid_result

                identity = load_harness(harness).identity
                write_invalid_result(
                    result_dir,
                    task={
                        "task_name": task.task_name,
                        "split": task.split,
                        "seed": task.seed,
                        "repeat_index": repeat_index,
                    },
                    harness={
                        "id": identity.harness_id,
                        "schema_version": identity.schema_version,
                        "commit": identity.commit,
                    },
                    agent={"kind": "subprocess", "returncode": returncode},
                    reason=reason,
                    error=error,
                    experiment={"episode_timeout_s": self.timeout_s},
                )
        if interruption is not None:
            raise interruption.with_traceback(interruption_traceback)
        return result_dir

    def _stop_process_group(self, process: subprocess.Popen[str]) -> int | None:
        """Terminate the complete RPent process group, escalating if needed."""
        if process.poll() is not None:
            return process.returncode
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=self.terminate_grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.wait()


def load_episode_result(result_dir: str | Path) -> EpisodeSummary:
    """Load the stable fields while tolerating harmless schema nesting changes."""
    root = Path(result_dir).resolve()
    manifest_path = root / "result.json"
    try:
        value = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return EpisodeSummary(
            root,
            "invalid",
            None,
            None,
            None,
            None,
            None,
            None,
            "missing_or_malformed_result",
        )

    raw_validity = value.get("validity", "invalid")
    if isinstance(raw_validity, bool):
        validity: Literal["valid", "invalid"] = "valid" if raw_validity else "invalid"
    else:
        validity = "valid" if raw_validity == "valid" else "invalid"
    outcome = value.get("outcome") if isinstance(value.get("outcome"), dict) else {}
    task_status = (
        value.get("task_status") if isinstance(value.get("task_status"), dict) else {}
    )
    official = value.get("official_success")
    if official is None:
        official = outcome.get("official_success")
    if official is None:
        official = task_status.get("official_success")
    if not isinstance(official, bool):
        official = None
    metrics = value.get("metrics") if isinstance(value.get("metrics"), dict) else {}

    def optional_int(*names: str) -> int | None:
        for name in names:
            candidate = metrics.get(name, value.get(name))
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
        return None

    def optional_float(*names: str) -> float | None:
        for name in names:
            candidate = metrics.get(name, value.get(name))
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return float(candidate)
        return None

    reason = value.get("terminal_reason", outcome.get("terminal_reason"))
    return EpisodeSummary(
        result_dir=root,
        validity=validity,
        official_success=official,
        model_requests=optional_int("model_requests", "llm_requests"),
        agent_turns=optional_int("agent_turns", "turns"),
        tool_calls=optional_int("tool_calls"),
        env_steps=optional_int("env_steps", "episode_steps"),
        elapsed_s=optional_float("elapsed_s", "wall_time_s"),
        terminal_reason=str(reason) if reason is not None else None,
    )


# Run protocol v1. These are persistence/read functions, not a second runner or
# scheduler. W4 dispatches their pending cells through the shared pool.

def _record_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{field} must be a plain identifier")
    return value


def _record_integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _record_fields(value: Any, fields: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != fields:
        raise ValueError(
            f"{label}: missing fields {sorted(fields - set(value))}; "
            f"unknown fields {sorted(set(value) - fields)}"
        )


def _record_commit(value: Any) -> str:
    # This validates an externally supplied full Git identity, never computes a
    # file, directory, configuration, or artifact digest.
    if not isinstance(value, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
        raise ValueError("harness_commit must be a full Git commit ID")
    return value


def _record_reference(value: Any, field: str) -> str:
    from pathlib import PurePosixPath

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a run-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts or value == ".":
        raise ValueError(f"{field} must be a canonical run-relative path")
    return value


def validate_evaluation_request(request: dict[str, Any]) -> None:
    """Validate the complete, immutable batch request before dispatch."""
    _record_fields(request, {
        "schema_version", "evaluation_id", "purpose", "scope_id",
        "harness_commit", "config_ref", "controller", "cells",
    }, "evaluation request")
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("unsupported evaluation request schema_version")
    _record_id(request["evaluation_id"], "evaluation_id")
    _record_reference(request["scope_id"], "scope_id")
    _record_reference(request["config_ref"], "config_ref")
    _record_commit(request["harness_commit"])
    if request["purpose"] not in {"formal", "baseline", "development", "diagnostic", "throughput"}:
        raise ValueError("unsupported evaluation purpose; retries retain the original purpose")
    if request["controller"] not in {"planner", "meta", "mixed"}:
        raise ValueError("unsupported controller")
    if request["purpose"] in {"formal", "baseline"} and request["controller"] != "planner":
        raise ValueError("formal/baseline evaluations require autonomous Planner control")
    if not isinstance(request["cells"], list) or not request["cells"]:
        raise ValueError("evaluation cells must be a non-empty list")
    ids, conditions = set(), set()
    for cell in request["cells"]:
        _record_fields(cell, {"cell_id", "task_id", "seed", "repeat"}, "evaluation cell")
        _record_id(cell["cell_id"], "cell_id")
        _record_id(cell["task_id"], "task_id")
        _record_integer(cell["seed"], "seed")
        _record_integer(cell["repeat"], "repeat")
        if cell["repeat"] >= 10000:
            raise ValueError("repeat must be < 10000 for the existing policy RNG formula")
        condition = (cell["task_id"], cell["seed"], cell["repeat"])
        if cell["cell_id"] in ids or condition in conditions:
            raise ValueError("duplicate cell ID or task/seed/repeat condition")
        ids.add(cell["cell_id"])
        conditions.add(condition)


def plan_evaluation(
    store: Any, *, evaluation_id: str, harness_commit: str, scope_id: str,
    purpose: str, cells: list[dict[str, Any]], controller: str = "planner",
    config_ref: str = "config.json",
) -> dict[str, Any]:
    """Persist a batch before pool submission, or verify an identical retry."""
    request = {
        "schema_version": 1, "evaluation_id": evaluation_id, "purpose": purpose,
        "scope_id": scope_id, "harness_commit": harness_commit,
        "config_ref": config_ref, "controller": controller,
        "cells": [dict(cell) for cell in cells],
    }
    validate_evaluation_request(request)
    store.read(config_ref)  # A request must point at already frozen conditions.
    store.write_once(f"evaluations/{evaluation_id}/request.json", request)
    return request


def _read_request(store: Any, evaluation_id: str) -> dict[str, Any]:
    _record_id(evaluation_id, "evaluation_id")
    request = store.read(f"evaluations/{evaluation_id}/request.json")
    validate_evaluation_request(request)
    if request["evaluation_id"] != evaluation_id:
        raise ValueError("request evaluation_id does not match its directory")
    return request


def _validate_episode(store: Any, record: dict[str, Any]) -> None:
    _record_fields(record, {
        "schema_version", "episode_id", "evaluation_id", "cell_id",
        "execution_attempt", "task_id", "seed", "repeat", "policy_seed",
        "harness_commit", "config_ref", "purpose", "controller", "resources",
        "started_at",
    }, "episode")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("unsupported episode schema_version")
    _record_id(record["episode_id"], "episode_id")
    _record_id(record["task_id"], "task_id")
    _record_integer(record["execution_attempt"], "execution_attempt", 1)
    for field in ("seed", "repeat", "policy_seed"):
        _record_integer(record[field], field)
    if record["repeat"] >= 10000 or record["policy_seed"] >= 2**63:
        raise ValueError("episode repeat or policy_seed is outside the runner RNG range")
    _record_commit(record["harness_commit"])
    _record_reference(record["config_ref"], "config_ref")
    if record["controller"] not in {"planner", "meta", "mixed"}:
        raise ValueError("unsupported episode controller")
    if record["purpose"] not in {"formal", "baseline", "development", "diagnostic", "throughput"}:
        raise ValueError("unsupported episode purpose")
    from datetime import datetime

    if not isinstance(record["started_at"], str):
        raise ValueError("started_at must be an ISO 8601 timestamp with timezone")
    try:
        timestamp = datetime.fromisoformat(record["started_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("started_at must be an ISO 8601 timestamp with timezone") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("started_at must include a timezone")
    _record_fields(record["resources"], {
        "env_gpu", "vla_endpoint", "sam_endpoint", "session_id",
    }, "episode resources")
    _record_integer(record["resources"]["env_gpu"], "env_gpu")
    _record_id(record["resources"]["session_id"], "session_id")
    config = store.read(record["config_ref"])
    for field in ("vla_endpoint", "sam_endpoint"):
        if field == "sam_endpoint" and config.get("robot") == "robotwin" and record["resources"][field] is None:
            continue
        if not isinstance(record["resources"][field], str) or not record["resources"][field]:
            raise ValueError(f"resources.{field} must identify the assigned endpoint")
    base_seed = config.get("episode", {}).get("policy_seed")
    _record_integer(base_seed, "config episode.policy_seed")
    expected_seed = base_seed + record["seed"] * 10000 + record["repeat"]
    if record["policy_seed"] != expected_seed:
        raise ValueError("episode policy_seed differs from the configured scene/repeat RNG")
    if record["evaluation_id"] is None:
        if record["cell_id"] is not None or record["purpose"] in {"formal", "baseline"}:
            raise ValueError("a standalone live episode cannot claim a formal evaluation cell")
        return
    request = _read_request(store, record["evaluation_id"])
    matching = [cell for cell in request["cells"] if cell["cell_id"] == record["cell_id"]]
    if not matching:
        raise ValueError("episode cell_id is absent from its evaluation request")
    expected = {**matching[0], **{field: request[field] for field in (
        "harness_commit", "config_ref", "purpose", "controller",
    )}}
    for field, value in expected.items():
        if record[field] != value or type(record[field]) is not type(value):
            raise ValueError(f"episode {field} differs from the fixed evaluation request")


def record_episode(store: Any, record: dict[str, Any]) -> None:
    """Allocate an attempt before dispatch; the worker owns its episode files.

    Small allocation references let recovery read only this evaluation's
    attempts. They precede episode.json so an interrupted allocation is still
    discoverable without scanning the run's images or all prior episodes.
    """
    _validate_episode(store, record)
    episode_id = record["episode_id"]
    episode_ref = f"episodes/{episode_id}/episode.json"
    # Reserve the global episode ID before publishing any cell allocation.
    # A crash between the following writes must not let another cell reuse it.
    store.write_once(f"episodes/{episode_id}/reservation.json", record)
    # Check ID reuse before publishing a reference to an existing episode.
    try:
        existing = store.read(episode_ref)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        store.write_once(episode_ref, record)
    if record["evaluation_id"] is not None:
        store.write_once(
            f"evaluations/{record['evaluation_id']}/attempts/{record['cell_id']}/"
            f"a{record['execution_attempt']:04d}.json",
            {"episode_id": episode_id, "episode_ref": episode_ref,
             "execution_attempt": record["execution_attempt"]},
        )
    store.write_once(episode_ref, record)


def _validate_recorded_result(episode: dict[str, Any], result: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise ValueError("episode result must be an object")
    _record_fields({k: v for k, v in result.items() if k != "deployment_time"}, {
        "schema_version", "episode_id", "evaluation_id", "cell_id",
        "execution_attempt", "controller", "validity", "official_success",
        "agent_claimed_success", "terminal_reason", "env_steps", "trace_complete",
        "elapsed_seconds", "completed_at",
    }, "episode result")
    for field in ("schema_version", "episode_id", "evaluation_id", "cell_id", "execution_attempt"):
        if result[field] != episode[field] or type(result[field]) is not type(episode[field]):
            raise ValueError(f"result {field} differs from episode identity")
    if result["controller"] not in {episode["controller"], "mixed"}:
        raise ValueError("result controller cannot relabel a manual episode as autonomous")
    if result["validity"] not in {"valid", "invalid"}:
        raise ValueError("result validity must be valid or invalid")
    if type(result["trace_complete"]) is not bool:
        raise ValueError("trace_complete must be a boolean")
    if result["validity"] == "valid":
        if type(result["official_success"]) is not bool or not result["trace_complete"]:
            raise ValueError("a valid result needs an official outcome and a drained, complete trace")
    elif result["official_success"] is not None:
        raise ValueError("invalid official_success must be null")
    if result["agent_claimed_success"] is not None and type(result["agent_claimed_success"]) is not bool:
        raise ValueError("agent_claimed_success must be a boolean or null")
    _record_integer(result["env_steps"], "env_steps")
    if "deployment_time" in result:
        from rpent.deployment_time import validate
        validate(result["deployment_time"], result["env_steps"])
    elapsed = result["elapsed_seconds"]
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("elapsed_seconds must be finite and nonnegative")
    if not isinstance(result["terminal_reason"], str) or not result["terminal_reason"]:
        raise ValueError("terminal_reason must describe how the episode ended")
    from datetime import datetime

    try:
        completed = datetime.fromisoformat(result["completed_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("completed_at must be the host's timestamp with timezone") from exc
    if completed.utcoffset() is None:
        raise ValueError("completed_at must include a timezone")


def record_episode_result(store: Any, episode_id: str, result: dict[str, Any]) -> None:
    """Publish the worker's final record only after evidence writer completion."""
    _record_id(episode_id, "episode_id")
    episode = store.read(f"episodes/{episode_id}/episode.json")
    _validate_episode(store, episode)
    if episode["episode_id"] != episode_id:
        raise ValueError("episode_id differs from its directory")
    from datetime import datetime, timezone

    # Completion time comes from the host, not from Agent self-report. Keep it
    # stable when the exact same worker publication is retried after a crash.
    result = dict(result)
    result_ref = f"episodes/{episode_id}/result.json"
    try:
        previous = store.read(result_ref)
    except FileNotFoundError:
        previous = None
    result["completed_at"] = (previous["completed_at"] if previous is not None
                              else datetime.now(timezone.utc).isoformat())
    _validate_recorded_result(episode, result)
    store.write_once(result_ref, result)


def collect_evaluation(store: Any, evaluation_id: str) -> dict[str, Any]:
    """Collect facts and acknowledge each cell's first valid completion once.

    Called by the pool/scheduler, this fixes the deduplication choice separately
    from the rebuildable summary. Workers still write only their own results;
    later duplicate completion never replaces an acknowledged result.
    """
    from datetime import datetime

    from meta_harness.storage import RecordConflict

    request = _read_request(store, evaluation_id)
    from meta_harness.resource_profile import current_revision
    profile_revision = current_revision(store)
    runtime_pointer = store.path("runtime-current.json")
    runtime_revision = store.read("runtime-current.json").get("revision") if runtime_pointer.exists() else None
    details, selected, by_task, by_seed = {}, {}, {}, {}
    completed = valid = invalid = non_autonomous = successes = 0
    attempt_count = finished_attempts = retry_attempts = 0
    elapsed_sum = 0.0
    for cell in request["cells"]:
        cell_id = cell["cell_id"]
        attempts = []
        folder = store.root / f"evaluations/{evaluation_id}/attempts/{cell_id}"
        # Only small allocation metadata is traversed; no history/asset scan.
        for path in sorted(folder.glob("a*.json")):
            allocation = store.read(path.relative_to(store.root).as_posix())
            _record_fields(allocation, {"episode_id", "episode_ref", "execution_attempt"}, "allocation")
            episode_id = _record_id(allocation["episode_id"], "episode_id")
            attempt = _record_integer(allocation["execution_attempt"], "execution_attempt", 1)
            if path.name != f"a{attempt:04d}.json" or allocation["episode_ref"] != f"episodes/{episode_id}/episode.json":
                raise ValueError("allocation identity differs from its path")
            entry = {"episode_id": episode_id, "execution_attempt": attempt, "result": None}
            job_path = store.path(f"pool/jobs/{episode_id}/spec.json")
            job_spec = store.read(f"pool/jobs/{episode_id}/spec.json") if job_path.exists() else {}
            job_profile = job_spec.get("resource_profile_revision")
            entry["current_profile"] = job_profile == profile_revision
            entry["current_execution"] = job_spec.get("runtime_revision") == runtime_revision
            entry["resource_profile_revision"] = job_profile
            try:
                episode = store.read(allocation["episode_ref"])
            except FileNotFoundError:
                attempts.append(entry)
                continue
            _validate_episode(store, episode)
            expected = {"evaluation_id": evaluation_id, "cell_id": cell_id,
                        "episode_id": episode_id, "execution_attempt": attempt}
            if any(episode[field] != value for field, value in expected.items()):
                raise ValueError("allocated episode belongs to a different evaluation/cell/attempt")
            try:
                result = store.read(f"episodes/{episode_id}/result.json")
            except FileNotFoundError:
                result = None
            if result is not None:
                _validate_recorded_result(episode, result)
                entry["result"] = result
                entry["backoff_exempt_after_repair"] = store.path(
                    f"pool/jobs/{episode_id}/retry-after-repair.json").exists()
            attempts.append(entry)
        attempts.sort(key=lambda entry: entry["execution_attempt"])
        finished = [entry for entry in attempts if entry["result"] is not None]
        usable = [entry for entry in finished if entry["result"]["validity"] == "valid"
                  and entry["result"]["controller"] == request["controller"]]
        usable.sort(key=lambda entry: (datetime.fromisoformat(entry["result"]["completed_at"]),
                                       entry["execution_attempt"]))
        chosen = usable[0] if usable else None
        acknowledgement = f"evaluations/{evaluation_id}/attempts/{cell_id}/first_valid.json"
        if chosen:
            try:
                store.write_once(acknowledgement, {"episode_id": chosen["episode_id"]})
            except RecordConflict:
                pass  # An earlier collection already acknowledged a result.
        try:
            first_valid = store.read(acknowledgement)
        except FileNotFoundError:
            first_valid = None
        if first_valid is not None:
            _record_fields(first_valid, {"episode_id"}, "first valid completion")
            matching = [entry for entry in usable if entry["episode_id"] == first_valid["episode_id"]]
            if not matching:
                # Another collector can acknowledge a result that arrived
                # after this collector read the attempts. Refresh that one
                # immutable fact instead of mistaking a stale view for damage.
                acknowledged_id = _record_id(first_valid["episode_id"], "episode_id")
                episode = store.read(f"episodes/{acknowledged_id}/episode.json")
                _validate_episode(store, episode)
                result = store.read(f"episodes/{acknowledged_id}/result.json")
                _validate_recorded_result(episode, result)
                if (episode["episode_id"] != acknowledged_id
                        or episode["evaluation_id"] != evaluation_id or episode["cell_id"] != cell_id
                        or result["validity"] != "valid" or result["controller"] != request["controller"]):
                    raise ValueError("acknowledged result belongs to another cell or is not valid")
                allocation = store.read(
                    f"evaluations/{evaluation_id}/attempts/{cell_id}/a{episode['execution_attempt']:04d}.json"
                )
                expected_allocation = {"episode_id": acknowledged_id,
                                       "episode_ref": f"episodes/{acknowledged_id}/episode.json",
                                       "execution_attempt": episode["execution_attempt"]}
                if allocation != expected_allocation:
                    raise ValueError("acknowledged result conflicts with its allocation")
                refreshed = {"episode_id": acknowledged_id, "current_profile": True,
                             "current_execution": True,
                             "execution_attempt": episode["execution_attempt"], "result": result}
                attempts = [entry for entry in attempts if entry["episode_id"] != acknowledged_id]
                attempts.append(refreshed)
                attempts.sort(key=lambda entry: entry["execution_attempt"])
                finished = [entry for entry in attempts if entry["result"] is not None]
                matching = [refreshed]
            chosen = matching[0]
        attempt_count += len(attempts)
        retry_attempts += sum(entry["execution_attempt"] > 1 for entry in attempts)
        finished_attempts += len(finished)
        elapsed_sum += sum(entry["result"]["elapsed_seconds"] for entry in finished)
        completed += bool(finished)
        valid += chosen is not None
        invalid += chosen is None and any(entry["result"]["validity"] == "invalid" for entry in finished)
        non_autonomous += chosen is None and any(
            entry["result"]["controller"] != request["controller"] for entry in finished
        )
        success = bool(chosen and chosen["result"]["official_success"])
        # Manual/mixed diagnostics retain official outcomes but never enter an
        # autonomous success-rate denominator.
        autonomous = request["controller"] == "planner"
        successes += int(success and autonomous)
        if chosen:
            selected[cell_id] = chosen["episode_id"]
        details[cell_id] = {
            **cell, "selected_episode": chosen["episode_id"] if chosen else None,
            "official_success": chosen["result"]["official_success"] if chosen else None,
            "deployment_time": chosen["result"].get("deployment_time") if chosen else None,
            "resource_profile_revision": chosen.get("resource_profile_revision") if chosen else None,
            "attempts": [{"episode_id": entry["episode_id"],
                          "execution_attempt": entry["execution_attempt"],
                          "current_profile": entry["current_profile"],
                          "current_execution": entry["current_execution"],
                          "backoff_exempt_after_repair": entry.get("backoff_exempt_after_repair", False),
                          "complete": entry["result"] is not None,
                          "validity": (entry["result"] or {}).get("validity"),
                          "completed_at": (entry["result"] or {}).get("completed_at")}
                         for entry in attempts],
        }
        for groups, key in ((by_task, cell["task_id"]), (by_seed, str(cell["seed"]))):
            group = groups.setdefault(key, {"requested": 0, "valid": 0, "successes": 0})
            group["requested"] += 1
            group["valid"] += int(chosen is not None and autonomous)
            group["successes"] += int(success and autonomous)
    denominator = valid if request["controller"] == "planner" else 0
    requested = len(request["cells"])
    successful_cells = [cell for cell in details.values() if cell["official_success"] is True]
    times = [cell["deployment_time"]["total_seconds"] for cell in successful_cells
             if (cell.get("deployment_time") or {}).get("available")]
    return {
        "schema_version": 1, "evaluation_id": evaluation_id,
        "resource_profile_revision": profile_revision,
        "harness_commit": request["harness_commit"], "config_ref": request["config_ref"],
        "purpose": request["purpose"], "controller": request["controller"],
        "state": "complete" if valid == requested else "incomplete",
        "comparable": denominator == requested,
        "requested": requested, "completed": completed, "valid": valid,
        "invalid": invalid, "non_autonomous": non_autonomous,
        "successes": successes, "denominator": denominator,
        "success_rate": successes / denominator if denominator else None,
        "pending_cells": [key for key, value in details.items() if value["selected_episode"] is None],
        "selected_episodes": selected, "cells": details,
        "by_task": by_task, "by_seed": by_seed,
        "deployment_time": {"successful_episodes": len(successful_cells),
                            "measured_successful_episodes": len(times),
                            "complete": bool(successful_cells) and len(times) == len(successful_cells),
                            "mean_seconds": sum(times) / len(times) if times else None},
        "cost": {"episode_attempts": attempt_count, "finished_attempts": finished_attempts,
                 "retry_attempts": retry_attempts, "episode_elapsed_seconds_sum": elapsed_sum},
    }


def pending_evaluation_cells(
    summary: dict[str, Any], *, max_retries: int | None, active_episode_ids: set[str],
    now: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Tell the pool what is retryable, backing off, or still owned by a worker.

    Process ownership is supplied by the pool's leases. This function never
    treats a missing result as proof that its process has stopped.
    Infrastructure failures are not task failures. Exhausted cells remain
    incomplete for operator recovery rather than consuming unlimited attempts.
    """
    if max_retries is not None:
        _record_integer(max_retries, "max_retries")
    now = time.time() if now is None else now
    pending: dict[str, list[dict[str, Any]]] = {"ready": [], "active": [], "exhausted": [], "backoff": []}
    for cell_id in summary["pending_cells"]:
        detail = summary["cells"][cell_id]
        attempts = detail["attempts"]
        next_attempt = max((entry["execution_attempt"] for entry in attempts), default=0) + 1
        cell = {field: detail[field] for field in ("cell_id", "task_id", "seed", "repeat")}
        cell["execution_attempt"] = next_attempt
        if any(entry["episode_id"] in active_episode_ids for entry in attempts):
            pending["active"].append(cell)
        elif max_retries is not None and next_attempt > 1 + max_retries:
            pending["exhausted"].append(cell)
        elif _in_infra_backoff(attempts, now):
            pending["backoff"].append(cell)
        else:
            pending["ready"].append(cell)
    return pending


#: Hold a cell this long after its latest infrastructure failure, so a brief
#: provider/simulator outage window passes before the next attempt is spent.
INFRA_RETRY_BACKOFF_S = 300.0


def _in_infra_backoff(attempts: list[dict[str, Any]], now: float) -> bool:
    latest: float | None = None
    for entry in attempts:
        if entry.get("backoff_exempt_after_repair"):
            continue
        if not entry.get("complete") or entry.get("validity") != "invalid":
            continue
        completed_at = entry.get("completed_at")
        if not completed_at:
            continue
        try:
            moment = datetime.fromisoformat(completed_at).timestamp()
        except (TypeError, ValueError):
            continue
        latest = moment if latest is None else max(latest, moment)
    return latest is not None and now - latest < INFRA_RETRY_BACKOFF_S
