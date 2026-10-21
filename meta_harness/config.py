# Copyright 2025 The RPent Team. All rights reserved.
"""Strict, secret-free experiment configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from meta_harness._files import read_json, resolve_repo_path

META_MODEL = "gpt-5.5"  # Provider fallback; formal runs pass their explicit model.
LOWER_PLANNER = "api"

# Interpreter sentinels for episode.command.
#
# A tracked config must never name an absolute interpreter path. Every config in
# this repository used to point at
# /data2/.../RPent-robocasa-baseline/.venv/bin/python, that directory was later
# moved under .archive/, and all of them broke at once while still looking
# valid. HOST_INTERPRETER_SENTINEL resolves to the interpreter actually running
# meta_harness, so the host side cannot drift from the code it is executing;
# there is nothing to update when the environment moves.
#
# The simulation interpreter stays a separate sentinel rather than collapsing
# into the same one: it owns mujoco/robosuite, which the host environment
# deliberately does not. It comes from the environment (configs/env.sh) because
# it is genuinely a second installation, not a second name for this one.
HOST_INTERPRETER_SENTINEL = "@RPENT_PYTHON@"
ENV_INTERPRETER_SENTINEL = "@RPENT_ENV_PYTHON@"
ENV_INTERPRETER_VARIABLE = "RPENT_ENV_PYTHON"


def resolve_command_sentinels(command: list[str]) -> list[str]:
    """Substitute the interpreter sentinels in an ``episode.command`` list."""
    resolved: list[str] = []
    for item in command:
        if item == HOST_INTERPRETER_SENTINEL:
            resolved.append(sys.executable)
        elif item == ENV_INTERPRETER_SENTINEL:
            value = os.environ.get(ENV_INTERPRETER_VARIABLE)
            if not value:
                raise ValueError(
                    f"episode.command requests {ENV_INTERPRETER_SENTINEL} but "
                    f"${ENV_INTERPRETER_VARIABLE} is unset; source configs/env.sh "
                    "or set it explicitly"
                )
            resolved.append(value)
        else:
            resolved.append(item)
    return resolved


def validate_model_name(value: Any, field: str = "model") -> str:
    """Validate an explicit model identifier, without choosing a provider/model."""
    if (not isinstance(value, str) or not value or value.startswith("-")
            or any(character.isspace() or ord(character) < 32 for character in value)):
        raise ValueError(f"{field} must be a non-empty model identifier without whitespace")
    return value


@dataclass(frozen=True)
class TaskCase:
    """One task and scene passed to the shared episode runner."""

    id: str
    task_name: str
    split: str = "target"
    seed: int = 0

    @classmethod
    def parse(cls, value: Any, index: int) -> "TaskCase":
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("task_name"), str)
            or not value["task_name"].strip()
        ):
            raise ValueError(f"tasks[{index}] must contain a string task_name")
        task_id = value.get("id") or f"task-{index:02d}"
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"tasks[{index}].id must be a non-empty string")
        if any(character in task_id for character in "/\\") or task_id in {".", ".."}:
            raise ValueError(f"tasks[{index}].id is not a safe directory name")
        split = value.get("split", "target")
        if split not in {"target", "pretrain", "all"}:
            raise ValueError(f"tasks[{index}].split is invalid: {split!r}")
        return cls(
            id=task_id,
            task_name=value["task_name"],
            split=split,
            seed=int(value.get("seed", 0)),
        )


def validate_env_cuda_devices(value: Any) -> tuple[int, ...] | None:
    """Validate an optional host-owned environment GPU pool."""
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(device) is not int or device < 0 for device in value)
            or len(set(value)) != len(value)):
        raise ValueError("env_cuda_devices must contain distinct nonnegative integer devices")
    return tuple(value)


def resolve_env_cuda_command(
    command: tuple[str, ...] | list[str], devices: tuple[int, ...] | None,
    scene_seed: int, repeat_index: int,
) -> list[str]:
    """Assign the same scene/repeat to the same GPU across all Harness versions."""
    devices = validate_env_cuda_devices(devices)
    if devices is None:
        return list(command)
    if any(type(value) is not int or value < 0 for value in (scene_seed, repeat_index)):
        raise ValueError("GPU assignment requires nonnegative integer scene/repeat indices")
    replacement = ["--env-cuda-device", str(devices[(scene_seed + repeat_index) % len(devices)])]
    result, index, replaced = [], 0, False
    while index < len(command):
        token = command[index]
        if token == "--env-cuda-device" or token.startswith("--env-cuda-device="):
            if token == "--env-cuda-device":
                if index + 1 >= len(command) or command[index + 1].startswith("--"):
                    raise ValueError("--env-cuda-device requires a value")
                index += 1
            if not replaced:
                result.extend(replacement)
                replaced = True
        else:
            result.append(token)
        index += 1
    if not replaced:
        result.extend(replacement)
    return result


@dataclass(frozen=True)
class EpisodeConfig:
    """RPent subprocess settings common to all evaluation cells."""

    command: tuple[str, ...]
    timeout_s: float = 14400.0
    required_env: tuple[str, ...] = ()
    env_passthrough: tuple[str, ...] = ()
    cwd: Path | None = None
    policy_seed: int | None = None
    heavy_cache_steps: int | None = None
    trace_mode: str | None = None
    env_cuda_devices: tuple[int, ...] | None = None

    @classmethod
    def parse(
        cls, value: Any, *, root: Path, resolve_interpreters: bool = True,
    ) -> "EpisodeConfig":
        if not isinstance(value, dict):
            raise ValueError("episode must be an object")
        command = value.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError("episode.command must be a non-empty list of strings")
        if resolve_interpreters:
            command = resolve_command_sentinels(list(command))
        cls._require_flag(command, "--planner", LOWER_PLANNER)
        cls._require_flag(command, "--model", None)
        reserved = {
            "--task-name",
            "--split",
            "--seed",
            "--harness-dir",
            "--result-dir",
            "--output-dir",
        }
        collision = reserved.intersection(command)
        if collision:
            raise ValueError(
                "episode.command must not contain per-episode flags: "
                + ", ".join(sorted(collision))
            )
        forbidden_flags = {"--api-key", "--token", "--password", "--secret"}
        if forbidden_flags.intersection(command):
            raise ValueError(
                "credentials must come from environment variables, not command flags"
            )
        timeout_s = float(value.get("timeout_s", 14400))
        if timeout_s <= 0:
            raise ValueError("episode.timeout_s must be positive")
        required = cls._environment_names(value.get("required_env", []), "required_env")
        passthrough = cls._environment_names(
            value.get("env_passthrough", []), "env_passthrough"
        )
        if set(required) - set(passthrough):
            raise ValueError(
                "every episode.required_env name must also appear in env_passthrough"
            )
        cwd_value = value.get("cwd")
        cwd = resolve_repo_path(cwd_value, root=root) if cwd_value else root
        policy_seed = value.get("policy_seed")
        if policy_seed is not None and (type(policy_seed) is not int or policy_seed < 0):
            raise ValueError("episode.policy_seed must be a nonnegative integer")
        cache = value.get("heavy_cache_steps")
        if cache is not None and (type(cache) is not int or cache < 1):
            raise ValueError("episode.heavy_cache_steps must be a positive integer")
        trace_mode = value.get("trace_mode")
        if trace_mode not in (None, "full_rgb", "full_rgbd"):
            raise ValueError("episode.trace_mode must be full_rgb or full_rgbd when configured")
        if "env_cuda_devices" in value and not isinstance(value["env_cuda_devices"], list):
            raise ValueError("episode.env_cuda_devices must be a list of integer devices")
        devices = validate_env_cuda_devices(value.get("env_cuda_devices"))
        return cls(tuple(command), timeout_s, required, passthrough, cwd, policy_seed, cache, trace_mode, devices)

    @staticmethod
    def _environment_names(value: Any, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(
            isinstance(name, str)
            and name
            and "=" not in name
            and name.replace("_", "A").isalnum()
            for name in value
        ):
            raise ValueError(
                f"episode.{field} must contain environment variable names, not values"
            )
        return tuple(dict.fromkeys(value))

    @staticmethod
    def _require_flag(command: list[str], flag: str, expected: str | None) -> None:
        positions = [index for index, item in enumerate(command)
                     if item == flag or item.startswith(flag + "=")]
        if (len(positions) != 1 or
                command[positions[0]] == flag and positions[0] + 1 >= len(command)):
            raise ValueError(
                f"episode.command must contain exactly one {flag} "
                + (expected if expected is not None else "MODEL")
            )
        token = command[positions[0]]
        actual = command[positions[0] + 1] if token == flag else token[len(flag) + 1:]
        if expected is None:
            validate_model_name(actual, flag)
        elif actual != expected:
            raise ValueError(f"{flag} must remain {expected!r}, got {actual!r}")


# Schema 1 above remains available only to the old entry points until the W6/W8
# migration. Schema 2 never delegates stage, budget, or selection semantics to it.
_PHASES = {"parallel": (10, 4), "joint": (3, 1)}


def _object(value: Any, field: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    unknown = set(value) - required - (optional or set())
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(sorted(unknown))}")
    missing = required - set(value)
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(sorted(missing))}")
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if (type(value) not in (int, float) or not isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _identifier(value: Any, field: str) -> str:
    value = _text(value, field)
    if (value in {".", ".."} or any(character in value for character in "/\\")
            or any(character.isspace() or ord(character) < 32 for character in value)):
        raise ValueError(f"{field} must be a safe identifier")
    return value


def _choice(value: Any, field: str, choices: set[str], *, pending: bool = False) -> str | None:
    if pending and value is None:
        return None
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


@dataclass(frozen=True)
class EvolutionTask:
    """Task identity; scene and policy random seeds belong to the matrix."""

    id: str
    task_name: str = ""
    split: str = "target"
    task_index: int | None = None
    backend_suite: str | None = None
    seeds: tuple[int, ...] | None = None

    @classmethod
    def parse(cls, value: Any, field: str, *, backend_suite: str | None = None, task_config: str | None = None) -> "EvolutionTask":
        if task_config is not None:
            value = _object(value, field, {"id", "task_name", "seeds"})
            seeds = value["seeds"]
            if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
                raise ValueError(f"{field}.seeds must be a nonempty distinct list")
            return cls(id=_identifier(value["id"], f"{field}.id"),
                       task_name=_text(value["task_name"], f"{field}.task_name"), split=task_config,
                       seeds=tuple(_integer(s, f"{field}.seeds") for s in seeds))
        if backend_suite is not None:
            value = _object(value, field, {"id", "task_index"})
            return cls(id=_identifier(value["id"], f"{field}.id"),
                       task_index=_integer(value["task_index"], f"{field}.task_index"),
                       backend_suite=backend_suite)
        value = _object(value, field, {"id", "task_name", "split"})
        return cls(
            _identifier(value["id"], f"{field}.id"),
            _text(value["task_name"], f"{field}.task_name"),
            _choice(value["split"], f"{field}.split", {"target", "pretrain", "all"}),
        )

    def public_record(self):
        if self.seeds is not None:
            return {"id": self.id, "task_name": self.task_name, "seeds": list(self.seeds)}
        if self.backend_suite is not None:
            return {"id": self.id, "task_index": self.task_index}
        return {"id": self.id, "task_name": self.task_name, "split": self.split}

    def launch_selector(self):
        result = self.public_record()
        if self.seeds is not None:
            result["task_config"] = self.split
        if self.backend_suite is not None:
            result["backend_suite"] = self.backend_suite
        return result


@dataclass(frozen=True)
class EvolutionSuite:
    id: str
    label: str
    tasks: tuple[EvolutionTask, ...]
    backend_suite: str | None = None

    @classmethod
    def parse(cls, index_value: Any, index: int, *, robot: str = "robocasa") -> "EvolutionSuite":
        field = f"suites[{index}]"
        value = _object(index_value, field, {"id", "label", "tasks"} |
                        ({"backend_suite"} if robot == "libero" else set()))
        if not isinstance(value["tasks"], list) or not value["tasks"]:
            raise ValueError(f"{field}.tasks must be a non-empty list")
        suite_id = _identifier(value["id"], f"{field}.id")
        backend = value.get("backend_suite")
        if backend is not None:
            _choice(_text(backend, f"{field}.backend_suite"), f"{field}.backend_suite",
                    {f"libero_{suite_id}_{variant}" for variant in ("task", "swap", "lan", "object")})
        return cls(
            suite_id,
            _text(value["label"], f"{field}.label"),
            tuple(EvolutionTask.parse(task, f"{field}.tasks[{i}]", backend_suite=backend,
                                      task_config=suite_id if robot == "robotwin" else None)
                  for i, task in enumerate(value["tasks"])),
            backend,
        )

    def public_record(self):
        result = {"id": self.id, "label": self.label,
                  "tasks": [task.public_record() for task in self.tasks]}
        if self.backend_suite is not None:
            result["backend_suite"] = self.backend_suite
        return result


@dataclass(frozen=True)
class PhaseConfig:
    rounds: int
    repeats: int

    @classmethod
    def parse(cls, value: Any, field: str) -> "PhaseConfig":
        value = _object(value, field, {"rounds", "repeats"})
        # joint.rounds=0 skips merge/joint after parallel (day-1 / speed tracks).
        rounds_min = 0 if field == "joint" else 1
        return cls(_integer(value["rounds"], f"{field}.rounds", rounds_min),
                   _integer(value["repeats"], f"{field}.repeats", 1))


@dataclass(frozen=True)
class MatrixCell:
    """An ordinary evaluation-local ID, never a content digest."""

    cell_id: str
    task_id: str
    seed: int
    repeat: int


@dataclass(frozen=True)
class SelectionConfig:
    donor: str | None
    accept: str | None
    tie: str | None
    round_completion: str | None
    merge_conflicts: str | None
    h0_state_visibility: str | None
    time_optimization: dict | None = None
    perfect_rounds: int = 0

    @classmethod
    def parse(cls, value: Any) -> "SelectionConfig":
        choices = {
            "donor": {"selected"},
            "accept": {"strict_improvement", "success_then_time"},
            "tie": {"incumbent"},
            "round_completion": {"complete_candidate_evaluation_and_selection"},
            "merge_conflicts": {"joint_meta_before_rounds"},
            "h0_state_visibility": {"reference_fields"},
        }
        value = _object(value, "selection", set(choices), {"time_optimization", "perfect_rounds"})
        settings = value.get("time_optimization")
        if value.get("accept") == "success_then_time":
            settings = _object(settings, "selection.time_optimization",
                               {"success_rate_threshold", "minimum_relative_improvement", "control_frequency_hz"})
            for key in settings:
                number = settings[key]
                if type(number) not in (int, float) or not isfinite(number) or number <= 0:
                    raise ValueError(f"selection.time_optimization.{key} must be finite and positive")
            if settings["success_rate_threshold"] > 1 or settings["minimum_relative_improvement"] >= 1:
                raise ValueError("invalid time optimization threshold")
        elif settings is not None:
            raise ValueError("time_optimization requires accept=success_then_time")
        return cls(**{key: _choice(value[key], f"selection.{key}", allowed, pending=True)
                      for key, allowed in choices.items()}, time_optimization=settings,
                   perfect_rounds=_integer(value.get("perfect_rounds", 0), "selection.perfect_rounds"))

    def public_record(self):
        value = asdict(self)
        if self.time_optimization is None:
            value.pop("time_optimization")
        return value


@dataclass(frozen=True)
class EvolutionDevelopmentConfig:
    """One attempt budget, with explicit unresolved values instead of defaults."""

    timeout_s: float | None
    finish_reserve_s: float | None
    max_evaluation_requests: int | None
    max_no_candidate_recoveries: int | None
    deadline_action: str | None
    time_limit_enabled: bool = True
    live_trials: int = 2
    live_enabled: bool = True
    max_provider_failures: int = 3
    max_no_progress_turns: int = 3

    @classmethod
    def parse(cls, value: Any) -> "EvolutionDevelopmentConfig":
        value = _object(value, "development", {
            "timeout_s", "finish_reserve_s", "max_evaluation_requests",
            "max_no_candidate_recoveries", "deadline_action",
        }, {"time_limit_enabled", "live_trials", "live_enabled", "max_provider_failures", "max_no_progress_turns"})
        enabled = value.get("time_limit_enabled", True)
        if type(enabled) is not bool:
            raise ValueError("development.time_limit_enabled must be boolean")
        parsed = {"time_limit_enabled": enabled}
        live_enabled = value.get("live_enabled", True)
        if type(live_enabled) is not bool:
            raise ValueError("development.live_enabled must be boolean")
        if not live_enabled and value.get("live_trials", 2) != 0:
            raise ValueError("disabled live requires development.live_trials=0")
        parsed["live_enabled"] = live_enabled
        for key, default, minimum in (("live_trials", 2, 0), ("max_provider_failures", 3, 1), ("max_no_progress_turns", 3, 1)):
            parsed[key] = _integer(value.get(key, default), f"development.{key}", minimum)
        for key in ("timeout_s", "finish_reserve_s"):
            parsed[key] = (None if value[key] is None else
                           _number(value[key], f"development.{key}", positive=key == "timeout_s"))
        for key in ("max_evaluation_requests", "max_no_candidate_recoveries"):
            parsed[key] = (None if value[key] is None else
                           _integer(value[key], f"development.{key}"))
        parsed["deadline_action"] = _choice(
            value["deadline_action"], "development.deadline_action",
            {"evaluate_loadable_snapshot"}, pending=True,
        )
        if (parsed["timeout_s"] is not None and parsed["finish_reserve_s"] is not None
                and parsed["finish_reserve_s"] >= parsed["timeout_s"]):
            raise ValueError("development.finish_reserve_s must be smaller than timeout_s")
        return cls(**parsed)


@dataclass(frozen=True)
class EvolutionMetaConfig:
    """Provider settings only; the development owner supplies the deadline."""

    model: str | None
    reasoning_effort: str | None
    provider: dict[str, str] | None

    @classmethod
    def parse(cls, value: Any) -> "EvolutionMetaConfig":
        value = _object(value, "meta", {"model", "reasoning_effort", "provider"})
        model = None if value["model"] is None else validate_model_name(value["model"], "meta.model")
        effort = _choice(value["reasoning_effort"], "meta.reasoning_effort", {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
        }, pending=True)
        provider = value["provider"]
        if provider is not None:
            provider = _object(provider, "meta.provider", {"name", "base_url", "env_key", "wire_api"})
            provider = {key: _text(item, f"meta.provider.{key}") for key, item in provider.items()}
            EpisodeConfig._environment_names([provider["env_key"]], "meta.provider.env_key")
        return cls(model, effort, provider)


@dataclass(frozen=True)
class PoolConfig:
    """Explicit service placement and independent concurrency limits."""

    gpu_devices: tuple[int, ...]
    profile: dict[str, Any] | None

    @classmethod
    def parse(cls, value: Any) -> "PoolConfig":
        value = _object(value, "pool", {"gpu_devices", "profile"})
        if not isinstance(value["gpu_devices"], list):
            raise ValueError("pool.gpu_devices must be a list")
        devices = validate_env_cuda_devices(value["gpu_devices"])
        profile = value["profile"]
        if profile is not None:
            profile = _object(profile, "pool.profile", {
                "name", "max_episodes", "max_meta", "max_planner_requests",
                "env_devices", "vla", "sam",
            }, {"development_slots", "planner_rpm"})
            for key in ("development_slots", "planner_rpm"):
                if key in profile:
                    _integer(profile[key], f"pool.profile.{key}", 1)
            if profile.get("development_slots", 0) >= profile["max_episodes"]:
                raise ValueError("development_slots must leave formal evaluation capacity")
            _text(profile["name"], "pool.profile.name")
            for field in ("max_episodes", "max_meta", "max_planner_requests"):
                _integer(profile[field], f"pool.profile.{field}", 1)
            env_devices = validate_env_cuda_devices(profile["env_devices"])
            if not env_devices or not set(env_devices) <= set(devices):
                raise ValueError("pool.profile.env_devices must be a nonempty subset of gpu_devices")
            for kind in ("vla", "sam"):
                replicas = profile[kind]
                if not isinstance(replicas, list) or (not replicas and kind == "vla"):
                    raise ValueError(f"pool.profile.{kind} needs explicit replicas")
                endpoints = set()
                for replica in replicas:
                    required = {"endpoint", "capacity"} | ({"session_protocol"} if kind == "vla" else set())
                    _object(replica, f"pool.profile.{kind} replica", required, {"launch", "gpu_device"})
                    endpoint = _text(replica["endpoint"], "replica.endpoint")
                    if endpoint in endpoints:
                        raise ValueError("duplicate replica endpoint")
                    endpoints.add(endpoint)
                    _integer(replica["capacity"], "replica.capacity", 1)
                    if kind == "vla":
                        _choice(replica["session_protocol"], "session_protocol", {"registered", "legacy", "pi05", "lingbot"})
                    if "gpu_device" in replica and replica["gpu_device"] not in devices:
                        raise ValueError("service gpu_device must belong to pool.gpu_devices")
                    if "launch" in replica:
                        launch = _object(replica["launch"], "service launch", {"command", "timeout_s", "env_passthrough"})
                        if (not isinstance(launch["command"], list) or not launch["command"]
                                or any(not isinstance(x, str) or not x for x in launch["command"])):
                            raise ValueError("service launch.command must be a nonempty argv list")
                        _number(launch["timeout_s"], "service launch.timeout_s", positive=True)
                        if (not isinstance(launch["env_passthrough"], list)
                                or any(not isinstance(x, str) or not x for x in launch["env_passthrough"])):
                            raise ValueError("service launch.env_passthrough must list variable names")
        return cls(devices, profile)


def _evolution_episode(value: Any, root: Path) -> EpisodeConfig:
    value = _object(value, "episode", {
        "command", "timeout_s", "required_env", "env_passthrough", "policy_seed", "trace_mode",
    }, {"cwd", "heavy_cache_steps"})
    _number(value["timeout_s"], "episode.timeout_s", positive=True)
    _integer(value["policy_seed"], "episode.policy_seed")
    if value["trace_mode"] not in {"full_rgbd", "full_rgb"}:
        raise ValueError("episode.trace_mode must be full_rgb or full_rgbd")
    # Flags injected per episode, or owned by the shared pool, must not hide in
    # --flag=value spelling. Schema 1's permissive command parser is unchanged.
    reserved = {
        "--task-name", "--split", "--seed", "--harness-dir", "--result-dir", "--output-dir",
        "--env-cuda-device", "--env-endpoint", "--vla-endpoint", "--sam3-endpoint", "--policy-seed",
        "--harness-commit", "--config-ref",
        "--suite", "--task", "--cuda-device", "--libero-type",
        "--api-key", "--token", "--password", "--secret",
    }
    command = value["command"]
    if isinstance(command, list):
        collision = {item.split("=", 1)[0] for item in command if isinstance(item, str)} & reserved
        if collision:
            raise ValueError("episode.command contains runtime-owned or credential flags: " + ", ".join(sorted(collision)))
    return EpisodeConfig.parse(value, root=root, resolve_interpreters=False)


@dataclass(frozen=True)
class EvolutionConfig:
    """The single schema-2 stage plan, shared by formal runs and small probes.

    Loading is inspection-only: no SDK, model service, GPU, or environment
    interpreter is needed. Null decisions remain visible and prevent execution;
    they are not recommendations silently promoted to experiment settings.
    """

    source: Path | None
    repo_root: Path
    mode: str
    initial_harness: Path
    capability_catalog: Path
    suites: tuple[EvolutionSuite, ...]
    seeds: tuple[int, ...]
    parallel: PhaseConfig
    joint: PhaseConfig
    selection: SelectionConfig
    development: EvolutionDevelopmentConfig
    evaluation_retries: int | None
    episode: EpisodeConfig
    meta: EvolutionMetaConfig
    pool: PoolConfig
    robot: str = "robocasa"
    benchmark: str = "robocasa"
    scheduling: str = "staged"

    @classmethod
    def load(cls, path: str | Path, *, root: Path | None = None) -> "EvolutionConfig":
        source = Path(path).expanduser().resolve()
        return cls.parse(read_json(source), root=root, source=source)

    @classmethod
    def parse(
        cls, value: Any, *, root: Path | None = None, source: Path | None = None,
    ) -> "EvolutionConfig":
        value = _object(value, "evolution", {
            "schema_version", "mode", "initial_harness", "capability_catalog", "suites", "seeds",
            "parallel", "joint", "selection", "development", "evaluation", "episode", "meta", "pool",
        }, {"robot", "benchmark", "scheduling"})
        if type(value["schema_version"]) is not int or value["schema_version"] != 2:
            raise ValueError("evolution schema_version must be 2")
        repository = (root or Path(__file__).resolve().parents[1]).resolve()
        mode = _choice(value["mode"], "mode", {"formal", "probe"})
        robot = _choice(value.get("robot", "robocasa"), "robot", {"robocasa", "libero", "robotwin"})
        scheduling = _choice(value.get("scheduling", "staged"), "scheduling", {"staged", "pipeline"})
        expected_benchmark = {"libero": "libero-pro", "robocasa": "robocasa", "robotwin": "robotwin"}[robot]
        benchmark = value.get("benchmark", expected_benchmark)
        if benchmark != expected_benchmark:
            raise ValueError(f"{robot} requires benchmark {expected_benchmark!r}")
        from rpent.robots import get_robot_spec
        suite_sizes = get_robot_spec(robot).evolution_suites
        if not suite_sizes:
            raise ValueError(f"{robot} does not provide an evolution task protocol")
        if not isinstance(value["suites"], list) or not value["suites"]:
            raise ValueError("suites must be a non-empty list")
        suites = tuple(EvolutionSuite.parse(item, i, robot=robot) for i, item in enumerate(value["suites"]))
        suite_ids = [suite.id for suite in suites]
        expected_ids = [name for name, _ in suite_sizes]
        if suite_ids != [name for name in expected_ids if name in suite_ids]:
            raise ValueError(f"suites must be unique and ordered as {expected_ids}")
        if mode == "formal" and suite_ids != expected_ids:
            raise ValueError(f"formal runs require all {len(expected_ids)} {robot} suites")
        for suite in suites:
            count = dict(suite_sizes)[suite.id]
            if len(suite.tasks) > count or (mode == "formal" and len(suite.tasks) != count):
                raise ValueError(f"{suite.id} requires {'exactly' if mode == 'formal' else 'at most'} {count} tasks")
            if robot == "libero":
                if suite.backend_suite not in {f"libero_{suite.id}_task", f"libero_{suite.id}_swap"}:
                    raise ValueError("LIBERO-PRO requires a task or swap backend matching its family")
                indices = [task.task_index for task in suite.tasks]
                if len(set(indices)) != len(indices) or any(i >= count for i in indices):
                    raise ValueError(f"{suite.id} task indices must be distinct and in 0..{count - 1}")
                if mode == "formal" and indices != list(range(count)):
                    raise ValueError(f"formal {suite.id} must enumerate task indices 0..{count - 1} in order")
        if robot == "libero" and mode == "formal" and len({suite.backend_suite.rsplit("_", 1)[1] for suite in suites}) != 1:
            raise ValueError("formal LIBERO-PRO requires one consistent perturbation across suites")
        tasks = [task for suite in suites for task in suite.tasks]
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("task ids must be unique across suites")
        selectors = [(t.backend_suite, t.task_index) if t.backend_suite is not None
                     else (t.task_name, t.split) for t in tasks]
        if len(set(selectors)) != len(tasks):
            raise ValueError("task selectors must be unique across suites")
        seeds_value = value["seeds"]
        if not isinstance(seeds_value, list) or not seeds_value:
            raise ValueError("seeds must be a non-empty list")
        seeds = tuple(_integer(seed, f"seeds[{i}]") for i, seed in enumerate(seeds_value))
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        if robot != "robotwin" and mode == "formal" and seeds != tuple(range(1000, 1008 if scheduling == "pipeline" else 1016)):
            raise ValueError("formal seeds must match the scheduling protocol (pipeline: 1000..1007; staged: 1000..1015)")
        if robot != "robotwin" and mode == "probe" and not set(seeds) <= set(range(1000, 1016)):
            raise ValueError("probe seeds must be a subset of the formal seeds")
        phases = {name: PhaseConfig.parse(value[name], name) for name in _PHASES}
        for name, (rounds, repeats) in _PHASES.items():
            if scheduling == "pipeline" and name == "parallel":
                repeats = 2
            phase = phases[name]
            if ((mode == "formal" and (phase.rounds, phase.repeats) != (rounds, repeats)
                 and not (scheduling == "pipeline" and name == "joint" and phase.rounds == 0))
                    or phase.rounds > rounds or phase.repeats > repeats):
                raise ValueError(f"{name} requires {'exactly' if mode == 'formal' else 'at most'} {rounds} rounds and {repeats} repeats")
        evaluation = _object(value["evaluation"], "evaluation", {"infrastructure_retries"})
        retries = evaluation["infrastructure_retries"]
        if retries is not None:
            retries = _integer(retries, "evaluation.infrastructure_retries")
        pool = PoolConfig.parse(value["pool"])
        if mode == "formal" and pool.gpu_devices != tuple(range(8)):
            raise ValueError("formal pool.gpu_devices must explicitly list GPU 0 through 7")
        episode = _evolution_episode(value["episode"], repository)
        EpisodeConfig._require_flag(list(episode.command), "--robot", robot)
        if pool.profile:
            for replica in pool.profile["vla"]:
                if robot == "robotwin" and replica["session_protocol"] != "lingbot":
                    raise ValueError("RoboTwin requires the LingBot service protocol")
                if (replica["session_protocol"] == "pi05") != (robot == "libero"):
                    raise ValueError("VLA session protocol must match the selected robot")
        return cls(
            source=source, repo_root=repository, mode=mode,
            initial_harness=resolve_repo_path(_text(value["initial_harness"], "initial_harness"), root=repository),
            capability_catalog=resolve_repo_path(_text(value["capability_catalog"], "capability_catalog"), root=repository),
            suites=suites, seeds=seeds, parallel=phases["parallel"], joint=phases["joint"],
            selection=SelectionConfig.parse(value["selection"]),
            development=EvolutionDevelopmentConfig.parse(value["development"]),
            evaluation_retries=retries, episode=episode,
            meta=EvolutionMetaConfig.parse(value["meta"]), pool=pool,
            robot=robot, benchmark=benchmark, scheduling=scheduling,
        )

    def suite(self, suite_id: str) -> EvolutionSuite:
        for suite in self.suites:
            if suite.id == suite_id:
                return suite
        raise ValueError(f"unknown suite: {suite_id!r}")

    def phase(self, phase: str) -> PhaseConfig:
        _choice(phase, "phase", set(_PHASES))
        return getattr(self, phase)

    def matrix(self, suite_id: str, phase: str, *, task_id: str | None = None) -> tuple[MatrixCell, ...]:
        """Expand the current suite, or one parallel scope, without allocating resources."""
        suite, schedule = self.suite(suite_id), self.phase(phase)
        tasks = suite.tasks
        if task_id is not None:
            if phase != "parallel":
                raise ValueError("joint matrix always contains the entire current suite")
            tasks = tuple(task for task in tasks if task.id == task_id)
            if not tasks:
                raise ValueError(f"task {task_id!r} does not belong to suite {suite_id!r}")
        cells = ((task.id, seed, repeat) for task in tasks
                 for seed in (task.seeds if task.seeds is not None else self.seeds)
                 for repeat in range(schedule.repeats))
        return tuple(MatrixCell(f"c{index:05d}", task_id, seed, repeat)
                     for index, (task_id, seed, repeat) in enumerate(cells, 1))

    def candidate_cost(self) -> dict[str, Any]:
        """Nominal candidate episodes only; never label these as total experiment cost."""
        suites = []
        for suite in self.suites:
            record = {"suite_id": suite.id, "tasks": len(suite.tasks)}
            total = 0
            for phase in _PHASES:
                schedule = self.phase(phase)
                per_round = sum(len(task.seeds if task.seeds is not None else self.seeds)
                                for task in suite.tasks) * schedule.repeats
                record[phase] = {"rounds": schedule.rounds, "repeats": schedule.repeats,
                                 "episodes_per_round": per_round,
                                 "candidate_episodes": per_round * schedule.rounds}
                total += per_round * schedule.rounds
            record["candidate_episodes"] = total
            suites.append(record)
        return {"suites": suites,
                "total_candidate_episodes": sum(suite["candidate_episodes"] for suite in suites),
                "excluded_costs": ["baseline", "development", "merge", "infrastructure_retry"]}

    def unresolved_decisions(self) -> tuple[str, ...]:
        pending = []
        for group in ("selection", "development"):
            pending.extend(f"{group}.{key}" for key, value in asdict(getattr(self, group)).items()
                           if value is None and key != "time_optimization"
                           and not (group == "development" and not self.development.time_limit_enabled
                                    and key in {"timeout_s", "finish_reserve_s", "deadline_action"}))
        if self.evaluation_retries is None:
            pending.append("evaluation.infrastructure_retries")
        if self.meta.model is None:
            pending.append("meta.model")
        if self.pool.profile is None:
            pending.append("pool.profile")
        return tuple(pending)

    def require_ready(self) -> None:
        pending = self.unresolved_decisions()
        if pending:
            raise ValueError("evolution configuration has unresolved settings: " + ", ".join(pending))

    def public_record(self) -> dict[str, Any]:
        """Serializable conditions for config.json; keep interpreter sentinels portable."""
        def path_value(path: Path) -> str:
            try:
                return str(path.relative_to(self.repo_root))
            except ValueError:
                return str(path)

        episode = {
            "command": list(self.episode.command), "timeout_s": self.episode.timeout_s,
            "required_env": list(self.episode.required_env), "env_passthrough": list(self.episode.env_passthrough),
            "policy_seed": self.episode.policy_seed, "trace_mode": self.episode.trace_mode,
            "cwd": path_value(self.episode.cwd),
        }
        if self.episode.heavy_cache_steps is not None:
            episode["heavy_cache_steps"] = self.episode.heavy_cache_steps
        development = asdict(self.development)
        # Preserve immutable legacy configs byte-for-byte in semantic content.
        if development["live_enabled"]:
            development.pop("live_enabled")
        return {
            "schema_version": 2, "mode": self.mode, "scheduling": self.scheduling,
            "initial_harness": path_value(self.initial_harness),
            "capability_catalog": path_value(self.capability_catalog),
            **({"robot": self.robot, "benchmark": self.benchmark} if self.robot != "robocasa" else {}),
            "suites": [suite.public_record() for suite in self.suites],
            "seeds": list(self.seeds), "parallel": asdict(self.parallel), "joint": asdict(self.joint),
            "selection": self.selection.public_record(), "development": development,
            "evaluation": {"infrastructure_retries": self.evaluation_retries},
            "episode": episode, "meta": asdict(self.meta),
            "pool": {"gpu_devices": list(self.pool.gpu_devices), "profile": self.pool.profile},
        }
