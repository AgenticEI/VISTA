# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical agent main CLI entrypoint."""

# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --robot libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.planner`, `rpent.robots`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.dashboard.events import (
    NullDashboardEventSink,
    RunStartedEvent,
)
from rpent.harness import (
    BoundHarness,
    HarnessError,
    HarnessPackage,
    bind_harness,
    load_harness,
)
from rpent.planner.base import REASONING_EFFORTS, build_planner
from rpent.result import EpisodeOutcome, write_episode_result, write_invalid_result
from rpent.robots import enumerate_robots, get_robot_spec, get_toolkit
from rpent.robots.robot_spec import RobotSpec
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

logger = get_logger("agent")


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {
            **{k: v for k, v in m.items() if k != "content"},
            "content": _strip_images(m.get("content")),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    known_robots = enumerate_robots()
    known_robots_text = ", ".join(known_robots) if known_robots else "none"
    ap = argparse.ArgumentParser(
        description="RPent: Agentic Infrastructure for the Physical World",
        add_help=False,
    )

    ap.add_argument(
        "--robot",
        dest="robot_name",
        required=False,
        choices=known_robots,
        help=f"Robot backend. Known robots: {known_robots_text}.",
    )
    ap.add_argument(
        "--env",
        dest="env_name",
        required=False,
        choices=known_robots,
        help="Deprecated alias for --robot; use --robot instead.",
    )

    # models
    ap.add_argument(
        "--planner",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' planner, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2). For claude_code/codex this "
        "overrides the backend default model.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the api planner (unset preserves the "
        "provider/model default).",
    )
    ap.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="none",
        help="Planner reasoning effort for api, claude_code, and "
        "codex. Higher effort may improve task success rate "
        "but increases runtime. Defaults to none.",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Never send image bytes to the model (api planner only). "
        "Use for text-only models that reject image input "
        "(e.g. 400 \"message type 'image_url' is not supported\"); "
        "read_image then returns the image name instead, with a notice.",
    )
    ap.add_argument(
        "--planner-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for api/claude_code/codex planner runs. "
        "Terminal interactive API/Claude sessions are exempt. "
        "Defaults to CODEX_TIMEOUT_S (codex only), "
        "CELL_TIMEOUT_S, or 1200.",
    )
    ap.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=None,
        help="Budget passed to claude -p --max-budget-usd. "
        "Defaults to MAX_BUDGET_USD env or 10.",
    )

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--harness-dir",
        default=None,
        help="File-backed Harness directory for a single non-interactive episode.",
    )
    ap.add_argument("--harness-commit", default=None, help="Frozen Harness Git commit used by this episode.")
    ap.add_argument("--config-ref", default=None, help="Run-relative frozen experiment configuration reference.")
    ap.add_argument(
        "--result-dir",
        default=None,
        help="Episode Result directory below runs/<run-id>/.",
    )
    ap.add_argument(
        "--memory-profile",
        choices=["hf", "local"],
        default=None,
        help="Memory profile (default: hf for evaluation, local for exploration).",
    )
    ap.add_argument(
        "--memory-dir",
        default=None,
        help="Local memory root (environment default when omitted).",
    )
    ap.add_argument(
        "--explore",
        action="store_true",
        help="Enable exploration and memory distillation.",
    )
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start a local dashboard server for this single run.",
    )
    ap.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host. Defaults to 127.0.0.1.",
    )
    ap.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Dashboard port. 0 asks the OS for a free port.",
    )
    ap.add_argument(
        "--dashboard-language",
        choices=["en", "zh-cn"],
        default="en",
        help="Dashboard UI language. 'zh-cn' serves the Chinese "
        "translation; defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )
    ap.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: opens an interactive cli session.",
    )

    return ap


def _resolve_result_dir(raw_path: str | Path) -> Path:
    """Resolve a Result directory and keep it inside this repository's run tree."""
    repository = get_repo_root().resolve()
    requested = Path(raw_path).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    requested = Path(os.path.abspath(requested))
    runs_root = repository / "runs"
    if runs_root.is_symlink():
        raise ValueError(
            "this repository's runs directory must not be a symlink"
        )
    if runs_root.exists() and not runs_root.is_dir():
        raise ValueError("this repository's runs path must be a directory")
    from rpent.utils.run_paths import migrated_result_dir
    if approved := migrated_result_dir(requested, repository):
        return approved
    try:
        relative = requested.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(
            "--result-dir must be below this repository's runs/<run-id>/"
        ) from exc
    if len(relative.parts) < 2 or relative.parts[0] in {"", ".", ".."}:
        raise ValueError(
            "--result-dir must name a run and an episode below runs/"
        )
    current = runs_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("--result-dir must not pass through a symlink")
    result_dir = requested.resolve()
    try:
        result_dir.relative_to(runs_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "--result-dir resolves outside this repository's runs/"
        ) from exc
    return result_dir


def _validate_harness_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    robot_spec: RobotSpec,
) -> Path | None:
    """Validate the deliberately small, single-episode Harness runner surface."""
    harness_requested = args.harness_dir is not None or args.result_dir is not None
    if not harness_requested:
        return None
    if args.harness_dir is None or args.result_dir is None:
        parser.error("--harness-dir and --result-dir must be provided together")
    if args.dashboard or args.interactive or args.explore:
        parser.error(
            "Harness mode supports one non-interactive, non-dashboard, "
            "non-exploration episode"
        )
    if args.planner != "api":
        parser.error(
            "Harness mode currently supports only the api planner so the restricted "
            "tool surface remains the complete robot authority boundary"
        )
    if robot_spec.make_harness_adapter is None:
        parser.error(f"robot {args.robot_name!r} does not support Harness mode")
    try:
        result_dir = _resolve_result_dir(args.result_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir != result_dir:
            parser.error(
                "Harness mode requires --output-dir to equal --result-dir so all "
                "episode artifacts stay in the canonical Result directory"
            )
    args.output_dir = str(result_dir)
    return result_dir


def _bind_active_harness(
    package: HarnessPackage,
    *,
    toolkit: Toolkit,
    robot_spec: RobotSpec,
    episode_dir: Path | None = None,
) -> tuple[BoundHarness, Any]:
    """Create the robot adapter, enforce its native allowlist, and bind Harness."""
    factory = robot_spec.make_harness_adapter
    if factory is None:
        raise HarnessError(f"robot {robot_spec.name!r} has no Harness adapter")
    adapter = factory(toolkit)
    allowed_provider = getattr(adapter, "allowed_native_tools", None)
    if not callable(allowed_provider):
        raise HarnessError("Harness adapter must define allowed_native_tools()")
    allowed_native = frozenset(allowed_provider())
    episode_active = getattr(adapter, "episode_active", None)
    if not callable(episode_active):
        raise HarnessError("Harness adapter must define episode_active()")
    unauthorized = sorted(set(package.native_tools).difference(allowed_native))
    if unauthorized:
        raise HarnessError(
            f"Harness requests native tools outside the robot allowlist: {unauthorized}"
        )
    return (
        bind_harness(
            package,
            toolkit=toolkit,
            capabilities=adapter.capabilities(),
            allowed_native_tools=allowed_native,
            episode_active=episode_active,
            allow_image_reader="read_image" in package.planner_tools,
            episode_dir=episode_dir,
        ),
        adapter,
    )


def _harness_system_prompt(system_prompt: str, instructions: str) -> str:
    """Append schema-1 skill instructions after the legacy robot contract."""
    return (
        f"{system_prompt.rstrip()}\n\n"
        "# Active Harness\n\n"
        "The following instructions define the active strategy and the only "
        "Harness-provided abstractions for this episode. Follow them while "
        "obeying the fixed safety and tool contracts above.\n\n"
        f"{instructions.strip()}\n"
    )


def _episode_prompt_bundle(robot_spec: RobotSpec, harness_package: HarnessPackage | None):
    """Use the ordinary package's declared prompts for schema-2 episodes."""
    if harness_package is not None and harness_package.identity.schema_version >= 2:
        from rpent.robots.prompt_bundle import PromptBundle

        return PromptBundle(
            system=lambda variables: harness_package.system_prompt,
            user=lambda variables: harness_package.user_prompt,
        )
    if harness_package is not None and robot_spec.harness_prompts is not None:
        return robot_spec.harness_prompts
    return robot_spec.prompts


def _episode_system_prompt(
    system_prompt: str,
    instructions: str,
    *,
    robot_spec: RobotSpec,
    package: HarnessPackage | None = None,
) -> str:
    """Schema-2 prompts are wholly package-owned; schema 1 is a legacy format."""
    if package is not None and package.identity.schema_version >= 2:
        return f"{system_prompt.rstrip()}\n\n{instructions.strip()}\n" if instructions.strip() else system_prompt
    return _harness_system_prompt(system_prompt, instructions)


def _harness_manifest(package: HarnessPackage | None, path: str | Path) -> dict:
    if package is None:
        return {"id": "<unloaded>", "path": str(Path(path))}
    identity = package.identity
    return {
        "id": identity.harness_id,
        "schema_version": identity.schema_version,
        "commit": identity.commit,
    }


def _task_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Return the small, non-secret task identity available before parse_config."""
    task: dict[str, Any] = {"robot": args.robot_name}
    for name in ("task_name", "suite", "task", "task_config", "split", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            task[name] = value
    return task


def _publish_harness_result(
    result_dir: Path,
    *,
    args: argparse.Namespace,
    task_desc: dict[str, Any],
    package: HarnessPackage,
    outcome: EpisodeOutcome | None,
    agent_error: str | None,
    finish_result: dict | None,
    messages: list[dict],
    stats: dict,
    elapsed: float,
    capability_calls: list[dict],
    state: Any,
) -> Path:
    """Publish one authoritative Result without trusting the agent's claim."""
    if outcome is None:
        final_outcome = EpisodeOutcome(
            validity="invalid",
            official_success=None,
            terminated=False,
            truncated=False,
            env_steps=None,
            terminal_reason="runner_error",
            error=agent_error or "environment outcome was not captured",
        )
    elif agent_error:
        final_outcome = EpisodeOutcome(
            validity="invalid",
            official_success=None,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            env_steps=outcome.env_steps,
            terminal_reason="runner_error",
            error=agent_error,
        )
    else:
        final_outcome = outcome

    actors = stats.get("actors")
    if actors is None:
        actors = ["planner", *(["meta"] if stats.get("manual_decisions", 0) else []),
                  *(["harness_program"] if stats.get("harness_program_decisions", 0) else [])]
    controller = "meta" if set(actors) == {"meta"} else "mixed" if "meta" in actors else "planner"
    deployment_time = {}
    if args.robot_name == "libero":
        from rpent.deployment_time import estimate
        deployment_time = {"deployment_time": estimate(
            result_dir, os.environ.get("RPENT_TIMING_DIR"), env_steps=final_outcome.env_steps,
            expected_planner_calls=stats.get("requests"),
        )}
    return write_episode_result(
        result_dir,
        task={"robot": args.robot_name, **task_desc},
        harness=_harness_manifest(package, args.harness_dir),
        agent={
            "planner": args.planner,
            "model": args.model,
            "finish_claim": finish_result,
            "error": agent_error,
        },
        outcome=final_outcome,
        metrics={
            "elapsed_s": round(elapsed, 3),
            "model_requests": stats.get("requests"),
            "agent_turns": stats.get("turns_used", stats.get("turns")),
            "tool_calls": stats.get("tool_calls"),
            "input_tokens": stats.get("total_input_tokens"),
            "output_tokens": stats.get("total_output_tokens"),
            "cache_read_tokens": stats.get("cache_read_tokens"),
            "cache_write_tokens": stats.get("cache_write_tokens"),
            "env_steps": final_outcome.env_steps,
            **deployment_time,
        },
        experiment={
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "no_images": args.no_images,
        },
        messages=_serialize_messages(messages),
        capability_calls=capability_calls,
        state=state,
        config_ref=getattr(args, "config_ref", None),
        controller=controller,
        agent_claimed_success=(finish_result.get("status") == "success") if finish_result else None,
        actors=actors,
    )


def _handoff_message(output_dir, session_number: int, session_max: int) -> str:
    """Build the opening message for a continuation session."""
    attempts_dir = Path(output_dir) / "attempts"
    prior = (
        sorted(p.name for p in attempts_dir.glob("attempt_*_failed.json"))
        if attempts_dir.is_dir()
        else []
    )
    return (
        f"You are agent {session_number} of up to {session_max} on this cell. "
        f"{len(prior)} attempt(s) by earlier agents are archived in "
        f"{attempts_dir}/ ({', '.join(prior) if prior else 'none yet'}), and their "
        "working notes are in the memory inbox under wip/.\n\n"
        "Read every archive and the working notes before acting. Do not repeat "
        "failed approaches. A fresh toolkit has already restored a clean scene; "
        "inspect it before acting."
    )


def _start_continuation_session(
    args,
    *,
    output_dir,
    recipe_tag,
    dashboard_events,
    prompt_bundle,
    prompt_vars,
    session_number: int,
    session_max: int,
):
    """Build a fresh planner and prompts for an exploration handoff."""
    logger.info("=== handing off to agent %d/%d ===", session_number, session_max)
    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        robot_name=args.robot_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        planner_timeout_s=args.planner_timeout_s,
        reasoning_effort=args.reasoning_effort,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard_events=dashboard_events,
        no_images=args.no_images,
    )
    system_prompt = prompt_bundle.render(
        "system",
        variables={
            **prompt_vars,
            "session_number": session_number,
            "session_max": session_max,
        },
    )
    session_message = _handoff_message(output_dir, session_number, session_max)
    return planner, system_prompt, session_message


def main() -> int:
    parser = _build_argparser()
    # Two-phase argparse: first grab --robot / --env / --dashboard so we know
    # which robot's flags to add and whether to make its required flags optional.
    early, _ = parser.parse_known_args()

    # --env is a deprecated alias for --robot; resolve it before loading the spec.
    if early.env_name is not None:
        if early.robot_name is not None:
            parser.error("--robot and --env are aliases; provide only one of them")
        logger.warning("--env is deprecated and will be removed; use --robot instead")
        early.robot_name = early.env_name
    if early.robot_name is None:
        if "-h" not in sys.argv and "--help" not in sys.argv:
            parser.error("--robot is required")
    else:
        robot_spec = get_robot_spec(early.robot_name)
        robot_spec.add_cli_args(parser, use_dashboard=early.dashboard)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    args = parser.parse_args()
    args.robot_name = early.robot_name
    if args.temperature is not None and args.planner != "api":
        parser.error("--temperature is supported only by the api planner")
    if args.dashboard and args.interactive:
        parser.error("--dashboard and --interactive cannot be used together")
    if args.explore and args.robot_name != "libero":
        parser.error("--explore is currently supported only for LIBERO")
    if args.explore and args.memory_profile == "hf":
        parser.error("--explore cannot be used with --memory-profile hf")
    if args.explore and getattr(args, "explore_sessions", 1) <= 0:
        parser.error("--explore-sessions must be greater than 0")
    args.memory_profile = args.memory_profile or (
        "local" if args.explore or args.harness_dir else "hf"
    )
    if args.memory_profile == "hf" and args.memory_dir is not None:
        parser.error("--memory-dir requires --memory-profile local or --explore")
    result_dir = _validate_harness_mode(parser, args, robot_spec)
    harness_package: HarnessPackage | None = None
    if result_dir is not None:
        try:
            harness_package = load_harness(args.harness_dir, commit=getattr(args, "harness_commit", None))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("failed to load Harness: %s", error)
            write_invalid_result(
                result_dir,
                task=_task_arguments(args),
                harness=_harness_manifest(None, args.harness_dir),
                agent={"planner": args.planner, "model": args.model},
                reason="harness_load_error",
                error=error,
            )
            return 1
    if args.dashboard:
        from rpent.cli.dashboard import run_dashboard_session

        return run_dashboard_session(args, robot_spec, parser=parser)

    try:
        run_config = robot_spec.parse_config(args)
    except Exception as exc:
        if result_dir is not None and harness_package is not None:
            write_invalid_result(
                result_dir,
                task=_task_arguments(args),
                harness=_harness_manifest(harness_package, args.harness_dir),
                agent={"planner": args.planner, "model": args.model},
                reason="task_configuration_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            return 1
        raise
    recipe_tag = run_config.recipe_tag
    if result_dir is not None:
        run_config = replace(run_config, output_dir=result_dir)
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    task_desc = run_config.task_desc

    robot_name = args.robot_name

    def _write_pre_episode_failure(reason: str, exc: Exception) -> None:
        if result_dir is None or harness_package is None:
            return
        write_invalid_result(
            result_dir,
            task={"robot": args.robot_name, **task_desc},
            harness=_harness_manifest(harness_package, args.harness_dir),
            agent={"planner": args.planner, "model": args.model},
            reason=reason,
            error=f"{type(exc).__name__}: {exc}",
            experiment={
                "max_turns": args.max_turns,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "reasoning_effort": args.reasoning_effort,
            },
        )

    # mkdir + logging wiring (robot-side already picked the path).
    try:
        output_dir = init_output_dir(output_dir, verbose=args.verbose)
    except Exception as exc:
        _write_pre_episode_failure("output_initialization_error", exc)
        if result_dir is not None:
            return 1
        raise
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    # Preserve the original HF-backed evaluation behavior by default.
    memory_profile = getattr(args, "memory_profile", "hf")
    try:
        if harness_package is None and not getattr(args, "explore", False) and memory_profile == "hf":
            ensure_resources(robot_spec)
        else:
            logger.info("resources: using local %s memory profile", memory_profile)
    except Exception as exc:
        _write_pre_episode_failure("resource_initialization_error", exc)
        if result_dir is not None:
            return 1
        raise

    dashboard_events = NullDashboardEventSink()

    try:
        planner = build_planner(
            args.planner,
            output_dir=output_dir,
            recipe_tag=recipe_tag,
            robot_name=robot_name,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            planner_timeout_s=args.planner_timeout_s,
            reasoning_effort=args.reasoning_effort,
            claude_code_max_budget_usd=args.claude_code_max_budget_usd,
            dashboard_events=dashboard_events,
            no_images=args.no_images,
        )
        prompt_bundle = _episode_prompt_bundle(robot_spec, harness_package)
        prompt_vars = {**prompt_vars, "output_dir": output_dir}
        if harness_package is not None and harness_package.identity.schema_version >= 2:
            prompt_vars.update(
                memory_dir=str(harness_package.memory_path or ""),
                notes_dir=str(output_dir / "notes"),
                harness_dir=str(harness_package.root),
            )
            if harness_package.episode_notes:
                (output_dir / "notes").mkdir(parents=True, exist_ok=True)
        system_prompt = prompt_bundle.render(
            "system",
            variables=prompt_vars,
        )
        user_msg = prompt_bundle.render(
            "user",
            variables=prompt_vars,
        )
    except Exception as exc:
        _write_pre_episode_failure("planner_initialization_error", exc)
        if result_dir is not None:
            return 1
        raise

    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    if args.interactive:
        input_queue = queue.Queue()
        # Pre-fill the first prompt with the rendered default task (editable
        # preset);
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )
        # Resolve the opening prompt on a background thread so the user can type
        # it while the (slow) env/VLA servers boot below.
        await_first_prompt = start_first_prompt_resolver(input_queue)

    # --- initialise robot runtime --------------------------------------------
    runtime_started = time.perf_counter()
    try:
        daemons, primitives_kwargs = robot_spec.init_runtime(
            args,
            output_dir,
            dashboard_events,
            None,
        )
        from rpent.utils.execution import timing
        timing("environment_startup", runtime_started)
    except Exception as exc:
        _write_pre_episode_failure("robot_runtime_initialization_error", exc)
        if result_dir is not None:
            return 1
        raise

    # --- agent loop --------------------------------------------------------
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    episode_outcome: EpisodeOutcome | None = None
    capability_calls: list[dict] = []
    result_state: Any = None
    first_user_msg: str | None = user_msg
    if await_first_prompt is not None:
        # Block until the opening prompt typed during startup is ready.
        first_user_msg = await_first_prompt()
        if first_user_msg is None:
            logger.info("no task entered; ending session before start.")
    # Exploration may hand off between independent planner contexts.
    sessions = max(1, int(getattr(args, "explore_sessions", 1) or 1))
    if not getattr(args, "explore", False):
        sessions = 1
    recipe_path = ""
    try:
        if first_user_msg is not None:
            dashboard_events.emit(RunStartedEvent())
        session_msg = first_user_msg
        for session_number in range(1, sessions + 1):
            if session_msg is None:
                break
            if session_number > 1:
                planner, system_prompt, session_msg = _start_continuation_session(
                    args,
                    output_dir=output_dir,
                    recipe_tag=recipe_tag,
                    dashboard_events=dashboard_events,
                    prompt_bundle=prompt_bundle,
                    prompt_vars=prompt_vars,
                    session_number=session_number,
                    session_max=sessions,
                )
            state_output_dir = output_dir
            if getattr(args, "explore", False):
                state_output_dir = (
                    output_dir / "sessions" / f"session_{session_number:03d}"
                )
            if robot_name == "libero":
                package_memory = None
                if harness_package is not None and harness_package.memory_path is not None:
                    from rpent.memory import MemoryManager

                    package_memory = MemoryManager(
                        root=harness_package.memory_path,
                        package_memory=harness_package.identity.schema_version == 2,
                    )
                toolkit = get_toolkit(
                    robot_name,
                    primitives_kwargs=primitives_kwargs,
                    dashboard_events=dashboard_events,
                    mode=("exploration" if args.explore else "evaluation"),
                    attempts_per_session=getattr(
                        args,
                        "explore_attempts_per_session",
                        0,
                    ),
                    state_output_dir=state_output_dir,
                    memory=package_memory,
                )
            else:
                toolkit_kwargs: dict[str, Any] = {
                    "primitives_kwargs": primitives_kwargs,
                    "dashboard_events": dashboard_events,
                }
                if (robot_name in {"robocasa", "robotwin"} and harness_package is not None
                        and harness_package.memory_path is not None):
                    from rpent.memory import MemoryManager

                    toolkit_kwargs["memory"] = MemoryManager(
                        root=harness_package.memory_path,
                        package_memory=harness_package.identity.schema_version == 2,
                    )
                toolkit = get_toolkit(robot_name, **toolkit_kwargs)
            solved = False
            harness_adapter: Any = None
            bound_harness: BoundHarness | None = None
            try:
                planner_toolkit = toolkit
                episode_system_prompt = system_prompt
                if harness_package is not None:
                    bound_harness, harness_adapter = _bind_active_harness(
                        harness_package,
                        toolkit=toolkit,
                        robot_spec=robot_spec,
                        episode_dir=output_dir,
                    )
                    planner_toolkit = bound_harness.toolkit
                    episode_system_prompt = _episode_system_prompt(
                        system_prompt,
                        bound_harness.instructions,
                        robot_spec=robot_spec,
                        package=harness_package,
                    )
                if bound_harness is not None:
                    from rpent.harness import inspect_harness
                    (output_dir / "harness-inspection.json").write_text(
                        json.dumps(inspect_harness(bound_harness), ensure_ascii=False, indent=2))
                live_bridge = None
                if os.environ.get("RPENT_LIVE_MAILBOX"):
                    from meta_harness.live import LiveBridge
                    live_bridge = LiveBridge(os.environ["RPENT_LIVE_MAILBOX"], planner=planner,
                                             adapter=harness_adapter, bound=bound_harness).start()
                try:
                    result = planner.solve(
                        system_prompt=episode_system_prompt,
                        user_message=session_msg,
                        toolkit=planner_toolkit,
                        max_turns=args.max_turns,
                        input_queue=input_queue,
                    )
                finally:
                    if live_bridge is not None:
                        live_bridge.close()
                finish_result = result.finish_result
                messages += result.messages
                stats = result.stats
                agent_error = result.error
                if robot_name == "libero":
                    solved = toolkit.solved()
                    if solved:
                        recipe_path = toolkit.write_recipe(recipe_tag)
            finally:
                if harness_adapter is not None:
                    try:
                        captured_outcome = harness_adapter.outcome()
                        if not isinstance(captured_outcome, EpisodeOutcome):
                            raise TypeError(
                                "Harness adapter outcome() must return EpisodeOutcome"
                            )
                        episode_outcome = captured_outcome
                    except Exception as exc:
                        capture_error = (
                            f"outcome capture failed: {type(exc).__name__}: {exc}"
                        )
                        agent_error = (
                            f"{agent_error}; {capture_error}"
                            if agent_error
                            else capture_error
                        )
                        logger.error("%s", capture_error)
                    try:
                        capability_calls = [
                            dict(call) for call in harness_adapter.capability_calls()
                        ]
                    except Exception as exc:
                        trace_error = (
                            "capability trace capture failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        agent_error = (
                            f"{agent_error}; {trace_error}"
                            if agent_error
                            else trace_error
                        )
                        logger.error("%s", trace_error)
                    try:
                        result_state = toolkit.state
                    except (AttributeError, RuntimeError):
                        result_state = None
                try:
                    toolkit.close()
                except Exception as exc:
                    close_error = f"episode finalization failed: {type(exc).__name__}: {exc}"
                    agent_error = f"{agent_error}; {close_error}" if agent_error else close_error
                    logger.error("%s", close_error)
                finally:
                    if bound_harness is not None:
                        bound_harness.close()
            if solved:
                break
            if agent_error:
                if (
                    getattr(args, "explore", False)
                    and session_number < sessions
                    and "timed out" in agent_error.lower()
                ):
                    logger.warning(
                        "session %d/%d timed out; continuing with a fresh handoff",
                        session_number,
                        sessions,
                    )
                    continue
                break
    except Exception as exc:
        loop_error = f"{type(exc).__name__}: {exc}"
        agent_error = f"{agent_error}; {loop_error}" if agent_error else loop_error
        logger.error("EXCEPTION in agent loop: %s", agent_error)
    finally:
        if recipe_path:
            logger.info("recipe: %s", recipe_path)
        else:
            logger.info("recipe: not written (cell unsolved)")
        if robot_name == "libero":
            # Exploration can rebuild Toolkits across several planner sessions
            # while sharing one model client. Release its episode stream once,
            # after every session, and before stopping an owned model server.
            close_model = getattr(primitives_kwargs.get("model"), "close", None)
            if close_model is not None:
                try:
                    close_model()
                except Exception as exc:
                    close_error = f"Pi0.5 session cleanup failed: {type(exc).__name__}: {exc}"
                    agent_error = f"{agent_error}; {close_error}" if agent_error else close_error
                    logger.error("%s", close_error)
        for d in daemons:
            try:
                d.stop()
            except Exception as exc:
                stop_error = f"daemon shutdown failed: {type(exc).__name__}: {exc}"
                agent_error = (
                    f"{agent_error}; {stop_error}" if agent_error else stop_error
                )
                logger.error("%s", stop_error)

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    try:
        with open(transcript_path, "a", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
    except Exception as exc:
        transcript_error = f"transcript write failed: {type(exc).__name__}: {exc}"
        agent_error = (
            f"{agent_error}; {transcript_error}" if agent_error else transcript_error
        )
        logger.error("%s", transcript_error)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)

    # Publish environment-specific artifacts after recipe export and shutdown.
    if robot_spec.finalize_run is not None and not agent_error:
        try:
            finalized = robot_spec.finalize_run(args, run_config)
            if finalized is not None:
                logger.info("run finalized: %s", finalized)
        except Exception as exc:
            agent_error = f"memory finalization failed: {type(exc).__name__}: {exc}"
            logger.error("%s", agent_error)

    if result_dir is not None and harness_package is not None:
        try:
            result_path = _publish_harness_result(
                result_dir,
                args=args,
                task_desc=task_desc,
                package=harness_package,
                outcome=episode_outcome,
                agent_error=agent_error,
                finish_result=finish_result,
                messages=messages,
                stats=stats,
                elapsed=elapsed,
                capability_calls=capability_calls,
                state=result_state,
            )
            logger.info("result: %s", result_path)
        except Exception as exc:
            result_error = f"result write failed: {type(exc).__name__}: {exc}"
            agent_error = (
                f"{agent_error}; {result_error}" if agent_error else result_error
            )
            logger.error("%s", result_error)

    invalid_harness_outcome = harness_package is not None and (
        episode_outcome is None or episode_outcome.validity != "valid"
    )
    return 1 if agent_error or invalid_harness_outcome else 0


if __name__ == "__main__":
    sys.exit(main())
