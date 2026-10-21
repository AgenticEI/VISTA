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

"""The durable episode result seam used by evaluation and Meta Harness.

The robot runtime may keep rich, privileged artifacts in its normal output
directory.  This module writes one small authoritative manifest and a separate
public projection.  Meta code is given only the public projection.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from rpent.trace import TRACE_INDEX, inspect_trajectory

if TYPE_CHECKING:
    from rpent.tools.state import EnvState, StepRecord

Validity = Literal["valid", "invalid"]

_PRIVATE_KEYS = {
    "ground_truth",
    "gt",
    "oracle",
    "robocasa_terminated",
    "success",
    "success_criteria",
    "task_progress",
    "eval_success",
}
_SECRET_FRAGMENTS = ("api_key", "apikey", "access_token", "secret", "password")
_PUBLIC_IMAGES = frozenset(
    {
        "agentview.png",
        "agentview_high.png",
        "navview.png",
        "navview_floor.png",
        "wrist.png",
        "wrist_high.png",
        "head_rgb.png",
        "left_wrist_rgb.png",
        "right_wrist_rgb.png",
    }
)
_PUBLIC_ARTIFACT_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".npy", ".npz", ".json", ".jsonl", ".txt", ".md"}
)


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Environment-authoritative terminal state for one episode."""

    validity: Validity
    official_success: bool | None
    terminated: bool
    truncated: bool
    env_steps: int | None
    terminal_reason: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.validity not in {"valid", "invalid"}:
            raise ValueError(f"unsupported validity: {self.validity!r}")
        if self.validity == "valid" and self.official_success is None:
            raise ValueError("a valid outcome requires official_success")
        if self.env_steps is not None and self.env_steps < 0:
            raise ValueError("env_steps must be non-negative")
        if not self.terminal_reason:
            raise ValueError("terminal_reason must not be empty")


def public_projection(value: Any) -> Any:
    """Return a JSON-safe copy with privileged and secret-shaped fields removed."""
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_")
            if normalized in _PRIVATE_KEYS:
                continue
            if any(fragment in normalized for fragment in _SECRET_FRAGMENTS):
                projected[key] = "<redacted>"
                continue
            projected[key] = public_projection(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [public_projection(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def write_episode_result(
    output_dir: str | Path,
    *,
    task: Mapping[str, Any],
    harness: Mapping[str, Any],
    agent: Mapping[str, Any],
    outcome: EpisodeOutcome,
    metrics: Mapping[str, Any],
    experiment: Mapping[str, Any] | None = None,
    config_ref: str | None = None,
    controller: Literal["planner", "meta", "mixed"] = "planner",
    agent_claimed_success: bool | None = None,
    actors: Sequence[str] = (),
    messages: Sequence[Mapping[str, Any]] = (),
    capability_calls: Sequence[Mapping[str, Any]] = (),
    state: "EnvState | None" = None,
) -> Path:
    """Publish after the environment's ``finish_trajectory`` has completed.

    A running or failed asynchronous recorder makes the episode invalid. The
    official outcome, agent's claim and any manual intervention remain separate.
    """
    if controller not in {"planner", "meta", "mixed"}:
        raise ValueError("controller must be planner, meta or mixed")
    observed_actors = list(dict.fromkeys(actors))
    if any(actor not in {"planner", "harness_program", "meta"} for actor in observed_actors):
        raise ValueError("actors must be planner, harness_program or meta")
    if "meta" in observed_actors and controller == "planner":
        controller = "mixed" if any(actor != "meta" for actor in observed_actors) else "meta"
    if agent_claimed_success is None:
        claim = agent.get("claimed_success")
        agent_claimed_success = claim if isinstance(claim, bool) else None
    if agent_claimed_success is not None and not isinstance(agent_claimed_success, bool):
        raise ValueError("agent_claimed_success must be boolean or null")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    public_dir = root / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    _atomic_json(
        public_dir / "transcript.json",
        {"messages": public_projection(list(messages))},
    )
    _atomic_jsonl(
        public_dir / "capability_calls.jsonl",
        [public_projection(call) for call in capability_calls],
    )
    if state is not None:
        _write_public_state(public_dir, state, capability_calls)

    trajectory = inspect_trajectory(root, outcome.env_steps)
    if trajectory is not None and trajectory["required"] and not trajectory["complete"]:
        outcome = EpisodeOutcome(
            validity="invalid",
            official_success=None,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            env_steps=outcome.env_steps,
            terminal_reason="trace_incomplete",
            error="; ".join(filter(None, [outcome.error, *trajectory["errors"]])),
        )
    if outcome.validity == "invalid" and outcome.official_success is not None:
        outcome = replace(outcome, official_success=None)

    experiment_public = public_projection(dict(experiment or {}))
    config_ref = config_ref or experiment_public.get("config_ref")
    if config_ref is not None and (not isinstance(config_ref, str) or not config_ref.strip()):
        raise ValueError("config_ref must be a nonempty configuration reference or null")
    harness_public = public_projection(dict(harness))
    manifest = {
        "schema_version": 2,
        "validity": outcome.validity,
        "task": public_projection(dict(task)),
        "harness": harness_public,
        "harness_commit": harness_public.get("commit"),
        "config_ref": config_ref,
        "controller": controller,
        "actors": observed_actors,
        "official_success": outcome.official_success,
        "agent_claimed_success": agent_claimed_success,
        "terminal_reason": outcome.terminal_reason,
        "env_steps": outcome.env_steps,
        "trace_complete": trajectory["complete"] if trajectory is not None else None,
        "agent": public_projection(dict(agent)),
        "outcome": asdict(outcome),
        "metrics": public_projection(dict(metrics)),
        "experiment": experiment_public,
        "public": {
            "transcript": "public/transcript.json",
            "capability_calls": "public/capability_calls.jsonl",
            "steps": "public/steps.jsonl" if state is not None else None,
            "frames": "public/frames" if state is not None else None,
            "artifacts": "public/artifacts.jsonl" if state is not None else None,
            "trajectory": TRACE_INDEX if trajectory is not None else None,
            "trajectory_metadata": trajectory,
        },
    }
    destination = root / "result.json"
    _atomic_json(destination, manifest)
    return destination


def write_invalid_result(
    output_dir: str | Path,
    *,
    task: Mapping[str, Any],
    harness: Mapping[str, Any],
    agent: Mapping[str, Any],
    reason: str,
    error: str,
    experiment: Mapping[str, Any] | None = None,
    config_ref: str | None = None,
    controller: Literal["planner", "meta", "mixed"] = "planner",
    actors: Sequence[str] = (),
) -> Path:
    """Publish the smallest useful result when the episode process failed early."""
    return write_episode_result(
        output_dir,
        task=task,
        harness=harness,
        agent=agent,
        outcome=EpisodeOutcome(
            validity="invalid",
            official_success=None,
            terminated=False,
            truncated=False,
            env_steps=None,
            terminal_reason=reason,
            error=error,
        ),
        metrics={},
        experiment=experiment,
        config_ref=config_ref,
        controller=controller,
        actors=actors,
    )


def _write_public_state(
    public_dir: Path,
    state: "EnvState",
    capability_calls: Sequence[Mapping[str, Any]],
) -> None:
    records = list(state.records())
    frames_dir = public_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = [_public_step(record) for record in records]
    _atomic_jsonl(public_dir / "steps.jsonl", rows)

    for record in records:
        for artifact in sorted(record.artifacts.intersection(_PUBLIC_IMAGES)):
            try:
                source = state.artifact_path(artifact, step=record.step_idx)
                suffix = source.suffix.lower()
                destination = (
                    frames_dir / f"{record.step_idx:04d}_{Path(artifact).stem}{suffix}"
                )
                _link_or_copy(source, destination)
            except (OSError, ValueError):
                continue
    _write_public_artifacts(public_dir, state, records, capability_calls)


def _write_public_artifacts(
    public_dir: Path,
    state: "EnvState",
    records: Sequence["StepRecord"],
    capability_calls: Sequence[Mapping[str, Any]],
) -> None:
    """Project only artifacts explicitly declared by trusted capability calls."""
    artifacts_dir = public_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    by_step = {record.step_idx: record for record in records}
    projected: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for call in capability_calls:
        declarations = call.get("public_artifacts", ())
        if not isinstance(declarations, (list, tuple)):
            continue
        for declaration in declarations:
            if not isinstance(declaration, Mapping):
                continue
            artifact = declaration.get("artifact")
            step = declaration.get("step")
            if (
                not isinstance(artifact, str)
                or Path(artifact).name != artifact
                or Path(artifact).suffix.lower() not in _PUBLIC_ARTIFACT_SUFFIXES
                or isinstance(step, bool)
                or not isinstance(step, int)
                or step not in by_step
                or artifact not in by_step[step].artifacts
                or (step, artifact) in seen
            ):
                continue
            try:
                source = state.artifact_path(artifact, step=step)
                if not source.is_file() or source.is_symlink():
                    continue
                destination = artifacts_dir / f"{step:04d}_{artifact}"
                _link_or_copy(source, destination)
            except (OSError, ValueError):
                continue
            seen.add((step, artifact))
            projected.append(
                public_projection(
                    {
                        "artifact": artifact,
                        "path": f"public/artifacts/{destination.name}",
                        "step": step,
                        "kind": declaration.get("kind"),
                        "capture_id": declaration.get("capture_id"),
                    }
                )
            )
    _atomic_jsonl(public_dir / "artifacts.jsonl", projected)


def _public_step(record: "StepRecord") -> dict[str, Any]:
    """Return only a media timeline, never the base Toolkit's raw payloads.

    Agent-visible state and operation summaries already live in the transcript
    and adapter-sanitized capability trace. Base ``state/result/extras`` may
    contain simulator contact, object identity, progress, or other fields that
    the Harness intentionally did not expose, so this generic writer must not
    attempt to guess an environment-specific safe subset.
    """
    return public_projection(
        {
            "step": record.step_idx,
            "elapsed_s": record.elapsed_s,
            "images": sorted(record.artifacts.intersection(_PUBLIC_IMAGES)),
        }
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
