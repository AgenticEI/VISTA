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

"""Read-only episode sources for the Dashboard presentation layer.

The live Dashboard historically read directly from :class:`DashboardState`.
This module places a small seam between that presentation and its data source:
one adapter preserves the live behaviour, while another projects a canonical
Result directory without reading simulator-private artifacts.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rpent.dashboard.state import DashboardState

_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
_VIDEO_STEP = re.compile(r"step_0*([0-9]+)(?:_[^/]*)?\.mp4\Z")
_TASK_FIELDS = ("robot", "task_name", "split", "seed", "max_episode_steps")
_HARNESS_FIELDS = ("id", "schema_version", "commit", "content_hash")
_AGENT_FIELDS = ("model", "planner")
_METRIC_FIELDS = (
    "model_requests",
    "agent_turns",
    "tool_calls",
    "env_steps",
    "elapsed_s",
    "input_tokens",
    "output_tokens",
)


class EpisodeSourceError(ValueError):
    """A Result directory cannot be safely projected as a Dashboard episode."""


@runtime_checkable
class EpisodeSource(Protocol):
    """The data seam consumed by an episode Dashboard.

    ``frame`` addresses a recorded public frame.  Omitting ``step`` means the
    latest available frame for that channel.  ``video_path`` uses ``step=None``
    for the whole episode and a numeric step for an action clip.
    """

    @property
    def run_id(self) -> str: ...

    @property
    def read_only(self) -> bool: ...

    def run_info(self) -> dict[str, Any]: ...

    def run_detail(self) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def events_since(self, since: int) -> list[dict[str, Any]]: ...

    def frame(self, kind: str, step: int | None = None) -> bytes | None: ...

    def video_path(self, step: int | None = None) -> Path | None: ...


@dataclass(frozen=True, slots=True)
class DashboardStateEpisodeSource:
    """Adapt the existing in-memory live Dashboard state to ``EpisodeSource``."""

    state: DashboardState

    @property
    def run_id(self) -> str:
        return self.state.run_id

    @property
    def read_only(self) -> bool:
        return False

    def run_info(self) -> dict[str, Any]:
        return self.state.run_info()

    def run_detail(self) -> dict[str, Any]:
        return self.state.run_detail()

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    def events_since(self, since: int) -> list[dict[str, Any]]:
        return self.state.events_since(since)

    def frame(self, kind: str, step: int | None = None) -> bytes | None:
        if step is not None:
            requested = _non_negative_int(step, label="frame step")
            if requested != self.state.snapshot().get("frame_idx"):
                return None
        return self.state.frame(kind)

    def video_path(self, step: int | None = None) -> Path | None:
        if step is not None:
            return self.state.action_video_path(
                _non_negative_int(step, label="action-video step")
            )
        return self.state.video_path if self.state.has_video() else None


class ResultDirectoryEpisodeSource:
    """Immutable Dashboard projection of one canonical Result directory.

    Only ``result.json``, paths declared below its ``public`` projection, and
    the two conventional video locations are inspected.  Raw simulator state,
    legacy transcripts, success criteria, and ``private/`` are never read.
    """

    def __init__(self, result_dir: str | Path, *, run_id: str | None = None) -> None:
        supplied_root = Path(result_dir)
        if supplied_root.is_symlink():
            raise EpisodeSourceError(
                f"Result directory must not be a symlink: {supplied_root}"
            )
        if not supplied_root.is_dir():
            raise EpisodeSourceError(f"Result directory not found: {supplied_root}")
        self._root = supplied_root.resolve()
        self._run_id = _run_identifier(run_id, self._root)
        self._public_root = self._root / "public"
        if self._public_root.is_symlink():
            raise EpisodeSourceError(
                f"public Result directory must not be a symlink: {self._public_root}"
            )

        manifest_path = self._required_file(
            self._root / "result.json",
            anchor=self._root,
            label="Result manifest",
        )
        self._manifest = _read_json_object(manifest_path, label="Result manifest")
        if self._manifest.get("schema_version") not in (1, 2):
            raise EpisodeSourceError(
                f"unsupported Result schema: {self._manifest.get('schema_version')!r}"
            )
        raw_public = self._manifest.get("public")
        if not isinstance(raw_public, dict):
            raise EpisodeSourceError("result.json public must be an object")
        self._public = raw_public

        transcript_path = self._declared_public_file(
            "transcript",
            default="public/transcript.json",
        )
        transcript = _read_json_object(transcript_path, label="public transcript")
        messages = transcript.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise EpisodeSourceError("public transcript messages must be objects")
        self._events = _project_transcript(messages)

        calls_path = self._declared_public_file(
            "capability_calls",
            default="public/capability_calls.jsonl",
        )
        calls = _read_jsonl(calls_path, label="public capability calls")

        steps_path = self._optional_declared_public_file("steps")
        steps = (
            _read_jsonl(steps_path, label="public steps")
            if steps_path is not None
            else []
        )
        frames_dir = self._optional_declared_public_dir("frames")
        self._frames, known_frame_kinds = self._index_frames(steps, frames_dir)
        self._known_frame_kinds = known_frame_kinds
        self._frame_aliases = self._build_frame_aliases(known_frame_kinds)

        self._episode_video = self._find_episode_video()
        self._declared_action_videos = self._load_declared_action_videos()
        self._timeline = self._project_timeline(calls)
        self._detail = self._build_detail()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def read_only(self) -> bool:
        return True

    def run_info(self) -> dict[str, Any]:
        """Return the catalog-sized description used by ``/api/runs``."""
        return {
            "id": self.run_id,
            "read_only": True,
            "validity": self._detail["validity"],
            "official_success": self._detail["official_success"],
            "task": copy.deepcopy(self._detail["task"]),
        }

    def run_detail(self) -> dict[str, Any]:
        """Return a detached historical projection in Dashboard vocabulary."""
        return copy.deepcopy(self._detail)

    def snapshot(self) -> dict[str, Any]:
        """Return the lightweight refresh projection, excluding the timeline."""
        value = self.run_detail()
        value.pop("timeline", None)
        return value

    def events_since(self, since: int) -> list[dict[str, Any]]:
        index = _non_negative_int(since, label="transcript offset")
        return copy.deepcopy(self._events[index:])

    def frame(self, kind: str, step: int | None = None) -> bytes | None:
        """Read one public frame, defaulting to the latest for its channel."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("frame kind must be a non-empty string")
        canonical = self._frame_aliases.get(kind, kind)
        if canonical not in self._known_frame_kinds:
            raise ValueError(f"unknown frame kind: {kind!r}")
        indexed = self._frames.get(canonical, {})
        if not indexed:
            return None
        if step is None:
            selected = indexed[max(indexed)]
        else:
            selected = indexed.get(_non_negative_int(step, label="frame step"))
            if selected is None:
                return None
        safe = self._required_file(
            selected,
            anchor=self._public_root,
            label="public frame",
        )
        return safe.read_bytes()

    def video_path(self, step: int | None = None) -> Path | None:
        """Return a contained episode video or action clip, when available."""
        if step is None:
            if self._episode_video is None:
                return None
            return self._required_file(
                self._episode_video,
                anchor=self._root,
                label="episode video",
            )
        action_step = _non_negative_int(step, label="action-video step")
        declared = self._declared_action_videos.get(action_step)
        if declared is not None:
            return self._required_file(
                declared,
                anchor=self._public_root,
                label="public action video",
            )
        return self._find_conventional_action_video(action_step)

    def _build_detail(self) -> dict[str, Any]:
        outcome = _object_field(self._manifest, "outcome")
        task = _object_field(self._manifest, "task")
        harness = _object_field(self._manifest, "harness")
        agent = _object_field(self._manifest, "agent")
        metrics = _object_field(self._manifest, "metrics")

        validity = self._manifest.get("validity", outcome.get("validity"))
        if validity not in {"valid", "invalid"}:
            raise EpisodeSourceError(f"unsupported Result validity: {validity!r}")
        official_success = outcome.get("official_success")
        if official_success is not None and not isinstance(official_success, bool):
            raise EpisodeSourceError("outcome.official_success must be boolean or null")
        if validity == "valid" and official_success is None:
            raise EpisodeSourceError("a valid Result requires official_success")

        terminated = _boolean_field(outcome, "terminated", default=False)
        truncated = _boolean_field(outcome, "truncated", default=False)
        raw_finish_claim = agent.get("finish_claim")
        finish_claim = (
            _selected_fields(raw_finish_claim, ("_finish", "status", "summary"))
            if isinstance(raw_finish_claim, dict)
            else None
        )
        frame_available = {
            kind: bool(self._frames.get(canonical))
            for kind, canonical in sorted(
                {
                    **{kind: kind for kind in self._known_frame_kinds},
                    **self._frame_aliases,
                }.items()
            )
        }
        frame_steps = [
            step for indexed in self._frames.values() for step in indexed.keys()
        ]
        usage = {
            "in": _metric_int(metrics, "input_tokens"),
            "out": _metric_int(metrics, "output_tokens"),
            "tool_calls": _metric_int(metrics, "tool_calls"),
        }
        public_task = _selected_fields(task, _TASK_FIELDS)
        public_harness = _selected_fields(harness, _HARNESS_FIELDS)
        public_agent = _selected_fields(agent, _AGENT_FIELDS)
        if finish_claim is not None:
            public_agent["finish_claim"] = copy.deepcopy(finish_claim)
        public_metrics = _selected_fields(metrics, _METRIC_FIELDS)
        recorded_error = "recorded Episode error" if outcome.get("error") else None
        public_outcome = {
            "validity": validity,
            "official_success": official_success,
            "terminated": terminated,
            "truncated": truncated,
            "env_steps": outcome.get("env_steps"),
            "terminal_reason": outcome.get("terminal_reason"),
            "error": recorded_error,
        }
        task_projection = {
            **copy.deepcopy(public_task),
            "parameters": copy.deepcopy(public_task),
            "label": _task_label(public_task),
        }
        state = "succeeded" if official_success is True else "failed"
        return {
            "state": state,
            "terminated": terminated,
            "truncated": truncated,
            "error": recorded_error,
            "usage": usage,
            "runtime": {},
            "timeline": copy.deepcopy(self._timeline),
            "has_video": self._episode_video is not None,
            "frame_idx": max(frame_steps, default=-1),
            "frame_available": frame_available,
            "n_steps": len(self._timeline),
            "interaction": {
                "session_id": self.run_id,
                "input_mode": "disabled",
                "planner_activity": "ended",
                "interrupt_requested": False,
                "messages": [],
                "last_error": None,
            },
            "session_state": "historical",
            "task_generation": 1,
            "current_task": task_projection,
            "pending_task": None,
            "control_feedback": [],
            "control_error": None,
            "historical": True,
            "read_only": True,
            "validity": validity,
            "official_success": official_success,
            "finish_claim": copy.deepcopy(finish_claim),
            "terminal_reason": outcome.get("terminal_reason"),
            "outcome": public_outcome,
            "task": public_task,
            "harness": public_harness,
            "agent": public_agent,
            "metrics": public_metrics,
        }

    def _project_timeline(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        outcome = _object_field(self._manifest, "outcome")
        for position, call in enumerate(calls, start=1):
            sequence = call.get("sequence", position)
            step = _non_negative_int(sequence, label="capability-call sequence")
            action = call.get("capability_id")
            if not isinstance(action, str) or not action:
                raise EpisodeSourceError("capability call requires capability_id")
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                raise EpisodeSourceError("capability call arguments must be an object")
            result = call.get("result_summary", {})
            if not isinstance(result, dict):
                result = {"value": result}
            error = call.get("error")
            if error is not None:
                result = {**result, "error": str(error)}
            timeline.append(
                {
                    "step": step,
                    "action": action,
                    "args": copy.deepcopy(arguments),
                    "result": copy.deepcopy(result),
                    "elapsed_s": call.get("elapsed_s"),
                    "env_steps_before": call.get("env_steps_before"),
                    "env_steps_after": call.get("env_steps_after"),
                    "frame_step": _result_frame_step(result),
                    "terminated": False,
                    "truncated": False,
                    "has_action_video": self.video_path(step) is not None,
                }
            )
        if timeline:
            timeline[-1]["terminated"] = bool(outcome.get("terminated", False))
            timeline[-1]["truncated"] = bool(outcome.get("truncated", False))
        return timeline

    def _index_frames(
        self,
        steps: list[dict[str, Any]],
        frames_dir: Path | None,
    ) -> tuple[dict[str, dict[int, Path]], set[str]]:
        indexed: dict[str, dict[int, Path]] = {}
        known: set[str] = set()
        seen_steps: set[int] = set()
        for row in steps:
            step = _non_negative_int(row.get("step"), label="public frame step")
            if step in seen_steps:
                raise EpisodeSourceError(f"duplicate public frame step: {step}")
            seen_steps.add(step)
            images = row.get("images", [])
            if not isinstance(images, list):
                raise EpisodeSourceError("public step images must be an array")
            for raw_image in images:
                if not isinstance(raw_image, str):
                    raise EpisodeSourceError("public step image names must be strings")
                image = Path(raw_image)
                if (
                    image.name != raw_image
                    or image.suffix.lower() not in _IMAGE_SUFFIXES
                ):
                    raise EpisodeSourceError(
                        f"unsafe public step image name: {raw_image!r}"
                    )
                kind = image.stem
                known.add(kind)
                if frames_dir is None:
                    continue
                candidate = frames_dir / f"{step:04d}_{kind}{image.suffix.lower()}"
                safe = self._optional_file(
                    candidate,
                    anchor=self._public_root,
                    label="public frame",
                )
                if safe is not None:
                    indexed.setdefault(kind, {})[step] = safe
        return indexed, known

    @staticmethod
    def _build_frame_aliases(kinds: set[str]) -> dict[str, str]:
        if "camera" in kinds:
            return {}
        if "agentview" in kinds:
            return {"camera": "agentview"}
        if "navview" in kinds:
            return {"camera": "navview"}
        return {}

    def _find_episode_video(self) -> Path | None:
        declared = self._public.get("video")
        if declared is not None:
            path = self._public_pointer(declared, label="public episode video")
            _require_mp4(path, label="public episode video")
            return self._required_file(
                path,
                anchor=self._public_root,
                label="public episode video",
            )
        for candidate, anchor in (
            (self._public_root / "episode.mp4", self._public_root),
            (self._root / "episode.mp4", self._root),
        ):
            found = self._optional_file(candidate, anchor=anchor, label="episode video")
            if found is not None:
                return found
        return None

    def _load_declared_action_videos(self) -> dict[int, Path]:
        raw = self._public.get("action_videos")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise EpisodeSourceError("public.action_videos must be an object")
        videos: dict[int, Path] = {}
        for raw_step, pointer in raw.items():
            try:
                step = _non_negative_int(int(raw_step), label="action-video step")
            except (TypeError, ValueError) as exc:
                raise EpisodeSourceError(
                    f"invalid public action-video step: {raw_step!r}"
                ) from exc
            path = self._public_pointer(
                pointer,
                label="public action video",
            )
            _require_mp4(path, label="public action video")
            videos[step] = self._required_file(
                path,
                anchor=self._public_root,
                label="public action video",
            )
        return videos

    def _find_conventional_action_video(self, step: int) -> Path | None:
        candidates: list[Path] = []
        for directory, anchor in (
            (self._public_root / "action_videos", self._public_root),
            (self._root / "action_videos", self._root),
        ):
            safe_dir = self._optional_dir(
                directory,
                anchor=anchor,
                label="action-video directory",
            )
            if safe_dir is None:
                continue
            for item in safe_dir.iterdir():
                match = _VIDEO_STEP.fullmatch(item.name)
                if match is None or int(match.group(1)) != step:
                    continue
                safe = self._required_file(
                    item,
                    anchor=anchor,
                    label="action video",
                )
                candidates.append(safe)
        unique = sorted(set(candidates))
        if len(unique) > 1:
            raise EpisodeSourceError(f"ambiguous action video for step {step}")
        return unique[0] if unique else None

    def _declared_public_file(self, key: str, *, default: str) -> Path:
        raw = self._public.get(key, default)
        if raw is None:
            raise EpisodeSourceError(f"result.json public.{key} is required")
        path = self._public_pointer(raw, label=f"public {key}")
        return self._required_file(
            path, anchor=self._public_root, label=f"public {key}"
        )

    def _optional_declared_public_file(self, key: str) -> Path | None:
        raw = self._public.get(key)
        if raw is None:
            return None
        path = self._public_pointer(raw, label=f"public {key}")
        return self._required_file(
            path, anchor=self._public_root, label=f"public {key}"
        )

    def _optional_declared_public_dir(self, key: str) -> Path | None:
        raw = self._public.get(key)
        if raw is None:
            return None
        path = self._public_pointer(raw, label=f"public {key}")
        return self._required_dir(path, anchor=self._public_root, label=f"public {key}")

    def _public_pointer(self, raw: Any, *, label: str) -> Path:
        if not isinstance(raw, str) or not raw:
            raise EpisodeSourceError(f"{label} path must be a non-empty string")
        relative = Path(raw)
        if relative.is_absolute():
            raise EpisodeSourceError(f"{label} path must be Result-relative: {raw}")
        if ".." in relative.parts:
            raise EpisodeSourceError(f"{label} path must not contain '..': {raw}")
        candidate = self._root / relative
        _require_lexical_containment(candidate, self._public_root, label=label)
        return candidate

    def _required_file(self, path: Path, *, anchor: Path, label: str) -> Path:
        safe = self._optional_file(path, anchor=anchor, label=label)
        if safe is None:
            raise EpisodeSourceError(f"{label} not found: {path}")
        return safe

    def _optional_file(self, path: Path, *, anchor: Path, label: str) -> Path | None:
        _reject_symlinks(path, anchor, label=label)
        _require_lexical_containment(path, anchor, label=label)
        if not path.exists():
            return None
        if not path.is_file():
            raise EpisodeSourceError(f"{label} must be a regular file: {path}")
        return path

    def _required_dir(self, path: Path, *, anchor: Path, label: str) -> Path:
        safe = self._optional_dir(path, anchor=anchor, label=label)
        if safe is None:
            raise EpisodeSourceError(f"{label} not found: {path}")
        return safe

    def _optional_dir(self, path: Path, *, anchor: Path, label: str) -> Path | None:
        _reject_symlinks(path, anchor, label=label)
        _require_lexical_containment(path, anchor, label=label)
        if not path.exists():
            return None
        if not path.is_dir():
            raise EpisodeSourceError(f"{label} must be a directory: {path}")
        return path


def _run_identifier(value: str | None, root: Path) -> str:
    candidate = root.name if value is None else value
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("run_id must be a non-empty string")
    return candidate


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise EpisodeSourceError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeSourceError(f"{label} must be a non-negative integer") from exc
    if result < 0 or str(result) != str(value):
        raise EpisodeSourceError(f"{label} must be a non-negative integer")
    return result


def _require_lexical_containment(path: Path, anchor: Path, *, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(anchor.resolve(strict=False))
    except ValueError as exc:
        raise EpisodeSourceError(
            f"{label} escapes its allowed directory: {path}"
        ) from exc


def _require_mp4(path: Path, *, label: str) -> None:
    if path.suffix.lower() != ".mp4":
        raise EpisodeSourceError(f"{label} must be an MP4 file: {path}")


def _reject_symlinks(path: Path, anchor: Path, *, label: str) -> None:
    current = path
    anchor_absolute = anchor.absolute()
    while True:
        if current.is_symlink():
            raise EpisodeSourceError(f"{label} must not use symlinks: {current}")
        if current.absolute() == anchor_absolute:
            return
        if current.parent == current:
            raise EpisodeSourceError(f"{label} escapes its allowed directory: {path}")
        current = current.parent


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeSourceError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EpisodeSourceError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EpisodeSourceError(f"cannot read {label}: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EpisodeSourceError(
                f"invalid {label} JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EpisodeSourceError(
                f"{label} row must be an object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _object_field(value: dict[str, Any], name: str) -> dict[str, Any]:
    raw = value.get(name, {})
    if not isinstance(raw, dict):
        raise EpisodeSourceError(f"result.json {name} must be an object")
    return raw


def _boolean_field(value: dict[str, Any], name: str, *, default: bool) -> bool:
    raw = value.get(name, default)
    if not isinstance(raw, bool):
        raise EpisodeSourceError(f"outcome.{name} must be boolean")
    return raw


def _metric_int(metrics: dict[str, Any], name: str) -> int:
    raw = metrics.get(name, 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def _result_frame_step(result: dict[str, Any]) -> int | None:
    """Return the public observation frame shared by one capability result."""
    observation = result.get("observation")
    projection = observation if isinstance(observation, dict) else result
    images = projection.get("images")
    if not isinstance(images, list):
        return None
    steps = {
        image.get("step")
        for image in images
        if isinstance(image, dict)
        and type(image.get("step")) is int
        and image["step"] >= 0
    }
    if len(steps) != 1:
        return None
    return steps.pop()


def _selected_fields(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy only fields that are part of the recorded Dashboard contract."""
    return {name: copy.deepcopy(value[name]) for name in fields if name in value}


def _task_label(task: dict[str, Any]) -> str:
    preferred = [task.get("task_name"), task.get("split")]
    if task.get("seed") is not None:
        preferred.append(f"seed {task['seed']}")
    values = [str(item) for item in preferred if item not in {None, ""}]
    if values:
        return " / ".join(values)
    return json.dumps(task, sort_keys=True, ensure_ascii=False, default=str)


def _project_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    first_user = True
    current_turn = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            text = _content_text(content)
            if first_user:
                events.append({"type": "initial_prompt", "turn": 0})
                first_user = False
            elif text:
                events.append({"type": "user", "text": text, "turn": current_turn})
            continue
        if role == "assistant":
            current_turn += 1
            for block in _content_blocks(content):
                block_type = block.get("type")
                if block_type in {"thinking", "reasoning"}:
                    text = block.get("thinking", block.get("text", ""))
                    if text:
                        events.append(
                            {
                                "type": "thinking",
                                "text": str(text),
                                "turn": current_turn,
                            }
                        )
                elif block_type in {"text", "output_text"}:
                    text = block.get("text", "")
                    if text:
                        events.append(
                            {
                                "type": "text",
                                "text": str(text),
                                "turn": current_turn,
                            }
                        )
                elif block_type in {"tool_use", "tool_call"}:
                    name = block.get("name", block.get("tool", "tool"))
                    arguments = block.get("input", block.get("args", {}))
                    events.append(
                        {
                            "type": "tool_call",
                            "tool": str(name),
                            "args": arguments,
                            "turn": current_turn,
                        }
                    )
                    tool_id = block.get("id")
                    if tool_id:
                        tool_names[str(tool_id)] = str(name)
            continue
        if role in {"codex_sdk", "claude_agent_sdk"}:
            current_turn += 1
            text = _content_text(content)
            if text:
                events.append(
                    {"type": "text", "text": text, "turn": current_turn}
                )
            continue
        if role == "tool":
            tool_id = message.get("tool_call_id")
            name = message.get("name") or tool_names.get(str(tool_id), "tool_result")
            events.append(
                {
                    "type": "tool_result",
                    "tool": str(name),
                    "turn": current_turn,
                    "result": {
                        "is_error": bool(message.get("is_error", False)),
                        "size": len(_content_text(content)),
                    },
                }
            )
    return events


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text", block.get("content"))
            if isinstance(text, str):
                texts.append(text)
        return "\n".join(texts)
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(content)


__all__ = [
    "DashboardStateEpisodeSource",
    "EpisodeSource",
    "EpisodeSourceError",
    "ResultDirectoryEpisodeSource",
]
