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

"""Public contracts for file-backed harnesses."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rpent.harness.runtime import HarnessToolkitView, PolicyRuntime
    from rpent.result import EpisodeOutcome


class HarnessError(ValueError):
    """A harness package is malformed or cannot be bound safely."""


@dataclass(frozen=True, slots=True)
class HarnessIdentity:
    """Git identity of a published harness, or ``None`` for an unfrozen draft."""

    harness_id: str
    schema_version: int
    commit: str | None


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """Validated, immutable snapshot of one declared skill."""

    name: str
    instructions: str
    requires: Mapping[str, str]
    tool_source: str | None
    tool_filename: str | None


@dataclass(frozen=True, slots=True)
class HarnessPackage:
    """Validated harness snapshot, independent of a live robot runtime."""

    root: Path
    identity: HarnessIdentity
    native_tools: tuple[str, ...]
    skills: tuple[SkillDefinition, ...]
    instructions: str
    system_prompt: str | None = None
    user_prompt: str | None = None
    planner_tools: tuple[str, ...] = ()
    policy_entry: str | None = None
    memory_path: Path | None = None
    episode_notes: bool = False
    resources: Mapping[str, Any] = field(default_factory=dict)
    source_paths: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, bytes] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class PlannerRequest:
    """SDK-independent request exposed to a package Policy.

    ``messages`` contains ordinary dicts/lists from the model-message adapter,
    with ``kind`` on messages and ``part_kind`` on parts. Images retain their
    bytes as ``{"kind": "binary", "data": bytes, "media_type": ...}``;
    datetimes and bytes remain Python values, not base64 text. RPent owns the
    conversion back to SDK objects. Tools use name/description/input_schema.
    A Policy may choose a tool subset, but cannot redefine bound schemas or
    change the host's model settings and budgets.
    """

    messages: list[dict[str, Any]]
    instructions: str | None
    model_settings: dict[str, Any]
    tools: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolDecision:
    """One package-programmed call through the normal validated tool path."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Public, serializable contract for one versioned robot capability."""

    id: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CapabilityBinding:
    """Public capability contract plus its trusted live implementation.

    ``invoke`` accepts one argument mapping. An adapter normally implements it
    by validating/clamping those arguments and then calling the backing
    ``Toolkit.execute_tool``. Capabilities are never exposed directly to the
    model; a skill can reach only the bindings named in its ``requires`` map.
    """

    spec: CapabilitySpec
    invoke: Callable[[Mapping[str, Any]], Any]

    @property
    def capability_id(self) -> str:
        """Return the versioned id used by Harness requirements."""
        return self.spec.id


class HarnessHost(Protocol):
    """The complete environment-specific seam used by the generic runner."""

    def allowed_native_tools(self) -> frozenset[str]:
        """Return the fixed base-tool allowlist for Harness mode."""
        ...

    def capabilities(self) -> Mapping[str, CapabilityBinding]:
        """Return the capabilities available in the current episode."""
        ...

    def episode_active(self) -> bool:
        """Whether the episode still accepts actions, without revealing why."""
        ...

    def outcome(self) -> EpisodeOutcome:
        """Read the trusted episode outcome before the runtime is closed."""
        ...

    def capability_calls(self) -> Sequence[Mapping[str, Any]]:
        """Return the sanitized audit trail of internal capability calls."""
        ...


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """The single optional agent-facing tool produced by a skill."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class BoundHarness:
    """A validated harness bound to one live episode toolkit."""

    identity: HarnessIdentity
    instructions: str
    toolkit: HarnessToolkitView
    package: HarnessPackage
    policy_runtime: PolicyRuntime
    memory_dir: Path | None = None
    notes_dir: Path | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Mutable state belonging only to this bound episode."""
        return self.policy_runtime.state

    def close(self) -> None:
        """Release only package-local resources; the host closes its Toolkit."""
        self.policy_runtime.close()
