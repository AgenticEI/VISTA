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

"""Load ordinary, Git-versioned Harness packages without executing their code."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from rpent.harness.contracts import (
    HarnessError,
    HarnessIdentity,
    HarnessPackage,
    SkillDefinition,
)

_V1_KEYS = {"schema_version", "id", "native_tools", "skills"}
_V2_REQUIRED_KEYS = _V1_KEYS | {"prompts", "planner_tools"}
_V2_OPTIONAL_KEYS = {"policy", "memory", "resources"}
_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SKILL_KEYS = {"name", "requires"}
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HARNESS_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_CAPABILITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*@[1-9][0-9]*$")


def _fail(message: str) -> HarnessError:
    return HarnessError(message)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_text(path: Path, *, label: str) -> tuple[str, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _fail(f"cannot read {label} at {path}: {exc}") from exc
    try:
        return raw.decode("utf-8"), raw
    except UnicodeDecodeError as exc:
        raise _fail(f"{label} must be UTF-8: {path}") from exc


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise _fail(f"{label} has invalid fields: {', '.join(details)}")


def _require_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _NAME_PATTERN.fullmatch(value) is None:
        raise _fail(f"{label} must match {_NAME_PATTERN.pattern!r}; got {value!r}")
    return value


def _validate_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise _fail(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise _fail(f"missing {label}: {path}")


def _validate_directory(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise _fail(f"{label} must not be a symlink: {path}")
    if not path.is_dir():
        raise _fail(f"missing {label}: {path}")


def _validate_directory_entries(
    directory: Path,
    *,
    allowed: set[str],
    label: str,
) -> None:
    try:
        actual = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise _fail(f"cannot list {label} at {directory}: {exc}") from exc
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if unexpected or missing:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise _fail(f"{label} has invalid entries: {', '.join(details)}")


def _parse_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    text, raw = _read_text(path, label="HARNESS.json")
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except HarnessError:
        raise
    except json.JSONDecodeError as exc:
        raise _fail(f"invalid HARNESS.json: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail("HARNESS.json must contain one JSON object")
    return value, raw


def _parse_native_tools(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _fail("native_tools must be a non-empty array")
    tools = tuple(
        _require_name(item, label=f"native_tools[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(tools)) != len(tools):
        raise _fail("native_tools must not contain duplicates")
    if "finish" not in tools:
        raise _fail("native_tools must include the fixed 'finish' tool")
    return tools


def _parse_requirement(
    value: Any,
    *,
    label: str,
    earlier_tool_names: set[str],
) -> str:
    if not isinstance(value, str):
        raise _fail(f"{label} must be a string")
    kind, separator, target = value.partition(":")
    if not separator or not target:
        raise _fail(f"{label} must be 'cap:<id>' or 'tool:<earlier-tool>'")
    if kind == "cap":
        if _CAPABILITY_ID_PATTERN.fullmatch(target) is None:
            raise _fail(f"{label} has invalid versioned capability id: {target!r}")
        return value
    if kind == "tool":
        if target not in earlier_tool_names:
            raise _fail(
                f"{label} must reference an earlier skill with tool.py; got {target!r}"
            )
        return value
    raise _fail(f"{label} has unsupported requirement kind: {kind!r}")


def _load_skills(
    root: Path,
    value: Any,
    files: list[tuple[str, bytes]],
    *,
    legacy: bool = True,
) -> tuple[SkillDefinition, ...]:
    if not isinstance(value, list) or (legacy and not value):
        raise _fail("skills must be an ordered array" + (" and non-empty" if legacy else ""))
    if not value:
        return ()

    skills_root = root / "skills"
    _validate_directory(skills_root, label="skills directory")
    definitions: list[SkillDefinition] = []
    declared_names: set[str] = set()
    earlier_tool_names: set[str] = set()

    for index, item in enumerate(value):
        label = f"skills[{index}]"
        if not isinstance(item, dict):
            raise _fail(f"{label} must be an object")
        _require_exact_keys(item, _SKILL_KEYS, label)
        name = _require_name(item["name"], label=f"{label}.name")
        if name in declared_names:
            raise _fail(f"duplicate skill name: {name!r}")
        declared_names.add(name)

        skill_dir = skills_root / name
        _validate_directory(skill_dir, label=f"skill directory {name!r}")
        instruction_path = skill_dir / "SKILL.md"
        _validate_regular_file(instruction_path, label=f"{name}/SKILL.md")
        tool_path = skill_dir / "tool.py"
        has_tool = tool_path.exists() or tool_path.is_symlink()
        allowed_entries = {
            "SKILL.md",
            *(("tool.py",) if has_tool else ()),
        }
        if legacy:
            _validate_directory_entries(
                skill_dir,
                allowed=allowed_entries,
                label=f"skill directory {name!r}",
            )

        instructions, instruction_bytes = _read_text(
            instruction_path, label=f"{name}/SKILL.md"
        )
        instructions = instructions.strip()
        if not instructions:
            raise _fail(f"{name}/SKILL.md must not be empty")
        files.append((f"skills/{name}/SKILL.md", instruction_bytes))

        tool_source: str | None = None
        tool_filename: str | None = None
        if has_tool:
            _validate_regular_file(tool_path, label=f"{name}/tool.py")
            tool_source, tool_bytes = _read_text(tool_path, label=f"{name}/tool.py")
            try:
                compile(tool_source, str(tool_path), "exec")
            except SyntaxError as exc:
                raise _fail(f"invalid Python in {name}/tool.py: {exc}") from exc
            tool_filename = str(tool_path)
            files.append((f"skills/{name}/tool.py", tool_bytes))

        requires_value = item["requires"]
        if not isinstance(requires_value, dict):
            raise _fail(f"{label}.requires must be an object")
        requires: dict[str, str] = {}
        for alias, requirement in requires_value.items():
            alias = _require_name(alias, label=f"{label}.requires alias")
            requires[alias] = _parse_requirement(
                requirement,
                label=f"{label}.requires[{alias!r}]",
                earlier_tool_names=earlier_tool_names,
            )

        definitions.append(
            SkillDefinition(
                name=name,
                instructions=instructions,
                requires=MappingProxyType(requires),
                tool_source=tool_source,
                tool_filename=tool_filename,
            )
        )
        if has_tool:
            earlier_tool_names.add(name)

    if legacy:
        _validate_directory_entries(
            skills_root,
            allowed=declared_names,
            label="skills directory",
        )
    return tuple(definitions)


def _build_instructions(harness_id: str, skills: tuple[SkillDefinition, ...]) -> str:
    sections = [f"## Active Harness: {harness_id}"]
    for skill in skills:
        sections.append(f"### Skill: {skill.name}\n\n{skill.instructions}")
    return "\n\n".join(sections)


def _package_path(root: Path, value: Any, *, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise _fail(f"{label} must be a relative package path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or not relative.parts
        or any(part in {".", "..", ".git", "__pycache__"} for part in relative.parts)
    ):
        raise _fail(f"{label} must be a canonical path inside the package")
    target = root
    for part in relative.parts:
        target /= part
        if target.is_symlink():
            raise _fail(f"{label} must not contain a symlink: {target}")
    return target


def _snapshot_files(root: Path) -> dict[str, bytes]:
    """Snapshot source and small resources; Git metadata is not package code."""
    files: dict[str, bytes] = {}

    def visit(directory: Path) -> None:
        for entry in sorted(directory.iterdir()):
            if entry.name in {".git", "__pycache__"}:
                continue
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                raise _fail(f"package entries must not be symlinks: {relative}")
            if entry.is_dir():
                visit(entry)
            elif entry.is_file():
                try:
                    data = entry.read_bytes()
                except OSError as exc:
                    raise _fail(f"cannot read package file {relative}: {exc}") from exc
                if entry.suffix == ".py":
                    try:
                        compile(data, str(entry), "exec")
                    except (SyntaxError, UnicodeDecodeError) as exc:
                        raise _fail(f"invalid Python in {relative}: {exc}") from exc
                files[relative] = data
            else:
                raise _fail(f"package entry must be a regular file: {relative}")

    visit(root)
    return files


def _git_commit(root: Path, explicit: str | None) -> str | None:
    if explicit is not None:
        if not isinstance(explicit, str) or not _COMMIT_PATTERN.fullmatch(explicit):
            raise _fail("commit must be a full lowercase Git commit id")
        # Archived snapshots may live outside their source Git repository. The
        # caller that froze/extracted them supplies their authoritative commit.
        return explicit
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if head.returncode != 0:
            return None
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", "HARNESS.json"],
            capture_output=True, text=True, check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", "."],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    commit = head.stdout.strip()
    if (
        tracked.returncode != 0 or not tracked.stdout.strip()
        or dirty.returncode != 0 or dirty.stdout.strip()
        or not _COMMIT_PATTERN.fullmatch(commit)
    ):
        return None
    return commit


def load_harness(
    harness_dir: Path | str, *, commit: str | None = None
) -> HarnessPackage:
    """Validate and snapshot a package without importing executable modules.

    Schema 2 accepts ordinary helper modules and resources and declares its
    prompts, optional Policy, memory, and Planner-side tools explicitly. Schema
    1 remains readable during migration. A supplied commit identifies a frozen
    Git archive; otherwise a clean tracked working tree resolves its own HEAD.
    An unfrozen or untracked draft has ``identity.commit is None``.
    """
    requested_root = Path(harness_dir)
    _validate_directory(requested_root, label="harness directory")
    root = requested_root.resolve()
    manifest_path = root / "HARNESS.json"
    _validate_regular_file(manifest_path, label="HARNESS.json")
    manifest, manifest_bytes = _parse_manifest(manifest_path)

    version = manifest.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2}:
        raise _fail(f"schema_version must be 1 or 2; got {version!r}")
    if version == 1:
        _require_exact_keys(manifest, _V1_KEYS, "HARNESS.json")
        # Git worktrees use a .git file; regular repositories use a directory.
        # Neither is part of the policy and neither must invalidate loading.
        allowed = {"HARNESS.json", "skills"}
        if (root / ".git").exists():
            allowed.add(".git")
        _validate_directory_entries(root, allowed=allowed, label="harness directory")
    else:
        missing = _V2_REQUIRED_KEYS - manifest.keys()
        unknown = manifest.keys() - (_V2_REQUIRED_KEYS | _V2_OPTIONAL_KEYS)
        if missing or unknown:
            raise _fail(f"HARNESS.json has invalid fields: missing {sorted(missing)}, unknown {sorted(unknown)}")

    harness_id = manifest["id"]
    if not isinstance(harness_id, str) or _HARNESS_ID_PATTERN.fullmatch(harness_id) is None:
        raise _fail(f"id must match {_HARNESS_ID_PATTERN.pattern!r}; got {harness_id!r}")
    native_tools = _parse_native_tools(manifest["native_tools"])
    skill_files = [("HARNESS.json", manifest_bytes)]
    skills = _load_skills(root, manifest["skills"], skill_files, legacy=version == 1)
    files = dict(skill_files) if version == 1 else _snapshot_files(root)
    system_prompt: str | None = None
    user_prompt: str | None = None
    planner_tools: tuple[str, ...] = ()
    policy_entry: str | None = None
    memory_path: Path | None = None
    episode_notes = False
    resources: dict[str, Any] = {}
    sources: dict[str, str] = {"manifest": "HARNESS.json"}
    if version == 2:
        prompts = manifest["prompts"]
        if not isinstance(prompts, dict):
            raise _fail("prompts must be an object")
        _require_exact_keys(prompts, {"system", "user"}, "prompts")
        loaded_prompts: dict[str, str] = {}
        for kind, value in prompts.items():
            path = _package_path(root, value, label=f"prompts.{kind}")
            _validate_regular_file(path, label=f"prompts.{kind}")
            loaded_prompts[kind], _ = _read_text(path, label=f"prompts.{kind}")
            sources[f"prompts.{kind}"] = value
        system_prompt = loaded_prompts["system"]
        user_prompt = loaded_prompts["user"]
        tools = manifest["planner_tools"]
        if not isinstance(tools, list) or any(name != "read_image" for name in tools):
            raise _fail("planner_tools must be an array containing only 'read_image'")
        if len(tools) != len(set(tools)):
            raise _fail("planner_tools must not contain duplicates")
        planner_tools = tuple(tools)
        collisions = set(planner_tools) & (
            set(native_tools) | {skill.name for skill in skills if skill.tool_source is not None}
        )
        if collisions:
            raise _fail(f"planner_tools collide with exposed package tools: {sorted(collisions)}")
        policy_entry = manifest.get("policy")
        if policy_entry is not None:
            if not isinstance(policy_entry, str):
                raise _fail("policy must be '<relative.py>:<class>' or null")
            filename, separator, class_name = policy_entry.partition(":")
            if not separator or not class_name.isidentifier() or not filename.endswith(".py"):
                raise _fail("policy must be '<relative.py>:<class>'")
            path = _package_path(root, filename, label="policy")
            if any(not part.isidentifier() for part in PurePosixPath(filename).with_suffix("").parts):
                raise _fail("policy path must use valid Python module names")
            _validate_regular_file(path, label="policy")
            sources["policy"] = policy_entry
        memory = manifest.get("memory")
        if memory is not None:
            if not isinstance(memory, dict):
                raise _fail("memory must be an object or null")
            _require_exact_keys(memory, {"path", "episode_notes"}, "memory")
            memory_path = _package_path(root, memory["path"], label="memory.path")
            _validate_directory(memory_path, label="memory.path")
            if not isinstance(memory["episode_notes"], bool):
                raise _fail("memory.episode_notes must be boolean")
            episode_notes = memory["episode_notes"]
            sources["memory"] = memory["path"]
        resources_entry = manifest.get("resources")
        if resources_entry is not None:
            path = _package_path(root, resources_entry, label="resources")
            _validate_regular_file(path, label="resources")
            text, _ = _read_text(path, label="resources")
            try:
                resources = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise _fail(f"invalid resources JSON: {exc}") from exc
            if not isinstance(resources, dict):
                raise _fail("resources must contain one JSON object")
            sources["resources"] = resources_entry
    for skill in skills:
        sources[f"skills.{skill.name}.instructions"] = f"skills/{skill.name}/SKILL.md"
        if skill.tool_source is not None:
            sources[f"skills.{skill.name}.tool"] = f"skills/{skill.name}/tool.py"
    return HarnessPackage(
        root=root,
        identity=HarnessIdentity(harness_id=harness_id, schema_version=version, commit=_git_commit(root, commit)),
        native_tools=native_tools,
        skills=skills,
        instructions=(
            _build_instructions(harness_id, skills)
            if version == 1
            else "\n\n".join(f"### Skill: {skill.name}\n\n{skill.instructions}" for skill in skills)
        ),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        planner_tools=planner_tools,
        policy_entry=policy_entry,
        memory_path=memory_path,
        episode_notes=episode_notes,
        resources=MappingProxyType(resources),
        source_paths=MappingProxyType(sources),
        files=MappingProxyType(files),
    )
