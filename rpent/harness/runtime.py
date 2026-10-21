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

"""Bind a validated harness to a live Toolkit without widening its authority."""

from __future__ import annotations

import copy
import inspect
import importlib
import importlib.abc
import importlib.util
import itertools
import json
import threading
import sys
import tempfile
import traceback
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from rpent.harness.contracts import (
    BoundHarness,
    CapabilityBinding,
    CapabilitySpec,
    HarnessError,
    HarnessPackage,
    PlannerRequest,
    ToolDecision,
    ToolBinding,
)
from rpent.harness.package import load_harness
from rpent.tools.toolkit import ToolResult

_Resolver = Callable[[Mapping[str, Any]], dict[str, Any]]


def _result_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, ToolResult):
        return dict(value.result)
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _agent_safe_result(value: Any) -> Any:
    """Remove capability-local arrays if a Skill accidentally returns them."""
    if isinstance(value, Mapping):
        return {
            str(key): _agent_safe_result(item)
            for key, item in value.items()
            if not (str(key).startswith("_") and str(key).endswith("_array"))
        }
    if isinstance(value, (list, tuple)):
        return [_agent_safe_result(item) for item in value]
    return value


class SkillRuntime:
    """The only authority passed to ``tool.py``.

    Aliases are the keys declared by that skill's ``requires`` object. The
    runtime is deliberately inactive while ``build(runtime)`` executes, so a
    builder cannot move the robot as a side effect of harness activation.
    """

    def __init__(
        self, resolvers: Mapping[str, _Resolver], *, state: dict[str, Any] | None = None
    ) -> None:
        self._resolvers = dict(resolvers)
        self._active = False
        self.state = state if state is not None else {}

    def _activate(self) -> None:
        self._active = True

    def call(
        self,
        alias: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke one declared dependency by its local alias."""
        if not self._active:
            raise HarnessError("skill dependencies cannot be called during build()")
        resolver = self._resolvers.get(alias)
        if resolver is None:
            raise HarnessError(f"skill attempted undeclared dependency: {alias!r}")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise HarnessError("runtime.call arguments must be a mapping")
        return resolver(dict(arguments))


def _invoke_tool(binding: ToolBinding, arguments: Mapping[str, Any]) -> Any:
    # ToolResult carries actual image content blocks; reducing it to a result
    # dict can silently turn a custom multimodal response into plain text.
    return binding.handler(**dict(arguments))


def _has_local_arrays(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (str(key).startswith("_") and str(key).endswith("_array"))
            or _has_local_arrays(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_local_arrays(item) for item in value)
    return False


_MODULE_IDS = itertools.count()


class _PackageModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Ordinary relative imports from a snapshot, isolated to one episode.

    Unique module names isolate helper globals as well as the Policy instance.
    Loading from retained bytes also avoids rewriting a shared Git snapshot with
    Python bytecode and keeps source paths usable in tracebacks and inspection.
    """

    def __init__(self, package: HarnessPackage) -> None:
        self.package = package
        self.prefix = f"_rpent_harness_episode_{next(_MODULE_IDS)}"
        self._closed = False
        self._files = dict(package.files)
        self._directories = {""}
        for filename in self._files:
            parent = Path(filename).parent
            while parent != Path("."):
                self._directories.add(parent.as_posix())
                parent = parent.parent
        sys.meta_path.insert(0, self)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self.prefix and not fullname.startswith(self.prefix + "."):
            return None
        relative = fullname[len(self.prefix):].lstrip(".").replace(".", "/")
        filename = f"{relative}/__init__.py" if relative else "__init__.py"
        is_package = relative in self._directories
        if not is_package:
            filename = relative + ".py"
            if filename not in self._files:
                return None
        origin = self.package.root / filename
        spec = importlib.util.spec_from_loader(fullname, self, origin=str(origin), is_package=is_package)
        spec.loader_state = filename if filename in self._files else None
        spec.has_location = True
        if is_package:
            spec.submodule_search_locations = [str(self.package.root / relative)]
        return spec

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        filename = module.__spec__.loader_state
        if filename is not None:
            exec(compile(self._files[filename], str(self.package.root / filename), "exec"), module.__dict__)

    def load(self, filename: str) -> ModuleType:
        name = filename.removesuffix(".py").replace("/", ".")
        if name == "__init__":
            name = ""
        elif name.endswith(".__init__"):
            name = name.removesuffix(".__init__")
        return importlib.import_module(self.prefix + ("." + name if name else ""))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for name in tuple(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + "."):
                sys.modules.pop(name, None)


class PolicyRuntime:
    """Three package hooks and the state owned by one live episode."""

    def __init__(
        self,
        package: HarnessPackage,
        *,
        modules: _PackageModules,
        episode_dir: Path | str | None = None,
    ) -> None:
        self._modules = modules
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.memory_dir = package.memory_path
        self.notes_dir: Path | None = None
        if package.episode_notes:
            if episode_dir is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="rpent-harness-episode-")
                episode_dir = self._temporary.name
            episode_root = Path(episode_dir).resolve()
            if episode_root == package.root or package.root in episode_root.parents:
                raise HarnessError("episode notes must be outside the published Harness package")
            self.notes_dir = episode_root / "notes"
            self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = {
            "package_root": str(package.root),
            "memory_dir": str(self.memory_dir) if self.memory_dir else None,
            "notes_dir": str(self.notes_dir) if self.notes_dir else None,
        }
        self.source_map: dict[str, Any] = {
            "package_root": str(package.root),
            "commit": package.identity.commit,
            **dict(package.source_paths),
            "helpers": [
                path for path in package.files
                if path.endswith(".py")
                and path not in {
                    entry.partition(":")[0] for entry in package.source_paths.values()
                }
            ],
        }
        self.capability_specs: list[dict[str, Any]] = []
        self.policy: Any = None
        if package.policy_entry is not None:
            filename, _, class_name = package.policy_entry.partition(":")
            try:
                module = modules.load(filename)
                policy_type = getattr(module, class_name)
                self.policy = policy_type()
            except Exception as exc:
                self.close()
                raise HarnessError(f"failed to instantiate {package.policy_entry}: {exc}") from exc
            for hook in ("prepare_request", "process_tool_result", "decide"):
                method = getattr(self.policy, hook, None)
                if method is not None and (not callable(method) or inspect.iscoroutinefunction(method)):
                    self.close()
                    raise HarnessError(f"Policy.{hook} must be a synchronous callable")

    def prepare_request(self, request: PlannerRequest) -> PlannerRequest:
        if not isinstance(request, PlannerRequest):
            raise HarnessError("prepare_request expects PlannerRequest")
        method = getattr(self.policy, "prepare_request", None)
        if method is None:
            return request
        visible = {spec["name"]: copy.deepcopy(spec) for spec in request.tools}
        settings = copy.deepcopy(request.model_settings)
        result = method(copy.deepcopy(request), self.state)
        if not isinstance(result, PlannerRequest):
            raise HarnessError("Policy.prepare_request must return PlannerRequest")
        if not isinstance(result.messages, list) or any(not isinstance(item, dict) for item in result.messages):
            raise HarnessError("Policy request.messages must be a list of message dicts")
        if result.instructions is not None and not isinstance(result.instructions, str):
            raise HarnessError("Policy request.instructions must be text or None")
        if result.model_settings != settings:
            raise HarnessError("Policy cannot change host model settings or budgets")
        if not isinstance(result.tools, list):
            raise HarnessError("Policy request.tools must be a list")
        seen: set[str] = set()
        for spec in result.tools:
            name = spec.get("name") if isinstance(spec, dict) else None
            if not isinstance(name, str) or name in seen or name not in visible or spec != visible[name]:
                raise HarnessError("Policy may only select a subset of the bound tools without changing their schemas")
            seen.add(name)
        return result

    def process_tool_result(self, call: dict[str, Any], raw_result: Any) -> Any:
        method = getattr(self.policy, "process_tool_result", None)
        if method is None:
            return raw_result
        # Policy changes cannot overwrite the raw evidence retained by RPent.
        result = method(copy.deepcopy(call), copy.deepcopy(raw_result), self.state)
        if not isinstance(result, (ToolResult, Mapping)):
            raise HarnessError("Policy.process_tool_result must return ToolResult or a result mapping")
        return result

    def decide(self, request: PlannerRequest) -> ToolDecision | None:
        method = getattr(self.policy, "decide", None)
        if method is None:
            return None
        result = method(copy.deepcopy(request), self.state)
        if result is not None and not isinstance(result, ToolDecision):
            raise HarnessError("Policy.decide must return ToolDecision or None")
        if result is not None:
            if result.tool_name not in {spec["name"] for spec in request.tools}:
                raise HarnessError("Policy decision names a tool outside this request's visible menu")
            if not isinstance(result.args, dict):
                raise HarnessError("Policy decision args must be a dict")
            if result.tool_call_id is not None and (not isinstance(result.tool_call_id, str) or not result.tool_call_id):
                raise HarnessError("Policy decision tool_call_id must be non-empty text or None")
        return result

    def close(self) -> None:
        self._modules.close()
        if self._temporary is not None:
            self._temporary.cleanup()


def inspect_harness(bound: BoundHarness) -> dict[str, Any]:
    """Describe an already loaded/bound package without calling its tools.

    The tool table is built by the same SDK adapter as the Planner, including
    ``read_image``. This is an inspectable binding preview; the request journal
    remains authoritative for history, Policy-selected subsets and sent bytes.
    """
    from rpent.planner.api_loop import _build_tools

    package = bound.package
    runtime = bound.policy_runtime
    tools = _build_tools(bound.toolkit)
    return {
        "kind": "bound_harness_inspection",
        "loaded": True,
        "actual_request": False,
        "identity": {
            "id": package.identity.harness_id,
            "schema_version": package.identity.schema_version,
            "commit": package.identity.commit,
        },
        "root": str(package.root),
        "prompts": {
            name: {"source": package.source_paths.get(f"prompts.{name}"), "text": text}
            for name, text in (("system", package.system_prompt), ("user", package.user_prompt))
        },
        "instructions": bound.instructions,
        "skills": [
            {"name": skill.name, "requires": dict(skill.requires), "instructions": skill.instructions}
            for skill in package.skills
        ],
        "native_tools": list(package.native_tools),
        "planner_tools": list(package.planner_tools),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": copy.deepcopy(tool.function_schema.json_schema),
            }
            for tool in tools
        ],
        "capabilities": copy.deepcopy(runtime.capability_specs),
        "memory": {
            "path": str(bound.memory_dir) if bound.memory_dir else None,
            "read_only": True,
            "episode_notes": package.episode_notes,
            "notes_dir": str(bound.notes_dir) if bound.notes_dir else None,
        },
        "policy": {
            "entry": package.policy_entry,
            "hooks": [
                name for name in ("prepare_request", "process_tool_result", "decide")
                if callable(getattr(runtime.policy, name, None))
            ],
        },
        "resources": copy.deepcopy(dict(package.resources)),
        "sources": copy.deepcopy(runtime.source_map),
        "files": list(package.files),
    }


def _load_tool_binding(
    skill_name: str, runtime: SkillRuntime, modules: _PackageModules,
) -> ToolBinding:
    try:
        module = modules.load(f"skills/{skill_name}/tool.py")
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise HarnessError(f"failed to load {skill_name}/tool.py: {exc}") from exc
    build = getattr(module, "build", None)
    if not callable(build):
        raise HarnessError(f"{skill_name}/tool.py must define callable build(runtime)")
    if inspect.iscoroutinefunction(build):
        raise HarnessError(f"{skill_name}/tool.py build(runtime) must be synchronous")
    try:
        binding = build(runtime)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise HarnessError(f"{skill_name}/tool.py build(runtime) failed: {exc}") from exc
    if not isinstance(binding, ToolBinding):
        raise HarnessError(f"{skill_name}/tool.py build(runtime) must return ToolBinding")
    return binding


def _validated_tool_binding(skill_name: str, binding: ToolBinding) -> ToolBinding:
    if binding.name != skill_name:
        raise HarnessError(
            f"skill {skill_name!r} must return ToolBinding(name={skill_name!r})"
        )
    if not isinstance(binding.description, str) or not binding.description.strip():
        raise HarnessError(f"tool {skill_name!r} must have a non-empty description")
    if not isinstance(binding.input_schema, Mapping):
        raise HarnessError(f"tool {skill_name!r} input_schema must be an object")
    schema = dict(binding.input_schema)
    if schema.get("type") != "object":
        raise HarnessError(f"tool {skill_name!r} input_schema.type must be 'object'")
    try:
        schema = json.loads(json.dumps(schema))
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            f"tool {skill_name!r} input_schema must be JSON serializable"
        ) from exc
    if not callable(binding.handler) or inspect.iscoroutinefunction(binding.handler):
        raise HarnessError(f"tool {skill_name!r} handler must be synchronous callable")
    return ToolBinding(
        name=binding.name,
        description=binding.description.strip(),
        input_schema=schema,
        handler=binding.handler,
    )


def _base_specs(toolkit: Any) -> dict[str, dict[str, Any]]:
    try:
        raw_specs = toolkit.get_tools_spec()
    except Exception as exc:
        raise HarnessError(f"cannot inspect base toolkit: {exc}") from exc
    if not isinstance(raw_specs, list):
        raise HarnessError("base toolkit get_tools_spec() must return a list")
    specs: dict[str, dict[str, Any]] = {}
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, Mapping):
            raise HarnessError(f"base tool spec {index} must be an object")
        spec = copy.deepcopy(dict(raw_spec))
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise HarnessError(f"base tool spec {index} has no valid name")
        if name in specs:
            raise HarnessError(f"base toolkit has duplicate tool name: {name!r}")
        specs[name] = spec
    return specs


def _validate_capability_spec(spec: CapabilitySpec) -> None:
    if not isinstance(spec.id, str) or not spec.id:
        raise HarnessError("capability spec id must be a non-empty string")
    if not isinstance(spec.description, str) or not spec.description.strip():
        raise HarnessError(f"capability {spec.id!r} must have a description")
    for field_name, schema in (
        ("input_schema", spec.input_schema),
        ("output_schema", spec.output_schema),
    ):
        if not isinstance(schema, Mapping):
            raise HarnessError(f"capability {spec.id!r} {field_name} must be an object")
        try:
            json.dumps(dict(schema))
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                f"capability {spec.id!r} {field_name} must be JSON serializable"
            ) from exc


class HarnessToolkitView:
    """Agent-facing Toolkit view for one bound harness.

    Native calls delegate to the base Toolkit. Skill handlers execute outside
    the base Toolkit's operation lock; only their capability subcalls enter the
    base Toolkit. This is what permits a composite tool to make several
    sequential robot calls without triggering a nested-lock rejection.
    """

    def __init__(
        self,
        *,
        base_toolkit: Any,
        native_specs: tuple[dict[str, Any], ...],
        skill_tools: tuple[ToolBinding, ...],
        episode_active: Callable[[], bool] | None = None,
        allow_image_reader: bool = False,
        harness_runtime: PolicyRuntime | None = None,
    ) -> None:
        self._base_toolkit = base_toolkit
        self._native_specs = native_specs
        self._native_names = frozenset(spec["name"] for spec in native_specs)
        self._skill_tools = {binding.name: binding for binding in skill_tools}
        self._episode_active = episode_active
        self._allow_image_reader = allow_image_reader
        self.harness_runtime = harness_runtime
        # Pydantic AI may execute several tool calls from one model response in
        # parallel.  A composite Harness tool is a transaction from the point
        # of view of the robot: letting another top-level tool interleave with
        # its capability subcalls makes controller-local state race and causes
        # the base Toolkit to reject overlapping operations.  Dependency tools
        # do not re-enter this method (SkillRuntime invokes their binding
        # directly), so a plain non-reentrant lock preserves composition without
        # creating the nested base-Toolkit lock that Harness binding avoids.
        self._top_level_operation_lock = threading.Lock()

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return only explicitly exposed native tools and harness tools."""
        specs = [copy.deepcopy(spec) for spec in self._native_specs]
        specs.extend(
            {
                "name": binding.name,
                "description": binding.description,
                "input_schema": copy.deepcopy(dict(binding.input_schema)),
            }
            for binding in self._skill_tools.values()
        )
        return specs

    @property
    def allow_image_reader(self) -> bool:
        """Whether the package explicitly selected the SDK image reader."""
        return self._allow_image_reader

    def episode_active(self) -> bool | None:
        """Return only action availability, never a terminal reason or outcome."""
        if self._episode_active is None:
            return None
        active = self._episode_active()
        if not isinstance(active, bool):
            raise HarnessError("Harness episode_active provider must return bool")
        return active

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch an exposed native or skill tool."""
        with self._top_level_operation_lock:
            if not isinstance(input_dict, Mapping):
                return ToolResult(
                    name=name,
                    result={"error": f"bad arguments for {name}: expected an object"},
                )
            if name in self._native_names:
                return self._base_toolkit.execute_tool(name, dict(input_dict))
            binding = self._skill_tools.get(name)
            if binding is None:
                return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
            try:
                result = _invoke_tool(binding, input_dict)
            except TypeError as exc:
                result = {
                    "error": f"bad arguments for {name}: {exc}",
                    "got": dict(input_dict),
                }
            except Exception as exc:
                result = {"error": str(exc), "traceback": traceback.format_exc()}
            if isinstance(result, ToolResult):
                # Preserve explicitly produced multimodal blocks. Standard
                # ToolResults already keep image bytes out of their text block.
                safe = copy.copy(result)
                safe.name = name
                safe.result = _agent_safe_result(result.result)
                safe.content_blocks = copy.deepcopy(result.content_blocks)
                if _has_local_arrays(result.result):
                    text = ToolResult(name=name, result=safe.result).content_blocks[0]
                    safe.content_blocks = [text] + [
                        block for block in safe.content_blocks if block.get("type") != "text"
                    ]
                return safe
            return ToolResult(name=name, result=_agent_safe_result(_result_dict(result)))

    def close(self) -> None:
        try:
            self._base_toolkit.close()
        finally:
            if self.harness_runtime is not None:
                self.harness_runtime.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate episode lifecycle operations to the base Toolkit."""
        return getattr(self._base_toolkit, name)


def bind_harness(
    harness: HarnessPackage | Path | str,
    *,
    toolkit: Any,
    capabilities: Mapping[str, CapabilityBinding],
    allowed_native_tools: Collection[str] | None = None,
    episode_active: Callable[[], bool] | None = None,
    allow_image_reader: bool = False,
    episode_dir: Path | str | None = None,
    commit: str | None = None,
) -> BoundHarness:
    """Validate, resolve, and bind a harness to one episode Toolkit.

    Args:
        harness: A directory path or a snapshot returned by
            :func:`load_harness`.
        toolkit: Live robot Toolkit to wrap.
        capabilities: Exact capability-id to live binding map supplied by the
            robot adapter.
        allowed_native_tools: Host-enforced base-tool ceiling. A Harness cannot
            grant itself a native tool outside this set. ``None`` preserves the
            legacy unrestricted binder for callers that do not execute generated
            Harnesses; production robot hosts should always provide the set.
        episode_active: Optional host callback exposing only whether another
            environment action is available. It does not expose success or the
            terminal reason.
        allow_image_reader: Legacy schema-1 compatibility; schema 2 always
            follows its explicit planner_tools declaration.
        episode_dir: Episode-owned output root for writable notes.
        commit: Frozen Git identity, when passing a directory archive.

    Returns:
        Instructions plus an agent-facing Toolkit view.

    Raises:
        HarnessError: If any dependency or generated tool is invalid.
    """
    package = harness if isinstance(harness, HarnessPackage) else load_harness(harness, commit=commit)
    if isinstance(harness, HarnessPackage) and commit is not None and commit != harness.identity.commit:
        raise HarnessError("commit differs from the loaded Harness identity")
    if allowed_native_tools is not None:
        allowed = frozenset(allowed_native_tools)
        forbidden = sorted(set(package.native_tools) - allowed)
        if forbidden:
            raise HarnessError(
                f"harness requests native tools forbidden by its host: {forbidden}"
            )
    if not isinstance(capabilities, Mapping):
        raise HarnessError("capabilities must be a mapping")
    capability_map: dict[str, CapabilityBinding] = {}
    for capability_id, binding in capabilities.items():
        if not isinstance(capability_id, str) or not isinstance(
            binding, CapabilityBinding
        ):
            raise HarnessError("capabilities must map ids to CapabilityBinding")
        if binding.capability_id != capability_id:
            raise HarnessError(
                f"capability key {capability_id!r} does not match binding id "
                f"{binding.capability_id!r}"
            )
        _validate_capability_spec(binding.spec)
        if not callable(binding.invoke):
            raise HarnessError(f"capability {capability_id!r} is not callable")
        capability_map[capability_id] = binding

    base_specs = _base_specs(toolkit)
    missing_native = [name for name in package.native_tools if name not in base_specs]
    if missing_native:
        raise HarnessError(f"base toolkit is missing native tools: {missing_native}")

    modules = _PackageModules(package)
    policy_runtime: PolicyRuntime | None = None
    try:
        policy_runtime = PolicyRuntime(package, modules=modules, episode_dir=episode_dir)
        policy_runtime.capability_specs = [
            {
                "id": cap.spec.id,
                "description": cap.spec.description,
                "input_schema": copy.deepcopy(dict(cap.spec.input_schema)),
                "output_schema": copy.deepcopy(dict(cap.spec.output_schema)),
            }
            for cap in capability_map.values()
        ]
        return _bind_episode(package, toolkit, capability_map, base_specs, modules,
                             episode_active, allow_image_reader, policy_runtime)
    except BaseException:
        if policy_runtime is not None:
            policy_runtime.close()
        else:
            modules.close()
        raise


def _bind_episode(
    package: HarnessPackage,
    toolkit: Any,
    capability_map: Mapping[str, CapabilityBinding],
    base_specs: dict[str, dict[str, Any]],
    modules: _PackageModules,
    episode_active: Callable[[], bool] | None,
    allow_image_reader: bool,
    policy_runtime: PolicyRuntime,
) -> BoundHarness:
    tool_bindings: dict[str, ToolBinding] = {}
    for skill in package.skills:
        resolvers: dict[str, _Resolver] = {}
        for alias, requirement in skill.requires.items():
            kind, _, target = requirement.partition(":")
            if kind == "cap":
                capability = capability_map.get(target)
                if capability is None:
                    raise HarnessError(
                        f"skill {skill.name!r} requires unavailable capability "
                        f"{target!r}"
                    )
                resolvers[alias] = lambda arguments, cap=capability: _result_dict(
                    cap.invoke(arguments)
                )
            else:
                dependency = tool_bindings.get(target)
                if dependency is None:
                    raise HarnessError(
                        f"skill {skill.name!r} requires unavailable tool {target!r}"
                    )
                resolvers[alias] = lambda arguments, tool=dependency: _result_dict(_invoke_tool(
                    tool, arguments
                ))

        if skill.tool_source is None:
            continue
        runtime = SkillRuntime(resolvers, state=policy_runtime.state)
        binding = _load_tool_binding(skill.name, runtime, modules)
        binding = _validated_tool_binding(skill.name, binding)
        # A Harness tool may deliberately replace a *hidden* native tool at the
        # agent surface.  Its handler still receives only the capabilities
        # declared by this skill, so sharing a name does not grant access to the
        # hidden Toolkit implementation.  Only an actually exposed native name
        # is ambiguous and therefore forbidden.
        if binding.name in package.native_tools:
            raise HarnessError(
                f"skill tool {binding.name!r} collides with an exposed native tool"
            )
        runtime._activate()
        tool_bindings[binding.name] = binding

    native_specs = tuple(base_specs[name] for name in package.native_tools)
    view = HarnessToolkitView(
        base_toolkit=toolkit,
        native_specs=native_specs,
        skill_tools=tuple(tool_bindings.values()),
        episode_active=episode_active,
        allow_image_reader=("read_image" in package.planner_tools if package.identity.schema_version == 2 else allow_image_reader),
        harness_runtime=policy_runtime,
    )
    return BoundHarness(
        identity=package.identity,
        instructions=package.instructions,
        toolkit=view,
        package=package,
        policy_runtime=policy_runtime,
        memory_dir=policy_runtime.memory_dir,
        notes_dir=policy_runtime.notes_dir,
    )
