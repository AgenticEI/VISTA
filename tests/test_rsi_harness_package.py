"""Schema-2 package contracts across real Git and the real tool binder."""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rpent.harness import (
    CapabilityBinding, CapabilitySpec, HarnessError, PlannerRequest, ToolDecision,
    bind_harness, inspect_harness, load_harness,
)
from rpent.tools.toolkit import ToolResult


class Toolkit:
    state = None

    def __init__(self):
        self.closed = False

    def get_tools_spec(self):
        return [{"name": "finish", "description": "Finish.", "input_schema": {"type": "object"}}]

    def execute_tool(self, name, arguments):
        return ToolResult(name=name, result={"_finish": name == "finish"})

    def close(self):
        self.closed = True


def package(root: Path, *, policy: str | None = None, skill: str | None = None) -> Path:
    root.mkdir()
    (root / "prompts").mkdir()
    (root / "prompts/system.md").write_text("System exactly.\n")
    (root / "prompts/user.md").write_text("Task: {{task}}\n")
    (root / "memory").mkdir()
    (root / "memory/experience.md").write_text("Shared fixed experience.")
    (root / "resources.json").write_text(json.dumps({"vla": {"path": "/external/model"}}))
    manifest = {
        "schema_version": 2, "id": "any-name",
        "prompts": {"system": "prompts/system.md", "user": "prompts/user.md"},
        "native_tools": ["finish"], "planner_tools": ["read_image"], "skills": [],
        "memory": {"path": "memory", "episode_notes": True}, "resources": "resources.json",
    }
    if policy:
        (root / "policy.py").write_text(policy)
        manifest["policy"] = "policy.py:Policy"
    if skill:
        (root / "skills/inspect").mkdir(parents=True)
        (root / "skills/inspect/SKILL.md").write_text("Inspect with images.")
        (root / "skills/inspect/tool.py").write_text(skill)
        manifest["skills"].append({"name": "inspect", "requires": {}})
    (root / "HARNESS.json").write_text(json.dumps(manifest))
    return root


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()


def request() -> PlannerRequest:
    return PlannerRequest(messages=[], instructions="System.", model_settings={"temperature": 0}, tools=Toolkit().get_tools_spec())


def test_loads_empty_skills_prompts_resources_and_explicit_identity(tmp_path):
    root = package(tmp_path / "package")
    loaded = load_harness(root, commit="a" * 40)
    assert loaded.identity.commit == "a" * 40
    assert not hasattr(loaded.identity, "content_hash")
    assert loaded.system_prompt == "System exactly.\n"
    assert loaded.user_prompt == "Task: {{task}}\n"
    assert loaded.instructions == ""
    assert loaded.skills == ()
    assert loaded.planner_tools == ("read_image",)
    assert loaded.resources["vla"]["path"] == "/external/model"
    assert loaded.source_paths["prompts.system"] == "prompts/system.md"
    bound = bind_harness(loaded, toolkit=Toolkit(), capabilities={})
    try:
        assert bound.toolkit.allow_image_reader
        assert bound.memory_dir == root / "memory"
        assert bound.notes_dir.is_dir()
        assert bound.notes_dir != root / "notes"
    finally:
        bound.close()


def test_real_git_worktree_and_dirty_draft_identity(tmp_path):
    root = package(tmp_path / "git-package")
    git(root, "init")
    git(root, "add", ".")
    git(root, "-c", "user.name=Harness Test", "-c", "user.email=harness@test.invalid", "commit", "-m", "H0")
    commit = git(root, "rev-parse", "HEAD")
    worktree = tmp_path / "worktree"
    git(root, "worktree", "add", "--detach", str(worktree), commit)
    assert (worktree / ".git").is_file()
    assert load_harness(worktree).identity.commit == commit
    (worktree / "prompts/user.md").write_text("Unfrozen change")
    assert load_harness(worktree).identity.commit is None


POLICY = '''
from .lib.counter import counter
from rpent.harness import ToolDecision
from rpent.tools.toolkit import ToolResult

class Policy:
    def prepare_request(self, request, state):
        counter.append("request")
        state["count"] = len(counter)
        request.instructions += " package addition"
        return request

    def process_tool_result(self, call, raw, state):
        raw.result["feedback"] = call["tool_name"]
        return ToolResult(name=raw.name, result=raw.result)

    def decide(self, request, state):
        return ToolDecision("finish", {}) if state.get("stop") else None
'''

TOOL = '''
from ...lib.counter import counter
from rpent.harness import ToolBinding
from rpent.tools.toolkit import ToolResult

def build(runtime):
    def inspect():
        counter.append("tool")
        return ToolResult(name="inspect", result={"count": len(counter), "_image_bytes": b"PNG image bytes"})
    return ToolBinding("inspect", "Inspect.", {"type": "object"}, inspect)
'''


def test_relative_imports_snapshot_and_episode_state_are_isolated(tmp_path):
    root = package(tmp_path / "package", policy=POLICY, skill=TOOL)
    (root / "lib").mkdir()
    (root / "lib/counter.py").write_text("counter = []\n")
    # Additional ordinary files inside a skill are legal, without a whitelist.
    (root / "skills/inspect/notes.md").write_text("Local helper notes")
    loaded = load_harness(root)
    (root / "lib/counter.py").write_text("raise RuntimeError('edited after load')\n")
    first = bind_harness(loaded, toolkit=Toolkit(), capabilities={}, episode_dir=tmp_path / "one")
    second = bind_harness(loaded, toolkit=Toolkit(), capabilities={}, episode_dir=tmp_path / "two")
    try:
        assert first.policy_runtime.policy is not second.policy_runtime.policy
        assert first.state is not second.state
        assert first.notes_dir != second.notes_dir
        first.policy_runtime.prepare_request(request())
        assert first.state["count"] == 1
        assert "count" not in second.state
        assert first.toolkit.execute_tool("inspect", {}).result["count"] == 2
        assert second.toolkit.execute_tool("inspect", {}).result["count"] == 1
        first.notes_dir.joinpath("lesson.md").write_text("Only this episode")
        assert not second.notes_dir.joinpath("lesson.md").exists()
        assert (root / "memory/experience.md").read_text() == "Shared fixed experience."
        assert not list(root.rglob("__pycache__"))
    finally:
        first.close()
        second.close()


def test_policy_feedback_keeps_raw_evidence_and_multimodal_payload(tmp_path):
    root = package(tmp_path / "package", policy=POLICY, skill=TOOL)
    (root / "lib").mkdir()
    (root / "lib/counter.py").write_text("counter = []\n")
    toolkit = Toolkit()
    bound = bind_harness(root, toolkit=toolkit, capabilities={})
    module_prefix = type(bound.policy_runtime.policy).__module__.split(".")[0]
    try:
        original = request()
        prepared = bound.policy_runtime.prepare_request(original)
        assert prepared.instructions == "System. package addition"
        assert original.instructions == "System."
        assert bound.policy_runtime.decide(prepared) is None
        bound.state["stop"] = True
        assert bound.policy_runtime.decide(prepared) == ToolDecision("finish", {})
        raw = bound.toolkit.execute_tool("inspect", {})
        feedback = bound.policy_runtime.process_tool_result({"tool_name": "inspect"}, raw)
        assert "feedback" not in raw.result
        assert feedback.result["feedback"] == "inspect"
        assert json.loads(feedback.content_blocks[0]["text"])["feedback"] == "inspect"
        assert feedback.content_blocks[1] == raw.content_blocks[1]
        assert base64.b64decode(raw.content_blocks[1]["source"]["data"]) == b"PNG image bytes"
        assert "PNG image bytes" not in raw.content_blocks[0]["text"]
    finally:
        bound.toolkit.close()
    assert toolkit.closed
    assert not any(name == module_prefix or name.startswith(module_prefix + ".") for name in sys.modules)


@pytest.mark.parametrize("mutation", [
    "request.tools[0]['input_schema']['properties'] = {'secret': {}}",
    "request.tools.append({'name': 'hidden', 'description': 'Hidden', 'input_schema': {}})",
    "request.model_settings['temperature'] = 1",
])
def test_policy_cannot_redefine_bound_schema_or_fixed_model_settings(tmp_path, mutation):
    source = "class Policy:\n    def prepare_request(self, request, state):\n        " + mutation + "\n        return request\n"
    root = package(tmp_path / "package", policy=source)
    bound = bind_harness(root, toolkit=Toolkit(), capabilities={})
    try:
        with pytest.raises(HarnessError):
            bound.policy_runtime.prepare_request(request())
    finally:
        bound.close()


def test_schema2_reader_is_manifest_selected_and_no_name_special_case(tmp_path):
    root = package(tmp_path / "package")
    manifest = json.loads((root / "HARNESS.json").read_text())
    manifest["id"] = "hvla_h0"
    manifest["planner_tools"] = []
    (root / "HARNESS.json").write_text(json.dumps(manifest))
    bound = bind_harness(root, toolkit=Toolkit(), capabilities={}, allow_image_reader=True)
    try:
        assert not bound.toolkit.allow_image_reader
    finally:
        bound.close()


@pytest.mark.parametrize("entry,value", [
    ("policy", "../outside.py:Policy"),
    ("resources", "../resources.json"),
    ("planner_tools", ["hidden"]),
])
def test_rejects_invalid_entry_paths_and_planner_tools(tmp_path, entry, value):
    root = package(tmp_path / "package")
    manifest = json.loads((root / "HARNESS.json").read_text())
    manifest[entry] = value
    (root / "HARNESS.json").write_text(json.dumps(manifest))
    with pytest.raises(HarnessError):
        load_harness(root)


def test_cannot_create_episode_notes_in_published_package(tmp_path):
    root = package(tmp_path / "package")
    with pytest.raises(HarnessError, match="outside"):
        bind_harness(root, toolkit=Toolkit(), capabilities={}, episode_dir=root / "runtime")
    assert not (root / "runtime").exists()


def test_inspection_reports_real_binding_without_executing_or_changing_state(tmp_path):
    root = package(tmp_path / "package", policy=POLICY, skill=TOOL)
    (root / "lib").mkdir()
    (root / "lib/counter.py").write_text("counter = []\n")

    def never_call(arguments):
        raise AssertionError("inspection called a robot capability")

    capability = CapabilityBinding(
        CapabilitySpec("control.fixture@1", "Fixture capability", {"type": "object"}, {"type": "object"}),
        never_call,
    )
    bound = bind_harness(root, toolkit=Toolkit(), capabilities={capability.spec.id: capability})
    try:
        before = dict(bound.state)
        inspected = inspect_harness(bound)
        assert inspected["loaded"] and inspected["actual_request"] is False
        assert [tool["name"] for tool in inspected["tools"]] == ["read_image", "finish", "inspect"]
        assert inspected["tools"][0]["input_schema"]["properties"]["step"]["default"] == -1
        assert inspected["capabilities"][0]["id"] == "control.fixture@1"
        assert inspected["memory"]["notes_dir"] == str(bound.notes_dir)
        assert inspected["policy"]["hooks"] == ["prepare_request", "process_tool_result", "decide"]
        assert "lib/counter.py" in inspected["sources"]["helpers"]
        assert bound.state == before
        json.dumps(inspected)
    finally:
        bound.close()
