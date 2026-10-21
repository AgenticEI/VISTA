# Copyright 2026 The RPent Authors.
"""Exercise actual SDK requests, HTTP retries, and one-turn manual takeover."""

from __future__ import annotations

import asyncio
import copy
import json

import httpx2
import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rpent.dashboard.events import NullDashboardEventSink
from rpent.harness.contracts import HarnessError, ToolDecision
from rpent.planner.api_loop import ApiAgentLoop, _build_tools
from rpent.planner.boundary import DecisionController, validate_history
from rpent.planner.recording import PlannerRecorder, current_decision_context
from rpent.tools.toolkit import ToolResult

PNG = b"\x89PNG\r\n\x1a\nfixture-pixels"


class Toolkit:
    allow_image_reader = False

    def __init__(self, runtime=None):
        self.harness_runtime = runtime
        self.calls = []
        self.provenance = []

    def get_tools_spec(self):
        return [
            {
                "name": name,
                "description": f"Fixture {name}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            }
            for name in ("observe", "hidden")
        ]

    def execute_tool(self, name, arguments):
        self.calls.append((name, arguments))
        self.provenance.append(current_decision_context())
        return ToolResult(
            name,
            {"raw_marker": "RAW", "value": arguments["value"], "_image_bytes": PNG},
        )

    def cancel_active_and_wait(self):
        pass


class PolicyRuntime:
    source_map = {
        "policy": "fixture/policy.py:Policy",
        "system": "fixture/prompts/system.md",
    }

    def __init__(self, *, program=False):
        self.program = program
        self.preparations = 0
        self.decisions = 0
        self.raw_seen = []

    def prepare_request(self, request):
        request = copy.deepcopy(request)
        self.preparations += 1
        request.instructions = "PACKAGE SYSTEM"
        request.tools = [tool for tool in request.tools if tool["name"] == "observe"]
        return request

    def process_tool_result(self, call, raw):
        self.raw_seen.append(raw.result)
        return ToolResult(
            raw.name,
            {"feedback_marker": "FEEDBACK", "_image_bytes": raw.result["_image_bytes"]},
        )

    def decide(self, request):
        self.decisions += 1
        if self.program and self.decisions == 1:
            return ToolDecision("observe", {"value": "7"})
        return None


def events(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "planner" / "events.jsonl").read_text().splitlines()
    ]


def completion(*, tool=False):
    message = {"role": "assistant", "content": "Done"}
    if tool:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "observe", "arguments": '{"value": 3}'},
                }
            ],
        }
    return {
        "id": "fixture-response",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3.8-max",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool else "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.mark.parametrize("model_name", ["qwen3.8-max", "Qwen3.8-Max"])
def test_policy_sdk_wire_retry_and_feedback_evidence(tmp_path, model_name):
    requests = []
    runtime = PolicyRuntime()
    toolkit = Toolkit(runtime)

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx2.Response(
                429,
                json={
                    "error": {"message": "fixture retry", "type": "rate_limit_error"}
                },
                headers={"retry-after-ms": "1"},
            )
        return httpx2.Response(200, json=completion(tool=len(requests) == 2))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
            provider = OpenAIProvider(
                base_url="https://fixture.invalid/v1",
                api_key="never-record-this",
                http_client=http,
            )
            planner = ApiAgentLoop(
                OpenAIChatModel(model_name, provider=provider),
                dashboard_events=NullDashboardEventSink(),
                output_dir=tmp_path,
            )
            return await planner._solve(
                system_prompt="HOST SYSTEM",
                user_message="Inspect.",
                toolkit=toolkit,
                max_turns=4,
            )

    result = asyncio.run(run())
    assert result.error is None
    assert len(requests) == 3
    assert runtime.preparations == 2
    assert toolkit.calls == [("observe", {"value": 3})]
    assert toolkit.provenance[0]["actor"] == "planner"
    assert requests[0]["messages"][0]["content"] == "PACKAGE SYSTEM"
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["observe"]
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[0]["model"] == model_name
    assert "FEEDBACK" in json.dumps(requests[2]) and "RAW" not in json.dumps(
        requests[2]
    )
    rows = events(tmp_path)
    sends = [row for row in rows if row["event"] == "http_request"]
    assert [(row["decision_id"], row["send_attempt"]) for row in sends] == [
        ("decision-000001", 1),
        ("decision-000001", 2),
        ("decision-000002", 1),
    ]
    assert [row["event"] for row in rows].index("sdk_request") < [
        row["event"] for row in rows
    ].index("http_request")
    assert "RAW" in json.dumps(next(row for row in rows if row["event"] == "tool_raw"))
    assert "FEEDBACK" in json.dumps(
        next(row for row in rows if row["event"] == "tool_feedback")
    )
    evidence_text = (tmp_path / "planner" / "events.jsonl").read_text()
    assert "never-record-this" not in evidence_text
    assert "data:image" not in evidence_text
    assert "sha256" not in evidence_text
    assert list((tmp_path / "planner" / "images").iterdir())[0].read_bytes() == PNG
    assert len(list((tmp_path / "planner" / "images").iterdir())) == 1
    assert result.stats["http_send_attempts"] == 3
    assert result.stats["autonomous"] is True


def test_manual_pause_replacement_validation_and_return_to_planner(tmp_path):
    controller = DecisionController(pause_next=True)
    toolkit = Toolkit(PolicyRuntime())
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=completion())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
            model = OpenAIChatModel(
                "qwen3.8-max",
                provider=OpenAIProvider(
                    base_url="https://fixture.invalid/v1",
                    api_key="fixture",
                    http_client=http,
                ),
            )
            planner = ApiAgentLoop(
                model,
                dashboard_events=NullDashboardEventSink(),
                output_dir=tmp_path,
                decision_controller=controller,
            )
            task = asyncio.create_task(
                planner._solve(
                    system_prompt="HOST",
                    user_message="Act.",
                    toolkit=toolkit,
                    max_turns=3,
                )
            )
            paused = await asyncio.to_thread(controller.wait_for_pause, 3)
            assert paused and paused["sent"] is False
            assert paused["request"].instructions == "PACKAGE SYSTEM"
            assert requests == [] and toolkit.calls == []
            with pytest.raises(HarnessError, match="not visible"):
                controller.replace(ToolDecision("hidden", {"value": 1}))
            with pytest.raises(HarnessError, match="invalid arguments"):
                controller.replace(ToolDecision("observe", {"value": "wrong"}))
            controller.replace(ToolDecision("observe", {"value": "9"}))
            return await task

    result = asyncio.run(run())
    assert result.error is None
    assert toolkit.calls == [("observe", {"value": 9})]
    assert toolkit.provenance[0]["actor"] == "meta"
    assert len(requests) == 1
    assert any(message["role"] == "tool" for message in requests[0]["messages"])
    rows = events(tmp_path)
    responses = [row for row in rows if row["event"] == "model_response"]
    assert [(row["actor"], row["sent"]) for row in responses] == [
        ("meta", False),
        ("planner", True),
    ]
    assert {row["decision_id"] for row in rows if row["event"] == "http_request"} == {
        "decision-000002"
    }
    assert result.stats["manual_decisions"] == 1
    assert result.stats["autonomous"] is False
    assert result.stats["requests"] == 1


def test_program_decision_has_no_fake_send_and_keeps_sdk_usage(tmp_path):
    requests = []

    def answer(messages, info):
        requests.append(messages)
        return ModelResponse(parts=[])

    toolkit = Toolkit(PolicyRuntime(program=True))
    planner = ApiAgentLoop(
        FunctionModel(answer),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path,
    )
    result = planner.solve(
        system_prompt="HOST", user_message="Act.", toolkit=toolkit, max_turns=2
    )
    assert result.error is None
    assert toolkit.calls == [("observe", {"value": 7})]
    assert toolkit.provenance[0]["actor"] == "harness_program"
    assert len(requests) == 1
    rows = events(tmp_path)
    first = next(row for row in rows if row["event"] == "model_response")
    assert first["actor"] == "harness_program" and first["sent"] is False
    assert not any(row["event"] == "http_request" for row in rows)
    assert result.stats["harness_program_decisions"] == 1
    assert result.stats["autonomous"] is True
    assert result.stats["requests"] == 1


def test_hidden_model_call_never_executes(tmp_path):
    requests = 0

    def answer(messages, info):
        nonlocal requests
        requests += 1
        return ModelResponse(
            parts=[
                ToolCallPart("hidden", {"value": 1}, tool_call_id=f"hidden-{requests}")
            ]
        )

    toolkit = Toolkit(PolicyRuntime())
    result = ApiAgentLoop(
        FunctionModel(answer),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path,
    ).solve(system_prompt="HOST", user_message="Act.", toolkit=toolkit, max_turns=3)
    assert toolkit.calls == []
    assert result.error is None
    assert requests == 3


def test_history_pair_validation_and_tools_are_sequential():
    request = ModelRequest(parts=[UserPromptPart("hello")])
    call = ModelResponse(parts=[ToolCallPart("observe", {}, tool_call_id="c1")])
    result = ModelRequest(
        parts=[ToolReturnPart("observe", "result", tool_call_id="c1")]
    )
    validate_history([request, call, result])
    with pytest.raises(HarnessError, match="without results"):
        validate_history([request, call, request])
    with pytest.raises(HarnessError, match="unmatched"):
        validate_history([request, result])
    assert all(tool.sequential for tool in _build_tools(Toolkit()))


def test_recorder_known_artifact_and_explicit_image_reference_reuse_id(tmp_path):
    path = tmp_path / "camera.png"
    path.write_bytes(PNG)
    recorder = PlannerRecorder(tmp_path)
    first = recorder.register_binary(
        PNG, media_type="image/png", path=path, observation_id="obs-1"
    )
    second = recorder.register_binary(
        bytes(bytearray(PNG)), media_type="image/png", image_id=first["image_id"]
    )
    assert first == second
    assert first["path"] == str(path) and first["observation_id"] == "obs-1"
    assert not (tmp_path / "planner" / "images").exists()


def test_package_policy_is_the_only_history_image_pruner(tmp_path):
    from pydantic_ai import BinaryContent

    class ImageToolkit(Toolkit):
        def execute_tool(self, name, arguments):
            return ToolResult(
                name,
                {
                    "_image_bytes": b"\x89PNG"
                    + bytes([arguments["value"]]) * (3 * 1024 * 1024)
                },
            )

    image_counts = []

    def answer(messages, info):
        images = [
            item
            for message in messages
            for part in message.parts
            if isinstance(getattr(part, "content", None), list)
            for item in part.content
            if isinstance(item, BinaryContent)
        ]
        image_counts.append(len(images))
        if len(image_counts) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "observe", {"value": index}, tool_call_id=f"img-{index}"
                    )
                    for index in range(3)
                ]
            )
        return ModelResponse(parts=[])

    # The pass-through fixture policy intentionally keeps > 4 MiB and > 2
    # images. A hidden host-level pruning pass would make the second count 2.
    result = ApiAgentLoop(
        FunctionModel(answer),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path,
    ).solve(
        system_prompt="HOST",
        user_message="Inspect.",
        toolkit=ImageToolkit(PolicyRuntime()),
        max_turns=2,
    )
    assert result.error is None
    assert image_counts == [0, 3]


def test_resume_releases_unchanged_boundary_to_real_model(tmp_path):
    controller = DecisionController(pause_next=True)
    called = []

    def answer(messages, info):
        called.append(True)
        return ModelResponse(parts=[])

    async def run():
        toolkit = Toolkit(PolicyRuntime(program=True))
        planner = ApiAgentLoop(
            FunctionModel(answer),
            dashboard_events=NullDashboardEventSink(),
            output_dir=tmp_path,
            decision_controller=controller,
        )
        task = asyncio.create_task(
            planner._solve(
                system_prompt="HOST",
                user_message="Inspect.",
                toolkit=toolkit,
                max_turns=1,
            )
        )
        assert await asyncio.to_thread(controller.wait_for_pause, 3)
        assert called == []
        controller.resume()
        result = await task
        assert toolkit.calls == []
        return result

    result = asyncio.run(run())
    assert result.error is None
    assert called == [True]
    assert result.stats["manual_decisions"] == 0
    assert result.stats["requests"] == 1


def test_same_pixels_keep_distinct_known_observation_paths(tmp_path):
    recorder = PlannerRecorder(tmp_path)
    first_path, second_path = tmp_path / "frame1.png", tmp_path / "frame2.png"
    first_path.write_bytes(PNG)
    second_path.write_bytes(PNG)
    first = recorder.register_binary(
        PNG, media_type="image/png", path=first_path, observation_id="step-1"
    )
    second = recorder.register_binary(
        PNG, media_type="image/png", path=second_path, observation_id="step-2"
    )
    assert first["image_id"] != second["image_id"]
    assert first["path"] != second["path"]
    assert (
        recorder.register_binary(
            PNG, media_type="image/png", image_id=first["image_id"]
        )
        == first
    )


def test_recorder_identity_cache_is_bounded_and_not_content_addressed(tmp_path):
    recorder = PlannerRecorder(tmp_path)
    recorder._identity_byte_limit = 8
    recorder._identity_item_limit = 2
    first_data = bytes(bytearray(b"same"))
    second_data = bytes(bytearray(b"same"))
    first = recorder.register_binary(first_data)
    second = recorder.register_binary(second_data)
    assert first["image_id"] != second["image_id"]
    for index in range(8):
        recorder.register_binary(bytes([index]) * 4)
    assert recorder._identity_bytes <= 8
    assert len(recorder._binary_by_identity) <= 2
    assert recorder.register_binary(first_data, image_id=first["image_id"]) == first


def test_evidence_sequence_and_ids_continue_when_recorder_reopens(tmp_path):
    recorder = PlannerRecorder(tmp_path, episode_id="episode-fixture")
    first = recorder.new_decision()
    recorder.write("sdk_request", {"data": PNG}, context=first)
    reopened = PlannerRecorder(tmp_path, episode_id="episode-fixture")
    second = reopened.new_decision()
    reopened.write("model_response", {"text": "OK"}, context=second, sent=True)
    rows = events(tmp_path)
    assert [row["seq"] for row in rows] == [1, 2]
    assert [row["schema_version"] for row in rows] == [1, 1]
    assert [row["episode_id"] for row in rows] == ["episode-fixture", "episode-fixture"]
    assert first["decision_id"] != second["decision_id"]
    assert reopened.inspect_decision(first["decision_id"])[0]["event"] == "sdk_request"


def test_repeated_invalid_arguments_are_feedback_and_do_not_exhaust_sdk_retries(
    tmp_path,
):
    requests = 0
    bad_values = ["1.5", "not-a-number", True]

    def answer(messages, info):
        nonlocal requests
        requests += 1
        if requests <= 4:
            value = bad_values[requests - 1] if requests <= 3 else "4"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "observe",
                        {"value": value},
                        tool_call_id=f"validation-{requests}",
                    )
                ]
            )
        return ModelResponse(parts=[])

    toolkit = Toolkit(PolicyRuntime())
    result = ApiAgentLoop(
        FunctionModel(answer),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path,
    ).solve(system_prompt="HOST", user_message="Act.", toolkit=toolkit, max_turns=5)
    assert result.error is None
    assert toolkit.calls == [("observe", {"value": 4})]
    rejected = [row for row in events(tmp_path) if row["event"] == "tool_rejected"]
    assert len(rejected) == 3 and all(
        row["payload"]["executed"] is False for row in rejected
    )
    assert result.stats["requests"] == 5


def test_numeric_strings_keep_constraints_and_reject_bool_or_fractional_integer():
    from rpent.harness.contracts import PlannerRequest
    from rpent.planner.boundary import validate_decision

    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer", "minimum": 0, "maximum": 10}},
        "required": ["value"],
        "additionalProperties": False,
    }
    request = PlannerRequest(
        messages=[],
        instructions=None,
        model_settings={},
        tools=[{"name": "observe", "input_schema": schema}],
    )
    validate_decision(ToolDecision("observe", {"value": "2"}), request)
    for value in ["1.5", "1.0", "1e2", "11", "-1", True, False, "NaN"]:
        with pytest.raises(HarnessError, match="invalid arguments"):
            validate_decision(ToolDecision("observe", {"value": value}), request)
    with pytest.raises(HarnessError, match="invalid arguments"):
        validate_decision(ToolDecision("observe", {"value": "2", "extra": 1}), request)
    with pytest.raises(HarnessError, match="not visible"):
        validate_decision(ToolDecision("private_capability", {"value": "2"}), request)


def test_libero_native_string_step_reaches_state_and_http_records_pixels(
    tmp_path, monkeypatch
):
    from functools import partial

    import numpy as np

    from robots.libero.tools import TOOLS_SPEC, view_env_state
    from rpent.tools.state import EnvState
    from rpent.tools.toolkit import Toolkit as NativeToolkit
    from rpent.utils import logging as log_context

    monkeypatch.setattr(log_context, "_output_dir", tmp_path)
    state = EnvState(tmp_path)
    with state.record_step(state={"fixture": "initial"}):
        state.save("agentview.png", np.zeros((4, 4, 3), dtype=np.uint8))
    toolkit = NativeToolkit(state=state, dashboard_events=NullDashboardEventSink())
    toolkit.add_tool(
        "view_env_state",
        next(spec for spec in TOOLS_SPEC if spec["name"] == "view_env_state"),
        partial(view_env_state, state=state),
    )
    # Both reference/current native functions return an ordinary error on raw
    # string input. The new boundary adds documented numeric normalization.
    assert "error" in view_env_state("0", state=state)
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        response = completion(tool=len(requests) < 3)
        if len(requests) < 3:
            call = response["choices"][0]["message"]["tool_calls"][0]
            call["id"] = f"view-{len(requests)}"
            call["function"] = {"name": "view_env_state", "arguments": '{"step":"0"}'}
        return httpx2.Response(200, json=response)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
            model = OpenAIChatModel(
                "qwen3.8-max",
                provider=OpenAIProvider(
                    base_url="https://fixture.invalid/v1",
                    api_key="fixture",
                    http_client=http,
                ),
            )
            return await ApiAgentLoop(
                model, dashboard_events=NullDashboardEventSink(), output_dir=tmp_path
            )._solve(
                system_prompt="Inspect.",
                user_message="Observe.",
                toolkit=toolkit,
                max_turns=4,
            )

    result = asyncio.run(run())
    assert result.error is None and len(requests) == 3
    assert any(
        part.get("type") == "image_url"
        for message in requests[1]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    rows = events(tmp_path)
    calls = [row["payload"] for row in rows if row["event"] == "tool_call"]
    assert all(
        row["raw_args"] == {"step": "0"} and row["args"] == {"step": 0} for row in calls
    )
    assert all(
        row["payload"]["result"]["step"] == 0
        for row in rows
        if row["event"] == "tool_raw"
    )
    refs = []

    def walk(value):
        if isinstance(value, dict):
            if "binary_ref" in value:
                refs.append(value["binary_ref"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for row in rows:
        if row["event"] in {"sdk_request", "http_request"}:
            walk(row["payload"])
    from pathlib import Path

    assert refs
    expected_pixels = state.artifact_path("agentview.png", step=0).read_bytes()
    assert all(Path(ref["path"]).read_bytes() == expected_pixels for ref in refs)


def test_pure_program_reports_only_the_actor_that_ran(tmp_path):
    def never_called(messages, info):
        raise AssertionError("a one-decision program run must not call the model")

    result = ApiAgentLoop(
        FunctionModel(never_called),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path,
    ).solve(
        system_prompt="HOST",
        user_message="Act.",
        toolkit=Toolkit(PolicyRuntime(program=True)),
        max_turns=1,
    )
    assert result.error is None
    assert result.stats["actors"] == ["harness_program"]
    assert result.stats["requests"] == 0


def test_pure_manual_reports_only_meta(tmp_path):
    controller = DecisionController(pause_next=True)

    def never_called(messages, info):
        raise AssertionError("a manual one-decision run must not call the model")

    async def run():
        planner = ApiAgentLoop(
            FunctionModel(never_called),
            dashboard_events=NullDashboardEventSink(),
            output_dir=tmp_path,
            decision_controller=controller,
        )
        task = asyncio.create_task(
            planner._solve(
                system_prompt="HOST",
                user_message="Act.",
                toolkit=Toolkit(PolicyRuntime()),
                max_turns=1,
            )
        )
        assert await asyncio.to_thread(controller.wait_for_pause, 3)
        controller.replace(ToolDecision("observe", {"value": "2"}))
        return await task

    result = asyncio.run(run())
    assert result.error is None
    assert result.stats["actors"] == ["meta"]
    assert result.stats["requests"] == 0
    assert result.stats["autonomous"] is False


def test_failed_http_turn_preserves_completed_usage_and_actual_send_count(tmp_path):
    requests = 0

    def handle(request):
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx2.Response(200, json=completion(tool=True))
        return httpx2.Response(
            500, json={"error": {"message": "fixture failure", "type": "server_error"}}
        )

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http:
            provider = OpenAIProvider(
                base_url="https://fixture.invalid/v1",
                api_key="fixture",
                http_client=http,
            )
            provider.client.max_retries = 0
            return await ApiAgentLoop(
                OpenAIChatModel("qwen3.8-max", provider=provider),
                dashboard_events=NullDashboardEventSink(),
                output_dir=tmp_path,
            )._solve(
                system_prompt="HOST",
                user_message="Act.",
                toolkit=Toolkit(PolicyRuntime()),
                max_turns=4,
            )

    result = asyncio.run(run())
    assert result.error is not None
    assert result.stats["total_input_tokens"] == 5
    assert result.stats["total_output_tokens"] == 2
    assert result.stats["actors"] == ["planner"]
    error = next(row for row in events(tmp_path) if row["event"] == "decision_error")
    assert "sent" not in error
    assert error["handler_called"] is True
    assert error["http_send_attempts"] == 1
    assert requests == result.stats["http_send_attempts"] == 2


def test_timeout_preserves_completed_usage_and_transcript_without_fake_http(tmp_path):
    from pydantic_ai.usage import RequestUsage

    requests = 0

    async def answer(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(
                parts=[ToolCallPart("observe", {"value": 2}, tool_call_id="cost-1")],
                usage=RequestUsage(input_tokens=11, output_tokens=3),
            )
        await asyncio.sleep(2)
        return ModelResponse(parts=[])

    result = ApiAgentLoop(
        FunctionModel(answer),
        dashboard_events=NullDashboardEventSink(),
        output_dir=tmp_path,
        timeout_s=0.2,
    ).solve(
        system_prompt="HOST",
        user_message="Act.",
        toolkit=Toolkit(PolicyRuntime()),
        max_turns=4,
    )
    assert "timed out" in result.error
    assert result.stats["total_input_tokens"] == 11
    assert result.stats["total_output_tokens"] == 3
    assert len(result.messages) >= 3
    assert result.stats["actors"] == ["planner"]
    error = next(row for row in events(tmp_path) if row["event"] == "decision_error")
    assert "sent" not in error and error["handler_called"] is True
    assert error["http_send_attempts"] == 0
