# Copyright 2026 The RPent Authors.
"""Public PydanticAI capabilities for package policy and decision inspection."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import math
import re
import threading
from decimal import Decimal, InvalidOperation
from typing import Any

import jsonschema
from pydantic_ai import BinaryContent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
)

from rpent.harness.contracts import HarnessError, PlannerRequest, ToolDecision
from rpent.planner.recording import PlannerRecorder, decision_scope


def _plain(value: Any) -> Any:
    # dataclass fields avoid BinaryContent.identifier's implicit content digest.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
        if getattr(value, "kind", None) == "binary" and getattr(
            value, "_identifier", None
        ):
            result["identifier"] = value._identifier
        return result
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def planner_request(context: Any) -> PlannerRequest:
    """Convert public SDK data to the package's plain, editable DTO."""
    return PlannerRequest(
        messages=_plain(context.messages),
        instructions=next(
            (
                message.instructions
                for message in reversed(context.messages)
                if isinstance(message, ModelRequest)
            ),
            None,
        ),
        model_settings=copy.deepcopy(context.model_settings or {}),
        tools=[
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": copy.deepcopy(tool.parameters_json_schema),
            }
            for tool in context.model_request_parameters.function_tools
        ],
    )


def validate_history(messages: list[Any]) -> None:
    """Reject policy histories that orphan tool calls or their results."""
    pending: dict[str, str] = {}
    seen: set[str] = set()
    for message in messages:
        for part in message.parts:
            kind = getattr(part, "part_kind", None)
            if kind == "tool-call":
                call_id = part.tool_call_id
                if not isinstance(call_id, str) or not call_id or call_id in seen:
                    raise HarnessError(
                        "history tool calls require nonempty, unique call IDs"
                    )
                pending[call_id] = part.tool_name
                seen.add(call_id)
            elif kind in {"tool-return", "retry-prompt"} and getattr(
                part, "tool_call_id", None
            ):
                call_id = part.tool_call_id
                if pending.pop(call_id, None) != getattr(part, "tool_name", None):
                    raise HarnessError(
                        f"history contains an unmatched tool result: {call_id}"
                    )
    if pending:
        raise HarnessError(
            f"history contains tool calls without results: {sorted(pending)}"
        )
    if not messages or not isinstance(messages[-1], ModelRequest):
        raise HarnessError("prepared history must end with a request")


class DecisionController:
    """Pause exactly before the next model send, then replace or release it.

    A host debugger calls ``pause_next()``, inspects ``wait_for_pause()``, then
    ``replace(ToolDecision(...))`` or ``resume()``. Replacement affects one
    decision; the following boundary returns to the real Planner. Nothing is
    executed while paused, and only the current request's tool menu is accepted.
    """

    def __init__(self, *, pause_next: bool = False) -> None:
        self._condition = threading.Condition()
        self._pause_next = pause_next
        self._paused = False
        self._request: PlannerRequest | None = None
        self._context: dict[str, Any] = {}
        self._selection: ToolDecision | None = None
        self._released = False

    def pause_next(self) -> None:
        with self._condition:
            self._pause_next = True

    def wait_for_pause(self, timeout: float | None = None) -> dict[str, Any] | None:
        with self._condition:
            if not self._condition.wait_for(lambda: self._paused, timeout=timeout):
                return None
            return {
                **self._context,
                "request": copy.deepcopy(self._request),
                "sent": False,
            }

    def replace(self, decision: ToolDecision) -> None:
        with self._condition:
            if not self._paused or self._request is None:
                raise RuntimeError("Planner is not paused at a decision boundary")
            validate_decision(decision, self._request)
            self._selection = copy.deepcopy(decision)
            self._released = True
            self._condition.notify_all()

    def resume(self) -> None:
        with self._condition:
            if not self._paused:
                raise RuntimeError("Planner is not paused at a decision boundary")
            self._selection = None
            self._released = True
            self._condition.notify_all()

    async def at_boundary(
        self, request: PlannerRequest, context: dict[str, Any]
    ) -> tuple[bool, ToolDecision | None]:
        with self._condition:
            if not self._pause_next:
                return False, None
            self._pause_next = False
            self._paused = True
            self._released = False
            self._request = copy.deepcopy(request)
            self._context = dict(context)
            self._selection = None
            self._condition.notify_all()
        try:
            while True:
                with self._condition:
                    if self._released:
                        return True, self._selection
                await asyncio.sleep(0.02)
        finally:
            with self._condition:
                self._paused = False
                self._request = None
                self._condition.notify_all()


def normalize_tool_arguments(value: Any, schema: dict[str, Any]) -> Any:
    """Normalize numeric strings without changing types or losing precision.

    This compatibility seam is shared by model, package, and manual decisions.
    Booleans are never numbers; decimal/exponent strings are never coerced to
    integers. Validation of ranges, required fields, and additional properties
    still runs against the unchanged, currently visible schema afterward.
    """
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties")
        value = {
            key: normalize_tool_arguments(
                item,
                properties.get(key, additional if isinstance(additional, dict) else {}),
            )
            for key, item in value.items()
        }
    elif isinstance(value, list):
        item_schema = schema.get("items", {})
        prefix = schema.get("prefixItems", [])
        value = [
            normalize_tool_arguments(
                item,
                prefix[index]
                if index < len(prefix)
                else item_schema
                if isinstance(item_schema, dict)
                else {},
            )
            for index, item in enumerate(value)
        ]
    elif isinstance(value, str):
        kinds = schema.get("type", [])
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if "string" not in kinds:
            if "integer" in kinds and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
                value = int(value)
            elif "number" in kinds:
                stripped = value.strip()
                if re.fullmatch(r"[+-]?[0-9]+", stripped):
                    value = int(stripped)
                else:
                    try:
                        number = float(stripped)
                        if math.isfinite(number) and Decimal(str(number)) == Decimal(
                            stripped
                        ):
                            value = number
                    except (ValueError, InvalidOperation):
                        pass
    for union in ("anyOf", "oneOf"):
        alternatives = schema.get(union)
        if not isinstance(alternatives, list):
            continue
        for alternative in alternatives:
            if jsonschema.Draft202012Validator(alternative).is_valid(value):
                return value
        for alternative in alternatives:
            normalized = normalize_tool_arguments(value, alternative)
            if jsonschema.Draft202012Validator(alternative).is_valid(normalized):
                return normalized
    return value


def validate_decision(decision: ToolDecision, request: PlannerRequest) -> None:
    if not isinstance(decision, ToolDecision):
        raise HarnessError("replacement must be a ToolDecision")
    tools = {tool["name"]: tool for tool in request.tools}
    if decision.tool_name not in tools:
        raise HarnessError(
            f"tool is not visible in this decision: {decision.tool_name}"
        )
    try:
        schema = tools[decision.tool_name]["input_schema"]
        jsonschema.validate(normalize_tool_arguments(decision.args, schema), schema)
    except jsonschema.ValidationError as exc:
        raise HarnessError(
            f"invalid arguments for {decision.tool_name}: {exc.message}"
        ) from exc


class PlannerBoundary(AbstractCapability[Any]):
    """Keep all actors inside the original Agent.iter tool execution path."""

    def __init__(
        self,
        *,
        runtime: Any = None,
        recorder: PlannerRecorder | None = None,
        controller: DecisionController | None = None,
    ) -> None:
        self.runtime = runtime
        self.recorder = recorder
        self.controller = controller
        self._number = 0
        self._calls: dict[str, dict[str, Any]] = {}
        self._visible: dict[str, dict[str, Any]] = {}
        self.manual_decisions = 0
        self.program_decisions = 0
        self.model_requests = 0
        self.actors: set[str] = set()

    async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
        if self.runtime is None:
            return request_context
        original = planner_request(request_context)
        prepared = self.runtime.prepare_request(original)
        # A Policy that replaces bytes gets a fresh identity, even if it left
        # the previous image identifier in its copied message structure.
        original_images = {
            item.get("identifier"): item["data"]
            for message in original.messages
            for part in message.get("parts", [])
            if isinstance(part.get("content"), list)
            for item in part["content"]
            if isinstance(item, dict) and item.get("kind") == "binary"
        }
        for message in prepared.messages:
            for part in message.get("parts", []):
                if isinstance(part.get("content"), list):
                    for item in part["content"]:
                        if (
                            isinstance(item, dict)
                            and item.get("kind") == "binary"
                            and original_images.get(item.get("identifier"))
                            is not item.get("data")
                        ):
                            item.pop("identifier", None)
        messages = ModelMessagesTypeAdapter.validate_python(prepared.messages)
        validate_history(messages)
        if self.recorder:
            for message in messages:
                for part in message.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, BinaryContent):
                                ref = self.recorder.register_binary(
                                    item.data,
                                    media_type=item.media_type,
                                    image_id=getattr(item, "_identifier", None),
                                )
                                item._identifier = ref["image_id"]
        # The newest effective instructions and the structured provider caching
        # representation must describe the same policy-selected text.
        messages[-1] = dataclasses.replace(
            messages[-1], instructions=prepared.instructions
        )
        parameters = request_context.model_request_parameters
        available = {tool.name: tool for tool in parameters.function_tools}
        parameters = dataclasses.replace(
            parameters,
            function_tools=[available[tool["name"]] for tool in prepared.tools],
        )
        if prepared.instructions != original.instructions:
            parameters = dataclasses.replace(
                parameters,
                instruction_parts=[InstructionPart(prepared.instructions)]
                if prepared.instructions
                else [],
            )
        request_context.messages = messages
        request_context.model_settings = prepared.model_settings
        request_context.model_request_parameters = parameters
        return request_context

    async def wrap_model_request(
        self, ctx: Any, *, request_context: Any, handler: Any
    ) -> ModelResponse:
        request = planner_request(request_context)
        self._visible = {tool["name"]: tool for tool in request.tools}
        self._number += 1
        context = (
            self.recorder.new_decision()
            if self.recorder
            else {"decision_id": f"decision-{self._number:06d}", "actor": "planner"}
        )
        if self.recorder:
            self.recorder.set_request_images(request_context.messages)
            self.recorder.write(
                "sdk_request",
                {
                    "request": request,
                    "parameters": request_context.model_request_parameters,
                    "model": request_context.model.model_name,
                    "sources": getattr(self.runtime, "source_map", None),
                },
                context=context,
            )
        manual_release, decision = (
            await self.controller.at_boundary(request, context)
            if self.controller
            else (False, None)
        )
        actor = "meta" if decision is not None else "planner"
        if not manual_release and self.runtime is not None:
            decision = self.runtime.decide(copy.deepcopy(request))
            if decision is not None:
                actor = "harness_program"
        context["actor"] = actor
        with decision_scope(self.recorder, context):
            if decision is not None:
                validate_decision(decision, request)
                self.actors.add(actor)
                if self.recorder:
                    self.recorder.mark_actor(actor)
                call_id = (
                    decision.tool_call_id or f"{context['decision_id']}-{actor}-call-1"
                )
                response = ModelResponse(
                    parts=[
                        ToolCallPart(
                            decision.tool_name, decision.args, tool_call_id=call_id
                        )
                    ],
                    model_name=request_context.model.model_name,
                )
                if actor == "meta":
                    self.manual_decisions += 1
                    if self.recorder:
                        self.recorder.manual_decisions += 1
                else:
                    self.program_decisions += 1
                    if self.recorder:
                        self.recorder.program_decisions += 1
            else:
                try:
                    self.actors.add("planner")
                    if self.recorder:
                        self.recorder.mark_actor("planner")
                    self.model_requests += 1
                    from rpent.utils.execution import planner_slot
                    async with planner_slot():
                        response = await handler(request_context)
                except BaseException as exc:
                    if self.recorder:
                        self.recorder.write(
                            "decision_error",
                            {"type": type(exc).__name__, "message": str(exc)},
                            handler_called=True,
                            http_send_attempts=self.recorder.observed_send_attempts(
                                context["decision_id"]
                            ),
                        )
                        self.recorder.release_request_images()
                    raise
            for part in response.tool_calls:
                if part.tool_call_id in self._calls:
                    raise HarnessError(
                        f"response reused a tool call ID: {part.tool_call_id}"
                    )
                self._calls[part.tool_call_id] = dict(context)
            if self.recorder:
                self.recorder.write("model_response", response, sent=decision is None)
                self.recorder.release_request_images()
        return response

    async def wrap_tool_execute(
        self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any
    ) -> Any:
        context = self._calls.get(call.tool_call_id, {"actor": "planner"})
        with decision_scope(self.recorder, context):
            try:
                visible = self._visible.get(call.tool_name)
                if visible is None:
                    return self._rejected_tool(
                        call,
                        f"tool is not visible in the current request: {call.tool_name}",
                    )
                raw_args = call.args_as_dict()
                normalized = normalize_tool_arguments(raw_args, visible["input_schema"])
                try:
                    jsonschema.validate(normalized, visible["input_schema"])
                except jsonschema.ValidationError as exc:
                    return self._rejected_tool(
                        call, f"invalid arguments for {call.tool_name}: {exc.message}"
                    )
                if self.recorder:
                    self.recorder.write(
                        "tool_call",
                        {
                            "tool_name": call.tool_name,
                            "args": normalized,
                            "raw_args": raw_args,
                            "tool_call_id": call.tool_call_id,
                        },
                    )
                feedback = await handler(normalized)
                if self.recorder:
                    self.recorder.write(
                        "tool_feedback",
                        {
                            "tool_call_id": call.tool_call_id,
                            "tool_name": call.tool_name,
                            "feedback": feedback,
                        },
                    )
                return feedback
            except Exception as exc:
                if self.recorder:
                    self.recorder.write(
                        "tool_error",
                        {"tool_call_id": call.tool_call_id, "error": str(exc)},
                    )
                raise

    def _rejected_tool(self, call: Any, reason: str) -> str:
        """Return ordinary tool feedback without spending SDK retry budget."""
        feedback = json.dumps(
            {"error": reason, "got": call.args_as_dict()}, ensure_ascii=False
        )
        if self.recorder:
            self.recorder.write(
                "tool_rejected",
                {
                    "tool_name": call.tool_name,
                    "tool_call_id": call.tool_call_id,
                    "args": call.args_as_dict(),
                    "error": reason,
                    "executed": False,
                },
            )
            self.recorder.write(
                "tool_feedback",
                {
                    "tool_name": call.tool_name,
                    "tool_call_id": call.tool_call_id,
                    "feedback": feedback,
                },
            )
        return feedback
