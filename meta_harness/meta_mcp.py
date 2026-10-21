# Copyright 2026 The RPent Authors.
"""Thin MCP presentation of the single Meta authority and shared pool."""

import asyncio
import inspect
import json
import sys
from pathlib import Path

from meta_harness.live import exchange
from meta_harness.meta import MetaAttempt

# These schemas are also available to opening material and ordinary tests.
_FIELDS = {
    "budget": {},
    "history_summary": {},
    "history_episode": {"episode_id": "string"},
    "history_episodes": {"task_id": "string", "seed": "integer", "repeat": "integer",
        "evaluation_id": "string", "controller": "string", "validity": "string", "offset": "integer", "limit": "integer"},
    "history_decisions": {"episode_id": "string"},
    "history_decision": {"episode_id": "string", "decision_id": "string"},
    "history_timeline": {"episode_id": "string", "camera": "string", "max_frames": "integer",
        "start_env_step": "integer", "end_env_step": "integer"},
    "history_frame": {
        "episode_id": "string",
        "env_step": "integer",
        "camera": "string",
    },
    "inspect_harness": {"commit": "string"},
    "inspect_decision": {"episode_id": "string", "decision_id": "string"},
    "evaluate": {"request_id": "string", "cells": "array"},
    "evaluation_status": {"evaluation_id": "string"},
    "select_parent": {"commit": "string"},
    "submit": {"notes": "string"},
    "live_start": {
        "session_id": "string",
        "task_id": "string",
        "seed": "integer",
        "mode": "string",
    },
    "live_decision": {"session_id": "string", "request_id": "string"},
    "live_reply": {
        "session_id": "string",
        "request_id": "string",
        "decision_id": "string",
        "tool_name": "string",
        "args": "object",
        "resume": "boolean",
    },
    "live_capability": {
        "session_id": "string",
        "request_id": "string",
        "capability": "string",
        "args": "object",
    },
    "live_observe": {"session_id": "string", "request_id": "string"},
    "live_close": {"session_id": "string"},
}
_DESCRIPTIONS = {
    "evaluate": "Freeze current candidate and enqueue fresh autonomous cells. Returns ID immediately. Each cell needs cell_id, task_id, seed, repeat. Retrying request_id is idempotent.",
    "live_decision": "Poll real Planner boundary; reuse request_id only to fetch that response. Use a fresh ID for the next poll.",
    "live_reply": "Replace current decision via its visible tool menu, or resume=true to return control to Planner.",
    "history_episodes": "List recorded attempts by task, scene seed, repeat, evaluation, controller or validity. Includes failed/invalid attempts, not just accepted outcomes; pagination never hides the total.",
    "history_decisions": "List the episode's actual decisions, tool calls, actors and executed physical-step ranges. before_env_step is the observation before the action.",
    "history_decision": "Read one exact decision's SDK request, HTTP attempts, model response, raw tools and actual Planner feedback, plus links to before/after physical frames. Records are not summarized away.",
    "history_timeline": "View a sampled overview or dense interval of recorded control-step images; camera=all shows each real camera. Sampling is labelled and is not a claim of full asset verification.",
    "history_frame": "Read exact recorded RGB, action, robot state and decision linkage at env_step. Depth is returned only if archived; full_rgb episodes do not contain historical depth. camera=all returns both LIBERO cameras.",
    "live_start": "Freeze the current package and queue a private environment. mode=planner pauses before its first real decision; use live_decision to inspect it. This does not consume an autonomous evaluation request.",
    "live_observe": "At a paused live decision, obtain current public state/images as separate Meta evidence. Returned artifact paths can be opened as images. It does not add observations to Planner history.",
    "live_capability": "At a paused live decision, try an exact capability ID/schema from the opening material. Results are Meta diagnostics; inspect resulting physical state before deriving a package change.",
    "live_close": "Close this live environment and release its slot; preserve its diagnostic records.",
}
_OPTIONAL = {
    "history_episodes": set(_FIELDS["history_episodes"]),
    "history_timeline": {"camera", "max_frames", "start_env_step", "end_env_step"},
    "history_frame": {"camera"},
    "live_start": {"mode"},
    "live_reply": {"tool_name", "args", "resume"},
    "live_capability": {"args"},
}


def definitions(*, live_enabled=True):
    definitions = [
        {
            "name": name,
            "description": _DESCRIPTIONS.get(name, name.replace("_", " ")),
            "inputSchema": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in fields.items()},
                "required": [k for k in fields if k not in _OPTIONAL.get(name, set())],
                "additionalProperties": False,
            },
        }
        for name, fields in _FIELDS.items()
        if live_enabled or not name.startswith("live_")
    ]
    for definition in definitions:
        if definition["name"] == "evaluate":
            definition["inputSchema"]["properties"]["cells"].update({
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "cell_id": {"type": "string"}, "task_id": {"type": "string"},
                        "seed": {"type": "integer", "minimum": 0},
                        "repeat": {"type": "integer", "minimum": 0},
                    },
                    "required": ["cell_id", "task_id", "seed", "repeat"],
                    "additionalProperties": False,
                },
            })
        if definition["name"] == "live_reply":
            definition["inputSchema"]["anyOf"] = [
                {"properties": {"resume": {"const": True}}, "required": ["resume"]},
                {"properties": {"resume": {"const": False}}, "required": ["tool_name"]},
            ]
        if definition["name"] == "live_start":
            definition["inputSchema"]["properties"]["mode"]["enum"] = ["planner", "capability"]
    return definitions


def invoke(attempt, name, args):
    if name.startswith("live_") and not attempt.config.development.live_enabled:
        raise PermissionError("Live tools are disabled for this run's ablation protocol")
    if name not in _FIELDS:
        raise ValueError("unknown Meta tool")
    import jsonschema

    jsonschema.validate(
        args, next(d["inputSchema"] for d in definitions() if d["name"] == name)
    )
    if name in {
        "budget",
        "history_summary",
        "history_episode",
        "history_episodes",
        "history_decisions",
        "history_decision",
        "inspect_decision",
        "evaluate",
        "select_parent",
        "submit",
    }:
        return getattr(attempt, name)(**args)
    if name == "inspect_harness":
        from meta_harness.inspection import inspect_package

        return inspect_package(
            attempt.harnesses.materialize(args["commit"]), args["commit"], robot=attempt.config.robot
        )
    if name == "evaluation_status":
        return attempt.evaluation_status(**args)
    if name in {"history_frame", "history_timeline"}:
        from meta_harness.trajectory import TrajectoryReader, history_frame, history_timeline

        episode = attempt.history_episode(args["episode_id"])
        operation = history_frame if name == "history_frame" else history_timeline
        return operation(TrajectoryReader.from_episode(Path(episode["rollout"])),
                         **{k: v for k, v in args.items() if k != "episode_id"})
    if name == "live_start":
        return attempt.live_start(**args)
    sid = args["session_id"]
    req = attempt.store.read(f"pool/live/{sid}/request.json")
    if req["scope_id"] != attempt.context["scope_id"]:
        raise ValueError("live session belongs to another scope")
    if name == "live_close":
        from meta_harness.pool import read_optional
        from rpent.utils.execution import signal_owned

        attempt.store.write_once(f"pool/live/{sid}/close.json", {"reason": "closed"})
        owner = read_optional(attempt.store, f"pool/jobs/live-{sid}/owner.json")
        if owner:
            signal_owned(owner)
        return {"state": "closing"}
    attempt._editable()
    response = exchange(
        attempt.store,
        sid,
        args["request_id"],
        name.removeprefix("live_"),
        {k: v for k, v in args.items() if k not in {"session_id", "request_id"}},
    )

    if name in {"live_observe", "live_capability"}:
        return live_image_content(response, attempt.store.root)
    return response


def live_image_content(response, root):
    """Render recorded PNG refs as MCP images; metadata stays compact JSON."""
    from meta_harness.trajectory import _image, _text
    blocks = [_text(response)]
    seen = set()
    def visit(value):
        if isinstance(value, dict):
            ref = value.get("binary_ref", {})
            if ref.get("media_type") == "image/png":
                path = Path(ref["path"]).resolve()
                if not path.is_relative_to(Path(root).resolve()):
                    raise ValueError("live image outside the run")
                if path not in seen:
                    seen.add(path)
                    blocks.extend(_image(path.read_bytes(), value.get("field_path", str(path))))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(response)
    return (response, blocks) if len(blocks) > 1 else response


async def serve(root, attempt_id):
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    attempt = MetaAttempt(root, attempt_id)

    async def call(name, arguments):
        try:
            result = await asyncio.to_thread(invoke, attempt, name, arguments or {})
            if isinstance(result, tuple):
                _, blocks = result
                content = [
                    types.ImageContent(**b)
                    if b["type"] == "image"
                    else types.TextContent(**b)
                    for b in blocks
                ]
            else:
                content = [
                    types.TextContent(
                        type="text", text=json.dumps(result, ensure_ascii=False)
                    )
                ]
            return types.CallToolResult(content=content, isError=False)
        except Exception as exc:
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=f"{type(exc).__name__}: {exc}")
                ],
                isError=True,
            )

    if "on_list_tools" in inspect.signature(Server).parameters:

        async def on_list(context, params):
            return types.ListToolsResult(tools=[types.Tool(**d) for d in definitions(live_enabled=attempt.config.development.live_enabled)])

        async def on_call(context, params):
            return await call(params.name, params.arguments)

        server = Server(
            "rpent-rsi", version="1", on_list_tools=on_list, on_call_tool=on_call
        )
    else:
        server = Server("rpent-rsi", version="1")

        @server.list_tools()
        async def list_tools():
            return [types.Tool(**d) for d in definitions(live_enabled=attempt.config.development.live_enabled)]

        server.call_tool()(call)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(serve(*sys.argv[1:]))
