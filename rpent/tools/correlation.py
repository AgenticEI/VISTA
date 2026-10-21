# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Propagate the model's root tool-call id into internal robot operations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_ROOT_TOOL_CALL_ID: ContextVar[str | None] = ContextVar(
    "rpent_root_tool_call_id", default=None
)


@contextmanager
def root_tool_call_scope(call_id: str | None) -> Iterator[None]:
    """Associate nested synchronous work with one model tool invocation."""
    normalized = call_id if isinstance(call_id, str) and call_id else None
    token = _ROOT_TOOL_CALL_ID.set(normalized)
    try:
        yield
    finally:
        _ROOT_TOOL_CALL_ID.reset(token)


def current_root_tool_call_id() -> str | None:
    """Return the model tool-call id inherited by the current operation."""
    return _ROOT_TOOL_CALL_ID.get()
