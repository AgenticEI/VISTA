"""Keep a live planner decision and simulator while its API is unavailable.

This wraps one model request below PydanticAI's agent loop.  A retriable
provider error therefore cannot unwind the tool loop or reset the scene.
There is no attempt or waiting-time limit.  A bounded *interval* between
attempts prevents a quota outage from causing a request storm.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError


def retry_reason(error: Exception) -> str | None:
    if isinstance(error, ModelHTTPError):
        code = error.status_code
        if code == 402:
            return "payment_required"
        if code == 403:
            body = json.dumps(error.body, default=str, ensure_ascii=False).lower()
            return ("insufficient_user_quota" if any(term in body for term in
                    ("insufficient_user_quota", "额度不足", "余额不足")) else None)
        if code in {408, 429} or code >= 500:
            return f"http_{code}"
        return None
    # PydanticAI wraps some OpenAI/httpx timeouts in ModelAPIError before
    # they reach the model.request boundary.  The provider's exact observed
    # message is "Request timed out."; do not retry arbitrary API errors.
    if isinstance(error, ModelAPIError):
        message = str(error).lower()
        if any(term in message for term in ("insufficient_user_quota", "额度不足", "余额不足")):
            return "insufficient_user_quota"
        if "timed out" in message or "timeout" in message:
            return "model_api_timeout"
        if "connection error" in message or "network error" in message:
            return "model_api_transport"
        return None
    if isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError)):
        return "transport"
    try:
        from openai import APIConnectionError, APITimeoutError
    except ImportError:
        return None
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return "transport"
    return None


class PersistentRequest:
    """A single request's repeated sends, retaining the original message list."""

    def __init__(self, original, state_path: Path):
        self.original = original
        self.state_path = state_path

    def _publish(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(state, stream, sort_keys=True)
        os.replace(temporary, self.state_path)

    async def __call__(self, messages, settings, parameters):
        started = None
        attempts = 0
        interval = 2.0
        last_reported = 0.0
        try:
            while True:
                try:
                    result = await self.original(messages, settings, parameters)
                except Exception as error:
                    reason = retry_reason(error)
                    if reason is None:
                        raise
                    now = time.time()
                    started = now if started is None else started
                    attempts += 1
                    if attempts == 1 or now - last_reported >= 300:
                        self._publish({"state": "waiting", "reason": reason,
                                       "since": started, "last_attempt_at": now,
                                       "attempts": attempts})
                        last_reported = now
                    await asyncio.sleep(interval)
                    interval = min(interval * 2.0, 60.0)
                else:
                    if started is not None:
                        self._publish({"state": "recovered", "since": started,
                                       "recovered_at": time.time(),
                                       "attempts": attempts})
                    return result
        except BaseException:
            if started is not None:
                self._publish({"state": "interrupted", "since": started,
                               "ended_at": time.time(), "attempts": attempts})
            raise


def install_persistent_requests(model, output_dir: str | Path):
    """Use the model's existing provider and wire format without replacing it."""
    record = Path(output_dir) / "planner/provider-wait.json"
    model.request = PersistentRequest(model.request, record)
    return model
