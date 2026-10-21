# Copyright 2026 The RPent Authors.
"""Request-boundary evidence with ordinary IDs and external binary artifacts.

HTTP hooks observe an existing provider client. They never replace its transport,
retry policy, authentication, or model adapter, and never record HTTP headers.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import datetime as dt
import json
import threading
from collections import OrderedDict
from contextvars import ContextVar
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

_DECISION_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "rpent_decision", default=None
)
_ACTIVE_RECORDER: ContextVar[PlannerRecorder | None] = ContextVar(
    "rpent_recorder", default=None
)


def current_decision_context() -> dict[str, Any]:
    """Return the current model/program/manual action provenance, if any."""
    return dict(_DECISION_CONTEXT.get() or {})


@contextlib.contextmanager
def decision_scope(
    recorder: PlannerRecorder | None, context: dict[str, Any]
) -> Iterator[None]:
    ctx_token = _DECISION_CONTEXT.set(dict(context))
    recorder_token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_RECORDER.reset(recorder_token)
        _DECISION_CONTEXT.reset(ctx_token)


class PlannerRecorder:
    """Append inspectable, per-episode evidence; images are stored only once.

    Old images retain only ID/path metadata. The object-identity cache is bounded
    by both bytes and item count; content is never used as a lookup key. Existing
    artifacts use ``register_binary(path=..., observation_id=...)``. Only the
    current request's images are held for position-by-position wire verification.
    """

    def __init__(
        self, output_dir: str | Path, *, episode_id: str | None = None
    ) -> None:
        self.root = Path(output_dir).resolve() / "planner"
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id or __import__("os").environ.get("RPENT_EPISODE_ID") or self.root.parent.name
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.RLock()
        self._decision_number = 0
        self._binary_number = 0
        self._seq = 0
        self._binary_by_identity: OrderedDict[
            int, tuple[bytes | str, dict[str, Any]]
        ] = OrderedDict()
        self._identity_bytes = 0
        self._identity_byte_limit = 16 * 1024 * 1024
        self._identity_item_limit = 128
        self._binary_by_id: dict[str, dict[str, Any]] = {}
        self._binary_by_location: dict[
            tuple[str, str | int | None], dict[str, Any]
        ] = {}
        self._attempts: dict[str, int] = {}
        self.manual_decisions = 0
        self.program_decisions = 0
        self.http_sends = 0
        self.actors: set[str] = set()
        self._fallback_context: dict[str, Any] | None = None
        self._request_images: list[tuple[bytes, dict[str, Any]]] = []
        self._wire_decision: str | None = None
        self._wire_images: dict[str, tuple[bytes, dict[str, Any]]] = {}
        if self.events_path.exists():
            with self.events_path.open(encoding="utf-8") as previous:
                for line in previous:
                    record = json.loads(line)
                    if record.get("event") in {
                        "http_request",
                        "model_response",
                        "tool_call",
                        "tool_rejected",
                    } and record.get("actor") in {"planner", "meta", "harness_program"}:
                        self.actors.add(record["actor"])
                    self._seq = max(self._seq + 1, record.get("seq", 0))
                    decision = record.get("decision_id") or ""
                    if decision.startswith("decision-") and decision[9:].isdigit():
                        self._decision_number = max(
                            self._decision_number, int(decision[9:])
                        )
                    self._restore_image_refs(record.get("payload"))

    def _restore_image_refs(self, value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("binary_ref")
            if isinstance(ref, dict) and isinstance(ref.get("image_id"), str):
                image_id = ref["image_id"]
                if image_id.startswith("image-") and image_id[6:].isdigit():
                    self._binary_number = max(self._binary_number, int(image_id[6:]))
                self._binary_by_id[image_id] = ref
            for child in value.values():
                self._restore_image_refs(child)
        elif isinstance(value, list):
            for child in value:
                self._restore_image_refs(child)

    def _remember(self, value: bytes | str, ref: dict[str, Any]) -> None:
        with self._lock:
            old = self._binary_by_identity.pop(id(value), None)
            if old is not None:
                self._identity_bytes -= len(old[0])
            if len(value) > self._identity_byte_limit:
                return
            self._binary_by_identity[id(value)] = (value, ref)
            self._identity_bytes += len(value)
            while (
                self._identity_bytes > self._identity_byte_limit
                or len(self._binary_by_identity) > self._identity_item_limit
            ):
                _, removed = self._binary_by_identity.popitem(last=False)
                self._identity_bytes -= len(removed[0])

    def _known_object(self, value: bytes | str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._binary_by_identity.get(id(value))
            if entry is not None and entry[0] is value:
                self._binary_by_identity.move_to_end(id(value))
                return dict(entry[1])
        return None

    def binary_from_base64(
        self, payload: str, media_type: str
    ) -> tuple[bytes, dict[str, Any]]:
        data = base64.b64decode(payload, validate=True)
        ref = self._known_object(payload) or self.register_binary(
            data, media_type=media_type
        )
        self._remember(payload, ref)
        self._remember(data, ref)
        return data, ref

    def link_tool_images(self, result: Any) -> None:
        """Link this ToolResult's declared byte fields to its feedback blocks.

        This verifies each declared field against the corresponding freshly
        generated block, never searches old image content. Modified blocks get
        their own artifact unless they already have an explicit object binding.
        """
        raw = result.result
        fields = [
            raw[key]
            for key in (
                "_image_bytes",
                "_image_cam_bytes",
                "_image_nav_bytes",
                "_image_wrist_bytes",
            )
            if isinstance(raw.get(key), bytes)
        ]
        fields.extend(raw[key] for key in raw
                      if key.startswith("_image_depth") and key.endswith("_bytes")
                      and isinstance(raw[key], bytes))
        blocks = [
            block for block in result.content_blocks if block.get("type") == "image"
        ]
        for index, block in enumerate(blocks):
            source = block.get("source", {})
            if source.get("type") != "base64" or not isinstance(
                source.get("data"), str
            ):
                continue
            payload = source["data"]
            if self._known_object(payload) is not None:
                continue
            media_type = source.get("media_type", "image/png")
            data = base64.b64decode(payload, validate=True)
            if index < len(fields) and data == fields[index]:
                ref = self.register_binary(fields[index], media_type=media_type)
            else:
                ref = self.register_binary(data, media_type=media_type)
            self._remember(payload, ref)
            self._remember(data, ref)

    def set_request_images(self, messages: list[Any]) -> None:
        self._request_images = []
        for message in messages:
            for part in message.parts:
                content = getattr(part, "content", None)
                if not isinstance(content, list):
                    continue
                for item in content:
                    if getattr(
                        item, "kind", None
                    ) == "binary" and item.media_type.startswith("image/"):
                        ref = self.register_binary(
                            item.data,
                            media_type=item.media_type,
                            image_id=getattr(item, "_identifier", None),
                        )
                        self._request_images.append((item.data, ref))

    def release_request_images(self) -> None:
        self._request_images = []
        self._wire_images = {}

    def wire_payload(self, value: Any, *, decision_id: str) -> Any:
        """Externalize actual HTTP JSON and verify SDK associations by position."""
        if self._wire_decision != decision_id:
            self._wire_decision = decision_id
            self._wire_images = {}
        position = 0

        def binary(
            payload: str, media_type: str, location: str, encoding: str
        ) -> dict[str, Any]:
            nonlocal position
            data = base64.b64decode(payload, validate=True)
            expected = (
                self._request_images[position]
                if position < len(self._request_images)
                else None
            )
            position += 1
            prior = self._wire_images.get(location)
            # Comparisons check one explicitly associated request position;
            # there is no content-based cache lookup or historical scan.
            if expected is not None and data == expected[0]:
                ref = expected[1]
            elif prior is not None and data == prior[0]:
                ref = prior[1]
            else:
                ref = self.register_binary(data, media_type=media_type)
            self._wire_images[location] = (data, ref)
            return {
                "binary_ref": ref,
                "encoding": encoding,
                "media_type": media_type,
                "field_path": location,
            }

        def walk(item: Any, location: str) -> Any:
            if (
                isinstance(item, str)
                and item.startswith("data:")
                and ";base64," in item
            ):
                header, payload = item.split(";base64,", 1)
                return binary(payload, header[5:], location, "data_uri")
            if isinstance(item, dict):
                if item.get("type") == "base64" and isinstance(item.get("data"), str):
                    return {
                        **item,
                        "data": binary(
                            item["data"],
                            item.get("media_type", "application/octet-stream"),
                            f"{location}.data",
                            "base64",
                        ),
                    }
                return {
                    key: walk(child, f"{location}.{key}") for key, child in item.items()
                }
            if isinstance(item, list):
                return [
                    walk(child, f"{location}[{index}]")
                    for index, child in enumerate(item)
                ]
            return item

        return walk(value, "$")

    def new_decision(self, actor: str = "planner") -> dict[str, Any]:
        with self._lock:
            self._decision_number += 1
            return {
                "episode_id": self.episode_id,
                "decision_id": f"decision-{self._decision_number:06d}",
                "actor": actor,
            }

    def mark_actor(self, actor: str) -> None:
        if actor not in {"planner", "meta", "harness_program"}:
            raise ValueError(f"unsupported decision actor: {actor}")
        self.actors.add(actor)

    def observed_send_attempts(self, decision_id: str) -> int:
        return self._attempts.get(decision_id, 0)

    def register_binary(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        path: str | Path | None = None,
        observation_id: str | int | None = None,
        image_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if image_id in self._binary_by_id:
                ref = self._binary_by_id[image_id]
                self._remember(data, ref)
                return dict(ref)
            location = (
                (str(Path(path).resolve()), observation_id)
                if path is not None
                else None
            )
            if location is not None and location in self._binary_by_location:
                ref = self._binary_by_location[location]
                self._remember(data, ref)
                return dict(ref)
            existing = self._known_object(data)
            if location is None and existing is not None:
                return existing
            self._binary_number += 1
            binary_id = f"image-{self._binary_number:06d}"
            if path is None:
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(media_type, ".bin")
                target = self.root / "images" / f"{binary_id}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            else:
                target = Path(path).resolve()
                if not target.is_file():
                    raise FileNotFoundError(f"image artifact does not exist: {target}")
            ref = {
                "image_id": binary_id,
                "path": str(target),
                "media_type": media_type,
                "bytes": len(data),
            }
            if observation_id is not None:
                ref["observation_id"] = observation_id
            if image_id is not None:
                ref["source_identifier"] = image_id
            self._remember(data, ref)
            self._binary_by_id[binary_id] = ref
            if location is not None:
                self._binary_by_location[location] = ref
            return dict(ref)

    def serializable(self, value: Any, *, field_path: str = "$") -> Any:
        """Preserve fields and replace binary payloads at their exact location."""
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError(f"nonfinite Decimal at {field_path}")
            return str(value)
        if isinstance(value, bytes):
            media_type = (
                "image/png"
                if value.startswith(b"\x89PNG")
                else "image/jpeg"
                if value.startswith(b"\xff\xd8")
                else "application/octet-stream"
            )
            return {
                "binary_ref": self.register_binary(value, media_type=media_type),
                "field_path": field_path,
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            if getattr(value, "kind", None) == "binary" and isinstance(
                getattr(value, "data", None), bytes
            ):
                ref = self.register_binary(
                    value.data,
                    media_type=value.media_type,
                    image_id=getattr(value, "_identifier", None),
                )
                return {
                    "kind": "binary",
                    "binary_ref": ref,
                    "media_type": value.media_type,
                    "vendor_metadata": value.vendor_metadata,
                    "field_path": field_path,
                }
            return {
                field.name: self.serializable(
                    getattr(value, field.name), field_path=f"{field_path}.{field.name}"
                )
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
            }
        if isinstance(value, dict):
            if value.get("kind") == "binary" and isinstance(value.get("data"), bytes):
                return {
                    **{
                        key: self.serializable(item, field_path=f"{field_path}.{key}")
                        for key, item in value.items()
                        if key != "data"
                    },
                    "data": {
                        "binary_ref": self.register_binary(
                            value["data"],
                            media_type=value.get(
                                "media_type", "application/octet-stream"
                            ),
                            image_id=value.get("identifier"),
                        ),
                        "field_path": f"{field_path}.data",
                    },
                }
            if value.get("type") == "base64" and isinstance(value.get("data"), str):
                _, ref = self.binary_from_base64(
                    value["data"], value.get("media_type", "application/octet-stream")
                )
                return {
                    **{
                        key: self.serializable(item, field_path=f"{field_path}.{key}")
                        for key, item in value.items()
                        if key != "data"
                    },
                    "data": {
                        "binary_ref": ref,
                        "encoding": "base64",
                        "field_path": f"{field_path}.data",
                    },
                }
            return {
                str(key): self.serializable(item, field_path=f"{field_path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                self.serializable(item, field_path=f"{field_path}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            header, payload = value.split(";base64,", 1)
            return {
                "binary_ref": self.register_binary(
                    base64.b64decode(payload, validate=True), media_type=header[5:]
                ),
                "encoding": "data_uri",
                "field_path": field_path,
            }
        if isinstance(value, (dt.datetime, dt.date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "tolist"):
            return self.serializable(value.tolist(), field_path=field_path)
        raise TypeError(f"cannot record {type(value).__name__} at {field_path}")

    def write(
        self,
        event: str,
        payload: Any,
        *,
        context: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = {
                "schema_version": 1,
                "seq": self._seq,
                "event": event,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "decision_id": None,
                "actor": None,
                **(
                    context
                    or current_decision_context()
                    or {"episode_id": self.episode_id}
                ),
                **extra,
                "payload": self.serializable(payload),
            }
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            return record

    def inspect_decision(self, decision_id: str) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        with self.events_path.open(encoding="utf-8") as stream:
            return [
                record
                for line in stream
                if (record := json.loads(line)).get("decision_id") == decision_id
            ]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "planner_evidence": str(self.events_path),
            "manual_decisions": self.manual_decisions,
            "harness_program_decisions": self.program_decisions,
            "http_send_attempts": self.http_sends,
            "autonomous": "meta" not in self.actors,
            "actors": sorted(self.actors),
        }


def attach_http_client(
    client: Any, recorder: PlannerRecorder | None = None, *, fallback: bool = False
):
    """Attach public HTTPX/httpx2 event hooks to a live HTTP client.

    ``recorder=None`` selects the recorder in ``decision_scope``. ``fallback=True``
    supports an unmodified reference loop without SDK hooks: retry count only
    determines correlation, and no header values are written to evidence.
    Returns an idempotent remover. The caller retains client lifecycle ownership.
    """
    hooks = client.event_hooks

    async def on_request(request: Any) -> None:
        from rpent.utils.execution import planner_rate_limit
        await planner_rate_limit()
        active = _ACTIVE_RECORDER.get()
        target = active if recorder is None else recorder
        if target is None or (active is not None and active is not target):
            return
        context = current_decision_context()
        if not context:
            if not fallback:
                return
            retry = request.headers.get("x-stainless-retry-count", "0")
            if retry == "0" or target._fallback_context is None:
                target._fallback_context = target.new_decision()
            context = dict(target._fallback_context)
        key = context["decision_id"]
        target.mark_actor(context.get("actor", "planner"))
        with target._lock:
            attempt = target._attempts.get(key, 0) + 1
            target._attempts[key] = attempt
            target.http_sends += 1
        request.extensions["rpent_evidence"] = (target, context, attempt)
        raw = await request.aread()
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = {"unparsed_body": raw}
        endpoint = f"{request.url.scheme}://{request.url.host}"
        if request.url.port:
            endpoint += f":{request.url.port}"
        endpoint += request.url.path
        target.write(
            "http_request",
            target.wire_payload(body, decision_id=key),
            context=context,
            sent=True,
            send_attempt=attempt,
            method=request.method,
            endpoint=endpoint,
        )

    async def on_response(response: Any) -> None:
        record = response.request.extensions.get("rpent_evidence")
        if record is None:
            return
        target, context, attempt = record
        if "text/event-stream" in response.headers.get("content-type", ""):
            target.write(
                "http_response",
                {"streamed": True},
                context=context,
                send_attempt=attempt,
                status_code=response.status_code,
            )
            return
        raw = await response.aread()
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = {"unparsed_body": raw}
        target.write(
            "http_response",
            body,
            context=context,
            send_attempt=attempt,
            status_code=response.status_code,
        )

    hooks.setdefault("request", []).append(on_request)
    hooks.setdefault("response", []).append(on_response)

    def detach() -> None:
        for kind, callback in (("request", on_request), ("response", on_response)):
            if callback in hooks[kind]:
                hooks[kind].remove(callback)

    return detach


def attach_provider_http(provider: Any) -> bool:
    """Install one context-driven hook on the provider's existing HTTP client.

    OpenAI/Anthropic expose their provider client publicly; those SDKs currently
    keep the HTTP client in ``_client``. This is the sole compatibility lookup;
    request interception itself uses the HTTP client's public event hooks.
    """
    sdk_client = getattr(provider, "client", None)
    http = getattr(sdk_client, "_client", None)
    if http is None and hasattr(sdk_client, "event_hooks"):
        http = sdk_client
    if http is None or not hasattr(http, "event_hooks"):
        return False
    if not getattr(http, "_rpent_evidence_hooks_installed", False):
        attach_http_client(http)
        http._rpent_evidence_hooks_installed = True
    return True
