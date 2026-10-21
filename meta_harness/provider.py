# Copyright 2026 The RPent Authors.
"""Codex process/provider adapter, independent of experiment orchestration."""

from __future__ import annotations

import io
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meta_harness._files import atomic_json, read_json
from meta_harness.config import META_MODEL, validate_model_name
from meta_harness.sandbox import CodexFileSandbox, SandboxUnavailable


class CodexError(RuntimeError):
    pass


def _mcp_server_flags(servers: dict[str, dict[str, Any]]) -> list[str]:
    """Pass explicit host-owned stdio servers without enabling user config."""
    if not isinstance(servers, dict):
        raise ValueError("mcp_servers must be an object")
    flags: list[str] = []
    allowed = {"command", "args", "env_vars", "startup_timeout_sec", "tool_timeout_sec"}
    for name, config in servers.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                "MCP server names must contain only letters, digits, '_' or '-'"
            )
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"unsupported MCP server configuration: {name}")
        command = config.get("command")
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise ValueError(f"MCP server {name} requires a command")
        args = config.get("args", [])
        if not isinstance(args, list) or any(
            not isinstance(arg, str) or "\x00" in arg for arg in args
        ):
            raise ValueError(f"MCP server {name} args must be a list of strings")
        env_vars = config.get("env_vars", [])
        if not isinstance(env_vars, list) or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
            for value in env_vars
        ):
            raise ValueError(
                f"MCP server {name} env_vars must be environment variable names"
            )
        for key in ("startup_timeout_sec", "tool_timeout_sec"):
            if key in config and (
                type(config[key]) not in (int, float)
                or not math.isfinite(config[key])
                or config[key] <= 0
            ):
                raise ValueError(f"MCP server {name} {key} must be positive and finite")
        for key, value in config.items():
            flags.extend(("-c", f"mcp_servers.{name}.{key}={json.dumps(value)}"))
    return flags


@dataclass(frozen=True)
class CodexTurn:
    thread_id: str
    last_message: str
    returncode: int


@dataclass
class CodexSession:
    """One logical Codex thread persisted across one task's proposal rounds."""

    project_dir: Path
    runtime_dir: Path
    session_record: Path
    event_log: Path
    model: str = META_MODEL
    timeout_s: float | None = 1800.0
    thread_id: str | None = None
    environment: dict[str, str] | None = None
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasoning_effort: str | None = None
    task_id: str | None = None
    provider: dict[str, str] | None = None
    inherited_fds: tuple[int, ...] = ()
    process_record: Path | None = None
    initial_images: tuple[Path, ...] = ()
    allow_session_migration: bool = False
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_model_name(self.model, "Meta Codex model")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("Codex timeout must be positive")
        if self.session_record.is_file():
            value = read_json(self.session_record)
            stored_model = value.get("model")
            if stored_model != self.model and not self.allow_session_migration:
                raise CodexError(
                    f"cannot resume {stored_model!r} session with {self.model!r}"
                )
            # A provider change moves the session to a different backend; the old
            # thread cannot be resumed there, so start a fresh one with the same files.
            provider_changed = value.get("provider") != (self.provider or {}).get(
                "name"
            )
            if provider_changed and not self.allow_session_migration:
                self.thread_id = None
            stored_id = value.get("thread_id")
            self.task_id = value.get("task_id")
            if (not provider_changed or self.allow_session_migration) and isinstance(stored_id, str) and stored_id:
                if self.thread_id and self.thread_id != stored_id:
                    raise CodexError("configured thread id disagrees with session.json")
                self.thread_id = stored_id
            if self.allow_session_migration and (stored_model != self.model or provider_changed):
                migration = {"at": time.time(), "thread_id": stored_id, "from_model": stored_model,
                             "to_model": self.model, "from_provider": value.get("provider"),
                             "to_provider": (self.provider or {}).get("name"),
                             "conversation_preserved": bool(self.thread_id)}
                with self.session_record.with_name("session-migrations.jsonl").open("a") as stream:
                    stream.write(json.dumps(migration) + "\n")
                value.update(model=self.model, provider=(self.provider or {}).get("name"))
                atomic_json(self.session_record, value)

    def begin_task(self, task_id: str | None) -> None:
        """Resume within a task; retain file history when starting the next task."""
        if self.task_id != task_id:
            # An older task-less session belongs to the currently resumed task.
            if self.task_id is not None or task_id is None:
                self.thread_id = None
            self.task_id = task_id
            atomic_json(
                self.session_record,
                {
                    "schema_version": 1,
                    "thread_id": self.thread_id,
                    "model": self.model,
                    "task_id": task_id,
                    "reasoning_effort": self.reasoning_effort,
                    "provider": (self.provider or {}).get("name"),
                },
            )
        if self._process is None or self._process.poll() is not None:
            self._cancelled = False

    def turn(self, prompt: str) -> CodexTurn:
        if self._cancelled:
            raise CodexError(
                "Codex session was cancelled; create a new session to resume"
            )
        if not prompt.strip():
            raise ValueError("Codex prompt must not be empty")
        try:
            sandbox = CodexFileSandbox.create(self.project_dir, self.runtime_dir)
        except SandboxUnavailable as exc:
            raise CodexError(str(exc)) from exc
        executable = str(sandbox.codex)
        command = sandbox.wrap(self._command(executable, prompt))
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        streamed = False
        try:
            # argv has a per-argument OS limit. A seekable stdin also avoids a
            # pipe writer blocking before the output/deadline monitor starts.
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as prompt_input:
                prompt_input.write(prompt)
                prompt_input.seek(0)
                process = subprocess.Popen(
                    command,
                    cwd=self.project_dir,
                    env=self.environment,
                    stdin=prompt_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                    pass_fds=self.inherited_fds,
                )
            if self.process_record is not None:
                from rpent.utils.execution import process_identity

                atomic_json(self.process_record, process_identity(process.pid))
            self._process = process
            if self._cancelled:
                self.cancel()
                raise CodexError("Codex session was cancelled")
            try:
                if isinstance(process.stdout, io.TextIOBase):
                    chunks: list[str] = []
                    stream_errors: list[Exception] = []

                    def drain_output() -> None:
                        for line in process.stdout:
                            chunks.append(line)
                            try:
                                self._append_log(line)
                            except Exception as error:
                                stream_errors.append(error)

                    reader = threading.Thread(target=drain_output, daemon=True)
                    streamed = True
                    reader.start()
                    try:
                        process.wait(timeout=self.timeout_s)
                    finally:
                        if process.poll() is not None:
                            reader.join(timeout=5)
                    output = "".join(chunks)
                    if reader.is_alive():
                        self.cancel()
                        reader.join(timeout=5)
                        raise CodexError("Codex exited but left its output stream open")
                    if stream_errors:
                        raise CodexError(
                            f"failed to persist Meta output: {stream_errors[0]}"
                        )
                else:
                    output, _ = process.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                self.cancel()
                if streamed:
                    reader.join(timeout=5)
                    output = "".join(chunks)
                else:
                    output, _ = process.communicate()
                    self._append_log(output or "")
                discovered = self.thread_id or self._thread_id(
                    self._events(output or "")
                )
                if discovered:
                    self.thread_id = discovered
                    self._persist_session(discovered, sandbox)
                raise CodexError(
                    f"Codex timed out after {self.timeout_s:.0f}s"
                ) from exc
            except BaseException:
                self.cancel()
                raise
        finally:
            self._process = None
        output = output or ""
        if not streamed:
            self._append_log(output)
        events = self._events(output)
        discovered = self.thread_id or self._thread_id(events)
        last_message = self._last_message(events)
        if discovered:
            self.thread_id = discovered
            self._persist_session(discovered, sandbox)
        if process.returncode != 0:
            # The last assistant message can precede a terminal API failure.
            # Preserve that failure so the worker can classify transient outages.
            terminal_errors = []
            for event in events:
                if event.get("type") == "turn.failed":
                    error = event.get("error", {})
                    message = error.get("message") if isinstance(error, dict) else str(error)
                    if message:
                        terminal_errors.append(message)
                elif event.get("type") == "error" and event.get("message"):
                    terminal_errors.append(event["message"])
            detail = terminal_errors[-1] if terminal_errors else (last_message or output[-1000:])
            raise CodexError(f"Codex exited with {process.returncode}: {detail}")
        if not discovered:
            raise CodexError("Codex completed without reporting a thread id")
        return CodexTurn(discovered, last_message, process.returncode)

    def cancel(self) -> None:
        """Stop the current Meta process group, including its waiting tool clients."""
        self._cancelled = True
        process = self._process
        if process is None:
            return
        # Signal the group even if the parent already exited: a tool subprocess
        # can still own the output pipe and keep communicate() blocked.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # Always reap remaining descendants, including a child ignoring TERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def _command(self, executable: str, prompt: str) -> list[str]:
        del prompt  # The complete prompt is supplied on stdin for start/resume.
        common = [
            executable,
            "exec",
            "-C",
            str(self.project_dir),
            "--sandbox",
            "danger-full-access",
            "-c",
            "shell_environment_policy.inherit=all",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--disable",
            "plugins",
            "--disable",
            "remote_plugin",
            "--disable",
            "multi_agent",
            "--disable",
            "multi_agent_v2",
            "--disable",
            "remote_compaction_v2",
            "-c",
            "features.multi_agent=false",
            "-c",
            "agents.max_threads=1",
            "-c",
            "agents.max_depth=0",
            "-c",
            'web_search="disabled"',
            "--model",
            self.model,
            "--json",
        ]
        common.extend(_mcp_server_flags(self.mcp_servers))
        if not self.thread_id:
            common.extend(f"--image={Path(path).resolve()}" for path in self.initial_images)
        if self.reasoning_effort is not None:
            common.extend(
                ("-c", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}")
            )
        if self.provider:
            name = self.provider["name"]
            common.extend(
                (
                    "-c",
                    f"model_provider={name}",
                    "-c",
                    f"model_providers.{name}.name={name}",
                    "-c",
                    f"model_providers.{name}.base_url={self.provider['base_url']}",
                    "-c",
                    f"model_providers.{name}.env_key={self.provider['env_key']}",
                    "-c",
                    f"model_providers.{name}.wire_api={self.provider.get('wire_api', 'responses')}",
                )
            )
        if self.thread_id:
            return [*common, "resume", self.thread_id, "-"]
        return [*common, "-"]

    def _append_log(self, output: str) -> None:
        with self.event_log.open("a") as stream:
            stream.write(output)
            if output and not output.endswith("\n"):
                stream.write("\n")
        # Persist the identity as soon as Codex announces it, before the first
        # long turn can be interrupted by the broker deadline or an operator.
        if '"thread.started"' in output:
            discovered = self._thread_id(self._events(output))
            if discovered:
                if self.thread_id and self.thread_id != discovered:
                    raise CodexError(
                        "Codex resumed a different thread; refusing to replace this task's session"
                    )
                self.thread_id = discovered
                record = (
                    read_json(self.session_record)
                    if self.session_record.is_file()
                    else {}
                )
                record.update(
                    schema_version=1,
                    thread_id=discovered,
                    model=self.model,
                    task_id=self.task_id,
                    reasoning_effort=self.reasoning_effort,
                    provider=(self.provider or {}).get("name"),
                )
                atomic_json(self.session_record, record)

    def _persist_session(self, thread_id: str, sandbox: CodexFileSandbox) -> None:
        try:
            version = subprocess.run(
                sandbox.version_command(),
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            version = "unknown"
        atomic_json(
            self.session_record,
            {
                "schema_version": 1,
                "thread_id": thread_id,
                "model": self.model,
                "task_id": self.task_id,
                "reasoning_effort": self.reasoning_effort,
                "provider": (self.provider or {}).get("name"),
                "codex_version": version,
            },
        )

    @staticmethod
    def _events(output: str) -> list[dict[str, Any]]:
        events = []
        for line in output.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    @staticmethod
    def _thread_id(events: list[dict[str, Any]]) -> str | None:
        for event in events:
            if event.get("type") == "thread.started" and event.get("thread_id"):
                return str(event["thread_id"])
        return None

    @staticmethod
    def _last_message(events: list[dict[str, Any]]) -> str:
        message = ""
        for event in events:
            item = event.get("item")
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and item.get("text")
            ):
                message = str(item["text"])
        return message
