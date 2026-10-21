# Copyright 2026 The RPent Authors.
"""Run-local launcher for the Meta Codex process."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class SandboxUnavailable(RuntimeError):
    """The host cannot provide the required Meta Agent isolation boundary."""


def _system_executable(name: str) -> Path:
    """Resolve a system executable without consulting personal wrapper paths."""
    executable = (os.environ.get("CODEX_BIN") if name == "codex" else None) or shutil.which(name, path="/usr/local/bin:/usr/bin:/bin")
    if executable is None:
        raise SandboxUnavailable(f"required system executable not found: {name}")
    path = Path(executable)
    if not path.exists() or not os.access(path, os.X_OK):
        raise SandboxUnavailable(f"system executable is not runnable: {path}")
    return path


@dataclass(frozen=True, slots=True)
class CodexFileSandbox:
    """Launch Codex from a run-local project without external file isolation.

    The class name is retained to keep the campaign seam small. Meta currently
    shares the host filesystem and environment so it can execute candidate code
    with the same dependencies as the lower Agent.
    """

    meta_dir: Path
    project_dir: Path
    runtime_dir: Path
    codex: Path

    @classmethod
    def create(cls, project_dir: Path, runtime_dir: Path) -> "CodexFileSandbox":
        project = project_dir.resolve()
        runtime = runtime_dir.resolve()
        meta = project.parent
        if runtime.parent != meta or project.name != "project":
            raise SandboxUnavailable(
                "Codex project and runtime must be sibling directories below run/meta"
            )
        if not project.is_dir() or not runtime.is_dir():
            raise SandboxUnavailable("Codex project/runtime directories are missing")
        return cls(
            meta_dir=meta,
            project_dir=project,
            runtime_dir=runtime,
            codex=_system_executable("codex"),
        )

    def wrap(self, codex_arguments: Sequence[str]) -> list[str]:
        """Return the direct Codex command after validating its executable."""
        if not codex_arguments or Path(codex_arguments[0]) != self.codex:
            raise ValueError("command must invoke the discovered system Codex")
        return list(codex_arguments)

    def version_command(self) -> list[str]:
        """Return a harmless command that proves the isolated CLI can start."""
        return self.wrap((str(self.codex), "--version"))

    def probe(self, *, environment: dict[str, str] | None = None) -> str:
        """Start the CLI without contacting the model provider."""
        try:
            completed = subprocess.run(
                self.version_command(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(
                f"Codex runtime probe failed: {exc}"
            ) from exc
        if completed.returncode:
            detail = (completed.stdout or "").strip()[-1000:]
            raise SandboxUnavailable(
                f"Codex runtime exited with {completed.returncode}: {detail}"
            )
        return (completed.stdout or "").strip()


__all__ = ["CodexFileSandbox", "SandboxUnavailable"]
