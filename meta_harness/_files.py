# Copyright 2026 The RPent Authors.
"""Small file and path helpers for the current Meta runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return the repository containing this package."""
    return Path(__file__).resolve().parents[1]


def require_contained_path(
    path: str | Path,
    *,
    anchor: str | Path,
    label: str = "path",
) -> Path:
    """Return a lexical path below ``anchor`` without following output symlinks.

    Unlike a plain ``resolve().relative_to(...)`` check, this function does not
    turn a symlinked output tree into a new trusted anchor.  Missing suffixes are
    allowed so callers can validate a destination before creating it.
    """
    anchor_path = Path(os.path.abspath(Path(anchor).expanduser()))
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = anchor_path / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(anchor_path)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {anchor_path}: {candidate}") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} is not canonical below {anchor_path}: {path}")

    current = anchor_path
    if current.is_symlink():
        raise ValueError(f"{label} anchor must not be a symlink: {current}")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} must not pass through a symlink: {current}")
    try:
        candidate.resolve().relative_to(anchor_path.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside {anchor_path}: {candidate}") from exc
    return candidate


def resolve_repo_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a repository-owned configuration path."""
    repository = (root or repo_root()).resolve()
    path = Path(value).expanduser()
    return (repository / path).resolve() if not path.is_absolute() else path.resolve()


def atomic_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file after creating its parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
