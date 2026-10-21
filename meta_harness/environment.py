# Copyright 2025 The RPent Team. All rights reserved.
"""Construct intentionally separate lower-agent and Meta-Agent environments."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Mapping

# Process/runtime variables needed by ordinary Python and CUDA subprocesses.
_COMMON_NAMES = {
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}

# Lower-agent credentials and runtime knobs that are safe and necessary for
# every episode. Environment-specific variables must be named explicitly by the
# experiment's ``env_passthrough`` list; prefix-wide inheritance is deliberately
# forbidden because variables such as ``RLDX_VIDEO_DIR`` can redirect run
# artifacts outside ``experiments/<run-id>``.
_LOWER_NAMES = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
}


def _copy_present(source: Mapping[str, str], names: set[str]) -> dict[str, str]:
    return {name: source[name] for name in names if source.get(name) is not None}


def lower_environment(
    *,
    source: Mapping[str, str] | None = None,
    extra_names: tuple[str, ...] = (),
) -> dict[str, str]:
    """Return the allowlisted environment for RPent and its child services."""
    source = os.environ if source is None else source
    names = _COMMON_NAMES | _LOWER_NAMES | set(extra_names)
    result = _copy_present(source, names)
    result["PYTHONUNBUFFERED"] = "1"
    # The lower agent imports RPent directly from the Clean source tree. Keep
    # Python's import cache from becoming an untracked runtime artifact there.
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = Path(__file__).resolve().parents[1]
    result["RPENT_REPO_ROOT"] = str(source_root)
    result["PYTHONPATH"] = os.pathsep.join((str(source_root / "tools/runtime_python"), str(source_root)))
    return result


def meta_environment(
    codex_runtime: Path,
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the campaign environment for the run-local Meta Codex process."""
    source = os.environ if source is None else source
    result = dict(source)
    python_bin = str(Path(sys.executable).parent)
    inherited_path = result.get("PATH", "")
    result["PATH"] = (
        python_bin if not inherited_path else python_bin + os.pathsep + inherited_path
    )
    repository = str(Path(__file__).resolve().parents[1])
    inherited_pythonpath = result.get("PYTHONPATH", "")
    result["RPENT_REPO_ROOT"] = repository
    result["PYTHONPATH"] = str(Path(repository) / "tools/runtime_python") + os.pathsep + (
        repository
        if not inherited_pythonpath
        else repository + os.pathsep + inherited_pythonpath
    )
    result["RPENT_PYTHON"] = sys.executable
    result["CODEX_HOME"] = str(codex_runtime.resolve())
    result["PYTHONUNBUFFERED"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def prepare_codex_auth(
    codex_runtime: Path,
    *,
    source: Mapping[str, str] | None = None,
) -> Path | None:
    """Copy only Codex authentication into a run-local ignored directory.

    No personal configuration, sessions, plugins, MCP servers, or user Skills
    are copied. Existing run-local auth is never overwritten.
    """
    source = os.environ if source is None else source
    codex_runtime.mkdir(parents=True, exist_ok=True)
    ignore = codex_runtime / ".gitignore"
    expected_ignore = "*\n!.gitignore\n"
    if ignore.exists() and ignore.read_text() != expected_ignore:
        raise RuntimeError(f"unexpected existing Codex runtime ignore file: {ignore}")
    if not ignore.exists():
        ignore.write_text(expected_ignore)

    target_auth = codex_runtime / "auth.json"
    if target_auth.is_file():
        target_auth.chmod(0o600)
        return target_auth

    source_home = Path(source.get("CODEX_HOME", str(Path.home() / ".codex")))
    source_auth = source_home / "auth.json"
    if not source_auth.is_file():
        return None
    shutil.copyfile(source_auth, target_auth)
    target_auth.chmod(0o600)
    return target_auth


def environment_presence(
    names: tuple[str, ...],
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    """Describe only presence, never values or hashes, for a preflight report."""
    source = os.environ if source is None else source
    return {name: bool(source.get(name)) for name in names}


def missing_lower_variables(
    required: tuple[str, ...],
    *,
    source: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    presence = environment_presence(required, source=source)
    return tuple(name for name, present in presence.items() if not present)
