# Copyright 2026 The RPent Authors.
"""Binding preview using the production schemas; evidence uses live bindings."""

import tempfile
from pathlib import Path

from rpent.harness import bind_harness, inspect_harness, load_harness


def inspect_package(path: Path, commit=None, *, robot="robocasa"):
    from rpent.robots import get_robot_spec

    spec = get_robot_spec(robot)
    if spec.make_harness_preview is None or spec.make_harness_adapter is None:
        raise ValueError(f"{robot} does not provide a Harness binding preview")
    with tempfile.TemporaryDirectory(prefix="rpent-binding-") as folder:
        toolkit = spec.make_harness_preview(Path(folder))
        adapter = spec.make_harness_adapter(toolkit)
        bound = bind_harness(
            load_harness(path, commit=commit),
            toolkit=toolkit,
            capabilities=adapter.capabilities(),
            allowed_native_tools=adapter.allowed_native_tools(),
            episode_active=lambda: False,
            episode_dir=Path(folder),
        )
        try:
            return inspect_harness(bound)
        finally:
            bound.close()


def validate_package(path: Path, *, timeout_s: float = 10, robot="robocasa"):
    """Candidate module initialization cannot consume the host's cleanup budget."""
    import subprocess
    import sys
    from meta_harness.environment import lower_environment

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "meta_harness.inspection", str(path), robot],
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout_s),
            cwd=Path(__file__).resolve().parents[1],
            env=lower_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            "candidate binding exceeded the validation time limit"
        ) from exc
    if completed.returncode:
        raise ValueError("candidate is not loadable: " + completed.stderr[-2000:])


if __name__ == "__main__":
    import sys

    inspect_package(Path(sys.argv[1]), robot=sys.argv[2] if len(sys.argv) > 2 else "robocasa")
