# Copyright 2025 The RPent Team. All rights reserved.
"""File-backed Meta Harness evolution loop.

The package deliberately sits outside :mod:`rpent`: RPent consumes one Harness
directory and produces one Result directory, while this package only arranges
those files and invokes RPent as a subprocess.
"""

from typing import Any

__all__ = ["EvolutionLoop", "EvolutionConfig", "TaskCase"]


def __getattr__(name: str) -> Any:
    if name == "EvolutionLoop":
        from meta_harness.loop import EvolutionLoop

        return EvolutionLoop
    if name in {"EvolutionConfig", "TaskCase"}:
        from meta_harness import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
