# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permission and
# limitations under the License.

"""Memory-root bindings for the shared file tools.

This fork does not port Harness VLA's HF sync or inbox merge. The manager
only roots ``read_text_file`` / ``write_text_file`` / ``list_dir`` on one
corpus so the planner-facing schemas and path checks match VLA eval.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class MemoryManager:
    """Root common file tools on one memory corpus."""

    def __init__(
        self,
        root: str | Path,
        *,
        memory_access: str = "read_only",
        inbox_cell_tag: str | None = None,
        package_memory: bool = False,
    ) -> None:
        self._root = Path(root).resolve()
        self._memory_access = memory_access
        self._inbox_cell_tag = inbox_cell_tag
        self._package_memory = package_memory

    @property
    def root(self) -> Path:
        """Resolved corpus root."""
        return self._root

    def get_common_tool_bindings(
        self,
    ) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
        """Return memory-aware bindings for shared file tools."""
        from functools import partial

        from rpent.memory import tools as memory_tools
        from rpent.tools import common

        handlers = {
            "read_text_file": partial(
                memory_tools.read_text_file,
                memory_root=self._root,
                memory_access=self._memory_access,
                cell_tag=self._inbox_cell_tag,
                package_memory=self._package_memory,
            ),
            "write_text_file": partial(
                memory_tools.write_text_file,
                memory_root=self._root,
                memory_access=self._memory_access,
                cell_tag=self._inbox_cell_tag,
                package_memory=self._package_memory,
            ),
            "list_dir": partial(
                memory_tools.list_dir,
                memory_root=self._root,
                memory_access=self._memory_access,
                cell_tag=self._inbox_cell_tag,
                package_memory=self._package_memory,
            ),
        }
        bindings: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}
        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            handler = handlers.get(name)
            if handler is None:
                continue
            tool_spec = dict(spec)
            tool_spec["description"] = spec["description"] + memory_tools.MEMORY_BOUNDARY_NOTE
            bindings[name] = (tool_spec, handler)
        return bindings
