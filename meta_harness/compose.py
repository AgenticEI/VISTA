# Copyright 2026 The RPent Authors.
"""Git composition for the sole schema-2 evolution loop."""

import subprocess
from collections.abc import Callable
from pathlib import Path


def compose_selected(
    harnesses,
    scope_id: str,
    base: str,
    donors: list[str],
    *,
    before_merge: Callable[[dict], None] | None = None,
) -> dict:
    """Three-way Git merges; preserve deletions, actual parents and conflicts.

    Re-entry detects ancestors already merged. A conflicted worktree is left
    intact for joint Meta; its resolution must finish before baseline scoring.
    """
    workspace = harnesses.workspace(scope_id, base)
    with harnesses._locked():
        pending = harnesses._git(
            "diff", "--name-only", "--diff-filter=U", cwd=workspace
        ).splitlines()
        if pending:
            return {
                "state": "conflict",
                "workspace": str(workspace),
                "files": pending,
                "donors": donors,
            }
        merge_head = Path(
            harnesses._git("rev-parse", "--git-path", "MERGE_HEAD", cwd=workspace)
        )
        if not merge_head.is_absolute():
            merge_head = workspace / merge_head
        if merge_head.exists():
            return {
                "state": "conflict",
                "workspace": str(workspace),
                "files": [],
                "donors": donors,
            }
        for donor in dict.fromkeys(donors):
            harnesses._commit(donor)
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "merge-base",
                    "--is-ancestor",
                    donor,
                    "HEAD",
                ],
                capture_output=True,
            )
            if ancestor.returncode == 0:
                continue
            if ancestor.returncode != 1:
                raise RuntimeError(
                    f"cannot inspect donor ancestry: {ancestor.stderr.decode().strip()}"
                )
            if before_merge is not None:
                before_merge(
                    {
                        "kind": "donor",
                        "scope_id": scope_id,
                        "donor": donor,
                        "head": harnesses._git("rev-parse", "HEAD", cwd=workspace),
                    }
                )
            try:
                harnesses._git("merge", "--no-ff", "--no-edit", donor, cwd=workspace)
            except RuntimeError:
                pending = harnesses._git(
                    "diff", "--name-only", "--diff-filter=U", cwd=workspace
                ).splitlines()
                if not pending:
                    raise
                return {
                    "state": "conflict",
                    "workspace": str(workspace),
                    "files": pending,
                    "donors": donors,
                }
        commit = harnesses._git("rev-parse", "HEAD", cwd=workspace)
        harnesses.retain(commit, f"merges/{commit}")
        return {"state": "complete", "commit": commit, "donors": donors}
