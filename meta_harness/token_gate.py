# Copyright 2026 The RPent Authors.
"""Per-round Meta token gate for token-matched ablations (enabled by token-gate.json)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

BLOCKED = (
    "Submission is not available yet: this development round has used {used:,} Meta tokens "
    "of its {target:,} budget. Continue evidence-driven development in the same workspace: "
    "inspect failing history episodes you have not yet examined, form a new falsifiable "
    "hypothesis, edit the harness, and verify it with autonomous evaluation. Call submit "
    "again after further verified work."
)
CONTINUE = (
    "The provider turn ended before the round budget was used ({used:,}/{target:,} Meta "
    "tokens). The round is not finished. Continue in the same workspace: re-examine failing "
    "episodes, test an alternative hypothesis with autonomous evaluation, and keep the best "
    "verified candidate. Do not submit until the host reports the budget is reached."
)
REACHED = (
    "The round's Meta token budget has been reached ({used:,}/{target:,} Meta tokens). "
    "Finalize and submit your best verified candidate now."
)


def _last_total(rollout):
    try:
        with open(rollout, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4_000_000))
            lines = fh.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        if '"token_count"' in line and '"info":{' in line:
            try:
                return int(json.loads(line)["payload"]["info"]["total_token_usage"]["total_tokens"])
            except (ValueError, KeyError, TypeError):
                continue
    return 0


class TokenGate:
    def __init__(self, attempt):
        path = attempt.store.path("token-gate.json")
        self.spec = json.loads(Path(path).read_text()) if Path(path).exists() else None
        self.attempt = attempt
        ctx = attempt.context
        self.target = None
        if self.spec and ctx["kind"] == "development":
            self.target = self.spec["targets"].get(ctx["scope_id"], {}).get(str(ctx["round_index"]))
        self.upper = int(self.target * self.spec.get("upper_ratio", 1.1)) if self.target else None
        self.db = attempt.store.path(f"workspaces/{ctx['scope_id']}/runtime/state_5.sqlite")
        self.marker = f"Harness for {ctx['task_ids']}; round {ctx['round_index']}, kind development"

    @property
    def enabled(self):
        return self.target is not None

    def used(self):
        if not self.enabled or not os.path.exists(self.db):
            return 0
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True, timeout=5)
        try:
            rows = con.execute(
                "select id, source, rollout_path, first_user_message, tokens_used from threads"
            ).fetchall()
        finally:
            con.close()
        parent, roots, rollout, recorded = {}, set(), {}, {}
        for tid, source, path, first, tokens in rows:
            rollout[tid] = path
            recorded[tid] = int(tokens or 0)
            if source and source.startswith("{"):
                try:
                    parent[tid] = json.loads(source)["subagent"]["thread_spawn"]["parent_thread_id"]
                except (ValueError, KeyError, TypeError):
                    pass
            elif first and self.marker in first:
                roots.add(tid)

        def root(tid):
            seen = set()
            while tid in parent and tid not in seen:
                seen.add(tid)
                tid = parent[tid]
            return tid

        return sum(
            max(recorded[t], _last_total(rollout[t]) if rollout[t] else 0) for t in rollout if root(t) in roots
        )

    def check_submit(self):
        if self.enabled:
            used = self.used()
            if used < self.target:
                raise ValueError(BLOCKED.format(used=used, target=self.target))

    def record(self, used):
        path = self.attempt.store.path(f"{self.attempt.prefix}/token-gate.json")
        Path(path).write_text(json.dumps({"target": self.target, "upper": self.upper, "used": used}) + "\n")
