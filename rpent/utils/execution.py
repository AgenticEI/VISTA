# Copyright 2026 The RPent Authors.
"""Process-owned capacity slots shared by the host and all episode workers."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import time
from pathlib import Path


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield stream.fileno()


def is_locked(path: Path) -> bool:
    try:
        with file_lock(path, blocking=False):
            return False
    except BlockingIOError:
        return True


def process_identity(pid: int | None = None) -> dict:
    pid = os.getpid() if pid is None else pid
    raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": raw[19],
        "pgid": int(raw[2]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def owns_process(owner: dict) -> bool:
    try:
        raw = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        if raw[0] in {"Z", "X", "x"}:
            return False
        return process_identity(owner["pid"]) == owner
    except (OSError, KeyError, ValueError, IndexError):
        return False


def signal_owned(owner: dict, sig: int = signal.SIGTERM) -> bool:
    if not owns_process(owner) or owner["pgid"] != owner["pid"]:
        return False
    try:
        os.killpg(owner["pgid"], sig)
        return True
    except ProcessLookupError:
        return False


def timing(stage: str, started: float, **fields):
    root = os.environ.get("RPENT_TIMING_DIR")
    if root:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        ended = time.perf_counter()
        value = {"stage": stage, "seconds": ended - started,
                 "start_perf_s": started, "end_perf_s": ended, **fields}
        with (path / f"{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")


async def planner_rate_limit():
    """Pace actual HTTP sends across workers, including SDK transport retries."""
    root = os.environ.get("RPENT_PLANNER_SLOTS")
    rpm = int(os.environ.get("RPENT_PLANNER_RPM", "0"))
    if not root or rpm <= 0:
        return
    path = Path(root) / "rate.json"
    while True:
        with file_lock(Path(root) / "rate.lock"):
            now = time.time()
            try:
                last = json.loads(path.read_text())["last"]
            except (FileNotFoundError, ValueError, KeyError):
                last = 0
            delay = max(0, last + 60.0 / rpm - now)
            if delay == 0:
                path.write_text(json.dumps({"last": now}))
                return
        await asyncio.sleep(min(delay, 0.1))


@contextlib.asynccontextmanager
async def planner_slot():
    """Cover the SDK request including retries; cancellation releases the slot."""
    root = os.environ.get("RPENT_PLANNER_SLOTS")
    if not root:
        started = time.perf_counter()
        try:
            yield
        finally:
            timing("planner_request", started)
        return
    limit = int(os.environ["RPENT_MAX_PLANNER_REQUESTS"])
    started = time.perf_counter()
    stream = None
    try:
        while stream is None:
            for index in range(limit):
                path = Path(root) / str(index)
                path.parent.mkdir(parents=True, exist_ok=True)
                candidate = path.open("a+")
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                else:
                    stream = candidate
                    break
            if stream is None:
                await asyncio.sleep(0.025)
        timing("planner_queue", started)
        started = time.perf_counter()
        try:
            yield
        finally:
            timing("planner_request", started)
    finally:
        if stream is not None:
            stream.close()


def local_service_owner(endpoint: str) -> dict:
    """Identify an existing local legacy service once, without GPU discovery."""
    import re
    import subprocess
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "legacy service requires a local process identity; use registered sessions remotely"
        )
    output = subprocess.check_output(
        ["ss", "-H", "-ltnp", f"sport = :{parsed.port}"], text=True
    )
    pids = set(re.findall(r"pid=(\d+)", output))
    if len(pids) != 1:
        raise RuntimeError(f"cannot identify the legacy service owner at {endpoint}")
    return process_identity(int(pids.pop()))


if __name__ == "__main__":
    # Publish the actual runner's identity before exec. Its inherited lease
    # survives worker loss and exec, so admission lasts until execution ends.
    import sys

    record, inherited_fd, *command = sys.argv[1:]
    os.fstat(int(inherited_fd))
    os.set_inheritable(int(inherited_fd), True)
    path = Path(record)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(process_identity()) + "\n")
    os.replace(temporary, path)
    os.execvpe(command[0], command, os.environ)
