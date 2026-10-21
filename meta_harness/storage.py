# Copyright 2026 The RPent Authors.

"""Small Git policy versions and run-relative, durable JSON records.

This module owns no experiment decisions. The scheduler owns ``state.json``;
workers and Meta use explicit IDs and immutable records in their own directories.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _relative(value: str | Path) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or any(ord(character) < 32 for character in text)
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in {".", ".."} for part in path.parts)
        or not path.parts
    ):
        raise ValueError(f"expected a canonical relative path: {text!r}")
    return text


def _contained(root: Path, relative: str | Path) -> Path:
    pointer = _relative(relative)
    current = root
    for part in PurePosixPath(pointer).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"record/file path must not contain a symlink: {current}")
    return current


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError("a durable record must be a JSON object")
    return (json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _differences(actual: Any, expected: Any, path: str = "$") -> list[str]:
    if type(actual) is not type(expected):
        return [path]
    if isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(actual.keys() | expected.keys()):
            child = f"{path}.{key}"
            if key not in actual or key not in expected:
                differences.append(child)
            else:
                differences.extend(_differences(actual[key], expected[key], child))
        return differences
    if isinstance(actual, list):
        differences = []
        if len(actual) != len(expected):
            differences.append(f"{path}.length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            differences.extend(_differences(left, right, f"{path}[{index}]"))
        return differences
    return [] if actual == expected else [path]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Linking a complete temporary file gives create-once semantics
            # without exposing a partially written destination to another writer.
            os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RecordConflict(ValueError):
    """The same durable ID was supplied with different JSON fields."""

    def __init__(self, relative: str, fields: Sequence[str]) -> None:
        self.relative = relative
        self.fields = tuple(fields)
        super().__init__(f"record conflict in {relative}: {', '.join(fields)}")


class RunStore:
    """Portable records and a process/thread-owned scheduler state lock."""

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).absolute()
        from rpent.utils.run_paths import migration_receipt
        migration = migration_receipt(requested)
        if requested.is_symlink():
            if (not migration or migration.get("kind") != "thin-ssd-run-20261013"
                    or migration.get("original_root") != str(requested)
                    or Path(migration["ssd_root"]).resolve() != requested.resolve()):
                raise ValueError(f"run root must not be a symlink: {requested}")
        requested.mkdir(parents=True, exist_ok=True)
        self.root = requested.resolve()
        self._migration = migration
        self._history_episode_ids = frozenset((migration or {}).get("history_episode_ids", ()))
        self._scheduler_owner: tuple[int, int] | None = None

    def path(self, relative: str | Path) -> Path:
        try:
            return _contained(self.root, relative)
        except ValueError:
            pointer = _relative(relative)
            parts = Path(pointer).parts
            if (self._migration and len(parts) >= 2 and parts[0] == "episodes"
                    and parts[1] in self._history_episode_ids):
                path = self.root / pointer
                permitted = Path(self._migration["history_root"]) / "episodes" / parts[1]
                if path.resolve().is_relative_to(permitted.resolve()):
                    return path
            raise

    def read(self, relative: str | Path) -> dict[str, Any]:
        return _read(self.path(relative))

    def iter_events(self, relative: str | Path) -> Iterator[dict[str, Any]]:
        """Read complete events in order, excluding an unfinished trailing line."""
        pointer = _relative(relative)
        with self.path(pointer).open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith(b"\n"):
                    return
                location = f"{pointer}: line {line_number}"
                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{location}: malformed event JSON: {error}") from error
                if not isinstance(event, dict):
                    raise ValueError(f"{location}: event must be a JSON object")
                if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
                    raise ValueError(f"{location}: schema_version must be 1")
                if type(event.get("seq")) is not int or event["seq"] != line_number:
                    raise ValueError(
                        f"{location}: expected seq {line_number}, got {event.get('seq')!r}"
                    )
                yield event

    def write_once(self, relative: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
        """Create an immutable record, or compare every field on an ID retry."""
        pointer = _relative(relative)
        if pointer in {"state.json", "run.lock"}:
            raise ValueError(f"{pointer} is owned by the scheduler lock")
        path = self.path(pointer)
        payload = _json_bytes(value)
        expected = json.loads(payload)
        try:
            actual = self.read(pointer)
        except FileNotFoundError:
            parts = Path(pointer).parts
            if len(parts) >= 2 and parts[0] == "episodes" and parts[1] in self._history_episode_ids:
                raise PermissionError("historical episode storage is read-only; use a new execution attempt")
            try:
                _publish(path, payload, replace=False)
            except FileExistsError:
                # Another writer may have published since the initial read.
                actual = self.read(pointer)
            else:
                return expected
        differences = _differences(actual, expected)
        if differences:
            raise RecordConflict(pointer, differences) from None
        return actual

    @contextlib.contextmanager
    def scheduler_lock(self) -> Iterator["RunStore"]:
        """Fail promptly when another scheduler already owns this run."""
        path = self.path("run.lock")
        with path.open("a+", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"another scheduler owns {path}") from None
            owner = (os.getpid(), threading.get_ident())
            self._scheduler_owner = owner
            try:
                yield self
            finally:
                self._scheduler_owner = None
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def write_state(self, value: Mapping[str, Any]) -> None:
        if self._scheduler_owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("write_state requires this scheduler's run.lock")
        path = self.path("state.json")
        payload = _json_bytes(value)
        try:
            if path.read_bytes() == payload:
                return
        except FileNotFoundError:
            pass
        _publish(path, payload, replace=True)


class HarnessStore:
    """Git is the sole version/parent authority; paths are materializations only.

    ``files`` always names individual UTF-8 small files, including explicit
    tracked deletions. No recursive staging, resource copies or extra digests
    are used. A workspace is private to its scope; frozen worktrees are shared.
    """

    def __init__(self, root: str | Path, *, max_file_bytes: int = 2 * 1024 * 1024) -> None:
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self.records = RunStore(root)
        self.root = self.records.root
        self.git_dir = self.records.path("harness.git")
        self.max_file_bytes = max_file_bytes
        self._mutex = threading.RLock()
        self._lock_depth = 0

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with self._mutex:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            with self.records.path("harness.lock").open("a+") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _git(self, *arguments: str, cwd: Path | None = None,
             extra_env: Mapping[str, str] | None = None) -> str:
        command = ["git", "--literal-pathspecs", "-c", "user.name=RPent Harness",
                   "-c", "user.email=harness@localhost", "-c", "commit.gpgsign=false",
                   "-c", "core.hooksPath=/dev/null"]
        if cwd is None:
            command += ["--git-dir", str(self.git_dir)]
        command.extend(arguments)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("GIT_")}
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment.update(extra_env or {})
        result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True,
                                text=True, encoding="utf-8", check=False)
        if result.returncode:
            raise RuntimeError(f"Git {' '.join(arguments[:2])} failed: {result.stderr.strip()}")
        return result.stdout.rstrip("\n")

    def _commit(self, value: str) -> str:
        if not isinstance(value, str) or not _COMMIT.fullmatch(value):
            raise ValueError("a policy version must be a full Git commit ID")
        actual = self._git("rev-parse", "--verify", f"{value}^{{commit}}")
        if actual != value:
            raise ValueError(f"expected an exact Git commit: {value}")
        return actual

    def _files(self, root: Path, files: Sequence[str], *, tracked: set[str] | None = None) -> list[str]:
        if isinstance(files, (str, bytes)):
            raise ValueError("files must be an explicit sequence of file paths")
        pointers = [_relative(item) for item in files]
        if len(set(pointers)) != len(pointers):
            raise ValueError("files must not contain duplicate paths")
        for pointer in pointers:
            if ".git" in PurePosixPath(pointer).parts:
                raise ValueError("Git metadata cannot be staged as policy content")
            path = _contained(root, pointer)
            if not path.exists() and tracked is not None and pointer in tracked:
                continue
            if not path.is_file():
                raise ValueError(f"policy files must name regular files: {pointer}")
            if path.stat().st_size > self.max_file_bytes:
                raise ValueError(f"large resources must stay outside Git: {pointer}")
            payload = path.read_bytes()
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"binary resources must stay outside Git: {pointer}") from None
            if b"\0" in payload:
                raise ValueError(f"binary resources must stay outside Git: {pointer}")
        return pointers

    def initialize(self, source: str | Path, *, files: Sequence[str]) -> str:
        """Record an initial small policy directly from an explicit source list."""
        source = Path(source).absolute()
        if source.is_symlink() or not source.is_dir():
            raise ValueError("initial policy source must be a regular directory")
        source = source.resolve()
        pointers = self._files(source, files)
        if not pointers:
            raise ValueError("an initial policy needs at least one explicit file")
        with self._locked():
            if not self.git_dir.exists():
                self._git("init", "--bare", "--quiet", str(self.git_dir), cwd=self.root)
            existing = self._git("for-each-ref", "--format=%(objectname)", "refs/heads/initial")
            if existing:
                previous = self._git("ls-tree", "-r", "--name-only", existing).splitlines()
                if set(previous) != set(pointers):
                    raise ValueError("initial policy file list conflicts with its recorded Git commit")
                for pointer in pointers:
                    # Git's own stored bytes are the comparison; no second identity.
                    actual = subprocess.run(
                        ["git", "--git-dir", str(self.git_dir), "show", f"{existing}:{pointer}"],
                        capture_output=True, check=True,
                    ).stdout
                    if actual != (source / pointer).read_bytes():
                        raise ValueError(f"initial policy conflicts at {pointer}")
                return existing
            descriptor, index = tempfile.mkstemp(prefix=".initial-index-", dir=self.root)
            os.close(descriptor)
            os.unlink(index)
            try:
                environment = {"GIT_INDEX_FILE": index}
                self._git("read-tree", "--empty", extra_env=environment)
                self._git("--work-tree", str(source), "add", "--all", "--", *pointers,
                          extra_env=environment)
                tree = self._git("write-tree", extra_env=environment)
                commit = self._git("commit-tree", tree, "-m", "Initial Harness")
                self._git("update-ref", "refs/heads/initial", commit)
                self._git("symbolic-ref", "HEAD", "refs/heads/initial")
                self.retain(commit, "initial")
                return commit
            finally:
                for path in (Path(index), Path(index + ".lock")):
                    if path.exists():
                        path.unlink()

    def workspace(self, scope_id: str, parent_commit: str) -> Path:
        """Create a scope worktree once; reopening never resets its draft."""
        scope = _relative(scope_id)
        branch = f"scopes/{scope}"
        self._git("check-ref-format", f"refs/heads/{branch}")
        with self._locked():
            parent = self._commit(parent_commit)
            path = self.records.path(f"workspaces/{scope}/candidate")
            if path.exists():
                self._workspace(path)
                return path
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._git("for-each-ref", "--format=%(objectname)", f"refs/heads/{branch}")
            if existing:
                self._git("worktree", "add", str(path), branch)
            else:
                self._git("worktree", "add", "-b", branch, str(path), parent)
            return path

    def _workspace(self, workspace: str | Path) -> Path:
        requested = Path(workspace).absolute()
        try:
            relative = requested.relative_to(self.root)
        except ValueError:
            raise ValueError("workspace must belong to this run") from None
        path = self.records.path(relative)
        if relative.parts[0] != "workspaces" or path.name != "candidate":
            raise ValueError("expected a scope's candidate worktree")
        common = Path(self._git("rev-parse", "--git-common-dir", cwd=path))
        if not common.is_absolute():
            common = path / common
        if common.resolve() != self.git_dir:
            raise ValueError("workspace belongs to another Git repository")
        return path

    def freeze(self, workspace: str | Path, *, files: Sequence[str], message: str) -> str:
        """Commit explicit changes on the actual current parent and retain them."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("a freeze needs a commit message")
        with self._locked():
            path = self._workspace(workspace)
            if self._git("diff", "--name-only", "--diff-filter=U", cwd=path):
                raise ValueError("resolve and stage all merge entries explicitly before freeze")
            tracked = set(filter(None, self._git("ls-files", "-z", cwd=path).split("\0")))
            pointers = self._files(path, files, tracked=tracked)
            changed = set(filter(None, self._git("diff", "--name-only", "-z", "HEAD", cwd=path).split("\0")))
            omitted = changed - set(pointers)
            if omitted:
                raise ValueError(f"explicit files omit tracked changes: {', '.join(sorted(omitted))}")
            if pointers:
                self._git("add", "--all", "--", *pointers, cwd=path)
            merge_head = Path(self._git("rev-parse", "--git-path", "MERGE_HEAD", cwd=path))
            if not merge_head.is_absolute():
                merge_head = path / merge_head
            if self._git("diff", "--cached", "--name-only", cwd=path) or merge_head.exists():
                self._git("commit", "--quiet", "-m", message, cwd=path)
            commit = self._git("rev-parse", "HEAD", cwd=path)
            self.retain(commit, f"candidates/{commit}")
            return commit

    def select_parent(self, workspace: str | Path, parent_commit: str, *,
                      files: Sequence[str], message: str = "Save draft before changing parent") -> str:
        """Save and retain all draft changes before moving this scope to a parent.

        Returns the saved draft commit so Meta can record the selection event.
        Unknown/unlisted files are never removed by a parent switch.
        """
        with self._locked():
            parent = self._commit(parent_commit)
            path = self._workspace(workspace)
            untracked = set(filter(None, self._git("ls-files", "--others", "--exclude-standard", "-z", cwd=path).split("\0")))
            omitted = untracked - set(files)
            if omitted:
                raise ValueError(f"save or move untracked files before changing parent: {', '.join(sorted(omitted))}")
            saved = self.freeze(path, files=files, message=message)
            self._git("reset", "--hard", parent, cwd=path)
            return saved

    def retain(self, commit: str, name: str) -> None:
        """Keep rejected candidates/drafts reachable without marking a winner."""
        name = _relative(name)
        reference = f"refs/retained/{name}"
        with self._locked():
            commit = self._commit(commit)
            self._git("check-ref-format", reference)
            existing = self._git("for-each-ref", "--format=%(objectname)", reference)
            if existing and existing != commit:
                raise ValueError(f"retained ref already identifies another commit: {name}")
            self._git("update-ref", reference, commit)

    def parents(self, commit: str) -> tuple[str, ...]:
        commit = self._commit(commit)
        return tuple(self._git("show", "-s", "--format=%P", commit).split())

    def materialize(self, commit: str) -> Path:
        """Reuse one numbered, read-only Git worktree per immutable commit."""
        with self._locked():
            commit = self._commit(commit)
            frozen = self.records.path("frozen")
            frozen.mkdir(exist_ok=True)
            worktrees = self._git("worktree", "list", "--porcelain")
            for entry in worktrees.split("\n\n"):
                values = dict(line.split(" ", 1) for line in entry.splitlines() if " " in line)
                if values.get("HEAD") != commit or "worktree" not in values:
                    continue
                path = Path(values["worktree"])
                if path.parent != frozen:
                    continue
                self.records.path(path.relative_to(self.root))
                if not path.is_dir():
                    raise RuntimeError(f"registered frozen worktree is missing: {path}")
                if self._git("status", "--porcelain", "--untracked-files=all", cwd=path):
                    raise RuntimeError(f"frozen worktree was modified: {path}")
                self._read_only(path)
                return path
            numbers = [int(path.name[1:]) for path in frozen.iterdir()
                       if re.fullmatch(r"f\d+", path.name)]
            path = frozen / f"f{max(numbers, default=0) + 1:03d}"
            self._git("worktree", "add", "--detach", str(path), commit)
            self._git("worktree", "lock", "--reason", "shared frozen Harness", str(path))
            self._read_only(path)
            return path

    @staticmethod
    def _read_only(root: Path) -> None:
        # This also removes write bits from the .git pointer. Git's worktree
        # metadata remains outside the frozen policy and is managed by Git.
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"frozen policy must not contain symlinks: {path}")
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
