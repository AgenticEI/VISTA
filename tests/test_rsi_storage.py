# Copyright 2026 The RPent Authors.

"""Real Git and interruption checks for the first RSI refactor checkpoint."""

from __future__ import annotations

import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from meta_harness.storage import HarnessStore, RecordConflict, RunStore
from meta_harness import storage


class RunStoreTests(unittest.TestCase):
    def test_existing_records_and_unchanged_state_do_not_republish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(temporary)
            store.write_once("result.json", {"count": 1})
            with store.scheduler_lock():
                store.write_state({"round": 1})
                with patch("meta_harness.storage._publish") as publish:
                    self.assertEqual(store.write_once("result.json", {"count": 1}), {"count": 1})
                    store.write_state({"round": 1})
                    with self.assertRaises(RecordConflict):
                        store.write_once("result.json", {"count": True})
                    publish.assert_not_called()
                store.write_state({"round": True})
                self.assertIs(store.read("state.json")["round"], True)
            with self.assertRaisesRegex(RuntimeError, "run.lock"):
                store.write_state({"round": True})

    def test_record_published_after_initial_read_is_still_compared(self) -> None:
        for concurrent_value in ({"count": 1}, {"count": True}):
            with self.subTest(concurrent_value=concurrent_value), tempfile.TemporaryDirectory() as temporary:
                store = RunStore(temporary)
                publish = storage._publish

                def publish_after_competitor(path, payload, *, replace):
                    publish(path, storage._json_bytes(concurrent_value), replace=False)
                    publish(path, payload, replace=replace)

                with patch("meta_harness.storage._publish", side_effect=publish_after_competitor):
                    if type(concurrent_value["count"]) is bool:
                        with self.assertRaises(RecordConflict):
                            store.write_once("result.json", {"count": 1})
                    else:
                        self.assertEqual(store.write_once("result.json", {"count": 1}), {"count": 1})
                self.assertEqual(list(Path(temporary).glob(".result.json.*")), [])

    def test_config_is_fixed_and_reports_the_conflicting_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(temporary)
            original = {"schema_version": 1, "cells": [{"seed": 4, "repeat": 0}],
                        "meta": {"model": "existing-model"}}
            store.write_once("config.json", original)
            RunStore(temporary).write_once("config.json", dict(original))
            changed = {**original, "cells": [{"seed": 5, "repeat": False}]}
            with self.assertRaises(RecordConflict) as caught:
                store.write_once("config.json", changed)
            self.assertEqual(caught.exception.fields, ("$.cells[0].repeat", "$.cells[0].seed"))
            self.assertEqual(store.read("config.json"), original)
            self.assertEqual(list(Path(temporary).glob(".config.json.*")), [])

    def test_state_has_one_process_and_thread_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(temporary)
            with self.assertRaisesRegex(RuntimeError, "run.lock"):
                store.write_state({"completed_rounds": 0})
            with self.assertRaisesRegex(ValueError, "scheduler"):
                store.write_once("state.json", {})
            errors: list[Exception] = []

            def write_from_another_thread() -> None:
                try:
                    store.write_state({"wrong_writer": True})
                except Exception as error:
                    errors.append(error)

            with store.scheduler_lock():
                store.write_state({"completed_rounds": 1})
                with self.assertRaisesRegex(RuntimeError, "another scheduler"):
                    with RunStore(temporary).scheduler_lock():
                        self.fail("a second scheduler acquired the lock")
                thread = threading.Thread(target=write_from_another_thread)
                thread.start()
                thread.join(timeout=5)
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], RuntimeError)
                self.assertEqual(store.read("state.json"), {"completed_rounds": 1})
            with RunStore(temporary).scheduler_lock() as resumed:
                resumed.write_state({"completed_rounds": 2})
            self.assertEqual(store.read("state.json"), {"completed_rounds": 2})

    def test_killed_scheduler_releases_os_lock_without_removing_its_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code = (
                "import sys, threading\n"
                "from meta_harness.storage import RunStore\n"
                "store = RunStore(sys.argv[1])\n"
                "with store.scheduler_lock():\n"
                "    store.write_state({'phase': 'evaluating_candidate'})\n"
                "    print('ready', flush=True)\n"
                "    threading.Event().wait(30)\n"
            )
            child = subprocess.Popen([sys.executable, "-c", code, temporary],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaisesRegex(RuntimeError, "another scheduler"):
                    with RunStore(temporary).scheduler_lock():
                        self.fail("a live scheduler's lock was acquired")
                child.kill()
                child.wait(timeout=5)
                with RunStore(temporary).scheduler_lock() as resumed:
                    self.assertEqual(resumed.read("state.json")["phase"], "evaluating_candidate")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                child.stdout.close()
                child.stderr.close()

    def test_failed_atomic_state_replace_keeps_the_previous_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(temporary)
            with store.scheduler_lock():
                store.write_state({"round": 1})
                with patch("meta_harness.storage.os.replace", side_effect=OSError("interrupted")):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        store.write_state({"round": 2})
            self.assertEqual(RunStore(temporary).read("state.json"), {"round": 1})
            self.assertEqual(list(Path(temporary).glob(".state.json.*")), [])

    def test_records_cannot_escape_or_follow_resource_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = RunStore(root / "run")
            (root / "outside").mkdir()
            (store.root / "alias").symlink_to(root / "outside", target_is_directory=True)
            for pointer in ("../outside/record.json", "/tmp/record.json", "alias/record.json"):
                with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                    store.write_once(pointer, {"x": 1})
            self.assertEqual(list((root / "outside").iterdir()), [])

    def test_event_recovery_ignores_only_the_unfinished_tail_and_rejects_bad_complete_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(temporary)
            pointer = "episodes/ep001/events.jsonl"
            path = store.path(pointer)
            path.parent.mkdir(parents=True)
            first = b'{"schema_version":1,"seq":1,"type":"start"}\n'
            second = b'{"schema_version":1,"seq":2,"type":"tool_call"}\n'
            # The incomplete final character is deliberate: text-mode buffered
            # decoding must not make earlier complete events unreadable.
            path.write_bytes(first + second + b'{"schema_version":1,"seq":3,"text":"\xe4\xb8')
            self.assertEqual([event["seq"] for event in store.iter_events(pointer)], [1, 2])
            invalid_lines = {
                "broken_json": (b'{"schema_version":1,"seq":\n', "malformed event JSON"),
                "sequence_gap": (b'{"schema_version":1,"seq":3}\n', "expected seq 2"),
                "wrong_schema": (b'{"schema_version":2,"seq":2}\n', "schema_version must be 1"),
            }
            for label, (line, message) in invalid_lines.items():
                with self.subTest(label=label):
                    path.write_bytes(first + line)
                    events = store.iter_events(pointer)
                    self.assertEqual(next(events)["seq"], 1)
                    with self.assertRaisesRegex(ValueError, f"line 2: {message}"):
                        next(events)


class HarnessStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "HARNESS.json").write_text('{"schema_version": 1, "id": "H0"}\n')
        (self.source / "prompt.md").write_text("Initial policy\n")
        (self.source / "large-resource.bin").write_bytes(b"\0" * 512)
        self.files = ["HARNESS.json", "prompt.md"]
        self.store = HarnessStore(self.root / "run")
        self.initial = self.store.initialize(self.source, files=self.files)

    def tearDown(self) -> None:
        # frozen worktrees are intentionally read-only; restore write bits only
        # in this test's temporary tree to let TemporaryDirectory remove it.
        for path in self.root.rglob("*"):
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        self.temporary.cleanup()

    def git(self, *arguments: str) -> str:
        return subprocess.check_output(["git", "--git-dir", str(self.store.git_dir), *arguments],
                                       text=True).strip()

    def test_initial_commit_records_only_explicit_small_files_and_is_reopenable(self) -> None:
        self.assertEqual(set(self.git("ls-tree", "-r", "--name-only", self.initial).splitlines()),
                         set(self.files))
        self.assertEqual(self.store.parents(self.initial), ())
        self.assertEqual(HarnessStore(self.store.root).initialize(self.source, files=self.files),
                         self.initial)
        self.assertTrue((self.source / "large-resource.bin").is_file())
        (self.source / "prompt.md").write_text("Uncommitted source change\n")
        with self.assertRaisesRegex(ValueError, "prompt.md"):
            self.store.initialize(self.source, files=self.files)
        self.assertEqual(self.git("show", f"{self.initial}:prompt.md"), "Initial policy")

    def test_rejected_parent_and_intermediate_freezes_keep_the_real_git_chain(self) -> None:
        first = self.store.workspace("atomic-seen/task-a", self.initial)
        second = self.store.workspace("atomic-seen/task-b", self.initial)
        (first / "prompt.md").write_text("Rejected proposal\n")
        rejected = self.store.freeze(first, files=self.files, message="First proposal")
        self.store.retain(rejected, "rejected/a001")
        (first / "prompt.md").write_text("Development trial\n")
        trial = self.store.freeze(first, files=self.files, message="Freeze small test")
        (first / "prompt.md").write_text("Final proposal after trial\n")
        final = self.store.freeze(first, files=self.files, message="Final candidate")
        self.assertEqual(self.store.parents(rejected), (self.initial,))
        self.assertEqual(self.store.parents(trial), (rejected,))
        self.assertEqual(self.store.parents(final), (trial,))
        self.assertEqual((second / "prompt.md").read_text(), "Initial policy\n")

        (second / "prompt.md").write_text("Unsaved research on other branch\n")
        saved = self.store.select_parent(second, rejected, files=self.files)
        self.assertEqual(self.store.parents(saved), (self.initial,))
        self.assertEqual(self.git("show", f"{saved}:prompt.md"), "Unsaved research on other branch")
        self.assertEqual((second / "prompt.md").read_text(), "Rejected proposal\n")
        (second / "prompt.md").write_text("Improvement developed from rejection\n")
        inherited = self.store.freeze(second, files=self.files, message="Inherit rejected candidate")
        self.assertEqual(self.store.parents(inherited), (rejected,))
        self.assertEqual(self.git("rev-parse", "refs/retained/rejected/a001"), rejected)
        self.assertEqual(self.git("rev-parse", f"refs/retained/candidates/{saved}"), saved)

    def test_workspace_reopen_and_frozen_reuse_preserve_draft_isolation(self) -> None:
        workspace = self.store.workspace("atomic-seen/task", self.initial)
        frozen = self.store.materialize(self.initial)
        (workspace / "prompt.md").write_text("Still editing\n")
        reopened = HarnessStore(self.store.root)
        self.assertEqual(reopened.workspace("atomic-seen/task", self.initial), workspace)
        self.assertEqual((workspace / "prompt.md").read_text(), "Still editing\n")
        self.assertEqual(reopened.materialize(self.initial), frozen)
        self.assertEqual((frozen / "prompt.md").read_text(), "Initial policy\n")
        self.assertEqual(frozen.name, "f001")
        self.assertEqual(list(frozen.parent.iterdir()), [frozen])
        for path in (frozen, frozen / "prompt.md", frozen / ".git"):
            self.assertEqual(path.stat().st_mode & 0o222, 0)

    def test_materialization_recovers_a_worktree_created_before_interruption(self) -> None:
        with patch.object(self.store, "_read_only", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.store.materialize(self.initial)
        recovered = HarnessStore(self.store.root).materialize(self.initial)
        self.assertEqual(recovered.name, "f001")
        self.assertEqual(list(recovered.parent.iterdir()), [recovered])
        self.assertEqual(recovered.stat().st_mode & 0o222, 0)

    def test_concurrent_consumers_share_one_frozen_worktree(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as workers:
            paths = list(workers.map(lambda _: HarnessStore(self.store.root).materialize(self.initial),
                                     range(4)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(list(paths[0].parent.iterdir()), [paths[0]])

    def test_explicit_filenames_are_not_git_glob_patterns(self) -> None:
        workspace = self.store.workspace("task", self.initial)
        (workspace / "notes[1].md").write_text("Selected file\n")
        (workspace / "notes1.md").write_text("Unselected draft\n")
        commit = self.store.freeze(workspace, files=["notes[1].md"], message="One explicit path")
        self.assertIn("notes[1].md", self.git("ls-tree", "-r", "--name-only", commit).splitlines())
        self.assertNotIn("notes1.md", self.git("ls-tree", "-r", "--name-only", commit).splitlines())

    def test_missing_files_and_modifications_do_not_silently_redefine_frozen_commit(self) -> None:
        frozen = self.store.materialize(self.initial)
        prompt = frozen / "prompt.md"
        prompt.chmod(0o644)
        prompt.write_text("Changed behind the store\n")
        with self.assertRaisesRegex(RuntimeError, "modified"):
            HarnessStore(self.store.root).materialize(self.initial)

    def test_large_binary_and_symlink_inputs_never_reach_git(self) -> None:
        workspace = self.store.workspace("task", self.initial)
        (workspace / "binary.dat").write_bytes(b"binary\0data")
        (workspace / "large.txt").write_text("x" * 2048)
        (workspace / "linked.md").symlink_to(self.source / "prompt.md")
        limited = HarnessStore(self.store.root, max_file_bytes=1024)
        for pointer in ("binary.dat", "large.txt", "linked.md", ".git", "../escape.md"):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                limited.freeze(workspace, files=[pointer], message="Must not publish resource")
        self.assertEqual(self.git("rev-parse", "refs/heads/scopes/task"), self.initial)
        self.assertEqual(set(self.git("ls-tree", "-r", "--name-only", self.initial).splitlines()),
                         set(self.files))

    def test_deletions_are_explicit_and_unlisted_changes_are_not_lost(self) -> None:
        workspace = self.store.workspace("task", self.initial)
        (workspace / "prompt.md").unlink()
        with self.assertRaisesRegex(ValueError, "omit tracked changes"):
            self.store.freeze(workspace, files=["HARNESS.json"], message="Missing deletion")
        deleted = self.store.freeze(workspace, files=self.files, message="Remove obsolete prompt")
        self.assertEqual(self.git("ls-tree", "-r", "--name-only", deleted), "HARNESS.json")
        self.assertEqual(self.store.parents(deleted), (self.initial,))
        (workspace / "notes.md").write_text("Unlisted new draft\n")
        with self.assertRaisesRegex(ValueError, "untracked"):
            self.store.select_parent(workspace, self.initial, files=["HARNESS.json"])
        self.assertEqual((workspace / "notes.md").read_text(), "Unlisted new draft\n")
        self.assertFalse((workspace / "prompt.md").exists())

    def test_parent_switch_interruption_still_preserves_the_saved_draft(self) -> None:
        workspace = self.store.workspace("task", self.initial)
        (workspace / "prompt.md").write_text("Save before switching\n")
        original_git = self.store._git

        def interrupt_reset(*arguments: str, **kwargs: object) -> str:
            if arguments[:2] == ("reset", "--hard"):
                raise RuntimeError("interrupted before switch")
            return original_git(*arguments, **kwargs)

        with patch.object(self.store, "_git", side_effect=interrupt_reset):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.store.select_parent(workspace, self.initial, files=self.files)
        saved = self.git("rev-parse", "refs/heads/scopes/task")
        self.assertEqual(self.git("show", f"{saved}:prompt.md"), "Save before switching")
        self.assertEqual(self.git("rev-parse", f"refs/retained/candidates/{saved}"), saved)
        self.assertEqual(HarnessStore(self.store.root).select_parent(workspace, self.initial,
                                                                   files=self.files), saved)


if __name__ == "__main__":
    unittest.main()
