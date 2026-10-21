# Copyright 2026 The RPent Authors.
"""Current Meta/episode environment and local authentication contracts."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from meta_harness.environment import (
    lower_environment,
    meta_environment,
    prepare_codex_auth,
)


class EnvironmentTests(unittest.TestCase):
    def test_meta_inherits_campaign_environment_and_uses_campaign_python(self) -> None:
        source = {
            "PATH": "/bin",
            "OPENAI_API_KEY": "lower-secret",
            "OPENAI_BASE_URL": "https://lower.invalid",
            "RLDX_ATTN_IMPL": "sdpa",
            "HF_HUB_OFFLINE": "1",
            "PERSONAL_SETTING": "do-not-copy",
        }
        lower = lower_environment(
            source=source,
            extra_names=("RLDX_ATTN_IMPL", "HF_HUB_OFFLINE"),
        )
        meta = meta_environment(Path("/tmp/meta-runtime"), source=source)
        self.assertEqual(lower["OPENAI_API_KEY"], "lower-secret")
        self.assertEqual(lower["RLDX_ATTN_IMPL"], "sdpa")
        self.assertEqual(meta["OPENAI_API_KEY"], "lower-secret")
        self.assertEqual(meta["OPENAI_BASE_URL"], "https://lower.invalid")
        self.assertEqual(meta["RLDX_ATTN_IMPL"], "sdpa")
        self.assertEqual(meta["HF_HUB_OFFLINE"], "1")
        self.assertEqual(meta["PERSONAL_SETTING"], "do-not-copy")
        self.assertEqual(meta["RPENT_PYTHON"], sys.executable)
        self.assertEqual(meta["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent))
        self.assertEqual(
            Path(meta["PYTHONPATH"].split(os.pathsep)[0]),
            Path(__file__).resolve().parents[1] / "tools/runtime_python",
        )
        self.assertEqual(Path(meta["RPENT_REPO_ROOT"]), Path(__file__).resolve().parents[1])
        self.assertEqual(meta["CODEX_HOME"], "/tmp/meta-runtime")
        self.assertEqual(meta["PYTHONDONTWRITEBYTECODE"], "1")

    def test_lower_environment_requires_explicit_names_and_contains_bytecode(
        self,
    ) -> None:
        source = {
            "PATH": "/bin",
            "OPENAI_API_KEY": "lower-secret",
            "RLDX_ATTN_IMPL": "sdpa",
            "RLDX_VIDEO_DIR": "/outside/run",
            "ROBOCASA_EXPERIMENT_OUTPUT": "/outside/robocasa",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "0",
        }

        lower = lower_environment(
            source=source,
            extra_names=("RLDX_ATTN_IMPL", "HF_HUB_OFFLINE"),
        )

        self.assertEqual(lower["RLDX_ATTN_IMPL"], "sdpa")
        self.assertEqual(lower["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("RLDX_VIDEO_DIR", lower)
        self.assertNotIn("ROBOCASA_EXPERIMENT_OUTPUT", lower)
        self.assertEqual(lower["PYTHONDONTWRITEBYTECODE"], "1")

    def test_only_auth_is_copied_to_run_local_codex_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_home = root / "source-codex"
            source_home.mkdir()
            (source_home / "auth.json").write_text("secret-auth")
            (source_home / "config.toml").write_text("personal=true")
            (source_home / "sessions").mkdir()
            target = root / "experiments" / "run" / "meta" / "codex-runtime"
            auth = prepare_codex_auth(target, source={"CODEX_HOME": str(source_home)})
            self.assertEqual(auth, target / "auth.json")
            self.assertEqual(auth.read_text(), "secret-auth")
            self.assertEqual(os.stat(auth).st_mode & 0o777, 0o600)
            self.assertFalse((target / "config.toml").exists())
            self.assertFalse((target / "sessions").exists())
            self.assertEqual((target / ".gitignore").read_text(), "*\n!.gitignore\n")
