# Copyright 2026 The RPent Authors.
"""Capture the small runtime source with Git, including authorized local edits."""

import os
import subprocess

from meta_harness.storage import HarnessStore, RunStore


def record_runtime(root, config):
    repository = config.repo_root
    candidates = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
         "rpent", "meta_harness", "robots", "pyproject.toml", "uv.lock", "configs/env.sh",
         "tools/run-rsi.sh", "tools/run-libero-pipeline.sh", "tools/launch_libero_pipeline.py", "tools/run-libero-suite.sh", "tools/resume_libero_suite.py", "tools/export_libero_suite.py", "tools/runtime_python", "tools/prepare_libero_pro.py", "tools/batch2_supervisor.py", "tools/run_batch2_supervisor.sh", "tools/run_robotwin_campaign_member.py", "tools/run_robotwin_campaign_member.sh", "tools/prepare_robotwin_resume_20261013.py"],
        cwd=repository, check=True, capture_output=True,
    ).stdout.decode().split("\0")
    suffixes = {".py", ".json", ".md", ".toml", ".lock", ".sh", ".yaml", ".yml",
                ".html", ".css", ".js", ".j2"}
    files = sorted({name for name in candidates if name
                    and (repository / name).suffix in suffixes
                    and (repository / name).is_file()})
    store = RunStore(root)
    from meta_harness.resource_profile import current_revision
    profile_revision = current_revision(store)
    revision = (store.read(f"pool/resource-profiles/{profile_revision}.json")["runtime_revision"]
                if profile_revision else None)
    if store.path("runtime-current.json").exists():
        from meta_harness.evaluation import _record_id
        revision = _record_id(store.read("runtime-current.json")["revision"], "runtime revision")
    source_ref = f"runtime-source-revisions/{revision}" if revision else "runtime-source"
    record_ref = f"runtime-source-revisions/{revision}.json" if revision else "runtime-source.json"
    source = HarnessStore(store.path(source_ref))
    # Git compares every recorded byte when reopening an existing snapshot.
    # Resume therefore cannot silently switch runtime code between candidates.
    commit = source.initialize(repository, files=files)
    store.write_once(record_ref, {"commit": commit, "files": files})
    names = ("RPENT_PYTHON", "RPENT_ENV_PYTHON", "RPENT_VLA_PYTHON", "RPENT_SAM3_PYTHON",
             "RPENT_VLA_MODEL_PATH", "RLDX_BASE_VLM_PATH", "SAM3_CHECKPOINT_PATH",
             "SAM3_BPE_PATH", "RLDX_ATTN_IMPL", "HF_HUB_OFFLINE", "PI05_CHECKPOINT_PATH",
             "LIBERO_TYPE", "RPENT_REPO_ROOT", "RPENT_RLINF_ROOT", "LIBEROPRO_CONFIG_PATH")
    store.write_once("runtime-environment.json", {
        "paths_and_settings": {name: os.environ.get(name) for name in names},
        "resources": "External model files are registered by path; source Git contains no weights or credentials.",
    })
    return commit
