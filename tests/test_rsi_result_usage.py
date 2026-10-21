"""Usage survives result publication, including a failed run that already spent tokens."""

import argparse
import json

import pytest
from pydantic_ai.usage import RunUsage

from rpent.cli.main import _publish_harness_result
from rpent.harness import load_harness
from rpent.planner.api_loop import _build_stats
from rpent.result import EpisodeOutcome


@pytest.mark.parametrize("error", [None, "Provider became unavailable after earlier completed requests"])
def test_sdk_cached_usage_survives_final_result(tmp_path, error):
    package = tmp_path / "package"
    package.mkdir()
    (package / "system.md").write_text("System")
    (package / "user.md").write_text("Task")
    (package / "HARNESS.json").write_text(json.dumps({
        "schema_version": 2, "id": "usage", "native_tools": ["finish"],
        "planner_tools": [], "skills": [],
        "prompts": {"system": "system.md", "user": "user.md"},
    }))
    stats = _build_stats(RunUsage(input_tokens=1000, cache_read_tokens=800,
                                output_tokens=50, requests=3), 3, 2)
    path = _publish_harness_result(
        tmp_path / "episode", args=argparse.Namespace(
            robot_name="libero", harness_dir=str(package), planner="api", model="openai-chat:Qwen3.8-Max",
            max_turns=100, max_tokens=8192, temperature=0, reasoning_effort="none", no_images=False,
        ), task_desc={"suite": "libero_spatial_task", "task": 0, "seed": 1000},
        package=load_harness(package), outcome=EpisodeOutcome(
            validity="valid", official_success=False, terminated=False, truncated=False,
            env_steps=5, terminal_reason="agent_stopped",
        ), agent_error=error, finish_result=None, messages=[], stats=stats, elapsed=1,
        capability_calls=[], state=None,
    )
    result = json.loads(path.read_text())
    assert result["metrics"]["input_tokens"] == 1000
    assert result["metrics"]["cache_read_tokens"] == 800
    assert result["metrics"]["cache_write_tokens"] == 0
    assert result["metrics"]["output_tokens"] == 50
    assert result["metrics"]["model_requests"] == 3
    assert result["validity"] == ("invalid" if error else "valid")
