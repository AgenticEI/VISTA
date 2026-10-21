from decimal import Decimal

import pytest

from rpent.planner.recording import PlannerRecorder


def test_model_usage_cost_preserves_decimal_precision():
    recorder = PlannerRecorder.__new__(PlannerRecorder)
    recorded = recorder.serializable({"request": {"messages": [{"usage": {"cost": Decimal("0.0001234567890123456789")}}]}})
    assert recorded["request"]["messages"][0]["usage"]["cost"] == "0.0001234567890123456789"


def test_nonfinite_cost_is_rejected():
    recorder = PlannerRecorder.__new__(PlannerRecorder)
    with pytest.raises(ValueError, match="nonfinite Decimal"):
        recorder.serializable(Decimal("NaN"))
