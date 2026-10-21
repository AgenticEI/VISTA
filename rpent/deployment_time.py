"""Serial deployment-time estimate: physical control time plus model RPC time.

Simulator stepping/rendering, recording, startup and local pool admission are
excluded. Model calls include their measured round-trip latency. This is a
specified timing model, not a claim of measured real-robot completion time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PROTOCOL = 'serial-physical-plus-models-v1'
STAGES = {'planner_request': 'planner_seconds', 'rpc.vla.predict': 'vla_seconds', 'rpc.segment': 'sam_seconds'}


def estimate(rollout, timing_dir, *, env_steps, expected_planner_calls=None):
    result = {'protocol': PROTOCOL, 'available': False, 'env_steps': env_steps,
              'overlap_policy': 'serial_no_overlap', 'excluded': ['simulator_runtime', 'rendering', 'trace_io', 'startup', 'pool_admission', 'other_overhead']}
    try:
        if type(env_steps) is not int or env_steps < 0:
            raise ValueError('invalid physical step count')
        clock = json.loads((Path(rollout) / 'public/deployment-clock.json').read_text())
        hz = clock.get('control_frequency_hz')
        start, end = clock.get('start_perf_s'), clock.get('end_perf_s')
        if type(hz) not in (int, float) or not math.isfinite(hz) or hz <= 0:
            raise ValueError('environment control frequency unavailable')
        if clock.get('env_steps') != env_steps:
            raise ValueError('clock and outcome step counts differ')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end)) or end < start:
            raise ValueError('environment task time boundaries unavailable')
        if not timing_dir or not Path(timing_dir).is_dir():
            raise ValueError('model timing directory unavailable')
        totals = {name: 0.0 for name in STAGES.values()}
        calls = {name: 0 for name in STAGES.values()}
        recorded_planner_calls = 0
        spans = []
        for path in Path(timing_dir).glob('*.jsonl'):
            for line in path.open():
                row = json.loads(line)
                key = STAGES.get(row.get('stage'))
                if key is None:
                    continue  # Never add nested rpc.server_execute/queue spans.
                if key == 'planner_seconds':
                    recorded_planner_calls += 1
                seconds = row.get('seconds')
                left, right = row.get('start_perf_s'), row.get('end_perf_s')
                if any(type(x) not in (int, float) or not math.isfinite(x) for x in (seconds, left, right)):
                    raise ValueError('model timing span unavailable or nonfinite')
                if seconds < 0 or right < left or not math.isclose(seconds, right-left, abs_tol=1e-5):
                    raise ValueError('inconsistent model timing span')
                if right <= start or left >= end:
                    continue
                left, right = max(left, start), min(right, end)
                totals[key] += right-left
                calls[key] += 1
                spans.append((left, right))
        last_end = -math.inf
        for left, right in sorted(spans):
            if left < last_end - 1e-5:
                raise ValueError('overlapping model calls require a different timing protocol')
            last_end = max(last_end, right)
        # Audit all recorded requests, but charge only their part inside the
        # physical task window. Post-success finish calls are excluded, not free
        # unrecorded inference. Missing logs must never mean zero inference time.
        if type(expected_planner_calls) is not int or expected_planner_calls != recorded_planner_calls:
            raise ValueError('Planner request count does not match recorded timing')
        physical = env_steps / hz
        result.update(available=True, control_frequency_hz=hz, physical_seconds=physical,
                      **totals, model_calls=calls, recorded_planner_calls=recorded_planner_calls,
                      total_seconds=physical + sum(totals.values()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['reason'] = str(exc)
    return result


def validate(value, env_steps):
    if not isinstance(value, dict) or value.get('protocol') != PROTOCOL or type(value.get('available')) is not bool:
        raise ValueError('invalid deployment time record')
    if not value['available']:
        return
    fields = ('control_frequency_hz', 'physical_seconds', 'planner_seconds', 'vla_seconds', 'sam_seconds', 'total_seconds')
    if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) or value[k] < 0 for k in fields):
        raise ValueError('deployment time components must be finite and nonnegative')
    if value['control_frequency_hz'] <= 0 or value.get('env_steps') != env_steps:
        raise ValueError('deployment timing identity mismatch')
    if not math.isclose(value['physical_seconds'], env_steps / value['control_frequency_hz'], abs_tol=1e-8):
        raise ValueError('deployment physical time mismatch')
    total = sum(value[k] for k in ('physical_seconds', 'planner_seconds', 'vla_seconds', 'sam_seconds'))
    if not math.isclose(value['total_seconds'], total, abs_tol=1e-8):
        raise ValueError('deployment time total mismatch')
