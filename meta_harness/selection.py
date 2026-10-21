"""Success-first selection with paired deployment-time improvement above threshold."""
from __future__ import annotations

from math import ceil, isclose
from rpent.deployment_time import PROTOCOL, validate


def objective(config, selected):
    settings = config.time_optimization
    threshold = ceil(settings['success_rate_threshold'] * selected['denominator'] - 1e-12) if settings else None
    speed = bool(settings and selected.get('comparable') and selected['denominator'] > 0 and selected['successes'] >= threshold)
    return {'mode': 'reduce_deployment_time' if speed else 'improve_success',
            'settings': settings, 'required_successes': threshold,
            'selected_successes': selected.get('successes'), 'denominator': selected.get('denominator'),
            'time_definition': 'env_steps / control_frequency_hz + Planner + VLA + SAM round-trip times; serial; no other overhead',
            'success_count_regression_allowed': False,
            'paired_success_regression_allowed_for_time_promotion': False,
            'instructions': ('For a time-only promotion preserve every previously successful paired cell; with equal success count, '
                             'reduce the mean deployment time by the configured minimum. More successes always take priority.'
                             if speed else 'Improve autonomous task success. Time alone cannot promote a below-threshold tie.')}


def compare(config, incumbent, candidate):
    if not incumbent.get('comparable') or not candidate.get('comparable') or incumbent['denominator'] != candidate['denominator']:
        raise ValueError('selection requires complete comparable matrices')
    decision = {'accepted': False, 'reason': 'success_tie', 'objective': objective(config, incumbent)}
    if candidate['successes'] != incumbent['successes']:
        better = candidate['successes'] > incumbent['successes']
        return {**decision, 'accepted': better, 'reason': 'success_improvement' if better else 'success_regression'}
    settings = config.time_optimization
    if not settings:
        return decision
    if decision['objective']['mode'] != 'reduce_deployment_time':
        return {**decision, 'reason': 'success_threshold_not_reached'}

    def cells(summary):
        rows = list(summary.get('cells', {}).values())
        keys = [(c['task_id'], c['seed'], c['repeat']) for c in rows]
        if len(rows) != summary['denominator'] or len(set(keys)) != len(keys):
            raise ValueError('missing or duplicate paired cells')
        if any(type(c.get('official_success')) is not bool for c in rows):
            raise ValueError('paired official outcomes unavailable')
        return dict(zip(keys, rows))

    try:
        left, right = cells(incumbent), cells(candidate)
        if left.keys() != right.keys():
            raise ValueError('paired scene identities differ')
        regressions = [k for k, c in left.items() if c['official_success'] and not right[k]['official_success']]
        if regressions:
            return {**decision, 'reason': 'paired_success_regression', 'regressed_cells': regressions}
        pairs = []
        for key, cell in left.items():
            if not cell['official_success']:
                continue
            if cell.get('resource_profile_revision') != right[key].get('resource_profile_revision'):
                raise ValueError('resource profiles differ; preserve success comparison but do not claim time improvement')
            values = [cell.get('deployment_time'), right[key].get('deployment_time')]
            for value in values:
                if not value or not value.get('available') or value.get('protocol') != PROTOCOL:
                    raise ValueError('paired deployment timing unavailable')
                validate(value, value.get('env_steps'))
                if not isclose(value['control_frequency_hz'], settings['control_frequency_hz'], abs_tol=1e-9):
                    raise ValueError('control frequency differs from the fixed selection protocol')
            pairs.append({'task_id': key[0], 'seed': key[1], 'repeat': key[2],
                          'incumbent_seconds': values[0]['total_seconds'], 'candidate_seconds': values[1]['total_seconds']})
    except (ValueError, KeyError, TypeError) as exc:
        return {**decision, 'reason': 'time_comparison_unavailable', 'detail': str(exc)}
    if not pairs:
        return {**decision, 'reason': 'no_paired_successes'}
    before = sum(p['incumbent_seconds'] for p in pairs) / len(pairs)
    after = sum(p['candidate_seconds'] for p in pairs) / len(pairs)
    improvement = (before-after)/before if before > 0 else 0
    accepted = before > 0 and improvement + 1e-12 >= settings['minimum_relative_improvement']
    return {**decision, 'accepted': accepted,
            'reason': 'deployment_time_improvement' if accepted else 'insufficient_time_improvement',
            'paired_successes': len(pairs), 'incumbent_mean_seconds': before, 'candidate_mean_seconds': after,
            'relative_improvement': improvement, 'pairs': pairs}
