"""Explicit, versioned resource-only changes before the first baseline completes.

The fixed experiment config and old episode records remain untouched. Resource
changes do not invalidate first-valid success evidence. Timing comparability is
checked separately by the selection rule.
"""
from dataclasses import replace

from meta_harness.config import PoolConfig
from meta_harness.evaluation import _record_id
from meta_harness.storage import RunStore, _publish, _json_bytes


def current_revision(store):
    path = store.path('pool/resource-profile-current.json')
    if not path.exists():
        return None
    value = store.read('pool/resource-profile-current.json')['revision']
    return _record_id(value, 'resource profile revision')


def effective_config(store, config, revision='current'):
    revision = current_revision(store) if revision == 'current' else revision
    if revision is None:
        return config, None
    _record_id(revision, 'resource profile revision')
    record = store.read(f'pool/resource-profiles/{revision}.json')
    pool = PoolConfig.parse(record['pool'])
    if pool.gpu_devices != config.pool.gpu_devices:
        raise ValueError('resource override must retain the declared GPU pool')
    return replace(config, pool=pool), revision


def activate(root, config, revision, pool, *, runtime_revision):
    store = RunStore(root)
    _record_id(revision, 'resource profile revision')
    _record_id(runtime_revision, 'runtime revision')
    parsed = PoolConfig.parse(pool)
    if parsed.gpu_devices != config.pool.gpu_devices:
        raise ValueError('GPU pool identity changed')
    with store.scheduler_lock():
        state = store.read('state.json')
        if (state.get('suite_index') != 0 or state.get('phase') != 'parallel'
                or any(s['step'] != 'baseline' for s in state['scopes'].values())):
            raise ValueError('resource changes are only supported before any initial baseline is selected')
        record = {'revision': revision, 'runtime_revision': runtime_revision, 'pool': pool,
                  'previous_revision': current_revision(store),
                  'timing_policy': 'retain first-valid success outcomes; reject unpaired cross-profile timing comparisons'}
        store.write_once(f'pool/resource-profiles/{revision}.json', record)
        _publish(store.path('pool/resource-profile-current.json'), _json_bytes({'revision': revision}), replace=True)
    return record
