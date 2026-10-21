from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from meta_harness.config import EvolutionConfig
from meta_harness.meta import MetaAttempt, runtime_description
from meta_harness.meta_mcp import definitions, invoke
from meta_harness.pool import ExecutionPool

ROOT=Path(__file__).resolve().parents[1]

def test_legacy_config_and_tool_menu_unchanged():
    cfg=EvolutionConfig.load(ROOT/'tests/fixtures/libero-pipeline.json')
    assert cfg.development.live_enabled
    assert 'live_enabled' not in cfg.public_record()['development']
    assert any(d['name']=='live_start' for d in definitions())

def test_disabled_run_blocks_live_at_all_entry_layers():
    cfg=EvolutionConfig.load(ROOT/'tests/fixtures/libero-pipeline.json')
    cfg=replace(cfg,development=replace(cfg.development,live_enabled=False,live_trials=0))
    assert EvolutionConfig.parse(cfg.public_record(),root=ROOT).development.live_enabled is False
    names={d['name'] for d in definitions(live_enabled=False)}
    assert {'evaluate','submit','history_frame','select_parent'} <= names
    assert not any(n.startswith('live_') for n in names)
    attempt=object.__new__(MetaAttempt);attempt.config=cfg
    for name in ('live_start','live_observe','live_reply','live_capability','live_decision','live_close'):
        with pytest.raises(PermissionError):invoke(attempt,name,{})
    with pytest.raises(PermissionError):attempt.live_start('a','spatial-t00',1000)
    pool=object.__new__(ExecutionPool);pool.config=cfg
    with pytest.raises(PermissionError):
        pool.live_start(session_id='a',scope_id='s',commit='a'*40,task_id='spatial-t00',seed=1000,deadline=None)
    prompt=runtime_description(cfg)
    assert 'interfaces are disabled' in prompt
    assert 'perform 2 distinct live trials' not in prompt

def test_contradictory_live_config_rejected():
    d=EvolutionConfig.load(ROOT/'tests/fixtures/libero-pipeline.json').public_record()
    d['development']['live_enabled']=False
    with pytest.raises(ValueError,match='live_trials=0'):EvolutionConfig.parse(d,root=ROOT)
