from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from meta_harness.loop import EvolutionLoop
from meta_harness.storage import RunStore


def setup_loop(tmp_path, count=0, confirmations=()):
    loop=EvolutionLoop.__new__(EvolutionLoop);loop.store=RunStore(tmp_path)
    loop.store.write_once('development-budget-current.json',{'revision':'five'})
    loop.store.write_once('development-budgets/five.json',{'max_candidate_rounds':5})
    scope={'scope_id':'suite/task','phase':'parallel','rounds_completed':count,'recoveries':0,'step':'develop','selected_commit':'old','selected_evaluation':'old-eval','candidate_commit':'new','perfect_streak':0}
    for n in range(1,count+1):loop.store.write_once(f'selections/suite/task/r{n:02d}.json',{'confirmation':n in confirmations})
    return loop,scope


def test_confirmation_and_retries_do_not_consume_candidate_budget(tmp_path):
    loop,scope=setup_loop(tmp_path,5,(3,));scope['recoveries']=99
    assert loop._development_budget(scope)==(5,4,'five')
    assert not loop._stop_at_development_budget(scope)


def test_five_completed_stops_without_new_development(tmp_path):
    loop,scope=setup_loop(tmp_path,5)
    loop._develop=Mock(side_effect=AssertionError('must not develop'))
    loop._advance_scope(scope,None)
    assert scope['step']=='complete'
    assert scope['convergence_reason']=='development_budget_exhausted'
    assert scope['selected_commit']=='old'


def test_inflight_overbudget_drains_without_formal_evaluation(tmp_path,monkeypatch):
    loop,scope=setup_loop(tmp_path,5)
    monkeypatch.setattr('meta_harness.loop.is_locked',lambda p:True)
    assert loop._stop_at_development_budget(scope)
    assert scope['step']=='develop' and scope['budget_stop_pending']
    monkeypatch.setattr('meta_harness.loop.is_locked',lambda p:False)
    loop._advance_scope(scope,None)
    assert scope['step']=='complete' and scope['selected_commit']=='old'


@pytest.mark.parametrize('perfect',[False,True])
def test_fifth_candidate_and_confirmation(tmp_path,monkeypatch,perfect):
    loop,scope=setup_loop(tmp_path,4);scope['step']='evaluate'
    loop.config=SimpleNamespace(selection=SimpleNamespace(perfect_rounds=2),phase=lambda p:SimpleNamespace(rounds=10))
    candidate={'comparable':True,'denominator':8,'successes':8 if perfect else 4}
    loop._evaluation=lambda *a:('candidate-eval',candidate)
    loop.pool=SimpleNamespace(evaluation_status=lambda e:{'comparable':True,'denominator':8,'successes':2})
    loop.harnesses=SimpleNamespace(retain=lambda *a:None)
    monkeypatch.setattr('meta_harness.selection.compare',lambda *a:{'accepted':True})
    loop._advance_scope(scope,None)
    assert scope['rounds_completed']==5
    if not perfect:
        assert scope['step']=='complete' and scope['convergence_reason']=='development_budget_exhausted'
    else:
        assert scope['step']=='confirm'
        loop._advance_scope(scope,None)
        assert scope['rounds_completed']==6
        assert loop._development_budget(scope)[1]==5
        assert scope['step']=='complete' and scope['convergence_reason']=='two_perfect_evaluations'
