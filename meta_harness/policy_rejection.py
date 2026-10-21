"""Reject a proposal that demonstrably violates the existing policy contract.

This does not score an invalid trial, rewrite evidence, or cap infrastructure
retries. It makes candidate ineligibility terminal for the active proposal.
"""
import hashlib
from pathlib import Path

ERROR = 'HarnessError: Policy may only select a subset of the bound tools without changing their schemas'
REASONS = {'candidate_policy_contract_violation', 'superseded_by_rejected_candidate'}


def read_optional(store, path):
    try:return store.read(path)
    except FileNotFoundError:return None


def rejection(store, evaluation_id):
    existing=read_optional(store,f'evaluations/{evaluation_id}/cancel.json')
    if existing:return existing if existing.get('reason') in REASONS else None
    control=read_optional(store,'policy-contract-rejection-control.json')
    if not control or control.get('enabled') is not True:return None
    request=store.read(f'evaluations/{evaluation_id}/request.json')
    if request['scope_id'] not in control.get('allowed_scopes',[]):return None
    state=store.read('state.json');scope=state['scopes'].get(request['scope_id'],{})
    if scope.get('step') in ('complete','incomplete')or not scope:return None
    key=scope['scope_id'].replace('/','-')
    number=scope['rounds_completed']+1
    formal=f'{key}-r{number:02d}'
    development=f'dev-{key}-development-r{number:02d}-'
    if evaluation_id!=formal and not evaluation_id.startswith(development):return None
    folder=store.path(f'evaluations/{evaluation_id}/attempts')
    for cell in sorted(folder.glob('*')):
        for allocation in sorted(cell.glob('a*.json')):
            item=store.read(allocation.relative_to(store.root).as_posix())
            episode=item['episode_id']
            raw_path=store.path(f'episodes/{episode}/rollout/result.json')
            normalized=read_optional(store,f'episodes/{episode}/result.json')
            raw=read_optional(store,f'episodes/{episode}/rollout/result.json')
            if not raw or not normalized:continue
            if (raw.get('validity')!='invalid' or raw.get('terminal_reason')!='runner_error'
                    or raw.get('trace_complete') is not True or not raw.get('config_ref')
                    or raw.get('harness_commit')!=request['harness_commit']
                    or raw.get('controller')!=request['controller']
                    or raw.get('agent',{}).get('error')!=ERROR):continue
            from meta_harness.evaluation import collect_evaluation
            observed=collect_evaluation(store,evaluation_id)
            proof={'schema_version':1,'reason':'candidate_policy_contract_violation',
                'evaluation_id':evaluation_id,'scope_id':request['scope_id'],
                'harness_commit':request['harness_commit'],'episode_id':episode,'cell_id':cell.name,
                'raw_result_ref':f'episodes/{episode}/rollout/result.json',
                'raw_result_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                'error':ERROR,'policy_revision':control['revision'],
                'observed_candidate_successes':observed['successes'],
                'observed_candidate_valid_cells':observed['valid'],
                'invalid_trial_reclassified_as_valid':False,'candidate_score':None,
                'existing_completed_workers_may_finish':True}
            try:store.write_once(f'evaluations/{evaluation_id}/cancel.json',proof)
            except Exception:
                # Another host/MCP reader may have published equivalent proof
                # at the same boundary. Read the winning immutable decision.
                winner=read_optional(store,f'evaluations/{evaluation_id}/cancel.json')
                if not winner or winner.get('reason')!='candidate_policy_contract_violation':raise
                return winner
            return proof
    return None


def retire_development(store,scope,formal_rejection):
    key=scope['scope_id'].replace('/','-')
    prefix=f'dev-{key}-development-r{scope["rounds_completed"]+1:02d}-'
    for path in sorted(store.path('evaluations').glob(prefix+'*/request.json')):
        eid=path.parent.name
        if store.path(f'evaluations/{eid}/complete.json').exists():continue
        if read_optional(store,f'evaluations/{eid}/cancel.json'):continue
        store.write_once(f'evaluations/{eid}/cancel.json',{
            'schema_version':1,'reason':'superseded_by_rejected_candidate',
            'evaluation_id':eid,'scope_id':scope['scope_id'],
            'formal_evaluation_id':formal_rejection['evaluation_id'],
            'existing_completed_workers_may_finish':True,'candidate_score':None})


def reject_candidate(loop,scope,evaluation_id,summary,proof):
    if proof['reason']!='candidate_policy_contract_violation':
        raise ValueError('A formal proposal requires a proven policy-contract violation')
    if proof['harness_commit']!=scope['candidate_commit']:
        raise ValueError('Rejection evidence names a different candidate')
    source=loop.store.path(proof['raw_result_ref'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=proof['raw_result_sha256']:
        raise ValueError('Rejection evidence changed')
    incumbent=loop.pool.evaluation_status(scope['selected_evaluation'])
    if not incumbent['comparable']:raise ValueError('Incumbent matrix must remain complete')
    number=scope['rounds_completed']+1
    record={'round':number,'confirmation':False,'perfect_streak':0,'converged':False,
        'incumbent_commit':scope['selected_commit'],'incumbent_evaluation':scope['selected_evaluation'],
        'candidate_commit':scope['candidate_commit'],'candidate_evaluation':evaluation_id,
        'accepted':False,'selected_commit':scope['selected_commit'],'selected_evaluation':scope['selected_evaluation'],
        'incumbent_successes':incumbent['successes'],'candidate_successes':None,
        'denominator':incumbent['denominator'],'candidate_eligible':False,
        'observed_candidate_successes':proof['observed_candidate_successes'],
        'observed_candidate_valid_cells':proof['observed_candidate_valid_cells'],
        'comparison':{'accepted':False,'reason':'candidate_policy_contract_violation','comparable':False},
        'candidate_rejection':proof}
    loop.store.write_once(f'selections/{scope["scope_id"]}/r{number:02d}.json',record)
    loop.harnesses.retain(scope['candidate_commit'],f'candidates/{scope["candidate_commit"]}')
    retire_development(loop.store,scope,proof)
    scope.update(rounds_completed=number,recoveries=0,perfect_streak=0,convergence_reason=None,step='develop')
    budget=loop._development_budget(scope)
    exhausted=(budget[1]>=budget[0])if budget else number>=loop.config.phase(scope['phase']).rounds
    if budget:scope['development_budget_revision']=budget[2]
    if exhausted:scope.update(step='complete',convergence_reason='development_budget_exhausted')
