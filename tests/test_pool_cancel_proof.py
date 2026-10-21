from meta_harness.pool import ExecutionPool
from meta_harness.storage import RunStore


def test_scope_cleanup_preserves_rejection_proof_and_is_replayable(tmp_path):
    store = RunStore(tmp_path)
    pool = ExecutionPool.__new__(ExecutionPool)
    pool.store = store
    pool.active = lambda: {}
    proof = {"reason": "candidate_policy_contract_violation", "raw_result_sha256": "evidence"}
    for eid, scope, purpose in [("rejected", "task", "development"),
                                ("pending", "task", "development"),
                                ("formal", "task", "formal"),
                                ("other", "other-task", "development")]:
        store.write_once(f"evaluations/{eid}/request.json", {
            "evaluation_id": eid, "scope_id": scope, "purpose": purpose})
    store.write_once("evaluations/rejected/cancel.json", proof)
    original = store.path("evaluations/rejected/cancel.json").read_bytes()
    pool.cancel_scope("task")
    pool.cancel_scope("task")
    assert store.path("evaluations/rejected/cancel.json").read_bytes() == original
    assert store.read("evaluations/pending/cancel.json") == {"reason": "development_finished"}
    assert not store.path("evaluations/formal/cancel.json").exists()
    assert not store.path("evaluations/other/cancel.json").exists()
