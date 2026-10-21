# Copyright 2026 The RPent Authors.
"""Run the existing Codex provider under one host-owned Meta deadline."""

import json
import os
import re
import signal
import sys
import threading
import time

from meta_harness import token_gate
from meta_harness.environment import meta_environment, prepare_codex_auth
from meta_harness.meta import MetaAttempt
from meta_harness.pool import replace_record
from meta_harness.provider import CodexSession
from meta_harness.token_gate import TokenGate
from rpent.utils.execution import process_identity


RESUME_PROMPT = (
    "The provider was interrupted and has now resumed. Continue the same development "
    "round in the same workspace and conversation from where you left off."
)


def is_payment_required(exc):
    return bool(re.search(r"\b402\b.{0,40}Payment Required", str(exc), re.IGNORECASE))


def provider_outage(exc):
    """Classify model availability failures without treating policy errors as transient."""
    message = str(exc).lower()
    if ("insufficient_user_quota" in message or "额度不足" in message
            or "余额不足" in message or "预扣费额度失败" in message
            or is_payment_required(exc)):
        return "quota"
    if re.search(r"\b(?:429|500|502|503|504|520|522|524)\b", message):
        return "provider_unavailable"
    if any(word in message for word in ("timed out", "timeout", "connection reset", "connection refused", "temporarily unavailable",
                                       "stream disconnected", "transport error", "network error",
                                       "error decoding response body", "connection failed", "error sending request")):
        return "transport"
    return None


def cancel_from_signal(session, stop, reason):
    """Defer Popen waits: a signal can interrupt its non-reentrant wait lock."""
    reason[0] = "interrupted"
    stop.set()
    threading.Thread(target=session.cancel, daemon=True,
                     name="meta-signal-cancel").start()


def run(root, attempt_id, fd):
    fd = int(fd)
    os.fstat(fd)
    attempt = MetaAttempt(root, attempt_id)
    store, prefix = attempt.store, attempt.prefix
    replace_record(store, f"{prefix}/owner.json", process_identity())
    # An earlier worker may have persisted submit() and then crashed or lost
    # its provider.  Finish directly, before rebuilding its Codex workspace.
    if attempt.budget()["submission"]:
        attempt.finish(reason="submitted")
        os.close(fd)
        return
    scope = store.path(f"workspaces/{attempt.context['scope_id']}")
    project, runtime = scope / "project", scope / "runtime"
    project.mkdir(parents=True, exist_ok=True)
    prepare_codex_auth(runtime)
    (project / "runtime.md").write_text(attempt.context["runtime"])
    env = meta_environment(runtime)
    opening_prompt = attempt.opening()
    opening_images = attempt.opening_images()
    overrides_path = store.path("meta-overrides.json")
    override = (json.loads(overrides_path.read_text()).get(attempt.context["scope_id"], {})
                if overrides_path.exists() else {})
    from meta_harness.campaign_runtime import control
    campaign = control(store)
    runtime_meta = campaign.get("meta", {})
    selected_provider=runtime_meta.get("provider",override.get("provider",attempt.config.meta.provider))
    from meta_harness.round_cap import RoundProxy, configured_limits, finalize, CLOSING_PROMPT
    limits=configured_limits(attempt)
    round_proxy=None
    if limits and attempt.context["kind"]=="development":
        round_proxy=RoundProxy(attempt,selected_provider,limits)
        if round_proxy.ledger.phase()=="exhausted":
            finalize(attempt);attempt.finish(reason="submitted")
            round_proxy.close();os.close(fd);return
        selected_provider=round_proxy.provider_config()
        env[selected_provider["env_key"]]=round_proxy.client_key
        env["OPENAI_API_KEY"]=round_proxy.client_key
        env["TOKKEN_API_KEY"]=round_proxy.client_key
        env["OPENAI_BASE_URL"]=selected_provider["base_url"]
        env["TOKKEN_BASE_URL"]=selected_provider["base_url"]
        from meta_harness._files import atomic_json
        atomic_json(runtime/"auth.json",{"OPENAI_API_KEY":round_proxy.client_key})
    session = CodexSession(
        project,
        runtime,
        store.path(f"{prefix}/session.json"),
        store.path(f"{prefix}/provider.jsonl"),
        model=attempt.config.meta.model,
        timeout_s=attempt.budget()["development_seconds_left"],
        environment=env,
        provider=selected_provider,
        reasoning_effort=runtime_meta.get("reasoning_effort", override.get("reasoning_effort", attempt.config.meta.reasoning_effort)),
        allow_session_migration=bool(campaign.get("allow_session_migration", False)),
        inherited_fds=(fd,),
        process_record=store.path(f"{prefix}/provider-owner.json"),
        initial_images=opening_images,
        mcp_servers={
            "rsi": {
                "command": sys.executable,
                "args": ["-m", "meta_harness.meta_mcp", str(store.root), attempt_id],
                "env_vars": ["PYTHONPATH", "RPENT_REPO_ROOT", "RPENT_PYTHON"],
                "tool_timeout_sec": 120,
            }
        },
    )
    # CodexSession loaded the durable thread ID from this exact attempt. Keep
    # it on worker recovery: the workspace and provider conversation both
    # belong to this attempt, so a fresh opening would waste its prior turns.
    resume_path = store.path(f"{prefix}/resume-thread.json")
    resume = json.loads(resume_path.read_text()) if resume_path.exists() else None
    if resume:
        session.thread_id = resume["thread_id"]
        session.task_id = attempt_id
    # Replacement workers resume the durable attempt; clear the old parked marker.
    wait_path = store.path(f"{prefix}/provider-wait.json")
    prior_wait = json.loads(wait_path.read_text()) if wait_path.exists() else {}
    if prior_wait.get("state") == "needs_operator":
        replace_record(store, f"{prefix}/provider-wait.json",
                       {"state": "resuming", "reason": "worker_recovery",
                        "thread_id": session.thread_id, "attempt_id": attempt_id,
                        "previous_wait": prior_wait, "at": time.time()})
    stop = threading.Event()
    reason = ["provider_exit"]

    def cancel(sig=None, frame=None):
        cancel_from_signal(session, stop, reason)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    gate = TokenGate(attempt)
    if round_proxy:gate.target=None  # A maximum must never force legacy minimum spending.

    def monitor():
        checked = 0.0
        while not stop.wait(0.1):
            budget = attempt.budget()
            if budget["submission"] or (budget["development_seconds_left"] is not None and budget["development_seconds_left"] <= 0):
                reason[0] = "submitted" if budget["submission"] else "deadline"
                session.cancel()
                return
            if gate.enabled and time.monotonic() - checked > 15:
                checked = time.monotonic()
                used = gate.used()
                gate.record(used)
                if used >= gate.upper:
                    reason[0] = "token_budget"
                    stop.set()
                    session.cancel()
                    return

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        prompt = (RESUME_PROMPT + " " + str(attempt.budget())) if resume or session.thread_id else (
            opening_prompt + f"\nAttached historical RGB identities: {store.path(prefix + '/opening-images.json')}")
        failures = 0
        no_progress = 0
        def progress():
            # End-of-turn progress only; long reasoning within a turn is unrestricted.
            import hashlib
            digest = hashlib.sha256()
            for path in sorted(attempt.workspace.rglob("*")):
                if path.is_file() and ".git" not in path.parts:
                    digest.update(str(path).encode())
                    digest.update(path.read_bytes())
            for path in sorted(store.path(prefix + "/evaluations").glob("*/request.json")):
                digest.update(path.read_bytes())
                eid = f"dev-{attempt.attempt_id}-{path.parent.name}"
                summary = attempt.pool.evaluation_status(eid)
                digest.update(json.dumps({key: summary.get(key) for key in
                    ("state", "valid", "invalid", "successes", "comparable", "exhausted")}, sort_keys=True).encode())
            for path in sorted(store.path("pool/live").glob("*/request.json")):
                request = json.loads(path.read_text())
                if request["scope_id"] == attempt.context["scope_id"]:
                    for response in sorted(path.parent.glob("responses/*.json")):
                        # Fresh physical evidence counts; repeated queue polling does not.
                        value = json.loads(response.read_text())
                        if value.get("result", {}).get("state") != "waiting_for_boundary":
                            digest.update(response.read_bytes())
            return digest.digest()
        previous_progress = progress()
        reported_outage = None
        last_outage_report = 0.0
        while not stop.is_set():
            if round_proxy and round_proxy.ledger.phase()=="exhausted":
                finalize(attempt);reason[0]="submitted";break
            if round_proxy and round_proxy.ledger.phase()=="finalization":
                round_proxy.ledger.acknowledge()
                prompt=CLOSING_PROMPT
            budget = attempt.budget()
            if budget["submission"] or (budget["development_seconds_left"] is not None and budget["development_seconds_left"] <= 0):
                reason[0] = "submitted" if budget["submission"] else "deadline"
                break
            try:
                session.timeout_s = budget["development_seconds_left"]
                session.turn(prompt)
                failures = 0
                if reported_outage is not None:
                    replace_record(store, f"{prefix}/provider-wait.json",
                                   {"state": "recovered", "reason": reported_outage,
                                    "thread_id": session.thread_id, "at": time.time()})
                    reported_outage = None
                current_progress = progress()
                pending = any(not (summary.get("comparable") or summary.get("exhausted"))
                              for summary in (attempt.pool.evaluation_status(
                                  f"dev-{attempt.attempt_id}-{path.parent.name}")
                                  for path in store.path(prefix + "/evaluations").glob("*/request.json")))
                no_progress = no_progress + 1 if current_progress == previous_progress and not pending else 0
                previous_progress = current_progress
                used = gate.used() if gate.enabled else None
                if gate.enabled:
                    gate.record(used)
                if gate.enabled and used < gate.target:
                    no_progress = 0
                    prompt = token_gate.CONTINUE.format(used=used, target=gate.target) + " " + str(attempt.budget())
                    continue
                if no_progress >= attempt.config.development.max_no_progress_turns:
                    reason[0] = "token_budget" if gate.enabled else "no_progress"
                    break
                prompt = (
                    token_gate.REACHED.format(used=used, target=gate.target) + " " + str(attempt.budget())
                    if gate.enabled else
                    "The provider turn ended without a submission. Continue in the same "
                    "workspace under the existing budget, or use submit now. "
                    + str(attempt.budget())
                )
            except Exception as exc:
                # A provider can fail just after submit() persisted the frozen
                # candidate.  Once that record exists, finishing the attempt
                # is independent of the provider and must not park this slot.
                if attempt.budget()["submission"]:
                    reason[0] = "submitted"
                    break
                if round_proxy and round_proxy.ledger.phase()!="development":
                    session.begin_task(attempt_id)
                    if round_proxy.ledger.phase()=="exhausted":
                        finalize(attempt);reason[0]="submitted";break
                    round_proxy.ledger.acknowledge();prompt=CLOSING_PROMPT
                    continue
                outage = provider_outage(exc)
                failures += 1
                replace_record(
                    store,
                    f"{prefix}/provider-error.json",
                    {"type": type(exc).__name__, "message": str(exc)},
                )
                if outage:
                    now = time.time()
                    if outage != reported_outage or now - last_outage_report >= 300:
                        replace_record(store, f"{prefix}/provider-wait.json",
                                       {"state": "waiting", "reason": outage,
                                        "thread_id": session.thread_id,
                                        "attempt_id": attempt_id, "at": now})
                        reported_outage, last_outage_report = outage, now
                    session.begin_task(attempt_id)
                    prompt = RESUME_PROMPT + " " + str(attempt.budget()) if session.thread_id else opening_prompt
                    # The interval is bounded, the wait and retry count are not.
                    stop.wait(60 if outage == "quota" else min(2 ** min(failures, 6), 60))
                    continue
                replace_record(store, f"{prefix}/provider-wait.json",
                               {"state": "needs_operator", "reason": "unclassified_provider_error",
                                "thread_id": session.thread_id,
                                "attempt_id": attempt_id, "at": time.time()})
                # A repair can stop this worker. The ordinary scheduler then
                # reopens this exact attempt and persisted provider thread.
                while not stop.wait(60):
                    if attempt.budget()["submission"]:
                        reason[0] = "submitted"
                        break
                break
    finally:
        stop.set()
        thread.join(timeout=3)
        session.cancel()
        if round_proxy:round_proxy.close()
    if reason[0] != "interrupted":
        attempt.finish(reason=reason[0])
    os.close(fd)


if __name__ == "__main__":
    try:
        run(*sys.argv[1:])
    except Exception as exc:
        # Setup/inspection errors must consume a recovery attempt too. Without
        # a wall-clock deadline, silently reopening the same crashed worker
        # would otherwise retry forever without ever producing finished.json.
        failed = MetaAttempt(sys.argv[1], sys.argv[2])
        replace_record(failed.store, failed.prefix + "/worker-error.json",
                       {"type": type(exc).__name__, "message": str(exc)})
        failed.finish(reason="worker_error")
        raise
