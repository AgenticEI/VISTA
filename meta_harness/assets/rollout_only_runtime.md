# Rollout-only harness development

Improve the inheritable harness through history inspection, code revision, and
fresh autonomous evaluations. Live scene creation, observation, intervention,
decision replacement, and manual robot action interfaces are disabled by the host
for this run. Do not attempt to bypass that boundary by constructing a simulator,
calling robot service endpoints, starting other runs, or importing live workers.
Do not read other experiments' evolved packages, traces, task cards, or notes.
Use only this run's initial package, public capability documentation, and history.

Planner uses the ordinary RPent loop. Package prompts/memory/tools and Policy
prepare_request/process_tool_result determine the model's actual input and feedback.
Skills are executable Python; the model reads prompts and tool returns, not Python
implicitly. Episode state is private; package memory is read-only during evaluation.
The initial public capabilities remain available for candidate skill development.
VLA, sensing, and native controllers may all be used inside autonomous candidates.

Inspect failing requests, actual tool arguments/results, and before/after RGB frames.
Use history_summary, history_episodes, history_decisions/history_decision,
history_timeline and history_frame. full_rgb archives both cameras, not depth.
Do not infer archived depth or treat an image path as visual inspection.
Distinguish missing/incorrect information, perception errors, physical failures,
and infrastructure failures. Propose falsifiable operational hypotheses and implement
the smallest useful revision in prompts, memory, Policy, or executable skills.

Call evaluate to freeze the candidate and run fresh autonomous episodes. Start
with 2–4 representative development seeds, one repeat each, including a failure
and a previously successful case when available. Expand only for a stated question.
Poll the same evaluation ID; don't duplicate queued evaluations. Invalid outcomes
are inconclusive and must not be silently treated as task failures or successes.
Use official host success, not Planner claims, to assess the package. If an idea
fails, revise the hypothesis or mechanism. Re-evaluate after code changes; an old
commit's result is not evidence for a new draft. Retain negative evidence in notes.

You may select earlier parents through select_parent; rejected candidates remain
useful history. Keep task-specific assumptions explicit. Don't disguise a fixed
pose or single-task recipe as a general rule. No hidden simulator state is allowed.
Write evidence, evaluation IDs, tested commits, limitations, and proposed physical
effects into package memory/notes. Respect the request quota and deadline; submit
an honest loadable candidate when ready. Lack of remaining small-test requests
does not prohibit submission; declare unverified final edits explicitly.

After submit, the host runs the complete formal matrix and applies the same
selection rule as the active condition. Ties retain the incumbent. Live/manual
success cannot count as autonomous success. The deadline may snapshot a loadable
unfinished draft; this does not claim Meta verification. Final held-out results
are not available for development or candidate selection.
