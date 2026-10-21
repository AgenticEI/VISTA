---
name: evolve-harness
description: Develop an evolvable Harness package with the current RSI Meta tools.
---

Planner uses the original RPent PydanticAI Agent.iter loop:
package prompts/memory/tools → Policy.prepare_request → actual SDK/provider request
→ Planner tool call → schema validation → bound handler/capability → raw result
→ Policy.process_tool_result → next model request. Policy.decide may provide a
harness_program tool decision through that same path. It does not replace the host loop.
Package Python executes; the model only reads the prompts, tool descriptions,
messages and explicit file/tool returns it receives. Package memory is read-only;
episode notes and Policy state are private to each episode. Skills retain
skills/<name>/SKILL.md and tool.py:build(runtime), with explicit requires.
The capability catalog is the backend menu; only the bound tools are visible to
Planner. The platform description specifies the real observation, control, VLA
instruction and session semantics. Each new environment has a private session.
Inspect actual requests and images before deciding information reached Planner.

History navigation uses the run's existing records, without copying parent runs:
history_summary → history_episodes (task_id / seed / repeat / validity filters)
→ history_episode → history_decisions → history_decision. A decision contains
separate SDK request, actual HTTP attempts, model response, raw tool result,
Policy-transformed feedback, and delivered tool feedback. Inspect the relevant
stage when testing whether a package change reached Planner. Rejected calls,
provider errors and multiple send_attempt values remain separate evidence.
history_decisions gives inclusive executed control-step ranges and the preceding
before_env_step. Use history_timeline(camera="all") for an overview, narrow its
start_env_step/end_env_step for a dense interval, and history_frame for exact
images, action and robot state. A frame links back through history_decision;
env_step is a physical control-step index, not a Planner turn or a tool-call index.
Episode identity includes task, scene seed, repeat, policy seed, execution attempt
and frozen Harness commit. Infrastructure retries are separate attempts of the
same cell; invalid is not task failure. Official success is host evidence, not
Planner's finish claim. Keep manual/mixed diagnostics separate from autonomous
Planner outcomes. full_rgb contains both LIBERO camera images but no archived
depth; online RGB-D localization remains available. Never invent historical depth.

Research improves the inheritable package. Follow history → live → evaluate → submit:

A. Observe. Align the failing decision's actual request, selected tool/arguments,
raw tool result and physical before/after frames. Use a live decision boundary
when history cannot distinguish the mechanisms; observe or take over as needed.
B. Diagnose. State a falsifiable operational hypothesis tied to a specific
choice, contact, grasp, geometry or feedback mechanism. Separate task failure
from OOM, invalid trajectories, provider failures and other infrastructure errors.
C. Intervene. Implement the smallest surface that tests the hypothesis: a tool,
Policy or memory. Keep prompts thin. A useful live observation or manual solution
must become an explicit change in the current package, not just conversation
state, project notes, hidden operator knowledge or a transient live action.
D. Validate internally. Freeze the updated workspace with evaluate, poll the same
evaluation ID until complete and comparable, and use official autonomous success
to accept or reject the hypothesis. Invalid is inconclusive. If falsified, change
the mechanism or hypothesis instead of repeating the same experiment. Evaluate
again after subsequent package changes; a prior commit's evidence is not evidence
for the new draft. Test quota exhaustion does not prohibit an honest submission;
state clearly if the final package has not completed internal validation.
Start each hypothesis with 2–4 representative seeds and one repeat per seed,
including a known failure and a previously successful case when available. For
joint development, target the affected tasks first. Do not default internal tests
to the host's full formal matrix. Expand only when the small test supports the
hypothesis and a specific unresolved regression or variability question requires
more evidence; state that question before expanding. A small test supports a
mechanism check, not a formal success-rate or promotion claim. The host runs the
full formal matrix after submission. Track outstanding evaluation IDs and reuse
their results; do not launch a replacement merely because a test is pending or
infrastructure-invalid. Concurrent tests must answer distinct stated questions.
E. Execute reliably. Scores must reflect the frozen package under the ordinary
Planner loop. A live reply replaces one real decision using its exact menu/schema.
Extra Meta observations are diagnostics and are not silently inserted into the
Planner's request. Mixed/manual success is never autonomous success or promotion.
F. Select formally. The host evaluates a submitted package in a separate complete
matrix. For strict_improvement, only more official successes promotes a package;
ties retain the incumbent. The host may stop iteration after the same selected
package achieves two consecutive complete perfect matrices. Live and internal
evaluate results never count toward that stopping rule. Legacy success_then_time
runs use the explicit objective and paired timing constraints in the opening.

Live protocol: the opening declares live_trials, a prompt-level ablation setting
(default 2). Complete that many distinct live scenes per development round;
repeated polling and observe calls are not new trials. With zero, use history and
autonomous evaluate without live. Choose observation versus intervention from
the evidence and package strength. Persist each trial's hypothesis, session ID,
observation, package change and evaluation outcome in package memory. Report and
replace infrastructure-invalid trials separately. Close a live scene after useful
evidence is collected; do not hold a slot while editing or thinking elsewhere.
There is no wall-clock target in unlimited mode. Thinking longer is allowed.
Avoid spending repeated attempts on an unchanged hypothesis without new evidence.

History remains full_rgb: both RGB cameras, no archived depth. During live,
Planner observe and Meta live_observe include pseudocolor depth PNGs and one-line
min/median/max in metres. Use geometry.libero.back_project@1 for metric positions
at the matching camera, step and resolution. Never put dense depth/world/RGB arrays
in JSON or MCP; do not infer historical depth from live captures. Open the actual
RGB and depth images, not only their paths. Preview colors are relative to the
reported range; metric geometry comes from back_project.

For live calls, pending means queued: poll the SAME request_id for its response.
After waiting_for_boundary, use a NEW request_id to inspect the next boundary.
Edit the candidate worktree with ordinary file tools and use Git for history.
live_start, evaluate and submit each freeze the workspace at the time of that
call, including current edits. Edit first, then freeze; a live session retains its
original frozen package, so restart a scene to test later code changes. Reusing an
evaluate request_id retrieves its original frozen test; new ideas need new IDs.

Packages from different tasks may later be merged. Keep task-specific knowledge
explicitly task-scoped, with triggering conditions, evidence and known limits.
Do not disguise a single-task recipe or fixed pose as a general rule. Reusable
mechanisms need evidence beyond the originating scene; keep specialized recipes
separate from shared tools/Policy where practical. Minimize unrelated edits and
explain any shared API change. During merge repair inspect all selected donors,
resolve both Git and semantic conflicts, preserve scoped dispatch and verify the
integrated package with autonomous evaluation. A clean Git merge does not prove
behavioral compatibility.

Submit when ready. Notes must identify change, intended physical effect, live
experience incorporated into the package, exact evaluated commit/evaluation IDs,
official outcomes, task scope and unresolved issues. With a configured legacy
deadline, the host may snapshot a loadable unfinished draft; this is not a claim
of Meta verification. In unlimited mode there is no deadline snapshot.
