# Developing this package

You may edit HARNESS.json, prompts, memory, policy.py and skills. Start by reading
both existing tool.py files: they are executable examples, not pseudocode.
capabilities.json contains all ten capability IDs with full input/output schemas.
These capabilities are available for development even though Planner initially
sees only observe and vla_step plus infrastructure tools.

## Adding a tool
Create skills/<name>/SKILL.md and tool.py. Export build(runtime) returning
ToolBinding(name, description, input_schema, handler) from rpent.harness.
Append {"name": "<name>", "requires": {"alias": "cap:<exact capability ID>"}}
to HARNESS.json skills. Call runtime.call("alias", arguments) inside the handler,
never during build. Only declared aliases are accessible. A later skill may use
"tool:<earlier skill name>". Do not import robots.libero.tools or use its advanced
native tools as a substitute for package code. Compose behavior here.

Pass observation/image bytes through in results; printing a path is not visual
inspection. Inspect the actual model request and feedback to verify delivery.
runtime.state is episode-local; package memory is read-only during evaluation.
Meta edits memory between evaluations; episode notes are private.

## Units and execution
World coordinates are metres. Robot state quaternion is xyzw; consult each exact
control schema for rotation conventions. OSC commands are normalized deltas.
Gripper +1 closes, -1 opens. RGB-D pixels have top-left origin, depth in metres;
back-project from the matching camera, step and resolution. VLA chunk executes
one batch, not a complete pick; batch length comes from the existing backend.
episode_active is not an official success oracle. Do not fabricate historical
depth: full_rgb archives contain RGB, actions and robot state, not stored depth.

## Evidence and iteration
Use history_episodes → history_decisions → history_decision, and history_timeline /
history_frame for visual evidence. For online diagnostics use live_start,
live_decision, live_observe, live_capability or live_reply, then live_close.
Manual intervention is diagnostic, not autonomous success. Freeze and evaluate
with evaluate; poll evaluation_status; submit the candidate with evidence.
Existing Meta opening supplies exact schemas and budget. Respect its objective:
success first, then paired deployment-time improvement when eligible. Do not
modify host code, records or scoring. A new tool must be bound, actually called,
and independently evaluated; a successful submission alone proves no improvement.
