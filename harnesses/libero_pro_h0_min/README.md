# LIBERO H0-min

Initial policy: observe → one raw VLA action batch → observe/reason again.
The two robot-facing tool implementations live under skills/. Native dependencies
are file access, image display and finish; robot sensing, model inference and
physical execution remain host capabilities. No host code changes are required.
No task recipes, advanced native robot tools or recovery routines are bound.
Policy only bounds image history using the reference policy; it does not choose actions.

## Meta development entry
Read development/README.md and development/capabilities.json before extending
this package. These are development references, not automatically injected into
Planner prompts. The existing Meta opening also supplies the actual bound tool
schemas and full backend capability catalog. The runtime catalog is authoritative.
