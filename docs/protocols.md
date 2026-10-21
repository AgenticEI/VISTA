# LIBERO-Pro protocols

| File | Suites | Tasks | Evaluation matrix per task |
| --- | --- | ---: | --- |
| `configs/protocols/libero-t.json` | Spatial-T, Object-T, Goal-T, LIBERO-10-T | 40 | 8 scenes × 2 repeats |
| `configs/protocols/libero-spatial-s.json` | Spatial-S | 10 | 8 scenes × 2 repeats |
| `configs/protocols/libero-object-s.json` | Object-S | 10 | 8 scenes × 2 repeats |
| `configs/protocols/libero-goal-s.json` | Goal-S | 10 | 8 scenes × 2 repeats |
| `configs/protocols/libero-long-s.json` | LIBERO-10-S | 10 | 8 scenes × 2 repeats |

All configurations start from `harnesses/libero_pro_h0_min`. This H₀ exposes observation and Pi0.5 execution skills. Meta development can implement further skills using the public capability catalog in `robots/libero/harness_capabilities.json`.

The runner focuses on one suite. When using the combined T template, select it explicitly with `tools/configure_run.py --suite spatial|object|goal|10` (choose one value), along with the required model, endpoint, template, and output arguments. This writes a standalone suite configuration with its original task/scene/repeat matrix and budgets. Use a separate run directory per suite. S templates already contain one suite each.

The native backend suffixes `*_task` and `*_swap` represent T and S respectively. The Long suite uses backend identifier `10`. Seeds 1000–1007 map to the first eight benchmark initial states. Each suite therefore has 160 autonomous evaluation episodes per complete condition.

Task lists, execution horizons, starting Harnesses, and scene/repeat matrices come from the research configurations. Private endpoints and machine paths have been replaced by explicit local configuration. Use `tools/configure_run.py` to select available planner and Meta models; protocol model IDs are recorded research settings, not a public model-availability guarantee.

The schema's `mode: probe` means a task subset or reduced schedule; it does not by itself imply an incomplete task matrix. Inspect the task, seed, repeat, round, and budget fields to understand the actual experiment. Full configurations retain their recording settings, which may include depth; the small example uses `full_rgb` to avoid storing depth arrays.

`configs/examples/libero.json` selects Goal-t09, one scene, one repeat, and one candidate round after H₀ evaluation. There is no joint phase. It allows two internal evaluation requests and one live trial, uses low concurrency, and retains the research execution horizon. It is an installation example, not the full paper evaluation.

A new development run can generate different candidate code and outcomes. Exact evaluation of the published taskwise selections additionally requires the historical evolved Harness commits, runtime snapshots, resource versions, and evaluation records. These artifacts are separate from this initial source release.

When resuming, keep the run configuration, source, and resources fixed. Create a new run directory when changing experimental conditions.
