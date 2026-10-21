# VISTA Harness development runtime

The shared entry point is:

```bash
python -m meta_harness.cli run --config CONFIG.json --run RUN_DIRECTORY
```

The same command initializes or resumes the durable experiment. See the repository
[README](../README.md) for editable installation, external model/simulator resources,
portable environment configuration, the LIBERO H0 package, and a one-task LIBERO-Pro example.

`python -m meta_harness.cli evolve-status RUN_DIRECTORY` inspects state. The
coordinator's `--seconds` flag limits that coordinator invocation; workers may
continue so the experiment can resume. It is not an offline check.

Use `python tools/check_config.py CONFIG.json` for a configuration/Harness check
that does not start any services or model requests.
