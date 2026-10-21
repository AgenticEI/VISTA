# VISTA

**Active Exploration for Self-Improvement in Embodied Agents**

Zhang Zhang, Shengjun Zhang, Simin Huang, Min Chen, Chong Xia, Xin Fei, Yueqi Duan

[Project page](https://agenticei.github.io/VISTA-page/) · [Paper](https://agenticei.github.io/VISTA-page/assets/vista-paper.pdf) · [中文说明](README.zh-CN.md)

VISTA lets a Meta-agent develop a robot's executable **Harness** through execution evidence, active exploration, and code revision. The Meta-agent can control live simulator experiments, implement skills and decision logic, and submit frozen versions for autonomous evaluation. Useful changes are retained for subsequent development. The underlying model parameters remain fixed.

This first code release supports **LIBERO-Pro**, including task/instruction perturbations (T) and position perturbations (S). It contains the complete development runtime, LIBERO adapter, initial Harness, experiment configurations, and offline tests. RoboCasa and RoboTwin integration will follow separately.

VISTA builds on [RPent](https://github.com/RLinf/RPent). The Python package and CLI names remain `rpent` and `rpent-evolve` for compatibility.

## Repository layout

| Path | Purpose |
| --- | --- |
| `meta_harness/` | Meta development, active exploration, candidate evaluation, selection, and version history |
| `rpent/` | Executing Agent, Harness runtime, model/tool interfaces, traces, and dashboard |
| `robots/libero/` | LIBERO environment, tools, public capabilities, and benchmark manifests |
| `harnesses/libero_pro_h0_min/` | The initial H₀ used by the LIBERO-Pro configurations |
| `configs/examples/libero.json` | One-task, one-scene setup example |
| `configs/protocols/` | LIBERO-Pro T/S experiment configurations |
| `tools/` | Portable configuration and launch commands |
| `tests/` | LIBERO and shared-runtime tests with controlled models and environments |

## Installation

Use Linux and Python 3.10–3.12. Install from the source checkout in **editable mode**: adapters, Harness packages, and configuration files are loaded from the repository.

```bash
git clone https://github.com/AgenticEI/VISTA.git
cd VISTA
python -m pip install -e '.[test]'
```

Install the corresponding dependencies in the Python environment that will run each service:

```bash
# Simulator environment
python -m pip install -e '.[libero-pro]'
liberopro-download-assets --skip-existing

# Pi0.5 policy environment
python -m pip install -e '.[openpi]'

# SAM3 perception environment
python -m pip install -e '.[sam3]'
```

These environments may be separate. The LIBERO extra pins MuJoCo to `3.3.0`, matching the simulator behavior expected by the experiment code. Prepare the LIBERO-Pro assets, Pi0.5 checkpoint, SAM3 checkpoint and BPE file using the relevant projects' installation instructions. Weights and simulator assets are external resources and are not included in this repository.

The host also needs a compatible Codex CLI for Meta development and access to the selected language models. Set `CODEX_BIN` if the executable is not in a standard location. Model IDs in the research configurations record the experiment settings; choose IDs available through your provider when starting a new run. The paper’s recorded model variants and ablation resource settings are listed in [experiment-models.json](docs/experiment-models.json).

## Start one small experiment

```bash
cp configs/env.example.sh configs/env.local.sh
# Fill in interpreter/checkpoint paths and credentials in this ignored local file.
source configs/env.sh

python tools/configure_run.py \
  --template configs/examples/libero.json \
  --output configs/local/libero.json \
  --planner-model 'openai-responses:YOUR_PLANNER_MODEL' \
  --meta-model 'YOUR_META_MODEL' \
  --api-base-url 'https://api.openai.com/v1'

# Checks configuration and H₀ without starting models or simulators.
python tools/check_config.py configs/local/libero.json

# Starts services and runs the experiment; repeating it resumes the same run.
bash tools/run-rsi.sh --config configs/local/libero.json --run runs/libero-example
```

The example uses **Goal-t09**, one scene, one repeat, and one candidate development round after the baseline. It permits up to two internal evaluation requests and one live trial. RGB observations, actions, and state are recorded using `full_rgb`; archived depth/geometry arrays are not written in this mode. This recording choice does not disable live sensing tools.

The example assigns model services to GPU 0 and environment workers to GPU 1. Adjust the resource profile for your hardware. Memory requirements depend on the external models. Running the experiment starts model services and can incur API costs.

Inspect progress with:

```bash
python -m meta_harness.cli evolve-status runs/libero-example
```

The Meta process executes candidate code with host filesystem access. Run it in an appropriate host/container with the resources you intend it to access. Interrupting the coordinator can leave valid services and workers running for resumption; its `--seconds` option is not a full experiment budget or a dry run.

## LIBERO-Pro configurations

`configs/protocols/libero-t.json` contains all four T suites. Four `libero-*-s.json` files contain the corresponding S suites. Each suite has 10 tasks and evaluates each task on eight scenes with two repetitions (160 episodes per condition).

Use the same configuration helper to create local copies of protocol files. For the multi-suite T template, add `--suite spatial`, `--suite object`, `--suite goal`, or `--suite 10`, and use a separate run directory for each suite. The runtime develops one focused suite per run; listing four suites does not automatically run all four. See [protocol details](docs/protocols.md) for task mappings, seeds, and reproduction scope.

This release includes the **initial H₀ and development code**. The paper's results use task-specific evolved Harness versions selected from historical runs. Those selected packages and research records are separate artifacts; starting a new development run does not guarantee the same candidate code or success rate.

## Experiment artifacts

The runtime preserves its native research structure:

```text
runs/<name>/
├── config.json, initial.json, state.json
├── benchmark.json            # resolved tasks and scene mappings
├── harness.git/              # versioned Harness code and lineage
├── meta/attempts/            # development conversations and tool evidence
├── workspaces/               # candidate Harness packages
├── evaluations/              # evaluation requests and results
├── episodes/                 # autonomous/live executions and traces
├── selections/               # version-selection decisions
└── runtime-source*/          # framework snapshot for the run
```

Original experiments, conversations, credentials, checkpoints, and machine-specific configurations are not part of this source release. Inspect and sanitize run artifacts before sharing them.

## Verification

After installing the LIBERO, policy/perception, and test dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -B -m pytest -q -p no:cacheprovider tests
```

The initial release passed **142 tests and 22 subtests**, all six shipped configuration checks, and source-package build checks.

The release checks cover Harness loading, LIBERO task/scene manifests, development and selection, planner/tool boundaries, private policy sessions, evaluation records, and RGB trace integrity. Tests use fake model/environment responses. They do not establish the task success rate of a newly installed simulator or a new development run. Fresh end-to-end model-driven validation of this assembled release is pending.

## License and citation

The code retains the [Apache-2.0 license](LICENSE) and original RPent copyright notices. External simulators, model weights, and assets keep their respective licenses.

```bibtex
@misc{zhang2026vista,
  title={VISTA: Active Exploration for Self-Improvement in Embodied Agents},
  author={Zhang Zhang and Shengjun Zhang and Simin Huang and Min Chen and Chong Xia and Xin Fei and Yueqi Duan},
  year={2026},
  url={https://agenticei.github.io/VISTA-page/}
}
```
