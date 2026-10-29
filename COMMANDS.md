# Pipeline and demo command guide

New to the runtime or scientific Python? Start with the
[21-page setup and learning PDF](docs/learning/EXPERIMENT_SETUP_WORKBOOK.pdf)
and its [editable source](docs/learning/EXPERIMENT_SETUP_WORKBOOK.md).
It includes six guided exercises, expected answers and setup troubleshooting.

## Final-submission controls (2026-10-28)

```bash
./run_submission_controls.sh plan
./run_submission_controls.sh audit
./run_submission_controls.sh run --quick
./run_submission_controls.sh run
```

Full `run` trains seven BC controls, evaluates them and zero native gait actions,
then produces checkpoint fidelity curves and reports. It starts/stops its own
servers and runs without interactive checkpoints. Existing results are preserved.
Use `train`, `evaluate`, `curves`, `report` for separate stages. `zero` independently
evaluates the untrained constant baseline; its report command is `report --zero-only`.
Optional `--diagnostics` must be chosen consistently for its separate batch;
it adds early-stop diagnostics and also includes the standard fits. See
[submission_controls/README.md](submission_controls/README.md) for budgets,
output locations, resume contracts and smoke limitations.

## New consolidated study (2026-10-15)

Use `./run_training.sh plan` / `./run_training.sh run` and
`./run_evaluation_all.sh plan` / `./run_evaluation_all.sh run` for the new
standardized sample-budget study. Both manage their own servers. Full details,
small smoke commands and optional selections: [consolidated/README.md](consolidated/README.md).
These create new original and varied hidden/input BC, mixed DAgger and two GAIL
lineages. The older commands below remain available for historical reproduction.

Command reference for the current Walker repository (2026-10-11). Run commands
from this directory. These are alternatives for different tasks, **not a sequence
to execute in full**. Existing experiments are completed; preserve their outputs.
For implementation, metrics and data contracts, see [PIPELINES.md](PIPELINES.md).
For current research scope, see [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

## 1. Environment and server ownership

```bash
cd /home/pamnt/robocup/Walker
conda activate fcportugal
python3 check_simulator_config.py
```

SimSpark must use `$agentSyncMode = true` and
`$enableRealTimeMode = false`. Restart the server after changing configuration.
Launchers support `PYTHON_BIN` when a different interpreter is needed.

| Pipeline | Start the simulator yourself? |
| --- | --- |
| Original BC, DAgger and GAIL training | Yes |
| `run_evaluation.sh` (original nominal evaluation) | Yes |
| `run_demo.sh` (interactive viewer demo) | Yes; also open RoboViz |
| `run_varied_slip.sh` smoke/fixed/sampled | No; launcher manages servers |
| Variable training and its live evaluations | No; launcher manages servers |
| Context/FiLM evaluations and BC repeatability | No; launcher manages servers |
| Context/FiLM training and all representation probing | No simulator required |

For managed live runs, leave ports **3100 and 3200 free** and run only one managed
job at a time. Do not leave a manually started simulator running. The launchers
own server startup, environment verification and teardown. A report-only command
does not need a server.

For the original pipelines or a demo, start this in a separate terminal:

```bash
rcssserver3d
```

Do not run another agent controller against that server at the same time.

## 2. Demo / test viewer

Start `rcssserver3d`, open your installed **RoboViz** viewer and connect it to the
server's monitor port, localhost **3200**. Then, in the activated project terminal:

```bash
./run_demo.sh
# Optional target/action random seed (default 101):
./run_demo.sh --seed 202
```

Select a controller in the terminal menu:

| Number | Controller | Checkpoint |
| --- | --- | --- |
| 1 | FC Portugal expert | Native Walk controller |
| 2 | BC | `bc_walk_policy.pth` |
| 3 | Original DAgger | `dagger_policy.pth` |
| 4 | Mixed DAgger | `dagger_mixed_policy.pth` |
| 5 | GAIL initialized from original DAgger | `gail_dagger_policy.pth` |
| 6 | GAIL initialized from BC | `gail_bc_policy.pth` |
| 7 | GAIL initialized randomly | `gail_random_policy.pth` |
| 8 | Uniform-random action baseline | Requires `expert_train_data.npz` for bounds |
| 0 | Exit | — |

Each invocation demonstrates **one controller for 1,500 active steps**, or 30
simulated seconds at 50 Hz, plus startup/fall recovery. Ctrl+C stops the demo.
Missing checkpoints cannot be selected. The demo does not launch RoboViz or the
server, and it does not save benchmark results. It currently selects the original
root checkpoints, not context/FiLM or varied-training replicas. There is no
`--policy` or `--checkpoint` flag. `start.sh` launches the soccer team, not this demo.

## 3. Original imitation training

These commands require a manually running server. They can update datasets and
checkpoints; do not rerun them merely to view completed agents.

### Behavioural cloning

```bash
./run_pipeline.sh
```

Defaults: 10 rounds, each with 5,000 long-run and 5,000 short-turn samples;
train/test split followed by BC fitting. Expert collection appends to the existing
dataset. Main outputs: `expert_walk_data.npz`, `expert_train_data.npz`,
`expert_test_data.npz`, `bc_walk_policy.pth` and `artifacts/bc/training_metrics.json`.

Example isolated new run, rather than changing the original artifacts:

```bash
mkdir -p experiments/manual_bc_v1
DATASET=experiments/manual_bc_v1/expert_walk_data.npz \
TRAIN_DATASET=experiments/manual_bc_v1/expert_train_data.npz \
TEST_DATASET=experiments/manual_bc_v1/expert_test_data.npz \
MODEL_OUTPUT=experiments/manual_bc_v1/policy.pth \
METRICS_OUTPUT=experiments/manual_bc_v1/training_metrics.json \
./run_pipeline.sh
```

### DAgger

```bash
./run_dagger.sh
./run_dagger_mixed.sh
```

Choose one: original DAgger uses a 20,000-step seed dataset; mixed DAgger uses
five long/short seed rounds totaling 100,000 samples. Both default to seven
10,000-step learner rollouts with oracle labeling and the existing expert/learner
mixing schedule. Artifacts are under `artifacts/dagger/` and
`artifacts/dagger_mixed/`; exported policies are `dagger_policy.pth` and
`dagger_mixed_policy.pth`.

Existing aggregate data can be reused, but rerunning these scripts is additional
training, not an exact completed-iteration resume. `RESET_DAGGER_DATASET=1` resets
the training dataset/state according to the selected launcher; use only when
intentionally starting over. Consult `PIPELINES.md` before resetting.

### GAIL

```bash
./run_gail.sh          # Original-DAgger initialization
./run_gail_bc.sh       # BC initialization
./run_gail_random.sh   # Historical random-initialization control
```

Defaults: 10 iterations with 4,000 on-policy steps each. Artifacts are under
`artifacts/gail_dagger/`, `artifacts/gail_bc/` and `artifacts/gail_random/`;
exports are the corresponding `<label>_policy.pth` files at the root.
The **expert remains FC Portugal demonstration data**; BC/DAgger only initialize
the actor. The recorded GAIL-DAgger lineage is original DAgger, not mixed DAgger.

`RESET_GAIL=1 ./run_gail_bc.sh`, for example, archives previous training state
before starting over. Without reset, existing state can seed another training
batch; this is not a reason to rerun a completed experiment.

## 4. Original nominal evaluation and plots

Requires a manually running server:

```bash
./run_evaluation.sh
```

Defaults: 3,000 active steps, seeds 101/202/303/404/505, available original
controllers, output `evaluation/`. **`RESET_EVALUATION` defaults to 1** and backs
up previous output. To retain completed runs and resume:

```bash
RESET_EVALUATION=0 RESUME_EVALUATION=1 ./run_evaluation.sh
```

For a separate comparison:

```bash
OUTPUT_DIR=experiments/nominal_evaluation_new ./run_evaluation.sh
```

The launcher produces `comparison.json` and `comparison.csv`. Offline summary
and plot commands for the default evaluation directory:

```bash
python3 summarize_evaluation.py
python3 plot_evaluation.py
```

To plot a different **legacy evaluation comparison**:

```bash
python3 plot_evaluation.py \
  --report experiments/nominal_evaluation_new/comparison.json \
  --output experiments/nominal_evaluation_new/evaluation_summary.png \
  --pdf experiments/nominal_evaluation_new/evaluation_summary.pdf
```

The figure includes the comparison table. Do not feed context, variable-training
or probing report JSON into this plotter; those pipelines have their own reports.

## 5. Frozen-policy slip evaluations

No manually running server:

```bash
./run_varied_slip.sh smoke
./run_varied_slip.sh fixed
./run_varied_slip.sh report-fixed
```

Current corrected scalar-slip grid: **0.025, 0.1, 0.2, 1 and 10**, with three seeds
101/202/303 and 3,000 active steps per evaluation. Full fixed evaluation covers
eight controllers, including random and oracle. Outputs are under
`experiments/phase2_domain_robustness/scalar_wide_v4/`.

Repeat the identical command to resume verified completed deployments. The
historical two-argument slip setter did not change effective physics; do not merge
those old results with the corrected scalar-slip experiment.

## 6. Varied-environment training and paired evaluation

The default plan is
`experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json`.
Training uses the fixed slip grid; evaluation draws a positive, bounded Gaussian
slip and holds it **fixed for the entire run**. All controllers in that round
receive the same slip and task seed. This is not per-contact surface noise.

```bash
./run_variable_training.sh plan
./run_variable_training.sh all --arm varied
./run_variable_training.sh evaluate --arm varied --suite grouped
./run_variable_training.sh report --arm varied --suite grouped
```

`all` trains; it does **not** automatically run the final evaluation. Individual
training stages are `collect`, `bc`, `dagger_mixed`, `gail_bc`, `gail_dagger`.
Run collection first when a new plan needs demonstrations.

`--arm nominal` is matched additional training at slip 0.1; `--arm varied` uses
the grid. Omitting `--arm` selects both. A nominal training arm is not the same
thing as testing retention on a nominal surface.

The default `grouped` suite uses five training-slip-centered Gaussian groups,
three draws each: 15 rounds, five controllers per varied-only round (oracle plus
four trained policies), or 75 deployments. Other suites are `gaussian`, `fixed`
and `all`. To evaluate original frozen checkpoints on the selected distribution:

```bash
./run_variable_training.sh baseline --suite grouped
./run_variable_training.sh report --frozen-only --suite grouped
```

Repeat the exact command to resume. Changed plan settings require a new `--plan`
path, not deletion of receipts. `--quick` is engineering-only, not research data.

## 7. Context-conditioned BC and FiLM

Training is offline using collected BC data; evaluations manage their servers.
These experiments are already completed. Commands below document reproduction.

### Hidden-context and input-conditioned BC

```bash
./run_context_training.sh audit
./run_context_training.sh train --device cuda
./run_context_evaluation.sh evaluate --suite smoke
./run_context_evaluation.sh evaluate --suite all --modes correct constant wrong
./run_context_evaluation.sh report --suite all --modes correct constant wrong
```

Default training root: `experiments/context_conditioning/bc_input_v1/`.
Training seeds 42/43/44 produce separate replicas for each architecture, not an
ensemble. The complete all-mode comparison contains 429 deployments; default
grouped/correct evaluation is smaller. Keep suite/mode flags identical on resume
and report generation.

### FiLM append-only comparison

```bash
./run_film_training.sh audit
./run_film_training.sh train --device cuda
./run_film_evaluation.sh evaluate --suite smoke
./run_film_evaluation.sh evaluate
./run_film_evaluation.sh report
```

Default training root: `experiments/context_conditioning/bc_film_v1/`.
Default evaluation is all suites with correct context: 99 new FiLM deployments,
reusing the verified prior comparison rather than rerunning its controls.
Reports are in the selected evaluation output directory (`comparison.md`).

### BC repeatability

```bash
./run_bc_repeatability.sh plan
./run_bc_repeatability.sh evaluate
./run_bc_repeatability.sh report
```

Default: training replica seed 42, five repeated deployments per selected case,
3,000 active steps, 45 runs total. This measures deployment repeatability, not
five additional independently trained models.

## 8. Offline internal-representation probing

No simulator or source-policy training. These analyses are completed; use `report`
for report regeneration. `audit` checks required inputs/provenance; `run` performs
the fits and supports verified resume.

```bash
# Main broad locomotion representation study:
./run_representation_probing.sh audit
./run_representation_probing.sh run
./run_representation_probing.sh report

# Expert comparison appended to that study:
./run_expert_probing.sh audit
./run_expert_probing.sh run
./run_expert_probing.sh report

# Earlier, narrower slip-focused study (historical):
./run_probing.sh audit
./run_probing.sh run
./run_probing.sh report
```

| Study | Default output directory |
| --- | --- |
| General representations | `experiments/representation_probing/nao_sources_v1/` |
| Expert/student comparison | `experiments/representation_probing/nao_expert_comparison_v1/` |
| Historical slip probes | `experiments/probing/original_policies_v1/` |

General probes cover BC, mixed DAgger, original-DAgger GAIL and GAIL-BC.
Expert probing adds the native expert and comparisons, preserving existing fits.
See [representation_probing/README.md](representation_probing/README.md) and
[representation_probing/EXPERT.md](representation_probing/EXPERT.md).
These are measured decodable properties, not an exhaustive inventory of all
representations or evidence of R1 transfer. R1 work is currently deferred.

## 9. Historical / specialist environment tools

`run_contact_slip.sh` and `run_slip_sensitivity.sh` belong to the environment
validation studies, not the standard trained-policy evaluation sequence. Inspect
their options before reproducing those experiments:

```bash
./run_contact_slip.sh --help
./run_slip_sensitivity.sh --help
./run_varied_slip.sh fixed --help
./run_variable_training.sh --help
```

The older `run_phase2_smoke.sh` and `run_phase2_condition.sh` expect an externally
configured environment/server; prefer `run_varied_slip.sh` for the corrected
managed-server frozen grid. Original training shell launchers use environment
variables, **not `--help` argument parsing**: do not invoke them with `--help`
expecting a harmless options listing.

## 10. Preserving results

Use a fresh output/plan location for changed checkpoints, data or protocol.
Managed pipelines verify identities and reject changed inputs rather than silently
combining experiments. Keep the same arguments when resuming. Do not remove locks,
receipts or prior experiment directories to bypass these checks. A simulator crash
does not require retraining the policies; resume the relevant evaluation once the
server/port problem is resolved.
# Select a model and slip for RoboViz

```bash
./run_viewer.sh
./run_viewer.sh --root experiments/consolidated/smoke_v1
./run_viewer.sh --list
./run_viewer.sh --model varied/input/seed_42/bc --slip 1
```

This viewer starts its own server; open RoboViz separately, connect to
localhost:3200, then press Enter in the launcher. Do not start rcssserver3d
manually or run this alongside training/evaluation. Choose `original` to preserve
the original scene, or a numeric fixed slip. Context models require numeric slip.
Only existing historical checkpoints and completed consolidated models are listed.

# Probe the completed consolidated models

```bash
./run_consolidated_probing.sh audit
./run_consolidated_probing.sh run
# Optional regeneration of completed reports:
./run_consolidated_probing.sh report
```

Offline CPU probes of28new policies plus expert, using existing oracle traces.
No server or retraining. Output: `experiments/representation_probing/consolidated_v1`.
Full protocol: [consolidated_probing/README.md](consolidated_probing/README.md).

# Latest consolidated results report

Read [the staged numerical report](evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md)
for experimental design, imitation/mechanical/environmental results, DAgger-specific
analysis and general/expert probing. To print its reproducible Markdown:

```bash
python evaluation/analysis/consolidated_summary.py
```

This reads saved reports only; no server or retraining is needed.

# Add scratch-model comparisons without rerunning existing results

```bash
./run_training.sh plan --include-scratch
./run_evaluation_all.sh audit --include-scratch
./run_training.sh run --include-scratch && \
./run_evaluation_all.sh run --include-scratch && \
./run_consolidated_probing.sh run --include-scratch
```

Existing models,1,020deployments and300probe fits are retained/reused. New artifacts
and combined reports go under `experiments/consolidated/v1/scratch_v1/`.
This adds varied scratch BC/DAgger and their GAIL variants, plus direct random-GAIL
in original and varied-hidden conditions. No new input-conditioned random-GAIL.
Read [the scratch comparison guide](scratch_comparison/README.md) for seeds,
budgets, optional smoke runs, flags and resume behaviour. No manual server needed.

# Optional study outside the paper — 2026-10-18

`./run_optional_robustness.sh plan` previews/pins the oracle stress design.
`./run_optional_robustness.sh smoke` runs seven slips with one seed each.
`./run_optional_robustness.sh evaluate` runs all21crossed slip/seed deployments.
Servers are managed; do not start one manually. Reports are automatic, or use
`report` / `report-smoke`. No RL training command yet. Full details:
[optional study guide](optional_robustness/README.md). Paper candidate text:
[final-paper draft](evaluation/FINAL_PAPER_DRAFT.md).

# Exploratory GAIL refinement — 2026-10-19

`./run_gail_improvement.sh plan` and `audit` are offline.
`./run_gail_improvement.sh run --quick` checks three initializers in a separate
smoke directory. `./run_gail_improvement.sh run` trains/evaluates the full350kstep
per actor study (nine actors). Servers are owned; do not prestart one. `report`
regenerates completed comparisons; `train` and `evaluate` split the workflow.
Same-budget legacy comparison: add `--recipes legacy adaptive` and a new output.
See [GAIL improvement guide](gail_experiments/README.md). This is outside the paper.

