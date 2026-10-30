# Convergence diagnostics and untrained-network engine controls

This is a separate copy/adaptation of `submission_controls`, with the original
files unchanged. `COPY_PROVENANCE.json` records the copied source hashes. All
historical checkpoints, datasets and evaluation receipts are read-only inputs.

## Run

```bash
# Original environment fits + full existing evaluation schedule (default).
./run_convergence_controls.sh run

# Varied-slip continued AND scratch BC/DAgger, hidden AND input-conditioned.
./run_convergence_controls.sh run --domain varied

# Both original and varied fits in one batch.
./run_convergence_controls.sh run --domain all
```

Choose the domain before starting and keep it for every stage of that batch.
The complete command trains, evaluates and reports without manual checkpoints.
Do not start a simulator yourself: the existing owned-server architecture starts
and stops a fresh server for every deployment, verifies scene import, checks
restoration, holds the shared simulator lock, and refuses occupied ports.

Individual stages:

```bash
./run_convergence_controls.sh plan --domain all
./run_convergence_controls.sh audit --domain all
./run_convergence_controls.sh train --domain all
./run_convergence_controls.sh evaluate --domain all
./run_convergence_controls.sh report --domain all

# Engine control can run before convergence training finishes.
./run_convergence_controls.sh engine --domain all
./run_convergence_controls.sh report --domain all --engine-only

# Small separate smoke batch covering both architectures and histories.
./run_convergence_controls.sh run --domain all --quick
```

`engine` needs no completed convergence fits. Its three networks and deployments
are stored in the same batch as the full evaluation; `evaluate`/`run` later skip
those completed jobs. The engine-only and full reports have separate schedules
and receipts. `report` is offline and refuses incomplete inputs instead of
launching work. `train` restores Adam and uses CUDA when available, otherwise CPU;
live evaluation follows the existing CPU evaluator. `PYTHON_BIN` overrides the
launcher's default fcportugal Python interpreter.

| Domain selector | Additional fits | New full deployments | Default output |
|---|---:|---:|---|
| `original` (default) | 2 | 190 | `experiments/convergence_controls/v1/` |
| `varied` | 24 | 906 | `experiments/convergence_controls/varied_v1/` |
| `all` | 26 | 982 | `experiments/convergence_controls/all_v1/` |

Counts include three untrained networks, 114 deployments altogether. Each varied
fit gets 33 surface runs; each original fit and each untrained network gets 38.
Choosing separate batches does not share their new receipts; use `--domain all`
if you want one combined study without repeating engine controls across batches.
`--quick` uses `smoke_<domain>_v1`, seed 42, tiny training/validation slices, four
additional epochs maximum and 32-step deployments in two scenes. The all-domain
smoke contains ten tiny fits plus one random network and 20 deployments, including
fresh short-horizon oracle/random/zero references. It is not research evidence.

`--parent` defaults to `experiments/consolidated/v1`; `--controls` defaults to
`experiments/submission_controls/v1`. Full varied scope needs the completed
parent `scratch_v1` extension too. `--output` selects an isolated destination,
which must not overlap either historical tree. Identities are immutable: changed
code, stopping rules, data or selected scope require a new output directory.
Completed jobs are checked and skipped; an incomplete job is archived and rerun
from its inherited checkpoint, not resumed halfway through an epoch.

## What is being trained

Every fit uses its existing 170,000-example training pool and the original
10,000-example validation split for that domain. No new oracle demonstrations,
DAgger rollouts or labels are collected. Long/short mixtures remain exactly as
recorded in the saved training job. Both methods share validation paths within
a domain; hidden/input models retain their original 63/64-input contracts.

| Diagnostic | Starting policy and optimizer | Fixed data |
|---|---|---|
| Original BC | Validation-selected matched-update BC (211,050 performed updates; currently selected endpoint) | Original 170k expert examples |
| Original mixed DAgger | Final round-7 selected policy/optimizer (211,050 performed updates) | Original 100k expert seed + 70k collected labels |
| Varied continued BC | Existing selected varied BC; original pretraining retained | Saved varied 170k expert pool |
| Varied continued DAgger | Existing selected varied round-7 policy; original pretraining retained | Its saved final varied aggregate |
| Varied scratch BC | Latest **200-epoch** varied-only BC control, selected policy/optimizer | Saved varied 170k expert pool |
| Varied scratch DAgger | Existing selected scratch varied round-7 policy | Its saved final varied aggregate |

Varied rows include hidden and input-conditioned architectures, seeds 42/43/44.
We reuse the latest completed 200-epoch scratch BC controls instead of repeating
those additional epochs from the older 100-epoch checkpoints. “Scratch” describes
the lineage: the new diagnostic itself continues a trained checkpoint, not random
weights. There are no GAIL training jobs in this study.

Historical performed updates and sample exposures are summed from audited source
metrics, including original-stage fitting for continued lineages. This includes
work in discarded epochs; it is **not** asserted to equal the optimizer's step
counter at its selected checkpoint. Final-fit best epoch and selected/performed
updates are recorded separately. Oracle-label and learner-step lineage costs are
also retained. No additional training queries or simulation steps are needed.

The original pair starts after matched performed updates; the varied histories
do not have equal inherited budgets. The purpose is validation-stopping sensitivity,
not a new equal-compute comparison. After stopping, each fit can have a different
amount of additional optimization.

## Stopping and checkpoint selection

Defaults for every fit:

- Adam learning rate **1e-4**, batch size **256**, restoring the synchronized saved
  optimizer. A different inherited learning rate is rejected.
- Evaluate validation MSE **before the first new update** (additional epoch 0),
  then after every additional epoch. The inherited model is a selection candidate.
- Save the minimum actual validation MSE as `policy.pth` + `optimizer.pth`.
- Patience **25 epochs**, reset by an absolute validation improvement greater than
  **1e-5** relative to the last material-improvement reference.
- Stop no earlier than **25 additional epochs**, hard cap **1,000 additional epochs**.
- Preserve `last_policy.pth` and `last_optimizer.pth` separately from the selected pair.

Custom settings, fixed before running and repeated for later stages:

```bash
./run_convergence_controls.sh run --domain all \
  --patience 30 --min-delta 0.000001 --minimum-epochs 30 --max-epochs 1500 \
  --output experiments/convergence_controls/all_stricter_v1
```

`stop_reason=validation_patience` means the declared plateau rule was met.
`stop_reason=budget_exhausted` means it was **not** met before the cap. Neither
proves mathematical convergence. For DAgger, this is a **fixed-final-dataset
supervised convergence diagnostic**, not convergence of the online state-collection
algorithm. Selection is never based on test MSE, deployment progress or falls.

## What the random-network engine test means

As clarified by the user, the engine receives outputs from an **untrained model**.
Three standard BC networks (63→256→256→16, ReLU hidden layers) use PyTorch's usual
initialization with seeds 42/43/44. Their weights are fixed and they receive the
real observations at each step. Their outputs are deterministic, state-dependent
native gait actions. No optimizer is constructed and no training is performed.

This differs from both all-zero actions and independent uniform-random actions
at each step. The existing zero and uniform-random evaluations are reused as
comparisons. Normal targets, gait phase, action filtering, foot offsets, arm swing,
IK, fall recovery and stabilization remain active; outputs are not rescaled to
expert bounds or forced to zero. A result therefore assesses this exact untrained
network plus the existing engine, not an engine with no feedback whatsoever.

## Evaluation and reports

Reuse the saved 38-round schedule: five original installed-scene tasks, three
corrected scalar-slip-0.1 tasks, and 30 sampled fixed-slip rounds (15 grouped +
15 confirmation). Varied policies skip the original installed scene, matching the
prior protocol. The sampled slip is constant for each entire run; no new draws
are made. All full deployments have 3,000 active 20 ms steps.

Original BC/DAgger, matched BC, zero, oracle and uniform-random references are
verified and reused. Each varied convergence policy is additionally compared with
its exact own pre-diagnostic initializer on the same rows. Corresponding new BC
and DAgger policies are paired within architecture, lineage and training seed.

Outputs under the selected batch:

- `plan.json`, `runtime_identity.json`: immutable scope, rules and dependency hashes.
- `training/<label>/`: job, best/last policy+optimizer, `history.json`, `metrics.json`,
  worker log and completion receipt. History includes epoch-0 validation and
  training loss (the latter averages predictions made throughout that epoch).
- `evaluation/schedule_all.json` / `schedule_engine.json`: scoped schedules sharing runs.
- `evaluation/runs/<round>/<label>/`: trace, summary, owned-server evidence and receipts.
- `evaluation/summary_all/` / `summary_engine/`: comparison Markdown/JSON/CSV,
  paired differences, and validation-curve PNG/PDF when training is in scope.

Metrics retain action MSE, joint occupancy SWD, progress, normalized return,
velocity, falls/exposure, jerk and torso RMS. Original/scalar nominal/sampled
summaries remain separate. The historical direction error retains the known
20 ms action / 40 ms localization caveat; this study does not silently replace
that metric. Random-network training seeds remain separately identifiable.

## Offline checks

```bash
python -m unittest convergence_controls.test_pipeline -v
```

Checks cover inherited checkpoint retention, optimizer-state restoration, plateau
versus cap reporting, validation overlap, immutable random-network initialization,
resume/tamper detection, and engine-only reporting independent of later training.
