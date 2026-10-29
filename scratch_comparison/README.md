# Scratch versus pretrained continuation — append-only extension

## Run

The completed consolidated batch remains the parent. Use the existing entrypoints
with an explicit flag to add this experiment:

```bash
# Read-only checks / counts; does not train or launch a server:
./run_training.sh plan --include-scratch
./run_evaluation_all.sh audit --include-scratch

# Train new models, then evaluate them, then append their representation probes:
./run_training.sh run --include-scratch && \
./run_evaluation_all.sh run --include-scratch && \
./run_consolidated_probing.sh run --include-scratch
```

Without `--include-scratch`, the original entrypoints and immutable experiment
contracts are unchanged. With the flag, old models are not trained again and old
deployment/probe results are verified and reused. Repeating the commands skips
completed **new** jobs too. A receipt is checked, not merely file existence.
Changed/corrupted inputs fail rather than silently skipping or overwriting results.
No manual server is needed; training/evaluation retain the owned server and shared
lock. Probing is offline. Do not run two writers for the same probe output.

Defaults:

- Parent: `experiments/consolidated/v1`.
- New training: `experiments/consolidated/v1/scratch_v1/policies`.
- New evaluation plus combined reports: `.../scratch_v1/evaluation/all_all_correct`.
- New probe fits plus combined reports: `.../scratch_v1/probing`.
- Old models, datasets, schedules, receipts and reports are never rewritten.

Training/evaluation accept `--output PATH` for the **existing parent batch**, and
`--extension-output PATH` for a separate new output. `--algorithm` selects training
only (`bc`, `dagger_mixed`, `gail_bc`, `gail_dagger_mixed`, `gail_random`, or `all`).
Required new BC/DAgger initializers are built automatically for their GAIL variants.
Full evaluation requires all new models. `report` rebuilds combined reports without
simulation; `audit` verifies parent data/evaluation artifacts without training.
The old `--domain`, `--cohort`, `--suite`, `--context-modes` flags do not apply to
the extension: its design intentionally uses the exact parent's full/correct-context
schedule. Custom constant/wrong-context expansions remain a separate study.

Probe overrides: `--batch` (parent), `--scratch` (scratch output), `--parent-probes`
(default `experiments/representation_probing/consolidated_v1`), `--output` (new
probe output). Parent probes must be complete; stride/splits/targets are inherited,
not adjustable independently. `audit` requires trained scratch checkpoints but
does not fit probes; probes can run after training without waiting for new evaluation.

## Conditions and lineage

| Condition | Original domain | Varied hidden | Varied input |
|---|---|---|---|
| Scratch BC | Existing original BC already covers this; no duplicate | 3 seeds | 3 seeds |
| Scratch mixed DAgger | Existing original DAgger already covers this | 3 seeds | 3 seeds |
| GAIL from scratch-trained BC | Existing original GAIL-BC already covers this | 3 seeds | 3 seeds |
| GAIL from scratch-trained mixed DAgger | Existing original GAIL-DM already covers this | 3 seeds | 3 seeds |
| Direct random-initialized GAIL | 1 seed (42) | 3 seeds | **Not added** |

This adds **28 checkpoints**: 24 scratch-lineage varied policies plus four
checkpoints for the two random-GAIL conditions requested by the user. The original
28 checkpoints remain, giving 56 learned checkpoints in the combined comparison.
The two random-GAIL *conditions* do not mean exactly two checkpoint files: the
varied condition follows the established three-replica convention (42/43/44).

Scratch BC and DAgger start with random weights and no original-domain ancestor.
For a paired seed, hidden and input actors begin with the same random function of
the original 63 observations; the added slip column is zero. This deliberately
matches initial behaviour while allowing context to learn. It is not copied from
a trained model. Standard actor sizes stay 63/64→256→256→16.

GAIL-BC and GAIL-DM still have supervised initialization, but **their entire
lineage is trained directly in the varied environment**, without original-domain
pretraining. Direct GAIL-random has no pretrained actor at all. It retains the
same discriminator regularization, PPO safeguards, and supervised anchor as the
existing GAIL implementation; it is not an unanchored-GAIL ablation. Its expert
remains FC Portugal and its static expert pool is unchanged.

## Budgets and shared data

Reuse the exact parent's verified oracle training/validation/test shards. Nothing
is recollected. BC uses 170,000 examples for 100 epochs. Mixed DAgger uses the
100,000 seed pool, then seven 10,000-step learner rounds with β=0.8/0.6/0.4/0.2/0/0/0,
50 supervised epochs initially and after each aggregation. Each GAIL uses 170,000
static examples plus 70,000 PPO rollout steps in seven rounds. The direct random
condition does not receive a preliminary BC fit. All varied rounds balance five
slips (0.025/0.1/0.2/1/10) and long/short target modes. Validation/test: 10,000 each.

These are **matched incremental varied-domain budgets**, not equal lifetime cost:
the continued policies additionally inherited original-environment training. The
study asks what that pretrained starting point adds under the same subsequent
budget. It does not isolate initialization from all accumulated historical compute.
Shared demonstrations are not counted as newly collected data in the extension's
training summary. DAgger/GAIL learner rollouts are newly collected; incomplete new
jobs are archived using the existing `_failed_*` retry mechanism.

## Evaluation reuse and reports

Use the identical saved rounds: five original-scene cases, 15 grouped draws,
15 confirmation draws and three corrected scalar-nominal cases. Hold slip fixed
within each deployment. Original random-GAIL is tested in all 38 rounds; the 27
new varied policies are tested in the 33 non-original rounds.

- Reuse **1,020** parent deployments, including oracle/random baselines.
- Add **38 + 27×33 = 929** new deployments, 3,000 active steps each.
- Combined report: **1,949** run records; no duplicate baseline deployments.

New results use the same held-out expert pools, normalization references, action
MSE, joint sliced Wasserstein, falls, velocity/tracking, torso RMS and jerk metrics.
`comparison.json/csv/md` contains both generations. Paired differences compare
scratch versus continuation at identical seed/round, and scratch input versus
scratch hidden. Direct random-GAIL remains an explicitly labelled extra condition,
not silently substituted for supervised-initialized GAIL. No claim of better
performance is made until this new batch has run.

## Probe reuse

Reuse the exact 30 oracle trajectories, 10/5/15 split, stride 7, 30 targets, ridge
penalties and random/shuffled controls of the completed consolidated probe study.
The old plan must match a fresh audit and all 300 old fit receipts must verify.
Fit only 56 new hidden-layer representations ×4 label modes = **224 new fits**.
The combined study has 131 representations, **524 fits and 15,720 target scores**.
Old expert/raw/random representations are not fitted again.

`report.json`, `probe_scores.csv`, `expert_comparison.csv` include both cohorts;
`initialization_comparison.csv` gives same-layer/property scratch-minus-continuation
R²/RMSE differences. New layers also get expert CKA, within-run permutation controls
and spectra. A higher R² or CKA is not a percentage of knowledge or causal evidence.
Existing staged interpretation reports are preserved and remain about the parent
batch until the new results are available.

## Small integration run and validation

To use the already completed quick parent without consuming a full research budget:

```bash
./run_training.sh run --include-scratch --quick --output experiments/consolidated/smoke_v1
./run_evaluation_all.sh run --include-scratch --quick --output experiments/consolidated/smoke_v1
```

This adds 10 models and 11 deployments to the quick parent. Full representation
probing intentionally rejects smoke data because it lacks the required 30 oracle
trajectories. Smoke checks wiring, not learning quality. Existing original smoke
artifacts remain separate and untouched.

Offline tests:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest scratch_comparison.test_pipeline
```

Checks cover counts, genuine random initialization, paired context initialization,
training lineage/data budgets, corruption rejection, and no-refit resume of
initializers/probes. Live scratch training/evaluation are user-launched, not run
automatically by adding this extension.
