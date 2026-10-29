# Fixed-grid training and Gaussian evaluation

## Current default: grouped Gaussians (no retraining)

```bash
./run_variable_training.sh evaluate --arm varied --suite grouped
./run_variable_training.sh report --arm varied --suite grouped
```

`grouped` is now the default CLI evaluation/report suite. Five parent means are
the training values [.025,.1,.2,1,10]; std is half each mean. Three independent
rejection-sampled draws per group, strictly positive and at most10. Each draw is
fixed for the full3,000-step run, shared with the same task seed across all models.
Varied-trained models+oracle:75 runs; both training arms+oracle:135 runs. Original
frozen models are not included by default. Servers are fresh per deployment.

New artifacts are isolated under evaluation_grouped_v1/ inside the existing plan
directory, including immutable schedule.json, per-round/controller receipts and
report.md/json. Reports include per-group means, progress, oracle ratios, falls
and targets, plus per-run slips/seeds and an overall summary. Parent means/stds
are before truncation: the mean10 group is upper-truncated and draws below10.
Lower bound is0 exclusive, not the training minimum .025; positive extrapolation
below the smallest training value is allowed. Three draws per group are a small
paired evaluation, not independent training replications.

Training plan/checkpoints are unchanged and remain usable. No retraining or plan
regeneration is needed. Previous five-round results/reports are retained; use
`--suite gaussian` explicitly to evaluate/report that older suite. Quick grouped
tests use one draw per group. Nineteen variable-training offline tests pass,
including grouped schedule, report isolation and per-group aggregation checks.

The five-round specification below describes the retained `gaussian` suite.

## Latest evaluation extension: five rounds

Production evaluation now uses five paired Gaussian rounds, matching the five
training slip values. The original training plan is deliberately unchanged so
completed checkpoints remain valid. The scheduler reproduces its original three
draws and appends two:5.774292699,6.790566449,3.413121168,4.054531697,6.099415186.
An immutable evaluation_schedule_5_rounds.json is written separately when evaluation
starts. Every model shares each round's slip/task seed. Existing matching completed
rounds resume normally; no training or original receipts are rewritten.

Use `evaluate --arm varied` then `report --arm varied`:25 runs including oracle.
Both arms give45 runs including oracle. No retraining or plan regeneration needed.
Quick plans still use one round; optional fixed suites still use three seeds.
References to three Gaussian rounds in the historical specification below describe
the original plan, not the extended production schedule.

This package trains **BC, mixed DAgger, GAIL-BC and GAIL-DAgger**. FC Portugal
remains unchanged and supplies demonstrations and the evaluation baseline.
Original DAgger and GAIL-random are not scheduled. GAIL-DAgger continues the
existing `gail_dagger_policy.pth`; its historical initializer is not relabelled as
mixed DAgger. Policies keep 63-D observations and 16-D latent actions; slip is hidden.

The corrected scalar-setting generation is now the default. Re-evaluate frozen
policies under the corrected physics before training. The launcher shares its
simulator lock with fixed evaluations and refuses concurrent use. Old data and
checkpoints are preserved, but legacy terrain-robustness claims are not validated.
Nominal and varied arms both use the corrected single-value field setter.

## Commands

Run from Walker with the fcportugal environment. The default immutable plan is
`experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json`.
Protocol: `fixed_grid_training_gaussian_evaluation_v4`; old plans cannot resume.

```bash
./run_variable_training.sh plan
./run_variable_training.sh all
./run_variable_training.sh evaluate
./run_variable_training.sh report
```

`evaluate` and `report` default to oracle plus newly trained policies ONLY, on
three Gaussian-drawn fixed surfaces. Each round uses the exact same slip and task
seed for every controller; each gets a fresh server. With both training arms there
are27 runs (3 x9 controllers); `--arm varied` gives15 runs (3 x5). Original frozen
models are excluded. The optional explicit `baseline`/`report --frozen-only` stages
remain available but are not required by this workflow. Frozen fixed tests remain
separate in run_varied_slip.sh. Explicit `--suite fixed`/`all` is optional; match
the reporting suite. Inspect fixed-test teacher viability before training. `all`
collects demonstrations and trains all four agents for both varied and nominal
control arms. It does not automatically run evaluation. Every command runs
without interactive prompts. You do not start rcssserver3d yourself.

Individual training commands (dependencies are collected/resumed automatically):

```bash
./run_variable_training.sh collect
./run_variable_training.sh bc
./run_variable_training.sh dagger_mixed
./run_variable_training.sh gail_bc
./run_variable_training.sh gail_dagger
```

Default `--arm both` includes budget-matched nominal-only continuation. Use
`--arm varied` for just the variable-slip arm and pass the same selection to
`evaluate` and `report`. Use `report --frozen-only` after only `baseline`.
Reports require every planned evaluation for the requested models; they do not
silently summarize an incomplete sweep. The unchanged oracle is included in both
arm selections; default reports omit original frozen policies, including fidelity
comparisons. Random-baseline normalized scores are deliberately
not reported: the requested scope excludes random. Reports include raw metrics,
oracle return ratios (not Scorenorm), nominal retention when the selected suite
includes nominal runs (otherwise null), held-out 16-D action MSE
(both per-element and squared L2/state) and collection costs.

For an engineering smoke test, use a separate tiny plan:

```bash
./run_variable_training.sh plan --quick --plan experiments/variable_slip_training/quick_grid_v4/plan.json
./run_variable_training.sh all --plan experiments/variable_slip_training/quick_grid_v4/plan.json
./run_variable_training.sh evaluate --plan experiments/variable_slip_training/quick_grid_v4/plan.json
./run_variable_training.sh report --plan experiments/variable_slip_training/quick_grid_v4/plan.json
```

Quick plans use 32-step segments/evaluations, one update epoch for supervised
learning, and one DAgger/GAIL iteration. They are not research results.

## Training settings and evaluation distribution

Training uses the fixed grid [.025,.1,.2,1,10], assigned round-robin to long/short
pairs. Training/validation/test cover every grid value; BC seed data is balanced
at20k samples per value. DAgger's seven rollout pairs cycle through the grid (not
exactly balanced); GAIL's20 pairs cover each value four times, matching the expert
and learner each iteration. Nominal controls replace all training slips with .1.
Slip is constant per segment, including recovery. Training data is NOT Gaussian.

The evaluation-only parent Gaussian has mean5.0125/std2.5, truncated to[.025,10]
by rejection sampling. One value per round stays fixed throughout all models'
deployments. Three rounds use distinct task seeds; all models share the round's
slip/seed. No spatial noise or contact resampling. Seed42 plan draws are
5.774292699386079, 6.790566449482138 and 3.413121168125631. These lie between sparse
training grid points, so this is a held-out-surface generalization study, not an
evaluation drawn from the training surface distribution. Offline fidelity uses
independent trajectories on the fixed grid, not the Gaussian test surfaces.
The plan saves assignments, draws, seeds and budgets; Gaussian parameter changes
do not change training assignments. Three rounds do not measure training-seed
variability. Results include explicit slip/seed rows and across-round aggregates.

Update the parameters after fixed results by creating a NEW plan path:

```bash
./run_variable_training.sh plan --plan experiments/variable_slip_training/grid_eval_alternative/plan.json \
  --low 0.025 --high 10 --mean 5.0125 --std 2.5 --seed 42
```

Supply `--plan` on every later command. A changed plan cannot overwrite an existing
one. Different training seeds require separate paths. Whole runs, not neighboring
frames, are allocated to training, validation and test.

| Component | Default budget per arm |
|---|---|
| BC/DAgger shared oracle train pool | 10 × (5,000 long + 5,000 short) = 100,000 labels |
| Oracle validation/test | 10,000 labels each; 5 fixed-surface pairs x (1,000 long + 1,000 short) |
| BC continuation | 100 epochs, batch 256, LR 1e-4 |
| Mixed DAgger | seed-pool update + 7 iterations × 10,000 new labels; 50 epochs/update |
| DAgger execution mixing | beta=max(0, 1−0.2×iteration); every active state labelled |
| GAIL shared expert pool | 40,000 labels on exact planned learner surfaces/modes |
| Each GAIL continuation | 10 updates × 4 segments × 1,000 steps = 40,000 steps |
| Default final evaluation | 3 paired Gaussian draws; 3,000 steps/run; fixed suites opt-in |

All four agents warm-start from their root checkpoints. Outputs never overwrite
those originals. Supervised continuation starts a fresh Adam optimizer, then
mixed DAgger carries the selected checkpoint's matching optimizer across iterations.
Fixed whole-run validation selects each supervised checkpoint. LR 1e-4 is an
explicit continuation default, lower than the original scratch trainer's 1e-3.

GAIL copies the full existing actor-critic. Discriminator and optimizers start
fresh in both arms, then persist across iterations. It preserves original PPO,
entropy, BC-anchor, gradient-penalty and KL safeguards. It matches expert/learner
slip and target-mode weights each iteration, using frozen normalization statistics
from the complete GAIL expert pool. All segments of an update use one checkpoint.
GAE runs separately per segment; falls terminate, upright horizons bootstrap.
KL safeguards may produce different accepted PPO update counts between arms;
environment budgets are matched and actual update counts are recorded.

## Server ownership, artifacts and resume

Each collection/evaluation client gets a fresh verified local scene and owned
server. Offline optimization runs without a connected server. Server death aborts
the client; cleanup reaps owned processes and checks the installed scene is intact.
No physical surface configuration is changed in the system installation.

Within each plan directory:

- `data/<arm>/<group>/<segment>/`: immutable expert shards, job specs, manifests,
  server/worker logs, metrics and completion receipts.
- `policies/<label>/<arm>/iteration_NNN/segments/`: learner rollouts.
- `policies/<label>/<arm>/iteration_NNN/update/`: versioned policy, optimizer,
  metrics and (GAIL) discriminator/merged batch files.
- `policies/<label>/<arm>/final.json`: checked pointer to the final model.
- `evaluation/<round>/<label>/`: oracle and continued-model results (frozen only when explicitly requested).
- `report.json`, `report.md`: comparison outputs.

Rerun the same command to resume. Each committed job must match plan, sources,
initializers, environment and dependency hashes, with every output checksum intact.
Data shards are shape/finite checked. Incomplete attempts are moved to
`*_failed_<id>` and rerun from the preceding committed version. Aggregates list
each shard once. Updates are committed only after all checkpoint components are
written, so retries do not double-apply updates. Resume is at segment/update
boundaries, not mid-epoch. Updates seed their RNG explicitly, making retries start
from the same committed input and seed; GPU numerical determinism is not promised.

Metrics distinguish active steps, received startup/recovery cycles, oracle queries,
oracle control steps, falls and targets. Demonstrations shared across BC/DAgger or
GAIL variants are physically collected once per arm; inherited supervision from
original checkpoints remains an additional historical cost. Failed-attempt logs
are retained but not counted as successful training data.

## Validation status

Evaluation-only resume compatibility: scheduler/report, receipt helper and test
edits do not invalidate identical rollouts. Exact audited legacy metadata hashes
are supported without rewriting original receipts. Actual evaluator/server,
checkpoint, plan, environment and output hashes remain protected. Training resume
stays strict. Evaluation-semantic changes must bump the protocol identifier.

Offline tests include real small CPU supervised/GAIL updates, warm-start weights,
Gaussian reproducibility/bounds, disjoint splits, segment GAE, hash-based resume,
and interrupted-attempt preservation:

```bash
python -m unittest variable_training.test_pipeline -v
```

Seventeen tests pass across this package and the existing isolated evaluation tests.
Live training was not launched during implementation. Use the quick plan once the
running fixed-slip sweep finishes, then review fixed/baseline viability before
the full research plan. This package does not implement explicit slip-conditioned
policies, actuator impairments or Unitree transfer.
