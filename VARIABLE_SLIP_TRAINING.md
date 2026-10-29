# Variable-slip training pipelines

LATEST 2026-10-09: the user selected FIXED-GRID TRAINING and Gaussian evaluation
only. Authoritative executable plan:
experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json.
Train on .025/.1/.2/1/10, with optional matched nominal controls. Evaluate trained
models and oracle on three paired constant-slip rounds drawn from truncated
Gaussian(mean5.0125,std2.5,bounds .025..10). Default evaluation excludes original
frozen policies. See variable_training/README.md and latest PIPELINES.md section;
earlier Gaussian-training specifications below are historical and superseded.

2026-10-09 correction: all executable domain pipelines now use the validated
single-value field setter. Default plan is
experiments/variable_slip_training/gaussian_scalar_v2_seed42/plan.json. Re-establish
corrected frozen baselines before nominal/varied continuation. Legacy results are
preserved but do not validate varied effective-slip learning. Current commands and
version isolation are described in PIPELINES.md and variable_training/README.md.

Status: implemented in the independent `variable_training/` package. See
[variable_training/README.md](variable_training/README.md) for authoritative
commands, defaults, scope and differences from the original design below.
Scope is BC, mixed DAgger, GAIL-BC and GAIL-DAgger, with unchanged FC Portugal as
teacher/baseline. Original DAgger and GAIL-random are excluded. Fixed-slip evaluation
code remains untouched. Live training has not been run; offline tests pass.

## Study and provisional distribution

First evaluate frozen policies, then teach the existing policies to walk over a
distribution of surfaces with slip hidden. The subsequent context-aware comparison
adds slip to observations. FC Portugal remains the oracle for every algorithm;
BC/DAgger initialization does not make those policies GAIL's expert.

Provisional distribution v1:

| Parameter | Proposed value |
|---|---:|
| Lower bound | 0.025 |
| Upper bound | 0.200 |
| Parent Gaussian mean | 0.100 |
| Parent Gaussian standard deviation | 0.035 |
| Parent variance | 0.001225 |
| Slip changes | Once per collection/rollout segment |
| Evaluation length | 3,000 active steps per run |

Only the bounds come from the current fixed experiment. The mean and standard
deviation are provisional design choices, not estimated findings. Draw from the
Gaussian conditional on falling inside the bounds, using rejection sampling or
an established truncated-normal implementation; do not clip values to endpoints.
Because the interval is asymmetric around 0.100, the truncated distribution's
mean/variance differ from the parent parameters. Store both theoretical truncated
moments and empirical moments of the realized draws in the plan.

Generate and save all draws before collection with a distribution version, RNG
algorithm/seed, task seeds, training seed, segment IDs and planned step counts.
Use separate RNG streams for slip, targets, action sampling and optimization.
Pair the exact evaluation draws across all policies and both baselines. Training,
validation and final evaluation use disjoint runs and draw seeds. Do not choose
checkpoints or Gaussian parameters using final held-out evaluation results.

Revise bounds/mean/std after the fixed results and FC Portugal viability review.
A revised distribution gets a new plan hash and experiment directory; never
silently change parameters in an active or resumed experiment. The current sampled
evaluation code is uniform: Gaussian plan generation AND validation must be
implemented before its next use. A five-draw evaluation remains a pilot.

## Server and job architecture

Reuse the lifecycle design of `run_varied_slip.py`, extracting a shared server
manager only after the active evaluation finishes. Do not wrap a whole training
loop in one simulator process. Each long collector, short collector, DAgger
segment and GAIL segment gets a fresh server with a workspace-local scene.

For every segment:

1. Resolve its immutable plan entry and verify input hashes and checkpoint.
2. Reject occupied ports; acquire the shared simulator lock. Start the owned
   process using `--script-path`, synchronous mode and disabled real-time mode.
3. Verify local slip, server-side scene hash/import marker, and startup hashes.
   Save the manifest. These checks are not independent ODE parameter readback.
4. Start exactly one collector/rollout client. A watchdog aborts it if the server
   exits. Keep startup/fall recovery outside the active sample count and report
   their simulator steps separately. Hold slip constant through fall recovery.
5. Validate outputs and stop/reap the owned processes. Optimize with no simulator
   connected; open a new server for the next segment.
6. Atomically commit the segment receipt with hashes and sample counts.

Keep simulator seed 0 initially, matching the new frozen-policy protocol, and
record it separately from policy/target seeds. A later simulator-seed experiment
must be declared. Keep own-half targets, 63-D observations and 16-D Walk latent
actions. Slip and segment metadata must not enter model input during this study.

## Storage and transaction contract

Proposed location (created by implementation, not by this document):

```text
experiments/variable_slip_training/<distribution_id>/
  plan.json
  data/<training_seed>/<split>/<segment_id>/
    scene/  server.log  collection.log  manifest.json  data.npz  completed.json
  policies/<variant>/<training_seed>/<varied|nominal_control>/
    initialization.json
    iterations/<iteration>/
      segments/<segment_id>/
      dataset_manifest.json
      checkpoint.pth  optimizer.pth  metrics.json  completed.json
  evaluation/<suite>/<variant>/<seed>/
  reports/
```

Never append directly to a shared dataset during collection. Write immutable
shards; build aggregates from a sorted, unique list of committed shard IDs.
Preserve each shard's observation/action arrays in the format expected by
`bc_data.load_dataset`, with metadata for slip, mode (long/short), source,
episode boundaries, oracle counts and initialization. Reject NaNs, wrong shapes,
truncated arrays, mismatched slips, wrong policy versions and duplicated shards.

Resume requires matching plan, source code, model, data and environment hashes.
File existence alone is insufficient. Archive incomplete attempts, retry the same
planned segment and preserve failed-attempt costs. Commit model, optimizer,
discriminator where applicable, RNG state and iteration receipt as one versioned
transaction. A restart must not apply an update twice or silently skip one.
Changed completed inputs require a new experiment, not automatic retraining in
place. Stop on unrecoverable errors; user reruns the same command to resume.

## BC: first implementation pilot

Start from a copy of `bc_walk_policy.pth` and compare two continuation arms:
varied slip versus additional nominal-only training at 0.100. Both start from the
same checkpoint and get equal new expert-label and optimizer budgets.

Proposed collection budget: 10 pairs of 5,000-step long runs and 5,000-step short
turns, giving 100,000 new labels. Each pair shares one slip draw, but uses separate
servers and task seeds, so terrain is not confounded with movement mode. Use the
same task seeds, lengths and server resets in the nominal control.

Collect validation and test demonstrations as separate whole runs with separate
draws, explicitly budgeted in addition to training data. Do not randomly split
adjacent frames into train/validation/test. This requires extending the current
trainer's frame-level validation interface. Keep validation membership fixed.

Add an explicit warm-start checkpoint argument to `train_policy`; currently it
constructs a fresh BCPolicy. Initialize a fresh optimizer for both continuation
arms unless matched optimizer histories are available. Keep the network unchanged.
Pilot the current batch size 256 and a declared update budget derived from the
current 100-epoch setting; match actual gradient updates and processed samples
between arms. Select learning rate and checkpoint only using validation. If old
nominal demonstrations are replayed, predeclare their fraction identically in
both arms and report repeated exposures separately from new labels.

Check oracle progress/falls across sampled surfaces before accepting the teacher
distribution. Do not silently discard difficult trajectories: flag failures and
revise the distribution as a new version if needed. Report fall/reset segments.

## DAgger: retain mixed long/short supervision

Continue only `dagger_mixed_policy.pth`. Original DAgger is outside the requested
training scope. Pair each varied arm with its
own nominal continuation from the exact same checkpoint.

Use a balanced 100,000-label long/short seed dataset, then seven iterations of
10,000 new learner-state labels each, following the mixed-seed pipeline's budget.
The BC demonstration pool can supply the seed dataset; account for shared reuse
and inherited costs explicitly. For a same-training-seed paired experiment,
nominal and varied arms must have equivalent seed-dataset sizes and allocations.

Split each 10,000-step iteration into two 5,000-step segments (long and short),
sharing an iteration's slip draw. Add explicit target-mode selection to DAgger
rollouts; do not assume existing target sampling already implements both modes.
Keep the learner checkpoint fixed during all segments of an iteration. Query FC
Portugal at every visited active state, store the oracle label even when the
learner acts, and record which controller supplied each executed action.

Preserve `default_beta(iteration)` from `rollout_dagger.py` rather than inventing
a new mixing schedule. Beta controls the executed controller, not whether labels
are collected. Preserve uniform aggregation without arbitrary reweighting: with
100,000 seed labels and 10,000 labels per iteration, the seed share is
100000 / (100000 + 10000*i), ending at 58.8% after iteration 7. This matches mixed
DAgger dilution; it is not the original 20,000-seed variant's dilution.

Warm-start each update from the preceding policy. The current train_dagger calls
train_policy, which currently starts from scratch; this is an explicit required
change for continuation. Keep the old retraining behavior available for historical
reproduction. Pair update counts, beta, long/short allocation and retained data
between varied and nominal arms. Exclude dedicated validation/test data from the
aggregate. Save each iteration's oracle count M, environment steps N and dataset
composition by source, slip and mode.

## GAIL: paired expert and learner surfaces

Continue the existing `gail_bc` and `gail_dagger` checkpoints as
separate variants, each with a matched nominal-only continuation. A fresh GAIL
initialized from newly adapted BC/DAgger is a separate
experiment and must not be conflated with continuation.

Copy each actor-critic into its run directory. For the initial adaptation study,
reset the discriminator and all optimizer states with paired RNG seeds in both
arms, and record that choice. Do not import a nominal discriminator into only
one arm. Record inherited initialization costs and all static FC Portugal data.

Proposed pilot: retain 10 iterations and 4,000 active learner steps per iteration,
but collect four 1,000-step segments as two paired long/short surface draws before one update.
All segments in an iteration use exactly the same actor-critic checkpoint.
Build the fixed expert demonstration pool in advance over these planned surfaces
with balanced long/short allocation. Match expert and learner sampling weights by
slip (exact draw IDs where possible); balancing by stored frame counts alone can
confound the discriminator if rollout lengths or resets differ. Slip stays metadata
only for both discriminator and policy in the hidden-slip study.

Preserve rollout obs/actions/logprobs/values/dones and a bootstrap value per segment,
plus policy hash, episode IDs, slip and segment ID. Compute GAE independently for
each segment BEFORE concatenating PPO training arrays. A fall is terminal; an
otherwise upright segment horizon is a truncation and bootstraps from its final
state. Never propagate advantages from one server or surface into the next. The
current trainer accepts one bootstrap scalar, so multi-segment training requires
an explicit batch-format and GAE change, not just concatenating NPZ arrays.

Initially retain the stabilization settings in run_gail.sh: PPO epochs 3,
discriminator epochs 1, batch size 256, actor LR 3e-6, critic LR 3e-4,
discriminator LR 1e-4, entropy 0.001, BC coefficient 0.05, gradient penalty 10,
target KL 0.02 and rollback KL 0.05. Version any tuning. Log accepted KL, rollback,
discriminator statistics, reward, entropy, falls and progress. Update and checkpoint
actor/critic/discriminator/optimizers atomically. Never reuse a rollout after the
actor changes. Report static labels and inherited labels even though GAIL makes
no new real-time oracle training queries during learner rollouts.

## Evaluation and comparisons

Use the isolated server lifecycle for every model, control and baseline. Compare
the original frozen checkpoint, nominal continuation and varied continuation for
each lineage. Evaluate nominal 0.100, fixed endpoints and held-out Gaussian draws;
collect the unchanged oracle baseline on the same surfaces and task seeds. Random
is excluded by the current scope, so report oracle ratios rather than Scorenorm. Evaluate final
policies deterministically using the existing evaluation action-selection rule.

Report raw progress, same-condition oracle return ratio, retention relative to each
adapted policy's nominal performance, and change relative to its original nominal
checkpoint. Include falls, targets, velocity, tracking error, joint occupancy
distance and jerk alongside held-out oracle action error. Preserve the distinction
between 16-D latent action MSE and 22-joint kinematics. Zero falls without progress
does not establish useful walking.

Pair training budgets within each algorithm; BC labels, DAgger real-time queries
and GAIL PPO steps are not interchangeable costs. Report wall time, simulated steps
including recovery/startup, active steps, unique labels, optimizer updates and
inherited costs separately. Use several independent training seeds when feasible;
deployment seeds alone cannot establish training reproducibility. Choose final
sample counts before inspecting held-out outcomes.

## Implementation order and proposed interface

1. Finish/analyse frozen fixed-slip evaluation. Freeze Gaussian plan v1 or revise it.
2. Implement Gaussian sampled frozen evaluation and review oracle viability.
3. Extract/test server management and immutable collection/receipt support.
4. Implement BC checkpoint loading, whole-run splits and paired continuation pilot.
5. Add DAgger target modes, aggregation transactions and continuation.
6. Add GAIL multi-segment GAE, matched expert selection and atomic updates.
7. Implement the separate explicit-context comparison: normalize slip using fixed
   training-plan constants and append it as observation 64. Match data/budgets;
   specify weight migration and discriminator context consistently. Decision
   Adapter and Unitree transfer remain later comparisons.

The earlier proposed interface below is superseded by the runnable commands in
`variable_training/README.md` (`dagger_mixed`, `gail_bc`, `gail_dagger` are explicit
stages; use `all` to train all four). Do not run these historical placeholders:

```text
run_variable_training.sh plan --low 0.025 --high 0.200 --mean 0.100 --std 0.035
run_variable_training.sh collect --plan <plan.json>
run_variable_training.sh bc --plan <plan.json> --arm both
run_variable_training.sh dagger --plan <plan.json> --arm both
run_variable_training.sh gail --plan <plan.json> --arm both
run_variable_training.sh evaluate --plan <plan.json>
run_variable_training.sh report --plan <plan.json>
```

Before live training, tests must cover truncated bounds/reproducibility, disjoint
splits, paired nominal budgets, server death and cleanup, interrupted shard commits,
exactly-once aggregation/update resume, BC warm-start, DAgger oracle accounting,
and independent GAE bootstraps at both falls and server truncations. User-run smoke
jobs then verify oracle viability and short training/evaluation cycles.
