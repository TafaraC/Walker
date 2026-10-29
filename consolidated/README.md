# Consolidated imitation study

Two public launchers wrap the existing learning and evaluation components:
`run_training.sh` and `run_evaluation_all.sh`. Both own fresh synchronous servers,
verify generated scenes, preserve failed attempts and resume verified jobs. Do
not start SimSpark manually. Existing checkpoints, experiments and launchers are
preserved. This is a new experiment generation, not continuation of old receipts.

## Commands

From Walker with the `fcportugal` environment activated:

```bash
# Offline plan / budget review; no simulator or training:
./run_training.sh plan
./run_evaluation_all.sh plan

# Full training, original first and then varied hidden/input models:
./run_training.sh run

# Full evaluation, including reports:
./run_evaluation_all.sh run

# Offline report regeneration:
./run_training.sh report
./run_evaluation_all.sh report
```

Launchers default to the installed fcportugal interpreter; override `PYTHON_BIN`
if needed. Ports 3100/3200 must be free. A shared lock prevents simultaneous
managed jobs; no server or unrelated process is killed. Runs continue without
interactive checkpoints. Configuration still requires agent synchronization and
disabled server real-time mode.

Smaller stages:

```bash
./run_training.sh run --domain original
./run_evaluation_all.sh run --suite original --cohort original
./run_training.sh run --domain varied
./run_evaluation_all.sh run --suite varied
./run_evaluation_all.sh report --suite varied
```

Each evaluation selection has its own output/schedule. Repeating the *same*
command resumes it. Results in separate selections are not automatically reused;
use the full default command consistently if you want a single resumable sweep.
`--cohort original --suite varied` tests frozen new original policies on the varied
schedule; it requires the varied expert pool for random bounds and held-out MSE.
`--cohort varied --suite varied` evaluates just new varied policies plus baselines.

Training `--algorithm bc|dagger_mixed|gail_bc|gail_dagger_mixed` selects a branch;
required initializers are trained automatically. `--domain varied` automatically
builds missing original parents. FiLM and random-initialized GAIL are excluded.

Optional input ablations, matching the earlier context protocol:

```bash
./run_evaluation_all.sh run --suite varied --context-modes correct constant wrong
./run_evaluation_all.sh report --suite varied --context-modes correct constant wrong
```

For an engineering smoke with tiny budgets, **use a separate output**:

```bash
./run_training.sh run --quick --output experiments/consolidated/smoke_v1
./run_evaluation_all.sh run --quick --output experiments/consolidated/smoke_v1
```

Quick mode is not research evidence. It has one varied training seed, one learner
iteration, short collection segments and 32-step evaluations. The agent has run
offline tests only; the live smoke is user-run.

## What “matched steps” means

The agreed primary accounting unit is **active training examples / learner
transitions**, not optimizer steps. Repeated epochs reuse examples. Validation and
test examples are extra, distinct whole collection segments, not withheld from the
170,000 training budget.

| Technique, per policy | Expert training data | Learner rollout steps | Initializer |
| --- | ---: | ---: | --- |
| BC | 170,000 static examples | 0 | Random in original; new original BC in varied |
| Mixed DAgger | 100,000 static seed + 70,000 learner-state labels | 70,000 | Random in original; new original mixed DAgger in varied |
| GAIL-BC | Shared 170,000-example static expert pool | 70,000 PPO transitions | Corresponding new BC |
| GAIL-mixed-DAgger | Same 170,000-example static expert pool | 70,000 PPO transitions | Corresponding new mixed DAgger |

Each environment also has 10,000 validation and 10,000 test examples. Data are
shared between branches where appropriate; inherited costs and shared datasets
must not be interpreted as fresh oracle queries per consumer. GAIL has zero
additional active labels in its PPO rollouts but uses expert demonstrations and
inherits its initializer's training cost. Total end-to-end costs are consequently
not equal. This distinction is explicit in each policy's `final.json` ancestry.

BC runs 100 epochs; DAgger runs 50 epochs initially and after each of seven
10,000-step learner batches. Thus optimizer steps/sample exposures are not matched.
Both use batch 256 and Adam 1e-4. DAgger retains beta `max(0,1-.2*iteration)`:
.8/.6/.4/.2/0/0/0, with an oracle label on every active learner state.
The seed and extra BC data are balanced long runs / short turns. Each learner
iteration also balances both modes over all five surfaces (original repeats the
same environment). Per-stage training metrics record updates and exposures.

GAIL uses seven 10,000-step PPO batches, three PPO epochs and one discriminator
epoch, reusing the existing stabilized update (small actor LR, gradient penalty,
BC anchor, KL checks/rollback). Each batch has ten independent segments. Segment
offsets/bootstrap values preserve GAE boundaries; falls terminate the preceding
transition. Every batch and the shared static expert pool have equal surface/mode
weights. Global normalization uses only the training expert pool.

## Environments, architectures and lineage

Original training uses a byte-for-byte copy of the installed scene. On this system
it contains `(setContactSlip 0.1 0.1)`, the historical pathway. This is deliberately
distinct from corrected scalar slip .1. Scene/config hashes are immutable inputs.
Fresh owned servers are used throughout, so archived original evaluation runs
are not pooled with these new deployments.

Varied training uses scalar slips .025/.1/.2/1/10, evenly weighted. Two variants:

- `hidden`: 63 observed features, no supplied slip.
- `input`: append `log1p(actual_scalar_slip)/log(11)` for 64 inputs; zero-initialize
  the extra first-layer column when expanding the original policy.

Both have two 256-unit ReLU layers and 16 latent Walk outputs. No FiLM. For GAIL,
actor, critic and discriminator all use the selected 63/64-input contract; expert
and learner data receive the same context encoding. The expert remains FC Portugal.
Predictions used for evaluation are deterministic actor means.

Original BC and mixed DAgger start from identical seed-42 random architecture
initialization. Original GAIL branches initialize from those new final policies.
Varied BC/mixed DAgger continue their corresponding new original checkpoint;
varied GAIL initializes from the corresponding **varied**, same-context, same-seed
BC/mixed DAgger final checkpoint. Consequently there is no old original-DAgger
GAIL checkpoint silently substituted for the newly requested mixed lineage.

There are four original models and 24 varied models: four techniques × hidden/input
× training seeds 42/43/44. Varied seeds share demonstrations and original parents;
they vary minibatches and learner rollout seeds. They are separate deployed policies.

## Evaluation protocol

Original-scene evaluation: oracle, uniform-random and four new original policies,
five task seeds 101/202/303/404/505, 3,000 active steps: **30 runs**.

Varied evaluation reuses the exact latest context schedule from
`context_training.evaluate.rounds(42, 'all')`: 15 grouped Gaussian draws, 15
previous confirmation draws and three scalar-nominal cases. Parent Gaussian
means .025/.1/.2/1/10, standard deviation half the mean, truncated to (0,10].
One slip stays fixed throughout a deployment; all controllers share the round's
slip/task seed and each has a fresh server. These are previously used test
settings, not newly blind confirmation evidence.

Each varied round evaluates oracle, random, four frozen new original policies and
24 new varied policies: **33 × 30 = 990 runs**, or **1,020** with original tests.
Default input mode is correct context. Constant/wrong modes supply .1 or the prior
extreme-context rule without changing physics; they add deployments.

Target sampling, 20-ms active timestep, fall recovery, deterministic actors and
trace quantities follow the original/context evaluators. Random actions use the
training expert action minima/maxima and an independent action RNG, preserving
the original random-policy rule without perturbing target RNG draws.

Reports include:

- Target-progress return, same-round random/oracle normalized score and oracle ratio.
- Forward velocity, vector tracking error, falls per active minute and target count.
- Held-out latent action MSE on the shared test pool of the evaluation domain,
  plus on-policy oracle MSE on visited states.
- Expert-standardized joint-occupancy sliced Wasserstein, torso RMS and smoothed
  integrated joint-angle jerk, excluding reset boundaries.
- Per-replica descriptive CIs, corrected-nominal retention when that suite exists,
  and paired input-minus-hidden / ablation-minus-correct differences.

Recovery/startup cycles are excluded from active budgets but logged by collectors.
Normalization is null for effectively zero oracle-minus-random denominators.
Sliced Wasserstein is not exact 22-D optimal transport; angular jerk is not torque
or energy. Replicas retain separate aggregate rows rather than treating all runs
as independent training repeats. Original-scene results and corrected nominal
results remain separate.

## Artifacts and recovery

Default root: `experiments/consolidated/v1/` (override `--output`).

```text
plan.json / runtime_identity.json
data/{original,varied}/{seed,extra,validation,test}/<segment>/
policies/<domain>/<hidden,input>/seed_<N>/<algorithm>/
  initialization/
  iteration_<N>/{segments/,update/}
  final.json
training_summary.json
evaluation/<suite>_<cohort>_<modes>/
  schedule.json / runs/<round>/<controller>/
  comparison.json / comparison.csv / comparison.md
```

Final models stay inside their policy directories. Plain tensor checkpoints are
paired with explicit architecture, lineage and resource manifests; do not load
64-input checkpoints through the legacy evaluator. This comparison JSON has its
own schema; it is not input for the legacy `plot_evaluation.py`.

Every completed job records hashes of inputs, outputs, source and environment.
Runtime identity also records Python/NumPy/PyTorch versions and selected CPU/CUDA
device. Final policy checks traverse completed input dependencies before resume.
Resume rejects changed completed artifacts. Incomplete attempts are renamed with
a unique `_failed_` suffix before retry. Default one retry for live jobs; no
mid-epoch training resume. Use identical flags to resume; changed budgets or
source contracts require a fresh output generation. Do not edit old receipts.

Offline validation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest consolidated.test_pipeline -v
```

The tests exercise exact balanced budgets, latest evaluation draws, context
initialization, mixed-DAgger GAIL ancestry, immutable receipts, original/scalar
scene preparation and a synthetic 64-input BC/PPO update with segmented GAE.
