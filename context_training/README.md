# Context-conditioned BC pilot

## Follow-ups: repeatability and appended FiLM

The first429-run comparison is complete. FiLM now has an isolated pipeline:
[FiLM training/evaluation guide](../film_training/README.md). Per user clarification,
FiLM evaluation appends only new FiLM runs on the exact prior slips/task seeds;
saved hidden/input/oracle evaluations are reused, not repeated.

Repeatability is a **separate optional diagnostic**, not a requirement built into
FiLM evaluation or a replacement for those saved controls:

```bash
./run_bc_repeatability.sh plan
./run_bc_repeatability.sh evaluate
# Automatically reported on completion, or regenerate offline:
./run_bc_repeatability.sh report
```

Defaults: five identical repeats per case/controller, seed42 hidden/input pair
plus oracle,3,000 active steps each:3 cases×3 controllers×5 repeats=45 deployments.
Cases retain the exact prior pairs: nominal.1/task4296000; low
.0002849946313024654/task4294000; high7.417564630443256/task4292014. The selected
checkpoint, task seed, physical slip and supplied context are unchanged across
repeats. Input uses correct context. Cases are interleaved between repeat blocks.
`--training-seed` selects a completed matched pair; it defaults to42, not a
best-performing seed. `--repeats` must be at least2. Changed settings require
a new `--output`.

The study deliberately preserves the current worker/startup behaviour. It does
not yet fix startup variation, beam the robot differently or change synchronization.
Fresh owned servers, shared lock, strict receipts and bounded retries are reused.
No manually started server is needed. Output defaults to bc_input_v1/repeatability_v1/
with plan.json, run traces/receipts, repeatability.md and repeatability.json.

Reports quantify progress mean/SD/range, falls, mean pairwise absolute progress
difference, and first-active-observation, planar position, joint-angle and target
differences across repeats. The raw trace also records initial actions, oracle
actions, head height and torso angles. These are first active control states,
not an assertion of identical simulator connection states. Mixed-unit observation
maxima are diagnostics, not a physical error unit. Pairwise differences share
runs and are not independent statistical samples. Do not remove outliers or
interpret correlation with startup variation as proof of a root cause.

Status: BC training and isolated live evaluation implemented; live evaluation
awaits user execution. FiLM and conditioned DAgger/GAIL remain planned. Do not
pass these checkpoints to the legacy 63-input evaluator.

## Commands

From the repository root, audit the existing demonstrations (no simulator or
training, no experiment writes):

```bash
./run_context_training.sh audit
```

When ready to train, run both matched arms and all three seeds:

```bash
./run_context_training.sh train
# Optional GPU version; choose this before the first training run:
./run_context_training.sh train --device cuda
```

These are alternatives, not successive commands for the same experiment. Device
and all other settings are immutable once the output plan exists. The launcher
uses the fcportugal interpreter, overridable with `PYTHON_BIN`. No server is
needed for either stage. Training runs to completion without internal prompts.

Defaults: architectures `hidden input`, seeds 42/43/44, 100 epochs, batch 256,
Adam learning rate 1e-4, CPU. Use `--output` for a new configuration; for example:

```bash
./run_context_training.sh train --epochs 1 --seeds 42 \
  --output experiments/context_conditioning/bc_input_engineering_check
```

This short run is an engineering test, not the research comparison. `--help`
lists `--data-plan`, `--initializer`, `--architectures`, and other overrides.
The default runs both architectures to avoid comparing extra context training
against an old checkpoint without the same update budget.

## Data and model contracts

Read-only source: `experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json`.
Uses its varied-arm oracle shards: 100k train, 10k validation, 10k test, with
whole trajectories already split and long/short modes retained. Audits verify
receipt hashes, plan/spec association, oracle-only commands, dimensions,
finite arrays, scene hashes, successful server import, scalar setter, and slip
agreement. Duplicate trajectories across splits are rejected. Historical source
hashes are preserved rather than requiring the old collector to equal today's code.

Each observation stays 63-D in the source data. Slip is attached in memory from
the verified scene metadata. Input conditioning appends
`log1p(slip)/log(11)`, producing 64-D inputs. The fixed transform is independent
of evaluation outcomes, accepts finite values in (0,10], and never clips values.
The hidden control discards the same context. Both use the existing 256/256 ReLU
backbone and output 16 latent actions, not joint commands. FiLM is deferred.

Both arms warm-start `bc_walk_policy.pth`, matching the original grid-training
starting point, not the already grid-trained final model. The input arm copies
the old first-layer weights and adds a zero slip column; all parameters remain
trainable. Paired seeds use identical shuffled batches. The input arm adds only
256 parameters. Selection uses validation MSE; the test split is scored only
after selection. Tests confirm initialization agreement within numerical tolerance.

## What the training seeds mean

The original experiments used one saved model per technique. This context BC
pilot instead trains **three checkpoints per architecture**, using seeds **42,
43 and 44**. There are still only two BC architectures in this comparison:
context hidden and input conditioned. The six checkpoints are separate policies,
not six techniques, an ensemble, or six policies acting together. Each checkpoint
controls the robot independently when evaluated.

### What changes between seeds?

The main controlled difference is **the order in which training demonstrations
are presented to the optimizer**. `train_one` seeds Python, NumPy and PyTorch
(including CUDA when selected). Its shuffled DataLoader uses a separate
`torch.Generator().manual_seed(seed)`, so seeds 42/43/44 yield different sequences
of minibatches over the epochs.

Each gradient update changes the network before the next batch is processed.
Consequently, changing batch order can produce different optimization paths,
validation-selected epochs and final weights even with the same demonstrations.
This is why evaluating multiple trained checkpoints can reveal whether an
apparent benefit depends on a particular fine-tuning run.

The current policy is a feed-forward ReLU network without dropout. Its random
constructor weights are overwritten by the common pretrained state, so different
training seeds do **not** give independently initialized backbones. Input models
all start with the same added slip-input weights: zero. The pipeline does not
enforce deterministic CUDA algorithms; a seed alone is not a guarantee of
bit-for-bit identical results across repeated executions, devices or versions.

### What stays the same?

For all three training seeds, the following remain fixed:

- Original `bc_walk_policy.pth` initializer and inherited pretrained weights.
- The same 100,000 expert training samples and existing 10,000-sample validation
  and 10,000-sample test splits; no new split is sampled per training seed.
- Demonstration surfaces 0.025/0.1/0.2/1/10 and long-run/short-turn composition.
- Context values associated with each trajectory and the fixed slip encoding.
- Architecture within each arm, optimizer settings, batch size and epoch budget.

The training seeds do **not** draw new surfaces, change the Gaussian, recollect
oracle demonstrations or change the physical environment. This BC stage is
offline. Both architectures receive the same existing observations/action labels;
only the input-conditioned arm is given the additional slip feature.

### Why pair seeds between architectures?

Hidden BC seed42 and input-conditioned BC seed42 receive the same shuffled
minibatch indices in the same order, with equal training budgets. The same applies
to seeds43 and44. The dedicated DataLoader generator preserves this pairing even
though constructing a 64-input layer consumes different random numbers from
constructing a 63-input layer. Their predictions and gradient updates can still
differ because their architectures differ.

Compare input42 with hidden42, input43 with hidden43, and input44 with hidden44.
These repetitions measure **fine-tuning variability conditional on a shared
initializer and dataset**. They are not independent pretraining runs or independent
demonstration datasets, and cannot establish variability of the full training
procedure from scratch.

### Training seeds versus evaluation seeds

| Seed role | What it controls here | What it does not control |
|---|---|---|
| BC training seed (42/43/44) | Minibatch ordering during offline fine-tuning | Training surface selection or new demonstrations |
| Evaluation schedule seed | Reproducible Gaussian slip draws and assigned task seeds | Trained checkpoint weights |
| Evaluation task seed | Random target sampling within a deployment | Physical slip within the run or training batch order |

Every controller within a round receives the same physical slip and task seed.
That pairs the task's random-number stream, not necessarily identical robot
trajectories or target-change timings: reaching targets and recovering from falls
can advance target sampling at different times. The server retains the existing
simulator seed setup; the evaluation task seed is not a new physics-engine seed.

### Why this increases evaluation runs

For the basic comparison, each round runs the oracle once, each of the three
hidden checkpoints once, and each of the three input checkpoints once:
**1 + 3 + 3 = 7 deployments per round**. Fifteen grouped rounds therefore require
**105 deployments**. These are not repeated identical evaluations of one model.

With all context ablations, each input checkpoint is evaluated with correct,
constant and wrong supplied context, while its weights and physical surface stay
fixed. That gives **1 + 3 + (3 × 3) = 13 deployments per round**. The full suite
contains 15 grouped, 15 confirmation and 3 nominal rounds, so **33 × 13 = 429**.
These are evaluation runs, not additional training runs or additional techniques.

A single matched checkpoint pair plus oracle would require 15 × 3 = 45 grouped
deployments. Choosing seed42 consistently in advance would be preferable to
selecting whichever seed scores best, but the current evaluator does not expose
a training-seed selector. This is an optional future reduction, not an implemented
command or a change to the current experiment. All six trained checkpoints remain
preserved for repeatability checks.

## Artifacts, resume and metrics

Default output: `experiments/context_conditioning/bc_input_v1/`:

```text
plan.json                      immutable data/model/source/settings identity
hidden/seed_42/                 repeated for each architecture and seed
  policy.pt                    metadata + model state, schema context_bc_v1
  optimizer.pt                 optimizer at selected validation epoch
  metrics.json                 training history and held-out fidelity
  completed.json               artifact checksums and plan identity
input/seed_42/...
summary.json                   completed offline runs; no locomotion claims
```

The plan records context encoding, data and receipt hashes, initializer hash,
source hashes, seeds/settings, device and NumPy/PyTorch versions. Completed jobs
skip only when their identity and artifact hashes match. Incomplete seed/arm
directories are archived on retry and restarted from the common initializer;
there is no mid-epoch resume. A per-output lock prevents simultaneous writers.
Input changes are checked again before writing each completion receipt.

Metrics include held-out element MSE and squared L2/state (16 times element MSE),
per-slip MSE, parameter count, updates, sample exposures, selected epoch and wall
time. Shared inherited demonstration counts are recorded separately from zero
new oracle queries/environment steps. These are latent-action errors, not 22-joint
command error; they do not establish stability or locomotion quality.

Python inference interface (no simulation):

```python
from context_training.models import load_checkpoint, predict
model, architecture = load_checkpoint(".../input/seed_42/policy.pt")
actions = predict(model, architecture, observations_63d, supplied_slip=0.1)
```

The explicit supplied-slip argument supports correct/constant/wrong-context
evaluation independently of physical slip. The isolated evaluator below uses
owned servers; old experiment identities are not modified for this architecture.

## Live BC evaluation

No manually started simulator should occupy ports 3100/3200. The runner owns a
fresh synchronous server for every deployment, verifies the local scalar scene,
and leaves installed settings unchanged. Commands are user-initiated and run
without internal approval prompts. No retraining or new oracle demonstrations
are required. All six training receipts/checkpoints are verified before evaluation.

First run a smoke evaluation (one paired surface/task round, all six checkpoints
plus oracle, still 3,000 active steps each):

```bash
./run_context_evaluation.sh evaluate --suite smoke
```

Then the complete planned BC comparison, including all context ablations:

```bash
./run_context_evaluation.sh evaluate --suite all --modes correct constant wrong
```

This is **429 deployments**: 33 rounds × 13 controllers. It includes 15 original
grouped draws, 15 new confirmation draws, and three nominal 0.1 rounds. Per round
there is one unchanged oracle, three hidden checkpoints, and three input
checkpoints under each of three supplied-context modes. Training seeds remain
42/43/44; each controller shares the round's task seed and physical slip.

For a smaller comparison without ablations (105 deployments):

```bash
./run_context_evaluation.sh evaluate --suite grouped
```

Other separate suites: `confirmation` (15 rounds), `nominal` (three rounds).
`--suite all` reports these suites separately, not as one pooled performance score.
Smoke results are engineering checks and are not automatically reused by other
suites. Physical slip stays fixed for each run, never per-contact randomized.

### Context modes and pairing

- `correct` (required, default): actual scalar scene slip, rounded exactly as the
  scene writer does, encoded using the checkpoint's fixed transform.
- `constant`: supply 0.1 regardless of the physical surface.
- `wrong`: supply 10 when physical slip is below 1; otherwise supply 0.025.
  This is a predeclared extreme mismatch test, not an estimate of sensor noise.
- Oracle and hidden controls receive no policy context; they run only once per
  round per checkpoint, not redundantly for every mode. At physical slip 0.1,
  correct and constant are intentionally identical inputs.

Original grouped draws/task seeds match the prior grouped suite. Confirmation
uses `SeedSequence([source_seed, group, 20261010])` with the same parent means,
std=0.5*mean and rejection bounds (0,10]; task seeds start at source_seed*100000+94000.
Nominal task seeds start at source_seed*100000+96000. The schedule is predeclared
in the saved evaluation plan, not selected based on new results. After inspection,
confirmation results must not be described as an untouched test for later tuning.

### Planning, reporting and resume

```bash
# Offline: validate checkpoints and save the immutable evaluation plan only.
./run_context_evaluation.sh plan --suite all --modes correct constant wrong
# Evaluation generates these reports automatically; regenerate offline if needed:
./run_context_evaluation.sh report --suite all --modes correct constant wrong
```

Default output is `<training-root>/evaluation_<suite>_v1/` with `plan.json`,
`runs/<round>/<controller>/`, and `report.md`, `report.csv`, `report.json`.
Each run saves job metadata, model hash, separate supplied/physical slip, server
logs, verified scene/manifest, result.json, trace.npz and hashed completion receipt.
Use `--training-root` for a different completed pilot. Use a new `--output` if
changing modes, steps, sources, checkpoints or settings within the same suite;
existing plans/results are never silently reinterpreted.

Rerun the identical evaluate command to resume verified completed deployments.
Default `--retries 1` allows one additional attempt for transient server/client
failures, with a fresh owned server and the incomplete directory archived. Invalid
data/model identities fail rather than being bypassed. Exhausted retries stop the
runner; after resolving the problem rerun the same command. No existing unrelated
server is killed. A shared simulator lock prevents overlapping experiments.

### Evaluation metrics and limits

Reports include progress, oracle return ratios (not random-normalized returns),
falls per active minute, targets, forward/planar speed, vector tracking error,
on-policy latent action MSE, smoothed joint-angle jerk and joint occupancy sliced
Wasserstein-1 against the oracle from the same round. Raw traces retain 63-D states,
16-D actions/oracle actions and 22 joint angles. Reset intervals are excluded from
active time and jerk derivatives. A fall from the final action is checked too;
older legacy evaluations did not perform this final check.

Sliced Wasserstein uses 256 projections, seed12345 and per-round oracle joint
standardization; it is not exact 22-D Wasserstein. Jerk uses a five-sample moving
average and third differences at 0.02 s; it is not a torque or energy measurement.
On-policy error must not be confused with the training report's held-out error.
JSON contains per-round paired progress/fall deltas: correct input versus hidden
at the same training seed, and each ablation versus that seed's correct input.
Tables retain each training seed and suite separately. No significance claim or
independent-pretraining assumption is made from these shared-initializer runs.

Offline orchestration tests (mock servers only):

```bash
python -m unittest context_training.test_evaluation
```

Plan and references: see the input-conditioning/FiLM follow-up in
[`EXPERIMENT_PHASES.md`](../EXPERIMENT_PHASES.md). Offline tests:

```bash
python -m unittest context_training.test_pipeline
```
