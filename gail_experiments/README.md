# Exploratory GAIL improvement study

This is separate from the completed paper and optional wide-slip oracle sweep.
The user selected **original environment first**. Goal: test whether a revised,
longer-budget adversarial update improves an initializer, not merely preserves it.
No improvement is guaranteed; no old checkpoint or report is overwritten.

## Run commands

The scripts own fresh synchronous servers. Do not start a server yourself or run
this concurrently with the optional stress sweep. They share the simulator lock.

```bash
# Inspect the full defaults; audit real datasets and initializers without simulation.
./run_gail_improvement.sh plan
./run_gail_improvement.sh audit

# Separate small live end-to-end check: all three initialization types, one seed.
./run_gail_improvement.sh run --quick

# Full training, periodic validation, confirmation evaluation and report.
./run_gail_improvement.sh run
```

Quick defaults to `experiments/gail_improvement/smoke_v1`, never the full directory.
It is only 200 rollout steps per actor, two updates, 100-step evaluations, one
training seed and one seed per evaluation split. It tests plumbing, not learning.
It reuses the full audited expert data and pretrained BC/DAgger, not newly fitted
tiny initializers. The full default output is
`experiments/gail_improvement/original_v1`.

Full defaults: **350,000 active rollout steps per actor**, 35 × 10,000-step
updates; initialization types random, BC and mixed DAgger; seeds42/43/44.
Thus the default means **nine actors and 3,150,000 training interaction steps**,
not 350,000 total. Each update has ten 1,000-step segments (five long/short pairs)
with fresh servers. Evaluation interactions and startup/recovery add extra cost.
Updates use CUDA when available; simulator clients run on CPU.

Separate stages and resume:

```bash
./run_gail_improvement.sh train
./run_gail_improvement.sh evaluate
./run_gail_improvement.sh report
```

`train` includes periodic validation. `evaluate` requires all checkpoints and
completes validation/confirmation. `report` verifies existing evaluations and
regenerates summaries; it does not launch a server or fit models. Repeating a
command skips hash-verified completed jobs. Interrupted jobs are archived by the
existing begin mechanism on retry, not counted as completed. Simulator failures
stop the command; there is no unbounded retry. No prompts occur within a run.

Use the same options for every command; incompatible saved plans are rejected.
For a single-seed pilot at the full budget:

```bash
./run_gail_improvement.sh run --seeds 42 --output experiments/gail_improvement/pilot_v1
```

For a controlled same-budget comparison with the old stabilized trainer:

```bash
./run_gail_improvement.sh run --recipes legacy adaptive --output experiments/gail_improvement/paired_recipes_v1
```

That doubles the default to18actors/6.3million training interactions. Both recipes
use identical initial exploration std0.2 and the same new evaluation protocol.
“Legacy” means the unchanged trainer, not byte-identical replication of the old
experiment (which used initial std≈0.135 and a smaller total budget). Comparing
the new recipe only against historical70k results confounds budget and algorithm.

`--steps`, `--rollout-steps`, `--eval-every`, `--seeds`, `--parent` and `--output`
are configurable. Budget must divide evenly into rollouts; rollout size divides
by ten. An optional `--domain varied` reuses the existing .025/.1/.2/1/10 grid
and same-domain hidden initializers, not untested large-slip values; use a new
output directory. The current requested experiment uses the default original
environment. No context-model architecture changes are introduced here.

## Why this might differ from the previous GAIL

The previous implementation became mechanically stable with a very small actor
learning rate, a persistent BC anchor, and strong discriminator regularization.
The new configuration tests plausible limitations of that recipe, not proven
causes. It deliberately changes several components; component ablations would
be required to attribute an effect to any one of them.

| Component | Adaptive recipe |
|---|---|
| Actor architecture | Same63→256→256→16 mean network and Gaussian exploration; gait/IK retained |
| Initial weights | Random, consolidated original BC, or consolidated original mixed DAgger |
| Auxiliary BC loss | None; expert data enter the discriminator only |
| Actor learning rate | Starts3e-5; adapts within1e-6..1e-4 |
| Policy constraint | Exact old-to-new diagonal-Gaussian KL over the full rollout; target.01, maximum.02 |
| Unsafe epoch | Restore actor and Adam state; retry with half LR, at most four attempts |
| Update budget | Up to four PPO epochs; independent four critic epochs; batch1024 |
| Discriminator | Logistic loss, up to two epochs, fresh-sample accuracy early stop above.85 on both classes |
| Regularizer | Zero-centered expert-input gradient penalty, coefficient1 (loss adds0.5×coefficient×squared norm) |
| Expert actions | Add Gaussian noise with the current actor's exploration std before discriminator training |
| Discriminator scaling | Fixed expert-pool mean/std; standard-deviation floor.05 |
| Reward | Positive `softplus(logit) = -log(1-D)` clipped at10; no task reward |

Noise matching addresses a potential shortcut: deterministic expert actions can
be distinguished from stochastic learner actions by noise alone. This changes
the discriminator target to an action-noise-convolved expert distribution; it is
not exact matching of deterministic expert actions. Held-out action MSE remains
against the original clean labels. Noise matching and R1 do not guarantee a
useful discriminator—accuracy, loss and reward variance are logged.

The positive reward avoids making early fall termination intrinsically attractive
solely because every surviving step incurs a negative reward. It can instead
introduce a survival preference; this is a reward-design trade-off, not an
unbiased universal fix. No explicit progress or fall reward is added. Correct
fall masks, segment-specific GAE/bootstrap and executed-action/log-probability
contracts are retained. Update reproduces saved log probabilities and rejects
an on-policy mismatch before training. Final KL is checked before writing files.

The learned action distribution remains unbounded Gaussian with a bounded std,
as in the existing interface. Samples are not silently clipped while retaining
incorrect log probabilities. Existing fall recovery remains in force; random
actors may still fail to discover walking.

## Data, initialization and resource accounting

Expert seed+extra shards (170k rows) and separate validation/test shards come
from `experiments/consolidated/v1`, audited through existing parent receipts.
No DAgger learner-state aggregate is used as expert occupancy. Original BC and
DAgger sources are the seed42 final checkpoints, shared across the three new
stochastic continuation seeds. These are not independent pretraining replicas.
The random actor receives no supervised warm start in the adaptive recipe.
The legacy trainer retains its BC anchor, including for random initialization.

Expert collection and BC/DAgger costs are inherited, not erased by reporting
zero online training oracle queries. Evaluation queries the oracle for fidelity
metrics; those are evaluation costs, not training supervision. Critic/discriminator
and actor optimizers persist between updates. New source/runtime/model identities
are pinned separately, leaving old experiment fingerprints unchanged.

## Evaluation and what counts as improvement

Iteration0 and every70ksteps (plus final) are evaluated on task seeds901/902/903,
3,000 active steps each. The **entire initializer actor** is measured, including
random initialization; GAIL rollout target/fall counts are not treated as
post-update deterministic evaluations.

Selection is predeclared: highest validation progress among checkpoints whose
total falls are no more than initializer falls+1; ties prefer the earlier
checkpoint. This is an operational screening rule, not a physical safety or
statistical guarantee. Iteration0 remains eligible. Validation action MSE is
reported but does not select checkpoints.

Initializer, selected checkpoint and final checkpoint are evaluated on separate
task seeds1901..1905 (shared across methods), together with fresh oracle and
uniform-random action controls. Duplicate selected/final/initializer checkpoints
are evaluated once and reused by role. At most15confirmation deployments per
actor, plus10shared reference runs. Original default periodic validation adds
162deployments (9actors×6checkpoints×3seeds). Confirmation data never select
checkpoints. The policy mean is used for deterministic evaluation, as previously.

Reports contain paired progress differences, falls and targets, forward velocity,
tracking error, expert/random-normalized return, test and on-policy action MSE,
joint occupancy SWD and jerk. New runs use the inherited random baseline's action
bounds from the first expert shard; that is held constant across recipes. A small
mean gain is descriptive—not proof of significance. GAIL does not directly
optimize exceeding the expert's task score, and better progress may accompany
worse imitation. Both selected and final checkpoints are reported; nothing is
automatically promoted over a paper model.

## Artifacts and diagnostics

- `plan.json`, `runtime_identity.json`: immutable settings/provenance.
- `policies/<recipe>/<initializer>/seed_N/iteration_NNN/`: source initialization,
  ten rollout segments, updates, checkpoints and resumable receipts.
- Each adaptive update saves discriminator loss/accuracies, R1 penalty, reward
  mean/std, exact KL, rejection counts, accepted epochs, current/next actor LR,
  exploration scale, actual mean-action shift, critic loss and expert-train MSE.
- `evaluation/validation/` and `evaluation/confirmation/`: separate run receipts,
  raw traces and results.
- `selection.json`: validation curves and selected iteration.
- `comparison.json/csv/md`: paired confirmation comparisons and caveats.

If mean-action shift is near zero, inspect exact KL, rejected epochs and LR.
If discriminator accuracies saturate and reward variance disappears, extra
budget alone may not help. If progress improves but action MSE deteriorates,
report the trade-off rather than label every metric improved.

## Method references and status

- [Orsini et al. (2021), What Matters for Adversarial Imitation Learning?](https://arxiv.org/abs/2106.00672)
  motivates examining reward and discriminator design rather than assuming one
  standard recipe is universally effective.
- [Mescheder, Geiger and Nowozin (2018), Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406)
  motivates zero-centered gradient regularization. Their GAN convergence analysis
  is not a guarantee for this PPO-controlled simulator experiment.
- [Ho and Ermon (2016)](https://arxiv.org/abs/1606.03476) and
  [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) remain the GAIL/PPO foundations.

This is an experimental variant, not a claimed new algorithm. CPU synthetic
updates, optimizer resume, exact-KL rejection and planning/selection tests are
provided. **No research simulator runs have been launched by the assistant.**
Run the separate quick batch before committing the larger budget.
