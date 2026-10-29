# FiLM BC and append-only evaluation

FiLM is the second context architecture after input concatenation. This package
is isolated so the completed hidden/input training and 429-run comparison retain
their original source identities. No existing model, plan or result is overwritten.

## What is trained

Only **three new FiLM checkpoints**, seeds42/43/44, by default. These are repeated
fine-tuning runs of one architecture, not an ensemble. Existing hidden/input BC
controls are reused, not trained or evaluated again for the FiLM comparison.

The 63 state features enter the original 256/256 ReLU backbone. After the second
ReLU, a separate linear branch maps one encoded slip value to 256 scale adjustments
and256 offsets:

`h_mod = (1 + delta_gamma(c)) * h + beta(c)`

The unchanged final layer maps to16 latent Walk actions. For batching, the model
accepts a packed64-value tensor; the extra value enters only the FiLM branch, not
the first state layer. Context encoding stays `log1p(slip)/log(11)` for (0,10].
All modulation weights and biases start at zero, making initial modulation exactly
identity. Backbone weights come from the original `bc_walk_policy.pth`. Both
backbone and conditioner train. Added parameter counts: input concatenation256,
FiLM1024; total hidden86288, input86544, FiLM87312. This pilot is not exactly
capacity-matched; report this limitation rather than attributing every gain to
conditioning structure alone.

Data, splits, batches and budgets match the completed BC pilot:100k train,10k
validation,10k test; fixed slips.025/.1/.2/1/10, long/short trajectories,100epochs,
batch256, Adam1e-4. Validation selects checkpoints; test is scored afterward.
Training seed controls shuffled batches, not new surfaces or initial backbone
weights. See ../context_training/README.md for detailed seed semantics.

## Train (offline, no simulator)

```bash
./run_film_training.sh audit
./run_film_training.sh train --device cuda
```

Use CUDA for this experiment because the completed controls used CUDA. The
trainer's generic default is CPU, but append evaluation rejects mismatched device,
library versions, data, initializer, seeds or training settings. The command above
matches current controls. Do not run both CPU and CUDA versions into the same
output. Other settings require a new `--output` and may require new matched controls.

Default output: experiments/context_conditioning/bc_film_v1/ with immutable
plan.json, film/seed_N/{policy.pt,optimizer.pt,metrics.json,completed.json}, and
summary.json. The new checkpoint schema is film_bc_v1, storing the precise FiLM
location/branch/initialization contract. Existing context_bc_v1 models remain
supported by the FiLM loader. Non-finite parameters and incompatible metadata fail.

As with the original trainer, completed jobs resume strictly; incomplete attempts
are archived and restarted. No mid-epoch resume. Source files, dataset receipts,
settings and initializer hashes are captured and rechecked. The training loop is
a versioned copy of the BC pilot loop to preserve the completed pilot's identity;
future changes must keep matched budgets explicit.

## Evaluate: append new models, reuse saved controls

```bash
./run_film_evaluation.sh evaluate --suite smoke
./run_film_evaluation.sh evaluate
```

Smoke runs the first prior round for three FiLM checkpoints (3 deployments).
Default evaluation is `--suite all --modes correct`: **99 new deployments**,
33 exact prior slip/task-seed rounds ×3 FiLM checkpoints,3,000 active steps each.
The saved oracle, hidden BC and input-correct BC results are used as references.
They are never scheduled as new live runs. Smoke has separate artifacts and is
not automatically reused by the full suite.

The schedule is read from
experiments/context_conditioning/bc_input_v1/evaluation_all_v1/plan.json, not
resampled:15 grouped,15 prior confirmation,3 nominal rounds. Because those
confirmation results were inspected before FiLM design, they are reused comparison
cases, **not untouched FiLM confirmation evidence**. Exact task seeds do not ensure
identical startup states; use repeatability diagnostics when interpreting small gains.

Optional FiLM context ablations require a different output if a correct-only
plan already exists:

```bash
./run_film_evaluation.sh evaluate --modes correct constant wrong \
  --output experiments/context_conditioning/bc_film_v1/evaluation_append_ablations_v1
```

This schedules **297 new FiLM deployments**, still zero control deployments.
Constant supplies.1; wrong supplies10 below physical slip1, otherwise.025. Physics
is unchanged by context mode. New pairs compare correct FiLM against saved hidden
and input-correct at matching training/task seeds; ablations compare against the
same FiLM checkpoint's correct mode.

The runner owns a fresh synchronous server per deployment, verifies the scalar
scene and uses the shared simulator lock. Do not start a server beforehand. There
are no internal human checkpoints. One retry is allowed by default, failed
attempts are archived, and rerunning the same command skips verified completions.

`plan` validates completed training/baseline artifacts and saves an offline plan.
`report` regenerates comparison.md and comparison.json offline; evaluation calls
it automatically. Repeat the same suite/modes/output flags when reporting:

```bash
./run_film_evaluation.sh report
```

Artifacts live under bc_film_v1/evaluation_append_<suite>_v1/ by default. The
combined report marks saved rows `reused_baseline=true`; it does not alter the
prior report. JSON retains progress, falls, targets, speed/tracking, on-policy
latent MSE, smoothed joint-angle jerk and standardized joint-occupancy sliced
Wasserstein with paired progress/fall differences. Oracle ratios are not
random-normalized returns; sliced Wasserstein is not exact22-D Wasserstein.

All baseline receipts and traces are validated. Physics/controller source or
environment changes since the prior comparison prevent appending silently. If
startup behaviour is fixed later, that requires a new matched comparison; do not
mix those new runs with old controls as though the protocol were unchanged.

## References

- Perez et al. (2018), [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871): affine feature modulation. Their visual-reasoning results are not evidence of locomotion gains here.
- Beukman et al. (2023), [Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e7b768198d24d883d69704eee57efb0-Paper-Conference.pdf): motivation for architecture comparisons; this implementation is FiLM, not their Decision Adapter.

Offline tests: `python -m unittest film_training.test_pipeline`. No research
training or live evaluation was launched during implementation.
