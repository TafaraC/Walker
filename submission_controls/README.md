# Final-submission controls

Additive experiments prompted by the 28 October audit. Existing consolidated
models, datasets, evaluations and probe artifacts remain read-only. The purpose
is to qualify optimization/history and gait-interface effects—not replace the
completed finite-budget study or start another GAIL experiment.

## Commands

From the Walker repository, with the fcportugal environment available:

```bash
./run_submission_controls.sh plan
./run_submission_controls.sh audit

# Separate small smoke batch: two short supervised fits, nine32-step deployments.
./run_submission_controls.sh run --quick

# Full standard controls: training, owned-server evaluation, curves, report.
./run_submission_controls.sh run
```

The script runs to completion without human checkpoints. Do not manually start
rcssserver3d. Close another simulator/training evaluation before live stages.
Open ports are refused; unrelated processes are never killed. Each deployment
owns a fresh server with the same scene-copy/import verification and restoration
checks as the consolidated pipeline. The existing shared simulator lock is used.

Stages can also be run separately:

```bash
./run_submission_controls.sh train
./run_submission_controls.sh evaluate
./run_submission_controls.sh curves
./run_submission_controls.sh report

# Zero-action evaluations without waiting for BC training:
./run_submission_controls.sh zero
./run_submission_controls.sh report --zero-only
```

`evaluate` and `zero` generate their reports after evaluation. `report` and
`curves` are offline: neither launches training nor a simulator. `train` uses
CUDA if available, otherwise CPU. Simulator evaluation is CPU-based; no parallel
servers are launched. `PYTHON_BIN` overrides the default fcportugal interpreter.

Default output: `experiments/submission_controls/v1/`.
`--quick` defaults to `experiments/submission_controls/smoke_v1/`, always separate.
`--parent` selects the completed consolidated batch (default
`experiments/consolidated/v1`). Its completed `scratch_v1` extension is required.
`--output` must be outside the parent tree. Do not change plan flags when resuming.

## The default training batch

| Control | Initialization and data | Allocation |
|---|---|---|
| Original matched-update BC | Saved original random BC initialization; same170k expert pool | Exactly211,050 Adam updates |
| Varied-only BC, hidden | Saved scratch random initialization; same varied170k pool; seeds42/43/44 | 200total epochs each |
| Varied-only BC, input | Saved scratch input initialization; same varied170k pool; seeds42/43/44 | 200total epochs each |
| Zero native gait action | Identically zero-output63→256→256→16 network | No learning |

There are seven trained BC policies plus the zero-action controller. All BC
fits restart from the saved initial checkpoint; no original100-epoch training
is inherited or counted twice. This avoids presenting a resume from a selected
earlier epoch as an exact total training budget. The same seed, optimizer type,
batch size256 and learning rate1e-4 are retained. Input models keep the existing
context encoding. No new demonstrations or DAgger queries are collected.

At170,000examples and batch256, an epoch has665updates including its final
16-example minibatch. The original matched budget is317complete epochs plus
245minibatches, exactly211,050updates and53,952,720example exposures. This matches
DAgger's update count, not exactly54million exposures or total query/collection
cost. The200-epoch varied fits use133,000updates and34million exposures each.

Validation is scored each complete epoch and at a partial budget endpoint.
`policy.pth`/`optimizer.pth` are the best validation pair within the allowed
budget. `last_policy.pth`/`last_optimizer.pth` preserve the actual last update.
Metrics distinguish **allocated/performed updates** from the **selected
checkpoint's update count**. No held-out test labels enter training or selection.
The default evaluation uses the validation-selected policy, not an independently
chosen best test checkpoint.

`--architectures hidden` narrows the varied controls to three hidden models.
Use a distinct output and keep that flag for every stage; defaults include both
architectures because the completed context comparison includes both.

## Optional stopping diagnostics

```bash
./run_submission_controls.sh plan --diagnostics
./run_submission_controls.sh run --diagnostics
```

This is a separate planned batch at `experiments/submission_controls/diagnostics_v1`
and **includes the seven standard BC fits again**. Choose this option before
launching if you want the diagnostics in the same batch; it does not silently
append flags to an already immutable standard plan.

Two extra controls:

- Original BC from the same saved random initialization: patience25epochs,
  minimum absolute validation improvement1e-5, stopping allowed after100epochs,
  hard cap500epochs.
- Original mixed DAgger: start from its final saved policy/optimizer and fit its
  **fixed final aggregate** with the same patience/delta and cap500additional
  epochs; stopping allowed after the first additional epoch. No new rollout or
  oracle query. Inherited211,050updates/54million exposures and70klearner steps
  remain explicit alongside additional diagnostic cost.

Best validation selection tracks every actual improvement; patience resets only
on improvement beyond the declared delta. `stop_reason` distinguishes reaching
patience from exhausting the cap. This establishes a validation stopping
criterion, not mathematical convergence or convergence of DAgger's iterative
state-distribution collection. Optional diagnostics add two actors and76full
deployments. No test feedback changes patience, delta, budget or checkpoint.

## Zero action means zero native gait offsets

BC/DAgger outputs are not standardized by dataset mean/std. All16zero outputs
are therefore different from the expert's constant mean action. We zero all
weights/biases of a standard BC network and use the **unchanged consolidated
evaluator**. Exact zero output is verified before committing its artifact.

The normal observation, target, action smoothing, step generator, fixed foot
offsets, arm swing, IK, fall reset and stabilization remain. It is not zero
joint commands, frozen joints or removal of the gait engine. Initial/reset
action-filter handling follows the same existing controller path. A high or
low result measures the retained interface's task capability under zero native
offsets; it does not by itself measure the contribution of each subcomponent.

The independent `zero` stage stores runs under `evaluation_zero/`; full
evaluation stores them under `evaluation/`. Choose one route if avoiding
duplicate zero deployments matters—these two receipts are intentionally separate.

## Evaluation and reuse

Reuse the exact saved38round schedule: five original-scene tasks,15grouped
draws,15confirmation draws and three corrected scalar0.1runs. Slip is fixed
throughout a run. Every new run has3,000active20ms steps (60simulated seconds).

- Original matched BC and zero action:38runs each.
- Six varied-only controls:33runs each (no original legacy scene).
- Total default:274new deployments; no old model is retrained or reevaluated.
- Hidden-only alternative:175new deployments.

Old oracle/random runs are verified and reused for normalization and occupancy.
Existing original BC/DAgger, varied continuation BC and scratch100-epoch BC are
read-only references. Comparison reports recompute metrics from verified traces
without writing to those old directories. Checkpoint/data/source/runtime hashes
must match; changed history is rejected rather than silently mixed.

Tables report target progress, normalized return, held-out **latent-action** MSE
alongside22-joint sliced Wasserstein distance, on-policy MSE, velocity, direction
error, falls/exposure, angular jerk rate and torso RMS. Grouped+confirmation
rounds are pooled as sampled; original and scalar nominal remain separate.
Training replicas retain individual labels, not pseudo-independent pooled frames.
Reports are descriptive; they do not declare significance or causal explanations.
`paired_differences.csv` also records new-minus-reference differences within each
matched round: matched BC versus original BC/DAgger,200-epoch BC versus its
continued and100-epoch scratch counterparts, and zero versus the oracle.

Smoke uses the full parent's saved initializations/data contracts but only64training
and32validation examples per fit, tiny update budgets, two rounds and32active
steps. It runs fresh32-step oracle/random controls so normalization is horizon-
matched. It is a plumbing check, not evidence for performance or convergence.

## Fidelity curves

`curves` uses the original-scene saved BC endpoint, DAgger rounds0–7 and GAIL
rounds0–7 (round0 is each GAIL initializer). It scores the existing original
validation/test pools without changing selection. Outputs:

- `curves/curves.csv` and `.json`: checkpoint hashes, MSE and resource columns.
- `curves/fidelity_resources.png`: DAgger against oracle labels including seed,
  GAIL against learner interactions including the initializer.

Columns separately record new/inherited learner steps, DAgger queries, unique
lineage labels and supervised optimization. GAIL-DM inherits70kDAgger interactions
and uses the shared170k static expert pool plus70kDAgger labels:240kunique labels
in that lineage; GAIL-BC uses170k. GAIL optimizer work is **not** represented by
its inherited supervised-update column. Shared physical dataset collection must
not be charged repeatedly as new queries across variants.

These are retrospective **fidelity-versus-resource** curves, not normalized-return
convergence or a way to select a test-optimal round. Full per-epoch control
validation curves are in each training job's `history.json`.

## Resume, artifacts and limits

Every completed training/evaluation job has a checksum receipt. Rerunning the
same command verifies and skips it. An interrupted *job* is moved to a unique
`_failed_...` sibling and restarted; intra-epoch/mid-job continuation is not
implemented. Failed data are retained. Input/source/plan changes require a new
output directory. Full training can be long; no research run is auto-launched by
installation or tests.

Main report: `evaluation/summary/comparison.md`, `.csv`, `.json`.
Independent zero report: `evaluation_zero/summary/`.
Policies/history/metrics/receipts: `training/<control>/`.
Server configuration, logs, traces and receipts: `evaluation/runs/<round>/<control>/`.

Offline engineering checks:

```bash
python -m unittest submission_controls.test_pipeline
```

No existing fingerprinted Python pipeline was edited. The final paper and slides
must not incorporate outcomes until these controls actually complete and are
reviewed. Existing convergence/causality/latent-interface caveats remain valid.
