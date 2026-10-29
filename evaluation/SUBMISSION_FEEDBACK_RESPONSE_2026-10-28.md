# Response to submission feedback: completed controls

28 October 2026. This addendum preserves the historical results and revises their
interpretation using the completed submission-control batch. It is not a new
algorithm-ranking claim or a convergence study.

## What was completed

Seven additional BC fits and **274 new deployments** are complete: one original
BC model matched to mixed DAgger's 211,050 optimizer updates, and six varied-only
BC models (hidden/input-conditioned, seeds 42/43/44) allocated 200 epochs each.
The zero-native-action controller was evaluated alongside them. Deployments use
3,000 active steps (60 simulated seconds), existing task/surface rounds and the
same control/reset machinery. Original-scene, corrected nominal and sampled
suites remain distinct. Historical references were reused, not measured again.
All checkpoints were selected using validation, not test performance.

The saved comparison and curve outputs passed their completion-receipt file-hash
checks during this write-up. This verifies saved output integrity, not an
independent rerun of the simulations. The optional early-stopping diagnostics
are not part of this completed default batch.

## 1. Was DAgger's fidelity advantage an optimization-budget effect?

The concern was justified. The original comparison allocated 66,500 updates to
BC and 211,050 to DAgger. Matching BC to the latter changes the ordering:

| Original-scene controller | Allocated updates | Held-out latent MSE | Progress (m) | Joint occupancy SWD | Falls / active minutes |
|---|---:|---:|---:|---:|---:|
| BC, original 100 epochs | 66,500 | 0.11047 | 44.630 | 0.04503 | 0 / 5 |
| Mixed DAgger | 211,050 | 0.01535 | 44.364 | 0.03909 | 0 / 5 |
| BC, matched updates | 211,050 | 0.01024 | 43.906 | 0.04716 | 0 / 5 |
| Oracle | — | — | 44.446 | 0 | 0 / 5 |

Matched BC reduces its error by approximately **90.7%** and has approximately
**33.3% lower MSE than DAgger** on this held-out expert set. The earlier 7.2-fold
DAgger advantage is therefore not robust to matching optimizer updates.
It is no longer appropriate to present that earlier result as evidence that
DAgger intrinsically imitates this expert better than BC.

This does not prove that learner-state labels are useless or quantify their
separate contribution. Datasets and supervisory costs still differ: BC has
170,000 expert-distribution examples; DAgger has 100,000 seed examples plus
70,000 learner-state labels. Equal updates also do not establish equal wall time
or equal collection cost. The matched BC result is one training seed.

The metrics do not all move together: matched BC has lower action error but
slightly lower mean progress and higher joint SWD than DAgger. This reinforces
the need to report fidelity, occupancy and task performance together.

## 2. Does original-domain pretraining still help after matching BC exposure?

Doubling varied-only BC from 100 to 200 epochs gives 34 million example
exposures, matching the allocated lifetime BC exposures of original-100 plus
varied-100 continuation. This matches exposure, not data distribution or
optimizer history; the distinct original and varied datasets remain a difference.

Values below average three training replicas on the same 30 sampled rounds
each. Falls are totals across 90 active minutes per row.

| BC lineage | Hidden MSE | Input-conditioned MSE | Hidden progress (m) | Input progress (m) | Falls, hidden / input |
|---|---:|---:|---:|---:|---:|
| Varied-only, 100 epochs | 0.11177 | 0.11226 | 33.192 | 33.274 | 4 / 4 |
| Varied-only, 200 epochs | 0.04210 | 0.03775 | 33.015 | 33.067 | 1 / 4 |
| Original 100 + varied 100 | 0.01995 | 0.01986 | 32.784 | 32.883 | 4 / 6 |

More optimization closes much of the observed fidelity gap: hidden varied-only
error falls by about 62.3%, input-conditioned by 66.4%. A residual gap remains:
the 200-epoch varied-only errors are about **2.11× and 1.90×** continuation's.
Thus the old 5.6-fold difference overstated what could be attributed to
pretraining independently of training exposure. Continuation still has better
fidelity in this finite-budget comparison, but there is no established universal
progress or stability benefit. Small differences and sparse falls are descriptive.

These are BC controls only; they do not resolve lifetime-budget confounds for
the DAgger and GAIL continuation/scratch comparisons. Nor do they establish
that explicit context is broadly useful: its effect depends on the metric and
training condition, and the per-replica results vary.

## 3. Did frozen BC require varied training to become more robust?

Longer original-scene BC training alone materially improves the sampled-surface
results, without adding varied-surface training examples:

| Frozen original-trained model | Sampled test MSE | On-policy MSE | Progress (m) | Joint SWD | Falls / 30 active minutes |
|---|---:|---:|---:|---:|---:|
| Original BC, 100 epochs | 0.17883 | 0.29098 | 32.828 | 0.09835 | 11 |
| Original mixed DAgger | 0.01634 | 0.02647 | 32.646 | 0.06570 | 6 |
| Original BC, matched updates | 0.00841 | 0.01343 | 33.151 | 0.04623 | 0 |

This is an important qualification to the earlier varied-training narrative.
The observed gains after continuation cannot all be credited to surface diversity:
optimization in the original environment also improved these measurements.
Matched original BC has more updates than the 200-epoch varied-only BC controls;
this is not an equal-budget comparison of which training domain is best.

The old rise in on-policy error and falls remains consistent with distribution
shift, but is not a unique diagnosis. Surface effects, visited-state difficulty
and insufficient fitting remain alternative or interacting explanations.

## 4. Was the gait engine doing the task without learned actions?

The zero baseline supplies zero in the **native 16-dimensional action space**,
not the mean of standardized expert actions. Gait generation, IK, observations,
commands and reset procedures remain active.

| Suite | Zero-action progress (m) | Oracle progress (m) | Zero falls / active minutes |
|---|---:|---:|---:|
| Original scene | -0.105 | 44.446 | 0 / 5 |
| Sampled surfaces | -0.303 | 32.885 | 12 / 30 |

Zero native actions do not produce useful target progress. In the original
scene, zero falls coexist with essentially no progress: stability alone is an
inadequate success criterion. The hypothesis that this particular zero-action
gait-engine baseline already matches oracle progress is not supported.
This does not test every constant vector, the expert-mean vector or removal of
the gait engine. It also does not turn the experiment into direct joint control.

## 5. GAIL and the efficiency analysis

Retained-checkpoint scoring is complete for 25 original checkpoints: BC's
endpoint, DAgger rounds 0–7 and both GAIL initializers plus rounds 1–7. The
artifacts include oracle labels, learner interactions and inherited costs.

DAgger held-out MSE declines from 0.21299 at round 0 to 0.01535 at round 7.
Both data aggregation and optimization accumulate across those rounds, so the
curve does not isolate the value of an additional oracle query.
GAIL-BC changes from 0.11047 to 0.11042; GAIL-DM from 0.01535 to 0.01572.
The curves strengthen the description of **supervised-initialized,
BC-regularized GAIL refinement** as producing little fidelity improvement at
this allocation. They are retrospective fidelity/resource curves, not
normalized-return convergence curves. The larger-budget GAIL study remains
outside this evidence.

The probes previously showed close selected readouts between GAIL and its
initializer. That is not proof that representations are unchanged. The newly
trained BC controls have not been probed here, so their performance must not
be explained using another checkpoint's probe scores.

## 6. What remains unresolved, and how the paper should change

- **Convergence:** original matched BC selected the final partial epoch 318;
  varied-only models selected epochs 197–200 of 200. Earlier DAgger selected
  epoch 50 of its final 50-epoch fit. All seven controls stopped at their budget.
  No two-method convergence claim is justified; optional early-stop diagnostics
  remain unrun in this batch.
- **Literature framing:** do not claim that 170,000 correlated demonstrations
  establish data sufficiency or that BC generally catches GAIL with abundant
  data. Report the empirical optimization sensitivity found here.
- **Fidelity wording:** use “latent-action fidelity,” with joint-occupancy SWD
  beside MSE. SWD discards temporal order and cannot prove that IK suppresses
  action differences. Jerk is kinematic, not torque or measured energy.
- **Proposal deviations:** explicitly state 16 native gait outputs rather than
  22 joint-velocity outputs; the regularized GAIL recipe; finite allocations and
  retrospective fidelity curves rather than demonstrated return convergence;
  kinematic jerk rather than torque smoothness.
- **Probing scope:** retain a short descriptive section, not a causal explanation
  of DAgger's advantage. That advantage changes under the optimization control.

The defensible central conclusion is now stronger methodologically: similar
progress can coexist with different fidelity, but **the fidelity ranking is
sensitive to optimization allocation**. Training history and domain exposure
must be separated before making algorithm or pretraining claims.

## Evidence and presentation

- [Control summary](../experiments/submission_controls/v1/evaluation/summary/comparison.md)
- [Full metrics, training costs and paired differences](../experiments/submission_controls/v1/evaluation/summary/comparison.json)
- [Retained-checkpoint curves](../experiments/submission_controls/v1/curves/curves.csv)
- [Curve figure](../experiments/submission_controls/v1/curves/fidelity_resources.png)
- [Control protocol](../submission_controls/README.md)
- [Updated presentation](presentation/CONSOLIDATED_PAPER_MARP.md)

Original reports are preserved. This addendum and updated presentation qualify
their earlier interpretations. The [revised paper draft](FINAL_PAPER_DRAFT.md)
now incorporates these controls, with a [LaTeX package](paper/README.md) and
reusable figures. Compilation and the final eight-body-page layout remain to
be checked before submission.
