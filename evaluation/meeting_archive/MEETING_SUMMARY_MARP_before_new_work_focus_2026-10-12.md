---
marp: true
theme: default
size: 16:9
paginate: true
title: "FC Portugal: imitation, domain sensitivity and representations"
description: "Research meeting update — 11 October 2026"
footer: "FC Portugal Walker · Research update · 11 October 2026"
style: |
  section { font-size: 26px; padding: 48px 60px; }
  h1 { font-size: 39px; color: #123b57; }
  h2 { font-size: 31px; color: #123b57; }
  table { font-size: 22px; width: 100%; }
  th { background: #eaf1f5; }
  blockquote { border-left: 5px solid #237a91; color: #123b57; }
  section.compact { font-size: 23px; }
  section.compact table { font-size: 20px; }
  section.references { font-size: 21px; }
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# Learning to walk like FC Portugal

## Imitation, domain sensitivity and internal representations

Research meeting · 11 October 2026

BC · Mixed DAgger · GAIL-DAgger · GAIL-BC

<!--
Scope runs from the original four-policy nominal comparison through completed
context, repeatability, FiLM and expert probing. R1 transfer is deferred.
Source: ../MEETING_SUMMARY_2026-10-10.md. This deck is a synthesis, not new experiments.
-->

---

# Main findings

- Nominal walking performance is near oracle parity; **BC is the simplest strong baseline**.
- Mixed DAgger improves fidelity; GAIL retains much of its initializer's profile.
- Effective slip needed independent validation; corrected high slip reduces progress.
- Input context and FiLM provide **no consistent practical BC benefit** here.
- Probes reveal accessible locomotion structure—not causal mechanisms or transfer utility.

<!--
No claim that all methods are equivalent, that context is always unnecessary, or
that representation similarity establishes inherited knowledge. Confidence and
control limitations accompany the relevant slides.
-->

---

# Research questions and evaluation contract

1. How faithfully do learned policies reproduce the oracle?
2. How do performance and stability change with dynamics?
3. Does explicit environment context help imitation?
4. Which locomotion properties remain accessible internally?

**63-D observations → network → 16-D latent Walk residuals → gait generator / IK → joints**

3,000 active steps = 60 simulated seconds; own-half targets.
Startup and fall recovery are excluded from active time.

<!--
The 16 outputs are not direct joint velocities or torques. The external stateful
generator and IK remain part of the controller and are outside the neural probes.
Nominal study uses five evaluation seeds; later protocols have separate schedules.
-->

---

# Original environment: performance versus fidelity

| Policy | Normalized return [95% CI] | Expert-test MSE | On-policy MSE |
| --- | --- | ---: | ---: |
| BC | 1.101 [0.906, 1.296] | .0084 | .00437 |
| Mixed DAgger | 1.096 [0.902, 1.290] | .0612 | .00464 |
| GAIL-DAgger | 1.106 [0.910, 1.302] | .2843 | .09880 |
| GAIL-BC | 1.104 [0.902, 1.307] | .0085 | .00444 |

Five seeds per policy; all four recorded zero falls.

> Similar task progress can coexist with very different action fidelity.

<!--
Source: ../PHASE1_RESULTS_REPORT.md. MSE is latent action error. Occupancy
Wasserstein: BC .0906, mixed .0808, GAIL-DAgger .1072, GAIL-BC .0817.
Intervals overlap oracle parity and each other; no policy superiority established.
Oracle had one fall over five minutes. Normalized score uses same-seed random
and oracle baselines, so scores above one partly reflect oracle variability.
-->

---

# What changed with mixed DAgger and GAIL?

**Mixed DAgger versus original DAgger**

- Expert-test MSE: .2848 → .0612 (**~78.5% reduction**).
- On-policy MSE: .09217 → .00464 (**~95% reduction**).
- Larger seed dataset **and** balanced long-run/short-turn coverage changed together.

**GAIL:** BC initialization preserves BC-like fidelity;
original-DAgger initialization retains much larger action error.

Random-start GAIL remains stationary under the tested budget.

<!--
Original DAgger: 20k seed +70k rollout labels. Mixed: 100k seed +70k labels.
Thus coverage and quantity are confounded. GAIL uses FC Portugal expert data,
not DAgger as expert. GAIL-DAgger is original, not mixed DAgger initialization.
Random-start GAIL normalized return -.011, forward velocity -.005 m/s, zero
targets. Low jerk/falls without progress are not successful locomotion.
-->

---

# Cost and interpretation matter

| Method | Supervision / interaction burden |
| --- | --- |
| BC | 90,000 static training labels; no on-policy training |
| Original DAgger | 90,000 labels total; 70,000 rollout steps |
| Mixed DAgger | 170,000 labels total; 70,000 rollout steps |
| GAIL | 40,000 additional rollout steps + static expert data + initializer costs |

**Zero active GAIL oracle queries ≠ zero expert supervision.**

BC provides the strongest nominal fidelity/simplicity trade-off.

<!--
Source nominal report. GAIL uses 90k static FC Portugal examples and inherits
pretraining cost; report marginal and end-to-end costs separately. Performance
claims are budget-specific, not a general ranking of imitation algorithms.
-->

---

# Early domain studies: an important correction

- Legacy fixed sweep: **120 runs** across configured .025/.1/.2.
- Initial Gaussian continuation comparison: **65 evaluations**, eight final checkpoints.
- Little apparent surface sensitivity; no clear varied-training progress gain.

**But the old two-argument setter did not vary logged solver slip.**

> Configuration/import verification was not enough to establish changed physics.

Preserve these results as legacy continuation evidence—not terrain robustness evidence.

<!--
Legacy expression: (setContactSlip v v). Passive audits found unchanged contact
classes even at large requested values. Five-pair varied-minus-nominal progress:
BC +.097m, mixed -.077m, GAIL-BC +.062m, GAIL-DAgger -.224m; all exploratory CIs
included zero. BC/mixed fidelity improved similarly with nominal continuation.
Do not pool legacy and corrected conditions or reuse legacy baseline normalization.
Sources: ../../experiments/phase2_domain_robustness/FIXED_RESULTS_INTERPRETATION.md
and ../../experiments/variable_slip_training/gaussian_v1_seed42/CROSS_PHASE_INTERPRETATION.md.
-->

---

# Corrected scalar slip changes locomotion

Single-argument setter changes enabled solver-bound contact parameters.

| Field setting | Oracle progress (m) | Retention versus corrected .1 |
| --- | ---: | ---: |
| .025 | 43.972 | 108.4% |
| .1 | 40.546 | 100.0% |
| .2 | 37.591 | 92.7% |
| 1 | 28.809 | 71.1% |
| 10 | 15.325 | 37.8% |

One-seed screen; **zero falls does not mean no dynamics impact**.

<!--
Source: ../../experiments/slip_sensitivity/oracle_scalar_v1/INTERPRETATION.md.
Five seed101 runs, 3000 active steps each. Correct expression (setContactSlip v).
Solver parameters are not measured foot sliding; exact legacy fallback unproven.
This validates responsiveness, not population effects or a universal safe range.
Original nominal configuration is not automatically corrected field slip .1.
-->

---

# Reliable experiments require server control

**Observed failure:** monitor backpressure / socket teardown crashes between runs.

**Managed experimental lifecycle**

Local scene → fresh server → verified configuration → deployment → receipt → teardown

- Preserve failed attempts and completed results.
- Resume only matching, verified identities.
- Separate infrastructure failures from policy falls.

<!--
Changes reduce lifecycle coupling but do not justify claiming all simulator races
are eliminated. New lifecycle differs from original/shared-server protocols, so
cross-phase comparisons are not automatically paired causal comparisons.
Corrected frozen tooling plans 5 slips x3 seeds x8 controllers=120 deployments.
No consolidated scalar_wide_v4 fixed report was located during the summary review;
do not substitute the legacy report for a verified corrected full-grid result.
-->

---

# Corrected varied training: evaluation design

Training grid: **.025 · .1 · .2 · 1 · 10**

For each grid value: Gaussian parent mean = value, std = half the value;
truncate draws to **(0, 10]**.

**5 groups × 3 draws × 5 controllers = 75 deployments**

Each round shares one slip and task seed across all controllers.
Slip stays fixed for the entire run; each deployment gets a fresh server.

This is **not per-contact Gaussian noise** or a spatial Gaussian field.

<!--
Source: ../../experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/evaluation_grouped_v1/report.md.
75 completed deployments, 750 referenced artifacts hash-verified in recorded
review. Group mean10 is upper-truncated; sampled mean need not equal parent mean.
Nominal training arm means continuation at .1, not just a nominal test.
Deployment rounds are not independent training seeds.
-->

---

# Grouped evaluation: policies largely track the oracle

| Controller | Mean progress (m) | Mean oracle ratio | Falls / 15 runs |
| --- | ---: | ---: | ---: |
| Oracle | 33.364 | 1.000 | 0 |
| BC varied | 33.292 | 1.005 | 0 |
| Mixed DAgger varied | 33.078 | .991 | 1 |
| GAIL-BC varied | 32.937 | .988 | 1 |
| GAIL-DAgger varied | 32.684 | .985 | 4 |

Oracle group progress declines **42.091 → 16.767 m** with severity.
No established algorithm-wide ranking from this descriptive comparison.

<!--
Mean per-round oracle ratios are not random-normalized scores, and not ratios of
aggregate means. Only three draws/group. Three GAIL-DAgger falls occur in group
centered1; not a robust population ranking. Grouped report lacks paired frozen
and nominal-continuation controls, so cannot show a causal varied-training gain.
Held-out varied element MSE: BC .0018185, mixed .00110956, GAIL-BC .00725441,
GAIL-DAgger .232627. Do not compare across different test sets as identical tasks.
-->

---

# Does supplying slip explicitly help BC?

Three architectures: **hidden context · input concatenation · FiLM**

Matched collected data, original BC initializer and fine-tuning budget.
Seeds 42/43/44 produce separate model replicas—not an ensemble.

| Set: mean progress (m) | Hidden | Input | FiLM |
| --- | ---: | ---: | ---: |
| Original sampled | 33.073 | 33.060 | 33.232 |
| Confirmation | 32.292 | 32.665 | 32.581 |
| Nominal | 39.272 | 39.175 | 39.219 |
| Falls / 99 runs | 5 | 4 | 5 |

> No consistent practical conditioning benefit in this setting.

<!--
Context study:429 runs including correct/constant/wrong inputs. FiLM:99 new
correct-context runs on prior rounds, verified controls reused. Confirmation
draws were previously inspected, not untouched FiLM tests. Inputs share data and
initializer; fine-tuning replicas mainly vary minibatch order, not independent
training datasets. FiLM correct-only does not separate capacity from context use.
Sources in ../../experiments/context_conditioning/: bc_input_v1/evaluation_all_v1/report.md
and bc_film_v1/evaluation_append_all_v1/comparison.md.
-->

---

# Small context effects need cautious interpretation

- Wrong/constant context changes actions, but does not consistently reduce progress.
- FiLM held-out MSE gain: **~.18%**; sampled occupancy distance is worse.
- One **+14.075 m** case dominates its confirmation progress gain.
- Repeatability: **45 runs**, same-controller progress SD **.24–.62 m**.
- Identical initial targets did not produce identical first-active physical states.

Keep difficult cases; do not use repeatability SD as a universal significance threshold.

<!--
Excluding no official observations: other44 FiLM-hidden pairs average -.0234m
as sensitivity diagnostic only. Jerk broadly similar. Unequal settling states
are observed, not proven to explain all variation. Repeatability used one trained
hidden/input pair, three cases and5 repetitions. Source:
../../experiments/context_conditioning/bc_input_v1/repeatability_v1/repeatability.md.
-->

---

# Why might context add little?

**Hypothesis:** ordinary observations already expose consequences of slip;
the oracle has no explicit slip input.

Imitation rewards reproducing oracle actions—not discovering a superior
surface-specific strategy.

**Decision:** pause context-conditioned DAgger/GAIL expansion.

**Not established:** context is useless, the expert never adapts, or other
methods/objectives cannot benefit.

<!--
Teacher has no explicit slip input but can respond to consequences implicitly.
This explanation is plausible, not proven. Context DAgger/GAIL are untested;
stopping is a research-priority decision, not their experimental failure.
-->

---

# Probing: from slip decoding to general locomotion structure

Frozen BC, mixed DAgger, GAIL-DAgger and GAIL-BC; shared oracle states.

| Study | Scope | Completed readouts |
| --- | --- | --- |
| Preliminary | Slip, posture, future velocity | 60 fits / 300 scores |
| General | 30 gait/body/force/command/future properties | 64 fits / 1,920 scores |
| Expert append | Expert + matched random controls | 16 new fits / 2,400 combined scores |

Continuous targets use **ridge regression**, with whole-run splits and
raw-input, random-network and shuffled-label controls.

<!--
Alain/Bengio linear-probe framework adapted to continuous regression;
Hewitt/Liang motivate controls. Random-feature controls are essential because
preliminary slip h1 R2 .440/.443/.461/.440 versus random .404–.424.
Decoding directly observed posture is retention, not discovery. General study
excludes slip target and includes36 in-memory layer-reset diagnostics.
Source model files unchanged. Scores count scalar readouts, not independent fits.
-->

---

# Strong candidate: phase-conditioned posture

First-layer linear decoding of **sin(gait phase) × torso roll**:

| Features | Held-out R² |
| --- | ---: |
| Raw observations | .332 |
| Random-network controls | .427–.452 |
| BC | .697 |
| Mixed DAgger | .703 |
| GAIL-DAgger | .586 |
| GAIL-BC | .697 |

Every learned h1 beats every random control on all 15 test runs.

<!--
Source ../../experiments/representation_probing/nao_sources_v1/INTERPRETATION.md.
Constructed interaction is a useful accessibility hypothesis, not a proven
balance mechanism. Shuffled scores are low for this target. GAIL-DAgger h2
retains more gyro-x/y; mixed modestly leads future-direction-error. BC/GAIL-BC
profiles similar. Periodic timing survives block shuffles; persistence weak for
averaged/oscillatory targets. Resets ~2.5–2.8 action RMS do not isolate semantics.
-->

---

# What resembles the expert?

Expert: **64 tanh units**; student layers: **256 ReLU units**.

| Representation | Pooled expert CKA |
| --- | ---: |
| Mixed DAgger h1 | .9355 |
| GAIL-DAgger h2 | .9314 |
| Raw observations | .8968 |

Expert exceeds matched random controls on command orientation,
phase-conditioned roll and future pitch.

**Similarity ≠ percentage of knowledge transferred.**

<!--
Source ../../experiments/representation_probing/nao_expert_comparison_v1/INTERPRETATION.md.
Both highlighted student layers beat same-layer random CKA controls on15 runs.
Raw/random similarities are high; shared observations explain substantial geometry.
All240 student-layer/property R2 gaps favor students, but width/nonlinearity
confounds prohibit superior-knowledge claims. Expert action agreement checked
over90k states, max abs6.56e-6. Generator/IK is not included in these comparisons.
-->

---

# Supported conclusions and open questions

**Supported**

- Strong nominal imitation; performance and fidelity are distinct.
- Corrected high slip reduces progress; context gains are marginal here.
- Teacher/student representations expose structured locomotion properties.

**Still open**

- Which features causally contribute to balance and control?
- How much is architecture, ordinary input geometry or learned organization?
- Do stronger periodic-null controls or nonlinear probes change the picture?

R1 transfer is deferred—not a negative-transfer result.

<!--
No claim of exhaustive internal representation discovery. Nonlinear regression
and stronger nulls need new/versioned studies. Joint/motor impairment remains
future work; do not present it as tested. R1 snag unspecified, separate directory.
Current focus is NAO evidence consolidation and warranted probe controls.
-->

---

<!-- _class: references -->

# Sources and reproducibility

- [Full meeting brief](../MEETING_SUMMARY_2026-10-10.md) · [command and demo guide](../../COMMANDS.md)
- [Original nominal results](../PHASE1_RESULTS_REPORT.md)
- [Corrected scalar validation](../../experiments/slip_sensitivity/oracle_scalar_v1/INTERPRETATION.md)
- [Grouped varied evaluation](../../experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/evaluation_grouped_v1/report.md)
- [General representations](../../experiments/representation_probing/nao_sources_v1/INTERPRETATION.md)
- [Expert comparison](../../experiments/representation_probing/nao_expert_comparison_v1/INTERPRETATION.md)
- Probe methods: [Alain & Bengio](https://arxiv.org/abs/1610.01644), [Hewitt & Liang](https://aclanthology.org/D19-1275/), [CKA contract and references](../../representation_probing/EXPERT.md)

Previous meeting brief is preserved in [meeting_archive](../meeting_archive/README.md).
No experiments were rerun for this presentation.

<!--
All local paths resolve relative to this Markdown file. Keep repository structure
for local source navigation; exported slides are standalone but local Markdown
source links may not be navigable outside the repository. Speaker notes carry
technical limitations and provenance; retain them when editing.
-->
