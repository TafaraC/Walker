---
marp: true
theme: default
size: 16:9
paginate: true
title: "FC Portugal: context conditioning and learned representations"
description: "Research meeting update — 12 October 2026"
footer: "FC Portugal Walker · Research update · 12 October 2026"
style: |
  section { font-size: 26px; padding: 46px 58px; }
  h1 { font-size: 38px; color: #123b57; }
  h2 { font-size: 30px; color: #123b57; }
  table { font-size: 22px; width: 100%; }
  th { background: #eaf1f5; }
  blockquote { border-left: 5px solid #237a91; color: #123b57; }
  section.compact { font-size: 23px; }
  section.compact table { font-size: 20px; }
  section.references { font-size: 21px; }
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# Context conditioning and learned representations

## What the newer NAO experiments show

Research meeting · 12 October 2026

BC · Mixed DAgger · GAIL-DAgger · GAIL-BC

<!--
Focus: context BC, FiLM, repeatability, general representation probes and expert
comparison. Two brief recap slides establish the earlier evidence. The broader
written summary remains ../MEETING_SUMMARY_2026-10-10.md. R1 is deferred.
-->

---

# Brief recap: a strong walking baseline

- All four selected policies achieved nominal progress near oracle parity.
- **BC:** strongest fidelity/simplicity trade-off.
- **Mixed DAgger:** improved original DAgger fidelity with more diverse, larger data.
- **GAIL:** retained much of its initializer's fidelity profile.

> Similar walking performance leaves an interesting question:
> what do the networks represent differently?

<!--
Source: ../PHASE1_RESULTS_REPORT.md. Five seeds, 3000 active steps per policy.
Normalized return BC1.101, mixed1.096, GAIL-DAgger1.106, GAIL-BC1.104; intervals
overlap parity. Test MSE .0084/.0612/.2843/.0085; no nominal falls among these four.
Mixed changes seed quantity and composition together; causal separation unavailable.
GAIL-DAgger uses original DAgger initialization; FC Portugal remains the expert.
16 outputs are latent gait residuals processed through the external generator/IK.
Historical cost details and random-start stationary failure remain in written report.
-->

---

# Brief recap: corrected dynamics change performance

Old scene settings did not vary logged solver slip; scalar setting fixed the manipulation.

| Corrected evidence | Main observation |
| --- | --- |
| Oracle smoke, slip .1 → 10 | Progress 40.546 → 15.325 m; zero falls |
| 75 grouped evaluations | Varied-trained policies largely track oracle progress |
| GAIL-DAgger in those 15 rounds | Four falls; other learned policies zero or one |

This established the setting for asking whether **explicit slip context helps**.

<!--
Scalar smoke is one seed, not population robustness evidence. The corrected
grouped study samples one slip per run, fixed throughout; 5 groups x3 draws
x5 controllers, each with fresh server. Mean progress oracle33.364, BC33.292,
mixed33.078, GAIL-BC32.937, GAIL-DAgger32.684m. Oracle ratios are not normalized
returns. No matched frozen/nominal arms in this report to isolate a training gain.
Legacy results preserved but cannot establish effective-slip robustness. No
consolidated corrected scalar_wide_v4 fixed-grid report was found in prior review.
Sources: ../../experiments/slip_sensitivity/oracle_scalar_v1/INTERPRETATION.md;
../../experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/evaluation_grouped_v1/report.md.
-->

---

# Three ways to use the same BC demonstrations

| Architecture | How slip enters | Extra parameters |
| --- | --- | ---: |
| Hidden-context BC | 63 observed state features only | 0 |
| Input-conditioned BC | Append one slip feature → 64 inputs | 256 |
| FiLM BC | Slip scales and shifts the final hidden features | 1,024 |

All use the original BC weights and a 256 / 256 ReLU backbone.

FiLM: **h′ = (1 + Δγ(c)) ⊙ h + β(c)**

Input slip weights start at zero; FiLM starts as the identity.

<!--
Context c=log1p(slip)/log(11), accepts(0,10]. Outputs remain16 latent actions.
FiLM context branch is linear, 1 to256 scale adjustments and256 offsets, applied
after second ReLU and before action head. Context never enters first state layer.
Both backbone and context parameters train. Total parameters86288/86544/87312;
not exactly capacity matched. Same initial action function up to numerical tolerance.
Sources ../../context_training/README.md and ../../film_training/README.md.
-->

---

# Training and evaluation controls

**Offline training:** 100k train / 10k validation / 10k test examples;
long runs and short turns at slips .025 / .1 / .2 / 1 / 10.

**Matched fine-tuning:** original BC initializer; 100 epochs;
Adam 10⁻⁴; batches of 256; validation selects checkpoints.

**Three training seeds:** 42 / 43 / 44, paired minibatch order.

**Live evaluation:** 33 rounds = 15 grouped + 15 confirmation + 3 nominal.

<!--
Whole trajectories split; no new demonstrations drawn per seed. Seeds mainly
vary minibatch ordering, conditional on shared pretrained weights and dataset.
3 checkpoints/architecture, each independently deployed, not an ensemble.
Hidden/input complete study429 runs:33 rounds x(oracle1+hidden3+input3x3 modes).
FiLM appends99 correct-context runs:33x3. Existing controls reused at matching
slips/task seeds. Confirmation cases were inspected before FiLM, not blind tests.
60 active seconds/deployment; fresh owned servers and verified completion receipts.
-->

---

# Context benefit is small and inconsistent

| Mean progress per run (m) | Hidden BC | Input BC | FiLM BC |
| --- | ---: | ---: | ---: |
| Grouped slips | 33.073 | 33.060 | 33.232 |
| Confirmation slips | 32.292 | 32.665 | 32.581 |
| Nominal | 39.272 | 39.175 | 39.219 |
| Total falls / 99 runs | 5 | 4 | 5 |

45 runs per architecture in each sampled suite; nine nominal.

> No architecture consistently improves progress and stability across suites.

<!--
Sources: ../../experiments/context_conditioning/bc_input_v1/evaluation_all_v1/report.md
and ../../experiments/context_conditioning/bc_film_v1/evaluation_append_all_v1/comparison.md.
Table uses correct context. These are descriptive means; three fine-tuning seeds
share initializer/data and do not give independent source-training replication.
-->

---

# Does the input-conditioned policy use slip?

| Supplied context | Held-out action response |
| --- | --- |
| Correct slip | Reference prediction |
| Constant slip .1 | Expert MSE increases ~0.41% |
| Extreme wrong slip | Expert MSE increases ~1.47%; action RMS change ~.0056 |

Context therefore affects predictions.

However, constant/wrong context does not consistently reduce **live walking progress**.

<!--
Read-only counterfactual inference on same10k held-out observations; this isolates
input changes from new simulator starts. Relative changes versus correct context.
Wrong supplies10 for physical slip<1, otherwise.025; constant supplies.1.
Correct-input expert error essentially equal/slightly worse than hidden BC.
Live effects combine context and run variability. Prediction sensitivity alone
does not establish context-dependent adaptation or task benefit.
Evidence recorded in SESSION_HANDOFF/context completion and original comparison.
-->

---

# Repeatability explains why small differences need care

Five repeats of each case, with the same checkpoint, slip and task seed.

| Case | Hidden progress mean ± SD (m) | Input progress mean ± SD (m) |
| --- | ---: | ---: |
| Low slip | 43.563 ± .439 | 43.771 ± .536 |
| Nominal | 37.493 ± .618 | 37.510 ± .241 |
| High slip | 17.122 ± .443 | 16.966 ± .452 |

Initial targets matched; first-active physical states differed.
Planar start spread reached **6.36–24.12 mm** across case/controller groups.

<!--
45 runs=3cases x3controllers(oracle,hidden42,input42) x5repeats. Cases slip
.0002849946/.1/7.4175646, same original task seeds. SD is deployment variability,
not standard error or a universal significance threshold. Cause not diagnosed.
Oracle included: SD .5002/.6002/.5925 low/nominal/high. Initial spread maximums
describe group ranges, not every pair. Source:
../../experiments/context_conditioning/bc_input_v1/repeatability_v1/repeatability.md.
-->

---

# FiLM adds flexibility, with limited practical gain

- Held-out latent MSE improves only **~0.18%** over hidden BC.
- Joint-occupancy sliced Wasserstein is worse on both sampled suites.
- Jerk is broadly similar.
- One **+14.075 m** pair dominates the confirmation progress gain.
- The other 44 pairs average **−.0234 m** relative to hidden BC.

All cases remain in the official results.

<!--
This exclusion calculation is sensitivity analysis, not a revised score.
Correct-context-only FiLM evaluation cannot isolate context use from added
capacity. Saved controls are paired by task/replica, not identical physical
starts. Previously inspected confirmation rounds are reused comparison evidence.
Input confirmation advantage is also concentrated:~76% comes from one seed44
hidden failure at slip.0002849946, below minimum training slip.025.
-->

---

# Why pause further context training?

The teacher receives state observations but no explicit slip parameter.
Those observations already contain physical consequences of the surface.

BC optimizes reproduction of teacher actions; this may leave little incentive
for a separate slip-specific strategy.

**Decision:** defer context DAgger/GAIL and examine the learned representations.

**Scope:** the explanation is plausible; other tasks or learning objectives may
make context more useful.

<!--
Context DAgger/GAIL remain untested. This is prioritization, not their failure.
Neither teacher non-adaptation nor universal context irrelevance is established.
R1 transfer remains deferred; its progress is not a prerequisite for this analysis.
-->

---

# Probing asks what can be read from a frozen network

**Shared observation → frozen hidden layer → fitted linear readout → property**

Fit only the diagnostic readout; source policies remain unchanged.

- Current sensor quantities: how much information remains accessible?
- Interactions: does the network expose useful combinations?
- Future outcomes: is short-horizon oracle motion predictable from the features?

Continuous targets use **ridge regression**; score with held-out R² and error.

<!--
Alain/Bengio frozen-layer probe framework adapted from classification to regression.
Not training another walking policy. R2=1 perfect,0 mean-baseline level,negative
worse; held-out pooled R2 and macro-run errors reported. Features standardized
using train-only statistics; alpha chosen per target by validation macro-run MSE.
Current readouts already are regression. Nonlinear regression would be a separate
extension. Future labels occur under oracle actions, not hypothetical student actions.
-->

---

<!-- _class: compact -->

# Which features were tested?

| Family | Examples | What the label measures |
| --- | --- | --- |
| Gait timing | Phase sine / cosine | External generator timing |
| Support proxies | Foot force norms; force >10 N | Sensor-based support indicators |
| Body motion | Height, vertical velocity, roll/pitch, gyro xyz | Current observed physical state |
| Limb geometry | Left/right ankle xyz | Position relative to hip centre |
| Commands | Forward/lateral displacement, orientation | Smoothed target information |
| Phase–posture | sin(phase) × roll; cos(phase) × pitch | Constructed feature interactions |
| Future motion | Velocity, direction error, posture, support | Oracle outcomes over / at +0.2 s |

30 labels span several aspects of locomotion; the general study has no slip target.

<!--
Counts2+4+7+6+3+2+6=30. Support indicators are force thresholds, not independent
ground-truth contacts. Generator phase is not observed stance. Commands are
displacements/orientation, not desired velocity. Ankle locations are not foot
contact points. Future velocity/error are interval means; posture/contact endpoint
labels. No COM/ZMP/balance-margin annotation. Decodable quantity may be directly
supplied by input, which is why controls matter.
-->

---

# A shared dataset makes the model comparison interpretable

30 verified oracle runs → whole-run **10 / 5 / 15** train / validation / test split.

After subsampling and reset-boundary exclusions:
**4,270 / 2,133 / 6,405 observations**.

Compare original BC, mixed DAgger, GAIL-DAgger and GAIL-BC on identical states.

- Raw observations and phase/command-only features.
- Three random networks matched to student architecture.
- True labels and three trajectory-shuffled controls.

<!--
Probe originals, not varied/context/FiLM models. GAIL uses actor mean.
Stride7 avoids repeatedly sampling fixed residue of nominal8-step half-cycle.
Future horizon10 steps=.2s; discard recovery/reset/unobserved endpoints.
Grouped draws0/1 train,draw2 validation;confirmation runs test. Test episodes
held out from fitting but previously inspected in other studies.
Rows correlated; one source checkpoint/technique, unequal source training budgets.
16 representations x4 label conditions=64 fits x30 targets=1920 scalar scores.
-->

---

# Several locomotion properties are accessible

| Property | Observed pattern |
| --- | --- |
| Command orientation | Strong decoding in all four learned models |
| Hip-relative ankle geometry | Strong decoding across trained layers |
| Torso gyro x/y | GAIL-DAgger retains more in its second layer |
| Phase-conditioned roll | Clear learned-over-random advantage |
| Future direction error | Modest advantage, strongest in mixed DAgger h1 |
| Future forward velocity | Similar scores across models; small random-control gap |
| Phase / future support | High scores, but periodic controls retain much signal |

These patterns distinguish retention, useful combinations and weakly supported predictions.

<!--
Source ../../experiments/representation_probing/nao_sources_v1/INTERPRETATION.md.
Commands/ankles/body-state quantities already appear in input: accessibility
does not by itself identify learned expertise. Future support proxy balanced
accuracy~91–93% trained h1, but shuffled R2~.57–.66 and restricted phase-command
R2 .671/.603 show periodic explanation. Do not present this as inferred contacts.
-->

---

# Phase–posture interaction gives the strongest evidence

First-layer decoding of **sin(generator phase) × torso roll**:

| Features | Held-out R² |
| --- | ---: |
| Raw / phase-command only | .332 / .017 |
| Random h1 controls | .427–.452 |
| BC / mixed DAgger | .697 / .703 |
| GAIL-DAgger / GAIL-BC | .586 / .697 |

All four learned h1 layers beat each random h1 control on **all 15 test runs**.

<!--
Trained shuffled h1 R2~.006–.043. Phase and roll are both observed inputs; this
reflects easier linear access to their interaction, not new physical information.
Constructed label is not a validated stability margin. Consistency across15 test
episodes is descriptive, not independent policy replication or a significance test.
-->

---

# Angular motion and future direction separate models

| Property / layer | BC | Mixed DAgger | GAIL-DAgger | Random controls |
| --- | ---: | ---: | ---: | ---: |
| Gyro x, h2 R² | .807 | .822 | **.889** | .811–.824 |
| Gyro y, h2 R² | .798 | .845 | **.883** | .817–.852 |
| Future direction error, h1 R² | .518 | **.538** | .515 | .475–.491 |

Future forward velocity h1: learned **.607–.616**, random **.599–.602**.

Different models preserve different accessible features; no universal winner.

<!--
GAIL-BC profiles remain close to BC; omitted in this table only to reduce width.
Gyro already provided in observations, so this is retention through layers.
Future direction error is evaluator vector discrepancy, not radians; oracle's
next.2s mean. Raw R2 direction.460,velocity.576. Incremental forward-velocity
gain is small; avoid claiming learned forward dynamics from that result alone.
Source general INTERPRETATION.md.
-->

---

# Adding the expert anchors the comparison

| Network | Hidden architecture | Comparisons |
| --- | --- | --- |
| FC Portugal expert | One 64-unit tanh layer | Expert h1 |
| Imitation students | Two 256-unit ReLU layers | Each h1 and h2 versus expert |
| Controls | Matched random tanh / ReLU networks | Learned versus untrained features |

Expert actions verified against **90,000 recorded states**.

16 new fits extend the existing study to **2,400 scalar scores**.

<!--
Actual robot0 behaviors/custom/Walk/walk_R0.pkl,63->64tanh->16.
Max action mismatch6.56e-6 within audited floating point tolerances.
Same30 targets,rows,splits,readout settings;4new representations x4conditions.
No expert h2 exists. Expert external smoothing/generator/IK not neural layers.
Architecture-matched random expert controls use PyTorch standard initialization,
not claimed recreation of the original expert RL initializer.
-->

---

<!-- _class: compact -->

# What remains accessible relative to the teacher?

Selected first-layer held-out R²:

| Property | Expert | BC | Mixed DAgger | GAIL-DAgger | GAIL-BC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Command orientation | .906 | .999 | .998 | .995 | .999 |
| Phase × roll | .281 | .697 | .703 | .586 | .697 |
| Future forward velocity | .501 | .615 | .612 | .607 | .616 |
| Future direction error | .314 | .518 | .538 | .515 | .518 |
| Future pitch | .522 | .619 | .617 | .611 | .619 |

All 240 student-layer/property gaps favor students under this readout protocol.

Wider ReLU layers can make linear decoding easier; this does not establish better control.

<!--
Source expert INTERPRETATION.md, 4students x2layers x30properties=240 comparisons.
Students may preserve/reorganize information in a more linearly accessible form.
Expert trained for action production need not reconstruct sensor variables.
Avoid percentages of knowledge transferred or claims of historical causal copying.
-->

---

# Which expert features exceed random controls?

| Expert h1 property | Trained R² | Random tanh R² range |
| --- | ---: | ---: |
| Command orientation | .906 | .748–.830 |
| Phase-conditioned roll | .281 | .207–.251 |
| Future pitch | .522 | .495–.504 |

Shuffled-label scores are much lower for these properties.

**Phase–posture organization** therefore has support in both expert and students.

Other readable sensor quantities may be explained by input and architecture alone.

<!--
Shuffled expert orientation approximately-.011 to-.001,phase-roll .0002–.0283,
future pitch .068–.094. Many other expert targets similar/worse than random.
The matched random comparison helps distinguish learned organization from generic
nonlinear features; it still does not prove causal use in locomotion.
-->

---

# Whole-layer geometry shows different resemblance patterns

Linear CKA compares activation relationships across the same states.

| Representation | Expert CKA: h1 | Expert CKA: h2 |
| --- | ---: | ---: |
| BC | .9162 | .9150 |
| Mixed DAgger | **.9355** | .9050 |
| GAIL-DAgger | .9115 | **.9314** |
| GAIL-BC | .9161 | .9148 |
| Random ReLU ranges | .8923–.8984 | .8458–.8841 |

Raw observations already score **.8968**: common input structure matters.

<!--
Pooled centered linear CKA, no activation standardization. Mixedh1 and GAIL-DAggerh2
beat each same-layer random control in all15 runs. Within-run row permutation
learned means~.016–.019, an alignment diagnostic, not calibrated pvalue.
CKA is geometric resemblance, not percent retained knowledge or neuron matching.
High CKA can follow dominant variance and common inputs; arbitrary rescaling
changes it. Kornblith etal2019, reference in appendix.
-->

---

# Activation structure and layer-reset diagnostics

| Network | Effective variance dimension h1 / h2 |
| --- | ---: |
| Expert | 8.38 / — |
| BC / GAIL-BC | 7.81 / 8.28 |
| Mixed DAgger | 7.95 / 8.09 |
| GAIL-DAgger | 9.03 / 10.38 |

Resetting individual layers in copies changes latent actions by **~2.5–2.8 RMS**.

Variance dimension describes activation concentration; reset effects show
functional sensitivity. Neither measure identifies a particular skill.

<!--
Participation ratio=(sum covariance eigenvalues)^2/sum squared eigenvalues.
Expert width64,student256; values depend on recorded state distribution.
Randomh1 5.26–5.37,h2 6.08–7.42. Near-constant h2 fractions BC/GAILBC11.33%,
mixed8.59%,GAILDag9.77%;expert0%. Not permanently dead units or pruning advice.
36interventions=4sources x3affine layers x3resetseeds, copies only, never deployed.
Exact reset means input/middle/output:BC2.642/2.709/2.638,mixed2.828/2.733/2.845,
GAILDag2.547/2.594/2.600,GAILBC2.640/2.708/2.637. Broken interfaces/scaling can
explain changes. This does not localize phase-roll computation.
-->

---

# Limits that shape the next experiments

- **Periodic controls:** shuffled phase/contact labels retain gait alignment.
- **Future baselines:** persistence is weak for averaged or oscillatory targets.
- **Probe capacity:** width/nonlinearity differ between expert and students.
- **Coverage:** shared oracle states, one source checkpoint per technique.
- **Regularization:** some targets still choose the weakest tested penalty.

Useful follow-ups: stronger phase controls, capacity checks and nonlinear
regression on a separately versioned protocol.

<!--
BC shuffled phase sin/cos .819/.870 vs true.983/.989; future contact shuffle
retains substantial signal. Persistence future velocity R2-2.589,direction-3.008;
beating it does not prove anticipation. General89/480 true scalar probes choose
alpha1e-6;expert6/30. Do not retune against already inspected test scores.
A phase-shifted or phase-matched residual control needs validation itself.
No proposal here is a completed result or authorization to launch research fits.
-->

---

# Meeting conclusions and discussion

1. Context modifies actions, but gave little consistent practical benefit here.
2. Probe results identify several accessible locomotion properties.
3. Phase–posture combinations provide the clearest learned-over-random evidence.
4. Mixed DAgger and GAIL-DAgger resemble the expert at different depths.
5. BC and GAIL-BC remain similar across measured feature profiles.

**Next question:** which of these properties contributes to actual control?

R1 transfer remains deferred; current conclusions concern NAO representations.

<!--
No exhaustive internal feature inventory, causal balance claim or R1 ranking.
Context DAgger/GAIL expansion remains paused; joint impairments remain future.
Existing ridge readouts already regress continuous locomotion quantities.
Potential interventions must separate meaningful disruption from generic interface
damage. This update revises presentation emphasis without changing the study.
-->

---

<!-- _class: references -->

# Evidence and methods

- [Context results](../../experiments/context_conditioning/bc_input_v1/evaluation_all_v1/report.md) · [repeatability](../../experiments/context_conditioning/bc_input_v1/repeatability_v1/repeatability.md) · [FiLM results](../../experiments/context_conditioning/bc_film_v1/evaluation_append_all_v1/comparison.md)
- [General probe interpretation](../../experiments/representation_probing/nao_sources_v1/INTERPRETATION.md) · [expert interpretation](../../experiments/representation_probing/nao_expert_comparison_v1/INTERPRETATION.md)
- [Alain & Bengio: linear probes](https://arxiv.org/abs/1610.01644)
- [Hewitt & Liang: probe controls](https://aclanthology.org/D19-1275/)
- [Kornblith et al.: representation similarity / CKA](https://proceedings.mlr.press/v97/kornblith19a.html)
- [Perez et al.: FiLM](https://arxiv.org/abs/1709.07871)
- [Full chronological brief](../MEETING_SUMMARY_2026-10-10.md) · [commands and demos](../../COMMANDS.md)

<!--
Local links resolve from this source file; exported files need repository access
to navigate local reports. Literature motivates methodology; locomotion readouts,
controls and findings are project-specific adaptations. Prior deck archived at
../meeting_archive/MEETING_SUMMARY_MARP_before_new_work_focus_2026-10-12.md.
No experimental results, checkpoints or fits changed. Marp conversion is user-run.
-->
