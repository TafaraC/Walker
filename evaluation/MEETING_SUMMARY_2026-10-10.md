# Meeting summary: nominal imitation, domain studies and representations

Updated 2026-10-11. This is the consolidated current meeting brief, covering work
from the original-environment comparison of BC, mixed DAgger, GAIL-DAgger and
GAIL-BC through completed context and expert probing studies. Original DAgger
and random-start GAIL are included as explanatory controls.

## Summary for discussion

- All four selected policies reproduced nominal walking performance near oracle
  parity; BC offered the strongest fidelity/simplicity trade-off.
- Larger, mixed long-run/short-turn data improved DAgger substantially, but data
  quantity and composition changed together.
- GAIL largely retained its initializer's fidelity characteristics; random-start
  GAIL did not bootstrap walking under the tested budget.
- Early slip experiments exposed a manipulation defect. Their artifacts remain
  useful historical records, but cannot demonstrate effective-slip robustness.
- Corrected scalar slip changes solver contact parameters and reduces walking
  progress at high settings. Varied-trained policies generally track the oracle
  across sampled fixed-slip rounds, without an established algorithmic winner.
- Explicit input context and FiLM did not establish consistent practical benefits
  in the tested BC setting. Context DAgger/GAIL expansion is therefore paused.
- General and expert probes reveal differences in accessible locomotion features
  and representation geometry, not causal control mechanisms or transfer benefits.
- R1 transfer is deferred pending progress in the separate workstream.

## 1. Original-environment comparison

Five evaluation seeds, 3,000 active synchronous steps (60 seconds) per policy.
Targets stayed within the player's half; startup and recovery were outside active
time. Normalized return uses same-seed random and oracle baselines.

| Selected policy | Normalized return, mean [95% CI] | Expert-test latent Action MSE | On-policy oracle MSE | Occupancy Wasserstein | Falls/min |
| --- | --- | ---: | ---: | ---: | ---: |
| BC | 1.101 [0.906, 1.296] | .0084 | .00437 | .0906 | 0 |
| Mixed DAgger | 1.096 [0.902, 1.290] | .0612 | .00464 | .0808 | 0 |
| GAIL-DAgger | 1.106 [0.910, 1.302] | .2843 | .09880 | .1072 | 0 |
| GAIL-BC | 1.104 [0.902, 1.307] | .0085 | .00444 | .0817 | 0 |

Intervals overlap oracle parity and one another; no superiority claim follows.
Oracle experienced one fall in five minutes. Scores above one partly reflect
baseline variability, not necessarily better learned control.

Mixed DAgger reduced original DAgger's expert-test MSE from .2848 to .0612
(~78.5%) and on-policy MSE from .09217 to .00464 (~95%). It used 100,000
balanced long/short seed examples plus 70,000 rollout labels, versus original
DAgger's 20,000 seed examples plus 70,000 rollout labels. Size and coverage are
confounded; the study does not isolate the effect of turning examples.

BC used 90,000 static training labels. GAIL used 40,000 additional environment
steps and the static 90,000-example FC Portugal dataset, plus inherited
initializer costs. Its zero *active* oracle-query count is not zero supervision.
GAIL-DAgger means initialization from **original DAgger**, not mixed DAgger;
FC Portugal remains the GAIL expert.

Random-start GAIL had normalized return -.011, forward velocity -.005 m/s and
zero targets: low falls/jerk reflected stationary failure, not successful walking.
The source policy's 16 outputs are latent Walk residuals processed by a stateful
gait generator and IK, not direct joint velocities or torques.

Source: [nominal results report](PHASE1_RESULTS_REPORT.md).

## 2. Early fixed-slip and Gaussian studies: preserved, but qualified

The initial isolated-server fixed sweep completed 120 runs: eight controllers,
three configured slips (.025/.100/.200), five seeds each. Selected policies
retained approximately 99–102% nominal progress with little separation.

The first per-run Gaussian experiment drew five settings and held each setting
constant for a run. Frozen baseline plus nominal/varied continuation evaluations
produced 65 verified receipts and eight final continuation checkpoints.
Varied-minus-nominal progress differences were BC +.097 m, mixed DAgger -.077 m,
GAIL-BC +.062 m and GAIL-DAgger -.224 m; exploratory five-pair intervals all
included zero. BC and mixed DAgger improved on-policy imitation error similarly
under both continuation arms.

**Subsequent physics auditing changed the interpretation.** The old generated
scene used `(setContactSlip v v)`; requested changes, even large ones, produced
the same logged solver-slip classes. Source/import/hash checks had verified
configuration, not effective contact dynamics. These batches cannot establish
robustness or adaptation to varied effective slip. Preserve datasets, checkpoints
and reports as legacy-pathway continuation evidence; do not pool them with
corrected experiments or reuse their normalization baselines.

Historical sources:
[fixed interpretation](../experiments/phase2_domain_robustness/FIXED_RESULTS_INTERPRETATION.md)
and [cross-phase interpretation](../experiments/variable_slip_training/gaussian_v1_seed42/CROSS_PHASE_INTERPRETATION.md).
Their recommendations/status describe the time of writing, not current next tasks.

## 3. Slip diagnosis, corrected method and server controls

The single-argument expression `(setContactSlip v)` changed enabled solver-bound
slip parameters. In five oracle seed-101 smoke runs:

| Requested field slip | Progress (m) | Retention relative to corrected .1 | Falls |
| --- | ---: | ---: | ---: |
| .025 | 43.972 | 108.4% | 0 |
| .100 | 40.546 | 100.0% | 0 |
| .200 | 37.591 | 92.7% | 0 |
| 1 | 28.809 | 71.1% | 0 |
| 10 | 15.325 | 37.8% | 0 |

This validates a responsive manipulation in those runs, not a population effect
size or universal safe range. Solver parameters are not measured foot sliding.
The exact legacy binding/fallback mechanism was not independently established.
Original nominal configuration is not automatically equivalent to corrected .1.

Corrected frozen evaluation tooling expanded to .025/.1/.2/1/10, three seeds and
eight controllers (120 planned deployments), under
`experiments/phase2_domain_robustness/scalar_wide_v4/`. A consolidated corrected
fixed-grid result table was not found in that directory during this update;
do not substitute the legacy fixed report or claim a newly verified full-grid
ranking from it. The scalar smoke and grouped evaluation below provide the
quantitative corrected evidence summarized here.

SimSpark monitor backpressure and socket teardown caused server crashes between
deployments. Managed pipelines now use fresh servers per deployment, isolated
local scenes, verified identities, failure preservation and receipt-based resume.
Infrastructure failures are not policy falls. These lifecycle changes also limit
causal comparisons with earlier shared-server/original batches.

Source: [scalar validation](../experiments/slip_sensitivity/oracle_scalar_v1/INTERPRETATION.md).
Implementation/failure history: [PIPELINES](../PIPELINES.md), [CHALLENGES](../CHALLENGES.md).

## 4. Corrected grid training and grouped Gaussian evaluation

The corrected varied training arm uses fixed slips .025/.1/.2/1/10. Its paired
nominal-continuation arm is training at .1, not merely a nominal test deployment.
Evaluation samples once per round and holds slip fixed: it is **not** an
independent Gaussian draw at each contact or a spatially varying Gaussian field.

Completed grouped evaluation: 15 rounds, five controllers each, 75 deployments
of 3,000 active steps. Five Gaussian parent groups are centered at the training
values with standard deviation half their means, truncated to (0,10]; three draws
per group. Every controller shares that round's slip and task seed, with a fresh
server. All 750 artifacts referenced by completion receipts were hash-verified
in the recorded completion review.

| Controller | Mean progress (m) | Mean per-round oracle ratio | Falls over 15 runs |
| --- | ---: | ---: | ---: |
| Oracle | 33.364 | 1.000 | 0 |
| BC varied | 33.292 | 1.005 | 0 |
| Mixed DAgger varied | 33.078 | .991 | 1 |
| GAIL-BC varied | 32.937 | .988 | 1 |
| GAIL-DAgger varied | 32.684 | .985 | 4 |

Absolute progress falls as surfaces become more challenging: oracle group means
42.091 m at the lowest-centered group and 16.767 m at the highest. The policies
mostly track that decline. GAIL-DAgger has more observed falls, particularly in
the group centered at 1; only three draws per group support descriptive evidence,
not a reliable algorithm-wide stability ranking.

On matched held-out varied expert data, latent element MSE is BC .0018185,
mixed DAgger .00110956, GAIL-BC .00725441 and GAIL-DAgger .232627. Similar task
progress can coexist with very different local imitation errors.

These are **oracle ratios**, not random-normalized scores. A mean of per-round
ratios is not the ratio of aggregate means. This grouped report contains the
varied policies and oracle, not a matched frozen/nominal-continuation comparison;
it alone cannot show that varied training caused an improvement. Deployment
rounds are not independent training replicas.

Source: [75-run grouped report](../experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/evaluation_grouped_v1/report.md).

## 5. Explicit context BC, repeatability and FiLM

Hidden-context BC and slip-input BC were fine-tuned on the same collected data,
initializer and budget, with training seeds 42/43/44. These are three separately
deployed replicas per architecture, not an ensemble or three independent source
datasets. Context evaluation completed 429 deployments including correct,
constant and wrong context; FiLM appended 99 correct-context deployments on the
same rounds, reusing verified controls rather than repeating them.

Mean target progress per 60-active-second deployment:

| Set | Hidden BC | Input-conditioned BC | FiLM BC |
| --- | ---: | ---: | ---: |
| Original sampled slips | 33.073 | 33.060 | 33.232 |
| Confirmation sampled slips | 32.292 | 32.665 | 32.581 |
| Nominal | 39.272 | 39.175 | 39.219 |
| Falls across 99 runs per architecture | 5 | 4 | 5 |

No consistent practical advantage was established. Input perturbations changed
actions, so context was not entirely ignored, but wrong/constant context did not
consistently reduce live progress. FiLM's held-out MSE gain over hidden BC was
~.18%; joint-occupancy sliced Wasserstein was worse on both sampled suites, with
broadly similar jerk. One +14.075 m case dominates its confirmation gain;
the other 44 pairs average approximately -.0234 m. Retain that case in results.

The 45-run repeatability diagnostic found same-controller progress SD .24–.62 m
and unequal first-active physical states despite identical initial targets.
This cautions against attributing small differences to conditioning; it is not
a universal significance threshold. Reused confirmation draws are not untouched
FiLM tests. Correct-context-only FiLM does not isolate context use from capacity.

Plausible explanation: the teacher observes physical consequences but receives
no explicit slip input; imitation rewards reproducing its actions rather than
finding a better surface-specific strategy. This is a hypothesis, not proof that
context cannot help. Context-conditioned DAgger/GAIL remain **untested**, with
expansion stopped as a research-priority decision.

Sources under `experiments/context_conditioning/`:
[context comparison](../experiments/context_conditioning/bc_input_v1/evaluation_all_v1/report.md),
[repeatability](../experiments/context_conditioning/bc_input_v1/repeatability_v1/repeatability.md),
[FiLM comparison](../experiments/context_conditioning/bc_film_v1/evaluation_append_all_v1/comparison.md).

## 6. Internal representations: preliminary, general and expert studies

All source policy weights remain unchanged. Main sources are BC, mixed DAgger,
original-DAgger GAIL and GAIL-BC. Probes use shared oracle observations, whole-run
splits, raw-input/random-network baselines and shuffled-label controls.

### Preliminary slip-focused study — complete

60 fits / 300 scalar scores. First-layer slip R2: BC .440, mixed DAgger .443,
GAIL-DAgger .461, GAIL-BC .440. Random-network controls already score .404–.424,
so much of decoding is not specifically learned. Directly observed posture is a
retention check, not discovery of a hidden physical concept. Some shuffle controls
retain signal; 74/75 true-label probes hit the minimum ridge penalty.

### Broad locomotion study — complete

64 multitarget fits / 1,920 scalar scores, 30 gait/body/force/command/future
properties, plus 36 in-memory layer-reset action-sensitivity diagnostics.

Phase-conditioned roll is the strongest learned-over-random candidate: h1 R2
BC .697, mixed DAgger .703, GAIL-DAgger .586, GAIL-BC .697, versus raw .332 and
random .427–.452. Each learned h1 beats each random control on all 15 test runs.
GAIL-DAgger h2 retains more gyro-x/y information; mixed DAgger modestly leads
future-direction-error decoding. BC/GAIL-BC profiles are very similar.

Periodic phase/contact signal survives block shuffles, weakening that null.
Future persistence controls are weak for averaged/oscillatory targets. Reset
effects (~2.5–2.8 latent-action RMS) show source action sensitivity, not which
property causes balance. No universal best representation is established.

### Expert comparison — complete

16 appended expert/random-tanh fits, 2,400 combined scalar scores, 240
expert/student comparisons and 20 geometric profiles. Oracle output agreement
was checked on 90,000 states. Expert: 63→64 tanh→16; students:
63→256 ReLU→256 ReLU→16. The external gait generator/IK is outside these probes.

Mixed DAgger h1 (pooled CKA .9355) and GAIL-DAgger h2 (.9314) are closest to
expert h1. Raw observations already reach .8968, with high random-network CKA.
These are geometric similarities, not percentages of transferred knowledge.

All 240 student-layer/property R2 gaps favor students under this protocol;
width/nonlinearity confounds prevent interpreting this as superior knowledge.
The expert beats its matched random controls on command orientation,
phase-conditioned roll and future pitch. Phase/posture organization is therefore
a convergent teacher/student candidate, not proven historical or causal transfer.

Sources:
[preliminary interpretation](../experiments/probing/original_policies_v1/INTERPRETATION.md),
[general interpretation](../experiments/representation_probing/nao_sources_v1/INTERPRETATION.md),
[expert interpretation](../experiments/representation_probing/nao_expert_comparison_v1/INTERPRETATION.md).
Methods: [Alain & Bengio](https://arxiv.org/abs/1610.01644),
[Hewitt & Liang](https://aclanthology.org/D19-1275/), and CKA methodology documented
in [EXPERT.md](../representation_probing/EXPERT.md). These references motivate
analysis; the locomotion adaptation and empirical findings are this project's.

## 7. Current decisions and next discussion

Consolidate the NAO evidence: nominal fidelity/cost, corrected environment
sensitivity, limited context benefit, and expert/student feature accessibility.
Do not claim exhaustive representation discovery, causal adaptation or R1 utility.

R1 transfer is excluded from current claims until the separate work progresses.
This is a scope decision, not a negative-transfer result. Joint/motor impairments,
context DAgger/GAIL, nonlinear regression probes and stronger periodic-null
controls remain potential follow-ups, not completed experiments.

The current continuous locomotion readouts already use ridge **regression**;
a nonlinear regression extension would be a new, versioned study.

Commands including demos are in [COMMANDS.md](../COMMANDS.md). No experiment was
rerun to prepare this summary.

## Meeting-history preservation

Before revising this current brief, archive its complete previous contents under
`evaluation/meeting_archive/` using a unique date/revision filename. Never overwrite
an archived snapshot. Keep obsolete plans in the archive rather than interleaving
them with current conclusions. Experimental source reports remain authoritative
and must not be rewritten to match a later narrative.

The full pre-consolidation brief is preserved at
[2026-10-11 snapshot](meeting_archive/MEETING_SUMMARY_2026-10-10_before_consolidation_2026-10-11.md).
Earlier historical versions are not reconstructed or claimed to have been archived.
