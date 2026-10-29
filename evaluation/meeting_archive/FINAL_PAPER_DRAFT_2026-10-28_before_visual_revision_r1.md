# Imitating RoboCup locomotion: fidelity, stability and generalization across surface dynamics

Working paper draft and candidate argument, 18 October 2026. Target: eight body
pages, references excluded. This is a synthesis of the completed consolidated
experiments, not a claim of a new imitation-learning algorithm. Numerical evidence,
full tables and exact artifact provenance remain in
[the consolidated report](CONSOLIDATED_RESULTS_REPORT_2026-10-15.md).
The optional wide-slip/RL study is explicitly excluded. No results from that
future study should be inserted as completed evidence.

## Abstract — candidate text

Imitation learning offers a practical route to locomotion control when an expert
controller is available, but reproducing task performance need not imply faithful
reproduction of expert actions. We compare behavioural cloning (BC), mixed
dataset aggregation (DAgger), and generative adversarial imitation learning
(GAIL) initialized from each supervised policy in RoboCup 3D simulation. Using
the native FC Portugal walking controller as an oracle, we evaluate task
progress, action fidelity, joint-state occupancy and mechanical stability in
the original environment and under controlled surface-slip changes. We then
compare varied-domain continuation, varied-only training and explicit slip
conditioning. The combined study contains 56 trained checkpoints and 1,949
evaluation deployments, including reused controls. All four original supervised-
initialized policies achieve approximately oracle-level progress, while their
action errors differ substantially. Mixed DAgger provides stronger action
fidelity than BC, including when trained directly on varied surfaces. Original-
environment pretraining followed by continuation yields substantially lower
action error than varied-only training, although scratch BC achieves slightly
greater progress. Direct random-initialized GAIL fails to acquire useful walking
under the tested configuration, and explicit context produces small, inconsistent
benefits. Representation probes further show that accessible locomotion
information and expert feature similarity are not reliable substitutes for
behavioural evaluation. These results support evaluating imitation as a
multi-objective trade-off rather than ranking policies by return alone.

## 1. Introduction and research questions

Stable bipedal locomotion combines sequential decision-making with sensitive
contact dynamics. An available expert allows a learned policy to acquire this
behaviour without independently discovering every aspect of gait control.
However, offline prediction error, closed-loop locomotion and mechanical
stability measure different properties. A policy may reproduce the expert's
distance travelled while issuing noticeably different actions, or exhibit low
jerk because it barely moves.

This study examines those distinctions in the FC Portugal NAO environment.
BC supplies an offline supervised baseline. DAgger adds oracle labels on
learner-visited states. GAIL uses a discriminator-based objective and simulator
interaction, with supervised initialization and stabilization in our implementation.
We ask:

1. **Stability:** How do the implemented imitation methods compare in fall
   incidence and kinematic smoothness while performing the same walking task?
2. **Resources:** What task performance and imitation fidelity do they attain
   at the tested data and interaction budgets, and what supervisory and
   optimization costs underlie those results?
3. **Expert performance:** Can learned policies attain oracle-level progress,
   velocity and target tracking, and is that accompanied by faithful imitation?

Controlled surface changes extend these questions to frozen and adapted
policies. Initialization and context comparisons help distinguish additional
training history from environmental conditioning. Probing is a supporting
analysis, not a separate claim of causal interpretability or cross-robot transfer.

The contributions are an auditable controlled comparison, evidence that
performance and fidelity can rank policies differently, and a characterization
of initialization/context effects under the tested budgets. Infrastructure is
necessary for reliable evidence but is not presented as a novel learning method.

## 2. Related work — compact framing

Use the full sources and attribution boundaries in [Appendix E of the report](CONSOLIDATED_RESULTS_REPORT_2026-10-15.md#appendix-e-paper-sources-and-final-write-up-citation-guide).
The core citations are Ross et al. (2011) for DAgger; Ho and Ermon (2016) for
GAIL; Schulman et al. (2017) and Schulman et al. (2015/2016) for PPO and GAE;
and Orsini et al. (2021) for adversarial imitation implementation choices and
random/expert normalization. Dynamics/context motivation can cite Peng et al.
(2018) and Beukman et al. (2023), without claiming to reproduce their architectures.
For the short probe section, cite Alain and Bengio (2016), Hewitt and Liang
(2019), and Kornblith et al. (2019). Do not spend scarce body space on a general
tutorial on reinforcement learning.

## 3. Experimental design

### 3.1 Controller and action interface

The native FC Portugal controller is an unchanged expert oracle. Learned
policies consume 63 walking-observation features and output 16 latent gait
actions, which feed the existing stateful gait-generation/IK machinery. These
are **not direct commands for 22 independent joint velocities**. Input-conditioned
policies receive one additional normalized slip feature. The retained gait
interface limits the scope of the comparison: learned walking is supported by
existing control structure rather than unrestricted joint-level gait discovery.

Simulation is synchronous. Targets remain within the permitted half-field.
Evaluation uses 3,000 active steps at 20 ms per step, corresponding to 60 active
seconds. A common fall detector triggers reset and stabilization; recovery is
excluded from active-step accounting. This assesses detected balance losses,
not autonomous recovery from collisions. Managed servers and configuration
verification prevent uncontrolled scene changes between experimental conditions.

### 3.2 Learning methods and resources

The four principal methods are BC, mixed DAgger, GAIL-BC and GAIL-DM. DM denotes
initialization from the mixed DAgger policy, not the older original-DAgger
checkpoint used in historical experiments. GAIL always uses FC Portugal
demonstrations as its expert data; its initializer is not a replacement expert.

| Method | Core training allocation | Additional accounting needed |
|---|---|---|
| BC | 170,000 expert examples | Epochs and supervised optimizer exposure |
| Mixed DAgger | 100,000 seed examples + 70,000 learner-state labels | Oracle queries, learner steps and repeated aggregate training |
| GAIL-BC / GAIL-DM | 170,000-example static pool + 70,000 PPO steps | Supervised initializer cost and adversarial optimization |

These allocations standardize selected resources, not total compute. BC uses
17 million supervised sample exposures and 66,500 gradient updates; mixed
DAgger uses 54 million exposures and 211,050 updates under this schedule.
GAIL inherits its initializer's cost and retains a supervised anchor and
conservative update safeguards. Claims should concern these implemented
variants, not all possible settings of each algorithm.

### 3.3 Experimental progression

1. Train four original-environment policies and compare them with oracle and
   random baselines over five task seeds.
2. Freeze those policies and evaluate them on controlled scalar-slip surfaces.
3. Continue training on a slip grid of 0.025, 0.1, 0.2, 1 and 10, using hidden
   and input-conditioned architectures with three continuation seeds.
4. Train the corresponding varied-only lineages from scratch and compare them
   with continuation. Add direct-random GAIL in the original and varied-hidden
   environments as separate initialization conditions.

Varied evaluation contains 15 grouped Gaussian draws, 15 confirmation draws
and three scalar-0.1 rounds. Gaussian parents have means at the training grid,
standard deviations half their means, and truncation to (0,10]. One sampled
slip is fixed throughout each run; controllers share the slip/task round.
This is not spatially random terrain or per-contact slip resampling. The
original installed scene and scalar-0.1 condition must remain distinct.

The combined dataset contains 56 checkpoints and 1,949 deployments. The scratch
extension reuses 1,020 parent deployments and adds 929; reused baselines are
not new independent measurements. Original policies have one training seed,
while continuation replicas share pretrained parents. The paired design helps
comparison but does not eliminate training-history or simulator variability.

### 3.4 Measurements

Report target progress and random/expert-normalized return alongside velocity
and tracking error. Score zero denotes the random reference, not an absolute
floor. Imitation measures are held-out latent-action element MSE, a separately
labelled on-policy oracle-action discrepancy, and sliced Wasserstein distance
between standardized 22-joint angle occupancies. The latter is not exact
multivariate trajectory transport and discards temporal ordering.

Mechanical measures include falls per active minute, torso RMS angles and
integrated/rate joint jerk after the documented smoothing and reset exclusions.
Jerk is kinematic smoothness, not a measured torque, energy or hardware-wear
quantity. Report both desirable and failure behaviours: low jerk without useful
progress is not successful locomotion.

## 4. Results — candidate numerical core

### 4.1 Similar original-environment performance conceals fidelity differences

| Controller | Mean target progress (m) | Normalized score | Held-out action MSE |
|---|---:|---:|---:|
| FC Portugal | 44.446 | 1.0000 | — |
| BC | 44.630 | 1.0044 | 0.11047 |
| Mixed DAgger | 44.364 | 0.9985 | 0.01535 |
| GAIL-BC | 44.556 | 1.0025 | 0.11042 |
| GAIL-DM | 44.150 | 0.9934 | 0.01572 |

All four learned policies record zero falls across their five original-scene
deployments. Their small progress differences do not establish general
superiority over the expert. In contrast, mixed DAgger's held-out action error
is roughly seven times lower than BC's. GAIL adds little final fidelity benefit
relative to its initializer in this configuration.

**Candidate conclusion:** matching the expert's task outcome is substantially
easier here than matching its action decisions. The experiment does not support
the assumption that BC necessarily collapses from compounding errors in this
particular supported walking task.

### 4.2 Changed surfaces reveal costs hidden by normalized return

Across the 30 sampled rounds, oracle progress averages approximately 32.885 m,
lower than in the original scene. Frozen original policies retain normalized
scores near one, but BC records 11 falls and mixed DAgger six across 30 active
minutes. Frozen BC's on-policy action MSE is approximately 0.291, versus 0.0265
for DAgger. Return normalization therefore needs absolute performance and
stability beside it: oracle parity can persist while everyone slows down.

Varied continuation substantially improves fidelity. Hidden BC test MSE becomes
0.01995; hidden DAgger reaches 0.00187. Their sampled progress remains about
32.8 m. This is evidence for useful adaptation of imitation accuracy, but
additional training and exposure to varied dynamics are confounded without an
equally trained nominal-only control.

### 4.3 Initialization changes fidelity more clearly than progress

Representative hidden-context families, each averaged over three seeds and
30 sampled rounds:

| Method | Continuation progress m | Scratch progress m | Continuation test MSE | Scratch test MSE |
|---|---:|---:|---:|---:|
| BC | 32.784 | 33.192 | 0.01995 | 0.11177 |
| Mixed DAgger | 32.845 | 32.917 | 0.00187 | 0.01035 |
| GAIL-BC | 32.838 | 33.237 | 0.02015 | 0.11199 |
| GAIL-DM | 32.851 | 32.912 | 0.00217 | 0.01061 |

Scratch BC gains about 0.408 m per deployment while its action error is about
5.6 times greater. Mixed DAgger still gives about 10.8-fold lower test MSE than
scratch BC, showing that its advantage is not solely inherited from original-
environment pretraining. However, longer supervised optimization is still a
competing explanation for part of the BC–DAgger gap.

Continuation improves fidelity but is not a universal progress or stability
winner. For hidden BC, both cohorts have four falls across 90 active minutes.
For hidden DAgger, continuation has five and scratch four. Sparse fall counts
should not be overinterpreted as strong population-level rankings.

Direct-random GAIL is qualitatively different: original-scene mean progress is
−0.121 m with no targets reached, while varied-trained random GAIL averages
−0.367 m over sampled conditions, also with no targets reached. Its normalized
score is approximately −0.0157. This configuration does not learn useful
locomotion at 70,000 PPO steps despite retaining the supervised anchor. This
is not a universal impossibility result for random-start GAIL.

### 4.4 Explicit context does not consistently improve control

For scratch models, input-minus-hidden mean sampled progress is +0.082 m for
BC, −0.121 m for mixed DAgger, +0.063 m for GAIL-BC and +0.053 m for GAIL-DM.
Continuation similarly gives small and inconsistent differences. These data
do not establish a general benefit from supplying slip explicitly.

The expert lacks an explicit slip input, but this alone is not an explanation:
its observations still respond to changed dynamics. Without incorrect/constant-
context interventions, marginal gains cannot distinguish unused context from
redundant information or insufficient benefit for the chosen task.

### 4.5 Probing supports interpretation, not causal explanations

The combined analysis comprises 524 multitarget ridge fits over 131
representations and 30 locomotion properties. Probes use common oracle-state
trajectories, episode-level splits, raw-input/random-feature controls and
shuffled-label diagnostics. Selected quantities include phase, support-force
proxies, posture, phase–posture interactions and 200-ms future motion.

For hidden DAgger, phase-conditioned roll R² is 0.7065 after continuation and
0.6935 after scratch training. Yet scratch has higher future-pitch R²
(0.6101 versus 0.5876) and higher expert CKA (0.9428 versus 0.9212), despite
worse action fidelity. Varied random-start GAIL has future-velocity R² near
0.6114, close to random-network features, while failing to walk usefully.

**Candidate conclusion:** decodability and representational resemblance are
not interchangeable with action fidelity or successful closed-loop control.
Differences in expert/student architecture and the use of oracle-state data
further constrain interpretation. CKA is not percentage knowledge retained.

## 5. Discussion and answers to the research questions

**RQ1 — stability:** viable supervised-initialized policies are generally
stable, but there is no dependable universal GAIL advantage in falls or jerk.
Read mechanical measures alongside progress to avoid treating inactivity as
smooth locomotion. Sparse events and limited independent training seeds bound
the strength of stability claims.

**RQ2 — resources:** this study establishes outcomes at specified data and
interaction budgets, with substantial differences in optimizer exposure and
inherited cost. It does not establish minimum sample complexity or universally
faster convergence. If comparable intermediate-checkpoint evaluation curves
are not added, use the term **budgeted resource comparison**, not a proved
convergence-efficiency ranking. This is narrower than the proposal's planned
convergence analysis and should be stated explicitly.

**RQ3 — expert performance:** learned supervised-initialized policies attain
approximately oracle-level task performance within the tested interface and
conditions. Small gains in progress do not imply better imitation or robust
superiority. DAgger's principal advantage is action fidelity, not a universal
return advantage.

The study's central result is the separation of performance, fidelity and
representation similarity. Additional original-domain training improves fidelity
without consistently improving progress; environmental context is not reliably
beneficial; and adversarial refinement remains dependent on initialization
under the chosen settings. These findings are useful empirical constraints,
not evidence that any algorithm is generally unnecessary or universally best.

## 6. Limitations and conclusion

Limitations include single-seed original training, shared continuation parents,
unmatched lifetime optimization, reused evaluation controls, correct-context-only
tests, reset-assisted deployment and the retained gait/IK structure. Varied
policies were not evaluated in the preserved original scene, so scalar-0.1
results are not original-environment retention. Surface slip remains fixed
within runs; collisions, motor faults, contact-wise randomness and other robots
are outside the evidence. Historical experiments with unverified effective
slip must not be pooled with the corrected scalar intervention.

The completed experiments support a focused conclusion: in this RoboCup
locomotion setting, successful task execution can coexist with substantially
different imitation fidelity. Mixed DAgger yields consistently lower action
error, GAIL's additional benefits are limited and initialization-dependent, and
explicit context does not consistently improve locomotion. Representation
analysis reinforces why no single metric should substitute for behavioural
and mechanical evaluation.

## Editorial plan: what belongs in eight pages

| Section | Body pages |
|---|---:|
| Introduction and questions | 0.75 |
| Related work | 0.75 |
| Methods and design | 1.5 |
| Original comparison | 1 |
| Environmental generalization/context | 1.5 |
| Initialization | 1 |
| Selected representation results | 0.75 |
| Discussion/conclusion | 0.75 |

References are outside the eight-page limit. Select at most a few combined
figures/tables: original fidelity/performance; slip response; paired
initialization/fidelity; one small probe comparison. Keep complete metric and
seed tables in the consolidated report or permitted supplementary material.
Describe superseded implementations only where needed to explain validity.

Do not add the optional wide-slip/RL exploration to this paper by default.
Do not claim physical energy savings, causal representation transfer,
statistical superiority from tiny mean differences, or recovery ability from
reset-based tests. Supervisor review should settle the narrower RQ2 framing
before the final submission.
