# Imitating RoboCup locomotion: optimization, fidelity and stability across surface dynamics

<!-- Updated 28 October 2026. Working paper: target eight body pages, references excluded.
Canonical editable source for paper/main.tex. Figures are generated from saved JSON.
Author and institution intentionally unset. Final template/page count require compilation. -->

## Abstract

Matching an expert's walking performance need not imply reproducing its actions.
We compare behavioural cloning (BC), mixed dataset aggregation (DAgger), and
supervised-initialized, BC-regularized generative adversarial imitation learning
(GAIL) using the native FC Portugal controller in RoboCup 3D simulation.
Experiments progress from the original scene to frozen-policy surface tests,
varied-surface training, explicit slip conditioning and varied-only initialization.
Evaluation combines target progress, latent-action error, joint occupancy and
mechanical stability. Initial results favoured DAgger's action fidelity, but
matching BC to DAgger's 211,050 optimizer updates reverses the original-scene
ordering: held-out mean squared error becomes 0.01024 for BC versus 0.01535 for
DAgger, compared with 0.11047 for the initial BC allocation. Increasing varied-only
BC training also narrows, without eliminating, the fidelity advantage associated
with original-domain pretraining. A zero-native-action baseline fails to make
useful progress despite retaining the gait engine. Explicit context offers no
consistent broad progress advantage, and the tested GAIL refinement changes
little from its initializer. Representation probes provide descriptive support,
not causal explanations. The principal finding is that task parity, action
fidelity and stability must be reported separately, and that fidelity rankings
depend materially on optimization allocation.

## 1. Introduction

Bipedal locomotion requires sequential control under sensitive contact dynamics.
An available expert makes imitation practical, but successful task execution
does not establish that a student has retained the expert's decisions. Low
offline error, stable deployment and similar joint motion are related but
distinct objectives. Conversely, an inactive controller may have low jerk
without performing useful locomotion.

We investigate these distinctions in FC Portugal's simulated NAO walking system.
The study is a controlled empirical comparison, not a new imitation-learning
algorithm. Its research questions are: (RQ1) how stable are the learned
controllers during useful walking; (RQ2) what fidelity and performance do their
stated supervisory, interaction and optimization allocations buy; and (RQ3) how
closely do they reproduce expert task performance and actions? Surface
variation, initialization and explicit context extend these questions.
A small representation analysis supports interpretation of the behavioural
results; cross-robot transfer is outside the evidence.

The contribution is an auditable staged study with follow-up controls that
change the interpretation of its initial results. Rather than retaining a
favourable algorithm ranking after discovering unequal optimization, we test
that confound. Matching updates reverses the original BC–DAgger fidelity
ordering, while leaving a different ordering in joint occupancy and progress.

## 2. Background

DAgger addresses sequential distribution shift by querying an expert at
learner-visited states and aggregating those labels with the training data
[@ross2011]. This motivates comparing held-out expert-state error with
closed-loop error, but does not make a rise in error a unique diagnosis of
covariate shift. The environment and difficulty of visited states also change.

GAIL learns from demonstrations through an adversarial objective
[@ho2016]. Practical adversarial imitation depends on implementation choices
[@orsini2021]. Our actor uses PPO [@schulman2017], supervised initialization,
a BC anchor and conservative update safeguards. We therefore refer to this arm
as BC-regularized GAIL refinement and limit conclusions to this configuration.
Neither a small learning rate nor a KL constraint alone invalidates adversarial
imitation; it is the complete tested recipe and allocation that bound the claim.
We do not infer from 170,000 correlated samples that demonstrations are
sufficient, or that BC generally matches GAIL when data are plentiful.

Probing examines whether information is accessible from frozen features
[@alain2016], with control tasks needed to constrain interpretation
[@hewitt2019]. Centered kernel alignment (CKA) compares representation geometry
on common observations [@kornblith2019]. These measurements do not directly
identify the features a controller causally uses.

## 3. Experimental design

### 3.1 Expert, interface and simulation

The unchanged native FC Portugal walking controller supplies demonstrations and
on-demand labels. The underlying skilled locomotion system is described by
Abreu et al. [@abreu2025]. Each ordinary student consumes 63 observation
features and produces 16 native gait actions. A stateful gait generator and
inverse kinematics convert them into joint commands. These are not direct
22-dimensional joint-velocity outputs. Input-conditioned policies append one
normalized slip scalar; hidden-context policies receive no explicit slip value.

Simulation is synchronous, targets remain inside the permitted half-field,
and managed launchers verify the scene and own each server's lifetime.
Each evaluation uses 3,000 active steps of 20 ms: 60 active simulated seconds.
Detected falls trigger common reset and stabilization procedures, excluded
from active-time accounting. Falls thus measure detected balance losses under
assisted reset, not autonomous recovery.

The original installed scene is preserved separately from corrected scalar slip
0.1. Varied training uses the grid 0.025, 0.1, 0.2, 1 and 10. Evaluation samples
positive truncated Gaussian draws centred on those values, with standard
deviation half the parent mean and support (0,10]. A draw remains fixed
throughout a run. This is not a spatially varying or per-contact random surface.

### 3.2 Methods and staged allocations

The four principal methods are BC, mixed DAgger, GAIL-BC and GAIL-DM. Mixed
DAgger combines long-run and short-turn data. GAIL-DM starts from that mixed
DAgger, not the older historical DAgger model. Both GAIL variants retain
FC Portugal demonstrations as the expert source.

| Method | Initial allocation | Supervised cost before GAIL |
|---|---|---|
| BC | 170,000 expert examples; 100 epochs | 66,500 updates; 17 million exposures |
| Mixed DAgger | 100,000 seed examples + seven 10,000-step queried rollouts | 211,050 updates; 54 million exposures |
| GAIL-BC / GAIL-DM | 170,000-example demonstration pool + 70,000 PPO steps | Inherits its BC / DAgger initializer |

An exposure counts one use of an example, not a new oracle query. These
allocations initially standardized selected data and interaction quantities,
not lifetime compute. DAgger repeatedly fits its growing aggregate. GAIL
inherits the cost and representations of its initializer.

First, four policies are trained and tested in the original scene over five
task seeds. Second, these frozen policies are evaluated on changed surfaces.
Third, training continues across the slip grid with hidden and explicit-input
architectures, each using three training seeds. Fourth, varied-only lineages
start without original-domain pretraining. Their GAIL policies initialize from
the corresponding varied-only supervised models. Separate random-actor GAIL
controls test original and varied-hidden training.

The sampled suite has 15 grouped and 15 confirmation rounds, plus three
corrected scalar-0.1 rounds reported separately. Controllers share surface/task
draws. Original models have one training seed; continued replicas share a
pretrained parent. Repeated task rounds do not create independent training
replicas. The consolidated and scratch stages contain 56 trained policies and
1,949 deployments; reused controls are counted once.

### 3.3 Follow-up controls prompted by budget confounds

We add seven BC fits and 274 deployments without replacing old results.
Original BC is retrained from its saved initial weights on its original
170,000 examples to exactly 211,050 updates: 317 complete epochs plus 245
minibatches at batch size 256, totalling 53,952,720 exposures. Updates are
matched to DAgger; collection costs and state distributions are not.

Six varied-only BC fits (hidden/input, seeds 42–44) receive 200 epochs, or
34 million allocated exposures, matching original-100 plus varied-100 BC
continuation. They preserve their saved initializations and data, but do not
match the continuation's data distribution or optimizer history.
All selection uses validation; test values are never used to select checkpoints.

A zero-native-action controller retains observations, targets, gait generation,
IK and resets. Zero is not an expert-mean action: the 16 outputs are not
dataset-standardized. Finally, 25 retained original checkpoints are scored to
recover retrospective fidelity/resource curves, including GAIL's inherited costs.
The combined evidence now contains 63 trained policies and 2,223 deployments,
plus the deterministic zero controller. Optional early-stopping diagnostics
were not run in this batch.

### 3.4 Measurements and interpretation

Target progress is reported in metres alongside forward velocity and directional
tracking error. Normalized score is
(learner progress − random progress)/(expert progress − random progress),
computed per matched round. A score of one denotes expert parity in that
condition; zero is a random reference, not an absolute floor.

Held-out action MSE averages squared error over the 16 native outputs and
expert test observations. On-policy oracle discrepancy instead uses states
visited during deployment. Joint occupancy is compared using sliced Wasserstein
distance (SWD) over standardized 22-joint angle observations. SWD discards
temporal ordering and is not exact multivariate trajectory transport.

Mechanical measurements include falls per active minute, torso-angle RMS and
smoothed angular jerk, with reset discontinuities excluded. Integrated squared
jerk has units rad²/s⁵; its time-normalized rate has units rad²/s⁶. Neither is
measured torque, energy consumption or hardware damage. Figures show means and,
where available, training-replica points. No significance claim is based on
pooled correlated timesteps or on sparse falls.

## 4. Results

### 4.1 Matching optimization reverses the original fidelity ranking

Initially, all four supervised-initialized policies achieve approximately
oracle-level progress with no recorded falls in five active minutes each.
BC, DAgger, GAIL-BC and GAIL-DM progress is respectively 44.630, 44.364, 44.556
and 44.150 m, versus the oracle's 44.446 m. Initial held-out MSE is 0.11047,
0.01535, 0.11042 and 0.01572. These endpoints appeared to favour DAgger's
fidelity by a factor of 7.2 over BC.

The matched-update control changes that interpretation (Figure 1).
BC MSE becomes 0.01024, approximately 90.7% below its original value and
33.3% below DAgger's. Its on-policy error is 0.00403, versus DAgger's 0.00782.
The earlier ranking cannot be attributed to algorithm alone. This does not
show that learner-state labels are useless: the two methods retain different
data distributions, supervision requirements and histories.

![Figure 1. Original-scene optimization control. Matching updates reverses latent MSE, but does not uniformly improve progress or joint occupancy. Means over five deployments; one training seed per controller.](presentation/figures_final/01_budget.png)

Matched BC progress is 43.906 m and joint SWD 0.04716, versus DAgger's
44.364 m and 0.03909. Original BC SWD was 0.04503. Thus the action-error
improvement does not yield a universal mechanical or task-performance ranking.
Occupancy distance cannot establish that the gait interface suppresses action
differences, because it ignores their temporal structure.

### 4.2 Surface changes and original-only optimization

Absolute performance declines across the changed surfaces even when normalized
scores remain near one (Figure 2). Oracle progress averages 32.885 m over the
30 sampled rounds. Frozen original BC and DAgger average 32.828 and 32.646 m,
with 11 and six falls in 30 active minutes. BC's on-policy error rises from
0.07444 in the original scene to 0.29098 on sampled surfaces; its sampled
expert-test error is 0.17883. This is consistent with distribution-shift
concerns, but surface effects and harder visited states are alternative
explanations rather than separately identified causes.

![Figure 2. Each point is a deployment at its actual fixed slip. Controllers share the 30 surface/task rounds. The horizontal axis is logarithmic; task seeds also differ between rounds, so this is not an isolated slip-only response curve.](presentation/figures_final/02_slip.png)

Varied hidden continuation lowers BC and DAgger test MSE to 0.01995 and
0.00187, while progress remains about 32.8 m. However, the new original-only
matched BC also improves substantially without varied training: sampled-test
MSE 0.00841, on-policy MSE 0.01343, progress 33.151 m, SWD 0.04623 and zero
recorded falls. Therefore, earlier improvements after continuation cannot all
be credited to surface diversity. This original-only model has more updates
than the 200-epoch varied-only controls, preventing a claim that original-domain
training is intrinsically better.

### 4.3 Matching BC exposure narrows the initialization gap

At the initial varied-only allocation, hidden BC has MSE 0.11177 versus
continuation's 0.01995, although its progress is slightly higher
(33.192 versus 32.784 m). Doubling varied-only training reduces hidden MSE to
0.04210 and input-conditioned MSE from 0.11226 to 0.03775 (Figure 3).
The residual gaps relative to continuation are 2.11× and 1.90×, rather than
approximately 5.6×. Increased fitting explains a substantial part of the
original difference, but not all of it at the tested exposure.

![Figure 3. BC fidelity across training histories. Dots are three training replicas; bars are means. The final two conditions both allocate 34 million example exposures but differ in datasets and optimizer history. Logarithmic MSE axes.](presentation/figures_final/05_pretraining.png)

The 200-epoch varied-only models average 33.015 m hidden and 33.067 m input
progress, compared with continuation's 32.784 and 32.883 m. Falls total one/four
for hidden/input 200-epoch models and four/six for continuation, across 90
active minutes each. Sparse events do not establish a population ranking.
A residual history-associated fidelity gap is supported; an isolated causal
benefit of pretraining is not. Equivalent lifetime controls have not been
completed for DAgger or GAIL.

Explicit slip likewise gives small, inconsistent progress changes.
In the initial varied-only batch, input-minus-hidden progress is +0.082 m for
BC, −0.121 m for DAgger, +0.063 m for GAIL-BC and +0.053 m for GAIL-DM.
The expert has no explicit slip input but can respond to surface effects
through its observations. Without incorrect-context interventions, these
results do not distinguish unused context from redundant information.

### 4.4 Mechanical stability and the zero-action control

Viable supervised policies generally have low fall incidence, but no universal
mechanical winner emerges (Figure 4). Hidden continuation/scratch falls are
4/4 for BC, 5/4 for DAgger, 4/2 for GAIL-BC and 6/2 for GAIL-DM, each over
90 active minutes. Mean jerk rates remain near 39–41 million rad²/s⁶.
These small differences should not be described as measured energy savings.

![Figure 4. Initial-allocation mechanical results for hidden models. Falls are totals over equal active time; jerk bars are means over three replicas and 30 rounds. Lower jerk must be interpreted alongside useful movement.](presentation/figures_final/06_mechanics.png)

Zero native actions produce −0.105 m original-scene progress versus 44.446 m
for the oracle, and −0.303 m sampled progress versus 32.885 m. The retained
engine with this particular constant action does not explain the learned
policies' near-oracle progress. It has zero original falls yet essentially no
progress, illustrating why inactivity must not be mistaken for successful
stability. Other constant vectors and removal of the gait engine are untested.

### 4.5 Retained checkpoints qualify GAIL and resource claims

DAgger held-out MSE decreases from 0.21299 after its seed fit to 0.01535 after
round seven. Both data aggregation and fitting accumulate, so this curve does
not isolate marginal oracle-query value (Figure 5). Matching the final BC
update count gives lower error, while the methods still differ in collection
and supervisory cost.

![Figure 5. Retrospective original-test fidelity curves. Left: DAgger and BC versus allocated supervised updates. Right: GAIL versus learner steps including inherited DAgger interactions; BC fitting is an additional inherited cost. These are not return-convergence curves.](presentation/figures_final/09_resources.png)

GAIL-BC changes from 0.11047 to 0.11042; GAIL-DM changes from 0.01535 to
0.01572. This supports limited fidelity improvement for the tested refinement,
not a universal limitation of adversarial imitation. Direct-random GAIL also
fails to acquire useful walking at 70,000 PPO steps: progress is −0.121 m in
the original environment and −0.367 m for varied-hidden training, with no
targets reached. Larger-budget GAIL work remains outside this paper.

### 4.6 Probes describe, but do not explain away, behavioural differences

The prior representation analysis contains 524 multitarget ridge fits across
131 representations and 30 properties. It uses shared oracle trajectories,
episode-level splits, training-only scaling, validation-selected penalties,
raw-observation/random-network controls and shuffled-label diagnostics.
Targets include gait phase, posture, support-force proxies and motion 200 ms
ahead; future labels follow oracle actions, not learned-policy futures.

Selected hidden continuation readouts remain close between GAIL and its
initializer (Figure 6). DAgger's phase-conditioned roll R² is 0.7065 versus
BC's 0.6911, while future-velocity R² is slightly lower (0.6294 versus 0.6330).
No single readout accounts for every performance measure.

![Figure 6. Selected first-layer ridge readouts on shared oracle states. Dots are training replicas and diamonds their means; vertical axes are deliberately zoomed. These are the prior models, not the new matched-update BC controls.](presentation/figures_final/07_probes.png)

For hidden DAgger, scratch has higher expert CKA than continuation
(0.9428 versus 0.9212) despite worse action fidelity. Random-start GAIL can
also have appreciable future-motion decodability while failing to walk.
Accessible information and expert resemblance are not substitutes for
closed-loop evaluation. The new BC controls were not probed; old readouts
cannot causally explain their improvement or rescue the initial algorithm
ranking after it reverses.

## 5. Discussion and limitations

RQ1 is answered by separating movement from mechanical measurements:
supervised-initialized controllers generally walk stably, but sparse falls,
assisted resets and small mean differences preclude a universal stability
ranking. The zero-action baseline demonstrates why low falls alone are
insufficient. RQ3 similarly requires task progress and fidelity together:
near-oracle progress coexists with materially different latent errors, while
joint occupancy need not follow their ordering.

For RQ2, the most important evidence is the budget control. An initial DAgger
fidelity advantage disappears at matched original-scene optimizer updates.
Matching BC lifetime exposures also reduces the continuation/scratch gap.
These results establish optimization sensitivity, not minimum sample complexity
or equal collection cost. Retrospective fidelity curves partially address
the proposed efficiency analysis but do not demonstrate normalized-return
convergence.

Convergence remains unresolved. Matched BC selects the last partial epoch 318;
the six 200-epoch fits select epochs 197–200. The earlier final DAgger fit
selected epoch 50 of 50. All new controls stop at their budgets, so neither
method has demonstrated convergence under this comparison. A single original
training seed, shared continuation parents, correlated trajectories and reused
baselines further limit statistical inference.

The study deviates from the initial proposal in four explicit ways: 16 native
gait actions replace 22 direct joint velocities; GAIL is supervised-initialized
and BC-regularized; finite-budget results and retrospective fidelity curves
replace proven return convergence; and kinematic jerk replaces torque smoothness.
Varied policies are tested at corrected scalar 0.1, not in the preserved
original scene, so original-scene retention is not established. Spatial terrain,
collisions, motor faults, physical hardware and transfer to another robot are
outside the evidence.

## 6. Conclusion

In this supported RoboCup walking task, task parity does not imply imitation
parity. The fidelity ranking is sensitive to optimization allocation:
matched-update BC overtakes the original mixed DAgger on held-out action MSE,
and additional varied-only BC fitting narrows the apparent pretraining benefit.
The retained engine with zero native actions does not solve the task.
Explicit slip and the tested conservative GAIL refinement offer no consistent
broad progress advantage. Probing remains descriptive. These findings support
reporting multiple outcome measures together with data, supervision and
optimization costs, rather than presenting a universal algorithm winner.

## Data and reproducibility

The companion source reports preserve full per-replica metrics, paired rounds,
training costs and artifact provenance. The presentation figure manifest records
input JSON hashes; figures are generated without retraining or reselecting
checkpoints. Original results remain intact. The submission-control addendum
documents the feedback response. Publication repository/archive URLs and final
author details remain to be inserted before submission.

<!-- References are in paper/references.bib. See paper/README.md for compilation
and the source report links; this paragraph is not a claim of public data release. -->

