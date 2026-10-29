# Slide-by-slide explanations

Companion to [MEETING_SUMMARY_MARP.md](MEETING_SUMMARY_MARP.md), revised
12 October 2026. Numbering counts the title as slide 1, even though its printed
page number is hidden. Each section explains the slide, the relevant terminology,
and a short takeaway you can say aloud. This is a presentation aid based on the
existing results; it introduces no new experiments.

## Slide 1 — Context conditioning and learned representations

This presentation moves from evaluating what the policies do to examining how
their inputs and internal features relate to that behaviour. The recent work has
two parts: giving BC explicit information about the surface, and examining what
locomotion information can be recovered from trained networks.

“Context” means the supplied slip parameter. A “representation” is the vector of
activations produced inside a network when it receives an observation. It need
not correspond to named concepts or individual interpretable neurons.

The policies are BC, mixed DAgger, GAIL initialized from original DAgger, and
GAIL initialized from BC. FC Portugal is the expert throughout. R1 transfer is
deferred, so this presentation makes claims about NAO experiments only.

**Say:** “The recent work asks whether explicit surface information helps walking,
and what information the imitation networks retain internally.”

## Slide 2 — Brief recap: a strong walking baseline

The original evaluation established that all four selected policies could make
roughly oracle-level progress in the nominal environment. Their uncertainty
intervals overlapped oracle parity, so their slightly higher mean scores do not
establish that they are better walkers than the expert.

“Fidelity” means closeness to the expert's actions or behaviour. BC had low action
error with relatively simple offline training. Mixed DAgger improved original
DAgger's fidelity after increasing both the amount and variety of seed data.
GAIL-BC stayed close to BC's fidelity, while GAIL-DAgger retained much larger
action errors despite comparable walking progress.

This motivates probing: policies can achieve similar task outcomes while computing
different actions or organizing their internal information differently.

**Say:** “Walking performance alone does not distinguish these policies well;
their imitation errors and internal representations may tell us more.”

## Slide 3 — Brief recap: corrected dynamics change performance

The early scene settings appeared to change slip, but inspection of parameters
passed to the physics solver showed that effective slip was unchanged. Switching
to the single-value setter produced actual parameter changes. Earlier results
remain recorded, but cannot support claims about adaptation to varied slip.

In the corrected oracle screen, increasing the field setting from 0.1 to 10
reduced progress from 40.546 m to 15.325 m. The robot did not fall, illustrating
that a controller can remain upright while its locomotion deteriorates. These
numbers come from a one-seed screen, so they demonstrate responsiveness rather
than a population-wide effect size.

The subsequent 75-run comparison tested the oracle and four varied-trained
policies over 15 shared slip/task rounds. Their progress generally followed the
oracle's decline. That comparison alone does not isolate the benefit of training
on varied surfaces.

**Say:** “Once slip was physically changing, the surface affected progress strongly,
even without falls. That gave us a meaningful setting for the context study.”

## Slide 4 — Three ways to use the same BC demonstrations

The hidden-context model receives the usual 63 state features. The input-conditioned
model receives those features plus one encoded slip value, giving 64 inputs. Both
produce the same 16 latent Walk actions.

FiLM means **feature-wise linear modulation**. It lets slip adjust the final hidden
features before the action prediction:

\[
h'=(1+\Delta\gamma(c))\odot h+\beta(c).
\]

Here, \(h\) is the hidden activation vector, \(c\) is encoded slip,
\(\Delta\gamma\) supplies a scale adjustment, and \(\beta\) supplies an offset.
The symbol \(\odot\) means multiply corresponding components. For example, a
particular hidden feature could be amplified or reduced depending on context.

Both approaches initially reproduce the original BC function: the extra input
weights start at zero, and FiLM starts with zero scale adjustments and offsets.
All parameters are then trainable. FiLM adds more parameters, so the comparison
does not perfectly separate conditioning design from model capacity.

**Say:** “We compare supplying slip at the input with letting it directly adjust
the network's internal features.”

## Slide 5 — Training and evaluation controls

The architectures use the same existing expert demonstrations: 100,000 examples
for fitting, 10,000 for validation and 10,000 for testing. Whole trajectories are
split so neighbouring frames from one run do not leak across those partitions.
Validation chooses the saved checkpoint; the test set assesses it afterward.

An **epoch** is one pass through the training data. A **batch** is the 256 examples
used for an update. Adam is the optimizer; its learning rate is 0.0001.

Seeds 42, 43 and 44 chiefly vary minibatch ordering. They share the pretrained BC
weights and dataset, so they measure fine-tuning variability under those shared
conditions. They are three separately deployed policies per architecture.

The 33 evaluation rounds comprise 15 grouped, 15 confirmation and three nominal
cases. Hidden/input evaluation totals 429 runs because input BC also receives
three context modes. FiLM adds 99 correct-context runs and reuses saved controls.

**Say:** “The data and update budgets are matched, and each architecture has three
fine-tuning replicas evaluated on shared surface and task settings.”

## Slide 6 — Context benefit is small and inconsistent

The table gives average target progress in metres during 60 active simulated
seconds. Each sampled suite has 15 rounds times three trained replicas, giving
45 runs per architecture. The nominal suite has nine.

Input conditioning is slightly below hidden BC on grouped slips, above it on
confirmation slips, and below it on nominal tests. FiLM also changes position
across the suites. Falls total five, four and five across the 99 correct-context
deployments for hidden, input and FiLM respectively.

There is no consistent pattern of one architecture improving both progress and
stability. These are descriptive comparisons; small differences should not be
treated as established advantages simply because one mean is larger.

**Say:** “The context models sometimes improve individual averages, but neither
produces a consistent practical advantage across the evaluation suites.”

## Slide 7 — Does the input-conditioned policy use slip?

This asks whether the slip input affects the network's predictions at all. We
reuse the same 10,000 held-out observations and the same model weights, changing
only the supplied context. This avoids simulator startup differences in this
particular comparison.

Correct slip is the reference. Supplying constant slip 0.1 increases mean squared
error against the expert by about **0.41%**. Supplying deliberately wrong slip
increases it by about **1.47%**. Wrong context is 10 when the physical slip is
below 1, and 0.025 otherwise. These are relative changes in action MSE, not
percentage-point changes in walking performance.

**RMS** means root mean square. For the action change:

\[
\mathrm{RMS\ change}
=\sqrt{\operatorname{mean}[(a_{\mathrm{wrong}}-a_{\mathrm{correct}})^2]}.
\]

Square the differences so positive and negative changes do not cancel, average
across action components and observations, then take the square root. The reported
value, approximately **0.0056**, is in latent Walk-action units. It is not a joint
angle, speed or percentage, and its magnitude depends on the action scaling.

The network therefore uses context to some extent. However, correct context did
not consistently improve live walking progress, and this sensitivity does not
establish an advantage over hidden-context BC.

**Say:** “The network responds to slip, but that response has not produced a
consistent improvement in locomotion.”

## Slide 8 — Repeatability explains why small differences need care

We repeated the same case five times while holding the policy, task seed and slip
fixed. The complete diagnostic includes three cases and three controllers—expert,
hidden BC and input BC—for 45 runs.

**SD**, or standard deviation, describes spread around the mean. For example,
hidden BC's nominal result of 37.493 ± 0.618 m reports its average progress and
run-to-run standard deviation. It is not a confidence interval or a guarantee
that every run lies within those limits.

Identical task seeds reproduced the initial targets but not exactly the same
first-active physical states. Maximum initial planar-position differences within
case/controller groups ranged from 6.36 to 24.12 mm. These unequal starts provide
one reason to investigate variability; they have not been proven to cause all
the return differences.

**Say:** “Even identical settings produce some deployment variation, so small
architecture differences need more evidence before we attribute them to context.”

## Slide 9 — FiLM adds flexibility, with limited practical gain

FiLM's held-out action MSE is about 0.18% lower than hidden BC's—a small relative
improvement. Its **joint-occupancy sliced Wasserstein distance** is worse on both
sampled suites. That metric compares distributions of visited joint configurations
through one-dimensional projections; lower values indicate greater similarity
under this metric. It does not compare exact trajectory timing.

**Jerk** describes how rapidly acceleration changes. Here it is derived from joint
angle motion, and it was broadly similar between models. It is a kinematic
smoothness measure rather than a direct torque or energy measurement.

One paired FiLM run gained 14.075 m over hidden BC. The remaining 44 confirmation
pairs average −0.0234 m. This explains how a positive overall mean can be driven
by one difficult case. The case stays in the official results; this calculation
examines sensitivity rather than changing which results count.

**Say:** “FiLM's average gain is concentrated in one case, while the other metrics
do not show a broad improvement.”

## Slide 10 — Why pause further context training?

The expert does not receive the slip parameter explicitly, but it does observe
consequences such as body motion and forces. A student may already have enough
information in those ordinary observations to reproduce the expert's actions.

BC is rewarded for action matching. It receives no direct objective encouraging
a better surface-specific walking strategy than the expert's. This is a plausible
reason why explicit slip contributes little under this training setup.

The decision is to prioritize representation analysis instead of expanding the
same context experiment to DAgger and GAIL immediately. Those conditioned methods
remain untested. The result does not show that context is universally unhelpful
or that the expert never adapts implicitly.

**Say:** “Given the limited BC gains, we are examining what the existing networks
already encode before expanding the context study.”

## Slide 11 — Probing asks what can be read from a frozen network

A **probe** is a small diagnostic model trained to predict a named property from
a policy's hidden activations. The policy is frozen: only the probe is fitted.
For example, a linear probe can try to reconstruct torso roll from a hidden layer.

**Ridge regression** fits a weighted sum of features plus an intercept, with a
penalty that discourages excessively large weights. Validation chooses the penalty
separately for each target. It already supports continuous locomotion quantities;
the current study is not restricted to classification.

**R²** measures error relative to predicting the evaluated target's mean:

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
\]

One means perfect prediction, zero means the same squared error as that mean
reference, and negative values mean worse. R² is not classification accuracy.
The implementation treats constant-target R² as undefined.

**Say:** “A probe tells us how accessible a property is to a simple readout; it
does not by itself show that the policy uses that property to control walking.”

## Slide 12 — Which features were tested?

The 30 targets cover seven families. Gait phase describes where the external
generator is in its cycle. Foot-force magnitudes and thresholded indicators are
proxies for support. Body measurements describe height, vertical motion, torso
tilt and angular velocity. Ankle coordinates describe limb geometry relative to
the hip centre. Command features describe smoothed target displacement and orientation.

The interaction targets combine phase and posture, such as sine of phase times
roll. Future targets describe oracle velocity, direction error, posture and support
over or at the next 0.2 seconds.

Some labels are already directly available in the observation. Their recovery
tests retention through the network. Interaction and future labels ask whether
features expose combinations or predictive relationships beyond directly copying
a single input coordinate.

Force above 10 N is a sensor-based support indicator, not independent ground-truth
contact. The study does not measure a physical balance margin or centre of pressure.

**Say:** “We examine timing, support, body motion, geometry, commands and short-term
outcomes, while keeping the meaning and limitations of each label explicit.”

## Slide 13 — A shared dataset makes the model comparison interpretable

All policies receive the same states from 30 verified oracle runs. This prevents
a model from appearing easier to probe simply because it visited easier states.
Ten complete runs train the probes, five select settings and 15 test them.
Subsampling and removal of windows crossing resets leave 4,270, 2,133 and 6,405
observations respectively.

Three controls help interpret the results. Raw-input probes show what was already
accessible before the network. Random networks show what a generic nonlinear
transformation can expose without policy training. Shuffled trajectory labels
test how much predictability survives after breaking the intended correspondence.

The models probed here are the original four policies, not the new context or
FiLM models. Whole-run splitting reduces adjacent-frame leakage, but observations
within a run remain correlated. Future labels describe the oracle's continuation.

**Say:** “Shared states and explicit controls let us ask what the trained features
add beyond the inputs and an untrained network.”

## Slide 14 — Several locomotion properties are accessible

This slide groups the findings by their interpretation. Command orientation and
ankle geometry remain easy to decode, showing that useful input information is
retained. GAIL-DAgger's second layer retains torso angular-motion information more
strongly on the measured x/y axes.

Phase-conditioned roll is more interesting because learned networks make the
combination of timing and posture easier to read than the controls do. Future
direction error has a modest learned advantage, strongest in mixed DAgger.
Future forward velocity differs little between trained and random features.

High phase and support scores need particular care: periodic walking structure
also produces strong control scores. Readability alone therefore provides different
strengths of evidence for different properties.

**Say:** “Several features are accessible, but phase–posture combinations provide
stronger evidence of learned organization than directly supplied sensor values.”

## Slide 15 — Phase–posture interaction gives the strongest evidence

The target is sine of generator phase multiplied by torso roll. It connects gait
timing with sideways body tilt. This is a constructed interaction, not a measured
stability margin or a claim that the network explicitly multiplies those inputs.

Raw inputs give linear-probe R² of 0.332. Random nonlinear features improve that
to 0.427–0.452, showing that architecture alone helps expose the interaction.
Learned first layers reach 0.586–0.703. Each learned model beats each random
control on every one of the 15 test runs by mean squared error.

Phase and roll were already present in the input. The result shows improved
linear accessibility of their combination. It does not prove that this combination
causes stable walking, or that mixed DAgger is universally the best representation.

**Say:** “Training consistently makes the relationship between gait timing and
torso roll easier to extract.”

## Slide 16 — Angular motion and future direction separate models

“h1” and “h2” mean the first and second hidden layers. Gyroscope x/y quantities
measure angular velocity around their recorded axes. GAIL-DAgger's h2 scores
of 0.889 and 0.883 exceed the corresponding random ranges and the listed BC and
mixed DAgger scores. This is evidence that those sensor quantities remain more
accessible deeper in that particular model.

Mixed DAgger has the highest listed h1 future-direction-error score, 0.538.
Direction error is a vector-discrepancy measure between movement and target
directions, not an angle in radians. The label concerns the oracle's next 0.2 seconds.

Future forward-velocity scores are only slightly above random controls. We should
therefore avoid turning these results into a broad claim of learned dynamics
prediction. Different layers and models perform well on different quantities.

**Say:** “GAIL-DAgger retains angular-motion information more strongly in h2,
while mixed DAgger has a modest advantage on future direction error.”

## Slide 17 — Adding the expert anchors the comparison

The expert extension asks whether properties readable from the students are also
accessible in the original teacher. The expert network has one 64-unit tanh
hidden layer; each student has two 256-unit ReLU layers. Both student layers are
compared against the expert's single hidden layer.

Tanh compresses activations into a bounded range, while ReLU replaces negative
values with zero. Width and activation differences can affect probe results even
when the underlying controller information is useful in both networks.

The loaded expert's actions were checked against 90,000 recorded observations.
Sixteen new multitarget probe fits—expert and random controls under true/shuffled
labels—bring the total to 2,400 scalar target scores. These are correlated scores,
not 2,400 independent experiments.

**Say:** “We now compare students with the actual teacher, while accounting for
the teacher's much smaller and differently structured network.”

## Slide 18 — What remains accessible relative to the teacher?

Each row compares the same property, states and linear-probe procedure. For
example, phase-conditioned roll scores 0.281 in expert h1 and 0.586–0.703 in
student h1. Command orientation, future velocity, future direction error and
future pitch also have higher student scores in this table.

Across all 30 properties, both student layers and four models, all 240 R² gaps
favor students. That is a statement about this readout procedure. A wider layer
offers more features for the probe to combine; different nonlinearities and
training objectives can also affect accessibility.

The expert was optimized to produce useful actions. It need not make every
sensor quantity easy to reconstruct linearly. These results cannot be converted
into a percentage of knowledge transferred or a claim of superior student control.

**Say:** “The students expose these properties more readily to linear probes,
but architecture differences prevent interpreting that as greater knowledge.”

## Slide 19 — Which expert features exceed random controls?

This comparison keeps the expert-style architecture fixed: trained expert features
versus three random 64-unit tanh networks. It helps assess whether a property is
especially accessible after learning, beyond what that architecture exposes by itself.

The expert exceeds its random controls on command orientation (0.906 versus
0.748–0.830), phase-conditioned roll (0.281 versus 0.207–0.251), and future pitch
(0.522 versus 0.495–0.504). Shuffled-label scores are much lower for these targets.

Phase–posture organization consequently has supporting evidence in both teacher
and students. This makes it a useful candidate for later functional investigation.
It still does not show that a particular student computation was causally copied
from the expert or that it is required for balance.

**Say:** “Several teacher features exceed random controls, with phase–posture
structure also appearing clearly in the imitation models.”

## Slide 20 — Whole-layer geometry shows different resemblance patterns

**CKA** stands for centered kernel alignment. Here linear CKA compares the pattern
of activation relationships across the same observations. Roughly, if two networks
organize the sampled states similarly in feature space, their score is high.
It can compare layers with different widths without matching individual neurons.

The table's columns refer to **student h1 or h2 compared with expert h1**. The
expert does not have a second hidden layer. Mixed DAgger h1 is closest at 0.9355;
GAIL-DAgger is closest at its h2, scoring 0.9314.

Raw observations already score 0.8968, and random-network scores are also high.
Common inputs therefore explain substantial similarity. A score of 0.9355 does
not mean 93.55% of knowledge was transferred. CKA describes geometry and can be
influenced by dominant variation in the sampled states.

**Say:** “Mixed DAgger resembles the expert most strongly in its first layer,
while GAIL-DAgger does so in its second, against an already high input baseline.”

## Slide 21 — Activation structure and layer-reset diagnostics

**Effective variance dimension** describes how many directions carry substantial
activation variation. It is calculated from the covariance eigenvalues using the
participation ratio:

\[
d_{\mathrm{eff}}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\]

If variation is equal in ten independent directions, this gives ten. If most
variation is concentrated in one direction, it approaches one. The expert's
8.38 therefore describes variance concentration across the recorded states;
it does not mean the expert uses exactly eight neurons or knows eight skills.

Separately, copies of each policy had one affine layer reset. Their actions changed
by roughly 2.5–2.8 RMS in latent action units. This confirms that disrupting the
layers changes output, but random resets also break feature interfaces and scaling.
They do not identify which semantic feature caused the change. The reset copies
were not deployed in the simulator.

**Say:** “The networks have different activation structure, and their outputs
depend on the learned layers, but these diagnostics do not isolate individual skills.”

## Slide 22 — Limits that shape the next experiments

The controls themselves exposed limitations. Shuffling entire trajectories can
leave periodic gait timing aligned, so a high phase or support score may survive
even when the intended association is broken. A better null must actually disrupt
the relationship being tested.

A **persistence baseline** predicts future quantities from their recent or current
values. It performs poorly for some averaged or oscillating targets here. Beating
that weak baseline is insufficient evidence of sophisticated prediction.

Some probes choose the weakest tested ridge penalty, suggesting remaining
sensitivity to the regularization range. Only one original checkpoint per method
is probed, on oracle states, which also limits algorithm-wide claims.

Stronger periodic controls, capacity comparisons and nonlinear regression are
possible extensions. A nonlinear probe may recover more information, but also
perform more computation itself; it would need the same careful controls.

**Say:** “The next methodological improvements should strengthen the controls
and distinguish readily accessible information from the probe's own capacity.”

## Slide 23 — Meeting conclusions and discussion

The context study establishes weak action sensitivity without consistent practical
improvement. The representation study finds several readable locomotion properties,
with phase–posture interactions providing the strongest learned-over-random pattern.

Models differ in where particular features and expert-like geometry are most
accessible. BC and GAIL-BC remain close on the measured profiles. These findings
help formulate hypotheses, but do not explain every aspect of walking behaviour.

The next scientific question is whether changing a candidate representation
changes meaningful control outcomes. Such interventions would need controls for
generic network damage and a new evaluation protocol. They have not been performed
by the current offline probing study. R1 remains deferred.

**Say:** “We have evidence about accessible features; the next step is to test
which of those features makes a functional contribution to control.”

## Slide 24 — Evidence and methods

The local reports provide the actual data, evaluation protocols and detailed
limitations behind the slide claims. The full meeting brief preserves the wider
chronology, including the slip-setting correction and earlier experiments.

The methodological references play different roles: Alain and Bengio motivate
frozen-layer probes; Hewitt and Liang motivate control comparisons; Kornblith and
colleagues provide CKA for representation similarity; Perez and colleagues provide
FiLM's feature-wise modulation approach. Their original studies do not establish
locomotion benefits for this project; those are assessed using our experiments.

Use the links on this slide to answer requests for exact results or methods.
The command guide includes reproduction commands and the walking demo launcher.

**Say:** “The presentation is backed by the saved reports and explicit control
protocols; the references explain the analytical tools we adapted.”
