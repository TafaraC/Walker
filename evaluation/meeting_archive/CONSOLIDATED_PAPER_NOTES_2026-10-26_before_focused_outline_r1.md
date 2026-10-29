# Consolidated-study presentation: background, explanations and speaking notes

Prepared 26 October 2026 for [CONSOLIDATED_PAPER_MARP.md](CONSOLIDATED_PAPER_MARP.md).
The 25 numbered sections below match the deck exactly, including the title slide.
This document is deliberately fuller than the slides: it supplies terminology,
experimental context, numerical interpretation and boundaries for questions.

The evidence cutoff is the completed scratch-initialization comparison and its
probes, reported on 16 October. The later optional wide-slip/RL and larger-budget
GAIL-improvement studies are outside this presentation. Historical context/FiLM
experiments are not silently pooled with the new consolidated models.

Authoritative numerical source: [consolidated results report](../CONSOLIDATED_RESULTS_REPORT_2026-10-15.md),
especially Sections 1–8 and Appendix F. The [paper draft](../FINAL_PAPER_DRAFT.md)
sets the eight-body-page narrative; references do not count toward that limit.
Numbers in this deck are rounded descriptive summaries, not newly fitted tests.

## 1. Imitating RoboCup locomotion

The central distinction is between achieving a task and reproducing the expert.
Two controllers can make similar progress toward targets while issuing different
actions, occupying different joint configurations or losing balance at different
rates. The study therefore asks how these measures agree or disagree, rather than
looking for a single winning method.

**Imitation learning (IL)** uses behaviour from an existing controller as a
learning signal. The **expert/oracle** here is the unchanged native FC Portugal
walking controller. “Oracle” means it can supply reference actions for queried
states; it does not imply perfect performance on every surface.

Suggested opening: “The main result is not that one method walks farther than
all the others. It is that apparently similar walking can conceal substantial
differences in fidelity, and the same separation appears in the initialization
and representation results.”

## 2. What the restart made comparable

Earlier experiments evolved as collection, synchronization, reset handling and
GAIL stabilization were improved. They remain historically useful, but combining
their results indiscriminately would confound algorithms with pipeline versions.
The consolidated restart created one explicitly budgeted comparison with shared
infrastructure and a defined set of model lineages.

**BC** is behavioural cloning. **DAgger** means Dataset Aggregation; “mixed”
identifies the long-run/short-turn pipeline and its oracle/learner execution
schedule. **GAIL** is Generative Adversarial Imitation Learning. **GAIL-DM** in
this batch starts from mixed DAgger, not the older DAgger checkpoint used in the
historical GAIL study. A **checkpoint** is a saved set of model parameters;
a **lineage** describes which prior checkpoints and training stages produced it.

Managed servers launch and stop the simulator for each deployment and verify
the intended scene. Synchronous execution prevents model computation or process
handoff delays from unintentionally becoming extra simulated motion. This is
experimental control, not an algorithmic novelty.

Paper placement: a brief motivation for the protocol in Methods. Do not spend
the introduction recounting every debugging incident. Source: report Section 1.

## 3. Questions that anchor the final paper

The proposal's questions concern stability, resource requirements and matching
the expert's locomotion. These remain the organizing questions. Surface changes
test generalization; initialization tests training history; context tests whether
known environmental information improves the learned controller.

For the resource question, this study supports comparisons **at the tested
allocations**. It does not identify the minimum number of samples each method
needs to converge. That stronger claim requires matched learning curves,
convergence criteria and more fully controlled optimization histories.

Likewise, “torque smoothness” should not be retained as the label for the measured
jerk metric. We measured derivatives of joint angles, not motor torque. The final
paper should say **kinematic smoothness**, while explaining the relation to the
proposal's original mechanical-stability objective.

Suggested transition: “These extensions stay within the same research question:
what does it mean for an imitation policy to reproduce useful, stable walking?”

## 4. Experimental progression

Stage 1 establishes performance in the original installed environment. Stage 2
changes the surface while keeping policy weights fixed. Stage 3 trains across
the varied grid, with and without an explicit slip input. Stage 4 removes
original-environment pretraining to compare complete training histories.
Stage 5 examines representations from these saved networks.

**Frozen** means no weight updates during the evaluation. The varied-trained
policies are also frozen when evaluated; “adapted” describes their earlier
training, not online adaptation during a test deployment.

This structure separates several questions that would otherwise be muddled:
robustness without further learning, effects after additional learning, effects
of context information and effects of initialization. It does not make every
comparison a perfect causal isolation: additional optimization remains part of
the continued-training treatment.

Use the same progression for Results. It is an experimental argument, not a
chronology of all repository work. Source: report Sections 1 and 3–8, Appendix F.

## 5. What the learned controller actually controls

The students use 63 observation features, two 256-unit hidden layers with ReLU
activations, and 16 latent gait-action outputs. A **hidden layer** is an internal
transformation of the observation; its **activation** is the feature vector it
produces for a particular state. ReLU applies a nonlinear threshold at zero.

The 16 outputs are interpreted by the existing stateful gait generator and
**inverse kinematics (IK)**. IK converts desired body/foot geometry into joint
configurations. This shared machinery is crucial: the experiment is not asking
a network to discover every aspect of bipedal mechanics through independent
joint commands. Mechanical evaluation nevertheless observes 22 joint angles.

The input-conditioned network has 64 inputs because it adds
`log1p(actual_slip) / log(11)`. It does not change into a 64-dimensional hidden
representation or a new 64-to-16 linear controller. For the continuation
condition, the new context input column starts at zero, preserving the initial
function, and the network is then trained. The expert's internal architecture
differs: its examined hidden representation has 64 tanh units. This matters when
comparing probe scores between expert and students.

Source: report Section 1 and the paper draft's controller-interface subsection.

## 6. Methods and the meaning of “expert”

BC learns a state-to-action mapping from recorded oracle examples. It can suffer
from **covariate shift**: small errors change the states it visits, and those
states may not resemble the expert training distribution.

DAgger addresses this by querying the oracle on states reached during collection
with the learner. New labelled examples are added to the accumulated dataset.
In this mixed schedule, the probability of executing the expert action declines
through 0.8, 0.6, 0.4, 0.2, 0, 0, 0. This **beta schedule** changes which action
is executed; an oracle label is still collected for every learner-collection step.

GAIL trains a **discriminator** to distinguish expert from learner examples,
using its output to provide a learning reward. The **actor** chooses actions;
the **critic** estimates expected return to help the update. **PPO**, Proximal
Policy Optimization, is the policy-update method used here. **KL divergence**
measures distributional change between policies; the safeguards limit abrupt
updates. A **BC anchor** adds a supervised imitation term that encourages
retention of the expert-like action mapping.

Initializing GAIL from BC or DAgger supplies starting weights. Its expert dataset
still comes from FC Portugal. This distinction prevents the mistaken conclusion
that GAIL is learning to imitate DAgger instead of the oracle.

Sources: report Sections 1, 6.4 and 7; Ross et al., Ho and Ermon, Schulman et al.

## 7. Standardized allocations—not equal total compute

The main allocations are 170,000 examples for BC, 100,000 seed examples plus
70,000 new learner-state labels for DAgger, and a 170,000-example expert pool plus
70,000 PPO interactions for each GAIL variant. Separate validation and test
expert partitions are not training examples. GAIL also inherits its supervised
initializer's full training cost.

These units must remain distinct:

- An **example** is a stored labelled state/action pair.
- An **environment step/transition** is one interaction with the simulator.
- An **epoch** is one pass through a training dataset.
- An **optimizer step** updates parameters using a minibatch.
- An **example exposure** counts reuse of an example during optimization.

BC uses 100 epochs, giving 17 million example exposures and 66,500 optimizer
updates. DAgger repeatedly retrains accumulated datasets, giving 54 million
exposures and 211,050 updates. Its stronger fidelity therefore comes with more
supervision at learner states **and** greater optimizer exposure. We cannot
attribute the entire difference to state coverage alone.

Collection combines long travel and short turns. A varied round contains
long/short segments at each of five surfaces; it is not a curriculum that only
introduces high slip late. GAIL uses static expert data during training rather
than active oracle queries on each PPO step; evaluation oracle queries belong
to measurement costs, not training costs. Source: report Sections 1.3 and 7.

## 8. Surface training and evaluation

The training grid is 0.025, 0.1, 0.2, 1 and 10. A **Gaussian parent** is the
distribution from which evaluation slips are drawn, not the actual test value.
For each parent mean μ, the standard deviation is 0.5μ, with draws restricted
to (0,10]. **Truncation** changes the resulting distribution: the parent centered
at 10 has draws below 10 and an empirical mean well below its parent mean.

Each round draws one value, then holds it constant for the full run and tests
all controllers at that value. There is no spatial Gaussian painted onto the
pitch, nor a new random value for every foot contact. Training uses a discrete
grid; testing measures behaviour at sampled values between and around it.

The main tables pool 15 grouped and 15 confirmation rounds. Three additional
scalar-0.1 rounds assess a separate retention condition. “Confirmation” is the
schedule's name: those draws have been reused/inspected, so they must not be
described as a newly untouched final holdout after every research decision.

The original installed scene and corrected scalar-0.1 scene are distinct
conditions. Avoid calling the original scene “zero slip.” The scalar is a
simulator contact parameter, not a measured real-world friction coefficient.
Scene/import verification also should not be described as a fresh live solver
readback for every deployment. Source: report Sections 1.4 and 4.1.

## 9. What the sample counts mean

The parent has 28 checkpoints: four original models, 12 varied-hidden models
and 12 varied-input models. Each varied group is four methods × three training
seeds. The scratch extension adds 24 varied-only lineage checkpoints plus four
direct-random GAIL checkpoints: one original and three varied-hidden.

The 1,020 parent deployments are reused; the extension adds 929, making 1,949
unique deployments—not 2,969. Oracle and random-action baselines are controls,
not additional trained checkpoints. Probe counts similarly add 224 new fits to
300 existing fits, rather than repeating the old evidence as new measurements.

There are different meanings of **seed**. A training seed changes initialization
or optimization randomness; a task seed determines the test target sequence;
a sampling seed determines surface draws. A family with three training seeds
and 30 evaluation rounds has 90 deployments, but only three trained replicas.
Continuation replicas also share original pretrained parents. Frames within a
trajectory are correlated and must not be treated as independent trials.

At a 20 ms simulation step, 3,000 active steps represent 60 simulated seconds.
This is not wall-clock duration. Startup, reset and stabilization are excluded
from active time. Original-scene results use five task seeds; sampled results
use 30 rounds per frozen model or 90 per varied family. Sources: Section 1,
Appendix F.1.

## 10. Success has more than one axis

**Target progress J** sums reductions in distance to the current target. It is
not total path length or final displacement from the spawn point. Normalization
subtracts the matched random baseline and divides by the oracle–random gap,
within each round, before averaging. A score may exceed 1 or be negative. Zero
is not a mathematical lower bound; a small denominator would make scores unstable.

**Action mean squared error (MSE)** averages squared differences across the 16
latent action elements on shared held-out expert states. It is not measured in
joint-velocity units. The proposal's squared-vector sum would be 16 times this
element-mean quantity. **On-policy MSE** instead compares learner and oracle
actions on states that learner visits; it is informative but uses different
state distributions for different policies.

**Sliced Wasserstein distance (SWD)** compares standardized 22-joint angle
occupancies using 256 one-dimensional projections. Occupancy describes which
states are visited, not their temporal order. This is an approximation based on
projections, not exact full-dimensional trajectory transport.

**Forward velocity** is translation projected onto the robot's heading.
**Direction error** compares unit movement and target directions; it is
dimensionless, not degrees. **RMS**, root mean square, is the square root of the
average squared value. Torso RMS measures roll/pitch excursions about zero and
is not the same as mean angle or standard deviation about a nonzero mean.

**Jerk** is the third derivative of joint angle. The metric uses radians,
five-frame smoothing and finite differences within reset-delimited segments.
Integrated squared jerk has units rad²/s⁵; dividing by valid duration gives the
reported rate in rad²/s⁶. It is sensitive to smoothing and sampling. It does not
directly measure torque, energy consumption or damage. Source: report Section 2.

## 11. Original scene: similar progress, different fidelity

BC's mean progress is 44.630 m versus the oracle's 44.446 m; DAgger is 44.364 m.
The small differences should not be presented as reliably beating or losing to
the expert. All four students have zero detected falls in these five runs, and
the reported intervals include oracle parity.

The stronger contrast is held-out action error: BC 0.11047 versus DAgger 0.01535,
about a 7.2-fold difference. The corresponding GAIL variants remain close to
their supervised initializer's action error. Locomotion can therefore look very
similar while local action agreement differs substantially.

Suggested sentence: “The original scene establishes why we need fidelity
metrics: task progress alone largely collapses four different models into a
tie.” Keep this table in the paper's main results. Source: report Section 3.

## 12. Frozen policies: normalization can hide slowdown

The table groups six actual sampled surfaces per Gaussian parent. Oracle
progress declines from 42.148 m in the lowest-parent group to 17.476 m in the
highest, and the frozen learners broadly follow that decline. Because the expert
also slows, a normalized score near 1 does not mean performance is unaffected.

The parent centered at 10 generated combined sampled values approximately
2.905–8.960, not repeated exact-slip-10 evaluations. Task seeds also differ among
groups. The table shows environmental burden, not a perfectly paired causal
estimate of increasing slip alone with everything else fixed.

Across all 30 rounds the oracle averages 32.885 m. The frozen BC and DAgger
means are 32.828 and 32.646 m. Use both the absolute and normalized values to
avoid claiming robustness solely from relative scores. Source: Section 4.1.

## 13. Varied training improves fidelity and fall incidence

Held-out varied-domain action error falls from 0.17883 to 0.01995 for BC and
from 0.01634 to 0.00187 for mixed DAgger. The latter is still about ten times
smaller than continued BC's error. This is the clearest DAgger result, rather
than a claim that it always walks farther.

Fall rates use unequal exposure correctly: frozen BC has 11 falls over 30 active
minutes; continued BC has four over 90. Those become approximately 0.367 and
0.044 falls/minute. Do not compare 11 and four without the denominators. A
detected fall triggers reset; this is not an autonomous get-up task or a count
restricted to irrecoverable collapses.

These are favourable outcomes after varied continuation, but the comparison
also adds training history and compute. The slide does not isolate environmental
diversity from every possible benefit of additional nominal training. Sources:
report Sections 4–6.

## 14. Does explicitly supplying slip help?

**Hidden context** means the surface parameter is not explicitly supplied to
the network. It does not mean the model lacks hidden layers, and the effects of
slip may still be visible in proprioceptive observations. **Input conditioning**
adds the known scalar surface value to the observation. Both models can update
their weights during training; neither learns during evaluation.

The progress changes are small and vary by method and lineage. For example,
input conditioning adds about 0.099 m for continued BC but subtracts 0.029 m
for continued GAIL-DM. Scratch DAgger changes in the opposite direction from
continued DAgger. This does not support a broad context benefit.

It is reasonable to hypothesize that the current imitation objective provides
little incentive to use explicit slip beyond what the expert behaviour and
observations already supply. It is not established that an expert without an
explicit context input prevents useful context learning. That is a mechanistic
hypothesis, not a conclusion. Historical FiLM is outside this consolidated batch.
Sources: Section 5.3, Appendix F.4.

## 15. The final extension: remove original pretraining

The extension asks whether original-domain training history is useful compared
with learning directly in the varied domain. **Continuation** starts from an
original-trained model and updates it on varied data. **From scratch** here
means removing that original-domain lineage, not removing the engineered gait
interface or all supervised initialization from every method.

Scratch BC and DAgger start with random parameters. Scratch GAIL-BC and GAIL-DM
start from those newly trained varied-only supervised policies. Direct-random
GAIL is an additional, distinct condition. There is no direct-random
input-conditioned GAIL cohort in this extension.

Same incremental budgets make the additional varied learning comparable, but
continuation has already paid for original pretraining. A difference can be
associated with complete training history; it cannot be assigned solely to the
source domain independently of lifetime optimization. Source: Appendix F.1.

## 16. More progress does not mean closer imitation

Scratch hidden BC gains roughly 0.41 m per active minute over continued BC but
its MSE increases from 0.01995 to 0.11177, about 5.6-fold. GAIL-BC follows the
same pattern. This is not a universal benefit of either continuation or scratch:
which looks better depends on whether the goal is progress or action fidelity.

Scratch DAgger remains much closer to the expert than scratch BC: 0.01035 versus
0.11177, about a 10.8-fold reduction. Continued DAgger is more precise again,
with MSE about 0.00187. Its slightly lower progress does not negate its stronger
imitation, nor does stronger imitation guarantee greater progress.

Suggested interpretation: “The initialization extension strengthens the main
paper argument rather than introducing a new one: performance and imitation
are distinct outcomes.” These are descriptive pooled differences, not declared
statistical significance across independent complete training histories.
Source: Appendix F.2.

## 17. Mechanics: no single winner either

Both cohorts have low fall counts across 90 active minutes per family. The
scratch GAIL families have two falls each in the hidden condition, compared
with four and six for continuation. Those small counts are encouraging but
insufficient by themselves to claim a reliable algorithm-wide stability benefit.

Scratch BC's SWD increases from 0.0502 to 0.0574 while jerk decreases. Thus it
can have smoother measured joint changes while visiting a less expert-like
joint distribution. DAgger's occupancy change is smaller and in the other
direction. These metrics describe different physical aspects.

The slide divides jerk rate by one million for legibility: 40.712 means about
40.712 million rad²/s⁶. The companion report also provides forward speed,
direction error and torso RMS. Include those in the supporting material; avoid
an unreadably wide main-paper table. Source: Appendix F.3.

## 18. GAIL depends strongly on the starting policy here

Direct-random GAIL fails to reach targets in the shown conditions. The
original-trained actor records zero falls in the original test yet makes
negative average progress. This exposes the problem with interpreting low fall
counts or low jerk without task success: standing or ineffective motion can
look mechanically quiet.

The supervised-initialized variants retain useful walking. However, their action
errors and selected probes remain close to their initializers, and the measured
task/mechanical changes do not establish a consistent broad improvement.

This conclusion applies to the consolidated 70,000-interaction configuration,
including its supervised anchor and conservative updates. It does not mean GAIL
cannot learn locomotion, and it does not evaluate the later optional 350,000-step
recipe. The random-action normalization baseline is also distinct from a
random-initialized network subsequently trained with GAIL. Sources: Section 6.4,
Appendices F.5–F.6.

## 19. Probing: what can be read from the networks?

A **regression probe** freezes the policy and trains a separate, simple readout
to predict a labelled physical property from its activations. The policy does
not change. This study uses **ridge regression**, a linear readout with an L2
penalty on its coefficients; the penalty is chosen on validation episodes.
Standardization is fitted on the training partition only.

There are 30 shared oracle traces: ten training, five validation and 15 test
episodes. Subsampling and valid future windows yield 4,270 training, 2,132
validation and 6,403 test states. Future labels are ten simulation steps, or
0.2 seconds, ahead; windows crossing resets are removed.

Targets include phase sine/cosine; force-threshold contact proxies; torso
orientation and angular motion; head height; ankle/hip geometry; desired motion;
phase-conditioned posture; and future movement/posture. A contact proxy based on
force exceeding 10 N is an operational label, not a perfect ground-truth support
state. This is a selected set of locomotion properties, not an exhaustive search
for every internal representation.

The combined analysis has 131 representations and 524 fits: true-label and
three shuffled-label fits, each reporting 30 target scores, yielding 15,720
scores. These are not 15,720 independent experimental replications. Expert
features, raw inputs and random-network features are essential controls.
Sources: Section 8 and Appendix F.6; Alain and Bengio; Hewitt and Liang.

## 20. Several locomotion properties are accessible

**R²** is `1 − squared prediction error / squared error of the test-mean baseline`.
One denotes perfect prediction, zero matches that constant baseline, and values
can be negative. It is not classification accuracy or a percentage of knowledge.
**RMSE**, root mean squared prediction error, instead reports prediction error
in the target's units; it is useful when a physical error magnitude matters.

“Phase × roll” is a labelled interaction between gait phase and torso roll. A
linear readout can benefit if the network has already formed nonlinear
relationships between its inputs. Its R² improves from 0.3624 on raw observations
and 0.4559 on random features to 0.6911 on BC and 0.7065 on DAgger.

Future velocity and pitch show smaller, mixed changes. Random nonlinear
features already expose much of their predictability, so a good probe alone
does not show that training discovered an entirely new physical model. Future
targets follow **oracle trajectories**: these are not predictions of what the
learner would do after choosing its own action.

Suggested sentence: “The networks expose several locomotion-related quantities,
but we need controls to distinguish useful reorganization from information
that was already available in the observation.” Source: Appendix F.6.

## 21. Representation similarity is not a success metric

**Centered kernel alignment (CKA)** compares representation geometry across the
same examples. In the linear case used for this type of comparison, it assesses
how similarly the representations organize those examples without requiring
one-to-one correspondence between neurons. It is a normalized similarity
coefficient, not a count of shared concepts or percent expert knowledge.

Scratch hidden DAgger has higher expert CKA than continuation (0.9428 versus
0.9212), and higher future-pitch R², yet worse action MSE. Thus these representation
measures cannot substitute for evaluating the action mapping. The failed
random-start GAIL network exposes future-velocity information about as well as
untrained random features, while not achieving useful walking.

Expert/student widths and nonlinearities differ. Some periodic shuffled labels
also remain predictable; the shuffled control is not universally a zero-score
null. No causal layer intervention shows that a particular decoded feature is
necessary for walking. “Accessible” is therefore more defensible than “used,”
“transferred knowledge” or “explains success.” No R1 transfer conclusion follows.
Sources: Section 8, Appendix F.6; Kornblith et al.

## 22. What the paper can conclude

The conclusion should retain all three original questions. On performance,
supervised-initialized policies broadly reach oracle-level progress under the
tested conditions. On fidelity, mixed DAgger is consistently strong. On stability,
the successful varied-trained families have low fall incidence but no uniformly
dominant method across all mechanical measures. On resources, the advantage
must be accompanied by explicit query and optimization accounting.

Initialization adds a progress–fidelity trade-off; explicit context produces
small, inconsistent changes. Probing gives supporting descriptions rather than
a causal explanation. The word “consistent” should refer to the observed
conditions, not all possible architectures, budgets or environments.

Avoid concluding that the oracle is optimal, that context is useless, that GAIL
cannot learn, or that DAgger wins every metric. The defensible contribution is
an auditable comparison showing why those overly broad rankings are misleading.

## 23. Boundaries to state plainly

**Confounding** means a comparison changes more than one potentially relevant
factor. Continued training differs from scratch in both source experience and
lifetime optimization. DAgger differs from BC in state coverage, query use and
repeated training exposure. These are comparisons of implemented training
protocols, not fully isolated effects of single components.

**Paired rounds** control surfaces and task schedules, but trajectories diverge
as actions differ. Reusing the same baselines is efficient and appropriate;
counting them as new independent samples is not. Likewise, many correlated
frames do not compensate for few independently trained models.

The engineered gait controller, fixed surfaces, fall resets and finite target
set limit external validity. Mechanical metrics are not hardware measurements.
The probes are descriptive, not causal interventions. Keep these limitations
specific: they define what the evidence supports without dismissing the work.

The optional later experiments belong in future work at most, without claims
about unreviewed results. R1 transfer remains outside this paper's evidence.

## 24. An eight-page paper, not a project diary

The proposed allocation totals eight body pages; the references are outside
that limit. It emphasizes the controlled comparison and its findings, not a
tutorial on every algorithm or a chronological list of infrastructure changes.

Suggested main exhibits:

1. One compact design/budget table with model lineage and cost caveats.
2. Original-scene progress and action-error comparison.
3. Environmental progress and fidelity/fall-rate comparison.
4. Continuation versus scratch progress–fidelity table or plot.
5. A small probe result illustrating why CKA/readout quality is not behaviour.

Context can be a compact subsection within varied training. Probes should be
supporting evidence rather than a second full paper. Full per-seed tables,
all 30 probe targets, complete mechanical profiles and infrastructure details
remain in the supporting report/repository; any formal supplement must follow
the eventual venue's rules.

This presentation is longer than the intended paper because the companion
supports discussion and terminology questions. Slide count is not a proposed
section count or page count. Source: final paper draft's editorial plan.

## 25. Sources and final takeaway

Use primary method references for what each method does, and repository results
for what happened in this experiment. A citation to GAIL does not establish that
our particular implementation should outperform its initializer. A probing
paper does not turn a correlational readout into a causal finding.

Selected primary references:

- Ross, Gordon and Bagnell (2011), [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html): DAgger and learner-state data collection.
- Ho and Ermon (2016), [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476): adversarial imitation.
- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): PPO policy updates.
- Orsini et al. (2021), [What Matters for Adversarial Imitation Learning?](https://arxiv.org/abs/2106.00672): implementation choices and evaluation context.
- Alain and Bengio (2016), [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644): representation-readout motivation; this project uses regression for continuous targets.
- Hewitt and Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/): the importance of controls; our locomotion controls are adapted, not a direct replication.
- Kornblith et al. (2019), [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html): CKA representation comparison.

See [report Appendix E](../CONSOLIDATED_RESULTS_REPORT_2026-10-15.md#appendix-e-paper-sources-and-final-write-up-citation-guide)
for the full verified bibliography, including context, occupancy and smoothness
background, and the precise attribution boundaries. See Appendix D and F.8 for
artifact provenance rather than citing rounded slides as primary experimental data.

Closing sentence: “The study shows why imitation should be evaluated jointly:
useful walking, agreement with the expert, mechanical stability and training
cost answer related—but different—questions.”

## Conversion and maintenance

The deck has Marp YAML metadata, 16:9 slides and inline speaker comments. It uses
no external image assets. From the repository root, with Marp already installed:

```bash
marp evaluation/presentation/CONSOLIDATED_PAPER_MARP.md --pdf \
  -o evaluation/presentation/CONSOLIDATED_PAPER.pdf
```

Conversion is user-run. Source structure and slide/notes alignment can be checked
without rendering; rendered layout still needs inspection after conversion.
The historical meeting deck and its notes are preserved separately. If this
deck's numerical claims change, update this companion with it and retain the
experimental source reports.
