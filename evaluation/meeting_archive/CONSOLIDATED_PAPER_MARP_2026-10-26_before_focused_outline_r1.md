---
marp: true
theme: default
paginate: true
size: 16:9
title: Imitating RoboCup locomotion
description: Consolidated experiments and the argument for the final paper
style: |
  section { font-size: 25px; }
  h1 { font-size: 39px; }
  h2 { font-size: 32px; }
  table { font-size: 21px; }
  footer { font-size: 13px; }
footer: 'Consolidated study · 26 October 2026 · Completed evidence through scratch comparison'
---

# Imitating RoboCup locomotion
## Fidelity, stability and surface generalization

From standardized original-environment training to varied-domain training, initialization comparisons and representation probes.

**The paper’s central argument:** matching the expert’s task performance does not necessarily mean matching its actions—or learning more useful internal representations.

Companion: [slide explanations and terminology](CONSOLIDATED_PAPER_NOTES.md).

<!-- Slide 1. Present this as a controlled empirical comparison, not a new learning algorithm. -->

---

# 2 · What the restart made comparable

The consolidated batch replaced comparisons between independently evolving historical pipelines with a common protocol:

- Four methods: **BC, mixed DAgger, GAIL-BC and GAIL-DM**.
- Shared observation/action interfaces, long runs and short turns.
- Explicit data and interaction budgets; managed synchronous servers.
- Common evaluation rounds, fall handling and fidelity metrics.

This presentation covers the completed consolidated study and its scratch extension—not historical FiLM, R1 transfer or later optional stress/GAIL experiments.

<!-- Slide 2. DM now means mixed DAgger. Historical GAIL initialized from the older DAgger is a different lineage. -->

---

# 3 · Questions that anchor the final paper

| Research question | Evidence to prioritize |
|---|---|
| How stable are the learned controllers? | Falls, joint jerk, posture and useful movement |
| What do the tested resources buy? | Fidelity and performance at stated data/interaction budgets |
| Do policies match the expert? | Progress, velocity, tracking and action error together |

Surface variation, initialization and context **extend these questions**.
Probing supports their interpretation; it is not a separate transfer claim.

<!-- Slide 3. Do not claim convergence/sample-complexity curves from endpoint budget comparisons. -->

---

# 4 · Experimental progression

| Stage | Comparison | Question |
|---|---|---|
| 1 | Original training → original scene | Can the expert be imitated? |
| 2 | Frozen original policies → varied surfaces | What generalizes without retraining? |
| 3 | Varied training: hidden vs explicit slip | Does exposure/context help? |
| 4 | Continuation vs varied-only initialization | Does original training history matter? |
| 5 | Shared-state probes, including the expert | What information is accessible? |

The final report should follow this sequence, ending with interpretation rather than opening another experimental direction.

<!-- Slide 4. Probes cover parent and scratch cohorts; they are analysis, not an additional control-training stage. -->

---

# 5 · What the learned controller actually controls

**63 observations** → **256 → 256 hidden units** → **16 latent gait actions**

↓ existing stateful gait generator and inverse kinematics ↓

**NAO joint motion**, measured across 22 joint angles

- Input-conditioned policies add one slip feature: **64 inputs**.
- The FC Portugal oracle remains unchanged.
- This is imitation within an existing walking interface—not learning unrestricted joint control from nothing.

<!-- Slide 5. Student hidden layers use ReLU. The native expert has a different 64-unit tanh representation. Sixteen outputs are not 22 direct joint-velocity commands. -->

---

# 6 · Methods and the meaning of “expert”

| Method | Learning signal |
|---|---|
| BC | Predict expert actions on recorded expert states |
| Mixed DAgger | Add expert labels for learner-visited states |
| GAIL-BC | Adversarial/PPO refinement initialized from BC |
| GAIL-DM | Same refinement initialized from mixed DAgger |

**GAIL always uses FC Portugal demonstrations as expert data.**
BC or DAgger supplies its starting weights, not a replacement oracle.

The tested GAIL recipe includes a supervised anchor and conservative policy-update safeguards.

<!-- Slide 6. Discuss implementation-specific findings; do not generalize the conservative recipe to all GAIL. -->

---

# 7 · Standardized allocations—not equal total compute

| Method | Main allocation |
|---|---|
| BC | 170,000 expert examples |
| Mixed DAgger | 100,000 seed examples + 7 × 10,000 learner-state labels |
| Each GAIL | 170,000-example expert pool + 70,000 PPO interactions |

BC: **17 million** supervised example exposures.
Mixed DAgger: **54 million** exposures through repeated aggregate training.

GAIL also inherits the cost of its initializer.
Report data, oracle queries, interactions and optimization separately.

<!-- Slide 7. BC has 66,500 gradient updates; DAgger 211,050. A unique example, an environment transition and an optimizer step are different units. -->

---

# 8 · Surface training and evaluation

Training grid: **0.025, 0.1, 0.2, 1, 10**; each varied round covers all five.

Evaluation samples from Gaussian parents centered on those values:

- Standard deviation = half the parent mean; truncated to **(0, 10]**.
- Draw one slip value and keep it **fixed for the entire deployment**.
- All controllers share that round’s surface and task seed.
- Main varied results pool 15 grouped + 15 confirmation rounds.

This is not per-contact randomness. The original installed scene is also distinct from the corrected scalar-slip 0.1 condition.

<!-- Slide 8. There are also three scalar-0.1 rounds, excluded from the pooled 30-round tables here. Confirmation is not a newly blind test after all decisions. -->

---

# 9 · What the sample counts mean

**56 trained checkpoints · 1,949 deployments · 524 probe fits**

| Component | Parent batch | Scratch extension |
|---|---:|---:|
| Trained checkpoints | 28 | +28 |
| Evaluation deployments | 1,020 | +929 |
| Probe fits | 300 | +224 |

Each deployment: **3,000 active steps = 60 simulated seconds**.
Frozen family: 30 sampled runs; varied family: 3 training seeds × 30 = 90.

Reused controls are not new measurements. Three replicas are not 90 independent trained models.

<!-- Slide 9. Original-scene comparison has five task seeds. Recovery time is excluded. Shared pretrained parents limit independence of continuation replicas. -->

---

# 10 · Success has more than one axis

| Axis | Main measurements |
|---|---|
| Task/environment performance | Target progress, normalized score, forward velocity, tracking |
| Imitation fidelity | Held-out action MSE; joint occupancy distance |
| Mechanical stability | Falls per active minute, torso RMS, angular jerk |
| Representation | Held-out probe R²; expert CKA similarity |

**Normalized score = (policy progress − random progress) / (expert progress − random progress).**

Score 1 means oracle parity **on that surface**, not unchanged absolute performance.

<!-- Slide 10. Score zero is a reference, not a floor. MSE is per latent-action element; jerk is kinematic, not torque. -->

---

# 11 · Original scene: similar progress, different fidelity

Five 60-second deployments per controller.

| Controller | Progress (m) ↑ | Normalized score | Test action MSE ↓ |
|---|---:|---:|---:|
| FC Portugal | 44.446 | 1.0000 | — |
| BC | 44.630 | 1.0044 | 0.11047 |
| Mixed DAgger | 44.364 | 0.9985 | 0.01535 |
| GAIL-BC | 44.556 | 1.0025 | 0.11042 |
| GAIL-DM | 44.150 | 0.9934 | 0.01572 |

All four learned methods recorded zero falls. DAgger’s action error is approximately **7.2× lower than BC’s**, despite similar progress.

<!-- Slide 11. Small progress differences do not establish superiority to the oracle; reported intervals include parity. Source: report Section 3. -->

---

# 12 · Frozen policies: normalization can hide slowdown

Mean progress across six sampled rounds per Gaussian parent:

| Parent mean—not actual fixed slip | Oracle (m) | Frozen BC (m) | Frozen DAgger (m) |
|---|---:|---:|---:|
| 0.025 | 42.148 | 43.008 | 42.531 |
| 0.1 | 39.341 | 38.679 | 39.361 |
| 0.2 | 37.190 | 37.190 | 36.404 |
| 1 | 28.270 | 27.584 | 27.690 |
| 10 | 17.476 | 17.679 | 17.245 |

Policies can remain near normalized score 1 while **both they and the oracle slow substantially**. Always report absolute progress too.

<!-- Slide 12. Parent 10 produces truncated draws below 10, not six tests at exactly 10. Task seeds differ across groups. Source: Section 4.1. -->

---

# 13 · Varied training improves fidelity and fall incidence

Hidden-context continuation versus frozen original policies:

| Method | Test MSE: frozen → trained | Falls/min: frozen → trained |
|---|---:|---:|
| BC | 0.17883 → 0.01995 | 0.367 → 0.044 |
| Mixed DAgger | 0.01634 → 0.00187 | 0.200 → 0.056 |
| GAIL-BC | 0.17918 → 0.02015 | 0.367 → 0.044 |
| GAIL-DM | 0.01665 → 0.00217 | 0.100 → 0.067 |

Progress remains around **32.8 m**, versus oracle **32.885 m**.
DAgger remains the strongest action-fidelity baseline in this comparison.

<!-- Slide 13. Frozen exposure is 30 minutes per method; continuation exposure 90. This includes more training history, not an isolated equal-compute effect of diversity. Sources: Sections 4–6. -->

---

# 14 · Does explicitly supplying slip help?

Input-conditioned minus hidden-context mean progress (m):

| Method | Continued lineage | Varied-only lineage |
|---|---:|---:|
| BC | +0.0989 | +0.0816 |
| Mixed DAgger | +0.0086 | −0.1206 |
| GAIL-BC | +0.0547 | +0.0630 |
| GAIL-DM | −0.0290 | +0.0532 |

Small, inconsistent benefits—not a general context advantage.

The oracle does not receive explicit slip, but can respond to its effects through observations. These results do **not** establish that context can never help.

<!-- Slide 14. Both columns use 90 deployments per architecture/family. Sources: Section 5.3 and Appendix F.4. -->

---

# 15 · The final extension: remove original pretraining

**Continuation:** original training → varied-domain training.

**Varied-only:** random BC/DAgger initialization → varied-domain training;
GAIL then starts from the corresponding varied-only BC/DAgger.

- Same incremental varied training allocations and evaluation rounds.
- Original policies and completed results preserved and reused.
- Additional direct-random GAIL tests isolate a different starting point.

**Important:** equal additional budgets do not equalize lifetime training history.

<!-- Slide 15. Scratch GAIL-BC is not an actor directly initialized randomly. Direct-random GAIL is a separate control. Source: Appendix F.1. -->

---

# 16 · More progress does not mean closer imitation

Hidden-context families; continuation → varied-only scratch:

| Method | Progress (m) ↑ | Test action MSE ↓ |
|---|---:|---:|
| BC | 32.784 → 33.192 | 0.01995 → 0.11177 |
| Mixed DAgger | 32.844 → 32.917 | 0.00187 → 0.01035 |
| GAIL-BC | 32.838 → 33.237 | 0.02015 → 0.11199 |
| GAIL-DM | 32.851 → 32.912 | 0.00217 → 0.01061 |

Scratch BC gains about **0.41 m**, but action error is **5.6× larger**.
Scratch DAgger still has about **10.8× lower error than scratch BC**.

<!-- Slide 16. All rows use 90 sampled deployments per cohort. Pretraining/history supports fidelity here, but is not uniformly best for progress. Source: Appendix F.2. -->

---

# 17 · Mechanics: no single winner either

Hidden families, continuation / scratch; **90 active minutes each**.

| Method | Falls | Joint occupancy SWD ↓ | Jerk rate ÷ 10⁶ ↓ |
|---|---:|---:|---:|
| BC | 4 / 4 | 0.0502 / 0.0574 | 40.712 / 39.352 |
| Mixed DAgger | 5 / 4 | 0.0435 / 0.0426 | 41.236 / 41.068 |
| GAIL-BC | 4 / 2 | 0.0468 / 0.0579 | 40.703 / 39.369 |
| GAIL-DM | 6 / 2 | 0.0428 / 0.0456 | 41.194 / 41.116 |

Sparse fall counts do not establish a robust population ranking.
Less jerk is a **kinematic smoothness** result, not measured energy efficiency.

<!-- Slide 17. Jerk-rate units are rad²/s⁶. SWD discards temporal order. Source: Appendix F.3. -->

---

# 18 · GAIL depends strongly on the starting policy here

Direct-random GAIL under the tested **70,000-step** recipe:

| Training / evaluation | Progress (m) | Targets reached | Falls / exposure |
|---|---:|---:|---:|
| Original / original | −0.121 | 0 | 0 / 5 min |
| Original / sampled | −0.216 | 0 | 12 / 30 min |
| Varied / sampled | −0.367 | 0 | 30 / 90 min |

Zero falls can mean unsuccessful standing, not good walking.
Supervised-initialized GAIL preserves useful walking, but shows no consistent broad improvement over its initializer.

<!-- Slide 18. This is the old consolidated anchored recipe, not the later optional 350k-step improvement experiment. Failure is configuration/budget-specific. Sources: Section 6.4, Appendix F.5. -->

---

# 19 · Probing: what can be read from the networks?

Freeze each network; fit simple regression readouts to its activations.

**30 locomotion targets**, including:

- Gait phase and support/contact proxies.
- Torso roll, pitch and angular motion.
- Foot geometry, commanded motion and phase-conditioned posture.
- Motion/posture 0.2 seconds ahead on oracle trajectories.

Shared oracle states; whole-episode train/validation/test splits.
Controls: raw observations, random networks and shuffled labels; expert included.

<!-- Slide 19. 30 traces split 10/5/15; 131 representations, 524 fits, 15,720 target scores. These are not independent experiments or all possible internal properties. -->

---

# 20 · Several locomotion properties are accessible

First-layer test R², continued hidden models:

| Representation | Phase × roll | Future velocity | Future pitch |
|---|---:|---:|---:|
| Raw observations | 0.3624 | 0.5951 | 0.5634 |
| Untrained random features | 0.4559 | 0.6114 | 0.5678 |
| BC | 0.6911 | 0.6330 | 0.5988 |
| Mixed DAgger | 0.7065 | 0.6294 | 0.5876 |

Training makes some relationships more linearly accessible.
However, information already present in inputs can also be decoded through random features.

<!-- Slide 20. Model entries average three training seeds. Future targets follow the oracle, not each learner's own actions. Source: Appendix F.6. -->

---

# 21 · Representation similarity is not a success metric

Hidden mixed DAgger:

| Measurement | Continuation | Scratch |
|---|---:|---:|
| Expert CKA similarity ↑ | 0.9212 | 0.9428 |
| Future-pitch probe R² ↑ | 0.5876 | 0.6101 |
| Action MSE ↓ | 0.00187 | 0.01035 |

Higher similarity and one better readout coexist with **worse action fidelity**.
Failed random-start GAIL also exposes future-velocity information (R² ≈ 0.6114).

CKA is not “percentage of expert knowledge”; decoding is not causal use.

<!-- Slide 21. Expert architecture differs; these comparisons support caution, not a mechanism explaining performance. Source: Appendix F.6. -->

---

# 22 · What the paper can conclude

1. **Task parity is not imitation parity.** Original progress is similar; action errors differ markedly.
2. **Mixed DAgger is a strong fidelity baseline**, including varied-only training, at its stated supervisory/optimization cost.
3. **Varied continuation retains fidelity and low fall incidence.** Scratch can gain progress without better imitation.
4. **Explicit slip and conservative GAIL add no consistent broad advantage** in this study.
5. **Probes complement behaviour; they cannot replace it.**

These are findings about the implemented protocols, not universal algorithm rankings.

<!-- Slide 22. Answer the proposal's stability, resources and expert-performance questions without introducing a new research space. -->

---

# 23 · Boundaries to state plainly

- Selected budgets are standardized; total compute and lifetime history are not.
- Few independent training replicas; continuation shares pretrained parents.
- Paired/reused rounds are valuable controls, not new independent evidence.
- Shared gait/IK structure limits claims about learning locomotion from scratch.
- Slip is a simulator parameter; tests are fixed-surface deployments.
- Probes show accessibility, not causality; no demonstrated R1 transfer.

Do not add the later optional wide-slip/RL or larger-budget GAIL results to this paper’s completed evidence by implication.

<!-- Slide 23. No claim that all internal representations were identified, or that jerk measures motor wear. -->

---

# 24 · An eight-page paper, not a project diary

| Body section | Approx. pages | Main purpose |
|---|---:|---|
| Introduction + related work | 1.5 | Questions and methodological context |
| Methods | 1.5 | Interfaces, budgets, paired evaluations |
| Original environment | 1.0 | Establish progress–fidelity distinction |
| Frozen + varied training | 1.5 | Environmental effects and context |
| Initialization comparison | 1.0 | Progress–fidelity trade-off |
| Probes + discussion | 1.5 | Supporting explanation and limitations |

References are outside the limit. Keep full tables and experimental history in the supporting report.

<!-- Slide 24. This is an editorial allocation, not a claim about a venue's supplementary-material policy. -->

---

# 25 · Sources and final takeaway

**Paper sources:** Ross et al. (2011), DAgger; Ho & Ermon (2016), GAIL;
Schulman et al. (2017), PPO; Orsini et al. (2021), adversarial imitation;
Alain & Bengio (2016), probes; Hewitt & Liang (2019), probe controls;
Kornblith et al. (2019), CKA.

**Project evidence:** [consolidated report, including Appendix F](../CONSOLIDATED_RESULTS_REPORT_2026-10-15.md), [candidate paper draft](../FINAL_PAPER_DRAFT.md), [slide companion](CONSOLIDATED_PAPER_NOTES.md).
Full references and attribution boundaries: report Appendix E.

**Evaluate imitation as a trade-off between useful walking, fidelity, stability and cost—not as a single leaderboard score.**

<!-- Slide 25. All numerical claims trace to the saved consolidated report. No simulator run or new result is introduced by this deck. -->
