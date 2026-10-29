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

# 0 · Imitating RoboCup locomotion
## What the consolidated experiments establish

Original environment → varied surfaces → initialization → representations

**Similar walking performance does not imply similar imitation fidelity.**

A focused narrative for the final eight-page report.
[Background and slide notes](CONSOLIDATED_PAPER_NOTES.md)

<!-- Slide 0. References are outside the eight-page body limit. Evidence ends with the completed scratch comparison and its probes. -->

---

# 1 · Redid experiments — how and why

**Why:** historical runs used evolving collection, recovery and training procedures, making comparisons harder to interpret.

**How:** one consolidated protocol for BC, mixed DAgger, GAIL-BC and GAIL-DM:

- Long runs and short turns; shared control interface.
- Explicit allocations and managed synchronous servers.
- Common test rounds, fall handling and metrics.

**Outcome:** compare training protocols, not different generations of tooling.

<!-- Slide 1. GAIL-DM starts from mixed DAgger; FC Portugal remains the expert for every method. The old historical DAgger/GAIL lineage is separate. -->

---

# 2 · Research questions and evidence

| Research question | Evidence to prioritize |
|---|---|
| How stable are the learned controllers? | Falls, joint jerk, posture and useful movement |
| What do the tested resources buy? | Fidelity and performance at stated data/interaction budgets |
| Do policies match the expert? | Progress, velocity, tracking and action error together |

Surface variation, initialization and context **extend these questions**.
Probing supports their interpretation; it is not a separate transfer claim.

<!-- Slide 2. Do not claim convergence/sample-complexity curves from endpoint budget comparisons. -->

---

# 3 · Experimental progression

| Stage | What changes? |
|---|---|
| Original scene | Train and evaluate the four principal methods |
| Frozen policies | Change slip without updating weights |
| Varied training | Train across surfaces; hide or supply slip |
| Remove pretraining | Compare continuation with varied-only learning |
| Probing | Examine shared-state features and expert similarity |

Training slips: **0.025, 0.1, 0.2, 1, 10**.
Evaluation: Gaussian draws around these values, **fixed throughout each run**.
Each deployment: **3,000 active steps / 60 simulated seconds**.

<!-- Slide 3. Gaussian standard deviation is half its mean, truncated to (0,10]. Main results pool 30 sampled rounds. Frozen models have 30 deployments; varied families have three training seeds and 90. -->

---

# 4 · Standardised allocations

| Method | Main allocation |
|---|---|
| BC | 170,000 expert examples |
| Mixed DAgger | 100,000 seed examples + 7 × 10,000 learner-state labels |
| Each GAIL | 170,000-example expert pool + 70,000 PPO interactions |

BC: **17 million** supervised example exposures.
Mixed DAgger: **54 million** exposures through repeated aggregate training.

GAIL also inherits the cost of its initializer.
Report data, oracle queries, interactions and optimization separately.

<!-- Slide 4. BC has 66,500 gradient updates; DAgger 211,050. A unique example, an environment transition and an optimizer step are different units. -->

---

# 5 · Success measures

| Axis | Main measurements |
|---|---|
| Task/environment performance | Target progress, normalized score, forward velocity, tracking |
| Imitation fidelity | Held-out action MSE; joint occupancy distance |
| Mechanical stability | Falls per active minute, torso RMS, angular jerk |
| Representation | Held-out probe R²; expert CKA similarity |

**Normalized score = (policy progress − random progress) / (expert progress − random progress).**

Score 1 means oracle parity **on that surface**, not unchanged absolute performance.

<!-- Slide 5. Score zero is a reference, not a floor. MSE is per latent-action element; jerk is kinematic, not torque. -->

---

# 6 · Original scene results

Five 60-second deployments per controller.

| Controller | Progress (m) ↑ | Normalized score | Test action MSE ↓ |
|---|---:|---:|---:|
| FC Portugal | 44.446 | 1.0000 | — |
| BC | 44.630 | 1.0044 | 0.11047 |
| Mixed DAgger | 44.364 | 0.9985 | 0.01535 |
| GAIL-BC | 44.556 | 1.0025 | 0.11042 |
| GAIL-DM | 44.150 | 0.9934 | 0.01572 |

**Compared with the earlier BC-led baseline:** consolidated mixed DAgger now has **7.2× lower action MSE**; BC still has slightly greater mean progress.

These are newly trained models—not relabelled checkpoints. DAgger also receives more optimizer exposure (54m vs 17m examples).

<!-- Slide 6. All four students recorded zero falls. Earlier BC-led refers to the historical baseline assessment, not a claim that old and new MSE values are directly comparable. Mixed DAgger uses100k seed examples plus70k learner-state labels with long runs/short turns. Sample exposures are not unique examples. Small progress differences do not establish superiority. Source: report Sections3 and7. -->

---

# 7 · Frozen policies on varied surfaces

Mean progress across six sampled rounds per Gaussian parent:

| Parent mean—not actual fixed slip | Oracle (m) | Frozen BC (m) | Frozen DAgger (m) |
|---|---:|---:|---:|
| 0.025 | 42.148 | 43.008 | 42.531 |
| 0.1 | 39.341 | 38.679 | 39.361 |
| 0.2 | 37.190 | 37.190 | 36.404 |
| 1 | 28.270 | 27.584 | 27.690 |
| 10 | 17.476 | 17.679 | 17.245 |

Policies can remain near normalized score 1 while **both they and the oracle slow substantially**. Always report absolute progress too.

<!-- Slide 7. Parent 10 produces truncated draws below 10, not six tests at exactly 10. Task seeds differ across groups. Source: Section 4.1. -->

---

# 8 · Varied training

Hidden-context continuation versus frozen original policies:

| Method | Test MSE: frozen → trained | Falls/min: frozen → trained |
|---|---:|---:|
| BC | 0.17883 → 0.01995 | 0.367 → 0.044 |
| Mixed DAgger | 0.01634 → 0.00187 | 0.200 → 0.056 |
| GAIL-BC | 0.17918 → 0.02015 | 0.367 → 0.044 |
| GAIL-DM | 0.01665 → 0.00217 | 0.100 → 0.067 |

Progress remains around **32.8 m**, versus oracle **32.885 m**.
DAgger remains the strongest action-fidelity baseline in this comparison.

<!-- Slide 8. Frozen exposure is 30 minutes per method; continuation exposure 90. This includes more training history, not an isolated equal-compute effect of diversity. Sources: Sections 4–6. -->

---

# 9 · Explicit slip supply

Input-conditioned minus hidden-context progress (m):

| Method | Continued lineage | Varied-only lineage |
|---|---:|---:|
| BC | +0.0989 | +0.0816 |
| Mixed DAgger | +0.0086 | −0.1206 |
| GAIL-BC | +0.0547 | +0.0630 |
| GAIL-DM | −0.0290 | +0.0532 |

**Small, inconsistent gains.** Knowing slip did not broadly improve this task.

The expert can respond to surface effects through observations even without an explicit slip input.

<!-- Slide 9. Hidden context means no explicit slip input, not no hidden layers. Input conditioning adds one normalized scalar. Sources: Section 5.3 and Appendix F.4. -->

---

# 10 · Remove original pretraining

**Continuation:** original training → varied-domain training.

**Varied-only:** random BC/DAgger initialization → varied-domain training;
GAIL then starts from the corresponding varied-only BC/DAgger.

- Same incremental varied training allocations and evaluation rounds.
- Original policies and completed results preserved and reused.
- Additional direct-random GAIL tests isolate a different starting point.

**Important:** equal additional budgets do not equalize lifetime training history.

<!-- Slide 10. Scratch GAIL-BC is not an actor directly initialized randomly. Direct-random GAIL is a separate control. Source: Appendix F.1. -->

---

# 11 · Without pretraining: results and meaning

Hidden-context families; continuation → varied-only scratch:

| Method | Progress (m) ↑ | Test action MSE ↓ |
|---|---:|---:|
| BC | 32.784 → 33.192 | 0.01995 → 0.11177 |
| Mixed DAgger | 32.844 → 32.917 | 0.00187 → 0.01035 |
| GAIL-BC | 32.838 → 33.237 | 0.02015 → 0.11199 |
| GAIL-DM | 32.851 → 32.912 | 0.00217 → 0.01061 |

Scratch BC gains about **0.41 m**, but action error is **5.6× larger**.
Scratch DAgger still has about **10.8× lower error than scratch BC**.

<!-- Slide 11. All rows use 90 sampled deployments per cohort. Pretraining/history supports fidelity here, but is not uniformly best for progress. Source: Appendix F.2. -->

---

# 12 · Mechanical results

Hidden families, continuation / scratch; **90 active minutes each**.

| Method | Falls | Joint occupancy SWD ↓ | Jerk rate ÷ 10⁶ ↓ |
|---|---:|---:|---:|
| BC | 4 / 4 | 0.0502 / 0.0574 | 40.712 / 39.352 |
| Mixed DAgger | 5 / 4 | 0.0435 / 0.0426 | 41.236 / 41.068 |
| GAIL-BC | 4 / 2 | 0.0468 / 0.0579 | 40.703 / 39.369 |
| GAIL-DM | 6 / 2 | 0.0428 / 0.0456 | 41.194 / 41.116 |

Sparse fall counts do not establish a robust population ranking.
Less jerk is a **kinematic smoothness** result, not measured energy efficiency.

<!-- Slide 12. Jerk-rate units are rad²/s⁶. SWD discards temporal order. Source: Appendix F.3. -->

---

# 13 · Probes interpret the behavioural results

| Earlier finding | What we ask of the representations |
|---|---|
| DAgger has lower action error | Is phase-conditioned posture more accessible? |
| Continuation preserves fidelity | What changes when pretraining is removed? |
| GAIL adds limited improvement | Do its readouts differ from its initializer? |

**Ridge regression** reads physical quantities from frozen features.
**CKA** compares expert/learner feature geometry on shared states.

Held-out episodes and raw/random/shuffled controls distinguish accessibility from mere input information—not causality.

<!-- Slide 13. These are post-hoc interpretive questions, not preregistered mechanisms. Selected first-layer readouts are on shared oracle trajectories; future labels follow oracle actions. Full methods and other targets remain in notes/report. -->

---

# 14 · Fidelity differences and limited GAIL change

Continued hidden models, evaluated on varied surfaces:

| Model | Action MSE ↓ | Phase × roll R² ↑ | Future velocity R² ↑ |
|---|---:|---:|---:|
| BC | 0.01995 | 0.6911 | 0.6330 |
| Mixed DAgger | 0.00187 | 0.7065 | 0.6294 |
| GAIL-BC | 0.02015 | 0.6910 | 0.6330 |
| GAIL-DM | 0.00217 | 0.7063 | 0.6294 |

DAgger’s fidelity advantage accompanies a better posture readout—not better prediction of every property.
GAIL’s nearly unchanged readouts are consistent with its limited behavioural change.

<!-- Slide 14. Sources: report Sections 5–6 and Appendix F.6. MSE and probes use their respective held-out datasets, not identical samples. Differences are descriptive; phase × roll does not establish the cause of lower action error or fewer falls. -->

---

# 15 · Pretraining: fidelity is not global similarity

Hidden mixed DAgger; continuation versus varied-only scratch:

| Measurement | Continuation | Scratch |
|---|---:|---:|
| Action MSE ↓ | 0.00187 | 0.01035 |
| Phase × roll R² ↑ | 0.7065 | 0.6935 |
| Future-pitch R² ↑ | 0.5876 | 0.6101 |
| Expert CKA ↑ | 0.9212 | 0.9428 |

Continuation combines closer actions with a better posture readout.
Scratch improves other readouts and similarity **without improving fidelity**.

**Interpretation:** training history changes what is accessible; “more expert-like features” alone does not explain better imitation.

<!-- Slide 15. Source: Appendix F.6. CKA is not percentage knowledge. No causal intervention establishes that phase-conditioned posture produces the fidelity advantage, or explains scratch progress gains. -->

---

# 16 · Paper conclusions

- **Task parity is not imitation parity:** similar progress masks large action-error differences.
- **Mixed DAgger is a strong fidelity baseline**, with greater query/optimization costs.
- **Training history matters:** scratch can gain progress while losing fidelity.
- **Explicit context and conservative GAIL show no consistent broad advantage.**
- **Probes support limited GAIL change, but do not reduce fidelity differences to a single representation score.**

Direct-random GAIL did not learn useful walking at the tested 70k-step budget; this is not a universal limit of GAIL.

<!-- Slide 16. Varied direct-random GAIL averaged -0.367m and reached no targets. The later larger-budget recipe is outside these results. -->

---

# 17 · Boundaries

- Standardised allocations **do not equalise total compute or lifetime training**.
- Few independent training replicas; continued models share pretrained parents.
- Paired/reused rounds are controls, not additional independent evidence.
- Policies retain the existing gait/IK machinery.
- Fixed-per-run slip is not spatially or continuously changing terrain.
- Jerk is kinematic, not measured energy; probes are descriptive, not causal.

**Scope:** conclusions about these implemented protocols—not universal algorithm rankings.

<!-- Slide 17. The original installed scene is distinct from corrected scalar slip 0.1. No demonstrated cross-robot transfer is claimed. -->

---

# 18 · Outside the paper — worth continuing

| Curiosity-driven direction | Question left open |
|---|---|
| Wider-slip stress tests → context-aware RL | Where does the oracle fail, and can reward-driven learning go beyond imitation? |
| Larger-budget GAIL: random, BC, mixed DAgger starts | Can less constrained refinement improve on its initializer? |
| R1 transfer and representation interventions | Which source-policy properties actually help another robot learn? |

These are **separate follow-on studies**, not evidence for the current paper.

Stress and GAIL pipelines are implemented; RL extensions remain conditional.
R1 work is separate and has no established transfer result here.

<!-- Slide 18. Wider-slip oracle sweep and the optional 350k-step GAIL pipeline are implemented; current live results were not assessed for this deck. Do not imply that context-aware RL or causal transfer experiments are complete. See notes for follow-on questions and sources. -->
