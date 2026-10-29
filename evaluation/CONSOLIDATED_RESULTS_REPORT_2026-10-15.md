# Consolidated imitation-learning results: experimental design, environmental robustness and representations

Originally reported 15 October 2026; updated 16 October 2026. Sections 1–10 and Appendices A–D describe the parent consolidated generation: 28 policies, 1,020 deployments and 300 probe fits. **Appendix F adds the completed scratch comparison**, bringing the total to 56 policies, 1,949 deployments and 524 fits. Appendix E retains the paper reference guide. Historical four-policy, slip-sensitivity and FiLM studies remain separate.

**Latest findings:** Scratch supervised lineages walk successfully without original-environment pretraining, but continuation retains approximately fivefold better action fidelity. Mixed DAgger remains the strongest supervised fidelity baseline. Direct random-initialized GAIL fails to acquire useful locomotion at this budget; explicit context remains inconsistent. See Appendix F for numerical comparisons, design and interpretation.

**Reading guide:** Sections 1–2 explain design and metrics; Section 3 covers original-environment results; Section 4 tests frozen policies on changed surfaces; Section 5 covers varied training and context; Section 6 focuses on DAgger; Section 7 accounts for training resources; Section 8 interprets probes; Sections 9–10 synthesize findings and limitations. Appendices retain additional numerical comparisons and source links.

## Executive summary

The main result is not a large gain in distance travelled. It is the separation between **task performance**, **imitation fidelity**, and **mechanical reliability**. In the original environment, all four learned controllers achieve approximately oracle-level target progress without recorded falls. Across the sampled surfaces, the original frozen policies remain close to oracle-normalized performance, but incur more falls and larger action errors. Varied-domain continuation produces substantially closer imitation and generally fewer falls, while average progress remains near the same oracle ceiling.

Mixed DAgger is particularly important: it has markedly lower action error than BC in this generation, and varied-trained DAgger retains that advantage. However, its training also uses more supervised optimizer work, so this cannot be attributed to learner-state queries alone. Explicit slip conditioning adds small and inconsistent behavioural gains; it is not a clear replacement for a strong hidden-context model.

The probes identify accessible phase, posture, contact-proxy, command and short-horizon motion information. They also show that the GAIL representations remain almost indistinguishable, at the displayed precision, from their supervised initializers. High decodability is not proof that a feature is used for control, nor proof of transfer to another robot.

## 1. Reading the experimental design

### 1.1 Three stages, followed by representation analysis

1. **Original environment:** train BC, mixed DAgger, GAIL-BC and GAIL-DM, then evaluate these four in the original installed scene.
2. **Frozen-policy environmental tests:** take those exact four checkpoints, change the environment, and evaluate without further learning.
3. **Varied-domain training and testing:** continue learning across five fixed training surfaces, with either hidden slip or explicit slip input; evaluate these new policies on the same sampled-slip rounds as the frozen models.
4. **Offline probes:** compare all actors and the expert on the same recorded oracle states, including matched random-network and raw-input controls.

“Original” is deliberately not called zero slip. The original arm preserves the installed legacy scene byte-for-byte. Corrected scalar slip 0.1 is a separate physical condition, not a synonymous nominal label. Likewise, “Gaussian evaluation” means **draw one positive slip per round, then keep it fixed for the whole deployment**. It is not a spatially varying field or a new draw at every foot contact.

### 1.2 Which models were trained?

DAgger-M denotes mixed long/short-run DAgger. GAIL-DM denotes GAIL initialized from this new **mixed DAgger**, not the original-DAgger GAIL of the historical study. In every case FC Portugal remains the expert; BC/DAgger initialize GAIL but do not replace the expert demonstration source.

| Family | Training environment | Network observations | Replicas | Models |
|---|---|---|---|---|
| Original four | Installed original scene | 63 inputs; slip hidden | Seed42 | 4 |
| Varied hidden four | Scalar slip0.025,0.1,0.2,1,10 | 63 inputs; slip hidden | Seeds42/43/44 | 12 |
| Varied input four | Same five surfaces | 63 inputs + normalized slip | Seeds42/43/44 | 12 |

Each actor has two256-unit ReLU hidden layers and16 outputs. Input-conditioned actors append `log1p(slip)/log(11)`, creating64inputs; the extra input column begins at zero when expanding the original63-input checkpoint. These are not64-to16 single-layer policies. The16 outputs are latent gait-generator actions/residuals executed through the existing walking machinery, not22 independent motor velocities.

Original BC and DAgger start from the same seeded random weights. Varied BC and DAgger start from their corresponding newly trained original controller. Each varied GAIL actor starts from the corresponding varied supervised controller with the same architecture and seed. Therefore the three varied replicas are **continuation replicas sharing an original pretrained ancestor and expert pools**, not three independent repetitions of the entire from-scratch experiment.

### 1.3 Training rounds and data composition

Data collection balances long runs and short target changes. Short targets are sampled 1–2 m from the current position in the own-half region, with a 0.4 m arrival threshold; long targets use the own-half sampler with a 0.5 m threshold. This maintains the long-run/short-turn combination requested for mixed DAgger. All controllers share the simulator contract, observation format and action interface.

| Procedure | Starting supervised data | New learner transitions | Final expert-labelled training pool | Update design |
|---|---|---|---|---|
| BC | 170,000 oracle examples | 0 | 170,000 | 100 supervised epochs |
| DAgger-M | 100,000 oracle examples | 7×10,000 | 100,000 +70,000 =170,000 | 50 epochs initially and after each aggregation |
| GAIL-BC | 170,000 static oracle examples +BC initializer | 7×10,000 | 170,000 static examples | Each round: discriminator1epoch +PPO up to3epochs |
| GAIL-DM | 170,000 static oracle examples +DAgger-M initializer | 7×10,000 | 170,000 static examples | Same GAIL update schedule |

The seed pool is 100,000 examples; an extra 70,000 oracle examples completes the shared BC/GAIL pool. Separate validation and test pools contain 10,000 examples each per environment domain. These are not included in the 170,000 training count. BC/GAIL share oracle data rather than collecting separate copies for every replica; DAgger adds its own learner-state labels.

For varied training, **each collection round has ten segments: five surfaces ×two target modes**. The 100,000-example seed pool has 10,000 per segment; the 70,000-example extra pool has 7,000; each validation/test pool has 1,000. Each10,000-step DAgger/GAIL learner round contributes 1,000 transitions per surface/mode segment. Thus every learner iteration sees all five training slips, rather than proceeding through a low-to-high-slip curriculum. Original-domain training retains the same ten-segment structure, but all segments use the unchanged original scene.

| DAgger stage | New learner labels | Aggregated training examples | Oracle execution probability β | Supervised epochs |
|---|---|---|---|---|
| Initial | 0 | 100,000 | Oracle seed data | 50 |
| 1 | 10,000 | 110,000 | 0.8 | 50 |
| 2 | 10,000 | 120,000 | 0.6 | 50 |
| 3 | 10,000 | 130,000 | 0.4 | 50 |
| 4 | 10,000 | 140,000 | 0.2 | 50 |
| 5 | 10,000 | 150,000 | 0.0 | 50 |
| 6 | 10,000 | 160,000 | 0.0 | 50 |
| 7 | 10,000 | 170,000 | 0.0 | 50 |

DAgger queries the oracle at every collected learner-state example, including when the learner executes the action. β chooses **which action is executed**, not whether an expert label is obtained. It decays 0.8,0.6,0.4,0.2,0,0,0. BC remains offline supervised training. GAIL instead samples its stochastic actor during collection, stores executed actions/log-probabilities/values, and learns from discriminator rewards; evaluation uses deterministic actor means.

BC/DAgger use Adam at 1e-4, batch256, selecting the lowest validation-MSE checkpoint within each update. GAIL uses actor/critic/discriminator learning rates 3e-6/3e-4/1e-4, gradient-penalty regularization, a supervised anchor and KL safeguards. Rollout segment ends and fall resets are handled as separate boundaries for return estimation. These conservative GAIL settings matter when interpreting the small later changes in representations.

### 1.4 Evaluation rounds and what the seeds mean

| Suite | Environmental rounds | Controllers each round | Deployments | Role |
|---|---|---|---|---|
| Original | 5 task seeds:101/202/303/404/505 | 4 original +oracle +random | 30 | Original-scene performance |
| Grouped Gaussian | 5parent means ×3draws | 28 learned +oracle +random | 450 | Common varied-environment comparison |
| Confirmation Gaussian | 5parent means ×3other draws | Same30 | 450 | Separate reused confirmation schedule |
| Corrected nominal | Scalar0.1 ×3task seeds | Same30 | 90 | Nominal retention under corrected physics |
| Total | 38environmental rounds | — | 1,020 | 3,060,000 active evaluation steps |

Every deployment has 3,000 active steps at 20 ms per step: 60 seconds of scored simulated control. Startup and fall recovery add simulator time outside that active duration. Each controller receives a fresh owned server, the same physical slip and task seed for a given round, own-half target sampling, and the same fall/reset rules. Checkpoints are frozen throughout evaluation. The server scene hash/import is verified; this is verification of the loaded configuration, not direct live ODE readback.

Training seeds 42/43/44 control continuation/training randomness. Evaluation seeds control the task sequence; the Gaussian schedule uses its own reproducible seed construction. They are different sources of variation. Pairing the seed makes starting/task randomness comparable, but achieved trajectories and target-switch timing can still diverge between policies.

The two sampled suites use parent means 0.025,0.1,0.2,1,10 and standard deviations equal to half each mean, truncated to(0,10]. A parent mean is not the empirical mean after truncation, particularly for the parent 10 distribution. The actual realized ranges are listed below. The existing schedules were previously inspected elsewhere; “confirmation” here does not mean a newly blind dataset.

| Parent μ | Parent σ | Unique draws | Realized minimum | Realized maximum | Realized mean |
|---|---|---|---|---|---|
| 0.025 | 0.0125 | 6 | 0.000285 | 0.053355 | 0.028859 |
| 0.1 | 0.05 | 6 | 0.062716 | 0.157545 | 0.111356 |
| 0.2 | 0.1 | 6 | 0.140220 | 0.408996 | 0.238566 |
| 1.0 | 0.5 | 6 | 0.212564 | 1.456805 | 0.980224 |
| 10.0 | 5.0 | 6 | 2.904625 | 8.959668 | 6.096283 |

### 1.5 How numbers are aggregated

Stage 1 averages five deployments per original policy. For the main environmental tables, grouped and confirmation are combined: 30 deployments per frozen policy and 30 per varied training replica, or 90 per varied family. Because each replica has the same 30 rounds, pooling these balanced runs gives the same point mean as averaging the three replica means. **It does not make 90 independent training experiments.** Fall counts are always shown with their exposure; replica ranges and paired context differences are reported separately.

The three corrected-nominal rounds are not mixed into the 30 sampled-round robustness summary. The original five-seed scene is not mixed with corrected scalar 0.1. Original checkpoints are only one training seed, and all varied replicas inherit shared parents. The report is primarily descriptive; it does not use individual frames as independent statistical observations or assert significance from small point differences.

## 2. Metric definitions and practical interpretation

| Metric | What it measures | How to read it |
|---|---|---|
| Target progress J (m) | Sum of reduction in distance to the current target | Higher is better; not total path length or final straight-line displacement |
| Normalized return | (Jpolicy−Jrandom)/(Joracle−Jrandom), paired within each round | 0=random reference;1=oracle parity; may exceed1 or be negative |
| Held-out action MSE | Mean squared error per16latent-action element on10,000held-out oracle examples | Lower=closer local imitation on shared expert states |
| On-policy action MSE | Same comparison on states each controller visits | Lower=closer local imitation, but state distributions differ |
| Joint occupancy SWD | 256-projection sliced Wasserstein over22joint angles, expert-standardized | Lower=closer occupancy; unitless and not exact22-D transport |
| Forward velocity (m/s) | Body-heading projected translation per active step | Higher normally indicates forward locomotion; target progress still matters |
| Direction tracking error | Norm of difference between unit movement and target directions | Lower=better; dimensionless, not degrees; undefined movement assigned1 |
| Falls/active minute | Detected balance losses divided by scored active minutes | Lower=better; reset time excluded; not a count restricted to unrecoverable falls |
| Torso roll/pitch RMS (degrees) | Square-root mean squared orientation about zero | Lower means less angular excursion, not necessarily better locomotion by itself |
| Angular jerk rate | Integrated squared third joint-angle derivative /valid differentiated duration | Lower=less high-frequency kinematic change, not measured torque or energy |
| Probe R² /RMSE | Held-out predictability from a representation | Higher R²/lower RMSE=more linearly accessible labelled information |
| CKA /participation ratio | Representation geometry similarity /effective variance dimension | Descriptive structure, not percent knowledge or causal usefulness |

The originally proposed squared-vector action error is 16 times the element-mean MSE tabulated here; the action interface is latent 16-D, not directly 22 joint-velocity commands. The occupancy metric discards temporal ordering, so “similar trajectories” must not be inferred from SWD alone. It is normalized using each matched oracle run’s joint variation.

Jerk uses joint positions converted to radians, five-frame smoothing, then a third finite difference at 20 ms. It is computed separately inside recovery-delimited episodes. The integrated quantity has units rad²/s⁵; its duration-normalized rate has units rad²/s⁶. Tables scale the rate by 10⁶ for readability. The metric is sensitive to derivative estimation, smoothing and episode edges. It is a **kinematic smoothness proxy**, not evidence of lower actuator torque, energy consumption or hardware wear.

Normalized scores are not clipped. A score near 1 means matching an oracle facing the same surface, even when both walk much more slowly than in the original environment. The random policy is a reference, not an absolute mathematical floor. All 1,020 normalization denominators in these results are valid.

## 3. Stage 1 — Original-environment policies in the original environment

### 3.1 Task performance and mechanical stability

| Controller | Progress m | Normalized score | Forward m/s | Direction error | Falls /5min | Jerk rate ×10⁶ |
|---|---|---|---|---|---|---|
| Oracle | 44.446 | 1.0000 | 0.7590 | 0.6571 | 0 | 46.564 |
| BC | 44.630 | 1.0044 | 0.7591 | 0.6545 | 0 | 44.995 |
| DAgger-M | 44.364 | 0.9985 | 0.7517 | 0.6583 | 0 | 46.474 |
| GAIL-BC | 44.556 | 1.0025 | 0.7621 | 0.6561 | 0 | 44.929 |
| GAIL-DM | 44.150 | 0.9934 | 0.7505 | 0.6601 | 0 | 46.429 |
| Random | -1.159 | 0.0000 | -0.0776 | 1.1504 | 151 | 196.065 |

All learned policies are close to the oracle’s 44.446 m target progress and have zero recorded falls in these five active minutes each. The best point score is BC 1.0044, but the original per-policy descriptive 95% intervals all include 1. This supports successful oracle-level task imitation, not a meaningful ordering of the four controllers by return. Zero observed falls over five minutes does not establish a zero true fall probability.

For numerical context, the stored normalized-return intervals are BC[0.99359,1.01517], DAgger-M[0.98766,1.00927], GAIL-BC[0.99586,1.00911], GAIL-DM[0.98661,1.00015]. These use mean±1.96×standard error across five deployments; they are approximate descriptive intervals, not training-seed uncertainty or a multiple-comparison test.

### 3.2 Imitation fidelity separates models that look equally successful

| Controller | Held-out MSE | On-policy MSE | Joint SWD | Roll RMS° | Pitch RMS° |
|---|---|---|---|---|---|
| BC | 0.110467 | 0.074445 | 0.04503 | 2.303 | 4.198 |
| DAgger-M | 0.015345 | 0.007821 | 0.03909 | 2.317 | 4.058 |
| GAIL-BC | 0.110423 | 0.072104 | 0.03596 | 2.361 | 4.157 |
| GAIL-DM | 0.015722 | 0.008524 | 0.03674 | 2.325 | 4.134 |

Mixed DAgger’s held-out MSE is 0.015345 versus BC 0.110467: approximately 86.1% lower. Its on-policy MSE is 0.007821 versus 0.074445: approximately 89.5% lower. This is a strong fidelity difference despite nearly identical progress. GAIL-DM largely retains the mixed-DAgger fidelity scale; GAIL-BC remains close to BC’s error scale.

BC/GAIL-BC have somewhat lower jerk rates, but DAgger’s joint occupancy is at least comparable and its action error is much smaller. No single scalar identifies the universally “best” gait. In particular, the original task is sufficiently forgiving that considerably different latent-action errors can produce similar progress.

This is a new generation with standardized unique-example counts, a changed mixed-DAgger GAIL lineage and newly collected data. Earlier reports in which BC led on some fidelity metrics should not be silently overwritten or pooled with these values. This study establishes the current batch’s ranking, not a universal algorithmic ordering across all previous experimental designs.

## 4. Stage 2 — Frozen original policies on sampled-slip surfaces

No policy weights change in this stage. This isolates how the original controllers behave when the environment changes, before any varied-domain continuation.

| Controller | Progress m | Normalized score | Held-out MSE | On-policy MSE | Joint SWD | Falls /30min | Falls/min |
|---|---|---|---|---|---|---|---|
| Oracle | 32.885 | 1.0000 | — | 0.000000 | 0.00000 | 3 | 0.100 |
| BC | 32.828 | 0.9984 | 0.178830 | 0.290978 | 0.09835 | 11 | 0.367 |
| DAgger-M | 32.646 | 0.9920 | 0.016337 | 0.026471 | 0.06570 | 6 | 0.200 |
| GAIL-BC | 32.774 | 0.9946 | 0.179184 | 0.223490 | 0.08420 | 11 | 0.367 |
| GAIL-DM | 32.497 | 0.9814 | 0.016648 | 0.028697 | 0.06279 | 3 | 0.100 |
| Random | 0.065 | 0.0000 | — | 22.000882 | 0.69450 | 704 | 23.467 |

Average oracle progress drops to 32.885 m across the sampled surfaces, about 26.0% below its original-scene mean. This broad comparison mixes different scene physics and task seeds, so it is descriptive rather than a matched estimate of a slip coefficient’s causal effect. Within the corrected varied suite, the slip-group analysis below makes the environmental burden much clearer.

The frozen policies still achieve scores 0.9814–0.9983 relative to the same-round oracle, yet the stability/fidelity picture changes. BC and GAIL-BC each record 11 falls in 30 active minutes, compared with 3 for the oracle. Mixed DAgger has 6, and GAIL-DM 3. Frozen BC’s varied-pool held-out MSE is 0.178830 and on-policy MSE 0.290978. Mixed DAgger remains much closer at 0.016337 and 0.026471 respectively.

This demonstrates why oracle normalization must be accompanied by absolute performance and stability: high normalized return does not mean the environment had little effect. A controller can recover, continue scoring progress and still have a materially worse fall profile.

### 4.1 Environmental burden by Gaussian parent group

Each row below combines the three grouped and three confirmation draws for that parent distribution. The labels identify Gaussian **parents**, not exact fixed physical values. In particular, parent 10 produced actual slips 2.905–8.960, not six tests at 10.

| Parent μ | Oracle progress m | Frozen BC m | Frozen DAgger-M m | Frozen GAIL-BC m | Frozen GAIL-DM m | Oracle falls /6min |
|---|---|---|---|---|---|---|
| 0.025 | 42.148 | 43.008 | 42.531 | 42.827 | 42.330 | 1 |
| 0.1 | 39.341 | 38.679 | 39.361 | 38.840 | 39.106 | 0 |
| 0.2 | 37.190 | 37.190 | 36.404 | 36.687 | 36.935 | 0 |
| 1.0 | 28.270 | 27.584 | 27.690 | 28.371 | 27.730 | 0 |
| 10.0 | 17.476 | 17.679 | 17.245 | 17.146 | 16.386 | 2 |

Oracle mean progress decreases from 42.148 m in the lowest parent group to 17.476 m in the highest, a 58.5% reduction. Frozen controllers follow a similar decline. These groups also use different task seeds, but the broad monotonic environmental trend is shared across controllers. A score near 1 at high slip should therefore be interpreted as keeping pace with a degraded oracle, not preserving low-slip speed.

## 5. Stage 3 — Training across varied slip

### 5.1 Main hidden-context and input-conditioned comparison

The following means cover the same 30 sampled rounds used for the frozen models. Each varied-family row contains three continuation replicas, 90 deployments in total. Hidden models must act using their ordinary sensors/history-related gait inputs; input models additionally receive the actual scalar slip.

| Family | Progress m | Normalized score | Held-out MSE | On-policy MSE | Joint SWD | Falls /90min |
|---|---|---|---|---|---|---|
| Varied hidden BC | 32.784 | 0.9984 | 0.019949 | 0.021086 | 0.05019 | 4 |
| Varied hidden DAgger-M | 32.844 | 0.9989 | 0.001866 | 0.002119 | 0.04346 | 5 |
| Varied hidden GAIL-BC | 32.838 | 1.0009 | 0.020149 | 0.021876 | 0.04680 | 4 |
| Varied hidden GAIL-DM | 32.851 | 1.0000 | 0.002167 | 0.002449 | 0.04285 | 6 |
| Varied input BC | 32.883 | 1.0018 | 0.019857 | 0.020813 | 0.04886 | 6 |
| Varied input DAgger-M | 32.853 | 1.0007 | 0.001902 | 0.002187 | 0.04827 | 8 |
| Varied input GAIL-BC | 32.893 | 1.0018 | 0.020138 | 0.021227 | 0.05401 | 5 |
| Varied input GAIL-DM | 32.822 | 0.9975 | 0.002278 | 0.002549 | 0.05091 | 4 |

Varied hidden BC reduces held-out error from the frozen BC’s 0.178830 to 0.019949, an 88.8% reduction on the same varied expert test pool. Varied hidden DAgger reduces 0.016337 to 0.001866, an 88.6% reduction. These are additional-learning comparisons: varied continuations inherited original training and then received more data/updates. This batch does **not** contain a matched extra-training-on-original-only continuation control, so the improvement cannot be attributed solely to environmental diversity.

Fall rates improve more clearly than average target progress. Hidden BC falls 4 times/90 min(0.044/min), versus frozen BC 11/30 min(0.367/min). Hidden DAgger falls 5/90 min(0.056/min), versus frozen DAgger 6/30 min(0.200/min). Hidden GAIL-BC falls 4/90 min and hidden GAIL-DM 6/90 min. Counts are small and exposure/replica structure differ; these are descriptive reductions, not formal causal risk estimates.

### 5.2 Mechanical behaviour, beyond falls

| Family | Forward m/s | Direction error | Roll RMS° | Pitch RMS° | Jerk rate ×10⁶ | Integrated jerk ×10⁹ |
|---|---|---|---|---|---|---|
| Frozen BC | 0.5217 | 0.7217 | 3.203 | 3.655 | 38.277 | 2.294 |
| Frozen DAgger-M | 0.5126 | 0.7242 | 2.841 | 3.532 | 40.943 | 2.454 |
| Frozen GAIL-BC | 0.5198 | 0.7223 | 2.777 | 3.627 | 38.095 | 2.283 |
| Frozen GAIL-DM | 0.5159 | 0.7285 | 2.768 | 3.501 | 40.974 | 2.456 |
| Varied hidden BC | 0.5181 | 0.7206 | 2.579 | 3.462 | 40.712 | 2.440 |
| Varied hidden DAgger-M | 0.5200 | 0.7216 | 2.606 | 3.498 | 41.236 | 2.472 |
| Varied hidden GAIL-BC | 0.5187 | 0.7198 | 2.663 | 3.503 | 40.703 | 2.440 |
| Varied hidden GAIL-DM | 0.5178 | 0.7216 | 2.659 | 3.457 | 41.194 | 2.469 |
| Varied input BC | 0.5200 | 0.7187 | 2.573 | 3.456 | 40.755 | 2.443 |
| Varied input DAgger-M | 0.5191 | 0.7207 | 2.679 | 3.484 | 41.312 | 2.476 |
| Varied input GAIL-BC | 0.5186 | 0.7191 | 2.572 | 3.454 | 40.710 | 2.440 |
| Varied input GAIL-DM | 0.5164 | 0.7210 | 2.586 | 3.450 | 41.305 | 2.476 |

Varied models cluster near 0.516–0.520 m/s forward velocity and roughly 2.6° roll RMS, close to the oracle’s 0.521 m/s and 2.607°. BC-derived varied models approach the oracle’s mechanical pattern rather than preserving the lower-jerk but less stable frozen BC pattern. DAgger-derived models remain around 41×10⁶ jerk-rate units. A lower jerk value alone would have incorrectly favoured frozen BC despite its higher fall frequency and action error.

### 5.3 What does explicit context add?

Compare each input policy with its hidden counterpart **at the same training seed and evaluation round**. The table averages the 90 paired differences over the 30 sampled rounds and three replicas. Fall differences are total input-minus-hidden counts over equal 90 minute exposure.

| Method | Mean Δprogress m | Relative Δprogress % | Δfalls /90min | Δprogress by seed42/43/44 m |
|---|---|---|---|---|
| BC | 0.0989 | 0.302 | 2 | 0.0664/0.0246/0.2056 |
| DAgger-M | 0.0086 | 0.026 | 3 | 0.0442/-0.0734/0.0551 |
| GAIL-BC | 0.0547 | 0.167 | 1 | 0.0483/0.0181/0.0977 |
| GAIL-DM | -0.0290 | -0.088 | -2 | 0.0605/-0.1331/-0.0144 |

The mean changes are small: about+0.099 m for BC,+0.009 m for DAgger,+0.055 m for GAIL-BC and−0.029 m for GAIL-DM over a 60 second deployment. There is no consistent reduction in falls or action error from supplying slip. Input DAgger has 8 falls versus hidden DAgger 5; input GAIL-DM has 4 versus 6, illustrating that even the direction of the stability difference depends on the method.

Only correct context was evaluated in this batch. There are no newly run wrong-context/constant-context interventions here, so small input-policy gains cannot establish beneficial causal use of slip. The oracle itself is not explicitly conditioned on slip; it reacts through its state inputs. That limits the direct supervised signal to exploit the extra scalar, but it does not prove the scalar is intrinsically useless—particularly for future-dynamics prediction or another learning objective.

### 5.4 Corrected-nominal retention

These are scalar 0.1 tests, three task seeds per replica. They test performance near a training condition, not return to the byte-identical original scene.

| Family | Progress m | Normalized score | Falls /deployments | Sampled/nominal progress % |
|---|---|---|---|---|
| Frozen BC | 38.651 | 0.9861 | 0/3 | 84.93 |
| Frozen DAgger-M | 39.353 | 1.0041 | 0/3 | 82.96 |
| Frozen GAIL-BC | 38.810 | 0.9899 | 0/3 | 84.45 |
| Frozen GAIL-DM | 39.279 | 1.0021 | 0/3 | 82.74 |
| Varied hidden BC | 39.048 | 0.9963 | 1/9 | 83.96 |
| Varied hidden DAgger-M | 39.196 | 1.0000 | 0/9 | 83.80 |
| Varied hidden GAIL-BC | 38.978 | 0.9946 | 0/9 | 84.25 |
| Varied hidden GAIL-DM | 39.327 | 1.0034 | 0/9 | 83.53 |
| Varied input BC | 39.145 | 0.9988 | 0/9 | 84.00 |
| Varied input DAgger-M | 39.404 | 1.0053 | 0/9 | 83.37 |
| Varied input GAIL-BC | 39.203 | 1.0002 | 0/9 | 83.90 |
| Varied input GAIL-DM | 39.249 | 1.0012 | 0/9 | 83.63 |

Corrected-nominal progress remains near the oracle’s 39.194 m. This is evidence against a large loss at scalar 0.1, not proof that the varied models retain their original-scene performance: the new varied policies were not evaluated in that original scene. The low nominal sample count also limits conclusions about rare falls. The final column is the ratio of mean progress over the 30 sampled rounds to mean progress in the three scalar-nominal rounds (balanced over replicas). It quantifies descriptive environmental retention, but uses different task seeds and is not a paired causal estimate of slip degradation.

## 6. Dedicated analysis — DAgger in varied-slip environments

### 6.1 Why DAgger is a stronger fidelity baseline here

Unlike BC, DAgger repeatedly asks the oracle what to do in states visited by its evolving controller. It also sees both target modes on every training surface in every aggregation round. This is a plausible reason for its substantially lower error on both expert and learner states, but the current design does not isolate that mechanism from its much larger supervised optimization exposure.

On sampled surfaces, hidden varied DAgger’s held-out MSE 0.001866 is 90.6% lower than hidden varied BC 0.019949; on-policy MSE 0.002119 is 90.0% lower than BC 0.021086. Joint occupancy SWD also improves, 0.043459 versus 0.050188. Yet progress differs by only 0.061 m per minute, and DAgger has 5 falls against BC 4 over 90 minutes. **DAgger is clearly closer to the oracle’s actions, but not clearly the universally safest or fastest controller.**

### 6.2 Where environmental gains appear

| Parent μ | Frozen DAgger progress m | Hidden varied m | Input varied m | Frozen falls /6min | Hidden falls /18min | Input falls /18min |
|---|---|---|---|---|---|---|
| 0.025 | 42.531 | 42.573 | 42.525 | 1 | 2 | 1 |
| 0.1 | 39.361 | 39.329 | 38.946 | 0 | 0 | 1 |
| 0.2 | 36.404 | 36.753 | 37.150 | 2 | 1 | 3 |
| 1.0 | 27.690 | 28.061 | 27.945 | 2 | 0 | 2 |
| 10.0 | 17.245 | 17.506 | 17.700 | 1 | 2 | 1 |

The improvement is not uniform. Around parent 1, frozen DAgger records 2 falls in 6 minutes; hidden varied DAgger records none in 18 minutes, while mean progress rises 27.690→28.061 m. At the highest parent group, progress rises 17.245→17.506 m for hidden and 17.700 m for input. At parent 0.1, hidden continuation is essentially unchanged in progress; the input version is lower. It would therefore be misleading to report only one favourable high-slip example or claim monotonic benefits from conditioning.

### 6.3 Repeatability across continuation seeds

| DAgger variant | Training seed | Mean normalized score | Progress m | Held-out MSE | Falls /30min |
|---|---|---|---|---|---|
| hidden | 42 | 0.9987 | 32.839 | 0.001879 | 2 |
| hidden | 43 | 0.9963 | 32.795 | 0.001813 | 2 |
| hidden | 44 | 1.0018 | 32.900 | 0.001905 | 1 |
| input | 42 | 1.0016 | 32.883 | 0.001906 | 3 |
| input | 43 | 0.9964 | 32.721 | 0.001910 | 3 |
| input | 44 | 1.0041 | 32.955 | 0.001890 | 2 |

All six varied DAgger replicas stay close to oracle-normalized progress. The hidden replicas have 1–2 falls per 30 minutes; input replicas 2–3. This makes hidden varied DAgger a particularly strong **fidelity-first reference**, without evidence that the context input is needed for the observed robustness. These are three continuations of a common ancestor; a stronger repeatability claim would require independent original training seeds as well.

### 6.4 Does adding GAIL improve the DAgger starting point?

For hidden varied policies, GAIL-DM changes progress 32.844→32.851 m, held-out MSE 0.001866→0.002167 and falls 5→6 over 90 minutes. The corresponding input GAIL-DM progress is slightly lower than input DAgger, while its fall count is lower. These trade-offs do not support a reliable improvement from the extra 70,000 PPO steps. The sensible conclusion is that GAIL mostly preserves a strong initializer under the tested regularization and budget, not that adversarial training is generally ineffective.

## 7. Algorithmic efficiency and resource accounting

The standardized 170,000 figure refers to unique expert-labelled examples, not optimizer steps. GAIL’s 70,000 figure refers to active on-policy environment transitions, not PPO gradient updates. Initializer and shared-data costs must be kept visible.

| Per training phase | BC | DAgger-M | Each GAIL variant |
|---|---|---|---|
| Expert training examples | 170,000 | 100,000seed +70,000learner labels | 170,000static |
| New learner transitions | 0 | 70,000 | 70,000 |
| Live oracle learner-state queries | 0 | 70,000 | 0 |
| Supervised sample exposures | 17,000,000 | 54,000,000 | Initializer cost +BC-anchor use |
| Supervised gradient updates | 66,500 | 211,050 | Initializer cost +PPO/discriminator updates |
| Inherited model cost | None original; originalBC if varied | None original; originalDAgger if varied | Corresponding trained supervised initializer |

DAgger’s eight supervised fits expose 100 k+110 k+…+170 k examples for 50 epochs each, giving 54 million sample exposures, versus BC’s 17 million. DAgger therefore receives about 3.18 times the supervised optimization exposure, even though both end with170,000 unique labels. The result supports the implemented DAgger pipeline, but it is not a clean test of equal-compute BC versus DAgger.

Across the batch, the recorded unique collection totals are 1,850,000 active steps and 870,000 oracle queries, including shared validation/test collection. The active-step total decomposes into 380,000 shared original+varied expert examples, 490,000 learner steps for seven DAgger replicas, and 980,000 for fourteen GAIL replicas. Of the 870,000 queries, 380,000 belong to the shared expert collections and 490,000 to DAgger learner-state labels. Startup/recovery counters are logged separately; these are not wall-clock or energy measurements.

All 98 GAIL update rounds completed without an actor rollback flag. This does not establish unconstrained learning: the actor learning rate, KL checks and supervised anchor still strongly limit drift. Final-policy evaluation alone does not supply normalized-return convergence curves over intermediate query counts or PPO steps. Claims about sample complexity or convergence speed should therefore remain narrower than claims about final quality.

## 8. Stage 4 — What the probing experiment actually tests

### 8.1 Shared states, held-out trajectories and controls

All 28 actors and the expert are frozen. The probes use the 30 oracle trajectories from the new grouped/confirmation evaluation, not the learner traces. Grouped draws 0/1 supply 10 training runs; grouped draw 2 supplies 5 validation runs; confirmation supplies 15 test runs. After stride 7 sampling and exclusion of reset/cross-recovery windows, the counts are 4,270 training, 2,132 validation and 6,403 test states. The future horizon is 10 steps, or 0.2 seconds.

The 30 targets span generator phase, force/contact proxies, torso/gyro state, ankle geometry, commands, phase-posture interactions and six future outcomes. Current properties decoded directly from observations are retention checks; future outcomes are under **oracle actions**, not the candidate’s own actions. Slip is not a target. A thresholded force indicator is not a separately measured contact event, and a phase-posture product is not a validated physical balance margin.

There are 75 representations: 56 student hidden layers, 12 random student layers across both 63/64 input architectures and three seeds, one expert layer, three random-expert layers, raw 63 input, raw 64 input including context, and restricted phase/command input. Four label conditions—true plus three trajectory-shuffled controls—produce 300 multitarget fits and 9,000 target scores. The expert’s saved weights reproduce recorded oracle actions before the audit accepts the data.

All probes are standardized linear ridge regressions. Scaling uses training data; each target selects its penalty using validation macro-run MSE, not the test set. Binary contact proxies also report thresholded accuracy/balanced accuracy, without claiming calibrated probabilities. No original policy is retrained and no layer-reset intervention is performed in this new probing batch.

### 8.2 Representative properties in the first hidden layer

Values below are held-out R². For varied models/random controls, each cell averages the corresponding three network seeds; this is a descriptive average of separate fits. Raw-input controls have no hidden layer. A value 1 means perfect prediction on these targets;0 matches the test-target mean-variance baseline;negative values are allowed.

| Representation | Phase sin | Left contact | Roll | Phase×roll | Future velocity | Future pitch |
|---|---|---|---|---|---|---|
| Raw63 | 0.9904 | 0.8601 | 1.0000 | 0.3624 | 0.5951 | 0.5634 |
| Raw64+slip | 0.9905 | 0.8628 | 1.0000 | 0.3698 | 0.6318 | 0.5645 |
| Phase/command only | 0.6870 | 0.2446 | 0.0023 | 0.0212 | 0.5005 | 0.4490 |
| Expert h1 | 0.8854 | 0.7530 | 0.6490 | 0.2809 | 0.5113 | 0.5151 |
| Random expert h1 | 0.9100 | 0.7783 | 0.6389 | 0.2453 | 0.5112 | 0.4995 |
| Random63 h1 | 0.9646 | 0.8315 | 0.8072 | 0.4559 | 0.6114 | 0.5678 |
| Random64 h1 | 0.9668 | 0.8328 | 0.8148 | 0.4681 | 0.6298 | 0.5777 |
| Frozen BC h1 | 0.9804 | 0.8617 | 0.9625 | 0.6847 | 0.6288 | 0.6012 |
| Frozen DAgger-M h1 | 0.9844 | 0.8679 | 0.9775 | 0.7121 | 0.6317 | 0.5930 |
| Frozen GAIL-BC h1 | 0.9804 | 0.8617 | 0.9625 | 0.6847 | 0.6288 | 0.6011 |
| Frozen GAIL-DM h1 | 0.9844 | 0.8679 | 0.9775 | 0.7121 | 0.6317 | 0.5930 |
| Varied hidden BC h1 | 0.9840 | 0.8689 | 0.9730 | 0.6911 | 0.6330 | 0.5988 |
| Varied hidden DAgger-M h1 | 0.9866 | 0.8720 | 0.9822 | 0.7065 | 0.6294 | 0.5876 |
| Varied hidden GAIL-BC h1 | 0.9840 | 0.8689 | 0.9730 | 0.6910 | 0.6330 | 0.5988 |
| Varied hidden GAIL-DM h1 | 0.9866 | 0.8720 | 0.9822 | 0.7063 | 0.6294 | 0.5876 |
| Varied input BC h1 | 0.9840 | 0.8686 | 0.9731 | 0.7000 | 0.6409 | 0.5988 |
| Varied input DAgger-M h1 | 0.9868 | 0.8728 | 0.9814 | 0.7073 | 0.6354 | 0.5862 |
| Varied input GAIL-BC h1 | 0.9840 | 0.8686 | 0.9731 | 0.7000 | 0.6408 | 0.5989 |
| Varied input GAIL-DM h1 | 0.9868 | 0.8728 | 0.9814 | 0.7073 | 0.6354 | 0.5862 |

The clearest learned-feature result is phase-conditioned posture. Raw 63 input yields R² 0.3624 for phase×roll; random 63 h1 reaches 0.4559; original BC h1 reaches 0.6847 and original DAgger h1 reaches 0.7121. This is a nonlinear interaction that a linear readout cannot recover simply by selecting one raw coordinate. The learned networks make it substantially more linearly accessible.

Conversely, raw observations predict current roll essentially perfectly because roll is directly supplied. Learned h1 roll R²around 0.96–0.98 is not “better information” than raw input; it indicates that much of the observable posture survives the transformation. Phase and contact scores also have high raw/random baselines, so their absolute R²alone is not evidence of a learned gait concept.

### 8.3 Does explicit slip improve accessible future information?

For future forward velocity, raw 63 R²is 0.5951 and raw 64 with slip is 0.6318. Thus slip is informative about oracle motion over the held-out surfaces even without a learned representation. Random 64 h1 also performs well, 0.6298, versus random 63 h1 at 0.6114.

Varied BC h1 improves 0.6330(hidden)→0.6409(input); varied DAgger h1 improves 0.6294→0.6354. These increments are small and should be assessed against the strong raw 64/random 64 controls, not just against hidden-policy features. They show that context-related predictive information can be present without producing a large behavioural improvement. They do not prove that the action head exploits it beneficially.

### 8.4 Layer 1 versus layer 2

| Family | h1phase×roll R² | h2phase×roll R² | h1future velocity R² | h2future velocity R² | h1future pitch R² | h2future pitch R² |
|---|---|---|---|---|---|---|
| Frozen BC | 0.6847 | 0.6475 | 0.6288 | 0.6179 | 0.6012 | 0.6015 |
| Frozen DAgger-M | 0.7121 | 0.6300 | 0.6317 | 0.6204 | 0.5930 | 0.6029 |
| Frozen GAIL-BC | 0.6847 | 0.6476 | 0.6288 | 0.6179 | 0.6011 | 0.6015 |
| Frozen GAIL-DM | 0.7121 | 0.6305 | 0.6317 | 0.6204 | 0.5930 | 0.6030 |
| Varied hidden BC | 0.6911 | 0.6117 | 0.6330 | 0.6199 | 0.5988 | 0.6049 |
| Varied hidden DAgger-M | 0.7065 | 0.6155 | 0.6294 | 0.6216 | 0.5876 | 0.6001 |
| Varied hidden GAIL-BC | 0.6910 | 0.6118 | 0.6330 | 0.6199 | 0.5988 | 0.6049 |
| Varied hidden GAIL-DM | 0.7063 | 0.6159 | 0.6294 | 0.6217 | 0.5876 | 0.6001 |
| Varied input BC | 0.7000 | 0.6126 | 0.6409 | 0.6218 | 0.5988 | 0.6056 |
| Varied input DAgger-M | 0.7073 | 0.6170 | 0.6354 | 0.6232 | 0.5862 | 0.6006 |
| Varied input GAIL-BC | 0.7000 | 0.6125 | 0.6408 | 0.6218 | 0.5989 | 0.6055 |
| Varied input GAIL-DM | 0.7073 | 0.6166 | 0.6354 | 0.6232 | 0.5862 | 0.6006 |

Layer 2 is not uniformly more decodable. Phase×roll and future velocity generally weaken in the second layer, while future pitch often remains similar or improves slightly. This is consistent with reorganization toward the action output, but the probe does not establish why the reorganization occurs. A lower readout score can reflect inaccessible nonlinear coding, not absence of the underlying information.

DAgger’s markedly better action fidelity is also not accompanied by uniformly higher probe scores on every target. For example, hidden varied DAgger h1 future pitch R² 0.5876 is below BC’s 0.5988, despite much lower action MSE. The probe targets are useful behavioural variables, not a complete list of the computations required for accurate control.

### 8.5 Expert comparison and representation geometry

All 1,680 student-layer/property comparisons have positive R²gaps relative to the expert hidden layer. It would be incorrect to interpret this as the students knowing more than the expert or transferring more than 100% of its knowledge. The teacher has one 64-unit tanh hidden layer; students have two 256-unit ReLU layers. Width, nonlinearity and task-specific compression strongly affect linear readability.

| Family | h1expert CKA | h2expert CKA | h1effective variance dimension | h2effective variance dimension |
|---|---|---|---|---|
| Frozen BC | 0.9393 | 0.9492 | 5.69 | 6.64 |
| Frozen DAgger-M | 0.9407 | 0.9148 | 6.34 | 6.28 |
| Frozen GAIL-BC | 0.9394 | 0.9492 | 5.69 | 6.64 |
| Frozen GAIL-DM | 0.9407 | 0.9148 | 6.34 | 6.28 |
| Varied hidden BC | 0.9409 | 0.9403 | 5.89 | 6.48 |
| Varied hidden DAgger-M | 0.9212 | 0.8849 | 6.74 | 5.75 |
| Varied hidden GAIL-BC | 0.9409 | 0.9403 | 5.89 | 6.48 |
| Varied hidden GAIL-DM | 0.9212 | 0.8849 | 6.74 | 5.75 |
| Varied input BC | 0.9409 | 0.9400 | 5.89 | 6.45 |
| Varied input DAgger-M | 0.9215 | 0.8846 | 6.73 | 5.74 |
| Varied input GAIL-BC | 0.9409 | 0.9399 | 5.89 | 6.45 |
| Varied input GAIL-DM | 0.9216 | 0.8846 | 6.73 | 5.74 |

Raw 63 input already has expert CKA 0.8968, and random 63 h1 averages 0.8944. Original BC h2 has 0.9492, whereas the far more faithful original DAgger h2 has 0.9148. Varied hidden DAgger h2 falls further to 0.8849 despite its much better action imitation. Therefore a higher whole-layer CKA is **not a ranking of control quality or expert-action fidelity**.

The effective variance dimensions of student layers are roughly 5.7–6.7 on this oracle-state dataset, despite 256 available units; expert h1 is 8.38. Participation ratio is a covariance-spectrum summary of the sampled activations. It is not a count of semantic concepts, proof that the network can be reduced to six neurons, or a general dimension estimate for all possible states.

### 8.6 GAIL and its initializer: near-identical measured representations

The first-layer phase×roll scores of original BC and GAIL-BC both round to 0.6847; DAgger and GAIL-DM both round to 0.7121. This near-equality persists for many h1/h2 properties and CKA values in the varied families. These results are consistent with conservative fine-tuning preserving the supervised representation, and with the small behavioural differences seen earlier. They do not prove identical weights or that no useful change occurred in a narrow state region.

### 8.7 Controls prevent overinterpretation

| Representation | Phase true /shuffled R² | Phase×roll true /shuffled R² | Future pitch true /shuffled R² |
|---|---|---|---|
| raw_observation | 0.9904 / 0.5324 | 0.3624 / 0.0302 | 0.5634 / 0.0701 |
| expert_h1 | 0.8854 / 0.5143 | 0.2809 / 0.0329 | 0.5151 / 0.0620 |
| original_hidden_42_dagger_mixed_h1 | 0.9844 / 0.6180 | 0.7121 / 0.0400 | 0.5930 / 0.0669 |
| varied_hidden_42_dagger_mixed_h1 | 0.9865 / 0.6166 | 0.7125 / 0.0361 | 0.5888 / 0.0663 |
| varied_input_42_dagger_mixed_correct_h1 | 0.9868 / 0.6171 | 0.7063 / 0.0358 | 0.5852 / 0.0662 |

Trajectory shuffling retains substantial periodic phase information: shuffled phase R²around 0.5–0.6 is not a zero-information null. For phase×roll and future pitch, the shuffled scores are much smaller, which makes those true-label comparisons more informative. Scores should not be called formal selectivity p-values; the null procedure is a project-specific temporal control.

The six future-target persistence baselines all have negative R². For example, carrying the previous forward velocity forward gives−2.6732, and keeping current pitch gives−0.4115. Learned readouts beat these baselines, but raw inputs/random nonlinear features already predict much of this short-horizon periodic behaviour. It is not evidence of an explicit learned world model or long-horizon balance planning.

## 9. Integrating behaviour and representations

1. **Task saturation hides fidelity differences.** Original BC and DAgger both walk at oracle-level progress, while DAgger’s action error is much lower. The progression into varied physics exposes additional stability differences that nominal return alone misses.
2. **Varied continuation improves local imitation and reliability more than speed.** This is supported across fidelity and fall metrics, but diversity and extra training are confounded without a matched nominal-only continuation arm.
3. **DAgger’s main advantage is faithful control, not an across-the-board return win.** Its strong results persist across the varied continuation replicas. The strongest candidate mechanism is better coverage/optimization of learner-relevant states; the current probes are consistent with retained phase/posture information but do not isolate that mechanism.
4. **Explicit context is measurable but not clearly necessary.** Slip improves predictability of future oracle motion in raw features. The learned input policies show small decoder gains, yet behavioural benefits are inconsistent. This is not a contradiction: available information need not be used in a way that improves the task.
5. **GAIL adds cost without a dependable final-policy advantage in this batch.** Its measured features mostly track its initializer, and fidelity sometimes worsens slightly. This conclusion applies to this anchored, conservative GAIL configuration and 70 krollout budget, not all adversarial imitation.
6. **Neither CKA nor a single probe target explains the ranking.** DAgger improves fidelity while some CKA/readout scores decrease. Representational similarity is an interpretation aid, not a surrogate benchmark.

## 10. Limitations and next evidence needed

- Original policies have one training seed. Varied replicas share parents/data; they are not full independent experimental replications.
- Shared 60-second tests and consistent resets improve comparability, but rare-fall estimates remain imprecise. Falls/min excludes recovery time, and target sequences can diverge when arrival timing differs.
- Training exposure differs substantially across BC/DAgger/GAIL despite standardized unique-data budgets. Matched compute and matched extra-nominal-training controls would strengthen causal conclusions.
- Evaluation samples fixed surfaces within(0,10]. It does not test online slip changes, heterogeneous terrain, motor damage, unseen slip beyond 10, or a different body.
- New varied policies are not tested in the preserved original scene. Scalar 0.1 retention is not a substitute for that missing condition.
- Only correct-context evaluation was run. Constant/wrong-context interventions and direct action sensitivity to the context channel would better establish whether the input is used beneficially.
- Probes cover selected properties on oracle states, not every representation or learner-state failure. Future outcomes are oracle-conditioned. Shared test data, multiple targets and random-feature baselines limit broad mechanistic claims.
- This report does not establish R1 transfer usefulness. The R1 study remains separate/deferred, and there are no causal representation-ablation results for this new generation.

The most valuable follow-up for the DAgger question is a matched-budget continuation comparison: additional nominal-only training versus varied training, with aligned optimizer exposure and independent original seeds. The most valuable context follow-up is wrong/constant-context testing on the same rounds. Both would answer more specific questions than expanding the model list without additional controls.

## Appendix A. Grouped versus confirmation scores and replica variation

Each varied row contains three replicas. Score SD below is the sample standard deviation of the **three replica means over 30 rounds**, not a confidence interval and not a frame-level statistic. Original models have one seed, so no training-replica SD is supplied.

| Family | Grouped score | Confirmation score | Replica-mean SD | Minimum replica mean | Maximum replica mean |
|---|---|---|---|---|---|
| Frozen BC | 1.0047 | 0.9920 | — | 0.9984 | 0.9984 |
| Frozen DAgger-M | 1.0030 | 0.9811 | — | 0.9920 | 0.9920 |
| Frozen GAIL-BC | 0.9922 | 0.9971 | — | 0.9946 | 0.9946 |
| Frozen GAIL-DM | 0.9707 | 0.9921 | — | 0.9814 | 0.9814 |
| Varied hidden BC | 1.0066 | 0.9902 | 0.00496 | 0.9927 | 1.0017 |
| Varied hidden DAgger-M | 1.0066 | 0.9912 | 0.00274 | 0.9963 | 1.0018 |
| Varied hidden GAIL-BC | 1.0094 | 0.9924 | 0.00192 | 0.9997 | 1.0031 |
| Varied hidden GAIL-DM | 1.0051 | 0.9948 | 0.00557 | 0.9936 | 1.0038 |
| Varied input BC | 1.0101 | 0.9934 | 0.00275 | 0.9989 | 1.0043 |
| Varied input DAgger-M | 1.0079 | 0.9935 | 0.00392 | 0.9964 | 1.0041 |
| Varied input GAIL-BC | 1.0101 | 0.9936 | 0.00447 | 0.9982 | 1.0068 |
| Varied input GAIL-DM | 1.0094 | 0.9856 | 0.00852 | 0.9876 | 1.0026 |

Grouped scores tend to be slightly above 1 and confirmation scores slightly below 1. This schedule dependence is another reason not to declare a winner from a pooled difference of a few thousandths. The detailed per-replica source report remains the authoritative deployment-level record.

## Appendix B. Complete 30-property first-layer probe profile

To keep this table readable, it lists the expert, raw input, matched random controls and the two supervised varied families with and without context. GAIL and original-model scores, both layers and all shuffled controls are preserved in the linked 9,000-row source CSV. All numbers here are held-out R²; varied/random values average three replicas.

| Property | Expert | Raw63 | Random63h1 | Random64h1 | Hidden BC | Input BC | Hidden DAgger | Input DAgger |
|---|---|---|---|---|---|---|---|---|
| generator_phase_sin | 0.8854 | 0.9904 | 0.9646 | 0.9668 | 0.9840 | 0.9840 | 0.9866 | 0.9868 |
| generator_phase_cos | 0.9380 | 0.9872 | 0.9857 | 0.9857 | 0.9896 | 0.9897 | 0.9912 | 0.9911 |
| left_force_contact_gt10N | 0.7530 | 0.8601 | 0.8315 | 0.8328 | 0.8689 | 0.8686 | 0.8720 | 0.8728 |
| right_force_contact_gt10N | 0.7114 | 0.8353 | 0.8109 | 0.8101 | 0.8402 | 0.8400 | 0.8400 | 0.8393 |
| left_force_norm_N | 0.7253 | 0.9815 | 0.9231 | 0.9208 | 0.9735 | 0.9735 | 0.9840 | 0.9841 |
| right_force_norm_N | 0.6728 | 0.9889 | 0.9114 | 0.9189 | 0.9739 | 0.9731 | 0.9819 | 0.9820 |
| head_height_m | 0.3043 | 1.0000 | 0.5561 | 0.5886 | 0.6925 | 0.6959 | 0.7578 | 0.7629 |
| head_vertical_velocity_mps | 0.2802 | 1.0000 | 0.4917 | 0.5027 | 0.7962 | 0.7956 | 0.9386 | 0.9335 |
| torso_roll_deg | 0.6490 | 1.0000 | 0.8072 | 0.8148 | 0.9730 | 0.9731 | 0.9822 | 0.9814 |
| torso_pitch_deg | 0.6222 | 1.0000 | 0.8433 | 0.8398 | 0.9717 | 0.9714 | 0.9846 | 0.9844 |
| gyro_x_degps | 0.5888 | 1.0000 | 0.9384 | 0.9426 | 0.9859 | 0.9856 | 0.9917 | 0.9921 |
| left_ankle_hip_x_m | 0.8969 | 1.0000 | 0.9710 | 0.9740 | 0.9933 | 0.9932 | 0.9978 | 0.9977 |
| right_ankle_hip_x_m | 0.8794 | 1.0000 | 0.9686 | 0.9724 | 0.9909 | 0.9909 | 0.9962 | 0.9962 |
| gyro_y_degps | 0.5819 | 1.0000 | 0.9403 | 0.9400 | 0.9825 | 0.9828 | 0.9919 | 0.9914 |
| left_ankle_hip_y_m | 0.5831 | 1.0000 | 0.9010 | 0.9015 | 0.9507 | 0.9500 | 0.9718 | 0.9708 |
| right_ankle_hip_y_m | 0.4764 | 1.0000 | 0.8507 | 0.8532 | 0.9394 | 0.9389 | 0.9608 | 0.9602 |
| gyro_z_degps | 0.7125 | 1.0000 | 0.9807 | 0.9836 | 0.9909 | 0.9906 | 0.9959 | 0.9959 |
| left_ankle_hip_z_m | 0.8569 | 1.0000 | 0.9486 | 0.9515 | 0.9666 | 0.9669 | 0.9740 | 0.9740 |
| right_ankle_hip_z_m | 0.8853 | 1.0000 | 0.9636 | 0.9678 | 0.9806 | 0.9803 | 0.9840 | 0.9841 |
| command_forward_displacement_m | 0.8952 | 1.0000 | 0.9705 | 0.9699 | 0.9943 | 0.9945 | 0.9982 | 0.9982 |
| command_lateral_displacement_m | 0.8410 | 1.0000 | 0.9422 | 0.9463 | 0.9945 | 0.9944 | 0.9979 | 0.9979 |
| command_orientation_deg | 0.9082 | 1.0000 | 0.9458 | 0.9463 | 0.9979 | 0.9979 | 0.9995 | 0.9995 |
| phase_sin_times_roll_deg | 0.2809 | 0.3624 | 0.4559 | 0.4681 | 0.6911 | 0.7000 | 0.7065 | 0.7073 |
| phase_cos_times_pitch_deg | 0.6043 | 0.7908 | 0.8247 | 0.8180 | 0.8740 | 0.8742 | 0.8836 | 0.8829 |
| future_200ms_forward_velocity_mps | 0.5113 | 0.5951 | 0.6114 | 0.6298 | 0.6330 | 0.6409 | 0.6294 | 0.6354 |
| future_200ms_direction_error | 0.3062 | 0.4936 | 0.4944 | 0.5225 | 0.5378 | 0.5447 | 0.5323 | 0.5367 |
| future_200ms_roll_deg | 0.2190 | 0.3400 | 0.3128 | 0.3317 | 0.3885 | 0.3906 | 0.3895 | 0.3895 |
| future_200ms_pitch_deg | 0.5151 | 0.5634 | 0.5678 | 0.5777 | 0.5988 | 0.5988 | 0.5876 | 0.5862 |
| future_200ms_left_force_contact_gt10N | 0.7138 | 0.7357 | 0.7375 | 0.7436 | 0.7552 | 0.7551 | 0.7570 | 0.7574 |
| future_200ms_right_force_contact_gt10N | 0.6845 | 0.7045 | 0.7172 | 0.7157 | 0.7320 | 0.7317 | 0.7290 | 0.7277 |

## Appendix C. Probe errors in physical units and contact classification

R² is useful for comparing targets of different scales, but RMSE makes prediction
error tangible. The table below uses the first hidden layer and the same held-out
test trajectories. Varied-policy values average the three separately fitted
replicas. Contact balanced accuracy averages sensitivity to positive and negative
force-threshold classes; it is not a calibrated contact probability.

| Representation | Current roll RMSE° | Future velocity RMSE m/s | Future pitch RMSE° | Left contact balanced accuracy | Right contact balanced accuracy |
|---|---|---|---|---|---|
| Raw63 | 0.0000 | 0.2319 | 1.5242 | 0.9669 | 0.9555 |
| Raw64+slip | 0.0000 | 0.2212 | 1.5223 | 0.9680 | 0.9566 |
| Expert h1 | 1.5620 | 0.2548 | 1.6063 | 0.9320 | 0.9132 |
| Frozen BC h1 | 0.5104 | 0.2221 | 1.4568 | 0.9673 | 0.9556 |
| Frozen DAgger-M h1 | 0.3957 | 0.2212 | 1.4715 | 0.9701 | 0.9570 |
| Frozen GAIL-BC h1 | 0.5104 | 0.2221 | 1.4568 | 0.9673 | 0.9556 |
| Frozen GAIL-DM h1 | 0.3957 | 0.2212 | 1.4716 | 0.9701 | 0.9570 |
| Varied hidden BC h1 | 0.4329 | 0.2208 | 1.4611 | 0.9708 | 0.9551 |
| Varied hidden DAgger-M h1 | 0.3520 | 0.2219 | 1.4814 | 0.9720 | 0.9567 |
| Varied hidden GAIL-BC h1 | 0.4329 | 0.2208 | 1.4611 | 0.9708 | 0.9550 |
| Varied hidden GAIL-DM h1 | 0.3520 | 0.2219 | 1.4813 | 0.9720 | 0.9570 |
| Varied input BC h1 | 0.4326 | 0.2184 | 1.4610 | 0.9703 | 0.9567 |
| Varied input DAgger-M h1 | 0.3596 | 0.2201 | 1.4838 | 0.9719 | 0.9556 |
| Varied input GAIL-BC h1 | 0.4326 | 0.2184 | 1.4610 | 0.9702 | 0.9566 |
| Varied input GAIL-DM h1 | 0.3597 | 0.2201 | 1.4838 | 0.9719 | 0.9556 |

## Appendix D. Source reports, provenance and reproducibility

- [Full evaluation JSON](../experiments/consolidated/v1/evaluation/all_all_correct/comparison.json): 1,020 run records, per-replica aggregates and paired context differences.
- [Evaluation CSV](../experiments/consolidated/v1/evaluation/all_all_correct/comparison.csv) and [readable source table](../experiments/consolidated/v1/evaluation/all_all_correct/comparison.md).
- [Training summary](../experiments/consolidated/v1/training_summary.json) and [training plan](../experiments/consolidated/v1/plan.json): budgets, nested initializer lineage and per-update diagnostics.
- [Probe JSON](../experiments/representation_probing/consolidated_v1/report.json), [all probe scores](../experiments/representation_probing/consolidated_v1/probe_scores.csv), [expert comparisons](../experiments/representation_probing/consolidated_v1/expert_comparison.csv) and [probe plan](../experiments/representation_probing/consolidated_v1/plan.json).
- [Training/evaluation protocol](../consolidated/README.md), [new probe protocol](../consolidated_probing/README.md) and [original target definitions](../representation_probing/README.md).
- [Historical meeting summary](MEETING_SUMMARY_2026-10-10.md) and [historical Phase 1 report](PHASE1_RESULTS_REPORT.md) remain unchanged. They concern earlier batches and should be cited as such.

Numerical tables in this report are generated from the saved JSON using [the read-only report builder](analysis/consolidated_summary.py). The builder prints Markdown; it does not retrain policies, recompute simulator traces or edit source artifacts. Source hashes below identify the exact input reports used. Probe methodology references remain documented in the linked protocol: Alain & Bengio (linear probes), Hewitt & Liang (control tasks), Kornblith et al. (CKA). The present implementation adapts linear decoding to continuous locomotion regression; it is not a reproduction of their complete experiments.

| Source | SHA-256 |
|---|---|
| experiments/consolidated/v1/evaluation/all_all_correct/comparison.json | e6077bbff3716a8befce17be9fa7106916009c63b0a507154ed9e2cf07090b8e |
| experiments/representation_probing/consolidated_v1/report.json | 69b9a07641f6bfc70b4eac5b28be2ddfb9a255cf60e35c48a4c55a8d903ff116 |
| experiments/consolidated/v1/training_summary.json | 0c6cc5932e624912cebcf6290b925229e4210874bc1e524a82197add5a9f30d6 |

## Appendix E. Paper sources and final-write-up citation guide

This appendix distinguishes the published foundations from this project's experimental choices. The numerical findings in this report come from the saved artifacts in Appendix D, not from the papers below. Reference keys are stable labels for use while drafting the dissertation; each entry links to a primary paper or publisher record.

### E.1 Which source belongs with which part of the report?

| Report topic | Suggested references | What is supported, and what is project-specific |
|---|---|---|
| Sections 1 and 3: supervised behavioural cloning | R1; R2 for the sequential prediction problem | Supervised neural control and the limitation of learning only from expert-visited states. The 63-to-16 architecture, long/short collection mixture and 170,000-example budget are our choices. |
| Sections 1, 6 and 7: mixed DAgger | R2 | Iterative learner-state collection, expert labelling and dataset aggregation. Our seed dataset, mixing schedule, seven collection rounds and optimizer exposure are implementation choices, not a replication of the paper's experiments. |
| Sections 1, 6.4 and 7: adversarial imitation | R3, R4, R5; R6 for regularization background; R7 for implementation sensitivity | GAIL provides the adversarial occupancy-matching framework; PPO and GAE provide the policy-update machinery. The BC anchor, pretrained initializers, KL safeguards, reset handling and 70,000-step budget define this project's particular variant. |
| Section 2: normalized return | R7 | Random-reference zero and expert-reference one. We apply this convention to paired target-progress returns within each evaluation round; this is not the paper's entire evaluation or learning-curve protocol. |
| Section 2: action MSE | R2 as supervised imitation context | Squared prediction error is a standard regression measure. Our averaging over 16 latent gait outputs, held-out trajectory partition and separate on-policy oracle comparison must be defined explicitly; no citation turns these outputs into joint velocities. |
| Section 2: joint-occupancy distance | R8 | One-dimensional projections as the basis of sliced Wasserstein comparisons. Our 256 projections, expert-based standardization and 22-joint occupancy representation are project choices. This is not exact multivariate Earth Mover's Distance or a temporally aligned trajectory distance. |
| Sections 2 and 5.2: jerk and stability | R9 for minimum-jerk background | A kinematic smoothness motivation, not validation of a NAO torque, energy or hardware-wear measure. Our angle smoothing, finite differences, reset exclusions, fall detector, velocity and tracking metrics are operational definitions documented in this report. |
| Sections 4 and 5: changed dynamics and context | R10, R11 | Motivation for training across dynamics and distinguishing hidden from explicit context. Our training uses a discrete slip grid; Gaussian evaluation draws choose one fixed slip per run. We do not implement spatially Gaussian terrain, contact-by-contact sampling, or Decision Adapter. |
| Section 8: layerwise probes | R12 | Inspecting information accessible to simple readouts from frozen intermediate layers. We adapt classifier probing to regularized multi-output regression of locomotion quantities, with episode-level splits. |
| Section 8.7: probe controls | R13 | The need to distinguish representational information from what a probe can learn. Our random networks, raw observations and trajectory-shuffled labels are adaptations, not the paper's linguistic control-task construction. |
| Section 8.5: CKA | R14 | Comparing representation geometry, including layers of different widths. CKA is not percentage knowledge retained, behavioural equivalence, causal feature importance or demonstrated transfer ability. |
| Earlier context study, outside this consolidated batch | R15 | FiLM's feature-wise affine conditioning. FiLM was investigated historically but is excluded from the 28-model consolidated results reported here. |

### E.2 Full references

**R1 — Pomerleau (1989).** Dean A. Pomerleau. *ALVINN: An Autonomous Land Vehicle in a Neural Network.* Advances in Neural Information Processing Systems 1, pp. 305–313. [Author-institution publication record](https://publications.ri.cmu.edu/alvinn-an-autonomous-land-vehicle-in-a-neural-network). Historical example of supervised neural control, not the source of our walking architecture. The conference was held in 1988; the proceedings publication is dated 1989.

**R2 — Ross, Gordon and Bagnell (2011).** Stéphane Ross, Geoffrey Gordon and Drew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS, Proceedings of Machine Learning Research 15, pp. 627–635. [Paper and publisher citation](https://proceedings.mlr.press/v15/ross11a.html). Primary DAgger reference, including the sequential-distribution problem motivating learner-state supervision.

**R3 — Ho and Ermon (2016).** Jonathan Ho and Stefano Ermon. *Generative Adversarial Imitation Learning.* Advances in Neural Information Processing Systems 29. [Paper](https://arxiv.org/abs/1606.03476). Cite for GAIL's adversarial imitation framework, not as evidence that our tuned implementation must outperform BC or DAgger.

**R4 — Schulman et al. (2017).** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. [Paper](https://arxiv.org/abs/1707.06347). Source for PPO; using PPO in this project is an implementation distinction from the original GAIL policy optimizer.

**R5 — Schulman et al. (2015 preprint; 2016 ICLR).** John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and Pieter Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438. [Paper](https://arxiv.org/abs/1506.02438). Source for GAE; our handling of reset boundaries and rollout truncations must still be specified independently.

**R6 — Gulrajani et al. (2017).** Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin and Aaron Courville. *Improved Training of Wasserstein GANs.* Advances in Neural Information Processing Systems 30. [Paper](https://arxiv.org/abs/1704.00028). Background for gradient-penalty regularization. Our regularized logistic discriminator is not a Wasserstein critic, and this citation does not make the implemented algorithm Wasserstein GAIL.

**R7 — Orsini et al. (2021).** Manu Orsini, Anton Raichuk, Léonard Hussenot, Damien Vincent, Robert Dadashi, Sertan Girgin, Matthieu Geist, Olivier Bachem, Olivier Pietquin and Marcin Andrychowicz. *What Matters for Adversarial Imitation Learning?* Advances in Neural Information Processing Systems 34. [Paper](https://arxiv.org/abs/2106.00672). Section 2 explicitly normalizes scores to random-policy zero and expert one. The study also motivates caution about reward, normalization and discriminator-design choices; it does not validate this project's chosen hyperparameters.

**R8 — Bonneel et al. (2015).** Nicolas Bonneel, Julien Rabin, Gabriel Peyré and Hanspeter Pfister. *Sliced and Radon Wasserstein Barycenters of Measures.* Journal of Mathematical Imaging and Vision 51, pp. 22–45. DOI: 10.1007/s10851-014-0506-3. [Author-lab paper](https://vcg.seas.harvard.edu/publications/20140101-sliced-and-radon-wasserstein-barycenters-of-measures/paper). Mathematical background for sliced transport, not a robotics-specific validation of our occupancy score.

**R9 — Flash and Hogan (1985).** Tamar Flash and Neville Hogan. *The coordination of arm movements: an experimentally confirmed mathematical model.* Journal of Neuroscience 5(7), pp. 1688–1703. DOI: 10.1523/JNEUROSCI.05-07-01688.1985. [Journal paper](https://www.jneurosci.org/content/jneuro/5/7/1688.full.pdf). Studies human arm movement and endpoint minimum jerk. Our rotational joint-jerk statistic is an adaptation, not the same measured quantity or a direct measure of mechanical energy.

**R10 — Peng et al. (2018).** Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba and Pieter Abbeel. *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.* IEEE International Conference on Robotics and Automation, pp. 3803–3810. DOI: 10.1109/ICRA.2018.8460528. [Paper, first posted in 2017](https://arxiv.org/abs/1710.06537). Motivation for varying dynamics during learning. Our experiments remain within SimSpark and do not demonstrate sim-to-real or NAO-to-R1 transfer.

**R11 — Beukman et al. (2023).** Michael Beukman, Devon Jarvis, Richard Klein, Steven James and Benjamin Rosman. *Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies.* Advances in Neural Information Processing Systems 36. [Paper](https://arxiv.org/abs/2310.16686). Motivation for investigating explicit environmental context. Our appended slip input is a simpler conditioning mechanism; neither the Decision Adapter architecture nor the paper's experimental results should be attributed to this implementation.

**R12 — Alain and Bengio (2016).** Guillaume Alain and Yoshua Bengio. *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644. [Paper](https://arxiv.org/abs/1610.01644). Foundational reference for the layerwise readout approach. Our continuous targets require regression probes, and accessible information is not proof of causal use by the policy.

**R13 — Hewitt and Liang (2019).** John Hewitt and Percy Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP, pp. 2733–2743. [Paper and publisher citation](https://aclanthology.org/D19-1275/). Supports probe controls and interpretive restraint. Periodic gait labels can remain predictable after our trajectory shuffle, so that control must not be described as a guaranteed zero-information null.

**R14 — Kornblith et al. (2019).** Simon Kornblith, Mohammad Norouzi, Honglak Lee and Geoffrey Hinton. *Similarity of Neural Network Representations Revisited.* ICML, Proceedings of Machine Learning Research 97, pp. 3519–3529. [Paper and publisher citation](https://proceedings.mlr.press/v97/kornblith19a.html). Source for CKA as a representation-comparison method; the observed expert/student alignment remains an association, not an explanation of performance by itself.

**R15 — Perez et al. (2018).** Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin and Aaron Courville. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI Conference on Artificial Intelligence. [Paper, first posted in 2017](https://arxiv.org/abs/1709.07871). Historical context-conditioning reference only; the present consolidated generation contains hidden-context and input-concatenated policies, not FiLM policies.

### E.3 Attribution boundaries for the final write-up

- **Separate literature from results.** Cite the paper for an algorithm or analytical concept, then cite this report and its saved artifacts for the measured NAO outcomes. Published findings elsewhere are not independent confirmation of our numerical results.
- **Preserve operational definitions.** State that action MSE concerns latent gait actions, occupancy distance is sliced, jerk is kinematic, and falls are detector-triggered losses followed by reset. Neither generic imitation papers nor human minimum-jerk work validate those measures as direct torque or energy estimates.
- **Do not equate accessible features with transferred knowledge.** Expert/student probes, raw-input controls and CKA describe decodability and geometry. The current results do not establish causal feature use or usefulness on R1; those require interventions or target-domain experiments.
- **Name project-specific design decisions.** The slip grid and truncated-Gaussian schedules, budgets, collection mixture, seed hierarchy, server verification, fall recovery, trajectory splits and probe targets are our protocol. Papers provide foundations, not authority for every selected constant.
- **Keep oracle provenance separate.** The native FC Portugal code, checkpoint and gait/IK interface are the implementation provenance. A generic locomotion or imitation paper is not evidence that it describes this exact oracle release. Preserve the code/checkpoint identity alongside any subsequently verified team publication.
- **Do not copy incomplete proposal references uncritically.** Author-year placeholders alone (for example a proposed locomotion citation without a verified title) should not be attached to claims about falls, wear or metric validity. The full, linked sources above identify what is actually being cited.
- **Keep batches distinct.** Historical FiLM experiments are outside the consolidated generation. The scratch-initialization comparison is a separate completed extension reported in Appendix F, not part of the original 28-model parent dataset in Sections 1–10. Cite the corresponding source reports for each cohort.

This appendix is maintained in `evaluation/analysis/consolidated_references.md` and included by the report builder so that regenerating numerical tables retains the reference guide.

## Appendix F. Completed scratch-initialization comparison — 16 October 2026

This extension answers whether original-environment pretraining was necessary for
the varied-domain results, and whether GAIL could learn useful walking directly
from random actor weights under the same configured rollout budget. It adds a
fourth behavioural comparison after original, frozen-on-varied, and continued
varied training. Sections 1–10 remain the parent-batch analysis, not pooled claims
about every new model. The revised conclusions below incorporate the new evidence.

### F.1 Completion, experimental design and accounting

All **28 additional checkpoints, 929 new deployments and 224 new multitarget
probe fits** are present. Combined with reused parent evidence, this gives
**56 trained checkpoints, 1,949 deployments, 524 fits, 131 representations and
15,720 target-level probe scores**. The 1,020 parent deployments were reused
exactly, not run again. Saved artifact hashes were checked for all 929 new
evaluation receipts, 224 new and 300 parent probe receipts; all 28 new checkpoint
hashes matched. Combined probe scores matched the individual metrics files.
These are saved-artifact integrity checks, not a new live physics audit.

The new supervised families start randomly directly in the varied environment:
BC and mixed DAgger, each hidden-context and input-conditioned, seeds 42/43/44.
GAIL-BC and GAIL-DM then start from those newly trained supervised policies.
Thus “scratch GAIL-BC” removes original-environment pretraining from its lineage;
it does **not** mean its actor starts randomly at the PPO stage. Separately,
direct GAIL-random has one original-environment seed and three varied-hidden
seeds. There is no input-conditioned direct-random GAIL condition.

BC still uses 170,000 examples; mixed DAgger uses 100,000 seed examples plus
70,000 learner-state oracle labels; every GAIL actor gets 70,000 PPO rollout
steps with the same static expert-pool contract and stabilization settings.
Shared expert data are reused. Incremental varied-domain budgets match the
continuation conditions, but lifetime training does not: continuation inherits
original-environment training. This is an initialization-history comparison,
not an isolation of pretraining from additional lifetime optimization.

The new collection summary records 1,540,000 active steps and 420,000 oracle
queries, with 1,541,188 rollout-plus-recovery steps and 7,700 startup received
steps reported separately. Reused demonstrations and inherited initializer costs
must not be counted as newly collected data or described as absent supervision.

The same 30 sampled rounds (15 grouped and 15 confirmation) and three scalar-0.1
rounds are reused. Each slip is fixed throughout a 3,000-active-step deployment.
Each varied family below pools three seeds × 30 sampled rounds = 90 active
minutes. Original GAIL-random additionally has five original-scene runs, giving
38 deployments; the other 27 new models have 33 each: 38 + 27 × 33 = 929.
Probe states, trajectory splits, penalties, targets and controls are unchanged.

### F.2 Scratch versus continuation: locomotion and imitation

Each row contrasts matching algorithm/architecture families over identical
sampled rounds. “Cont.” denotes original-pretrained continuation; “scratch”
denotes the new lineage without that original pretraining. Progress and scores
are deployment means. MSE is elementwise latent-action error; lower is better.
The normalized score is averaged per round, not computed from pooled returns.

| Family | Cont. m | Scratch m | Δ m | Cont. score | Scratch score | Cont. test MSE | Scratch test MSE | Cont. on-policy MSE | Scratch on-policy MSE |
|---|---|---|---|---|---|---|---|---|---|
| BC / hidden | 32.7837 | 33.1921 | 0.4085 | 0.9984 | 1.0103 | 0.0199 | 0.1118 | 0.0211 | 0.0992 |
| BC / input | 32.8826 | 33.2738 | 0.3912 | 1.0018 | 1.0113 | 0.0199 | 0.1123 | 0.0208 | 0.0993 |
| DAgger-M / hidden | 32.8445 | 32.9173 | 0.0729 | 0.9989 | 1.0014 | 0.0019 | 0.0103 | 0.0021 | 0.0123 |
| DAgger-M / input | 32.8531 | 32.7968 | -0.0563 | 1.0007 | 0.9995 | 0.0019 | 0.0098 | 0.0022 | 0.0122 |
| GAIL-BC / hidden | 32.8378 | 33.2367 | 0.3989 | 1.0009 | 1.0122 | 0.0201 | 0.1120 | 0.0219 | 0.0994 |
| GAIL-BC / input | 32.8926 | 33.2997 | 0.4071 | 1.0018 | 1.0152 | 0.0201 | 0.1125 | 0.0212 | 0.1003 |
| GAIL-DM / hidden | 32.8514 | 32.9119 | 0.0605 | 1.0000 | 1.0015 | 0.0022 | 0.0106 | 0.0024 | 0.0127 |
| GAIL-DM / input | 32.8224 | 32.9651 | 0.1427 | 0.9975 | 1.0048 | 0.0023 | 0.0101 | 0.0025 | 0.0120 |

The clearest new finding is a **progress–fidelity trade-off**, not a universal
benefit of pretraining. Scratch BC and GAIL-BC gain about 0.39–0.41 m per minute
against their continuation counterparts, but their held-out action errors are
roughly 5.6 times larger. Scratch hidden BC reaches 33.192 m versus 32.784 m,
while its test MSE rises from 0.01995 to 0.11177. More progress therefore does
not mean closer reproduction of the oracle's actions.

Mixed DAgger retains a substantial fidelity advantage even without original
pretraining: scratch hidden DAgger test MSE is 0.01035 versus scratch BC 0.11177,
approximately a 10.8-fold reduction. Yet continuation DAgger is still much more
precise (0.00187). Its roughly 5.5-fold advantage over scratch DAgger is evidence
for the value of the complete training history at these budgets, not proof
that original-environment data alone caused it. The optimizer-exposure caveat
between BC and DAgger from Section 7 continues to apply.

Task progress for the DAgger lineages remains much closer: scratch-minus-
continuation changes range from −0.056 m for input DAgger to +0.143 m for input
GAIL-DM. These small pooled differences do not establish a robust algorithm
ranking or statistical significance across independent full training histories.

### F.3 Mechanical stability and occupancy

Each entry is “continuation / scratch”. Falls are totals over 90 active minutes
per family; divide by 90 for falls per active minute. Jerk rate is displayed in
millions of rad²/s⁶. SWD and direction error use the definitions in Section 2.

| Family | Falls | Forward m/s | Direction error | Joint SWD | Jerk rate ×10⁶ | Roll RMS° | Pitch RMS° |
|---|---|---|---|---|---|---|---|
| BC / hidden | 4 / 4 | 0.5181 / 0.5242 | 0.7206 / 0.7183 | 0.0502 / 0.0574 | 40.7119 / 39.3517 | 2.5791 / 2.5991 | 3.4620 / 3.4578 |
| BC / input | 6 / 4 | 0.5200 / 0.5256 | 0.7187 / 0.7193 | 0.0489 / 0.0617 | 40.7547 / 39.5241 | 2.5726 / 2.5506 | 3.4565 / 3.4295 |
| DAgger-M / hidden | 5 / 4 | 0.5200 / 0.5236 | 0.7216 / 0.7205 | 0.0435 / 0.0426 | 41.2362 / 41.0679 | 2.6058 / 2.5993 | 3.4985 / 3.4792 |
| DAgger-M / input | 8 / 8 | 0.5191 / 0.5197 | 0.7207 / 0.7216 | 0.0483 / 0.0493 | 41.3123 / 41.0642 | 2.6794 / 2.7375 | 3.4845 / 3.5764 |
| GAIL-BC / hidden | 4 / 2 | 0.5187 / 0.5273 | 0.7198 / 0.7179 | 0.0468 / 0.0579 | 40.7026 / 39.3691 | 2.6625 / 2.5117 | 3.5026 / 3.4396 |
| GAIL-BC / input | 5 / 2 | 0.5186 / 0.5250 | 0.7191 / 0.7175 | 0.0540 / 0.0584 | 40.7098 / 39.5994 | 2.5718 / 2.5187 | 3.4542 / 3.4215 |
| GAIL-DM / hidden | 6 / 2 | 0.5178 / 0.5220 | 0.7216 / 0.7213 | 0.0428 / 0.0456 | 41.1945 / 41.1159 | 2.6590 / 2.5736 | 3.4573 / 3.4776 |
| GAIL-DM / input | 4 / 5 | 0.5164 / 0.5227 | 0.7210 / 0.7198 | 0.0509 / 0.0494 | 41.3052 / 41.1933 | 2.5860 / 2.6438 | 3.4497 / 3.4623 |

No new supervised-initialized family shows a gross balance collapse. Scratch
GAIL-BC has two falls per architecture versus four/five for continuation, but
these are sparse counts, not a demonstrated population-level reduction. Input
DAgger has eight falls in both cohorts; scratch input GAIL-DM has five versus
four. There is no uniform scratch or context stability advantage.

BC's extra progress also accompanies a larger occupancy distance: hidden SWD
increases from 0.05019 to 0.05737 and input SWD from 0.04886 to 0.06166. DAgger's
occupancy differences are much smaller and not uniformly worse. Lower jerk
alone does not resolve this trade-off and must not be interpreted as energy
efficiency without an energy measurement.

### F.4 Environment groups, replica variation and context

The following table separates the grouped and confirmation schedules and shows
the range of the three seed-level paired mean progress differences. Seeds share
data and evaluation rounds; this range is descriptive, not a confidence interval.
The scalar-0.1 column is the nine-run scratch mean, not original-scene retention.

| Family | Grouped m | Confirmation m | Scalar 0.1 m | Min seed Δm | Max seed Δm |
|---|---|---|---|---|---|
| BC / hidden | 33.4999 | 32.8844 | 39.5576 | 0.3643 | 0.4405 |
| BC / input | 33.6100 | 32.9375 | 39.6216 | 0.1840 | 0.6768 |
| DAgger-M / hidden | 33.1447 | 32.6900 | 39.3504 | -0.1034 | 0.1968 |
| DAgger-M / input | 33.2549 | 32.3387 | 39.4925 | -0.4476 | 0.2010 |
| GAIL-BC / hidden | 33.5871 | 32.8864 | 39.2813 | 0.2277 | 0.6052 |
| GAIL-BC / input | 33.6668 | 32.9326 | 39.6632 | 0.3138 | 0.5150 |
| GAIL-DM / hidden | 33.1364 | 32.6874 | 39.1930 | -0.0284 | 0.1638 |
| GAIL-DM / input | 33.1320 | 32.7982 | 39.2616 | -0.1458 | 0.4412 |

| Scratch input minus hidden | Progress Δm | Test MSE Δ | Fall-count Δ |
|---|---|---|---|
| BC | 0.0816 | 0.0005 | 0 |
| DAgger-M | -0.1206 | -0.0005 | 4 |
| GAIL-BC | 0.0630 | 0.0006 | 0 |
| GAIL-DM | 0.0532 | -0.0005 | 3 |

Context remains a small, inconsistent intervention: input BC adds about
0.082 m and input GAIL-BC 0.063 m, whereas input DAgger loses about 0.120 m.
Input GAIL-DM gains about 0.053 m. There is no broad reversal of the earlier
conclusion that a strong hidden-context policy is competitive. Correct-context
evaluation still cannot distinguish useful conditioning from extra capacity;
wrong/constant-context interventions remain missing.

For completeness, the next table reports scratch means by Gaussian parent group,
pooling hidden/input and their three seeds (18 runs per method per parent mean).
The oracle has three rounds per group and is not duplicated as independent data.
Parent mean is a sampling label, not the actual fixed slip used in every run.

| Parent group | Actual slip range | Oracle m | Scratch BC m | Scratch DAgger m | Scratch GAIL-BC m | Scratch GAIL-DM m |
|---|---|---|---|---|---|---|
| group_0 | 0.0271–0.0379 | 41.3810 | 42.3457 | 41.6845 | 41.8983 | 41.2836 |
| group_1 | 0.0627–0.1231 | 40.5287 | 40.7818 | 40.5982 | 40.7911 | 40.5558 |
| group_2 | 0.1402–0.4090 | 36.6136 | 37.2188 | 36.8209 | 37.5899 | 36.8375 |
| group_3 | 0.2126–1.0581 | 30.2381 | 30.6213 | 30.1681 | 30.7043 | 30.2777 |
| group_4 | 5.0133–7.4176 | 16.3026 | 16.8069 | 16.7271 | 17.1510 | 16.7165 |

### F.5 Direct random-initialized GAIL: stable standing is not successful walking

These actors start randomly at the GAIL stage, unlike the supervised-initialized
families. They still retain the configured supervised anchor and conservative
PPO/discriminator settings; this is not unrestricted or purely adversarial
learning from scratch. The result tests this configuration and 70,000-step
budget, not whether GAIL can ever learn locomotion without pretraining.

| Condition | Runs | Progress m | Score | Targets total | Falls total | Test MSE | Joint SWD | Jerk rate ×10⁶ |
|---|---|---|---|---|---|---|---|---|
| original trained / original scene | 5 | -0.1213 | 0.0222 | 0 | 0 | 7.8971 | 0.9075 | 7.6320 |
| original trained / sampled | 30 | -0.2155 | -0.0111 | 0 | 12 | 7.5271 | 0.6776 | 7.6671 |
| varied trained / sampled | 90 | -0.3672 | -0.0157 | 0 | 30 | 7.5900 | 0.6937 | 7.6470 |

Both direct-random conditions fail to acquire useful target locomotion in
this batch. Varied GAIL-random averages −0.367 m and a normalized score of
−0.0157, compared with scores near one for the supervised initializers.
Its much lower jerk rate is not evidence of better walking: the behavioural
objective is largely unmet. Original GAIL-random has zero falls in the original
scene despite near-zero progress, illustrating why falls cannot stand alone as
a success measure. Negative normalized scores are valid: zero is a random
reference, not an absolute mathematical floor.

### F.6 What the new probes add

Values below are mean first-layer test R² over three trained seeds, with pooled
expert CKA averaged over those same representations. “Cont. / scratch” retains
the direction used above. These are readouts on the **same oracle states**, not
predictions of each learner's own future trajectories.

| Family | Phase × roll R² | Future velocity R² | Future pitch R² | Expert CKA |
|---|---|---|---|---|
| BC / hidden | 0.6911 / 0.6643 | 0.6330 / 0.6287 | 0.5988 / 0.6044 | 0.9409 / 0.9387 |
| BC / input | 0.7000 / 0.6626 | 0.6409 / 0.6406 | 0.5988 / 0.6031 | 0.9409 / 0.9391 |
| DAgger-M / hidden | 0.7065 / 0.6935 | 0.6294 / 0.6340 | 0.5876 / 0.6101 | 0.9212 / 0.9428 |
| DAgger-M / input | 0.7073 / 0.6854 | 0.6354 / 0.6385 | 0.5862 / 0.6112 | 0.9215 / 0.9381 |
| GAIL-BC / hidden | 0.6910 / 0.6644 | 0.6330 / 0.6287 | 0.5988 / 0.6044 | 0.9409 / 0.9387 |
| GAIL-BC / input | 0.7000 / 0.6626 | 0.6408 / 0.6406 | 0.5989 / 0.6031 | 0.9409 / 0.9391 |
| GAIL-DM / hidden | 0.7063 / 0.6936 | 0.6294 / 0.6340 | 0.5876 / 0.6101 | 0.9212 / 0.9428 |
| GAIL-DM / input | 0.7073 / 0.6854 | 0.6354 / 0.6385 | 0.5862 / 0.6112 | 0.9216 / 0.9381 |

Continuation preserves more linearly accessible phase-conditioned roll
information across these families. Hidden DAgger is 0.7065 versus scratch
0.6935; input DAgger is 0.7073 versus 0.6854. This is consistent with different
representational organization after longer training, but it does not identify
the cause of the fidelity advantage.

Other properties go the opposite way: scratch DAgger improves future-pitch
readout to about 0.610–0.611 from 0.586–0.588. Scratch hidden DAgger also has
higher expert CKA (0.9428 versus 0.9212), despite worse action MSE. Therefore,
neither higher CKA nor higher R² on one future target is a reliable proxy for
better imitation. GAIL's selected readouts remain very close to its initializer
in both cohorts, reinforcing the conclusion that this conservative update
changes the measured hidden representation little.

The direct-random actors are an especially useful cautionary example: they can
expose information on oracle states while failing to walk themselves. The next
table includes their first layer and the existing random-feature/raw controls.

| Representation | Phase sin R² | Phase × roll R² | Future velocity R² | Future pitch R² |
|---|---|---|---|---|
| Raw63 | 0.9904 | 0.3624 | 0.5951 | 0.5634 |
| Untrained random63 h1 | 0.9646 | 0.4559 | 0.6114 | 0.5678 |
| Varied GAIL-random h1 | 0.9646 | 0.4557 | 0.6114 | 0.5677 |

Varied GAIL-random future-velocity R² is about 0.6114—close to the untrained
random-feature control—despite failed locomotion. Decoding information supplied
by oracle observations is not evidence that the policy learned to exploit it.
The original width/nonlinearity confound with the expert, periodic shuffled-label
controls and absence of causal interventions still apply. No R1 claim follows.

### F.7 Revised overall conclusions

1. **Original-environment pretraining is not necessary for viable varied-domain walking at these budgets.** Scratch BC and mixed DAgger, and GAIL initialized from them, reach approximately oracle-level progress. Their complete training histories differ from continuation, so this is not a matched-lifetime compute claim.
2. **Pretrained continuation remains substantially better at matching actions.** Approximately fivefold lower errors coexist with similar or slightly lower progress. Fidelity and performance should be reported separately, not collapsed into one winner.
3. **DAgger's fidelity advantage survives scratch initialization.** The varied-domain result is not simply inherited from the original DAgger parent. Extra supervised optimization remains a competing explanation alongside learner-state coverage.
4. **GAIL strongly depends on a competent initializer under this configuration.** Direct random starts fail; supervised starts work. Additional GAIL yields small, mixed behavioural differences and near-initializer probe profiles, not a demonstrated general improvement.
5. **Explicit slip context remains marginal and inconsistent.** Scratch initialization does not reveal a broad context advantage absent in the continuation batch.
6. **Representational similarity, decodability and control success are distinct.** Higher scratch DAgger CKA coexists with worse fidelity, and failed GAIL-random still admits informative probes. These are stronger reasons to avoid interpreting CKA as a percentage of transferred knowledge.

These comparisons reuse controls and draws; they are not new independent
confirmation experiments. Original-scene retention of varied-trained models,
matched lifetime optimization, independent full training histories, context
interventions and causal feature ablations remain open. The new data strengthen
the descriptive findings without resolving those causal questions.

### F.8 New source artifacts and reproducibility

- [Combined evaluation JSON](../experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json) and [per-replica CSV](../experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.csv).
- [Scratch training summary](../experiments/consolidated/v1/scratch_v1/training_summary.json).
- [Combined probe report](../experiments/consolidated/v1/scratch_v1/probing/report.json) and [paired initialization probe comparisons](../experiments/consolidated/v1/scratch_v1/probing/initialization_comparison.csv).
- [Scratch protocol](../scratch_comparison/README.md); [read-only extension builder](analysis/scratch_summary.py).
- [Preserved report before this extension](report_archive/CONSOLIDATED_RESULTS_REPORT_2026-10-15_before_scratch_2026-10-16.md). Earlier meeting summaries and source experiments are unchanged.

The papers in Appendix E remain the methodological references. This extension
does not introduce a new learning algorithm or claim that literature establishes
the measured outcomes. Source hashes below fix the new evidence used here.

| Source | SHA-256 |
|---|---|
| experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json | 788a1fb23f05052c7920aed82591bc5422a451d050cdfa53b0e584549d1e04e7 |
| experiments/consolidated/v1/scratch_v1/probing/report.json | 0aeeb19e93e5a89816b8404f1c8952d33fb90b8e1360fd71a8728ade7f928066 |
| experiments/consolidated/v1/scratch_v1/training_summary.json | 37fa86b2bbbb96ccb68285a2625ae7e30c2f584004c4f26c17547bbcaf54c997 |
