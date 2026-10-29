# Next meeting: context findings and representation probing

## Scope update — 2026-10-11

R1 transfer is excluded from the current presentation/study until the separate
work makes progress. The reported snag has no diagnosed cause here and is not a
negative-transfer finding. Treat R1 as deferred future work; do not rely on its
results to justify the completed NAO experiments.

The current research narrative compares imitation methods by locomotion accuracy,
performance, stability and resource costs; examines controlled domain changes and
the marginal effects of explicit context; and characterizes which locomotion
properties are accessible in expert and imitation representations. The completed
expert/student probes support descriptive retention/reorganization findings on
NAO states. They do not establish that any source is a better R1 initialization.
Earlier proposed links to target learning/ablation outcomes below are deferred.
Keep all source results, caveats and future transfer plans for later reference.

## Completed expert comparison

All appended16 fits and parent artifacts verified. Mixed DAgger h1 and
GAIL-DAgger h2 are geometrically closest to expert h1 (CKA .936/.931); both beat
same-layer random controls on all15 test runs. Raw/random similarity is already
high, so this is not a fraction of transferred knowledge or a causal finding.

Students outperform expert linear probes on every tested layer/property pair,
but have256-ReLU layers versus the expert's64-tanh layer. Interpret as greater
accessibility under this protocol, not superior knowledge/control. Expert exceeds
its random controls particularly on command orientation, phase-conditioned roll
and future pitch. These strengthen candidate mechanisms for comparison with R1
learning. No R1 utility is established; teacher's external generator/IK is outside
the neural probes. BC/GAIL-BC profiles remain similar. Full interpretation:
experiments/representation_probing/nao_expert_comparison_v1/INTERPRETATION.md.
Earlier pending-expert-run wording below is now historical.

## Expert comparison now implemented, awaiting research run

Add the FC Portugal expert's actual64-tanh hidden layer to the same30-property
study. Compare both256-ReLU student layers against it, with matched random-tanh
controls. Reuse completed64 fits;16 new fits only. Reports quantify property
decoding gaps and broad whole-layer geometry via CKA/spectra, not a percentage
of knowledge transferred. No claim of discovering every internal representation.
Expert predictions verified against90000 recorded actions. Full details/reference
to Kornblith et al.2019 CKA: representation_probing/EXPERT.md. User command:
./run_expert_probing.sh run. R1 utility remains established by the separate study.

## General probing completed: main findings

Verified64 fits/1920 target scores and36 in-memory reset diagnostics. Strongest
candidate property is phase-conditioned roll (sin phase x torso roll): h1 R2
BC .697, mixed DAgger .703, GAIL-DAgger .586, GAIL-BC .697; raw .332 and random
.427–.452. All four learned h1 representations beat each random control on all15
test runs for this target. Shuffled scores are low. This supports useful feature
organization as a hypothesis, NOT demonstrated balance control or R1 benefit.

GAIL-DAgger retains more gyro-x/y information in h2; mixed DAgger modestly leads
future-direction-error decoding. BC/GAIL-BC remain nearly identical on these
readouts. No universal best representation or transfer source is established.

Regular gait timing survives whole-run shuffles, making phase/contact nulls weak;
future persistence baselines also perform poorly. Neither high phase scores nor
beating persistence alone demonstrates learned prediction. Layer resets change
latent actions by roughly2.5–2.8 RMS but cannot rank R1 utility or isolate semantics.
Next: compare these hypotheses with independent R1 learning outcomes and matched
target-domain interventions. Full interpretation:
experiments/representation_probing/nao_sources_v1/INTERPRETATION.md.
General probing is complete; earlier pending-run wording below is historical.

## Scope correction: general representations for transfer

The research question is which learned NAO properties make useful R1 initialization
baselines. R1 learning tests usefulness; reproducing NAO walking or decoding slip
alone does not. Retain the first probe study as preliminary. The main pipeline is
now ./run_representation_probing.sh with30 gait/force/body/ankle/command/future
labels, raw/phase-command/random/shuffle controls, and in-memory layer-reset
source-action diagnostics. General labels were audited against the actual63-D
observation mapping, with force/phase proxies explicitly distinguished from ground
truth stability/contact annotations. No slip target. Original models preserved.

R1 adapters and fine-tuning run separately; comparisons with matched scratch,
frozen random and controlled layer resets there must establish utility. Source
probes describe accessibility and generate hypotheses, not causal transfer proofs.
Implementation/audit/tests complete; general research fits await user launch.
See representation_probing/README.md for source/target checkpoint-recording contract.

## Update: probing completed and reviewed

All60 probe fits and300 scores verified. Slip R2 at h1/h2: BC .440/.384,
mixed DAgger .443/.374, GAIL-DAgger .461/.427, GAIL-BC .440/.384.
Raw linear slip R2=.285, but random-network ranges are .404–.424 (h1) and
.361–.396 (h2): much of the decoding benefit already exists without policy
learning. Velocity scores similarly differ only modestly from random controls.
GAIL-DAgger leads descriptively, not conclusively; BC/GAIL-BC probe profiles
are almost identical. Trained features retain posture better than random ones,
but posture is directly available in the input, so this is a retention check.

Shuffled slip/velocity scores are much lower than true labels. Some posture
signal survives whole-run shuffling; do not treat the controls as a perfect null.
74/75 true-label probes select the minimum tested ridge penalty .01, motivating
a separately versioned validation-selected regularization sensitivity check
before strong rankings. No new fitting or R1 work performed during review.

Interpretation: the models have access to environmental consequences without
explicit context, compatible with marginal context-model benefits, but neither
causal adaptation nor an explanation of R1 transfer has been demonstrated.
Full analysis: experiments/probing/original_policies_v1/INTERPRETATION.md.
The earlier "research fitting remains user-run" statement below records the
pre-completion plan and is now superseded.

## Decision

Stop expansion into context-conditioned DAgger/GAIL for now. Under the tested
BC data, observations and training budget, supplying slip through input
concatenation or FiLM did not establish consistent practical improvements.
The R1 RL-transfer study is proceeding separately in the adjacent directory;
Walker now focuses on frozen-policy representation diagnostics.

## Evidence from completed comparisons

Mean target progress in metres per 60-active-second run, across three fine-tuning
seeds (45 runs per architecture per sampled set, 9 nominal):

| Set | Hidden BC | Input-conditioned BC | FiLM BC |
|---|---:|---:|---:|
| Original sampled slips | 33.073 | 33.060 | 33.232 |
| Confirmation sampled slips | 32.292 | 32.665 | 32.581 |
| Nominal | 39.272 | 39.175 | 39.219 |
| Falls across 99 runs per architecture | 5 | 4 | 5 |

FiLM's held-out latent-action MSE improves only approximately .18% versus hidden
BC. Its sampled-suite joint-occupancy sliced Wasserstein distance is worse than
both BC controls, and jerk is broadly similar. Its confirmation advantage over
hidden BC is dominated by one +14.075m paired case: the other44 pairs average
about -.0234m. Keep the difficult case in official results; this is sensitivity
analysis, not an exclusion. Input conditioning also lacked consistent benefits.

The45-run repeatability diagnostic found same-controller progress SD .24–.62m,
unequal first-active states, and identical initial targets within repeated cases.
Small differences are therefore difficult to attribute; this spread is not a
universal significance threshold. Controls were reused for FiLM at matching
slips/task seeds, not identical physical starting states. Three fine-tuning seeds
share the original BC initializer and demonstrations. Confirmation draws were
previously inspected, so they are not untouched FiLM evidence.

Sources: context_conditioning/bc_input_v1/evaluation_all_v1/report.md,
bc_input_v1/repeatability_v1/repeatability.md and
bc_film_v1/evaluation_append_all_v1/comparison.md under experiments/.

## Interpretation and motivation

The oracle has no explicit slip input, but observes the physical consequences of
slip and can react implicitly. BC rewards reproducing those oracle actions, not
discovering a superior slip-specific strategy. Explicit context may add little
when ordinary observations already explain the teacher's actions. This is a
plausible explanation, not a proven mechanism or theorem that future gains must
remain superficial. Different learning incentives, tasks or dynamics could make
context more useful. RL's task objective in the separate R1 project offers a
distinct motivation to adapt rather than merely copy the teacher.

Supported conclusion: explicit context provided no consistent practical benefit
in this tested BC setting. Unsupported claims: context is always unnecessary;
the expert never adapts; the students ignore context; or FiLM would never help
DAgger/GAIL. The decision not to extend is a research-priority decision, not an
experimental result about those untested methods.

## Next work in Walker

New offline pipeline: ./run_probing.sh audit, then ./run_probing.sh run.
Policies: original BC, mixed DAgger, original-DAgger-initialized GAIL, BC-initialized
GAIL. User confirmed the GAIL mixed-DAgger description was a typo.
Frozen h1/h2 linear regression probes use identical oracle observations, whole-run
splits, raw-state and random-feature baselines, and shuffled-trajectory controls.
Targets are slip, current posture and next0.2s oracle forward velocity.
No policy retraining or simulator run; research probe fitting remains user-run.

Method: [Alain & Bengio, linear probes](https://arxiv.org/abs/1610.01644), adapted
from classification to ridge regression; control motivation from
[Hewitt & Liang (2019)](https://aclanthology.org/D19-1275/).
Details and commands: ../probing/README.md. Probes measure decodability, not causal
policy use or demonstrated R1 transfer. Those claims require further experiments.
