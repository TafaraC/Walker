# General representation study: existing NAO source models

Expert extension: use `./run_expert_probing.sh audit`, then `run` to append the
FC Portugal64-unit tanh expert and matched random controls to these completed
results, without rerunning students. Adds property-retention comparisons and
label-free CKA/spectral diagnostics. Full protocol: [EXPERT.md](EXPERT.md).

This is the main **source-side** probing pipeline for the R1 initialization
question. It supersedes the slip-centred research direction, not the historical
results in experiments/probing/original_policies_v1. Neither code nor receipts
from that completed study are changed.

## Run

```bash
./run_representation_probing.sh audit
./run_representation_probing.sh run
./run_representation_probing.sh report
```

Run from Walker. No SimSpark, GPU, policy retraining or R1 process is required.
The user launches the research fits. The launcher limits CPU numerical libraries
to one thread and uses fcportugal Python; PYTHON_BIN overrides its location.
Report generation is automatic after run. Output:
`experiments/representation_probing/nao_sources_v1/`.

Only these original checkpoint files are used, frozen:

- BC: bc_walk_policy.pth.
- Mixed DAgger: dagger_mixed_policy.pth.
- GAIL initialized from **original DAgger**: gail_dagger_policy.pth.
- GAIL initialized from BC: gail_bc_policy.pth.

No FiLM, input-conditioned BC, varied-arm continuation, plain DAgger or
GAIL-random is substituted. Random networks are explicit diagnostic controls.
GAIL uses its actor mean, not critic or exploration variance.

## Research question and scope

Which gait, sensorimotor and short-horizon balance properties are accessible
in the original learned representations? Which source layers influence the
original actor's outputs when disrupted? These create hypotheses to compare
with R1 learning outcomes. They do **not** establish which source makes the best
R1 initialization: that requires the target-domain learning experiment.

The source actor is63 ->256 ReLU ->256 ReLU ->16 gait-generator residuals.
Outputs are not R1 joint targets. This pipeline does not implement adapters,
IK/WBC transfer, R1 fine-tuning, or an R1 observation conversion. R1 remains in
the alternate repository; no files there are read or changed by this launcher.

## Thirty properties, without a slip target

The observation mapping was inspected in behaviors/custom/Walk/Env.py. All
targets are labelled explicitly by how they are obtained; nothing assumes a
missing contact/centre-of-mass sensor exists.

| Group | Labels and interpretation |
|---|---|
| Generator phase (2) | sin/cos of pi*(step progress + right-active indicator); phase of the stateful generator, not observed stance |
| Foot force/support proxy (4) | left/right force norms in N, and norm>10N indicators; force-sensor threshold, not true ground-contact annotation |
| Body state (7) | head height, vertical velocity, roll, pitch, gyro xyz |
| Limb geometry (6) | left/right ankle xyz relative to the hip centre, metres |
| Command state (3) | smoothed forward/lateral target displacement in metres and relative orientation in degrees; **not desired velocity** |
| Phase-conditioned posture (2) | sin(phase)*roll and cos(phase)*pitch, degrees; constructed interaction labels, not validated stability margins |
| Future dynamics (6) | next0.2s mean oracle forward velocity/direction error; endpoint roll/pitch and two force-contact indicators |

Current quantities are recovered from63-D observations: height obs1/3,
vertical velocity obs2*2, torso angles obs3:5*15, gyro obs5:8*100,
force vectors obs14:17 and20:23 times100, ankle coords obs23:29 divided by
(8,8,5) per leg; phase obs55:58; commands obs58:61 scaled by(.5,.5,45).
The compatibility bug duplicating target-x velocity at obs62 is left unchanged;
no invented target-y velocity label is used. Foot **ankle** positions are not
reported as foot contact points or body-scale-normalized R1 coordinates.

Direction error is the recorded norm between movement and target unit vectors
(the evaluator uses1 when direction is undefined), not an angle in radians.
Future outcomes are under the **oracle's** actions, not each candidate's actions.
Current stance/phase/posture are primarily retention checks because their inputs
are directly supplied. Interaction and future labels ask more than direct copying.

Missing here: true contact events independent of force thresholds, COM/ZMP/capture
point, labelled balance margin, recovery success and desired-versus-achieved
velocity commands. Do not interpret the above proxies as those measurements.

## Data, splits and baselines

The default dataset reuses30 receipt-verified corrected-scalar oracle runs from
experiments/context_conditioning/bc_input_v1/evaluation_all_v1. All four models
see exactly the same states. Data span several surfaces, but slip is only source
provenance/stratification metadata and is not a decoding target. Thus these remain
oracle-state diagnostics across the existing distribution, not all possible NAO
behaviours or learner on-policy states. Broader source coverage needs new data.

Whole-run split: grouped draws0/1 ->train(10 runs); grouped draw2 ->validation(5);
confirmation ->test(15). Previously inspected confirmation cases are held out from
probe fitting, not newly blind experiments. No shuffling of adjacent frames into
different splits. Default stride7 avoids repeatedly sampling a fixed residue of
the nominal8-step half-cycle. Horizon10 steps=.2s. Drop windows crossing recovery,
unobserved endpoints, reset observations and states with no same-episode previous
transition. Audit counts:4270/2133/6405 rows. Rows are not independent trials.

Representations: raw63-D input; restricted phase/command obs55:61 (6-D); h1/h2
of each frozen actor; h1/h2 of three architecture-matched random networks, seeds
42/43/44. Sixteen representations total. Restricted-input probes help identify
properties explainable by phase/command alone rather than richer sensorimotor
state. They are not source-policy training baselines.

Each representation gets true labels and three whole-trajectory shuffled-target
controls, using the existing block-shuffle procedure. Independent permutations
within train/validation/test; donor relative-time rows are mapped when lengths
differ. This preserves some shared periodic structure and is **not** a perfect
zero-information null, especially for phase/posture. Inspect controls directly;
do not call score differences formal causal selectivity or p-values.

Readouts: train-only standardized ridge regression, alpha candidates
1e-6/1e-5/1e-4/.001/.01/.1/1/10/100. Choose separately for each target using
validation macro-run MSE; never test scores. Objective: mean squared residual
plus alpha*sum(weight^2), unpenalized intercept. We extended the grid because the
previous study often selected its lowest candidate; still inspect boundary hits.

Contact indicators use ridge scores thresholded at.5 for accuracy and balanced
accuracy; no probability calibration claim. Binary-target RMSE/R2 also remain
available. If only one class occurs, balanced accuracy is null. A mean-target
baseline is reported for all targets. Six future targets additionally have
persistence baselines: previous completed transition's velocity/error, or current
posture/force-contact state. No future observations enter persistence inputs.

64 multitarget fits (16 representations x4 label conditions),30 scalar readouts
each. Train only these diagnostic readouts; original model weights never update.
Seeds here randomize controls, not new source-policy training replicas.

## Controlled disruption of the original source model

After probes, create **in-memory copies** of each actor. Reset exactly one affine
layer (input, middle or output, including bias) using the same PyTorch random
initializer as the scratch architecture, at each of seeds42/43/44. Other layers
stay byte-identical. Never save over or modify source files.36 interventions total.

Measure latent-action RMS change versus the intact actor on shared test states,
overall and per run. This is functional sensitivity of the original source model.
No disrupted model is deployed. A large effect can result from a broken feature
interface or scaling mismatch; it does not identify a semantic representation,
show beneficial knowledge, or prove a loss of R1 transfer advantage. Output-layer
reset is a useful sensitivity comparison, not an encoder-information test.

`reset_layer` is a reproducible source-actor manipulation function; its actor-only
keys are0/2/4, not the original net./actor_mean. prefixes. R1 integration must map
them deliberately. This pipeline does not export replacement transfer checkpoints.

## Artifacts, resuming and reporting

plan.json fixes sources/checkpoint/data hashes, observation-decoder source hash,
30 target names, split counts, strides, penalties and library versions.
Each probes/<representation>/<true|shuffled_seed>/ contains probe.npz (weights,
scalers, chosen penalties), predictions.npz (targets, predictions, run/step indices),
metrics.json and hashed completed.json. interventions/ has hashed metrics with
per-run action changes. report.md is the true-label table; report.csv/json include
all controls, RMSE/R2, macro-run RMSE, contact scores; JSON additionally includes
persistence and intervention results. Negative R2 is valid; constant targets give
null R2. Metrics in different physical units must not be averaged as one score.

Flags: --data, --output, --stride (default7). Same flags are needed for report.
Changed inputs require a new output; completed subprobes resume only after hash
checks. An interrupted nonempty unreceipted subprobe is preserved and rejected:
archive just that directory before retry, or choose a new output. Only one writer
per output directory. Never edit old receipts or claim a changed run is a resume.

## Connecting to R1 results later

Retain the source SHA-256 and exact adapter/actor architecture, target training
seed, source-frozen/trainable status and reset layer/seed in the R1 run manifest.
Save intact-before-transfer, after-adapter-only learning, intermediate and final
actor checkpoints. Record held-out R1 episodes with observations, source-network
inputs, h1/h2, meaningful physical labels, actions, resets and simulation timing.
Keep training/evaluation episodes and units explicit. Cross-domain phase/contact
definitions must be reconciled; do not apply NAO63-D indices to R1 arrays.

Compare matched-architecture scratch, frozen random with identical adapters,
frozen learned source, intact fine-tuning and reset-source fine-tuning under the
same R1 budget/rewards. Native R1 PPO is a separate practical benchmark. Probe
associations can suggest a mechanism; loss of transfer advantage after controlled
disruption is stronger evidence, subject to the interface-disruption caveat.
No causal transfer ranking is generated by this source-side pipeline.

## References and validation

Primary methodology: [Alain & Bengio, Understanding intermediate layers using
linear classifier probes (2016; revised2018)](https://arxiv.org/abs/1610.01644).
We adapt frozen-layer linear decoding to continuous regression, not reproduce
their classification experiments. Control motivation: [Hewitt & Liang (2019),
Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/).
Our trajectory controls and layer resets are project-specific adaptations, not
their exact word-type control task. Layer-reset action sensitivity is a separate
engineering diagnostic, not a result claimed by either paper.

Tests: `python -m unittest representation_probing.test_pipeline probing.test_pipeline`.
Synthetic tests check label units/timing, reset boundaries, layer isolation,
frozen parameters, scoring, end-to-end reports, resume and artifact integrity.
