# Unitree R1 RL baseline and FC Portugal transfer: agent handoff

## Status update — 2026-10-11

The user has deferred R1 transfer from the current Walker study until progress
resumes in the alternate directory. This handoff remains future-work reference;
its proposed experiments are not current Walker deliverables. No failure cause
or negative-transfer result is established by this decision. Continue the NAO
analysis independently, and do not claim R1 utility from source probes. This update
does not request deletion of target artifacts, cancellation of processes or changes
in the adjacent repository. Earlier implementation plans below remain historical.

Prepared 2026-10-10 from the local repositories. This is a task brief and verified
source-model inventory, **not a claim that transfer is implemented or successful**.

## 1. User intent and workspace boundaries

The user wants two separate workstreams:

1. Continue context-hidden/input-conditioned/FiLM walking studies in **Walker**.
2. In the adjacent **Unitree R1 RL project**, first establish a normal target-robot
   RL agent trained from scratch. Then compare original FC Portugal imitation
   policies as sources of initialization/representations for target RL learning.
   Later probe the source/transferred models to investigate what may explain
   different transfer performance, if a reproducible difference exists.

Paths on this machine:

- Source repository: `/home/pamnt/robocup/Walker`
- Target repository: `/home/pamnt/robocup/unitree_rl_mjlab`
- This brief: `/home/pamnt/robocup/Walker/UNITREE_R1_TRANSFER_HANDOFF.md`

The receiving agent should open the **target repository** as its workspace and
read its own instructions/handoff first, if present. Walker source models are
read-only inputs. Do not modify its models, collectors, context experiments,
simulator settings, receipts or immutable plans. Changes in Walker's Python sources
can invalidate ongoing experiment identities. Keep target code, adapters, training
logs, checkpoints and transfer manifests in the target repository. This handoff
was written in Walker; the adjacent repository was only inspected, not modified.

Do not start long training, install dependencies, alter drivers or deploy to real
hardware merely because this document includes commands. The established workflow
is to prepare/test tooling offline and let the user initiate experiments. This
study is simulation-only unless hardware work is separately authorized. Training
shares machine/GPU resources with Walker: coordinate before concurrent GPU jobs.

## 2. First deliverable: an ordinary R1 RL baseline

Interpret "normal baseline agent" as a native R1 PPO agent trained **from random
initialization**, not a policy that executes uniform random actions forever.
Start with the registered flat-ground velocity task `Unitree-R1-Flat`. Preserve
the repository's native reward, observation, action, termination and randomization
settings for this engineering baseline. Do not add source-policy inputs yet.

Local files inspected:

| Purpose | Target-repository path |
|---|---|
| Training and playback | `scripts/train.py`, `scripts/play.py` |
| Task registration | `src/tasks/velocity/config/r1/__init__.py` |
| Flat/rough R1 configuration | `src/tasks/velocity/config/r1/env_cfgs.py` |
| PPO defaults | `src/tasks/velocity/config/r1/rl_cfg.py` |
| Shared observations/actions/rewards | `src/tasks/velocity/velocity_env_cfg.py` |
| R1 mechanics and actuator scaling | `src/assets/robots/unitree_r1/r1_constants.py` |
| Robot model | `src/assets/robots/unitree_r1/xmls/r1.xml` |
| Setup guidance | `doc/setup_en.md`, `README.md` |

Source-inspected PPO defaults: actor and critic hidden widths512/256/128, ELU,
observation normalization enabled, Gaussian actor with initial scalar std1.0;
24 rollout steps/environment,5 learning epochs,4 minibatches, learning rate1e-3,
adaptive schedule, gamma.99, GAE lambda.95, desired KL.01, clip.2, entropy.01,
max grad norm1.0, max_iterations10001, save_interval100. Record resolved runtime
configuration rather than relying on these notes if the target repository changes.

Shared physics configuration uses0.005s steps and decimation4 (nominal0.02s control
interval), with20s episodes. Actor terms include angular velocity, projected
gravity, velocity command, phase, relative joint positions/velocities and previous
actions. Actor observation corruption is enabled; critic includes additional
privileged quantities. Flat R1 removes terrain height scans. JointPositionAction
uses default pose offsets and R1-specific per-actuator scaling. These semantics
are not those of the NAO policy, despite a similar nominal control frequency.

The XML contains24 named hinge joints plus a floating base; **measure the resolved
action and actor/critic observation dimensions/order at runtime**, including any
configured exclusions or version-dependent term outputs. Do not infer an observation
dimension from this XML or assume this model is identical to a particular physical
R1 EDU SKU. Asset names and actual configurations take precedence over robot branding.

Candidate user-run commands, to verify against local CLI help/environment first:

```bash
cd /home/pamnt/robocup/unitree_rl_mjlab
python scripts/train.py Unitree-R1-Flat --help
# After an agreed smoke/budget check, an example native training invocation:
python scripts/train.py Unitree-R1-Flat --env.scene.num-envs=4096 --agent.seed=42
```

These commands were not executed here.4096 environments is a repository usage
example, not a memory-safe guarantee; select environment count/smoke horizon for
the available GPU. Read `doc/setup_en.md`: it proposes a separate Python3.11
environment named unitree_rl_mjlab. Walker currently uses a Python3.9 fcportugal
environment with native SimSpark modules. Do not reuse that environment blindly.
Inspect installed mjlab/RSL-RL versions and run task/config import checks before
choosing launch flags. `train.py` sets both agent and environment seed and offsets
seed by distributed rank. Capture number of environments, ranks and total samples.

Before transfer work, deliver:

- A reproducible native baseline launcher/config and a short smoke validation.
- Saved resolved config, software/GPU versions, source revision and seed manifest.
- Periodic checkpoints, observation-normalizer state and training/evaluation logs.
- A target-domain evaluation protocol: held-out command sequences, resets/seeds,
  deterministic actor means versus stochastic evaluation stated explicitly.
- Fixed-budget learning curves and a final baseline evaluation; do not equate
  training reward alone with successful, stable locomotion.

## 3. Original Walker checkpoints available

These are **original context-free root checkpoints**, not the newer varied-slip,
input-conditioned or FiLM outputs. Verified to load as PyTorch state dictionaries
on2026-10-10. Recompute hashes before copying/using. Paths are relative to Walker.

| File | Meaning | Network |
|---|---|---|
| `bc_walk_policy.pth` | Original behavioural cloning |63→256→256→16, ReLU |
| `dagger_policy.pth` | Original DAgger | Same BCPolicy architecture |
| `dagger_mixed_policy.pth` | Later mixed long-run/short-turn DAgger variant | Same BCPolicy architecture |
| `gail_dagger_policy.pth` | GAIL with DAgger-derived initialization | Actor63→256→256→16; separate critic |
| `gail_bc_policy.pth` | GAIL with BC-derived initialization | Same GAIL actor/critic format |
| `gail_random_policy.pth` | GAIL trained from random initialization | Historical additional comparator, not the target scratch-RL baseline |

`gail_policy.pth` is **absent** at inspection. Do not invent it or silently replace
it with whichever checkpoint is newest. "Original three models" historically
means BC, original DAgger and the original GAIL lineage; newer work selected mixed
DAgger, GAIL-BC and GAIL-DAgger. Present this inventory and confirm the final source
comparison set before launching an expensive sweep. Retain distinct labels rather
than merging both DAgger variants or both GAIL initializations under one name.

SHA-256 inventory:

```text
bc_walk_policy.pth     cf66a37641baa6b1307758b9a66bc0e0092ce84049fa0cf70fc08c63fa63c580
dagger_policy.pth      4b8657ed65fc4ee0a4f307e74b24723ae5d8f98cde3c727dfb84f2ae08c68b21
dagger_mixed_policy.pth 64d667bc74c43fec072df7958a4ce8d199b5cbf2e08c97a82c83e162d8d9849b
gail_dagger_policy.pth  2f747931dc18891f74ea85804e4f47ac804ba7ae3895acc4142171b475ac9ba0
gail_bc_policy.pth     bf5d6ea9a903565262e6f6ade33d5c68a2e28d1360a2b6eaab16d70239018e8e
gail_random_policy.pth f0c72238dff35d6cdc5c450325a27284ef402672f483389e5fe88806a03ece74
```

BC/DAgger class: `train.py:BCPolicy`. Weights/biases use `net.0`, `net.2`, `net.4`;
weight shapes respectively[256,63], [256,256], [16,256]. Linear final output,
no output tanh. Normalizations below are part of the hand-engineered observation
contract, not a learned state-normalization object embedded in these checkpoints.

GAIL class: `gail_models.py:GAILActorCritic`. Actor tensors use
`actor_mean.0/.2/.4` with the same dimensions; critic uses `critic.0/.2/.4`, ending
in[1,256]. `actor_log_std` has shape[1,16]. The source forward pass clamps logstd
to[-5,-1]; deterministic evaluation uses `actor_mean`, not samples. For actor-only
transfer, extract the actor mean weights. **Do not transplant the GAIL critic,
discriminator, optimizer state or exploration scale into R1 by default**: their
reward/action meanings differ. GAIL's name describes how its source actor was
trained; the target experiment can use the same PPO algorithm for every source.

Minimal source inference without a simulator (run from Walker or use its path
on PYTHONPATH; trusted local checkpoints only):

```python
import torch
from train import BCPolicy
from gail_models import GAILActorCritic

bc = BCPolicy(63, 16)
bc.load_state_dict(torch.load("bc_walk_policy.pth", map_location="cpu", weights_only=True))
bc.eval()
gail = GAILActorCritic(63, 16)
gail.load_state_dict(torch.load("gail_dagger_policy.pth", map_location="cpu", weights_only=True))
gail.eval()
with torch.no_grad():
    obs = torch.zeros(1, 63)  # shape smoke only, NOT a realistic walking state
    bc_action = bc(obs)
    gail_action = gail.actor_mean(obs)
```

Avoid importing `Walk`/`Base_Agent` just to inspect networks: those pull in native
FC Portugal modules. Reimplementing the small torch architecture in the target
repository is acceptable if state keys, shapes, activations and numerical outputs
are checked against source inference. Export/copy with hashes; never overwrite
source checkpoints. Whole module pickle loading is unnecessary for these .pth files.

## 4. Exact source observation and action contract

Authoritative files: `behaviors/custom/Walk/Walk.py`, `behaviors/custom/Walk/Env.py`,
`behaviors/custom/Step/Step_Generator.py`, and the robot/IK implementations.
The source robot is SimSpark NAO type0 in the current collectors, with22 joints.
Control actions are issued at0.02s intervals. Python slices below exclude the end.

| Indices | Source meaning/scaling |
|---|---|
|0|`min(step_counter,120)/100` |
|1,2|`loc_head_z*3`, `loc_head_z_vel/2`; inspect the actual sensor/localization definitions rather than relabeling as R1 base height |
|3,4|Torso roll/pitch in degrees divided by15 |
|5:8,8:11|Gyro/100, accelerometer/10; verify source frames/units before mapping |
|11:17,17:23|Left/right foot contact point+force, multiplied by(10,10,10,.01,.01,.01); zero if absent |
|23:26,26:29|Left/right ankle positions relative to hip center, multiplied by(8,8,5) |
|29:32,32:35|Left/right foot roll/pitch/yaw relative to torso, degrees/20 |
|35:39|Source arm joint positions[14:18]/100 |
|39:55|`joints_target_last_speed[2:18]`, NOT measured joint velocity or a generic last16 actions |
|55:58|Step-generator progress and left/right active-leg flags |
|58:60|Smoothed planar target divided by.5m |
|60|Smoothed orientation divided by45degrees |
|61,62|Both contain internal target **x** change divided by.014m |

The duplicated x value at index62 is a documented compatibility bug intentionally
preserved for these trained models. Do not silently fix it during a purported
faithful source-policy mapping. Target/orientation smoothing and phase/reset state
are part of the controller: max target increment.014m, max orientation increment
1.6degrees per step. The oracle wrapper converts absolute navigation targets to
robot-relative targets; this is not directly the R1 velocity-command interface.

The16 output values mean:

- 0:3 and3:6: left/right ankle-position residuals.
- 6:9 and9:12: left/right foot-rotation residuals.
- 12:16: left/right arm pitch and roll residuals.

They pass through target-distance scaling, an exponential moving average
`act = .8*act + .2*action*action_mult*.7`, a stateful step generator, geometric
offsets/limits and NAO inverse kinematics. Position residual scales are.02/.02/.01m;
foot rotation scales3/3/5degrees plus yaw biases/limits. Arm commands add4degree
scaled residuals to a default posture and phase-dependent swing. Source joint
position targets are then converted through the robot's controller. **These16
numbers are neither22 joint velocities nor24 R1 joint targets/torques.**

The FC Portugal oracle is a separate NumPy MLP (`walk_R0.pkl` for type0), evaluated
with tanh hidden activations via `math_ops/Neural_Network.py:run_mlp`. Do not confuse
it with the ReLU imitation networks or claim that unchanged FC Portugal is a
ready-made R1 oracle. Different morphology requires a new adapter/controller and
target validation.

## 5. Transfer design: isolate what actually transfers

Do not directly load a63→16 policy into the native R1 PPO model. Both outer-layer
dimensions and internal architecture/activation differ. Padding observations to63
or copying the first16 target joints does not create semantic compatibility.
ODE/SimSpark slip numbers are not MuJoCo friction coefficients either.

Recommended staged experimental design (implementation choices, not existing code):

1. **Native scratch PPO:** standard R1 task/architecture. Establish that the target
   installation learns a stable gait and that evaluation works.
2. **Architecture-matched scratch PPO:** choose a transfer-compatible architecture
   and train it with all actor weights random. Keep target critic architecture,
   reward, data budget and exploration treatment matched to transfer arms.
   This is essential: comparing a256/256 ReLU transferred actor against only the
   native512/256/128 ELU baseline confounds transfer with architecture.
3. **Pilot middle-layer transfer:** target actor `D_R1→256→256→A_R1`, ReLU. New
   target-specific input/output layers; copy only the source256×256 middle-layer
   weights and256 bias (`net.2` or `actor_mean.2`). This is a conservative,
   mechanically compatible test of representation initialization, **not direct
   gait transfer**. Pair random outer-layer initialization across sources/seeds.
4. **Optional richer transfer:** a learned/engineered observation adapter into the
   source representation and a target-specific action head/decoder. Specify its
   semantics, trainable/frozen blocks, losses and supervision. A learned63-D latent
   adapter output is not automatically a physically faithful NAO observation.
   Retaining the source16-D action bottleneck requires an explicitly validated
   R1 decoder/gait controller; it is a distinct study with a matching random-source
   adapter control. Never reuse NAO IK geometry as R1 IK without adaptation.
5. Fine-tune with **the same target PPO objective** for BC-, DAgger- and GAIL-derived
   initializations. Start target critic and running observation normalizer fresh
   and identically across arms; use matched target exploration and optimizer state.
   Any reuse of source critic/statistics must be a separately justified ablation.

The middle-layer pilot avoids claiming a zero-shot walking controller. If the
user instead intends direct zero-shot motor transfer, stop and agree an observation/
action correspondence before implementing it. The target agent must document every
tensor copied versus initialized, and check shapes, finite actions, limits and
load-time numerical consistency. Do not use permissive `strict=False` loading as
a substitute for an explicit transfer mapping.

## 6. Fair target learning comparisons

- Match target rewards, commands, terrain, resets, randomization, terminations,
  control frequency, number of environments, rollout length and update budgets.
- Report learning curves against **total target environment transitions**, not
  iteration count alone; vectorized sample count depends on environments/ranks.
  Track wall-clock cost and distinguish reset/initialization overhead if relevant.
- Evaluate initial, intermediate and final policies on predeclared held-out target
  cases. Measure command-tracking error, speed, fall/termination rate, survival,
  joint-limit violations and a consistently defined action/joint smoothness metric.
- Predefine convergence thresholds/time-to-threshold and a fixed-budget score;
  keep native and architecture-matched scratch baselines separately labelled.
- Propose multiple paired target training seeds, with their cost stated before
  launch. These seeds affect target sampling/initialization/optimization; they
  are not the Walker BC minibatch-only repeats. Source checkpoints may each be
  single pretrained realizations, so do not overstate source-algorithm independence.
- Keep inherited source supervision/compute separate from marginal R1 learning
  cost. GAIL-DAgger and mixed DAgger carry different pretraining histories.
- Do not compare scalar rewards across NAO and R1 as if they share a scale. If
  normalizing within R1, define a fixed target-domain reference and denominator;
  FC Portugal's NAO return is not an R1 expert denominator.
- Cross-morphology normalized posture/phase representations require a declared
  mapping; raw22-D NAO versus24-D R1 joint Wasserstein is not directly meaningful.

## 7. Prepare for later probing without assuming positive transfer

Save artifacts now so later analysis is possible:

- Immutable original source actor state/hash and source-identity manifest.
- Exact target actor immediately after initialization/adaptation, before PPO.
- Intermediate and final actor/critic/normalizer/optimizer checkpoints, with total
  target steps, task configuration, training seed and source provenance.
- Reproducible held-out observation/transition probes with commands, contact/phase,
  base motion, joint states, actions, episode/reset boundaries and units/frames.
- Named activation capture points (source ReLU layers0/2; source outputs; target
  encoder, transferred block and head). Store raw and normalized inputs separately.

After establishing repeatable learning differences, probe phase/contact/velocity/
command information in hidden activations, source-to-target representation changes,
and policy sensitivity to physically meaningful state changes. Use episode- or
trajectory-level probe splits, held-out command regimes and appropriate shuffled-
label/random-feature controls; no random frame splits across near-identical traces.
Compare frozen-transfer versus fine-tuned representations and random-initialized
matched baselines. Layer-reset/freeze ablations can test which transferred blocks
help; equalize additional training budgets for those interventions.

Decodability or representational similarity alone is not causal evidence. Probes
may merely read information already present in inputs; pair them with interventions
and measured learning-curve effects. Report negative transfer or no effect as valid
outcomes. Do not assume that a source policy with best NAO return will transfer best.

## 8. Relevant Walker history and cautions

- Context work stays in Walker. Current context BC comparison completed429 runs;
  correct slip produced weak offline sensitivity but no consistent live locomotion
  gain. Identical nominal-context repeats had differing first states and~.463m
  mean absolute progress differences. A separate repeatability diagnostic exists;
  synchronous control/seeding alone does not guarantee identical starts.
- FiLM and append-only comparison tooling is ready; no FiLM research result is
  asserted by this handoff. These context models are **not** the original source
  models requested for the first R1 transfer study.
- Earlier two-argument SimSpark slip edits did not establish changed effective
  contact dynamics. Later scalar-setting tests corrected that. Preserve old
  results, but do not cite legacy apparent robustness as evidence of transferability.
- Reports may use16-D latent-action MSE, standardized sliced joint Wasserstein and
  smoothed angular jerk; these are not direct motor-command error, exact optimal
  transport, torque smoothness or energy expenditure.
- `SESSION_HANDOFF.md`, `EXPERIMENT_PHASES.md`, `PIPELINES.md`, `CHALLENGES.md` and
  `UNITREE_R1_TRANSFER_HANDOFF.md` provide project context. Root checkpoint hash
  verification is the minimum provenance check; inspect archives/training metrics
  before asserting exact source budgets or historical initialization ancestry.

## 9. Receiving agent's immediate checklist

1. Inspect target instructions, git status, installed dependencies, registered R1
   tasks, asset variant and available compute without altering Walker.
2. Resolve and print actor/critic dimensions, term ordering/scaling, action joint
   order/limits and timestep from the actual target environment/configuration.
3. Prepare the native from-scratch baseline and evaluation protocol; agree budget
   and seed count with the user before long runs. User executes live jobs.
4. Inventory source checkpoints by hashes above; agree which DAgger/GAIL variants
   are in the first comparison. Do not substitute context-conditioned checkpoints.
5. Implement and test an explicit transfer mapping plus architecture-matched scratch
   control, recording every copied tensor and newly initialized component.
6. Preserve initialization/intermediate/final checkpoints and probe-ready traces.
   Interpret target learning results before pursuing explanations of "better transfer".

No R1 baseline training, transfer implementation, adjacent-repository edit or
hardware deployment was performed in preparing this document.
