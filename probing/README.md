# Frozen walking-policy representation probes

## Scope and research question

Which physical properties are linearly recoverable from the frozen hidden layers
of the original BC, mixed DAgger, GAIL-DAgger and GAIL-BC actors, when all see the
same states? This is an offline representation comparison, not another walking
evaluation, policy update, causal intervention, or R1 transfer experiment.
R1 work remains in the adjacent repository. No native simulator imports/startup
or server ports are needed by this pipeline.

The user confirmed that GAIL-DAgger means the **original DAgger-initialized GAIL**,
not mixed-DAgger initialization. Only these four existing root checkpoints are used:

| Label | File | Representation |
|---|---|---|
| bc | bc_walk_policy.pth | net, two post-ReLU hidden layers |
| dagger_mixed | dagger_mixed_policy.pth | net, two post-ReLU hidden layers |
| gail_dagger | gail_dagger_policy.pth | actor_mean, two post-ReLU hidden layers |
| gail_bc | gail_bc_policy.pth | actor_mean, two post-ReLU hidden layers |

All actors are 63 -> 256 -> 256 -> 16. GAIL critic and stochastic action variance
are excluded. Models are evaluated frozen with gradients disabled. No context-BC,
FiLM, original plain DAgger, GAIL-random or varied-arm continuation checkpoints
are silently substituted. Checkpoint paths and SHA-256 hashes enter the plan.

## Paper basis

Primary: Guillaume Alain and Yoshua Bengio (2016; revised 2018),
[Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644).
Their method trains separate linear readouts on frozen intermediate features.
We adapt the readout to ridge **regression**, because slip, posture and velocity
are continuous. This is an adaptation, not a reproduction of their classification
experiments or evidence of their findings applying to locomotion.

Control motivation: John Hewitt and Percy Liang (2019),
[Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/),
EMNLP-IJCNLP, DOI 10.18653/v1/D19-1275. They distinguish probe performance from
representation quality using control tasks. Our whole-trajectory target shuffles
and random-feature baselines follow that motivation, **not** their word-type
control construction or exact selectivity statistic.

## Commands

From Walker in the fcportugal environment:

```bash
./run_probing.sh audit
./run_probing.sh run
./run_probing.sh report
```

`audit` validates existing trajectories and model shapes without writing research
outputs. `run` fits small CPU linear probes only; it never trains the policies.
`report` validates saved probe receipts and regenerates reports without refitting.
The user initiates the research run. Set PYTHON_BIN to override the interpreter.
BLAS and Torch use one CPU thread by default to avoid competing with R1 training.

Defaults:

- Data: experiments/context_conditioning/bc_input_v1/evaluation_all_v1.
- Output: experiments/probing/original_policies_v1.
- Sampling stride: 10 active simulator steps (0.2 seconds).
- Prediction horizon: 10 steps (0.2 seconds), fixed in protocol v1.
- Ridge penalties: .01, .1, 1, 10, 100, selected separately for each target on
  validation macro-trajectory MSE. Objective uses **mean** squared error plus
  alpha times squared weights, with unpenalized intercept.
- Random-network and shuffled-label control seeds: 42, 43, 44. These are diagnostic
  controls, not new training replicas of the four policies.

`--data`, `--output` and `--stride` are supported. Changing inputs, sources,
checkpoints, libraries or stride requires a new output directory. Example:
`./run_probing.sh run --stride 20 --output experiments/probing/stride20_v1`.
Keep identical flags when reporting. Completed subprobes resume with strict hashes.
An interrupted nonempty subprobe without a receipt is preserved and rejected:
archive that specific incomplete directory before retrying, or use a new output.
Do not delete or edit the historical data or relax fingerprints to resume.
Run only one probing process per output directory.

## Shared-state data and leakage controls

Use only the 30 completed oracle runs from the corrected scalar-slip context
comparison. The original four policies did not train on these evaluation traces.
Every actor receives the identical 63-D observation rows; it does **not** receive
the slip label. This isolates representation differences from on-policy state
occupancy differences. It is an oracle-state diagnostic, not a claim about the
states each learner visits after its own errors.

Splits are deterministic and by **entire run**:

- Train: original grouped draws 0 and 1 from each of five parent groups (10 runs).
- Validation: original grouped draw 2 from each group (5 runs).
- Test: all 15 confirmation runs. Nominal runs are excluded from fitting/scoring.

Every run holds one Gaussian-drawn slip constant. No per-contact random surface
is assumed. Validation/test do not overlap training trajectories/task seeds.
Exact trace duplication is rejected. Receipts, artifact hashes, complete 3,000-step
traces, scalar-scene import and physical slip metadata are checked through existing
validators. The plan records trace hashes, receipt hashes and parent provenance.
These confirmation cases were already inspected in previous performance analysis:
they are held out from **probe fitting**, not a newly blind experimental dataset.

Each sampled state has five regression targets:

| Target | Definition / interpretation |
|---|---|
| log1p_slip | log(1 + configured scalar field slip), constant within run |
| head_height_m | measured current head height |
| torso_roll_deg | measured current roll |
| torso_pitch_deg | measured current pitch |
| next_200ms_mean_forward_velocity_mps | mean of the next ten recorded transition velocities under the oracle |

Velocity[t] describes transition t -> t+1. A window is retained only if all states
from t through t+10 have the same episode_id. This excludes recovery crossings and
the last ten states without a fully observed endpoint. Recovery time is not inserted
as walking time. Indices and trajectory IDs are saved alongside predictions.
The audit found 2,990 train, 1,493 validation and 4,485 test rows with defaults.

Posture probes are **information-retention checks**: posture is already directly
represented in the input. The raw-input probe should therefore be strong. High
posture scores alone do not demonstrate learned dynamics. Future velocity is an
oracle continuation label, not the candidate actor's predicted consequence.
Slip can correlate with posture, speed, task/phase and target sequence. Decoding
it does not show an explicit internal friction estimator or causal adaptation.

## Probes and controls

Features are raw observations (63-D), each trained actor's h1/h2 (256-D each),
and h1/h2 of three independently initialized architecture-matched random networks.
Random networks share the same initialization distribution across comparisons and
are never pretrained. A train-target mean baseline is reported for every target.

Feature/target means and standard deviations are learned from training data only;
constant columns use scale 1. Hyperparameters use validation labels only. Test
labels never enter fitting, scaling or selection. A linear model with L2 penalty
keeps readout complexity controlled; the raw-input dimension differs, so compare
trained layers primarily against same-width random layers as well as raw inputs.

Each representation has true labels and three shuffled-label controls. Within
each split, whole target blocks are cyclically reassigned using a seeded random
trajectory order (no trajectory receives its own block). Donor rows are mapped by
relative index if lengths differ after recovery filtering. Train/validation/test
permutations are independently seeded. This preserves broad temporal structure
but destroys the intended state-label association. Some correlated/confounded
signal may survive a shuffle; these controls are diagnostics, not a p-value or
exact replication of the cited NLP control task.

There are 15 representations x 4 label conditions = 60 independently resumable
multitarget fits, each fitting five scalar probes and selecting their penalties.
No policy gradients or optimizer state are created.

## Artifacts and interpretation

The output contains immutable plan.json, probes/<representation>/<label_mode>/
with probe.npz (weights, scalers, selected alpha), test_predictions.npz (predictions,
labels, trajectory and timestep indices), metrics.json and hashed completed.json.
Reports are report.md, report.csv and report.json. They include true and shuffled
results, test R2, RMSE, MAE, macro-run RMSE and train-mean baseline RMSE. R2 can be
negative; constant-label test targets have undefined/null R2. Slip errors are in
log1p units, not raw slip. CSV/JSON include all control scores.

Do not treat thousands of adjacent rows as independent experimental trials.
Analyze run-level errors, parent groups and sensitivity to probe settings; this
v1 makes no significance claim or automatic model ranking. Random-control seeds
do not replace independently trained policy seeds. Original policies also differ
in training resources and lineage, so this is not an algorithm-only causal test.

Stronger decoding than random features supports accessible information. Stronger
decoding than raw inputs supports easier **linear accessibility**, not creation of
new information by a deterministic network. Similar scores can reflect redundant
state information, not proof that the models have identical representations.
Neither result proves a policy uses the decoded quantity for its decisions.

After results, consider matched layer interventions and negative controls if a
specific hypothesis warrants them. R1 transfer explanations require independently
measured transfer outcomes and matched source/target checkpoints; this pipeline
cannot infer that one source will transfer better. Keep R1 conclusions separate.

Engineering checks:
`python -m unittest probing.test_pipeline`. Synthetic tests are not research results.
