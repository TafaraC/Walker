# Consolidated generation: general locomotion probes

```bash
./run_consolidated_probing.sh audit
./run_consolidated_probing.sh run
./run_consolidated_probing.sh report
```

Run automatically writes reports; `report` rebuilds them from verified fits.
No simulator, GPU, policy training or new trajectories are required. CPU numerical
libraries use one thread. Only one invocation may write an output at a time.
Default source: `experiments/consolidated/v1/evaluation/all_all_correct`.
Default output: `experiments/representation_probing/consolidated_v1`.
`--data`, `--output`, `--stride` (default7) override these; repeat flags on resume.
An interrupted, unreceipted subprobe must be archived before retry. Changed inputs
require a new output, never edited receipts. Historical probes remain untouched.

## Scope and fair comparisons

All28 consolidated actors are frozen:4original,12varied hidden,12varied input.
Keep training seeds42/43/44 separate. GAIL's DAgger initializer in this generation
is **mixed DAgger**, unlike the historical GAIL-Dagger checkpoint. The evaluator's
checkpoint hashes, controller metadata and native expert hash enter the probe plan.
GAIL probes actor means only, not discriminator, critic or exploration variance.

Every actor and the expert sees identical oracle states from the completed varied
evaluation. Input models receive the recorded,12-significant-digit fixed scalar
slip through log1p(slip)/log(11). This tests representations across common states,
not each model's on-policy state distribution. Original-scene and scalar-nominal
retention runs are excluded from probe fitting; grouped draws0/1 train(10runs),
draw2 validates(5), confirmation tests(15). These previously evaluated cases are
held out from probe fitting, not newly blind experiments. Recovery-crossing windows
and synthetic reset observations are excluded. Horizon10steps=.2s, stride7.

## Properties and controls

Reuse the30 labelled properties and units from
[the general representation protocol](../representation_probing/README.md):
generator phase; force/contact proxies; torso posture/angular velocity; ankle
geometry; command displacement/orientation; phase-posture interactions; six
short-horizon outcomes under oracle actions. **Slip is not a prediction target.**
This cannot discover every internal representation; it tests these candidate
properties and descriptive whole-layer geometry.

75 representations:56student layers,12random student layers (both63/64input
architectures,3seeds), expert h1,3expert-random controls, raw63inputs,
raw64inputs including context, and restricted phase/command inputs.
Each gets true labels plus3whole-trajectory shuffled-label controls:300multitarget
ridge fits,9000scalar property scores. Scalers use training data only; each target
chooses ridge alpha by validation macro-run MSE. Test reports include RMSE,R2,
macro-run RMSE, mean baselines and thresholded contact accuracy/balanced accuracy.
Future targets also have persistence controls. Negative R2 is valid; constant-target
R2 is undefined. Do not average different physical-unit RMSEs.

Reports include student/expert R2 gaps, pooled and per-run linear CKA, within-run
permutation CKA controls, and representation spectra. Expert has64tanh units;
student layers have256ReLU units. Width/nonlinearity and directly encoded inputs
confound claims of transferred knowledge; a gap is **not a percentage retained**.
Random64-input and raw64-input controls matter because context itself is extra
information. Shuffling trajectories preserves some gait periodicity and is not
a perfect zero-information null. These are descriptive diagnostics, not causal
skills, environmental adaptation proof or R1-transfer evidence. Layer-reset
interventions from the historical pipeline are not rerun by this launcher.

## Outputs and methods

`plan.json` records data/checkpoint/source hashes, split counts, targets and library
versions. `probes/<representation>/<true|shuffled_seed>/` contains fitted weights,
scalers, predictions, metrics and integrity receipts. Reports: `report.json`,
`report.md`, `probe_scores.csv`, `expert_comparison.csv`. No checkpoints overwritten.

Same methodological references as the existing studies:
[Alain & Bengio, linear probes](https://arxiv.org/abs/1610.01644), adapted here to
continuous ridge regression;
[Hewitt & Liang, control tasks](https://aclanthology.org/D19-1275/), motivating
controls but not claiming to reproduce their NLP control task;
[Kornblith et al., CKA](https://proceedings.mlr.press/v97/kornblith19a.html).

Tests: `python -m unittest consolidated_probing.test_pipeline`.
