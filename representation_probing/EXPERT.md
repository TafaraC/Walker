# Appended expert comparison and broad representation structure

Run from Walker:

```bash
./run_expert_probing.sh audit
./run_expert_probing.sh run
./run_expert_probing.sh report
```

Default parent: experiments/representation_probing/nao_sources_v1.
Default output: experiments/representation_probing/nao_expert_comparison_v1.
`--baseline` and `--output` select other directories; use the same flags for report.
No simulator/GPU, policy training, R1 jobs or alterations to originals. The user
launches the actual research fits; audit is read-only. Existing64 fits and36 reset
diagnostics are verified/reused, not retrained. Completed parent sources stay
unchanged, so the original launcher/report remains valid.

## Which expert is now probed?

The robot-type0 FC Portugal Walk oracle at behaviors/custom/Walk/walk_R0.pkl:
63 inputs ->64 tanh units ->16 linear gait outputs. It has **one** hidden layer.
The four students each have two256-unit ReLU layers. The loader uses the trusted
repository pickle, verifies its bias/kernel shapes/finiteness, and reproduces the
native math_ops.Neural_Network.run_mlp equations without importing a simulator.
Audit checks all90000 recorded oracle actions from the30 source trajectories;
rtol1e-4/atol1e-5 allow float32 batched-matmul differences. Observed max absolute
error in implementation audit:6.56e-6. Any action mismatch blocks the pipeline.
Model SHA-256 is recorded; a different pickle cannot silently become the teacher.

This probes the expert neural network, not all of the external stateful step
generator, smoothing and IK that also contribute to the complete walking oracle.
Their state may be represented in the observed inputs, but is not a neural layer.

## Same property study, now anchored to the teacher

Keep all30 gait/support/body/limb/command/interaction/future labels, stride7,
the same4270/2133/6405 rows, whole-run splits, train-only normalization,
validation-selected ridge grid and shuffle seeds. No slip label or narrowing
to properties on which imitation is already good. Details: README.md.

Add expert_h1 and three random63->64 tanh feature controls (seeds42/43/44).
The random controls use PyTorch Linear's standard initialization, **not** a claimed
reproduction of the original expert RL initialization. Existing random256-ReLU
controls remain present. Four representations x four label conditions =16 new
multitarget fits /480 scalar scores. Combined reports contain2400 scores.

For every property compare **both** student layers with expert h1, not an assumed
one-to-one depth mapping. Save expert/student R2, R2 difference, RMSE ratio to
expert and macro-run RMSE difference. An RMSE ratio near1 indicates comparable
decoding error for that target under this protocol; it is not "100% knowledge
transferred." Ratios are undefined if expert error is effectively zero, and R2
differences are null for undefined R2. Better student decoding can arise from
width, activation, training or reorganization; worse decoding need not mean the
property was lost from every nonlinear representation. Do not select the best
student layer after inspecting test results and call it a confirmatory score.

Compare the trained model to its own architecture-matched random controls and
raw-input/phase-command controls before calling a property learned. The expert's
smaller tanh architecture differs from the students, so probe capacities and
conditioning differ even though the ridge protocol is the same. No architectural
or causal transfer conclusion follows from a raw expert/student score gap.

## Broad structure beyond the selected labels

On the same held-out states, compute centered **linear CKA** between expert h1
and each existing representation (all student layers and controls), plus expert
random controls and expert self-check. Report pooled CKA and separate within-run
CKA. This compares sample-wise feature geometry with different feature widths,
without neuron matching or an invented expert h2. Per-run means are descriptive,
not significance estimates. No activation standardization is applied: centering
only; CKA is insensitive to uniform scale but not arbitrary feature rescaling.

A seed2026+trajectory-index within-run row permutation provides a correspondence
control. It breaks sample alignment but does not establish a calibrated null
distribution; temporal dependence and high dimensionality still matter. A single
permutation is a diagnostic, not a p-value. CKA self-similarity should be1 for
nonconstant activations; zero-variance cases return null.

Also save centered covariance eigenvalue spectra, participation ratio
(sum eigenvalues)^2/sum eigenvalues^2, width and fraction of near-constant units
(std<1e-8), for expert, students and controls. Participation ratio is a descriptive
effective dimensionality under the recorded state distribution, not a count of
semantic skills. Compare absolute values alongside width; tanh/ReLU scales differ.
These label-free diagnostics preserve the broader internal-representation direction,
but **no finite probe set or CKA analysis finds all internal representations**.

## Artifacts and interpretation

The new plan hashes parent plan and65 completion receipts, existing sources,
expert/source decoder, new module, model and libraries. Strict parent validation
prevents stale/copied aggregate reports being used as evidence. New probe folders
save weights/scalers, row-aligned test predictions/labels, metrics and receipts.
geometry/ contains hashed CKA/spectral diagnostics. Old36 reset results remain
in the verified parent; no new expert resets or deployments are performed.

Outputs: report.md (property gaps plus structure table), report.json (all controls,
per-run CKA/permutations/spectra), expert_comparison.csv and probe_scores.csv.
Original reports remain unchanged. Completed new subprobes resume after hashes;
changed inputs require new --output. Archive incomplete unreceipted subfolders
before retry, or choose a new output. One writer per output directory.

Use property gaps to describe retained **linear accessibility** and CKA to describe
shared **geometry**. Agreement across them is useful evidence of resemblance to
the teacher, not historical proof that information was causally copied from it.
Students can reorganize features or learn equivalent computations. All comparisons
remain on shared oracle states; on-policy/R1 states may differ substantially.
Periodic shuffle weaknesses, previously inspected test data, source training-budget
differences and lower-alpha boundary caveats from README.md still apply.

To test R1 utility, compare intact and disrupted transferred sources against
matched scratch/frozen-random adapters under fixed R1 budgets. Do not equate
teacher resemblance with transfer success. No R1 architecture assumptions or
files are changed by this extension.

## References

Property probes retain [Alain & Bengio's frozen-layer linear-probe approach](https://arxiv.org/abs/1610.01644),
adapted to regression, and [Hewitt & Liang's control-task motivation](https://aclanthology.org/D19-1275/).

Label-free geometry follows [Kornblith, Norouzi, Lee & Hinton (2019), Similarity
of Neural Network Representations Revisited, ICML/PMLR97:3519–3529](https://proceedings.mlr.press/v97/kornblith19a.html).
Their linear CKA compares representational similarity across networks, including
different widths. Here it is used descriptively on shared locomotion states,
not as a causal transfer score. Per-run controls and spectral summaries are
project-specific diagnostics, not claimed reproductions of the paper's experiments.

Tests: python -m unittest representation_probing.test_expert
representation_probing.test_pipeline probing.test_pipeline (one shell command).
