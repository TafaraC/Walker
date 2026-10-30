# Convergence follow-up analysis

- [Interpretation and numerical results](REPORT.md)
- [11-page figure atlas](EDA_FIGURES.pdf)
- `figures/`: PNG and vector PDF versions, suitable for slides/paper selection.
- CSVs: full deduplicated deployment data, checkpoint/family summaries, training
  resources, paired effects, variance/correlations, exact-weight repeats, all probe
  targets, shuffled controls, expert comparisons and initializer CKA.
- `manifest.json`: EDA sources/hashes, versions and interval settings.

From the repository root, using the existing fcportugal environment:

```bash
/home/pamnt/miniconda3/envs/fcportugal/bin/python evaluation/convergence_2026-10-30/probe.py audit
/home/pamnt/miniconda3/envs/fcportugal/bin/python evaluation/convergence_2026-10-30/probe.py run
/home/pamnt/miniconda3/envs/fcportugal/bin/python evaluation/convergence_2026-10-30/analyze.py
```

`probe.py report` verifies/reuses completed fits and rebuilds its report and paired
geometry. `run` skips verified fits; interrupted unreceipted fits must be archived
before retry, as in the existing probing pipeline. No simulator or GPU is needed.
Numerical fitting uses one CPU thread. Run one writer at a time.

The probe wrapper reuses the existing shared-oracle dataset, ridge fitting,
trajectory shuffles, expert extraction, CKA and scoring code. It audits all 982
new deployment receipts and all 29 new model receipts. Default raw probe output:
`experiments/representation_probing/convergence_all_v1/`. `--batch` and `--output`
can select a separate compatible completed batch/output; incompatible plans fail
rather than overwrite prior fits. This wrapper requires the full combined batch,
not a smoke or original-only run.

The complete probe plan has 56 checkpoints, 131 representations, 524 multitarget
fits and 15,720 target scores. Inputs are 10/5/15 complete oracle trajectories for
train/validation/test, stride 7 and horizon 10. Both pre/post hidden layers, raw
inputs with/without context, restricted phase/command inputs, native expert and
matched random networks are included. There is no slip prediction target.

The EDA merges the historical consolidated/scratch/control reports with the new
convergence report, checks duplicate measurements, and verifies 3,205 unique
deployments. It preserves all historical metrics, including the documented
localization/direction-error limitation. It does not select checkpoints or launch
policy training. Prior EDA is preserved in `../exploratory_2026-10-29/`.

Large raw data/checkpoints/probe fits remain ignored by Git. Reproduction requires
those local artifacts in addition to this repository snapshot. Published CSVs
contain summaries/scores rather than training tensors or fitted probe weights.

Validation completed: 11 existing convergence/probing unit tests passed; all524
new probe receipts and15,720target scores verified; all56checkpoint and30oracle
trace hashes remained unchanged. The5,640overlapping historical probe scores
reproduced within1e-9RMSE. The generated atlas has11pages and selected figures were
visually checked. EDA asserts coverage, duplicate consistency, equal horizons,
balanced panels and sum-of-squares decomposition; two epoch-zero comparisons check
every initial/final weight tensor for exact equality.
