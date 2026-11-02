# Paper figures

`make_paper_figures.py` draws the six figures used in the paper, at IEEE print size (Figures 1, 2: full text width; Figures 3–6: one column).

| Figure | Content |
|---|---|
| 1 | Held-out action error against optimizer updates, and progress relative to the expert (original field) |
| 2 | Varied-surface families: held-out error before and after validation-stopped fitting, and paired progress change |
| 3 | Progress and total falls of the expert, a trained policy and the engine controls |
| 4 | Share of runs with at least one fall, per cohort |
| 5 | GAIL training diagnostics before and after the implementation changes |
| 6 | Learner progress gap and falls by slip group |

Figures 1–3 use the same calculations as `../figures_2026-10-30/make_figures.py`. Figures 4–6 reuse the data loading and calculations of the earlier exploratory analysis scripts; the script's docstring records where each comes from.

## Run

From the repository root (NumPy, pandas, Matplotlib ≥ 3.10):

```bash
python3 evaluation/figures_2026-10-31_paper/make_paper_figures.py --out <folder>
```

It writes `fig1_allocation` … `fig6_surfaces` as PDF and PNG, plus `figure_numbers.json` with the counts behind Figures 1, 3, 4 and 6.

## Inputs

- Tracked: `evaluation/convergence_2026-10-30/deployments.csv` and `training_summary.csv`.
- Not in Git (large research artifacts; see [REPOSITORY.md](../../REPOSITORY.md)):
  `experiments/submission_controls/v1/curves/curves.json`,
  `experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json`,
  `experiments/submission_controls/v1/evaluation/summary/comparison.json`,
  the GAIL histories in `experiments/archive_20260905_pre_variants/`, and
  `artifacts/gail_bc`, `artifacts/gail_random` (override with `--legacy <folder>`).
