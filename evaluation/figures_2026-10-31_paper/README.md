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

By default the script reads tracked files only, so it runs on a fresh clone:

- [`evaluation/paper_inputs/`](../paper_inputs/README.md): copies of the evaluation records it needs (about 6 MB, with SHA-256 checksums).
- `evaluation/convergence_2026-10-30/deployments.csv` and `training_summary.csv`.

With `--from-experiments` it reads the same records from their original, Git-ignored locations under `experiments/` and `artifacts/` (override the latter with `--legacy <folder>`). Both modes give identical figures and numbers.
