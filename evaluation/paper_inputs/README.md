# Paper figure inputs

Copies of the evaluation records that `../figures_2026-10-31_paper/make_paper_figures.py` reads, so that the six paper figures (and the counts behind them) can be regenerated from a fresh clone. The full experiment outputs, datasets and checkpoints are not in Git; see [REPOSITORY.md](../../REPOSITORY.md).

The files are unmodified copies; `SHA256SUMS` lists their checksums (`sha256sum -c SHA256SUMS`). Absolute paths inside the JSON files are provenance from the machine that produced them, not download links.

| File or folder | Used for | Original location |
|---|---|---|
| `submission_controls/curves.json` | Fig. 1 (DAgger learning curve) | `experiments/submission_controls/v1/curves/curves.json` |
| `submission_controls/comparison.json` | Figs. 4 and 6 | `experiments/submission_controls/v1/evaluation/summary/comparison.json` |
| `consolidated_scratch/comparison.json` | Figs. 4 and 6 | `experiments/consolidated/v1/scratch_v1/evaluation/all_all_correct/comparison.json` |
| `gail_history/before_changes/` | Fig. 5 (first GAIL implementation) | `experiments/archive_20260905_pre_variants/legacy_gail_backups/gail_backup_20260905-022910/` |
| `gail_history/after_dagger_init/` | Fig. 5 | `experiments/archive_20260905_pre_variants/gail_dagger/artifacts/` |
| `gail_history/after_bc_init/` | Fig. 5 | `artifacts/gail_bc/` |
| `gail_history/after_random_init/` | Fig. 5 | `artifacts/gail_random/` |

Figures 1–3 also read `../convergence_2026-10-30/deployments.csv` and `training_summary.csv`, which are tracked in place. GAIL folders contain only the per-iteration training-metric JSON files, not checkpoints.
