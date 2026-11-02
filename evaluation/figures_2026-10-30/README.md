# Presentation figures for the extended-fitting batch (30 October 2026)

These are clean, consistently styled versions of the main convergence results, intended for reports and
presentations. The analysis itself is in [convergence_2026-10-30/REPORT.md](../convergence_2026-10-30/REPORT.md).
No new training, simulation or checkpoint selection was performed.

| Figure | Shows | Main reading |
|---|---|---|
| `01_original_allocation` | Held-out MSE against optimizer updates performed for BC and DAgger (original scene), plus sampled-surface progress relative to the oracle | The fidelity ranking flips twice depending on how much fitting each method gets. Mean progress relative to the oracle stays within ±0.3 m. |
| `02_varied_families` | Initializer → validation-stopped held-out MSE for all 8 varied families, and the paired progress change | BC's 4–11× fidelity deficit shrinks to 0.9–1.1×. The progress changes are small; only varied-only BC loses clearly (−0.27 and −0.13 m per 60 s). |
| `03_engine_control` | Progress and falls of the oracle, a trained policy, zero actions, three untrained networks and uniform random actions | The gait engine with untrained or zero inputs makes no useful progress. Low fall counts there mean inactivity, not stability. |
| `04_damping_check` (supplementary) | Whether losing the slight action damping under-fitted models show predicts the progress change | It does not: Spearman ρ = −0.19 (permutation p ≈ 0.39, 24 pairs). The initializers were already close to undamped, so the test has limited range. |

**Conventions.** Blue = BC, orange = DAgger, purple = controls, grey = oracle; the palette passes colour-blind separation checks.
Box = interquartile range, line = median, ◇ = mean, whiskers = 1.5 × IQR, ○ = outlying rounds. Progress is metres per 60 active
seconds on the 30 sampled surface/task rounds. Falls are counts over the stated active minutes.

**Noise floor.** Re-evaluating identical weights gives per-round progress SDs of 0.78–0.88 m (identical-weight repeats in the
convergence report; an earlier repeatability study gives a similar ~0.7 m). Family-mean progress changes smaller than about 0.2 m
should be read as noise.

## Reproduce

From the repository root (NumPy, pandas, Matplotlib):

```bash
python3 evaluation/figures_2026-10-30/make_figures.py
```

This reads `evaluation/convergence_2026-10-30/{deployments,training_summary}.csv`,
`experiments/submission_controls/v1/curves/curves.json` and `damping_pairs.csv` (in this folder).

`damping_pairs.csv` comes from the per-step traces, which are not in the Git snapshot. To rebuild it:

```bash
# 1. reduce traces of the extended models and their initializers (any folder of runs/<round>/<label>/trace.npz)
python3 evaluation/figures_2026-10-30/reduce_traces.py <trace_root> <reduced_dir>
# 2. regenerate the damping table and all figures
python3 evaluation/figures_2026-10-30/make_figures.py --damping-traces <reduced_dir>/trace_runs.csv
```

On 30 October the trace root held 1,580 runs: the 26 extended models and their exact initializers on 30 sampled and
5 original rounds. The reduced table is kept outside Git at `artifacts/convergence_damping/trace_runs.csv`.

Other outputs: `figure_numbers.json` (every number quoted above) and `varied_family_changes.csv` (per-family MSE,
progress change and falls).
