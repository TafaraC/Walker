# Exploratory results, 29 October 2026

Read [REPORT.md](REPORT.md) for interpretations and limits; open
[EDA_FIGURES.pdf](EDA_FIGURES.pdf) for a portable 12-page graph atlas.
The main paper, slide deck, prior reports, checkpoints and evaluator are unchanged.

## Reproduce

From the repository root:

```bash
conda run --no-capture-output -n fcportugal python evaluation/exploratory_2026-10-29/analyze.py
```

Requires NumPy, pandas, SciPy and Matplotlib. Exact installed versions are recorded
in `manifest.json`. The script works from any current directory, resolves the
repository from its own location, and writes only into this analysis directory.
It does not start a server, run policies, train models, or mutate source artifacts.
It deliberately regenerates its own CSVs/figures/manifest; the interpretive report
is manually reviewed and is not rewritten by the script.

The source experiment artifacts under `experiments/` are required and are excluded
from Git. The shipped tables and figures remain readable without those artifacts.
`manifest.json` lists source paths and SHA-256 hashes, including every inspected
trace. The compatible experiment batch is intentionally asserted at 2,223 unique
runs: a different batch should receive a new dated analysis and revised review.

## Outputs and units

- `deployments.csv`: deduplicated long-form deployment metrics and controller identity.
- `coverage.csv`: suite and family coverage, checkpoint counts and active minutes.
- `checkpoint_summary.csv`: 30-round means, SD, quartiles, paired deficits, tails and falls.
- `family_summary.csv`: replica means and separation of within-model round SD from
  between-replica SD of means. A blank SD for a single checkpoint is intentional.
- `paired_effects.csv`: context, history, GAIL and optimization contrasts. Positive
  means left minus right; for MSE and falls, a positive difference is adverse.
- `variance_decomposition.csv`: balanced two-way descriptive partition; residual
  mixes interactions and simulator variation. It is not an ANOVA significance test.
- `variance_sensitivity.csv`: shared-round progress fraction separately by algorithm.
- `surface_groups.csv`: counts and progress in the five Gaussian-parent groups.
- `correlation_raw.csv`, `correlation_round_centered.csv`: descriptive Spearman matrices.
- `trace_summary.csv`, `trace_means_by_suite.csv`, `trace_means.csv`: error concentration
  and position-rate diagnostics. The latter combines sampled suites only. Legacy
  per-frame low-speed fractions are **not valid stopping-time estimates**; see report.
- `probe_target_comparisons.csv`: all existing student targets/layers against raw and
  matching random-feature controls, without fitting or changing probes.
- `resource_curves.csv`: retained checkpoint resource accounting from the controls report.
- `figures/`: individual PNG and vector PDF charts; `EDA_FIGURES.pdf` combines them.
- `manifest.json`: provenance, versions, bootstrap settings and figure captions.

Progress/deficits are metres over 60 active seconds. Velocity is m/s; direction
error is unit-vector distance, not radians. MSE is per-element 16-action squared
error. SWD is the existing standardized joint-occupancy metric. Jerk rate is
rad²/s⁶. Fractions in trace CSVs range 0–1; plotted percentages multiply by 100.
The 40 ms direction diagnostic only uses valid, defined-direction windows and
must not silently replace the historical tracking metric.

## Checks performed by the script

Duplicate round/checkpoint identities and common numeric metrics must agree;
all runs must have 60 active seconds; every sampled controller must have all 30
rounds; normalized returns are recomputed and checked against saved values;
the two-way sums of squares must add to the total; trace MSE/progress must agree
with summaries; each probe random control must contain three matching seeds.
Plots and report values are also reviewed manually. No policy tests are needed
because this analysis changes no policy, simulator or evaluation code.
