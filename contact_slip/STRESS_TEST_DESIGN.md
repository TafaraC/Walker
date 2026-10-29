# Oracle large-slip diagnostic

Purpose: test whether much larger **scene settings** reach ODE and whether the
unchanged FC Portugal oracle's locomotion responds. This is a diagnostic, not a
validated range for randomisation or a friction-coefficient sweep. Do not interpret
a 100x configured slip as 100x actual foot sliding or lower Coulomb friction.

## Stage A: six-run screen

```bash
./run_slip_sensitivity.sh smoke --suite stress
```

| Scene slip | Multiple of nominal .1 | Role |
|---|---:|---|
| .1 | 1x | Same-batch nominal reference |
| .5 | 5x | Moderate extension |
| 1 | 10x | Order-of-magnitude extension |
| 2 | 20x | Intermediate stress |
| 5 | 50x | Extreme diagnostic |
| 10 | 100x | Extreme diagnostic |

Oracle only, task seed 101 at every setting, 3,000 active steps / 60 active seconds
per run, fresh owned server each time, standard startup/targets/fall recovery.
No pre-started simulator; ports 3100/3200 must be free. No interactive checkpoints.
The oracle is not retrained or modified. Contact auditing is passive: scene edits
and their existing two-argument setter syntax are deliberately preserved to test
the current configuration pathway. No effective-contact override or Gaussian draws.

All six settings run automatically unless infrastructure/data validation fails.
Falls are outcomes, not automatic abort criteria. Active time excludes recovery;
therefore a poor controller can take longer than 60 seconds of total simulated time.
One seed is a screen, not evidence of statistical significance. Seeds match target
generation, not necessarily event timing once trajectories diverge.

## Evidence and decision

Review `experiments/slip_sensitivity/oracle_stress_v1/smoke_report.md` and JSON.
Every run preserves scene checksums, server logs, solver slip values and enabled
flags, contact-class counts, evaluation metrics, trace and completion hashes.
Contact statistics include startup/recovery and ball contacts, not identified feet.

1. Same solver-slip/mode classes throughout: stop expansion; diagnose binding or
   surface combination. Changes in aggregate means/contact counts or progress are
   not proof of successful physics manipulation. Report emits a warning.
2. Changed enabled solver values, little locomotion change: manipulation now has
   supporting evidence, but inspect contact classes and confirm with more seeds.
3. Changed values and degraded walking: identify the onset using progress, falls,
   target counts, forward velocity and torso metrics; verify the oracle remains a
   viable teacher before adopting any range for training.
4. Native crash/nonfinite data: engineering failure, not a policy fall. Preserve
   failed attempt and diagnose before rerunning or changing bounds.

The class comparison is exact and descriptive; class changes alone could reflect
different body contacts, and do not by themselves prove a scene-parameter effect.
Body-specific instrumentation may still be needed. Do not infer adaptation from
one run or adjust the existing Gaussian training distribution automatically.

## Stage B: optional confirmation after review

```bash
./run_slip_sensitivity.sh evaluate --suite stress
```

Six settings x seeds 101,202,303,404,505 = 30 oracle runs. Same 3,000-step horizon;
report generated automatically. Uses separate `runs/` from Stage A so the smoke
is not silently pooled with confirmation. Reports paired progress differences
against .1 on matching seeds, total falls and solver-slip ranges; JSON preserves
all per-run evaluator metrics. No learned-policy evaluation or training is added.

Regenerate summaries without a simulator:

```bash
./run_slip_sensitivity.sh report-smoke --suite stress
./run_slip_sensitivity.sh report --suite stress
```

Prior `v1/` smoke and fixed/Gaussian experiments are preserved. Receipt hashes
remain strict; never relax them to resume old runs after code changes. New inputs
require a new `--output` directory. Seven offline tests pass; no stress simulations
have been launched by the assistant.
