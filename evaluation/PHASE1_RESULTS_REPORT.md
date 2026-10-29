# Phase 1: Nominal-domain evaluation report

## Scope

This report interprets the completed nominal FC Portugal evaluation before any
friction, noise, actuator, or joint-failure modifications. Each policy was run
for 3,000 active synchronous simulator steps (60 seconds) under five seeds: 101,
202, 303, 404, and 505. All policies used identical target generation, own-half
constraints, fall accounting, and reset handling.

The evaluated policies were the FC Portugal oracle, behavioural cloning (BC),
original DAgger, mixed-seed DAgger, and GAIL actors initialized from DAgger, BC,
or random parameters. The uniform-random action policy defines normalized score
zero and FC Portugal defines score one separately for every seed.

## Principal results

| Policy | Normalized return, mean [95% CI] | Expert-test Action MSE | On-policy oracle MSE | Wasserstein | Forward velocity (m/s) | Falls/min | Active oracle labels | Additional environment steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FC Portugal oracle | 1.000 [1.000, 1.000] | 0.0000 | 0.0000 | 0.0000 | 0.693 | 0.20 | 0 | 0 |
| BC | 1.101 [0.906, 1.296] | 0.0084 | 0.00437 | 0.0906 | 0.734 | 0.00 | 90,000 | 0 |
| DAgger | 1.140 [0.923, 1.356] | 0.2848 | 0.09217 | 0.1056 | 0.757 | 0.00 | 90,000 | 70,000 |
| DAgger mixed | 1.096 [0.902, 1.290] | 0.0612 | 0.00464 | 0.0808 | 0.731 | 0.00 | 170,000 | 70,000 |
| GAIL-DAgger | 1.106 [0.910, 1.302] | 0.2843 | 0.09880 | 0.1072 | 0.742 | 0.00 | 0 active | 40,000 |
| GAIL-BC | 1.104 [0.902, 1.307] | 0.0085 | 0.00444 | 0.0817 | 0.742 | 0.00 | 0 active | 40,000 |
| GAIL-random | -0.011 [-0.063, 0.040] | 7.4336 | 6.0120 | 0.8632 | -0.005 | 0.00 | 0 active | 40,000 |

The complete precision values and all other metrics are retained in
`evaluation/comparison.json` and `evaluation/comparison.csv`.

## Interpretation

### Nominal task performance

BC, both DAgger policies, GAIL-DAgger, and GAIL-BC all achieved mean normalized
return near or slightly above oracle parity. Their 95% intervals overlap one
another and 1.0 substantially. The results therefore do not establish that any
learned walking policy outperforms FC Portugal. They establish statistical
compatibility with oracle-level target progress in the nominal domain.

The learned-policy means above one partly reflect variation in the oracle
baseline: FC Portugal experienced one fall across its five evaluation minutes,
whereas these learned policies recorded none. With only five seeds, the return
intervals remain broad.

### Effect of the larger mixed DAgger seed

Mixed DAgger substantially improved imitation fidelity over original DAgger:

- Expert-test Action MSE fell from 0.2848 to 0.0612, a reduction of about 78.5%.
- On-policy oracle MSE fell from 0.09217 to 0.00464, a reduction of about 95.0%.
- Trajectory Wasserstein distance fell from 0.1056 to 0.0808.
- Falls remained zero and normalized return remained compatible with parity.

This supports the hypothesis that the original 20,000-sample, long-target-only
seed provided insufficient initial coverage of turning behavior. Alternating
long-range and short-turn oracle demonstrations materially improved the final
DAgger policy. However, this experiment changed both composition and size: mixed
DAgger used 170,000 oracle-labelled states versus 90,000 for original DAgger.
It therefore cannot yet attribute the gain solely to short-turn coverage.

Despite the improvement, mixed DAgger's offline Action MSE remained about 7.3
times BC's. Its on-policy MSE was only about 6% above BC's, suggesting that its
largest residual errors occur on held-out expert states rather than the states
the learned controller commonly visits.

### GAIL initialization dominates the outcome

GAIL largely retained the properties of its initialization within the 40,000
step training budget. GAIL-BC preserved BC-level Action MSE and stable walking;
GAIL-DAgger preserved original DAgger's much larger action error while retaining
task performance. GAIL did not erase the fidelity gap between its predecessor
policies.

Randomly initialized GAIL failed to bootstrap locomotion. Its normalized return
was statistically compatible with the random floor, forward velocity was
slightly negative, and Action MSE and occupancy distance were very large. Its
zero fall rate and low jerk represent a stationary robot, not successful balance
control. This is direct evidence that low fall and jerk values must always be
conditioned on locomotion performance.

### Efficiency and supervision

BC remains the strongest nominal-domain cost/performance result. It reached
oracle-level return and the best imitation fidelity using 90,000 static labels,
without on-policy simulator training. Mixed DAgger improved over original DAgger
but required 80,000 additional labels and retained the same 70,000 rollout-step
cost.

The table's GAIL label count is **marginal active-query cost**, not end-to-end
supervision. Every GAIL variant still used the 90,000-sample FC Portugal expert
dataset. GAIL-BC and GAIL-DAgger also inherit the costs of their pretrained
initializers. Future reporting should show marginal GAIL cost and total pipeline
cost separately to avoid implying that GAIL used no expert information.

## Phase 1 conclusions

1. BC is the best nominal-domain imitation policy by fidelity and simplicity.
2. Rich long/short seed coverage greatly improves DAgger, especially on-policy.
3. The current evidence does not show a task-return advantage for DAgger or GAIL.
4. GAIL's outcome is strongly determined by actor initialization.
5. Random-start GAIL converges to stationary failure under the shared budget.
6. Zero falls or low jerk cannot be interpreted independently of forward progress.

## Next experiment

Proceed to frozen-policy, zero-shot parametric domain evaluation. Begin with a
friction sweep while preserving seeds, active duration, targets, and metrics.
The primary comparison should emphasize BC, original and mixed DAgger,
GAIL-DAgger, and GAIL-BC. Retain GAIL-random as a failure-control policy rather
than treating it as a viable locomotion controller.
