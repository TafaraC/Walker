# Optional wide-slip exploration (outside the paper)

The completed paper remains on its existing scope. This package implements
only the next empirical prerequisite: an oracle-only stress sweep using verified
scalar scene settings and passive ODE contact auditing. It does not train RL,
alter learned checkpoints, or select training bounds automatically.

## Commands

From Walker, with no simulator already running:

```bash
./run_optional_robustness.sh plan
./run_optional_robustness.sh smoke
./run_optional_robustness.sh evaluate
```

Plan is offline. Smoke runs all seven slips with the first seed (seven runs).
Full evaluation runs seven slips × three seeds = 21 runs, each 3,000 active
steps/60 active seconds. Smoke and full have separate receipts; the first seed
is deliberately repeated in full, not silently counted as independent evidence.
Each initiated command runs to completion without internal human checkpoints.
The user initiates live runs; the assistant has not run this batch.

Defaults: slip `0.1 1 5 10 20 50 100`; task seeds `101 202 303`.
These are search settings, not validated feasible terrain or a training range.
Reports are automatic, or regenerate without simulation:

```bash
./run_optional_robustness.sh report-smoke
./run_optional_robustness.sh report
```

For a refined grid, use a new output directory, include 0.1, and pass the same
options for every stage (saved plans reject inconsistent arguments):

```bash
./run_optional_robustness.sh plan --output experiments/optional_robustness/refined_v1 --slips 0.1 10 15 20 30 50
./run_optional_robustness.sh evaluate --output experiments/optional_robustness/refined_v1 --slips 0.1 10 15 20 30 50
```

## Reused architecture, data and safety

Reuses contact_slip.sensitivity's evaluator, native passive audit and owned
server, plus the existing shared simulator lock and scalar scene writer. A new
server is created for each deployment; the installed scene is not edited.
Existing servers are refused, not killed. A server-death watcher stops the client.
Completed receipts verify inputs/artifact hashes and skip on resume; incomplete
attempts use the existing begin/archive handling. On an error the command stops;
rerun the same command after checking the logs. There is no automatic endless retry.

Dependencies remain the FC Portugal Python environment, server installation,
native audit build toolchain and existing `expert_train_data.npz`/legacy model
files referenced by the shared environment manifest. The package does not
silently remove these inherited dependencies. Large per-contact CSV logs can
consume substantial storage; preserve them as physics evidence.

Default artifacts: `experiments/optional_robustness/oracle_wide_v1/` with
`plan.json`, `smoke/`, `runs/`, raw traces/results, server logs/manifests,
`contacts.csv`, `contact_audit.json`, `completed.json`, source reports and
`interpretation.json/md` (or `smoke_interpretation`). Plan pins optional source
hashes; existing evaluator receipts pin their source/runtime/environment inputs.
The passive hook observes solver slip; it does not override contact dynamics.

## Read the outcome before extending scope

Inspect absolute progress, velocity, targets, direction error and falls/min,
alongside paired progress retention relative to scalar 0.1. Retention is
undefined when reference progress is nonpositive/near zero. Do not use
oracle-normalized return as the primary metric when the oracle itself fails.
No pass/fail threshold is imposed: upright inactivity can be functional failure,
but an arbitrary target-progress cutoff should not masquerade as a physical law.

Audit enabled solver slip classes, not just configured scene text or contact
counts. Reports flag identical recorded classes, but differing classes alone
do not identify feet, sliding speed or prove every contact was modified.
The existing audit includes ball, startup and recovery contacts. Expert failure
does not establish that another policy can solve the condition; very high slip
may make the task infeasible within the shared gait interface.

## Later stages — design only, not implemented here

1. Use measured degradation to refine the sweep and freeze viable/transition/
   failure conditions. No Gaussian or training bounds are chosen in advance.
2. Evaluate selected existing frozen controllers on those same conditions in a
   separately versioned schedule. Do not edit the completed paper evaluations.
3. Compare original-to-wide IL with extended varied-only BC/mixed DAgger, matching
   within-algorithm optimization, data and query budgets explicitly. Distinguish
   more epochs from more demonstrations; these test different explanations.
4. Pilot hidden-context and input-conditioned PPO with task reward, unchanged
   oracle as evaluation reference, and the existing 16-action gait/IK interface.
   Validate reset-excluded progress, fall penalties, termination masks and reward
   accounting before training all seeds. Random RL and IL-initialized RL are
   distinct conditions. Expert-action rewards/anchors would be expert-assisted RL.
5. Freeze reward/range/budget choices before held-out confirmation, retain all
   tuning provenance, and report absolute outcomes where the expert fails.

These stages answer a separate curiosity-driven question, not a missing
requirement for the eight-page imitation paper. The CLI intentionally exposes
no training command until the stress results support choosing its environment.
