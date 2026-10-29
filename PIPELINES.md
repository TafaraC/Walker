# Imitation-Learning Pipelines

## Submission controls — 2026-10-28

`./run_submission_controls.sh plan|audit|train|evaluate|zero|curves|report|run`
adds isolated controls under `experiments/submission_controls/v1/`.
Technical/command reference: [submission_controls/README.md](submission_controls/README.md).
Historical consolidated/scratch Python sources, models and results are unchanged.

Default training: original BC from its saved random initialization for exactly
211,050 Adam updates (317 full epochs plus245minibatches, batch256), and varied-only
BC for200total epochs, hidden/input × seeds42/43/44. Reuses170k expert datasets,
separate validation/test pools, lr1e-4, original architecture/context contracts.
There are seven new BC fits; all start from saved initial weights, not selected
100-epoch weights. `--architectures hidden` reduces varied training to three fits.
No additional oracle demonstrations/learner rollouts are collected for training.

Validation-only best checkpoint and synchronized optimizer are saved separately
from the actual budget endpoint. `history.json` records each epoch/partial endpoint;
`metrics.json` distinguishes performed budgets and selected checkpoint costs.
Original matched BC uses53,952,720example exposures, not exactly54million.
Optional `--diagnostics` uses separate diagnostics_v1 output and includes the
standard fits plus early-stopped original BC and a fixed-final-aggregate DAgger
continuation. Patience25, absolute min_delta1e-5, cap500epochs; minimum100forBC,
one additional epoch forDAgger. DAgger inherited costs remain explicit. These are
validation stopping diagnostics, not a claim of algorithm convergence.

Zero baseline: an all-zero-output native16-D BC-shaped network, no dataset action
denormalization. The unchanged evaluator retains gait generation, filtering, IK,
commands and fall handling; this is not zero joint commands. `zero` can evaluate
it independently under evaluation_zero; the full run uses evaluation instead.

Full evaluation reuses the saved38round schedule,3,000active20ms steps per run:
38original matched BC +198varied BC +38zero =274new deployments. Reference oracle,
random, original BC/DAgger and matched varied continuation/scratch BC records
are hash-verified and reused. Shared managed-server lock, startup scene import
verification, retries and installed-scene restoration are inherited. No manual
server or in-script human checkpoints. `--quick` uses isolated smoke_v1, two tiny
fits, two rounds and nine32-step deployments with fresh horizon-matched baselines.

Reports place latent-action MSE beside joint SWD, progress/normalization, velocity,
tracking, falls/exposure, jerk and torso RMS; original/sampled/nominal remain
separate. Reports live in evaluation/summary/comparison.{md,csv,json}. `curves`
also has a separate entry point; paired_differences.csv records same-round
new-minus-reference deltas. The curves stage
scores original BC, DAgger0–7 and GAIL0–7 saved checkpoints on fixed validation/test
pools, producing CSV/JSON and fidelity_resources.png with explicit inherited
resource columns. No test selection; these are fidelity curves, not return curves.

Receipts enforce identical plan, sources and input hashes; completed jobs skip.
Incomplete attempts are archived and restarted at job boundaries, not resumed
mid-epoch. Offline stages do not own a simulator. CUDA is used for training when
available; CPU fallback is supported. Live smoke/research execution is user-run.

## Exploratory GAIL improvement pipeline — 2026-10-19

`./run_gail_improvement.sh plan|audit|train|evaluate|run|report` is a separate
experiment under `experiments/gail_improvement/original_v1`. User confirmed
original environment first. Defaults: random/BC/mixed-DAgger initialization,
seeds42/43/44, 350k rollout steps each (nine actors,3.15m total),35updates.
Reuses consolidated source data, fall/GAE/action contracts and isolated servers.
No completed paper source module/checkpoint is modified. `run --quick` routes
to smoke_v1: three actors,200steps each,100-step evaluations, one seed per split.

Adaptive trainer has no BC anchor, noise-matched expert actions, zero-centered
discriminator gradient penalty, positive clipped occupancy reward, adaptive actor
LR and exact-Gaussian-KL guarded epoch retries. The optional `--recipes legacy
adaptive` doubles the run count for a same-budget trainer comparison. These
changes are experimental, not guaranteed gains; original expert occupancy remains
the source, not the DAgger labelled aggregate. See gail_experiments/README.md for
all settings, clipping/normalization/data contracts and attribution caveats.

Initializer and70k-spaced checkpoints get validation seeds901–903; initializer,
validation-selected and final get separate confirmation seeds1901–1905, plus
fresh oracle/random references. Reports show paired progress/falls, held-out and
on-policy MSE, SWD/jerk and learning curves. Selection retains initializer if
later checkpoints do not qualify; no automatic replacement of existing models.
Artifacts include immutable plans, runtime/job receipts, model/discriminator/
optimizer states, per-update diagnostics, selection.json and comparison.json/csv/md.
Audit is offline; run/train/evaluate own the server and finish without prompts.

## Optional oracle wide-slip sweep — 2026-10-18

New isolated `optional_robustness/` and `./run_optional_robustness.sh` reuse the
passive contact_slip audit/evaluator and owned-server lifecycle, without editing
completed training/probing sources. Commands: plan, smoke, evaluate, report,
report-smoke. Defaults scalar slips .1/1/5/10/20/50/100, three task seeds
101/202/303, 3000 active steps. Smoke is seven one-seed deployments; full is
21 deployments in separate run directories. No internal prompts or auto-training.
Output experiments/optional_robustness/oracle_wide_v1; per-run scene/contact
evidence, traces, receipts and absolute/paired-retention reports. Hash-verified
resume skips complete runs; changed plans/sources reject reuse. See
optional_robustness/README.md for dependencies, commands and future design.
No live sweep launched; offline plan/metric and inherited audit tests pass.

Paper candidate draft: evaluation/FINAL_PAPER_DRAFT.md. It keeps this optional
exploration outside the completed 56-policy paper and labels convergence claims
as limited to budgeted resource comparison without new checkpoint curves.

## Completed scratch-report integration — 2026-10-16

The read-only consolidated report builder now imports
`evaluation/analysis/scratch_summary.py` and appends Appendix F from saved
scratch evaluation/training/probe JSONs. It asserts 1,949 combined runs,
15,720 combined probe scores and 28 new policies; no simulation or fitting occurs.
Parent analysis and paper references remain; opening text distinguishes parent
and combined counts. Previous full report is under evaluation/report_archive/.
Training/evaluation/probe implementation and fingerprinted sources are unchanged.

## Consolidated report references — 2026-10-16

`evaluation/analysis/consolidated_summary.py` now includes the maintained
`evaluation/analysis/consolidated_references.md` as Appendix E in its Markdown
output. This provides a method-to-paper citation map and full linked references
without changing training, evaluation, probing or source-artifact contracts.
The report remains `evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md`.

## Consolidated matched-sample study — 2026-10-15

New entry points: `./run_training.sh plan|run|report` and
`./run_evaluation_all.sh plan|run|report`. Full commands, contracts, artifacts,
lineage and caveats: [consolidated/README.md](consolidated/README.md). Defaults to
`experiments/consolidated/v1`; no old checkpoints/launchers replaced. Both live
pipelines own fresh servers, share the existing lock, archive failures and use
hash-verified resume. Do not manually start SimSpark.

BC has 170k training examples (validation/test extra 10k each); mixed DAgger has
100k balanced seed +70k learner-state oracle labels, seven 10k rollouts and beta
.8/.6/.4/.2/0/0/0. Per user choice, each GAIL has 70k PPO steps, 170k shared static
expert examples and inherited initializer cost. Counts match selected resource
axes, not gradient updates or total end-to-end cost. BC100/DAgger50 epochs,
batch256/Adam1e-4; GAIL reuses variable_training.gail_update defaults with segmented
GAE,3 PPO epochs/1 discriminator epoch. Per-stage updates/exposures and unique
collection totals are reported separately.

Four original policies (BC,mixed DAgger,GAIL-BC,GAIL-mixed-DAgger), seed42; original
BC/DAgger start from seeded scratch. Varied hidden/input policies use three seeds
42/43/44 (24 models), corresponding NEW original BC/DAgger parents, and GAIL actors
initialized from their same-domain/context/seed final BC/mixed DAgger. FiLM excluded.
63/64 observations,256/256 ReLU,16 latent outputs; GAIL actor/critic/discriminator
all include encoded context for input architecture. Existing PPO/metrics/context
encoding reused; isolated worker adapters handle DAgger/GAIL context transitions.

Original scene bytes are preserved; corrected varied surfaces use scalar setters
at .025/.1/.2/1/10. Evaluation original=5seeds×6controllers=30; latest33 context
rounds×(2baselines+4frozen original+24varied)=990; default total1020×3000active
steps. Correct context default; optional constant/wrong modes add runs. Reports
include same-round random/oracle normalization, matched evaluation-domain held-out
latent MSE (correct supplied context), visited-state MSE, velocity/tracking/falls,
torso RMS, sliced Wasserstein, angular jerk, per-replica CIs, corrected-nominal
retention and paired context deltas. New comparison schema has its own report,
not the legacy plotter contract. Synthetic offline tests pass; live smoke remains
user-run. Research commands are never launched automatically by an agent.

For a task-oriented command reference, server ownership requirements and the
interactive RoboViz demo menu, see [COMMANDS.md](COMMANDS.md).

## Expert-anchored append-only representations (2026-10-10)

`./run_expert_probing.sh audit|run|report`, isolated representation_probing/expert.py.
Full contract/references: representation_probing/EXPERT.md. Reuses verified64
general probes and36 reset diagnostics from nao_sources_v1 without changing their
sources/results. Default output experiments/representation_probing/nao_expert_comparison_v1;
flags --baseline/--output. No simulator, R1 change or source-policy training.

Expert walk_R0.pkl is63->64 tanh->16 linear; audit architecture and all90000
recorded oracle actions (rtol1e-4/atol1e-5, observed max abs6.56e-6). One expert
hidden layer, compared to both256-ReLU student layers. Same30 targets and exact
rows/splits/readout settings; three64-tanh random controls.16 new multitarget fits,
480 new scalar scores,2400 combined with reused controls. Per-target expert/student
R2 gaps, RMSE ratios and macro-run error gaps, not percentages of knowledge.

Broad structure: unscaled centered linear CKA pooled and per-run, within-run
row-permutation controls (seed2026+run index), covariance spectra/participation
ratios/near-constant-unit fractions. No neuron matching or exhaustive feature claim.
New hashed plan/receipts/probe weights/predictions, geometry/metrics.json;
report.md/json, expert_comparison.csv, probe_scores.csv. Strict resume and parent
source/model/data checks; nonempty incomplete outputs preserved/rejected. Original
expert/student weights unchanged; no expert layer resets. User runs research fits.

## General source representations for R1 hypotheses (2026-10-10)

New isolated `representation_probing/`, `run_representation_probing.sh`:
`audit`, `run`, `report`. Full contracts/references in representation_probing/README.md.
This replaces slip-centred probing as the main research direction while preserving
all earlier probing code/results. No simulator, GPU, policy training or R1 changes.
Root BC/mixed DAgger/original-DAgger GAIL/GAIL-BC only, shared audited oracle states.

Thirty labels: cyclic generator phase, foot-force/support-threshold proxies,
body/gyro state, hip-relative ankles, displacement/orientation commands,
phase-posture interactions and .2s future oracle motion/posture/support proxies.
No slip label; not true balance margin/contact ground truth. Defaults stride7,
horizon10, whole-run10/5/15 split,4270/2133/6405 rows.16 representations (raw,
phase+command,8 trained hidden layers,6 random layers) x4 label modes =64 fits.
Train-only scaled ridge penalties1e-6..100 chosen by validation macro-run MSE.
Three block-shuffle controls; random seeds42/43/44; mean and future-persistence
baselines. Test RMSE/R2/macro-run RMSE, threshold contact accuracy/balanced accuracy
and positive fraction; no calibrated probability claim or independent-row CI.

36 in-memory single-layer reset diagnostics measure latent-action RMS change on
shared test states: four policies x3 affine layers x3 seeds. Originals stay frozen
and unchanged; no corrupted checkpoints exported or deployed. This is source
functional sensitivity, NOT R1 learning benefit or property-specific causality.

Output experiments/representation_probing/nao_sources_v1: immutable plan with
source/data/checkpoint hashes and target schema; per-probe scalers/weights,
predictions with run/step indices, metrics/receipts; intervention metrics/receipt;
report.md/csv/json (JSON includes persistence/reset diagnostics). Strict resume,
new output for changed inputs. Preserve/archive incomplete unreceipted subprobe
before retry. Only one writer per output. Flags --data/--output/--stride.

## Offline frozen-policy probing (2026-10-10)

Implemented `probing/run.py` and `run_probing.sh`; full technical contract and
paper references are in `probing/README.md`. Commands: `./run_probing.sh audit`,
`./run_probing.sh run`, `./run_probing.sh report`. CPU-only readouts, no simulator,
policy training, adjacent R1 changes or new rollouts. User launches research fits.

Only root BC, mixed DAgger, original-DAgger-initialized GAIL and GAIL-BC checkpoints
are loaded, frozen; GAIL actor mean only. All receive common63-D oracle states.
Features: raw63, two256-D post-ReLU layers per actor, two layers per three random
networks. Targets: log1p(field slip), current head height/roll/pitch and next0.2s
mean oracle forward velocity. Stride10; exclude windows crossing recovery or the
unobserved final endpoint. Existing corrected-scalar oracle receipts/traces from
bc_input_v1/evaluation_all_v1 are audited; grouped draws0/1 train(10 runs), draw2
validation(5), confirmation test(15). Default eligible rows2990/1493/4485.

Train-only scaling; ridge mean-loss penalties .01/.1/1/10/100 selected per target
by validation macro-run MSE. Three independent within-split block-label shuffles
and random-network seeds42/43/44; mean-target baseline. Fifteen representations
times four label conditions =60 multitarget fits. Test RMSE/MAE/R2/macro-run RMSE,
not formal significance or causal context use. Report counts are not policy seeds.

Default experiments/probing/original_policies_v1 contains immutable source/data/
model-hashed plan, per-representation/label-mode readout weights/scalers, test
predictions with run/step indices, metrics and completion receipts, then report
MD/CSV/JSON. Strict resume; changed inputs require new output. Interrupted nonempty
unreceipted subprobe is preserved and rejected; archive specifically before retry
or choose new output. `--data`, `--output`, `--stride`; same flags for report.
Do not run concurrent writers to the same output. Existing pipelines unchanged.

## Repeatability diagnostics and append-only FiLM (2026-10-10)

Repeatability: `./run_bc_repeatability.sh evaluate` (optional standalone study).
Default45 deployments:5 identical repeats×3 surfaces×(oracle+hidden42+input42).
Same prior task/slip pairs, fresh server per run, unchanged startup protocol.
`plan` is offline; `report` regenerates repeatability.md/json. Raw first-active
state/action/joint/target traces support startup diagnostics; summaries include
progress SD/range, pairwise variation and initial-state differences. Default output
bc_input_v1/repeatability_v1. Shared lock, one retry and strict resume as before.
This measures variability, not an implemented simulator reset fix.

FiLM: `./run_film_training.sh audit`, then user-run
`./run_film_training.sh train --device cuda` to match completed controls. Defaults
train only FiLM seeds42/43/44 from original BC weights, same100k/10k/10k data,
100epochs/batch256/Adam1e-4. Separate package/output bc_film_v1 keeps original
source fingerprints/checkpoints intact. Last256 hidden features receive affine
modulation from one encoded slip scalar;1→512 linear branch outputs delta-gamma
and beta, initialized zero. Adds1024 parameters. Original63-input backbone and
16 latent outputs unchanged; packed64 transport is not64-input backbone FiLM.
Metadata schema film_bc_v1 records layout; validation-best checkpoints/test
fidelity and resource counters retain the BC training contract.

Per user clarification, FiLM evaluation runs **only FiLM**, never repeats controls:
`./run_film_evaluation.sh evaluate --suite smoke` (3 deployments), then
`./run_film_evaluation.sh evaluate` (99 new deployments:33 saved rounds×3 models).
Reads exact slips/task seeds from completed evaluation_all_v1 plan. Reuses verified
oracle/hidden/input-correct raw results in a separate combined report. Optional
`--modes correct constant wrong` yields297 new FiLM runs, not extra control runs;
use new --output when changing modes. Reports comparison.md/json include provenance
flags and paired deltas. Prior confirmation rounds are not untouched FiLM tests.
`plan`/`report` are offline. Data/initializer/budgets/seeds/device/library matching,
old physics/source compatibility, receipt hashes and scene verification are strict.
Prior controls used CUDA: CPU training is supported generically but will not pass
this matched-append check. Live smoke remains user-run; no experiments launched.
Details and references: [FiLM guide](film_training/README.md),
[repeatability guide](context_training/README.md#follow-ups-repeatability-and-appended-film).
Verification:38 offline tests passed, including synthetic FiLM updates/checkpoint
loading, identity initialization, repeat diagnostics, append-only scheduling,
comparison/resume and legacy regressions. Real oracle data audit passed and the
prior429-run evaluation identity remains unchanged. No live jobs started.

## Context BC seed semantics and replication scope

Detailed explanation: [What the training seeds mean](context_training/README.md#what-the-training-seeds-mean).
Seeds42/43/44 produce three checkpoints per architecture, not three techniques
or an ensemble. They mainly change shuffled minibatch order. Original pretrained
weights, demonstrations, whole-run splits, slip values, context encoding and
training budgets are shared; the extra context weights always start at zero.
Hidden/input models with matching seeds use identical batch indices via a
dedicated DataLoader generator. This measures fine-tuning variability conditional
on a common initializer/data, not independent training from scratch. CUDA seeds
alone do not guarantee bitwise determinism.

Evaluation schedule draws and task seeds are separate from these training seeds.
Each round pairs physical slip and task seed across checkpoints; target sampling
timing may diverge with falls or target arrivals. Default grouped comparison:
15×(1 oracle+3 hidden+3 input)=105 runs. Full suite with three supplied-context
modes: 33×(1+3+3×3)=429. A proposed one-pair comparison would be45 grouped runs,
but no evaluator training-seed selector is implemented; defaults are unchanged.

## Context BC live evaluation pipeline (2026-10-09)

Implemented `context_training.evaluate` (owned-server orchestration/report) and
`context_training.eval_worker` (one client), launched by `run_context_evaluation.sh`.
No legacy evaluator/training sources changed. The completed six BC checkpoints
remain valid. [Operational guide](context_training/README.md#live-bc-evaluation).

Commands: `./run_context_evaluation.sh evaluate --suite smoke` first (7 deployments),
then `./run_context_evaluation.sh evaluate --suite all --modes correct constant wrong`
(429 deployments). Default suite grouped/correct is 105 deployments. Each has
3,000 active steps. Stage `plan` saves an offline plan; `report` regenerates reports
offline with matching suite/modes. Evaluate automatically reports on completion.
No prestarted server or internal checkpoints; user starts the command once.

All comprises 15 historical grouped paired draws, 15 separate confirmation draws,
and 3 nominal .1 rounds. Three training seeds per architecture. Oracle runs once
per round. Input modes: correct actual scene slip; constant .1; wrong 10 below
physical slip1, otherwise .025. Physical slip never changes because of an ablation.
Each deployment gets an isolated verified scalar-slip server under the shared lock.
One retry by default for transient failures; archive incomplete attempts, strictly
verify completed artifacts on resume, never kill an unrelated server.

Artifact root: <training-root>/evaluation_<suite>_v1; immutable plan, per-round and
controller job/scene/log/manifest/result/trace/receipt, report.md/csv/json. Supplied
context and physical slip recorded separately. Reports retain training seed and
suite and provide paired input-hidden and ablation-correct progress/fall deltas.
Metrics: progress, oracle ratio, falls/minute, targets, speeds, tracking, on-policy
latent MSE, 5-sample-smoothed angular jerk, 256-projection standardized joint-space
sliced Wasserstein against paired oracle. No random normalization, exact Wasserstein
or torque claims. Final-action falls are checked; reset intervals excluded from
active steps and jerk. Existing training held-out MSE remains a separate metric.
No live jobs launched during development; user smoke validation remains required.
Thirty-three offline tests passed, including mocked worker, retry/resume and
report validation; all six real checkpoints passed offline schedule construction.

## Input-conditioned BC pilot: offline milestone (2026-10-09)

New isolated `context_training/` package and `run_context_training.sh`; existing
collectors, evaluators, training modules and historical source fingerprints are
unchanged. Technical contracts/commands: [context_training/README.md](context_training/README.md).
`./run_context_training.sh audit` validates existing varied-arm oracle data without
training or experiment writes. `./run_context_training.sh train` performs matched
hidden/input BC training, seeds 42/43/44, 100 epochs, batch 256, Adam 1e-4, CPU;
`--device cuda` is explicit and immutable within the chosen output experiment.

Inputs: verified whole-run splits under
`experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/` (100k/10k/10k).
Slip comes from hashed, server-import-verified scalar scenes, not guessed from
trajectory filenames. Original 63-D observations remain unchanged on disk.
`log1p(slip)/log(11)` is appended for the input arm (64-D), accepting only (0,10].
Both arms output 16 latent actions and warm-start the original `bc_walk_policy.pth`.
The new input column is zero-initialized; copied backbone parameters remain
trainable. Matching hidden control gets identical data, batches and update budget.

Artifacts live in `experiments/context_conditioning/bc_input_v1/`: immutable
`plan.json`, per-architecture/seed `policy.pt`, `optimizer.pt`, `metrics.json`,
hashed `completed.json`, and aggregate `summary.json`. The new checkpoint bundles
architecture/context metadata and state; it is not a legacy bare state dict.
Validation selects the best epoch; test reports latent element MSE, squared
L2/state and per-slip MSE. Metrics include parameters, updates, exposures, inherited
demonstration counts and zero new oracle queries/simulation steps. Resume verifies
identities/artifacts, archives incomplete attempts and restarts the affected job.

This milestone has no live-evaluation command, FiLM, or conditioned DAgger/GAIL.
Do not load these checkpoints with the legacy evaluator. Next: an isolated owned-
server evaluator with physical versus supplied context tracked separately. No
research training or live jobs were launched during implementation.
Verification: seven context regression tests plus twenty existing variable-training
tests passed; real-data audit, launcher syntax and Python compilation passed.

## Grouped evaluation rounding fix (2026-10-09)

Scene writer remains12 significant digits. status() now accepts either the prior
1e-12 numerical check or exact equality to float(format(expected,'.12g')), matching
the writer's encoding. This fixes slip5.01334232429452 being written5.01334232429
(error4.52e-12) and rejected before server startup. No scene/solver values change.
Evaluation identity canonicalizes ONLY the two audited pre/post validation source
hashes of configure_field_slip.py; arbitrary source changes, model/plan/row changes
and corrupted artifacts remain rejected. Original receipts are never rewritten.
All60 completed grouped runs verify for resume, all15 scene preparations pass
offline,20 variable-training tests pass. Repeat the same grouped evaluate command;
completed runs skip, failed preparation is archived, remaining15 deployments run.

## Current default: grouped-Gaussian evaluation (2026-10-09)

CLI evaluate/report now defaults to --suite grouped. Parent means are the five
training slips [.025,.1,.2,1,10], parent std=.5*mean. Three draws per group
(quick plans one); rejection sampling enforces 0<slip<=10 without clipping.
SeedSequence([training_seed,group_index,20261009]) produces independent deterministic
group streams. Each row records group, parent moments, bounds, exact fixed slip
and task seed;15 rounds total. All models share each row's slip/seed; fresh owned
server per model,3,000 active steps. Corrected scalar setting and passive-policy
observation/action contracts remain unchanged. No per-contact or spatial noise.

Command: `./run_variable_training.sh evaluate --arm varied --suite grouped`, then
`./run_variable_training.sh report --arm varied --suite grouped`.75 deployments
for four grid-trained models+oracle; both arms135. Frozen originals excluded unless
explicit optional baseline is requested. Prior five-round evaluation remains
available via --suite gaussian; fixed/all suites retain previous meanings.

Training plan/checkpoint files are unchanged. A separate immutable schedule.json
is saved at <plan directory>/evaluation_grouped_v1/, with training-plan SHA and
distribution specification. Per-round/controller logs/manifests/raw data/hashed
receipts also live there. Report.md/json in that directory cannot overwrite the
previous root reports. Grouped report checks receipt row/label pairing and hashes;
per-group and overall summaries accompany full run data and held-out-grid fidelity.
Parent moments do not equal post-truncation moments. Mean10 group samples below10;
the positive lower bound permits slips below training minimum. No claim of
statistical equivalence or training-seed uncertainty from three draws per group.
Nineteen variable-training tests pass; existing final checkpoint/plan references
verified. No live runs launched by assistant; no retraining necessary.

## Five Gaussian evaluation rounds (latest, 2026-10-09)

User requested as many evaluation rounds as training slip values: five. Production
Experiment derives this count from training_environment.slips. It reconstructs
the evaluation-only seeded Gaussian stream, verifies the original three-row prefix
is unchanged, and appends sampled_3/slip4.054531697160031/seed4290003 and
sampled_4/slip6.099415186166547/seed4290004. Each model retains paired slip/task seed,
fresh server and3,000 active steps. Quick plans stay one round; fixed suites stay
three seeds per condition. Training data/budgets/plan.json are NOT changed.

Default evaluate/report use five rounds. `--arm varied`:25 deployments (four trained
models+oracle); both arms:45. Run `./run_variable_training.sh evaluate --arm varied`
then `./run_variable_training.sh report --arm varied`. No retraining or new plan
required. Evaluation persists evaluation_schedule_5_rounds.json with original
training-plan hash, Gaussian specification and exact rows; conflicting schedules
are rejected. Existing matching per-run receipts can resume; appended rows are new.
Reports/aggregates include all five rounds. Original three-round artifacts and
the eight completed checkpoints are preserved. Their hashes and plan references
were verified offline. Seventeen variable-training tests passed, including prefix,
plan-immutability, paired evaluation and report checks. No live jobs launched.

## Authoritative workflow: grid training, sampled fixed evaluation (2026-10-09)

User clarified: models train on fixed slips [.025,.1,.2,1,10]; Gaussian is for
evaluation ONLY. Protocol `fixed_grid_training_gaussian_evaluation_v4`, default
plan `experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/plan.json`
(created offline). New identities isolate every preceding generation. Corrected
scalar scene setting, 63-observation/16-action interface and hidden slip persist.

The varied arm assigns grid values round-robin per long/short pair; nominal control
uses .1 with matched task seeds/budgets. BC/mixed DAgger's100k seed pool is20k per
setting. Validation/test each remain10k, now five pairs of1k+1k to cover every
setting. DAgger's seven10k rollout pairs cycle settings, with beta schedule unchanged
(seven pairs are not perfectly balanced across five settings). GAIL's ten4k
updates comprise20 pairs (four per setting), expert/learner surfaces matched.
Other optimizer settings and original checkpoint initializers are unchanged.

Evaluation Gaussian: parent mean5.0125/std2.5, bounds[.025,10], rejection sampling.
Three preplanned rounds at5.774292699386079,6.790566449482138,3.413121168125631 for
seed42. Every model gets identical slip and task seed in a round, held constant
for the full3,000 active steps. Fresh server per model prevents state carryover.
No online slip changes, spatial random field or per-contact noise. Values are
held-out surfaces between sparse grid points; this is a distribution shift from
training. Fidelity test shards remain held-out trajectories on the fixed grid.

Default evaluate/report includes oracle plus trained variants only, not original
frozen policies. Both arms:27 runs (3 rounds x9 controllers); --arm varied:15 runs
(3 x5). Explicit baseline/report --frozen-only remains optional but is not part of
the new required workflow. Original frozen fixed tests still use
`./run_varied_slip.sh fixed` independently; do not change their running inputs.

Commands (after reviewing corrected fixed results and finishing their live job):

```bash
./run_variable_training.sh plan
./run_variable_training.sh all
./run_variable_training.sh evaluate
./run_variable_training.sh report
```

Append --arm varied to all/evaluate/report to omit nominal-only training controls.
All trains to completion but does not automatically evaluate. Lock/port checks,
owned-server startup/cleanup, immutable provenance and resume remain in place.
Reports show slip/seed per round and across-round mean progress, oracle ratios,
total falls, mean targets, held-out action fidelity and resource costs. No random
baseline in this workflow: oracle ratios are NOT Scorenorm. Three rounds are not
independent training replicas. Twenty-six offline tests, including mocked paired
15-run evaluation and report checks, pass; no live training/evaluation launched.

## Three-seed, expanded-range revision (2026-10-09)

User requested three evaluation seeds and an upper field-slip bound of 10.
Fixed default is now `experiments/phase2_domain_robustness/scalar_wide_v4/`, protocol
isolated_server_scalar_wide_slip_v4. Seeds: 101/202/303. Conditions retain
.025/.1/.2 and add 1/10 (CLI names slip_1/slip_10). Eight controllers x five
conditions x three seeds = 120 runs, 3,000 active steps each. Report/optional
uniform sampled planner follow the same three seeds and accept bounds up to 10.
Prior outputs are untouched; the common lock remains unchanged.

Gaussian default plan path is now
`experiments/variable_slip_training/gaussian_scalar_wide_v3_seed42/plan.json`,
protocol variable_slip_hidden_scalar_wide_v3. Run `./run_variable_training.sh plan`
before baseline/training. Bounds [.025,10]; Gaussian evaluation uses three paired
draws (oracle+four frozen policies = 15 runs). Optional fixed suite uses three
seeds per condition. Quick plans retain one evaluation seed. Training segment
counts, training seed42, budgets, nominal .1 control, architectures and recovery
remain unchanged. The corrected scalar setter remains active throughout.

Only the upper Gaussian bound has changed: parent mean .1/std .035 remain pending
the user's distribution preference. This distribution almost never samples large
slips; raising the cap is NOT broad coverage. For broad exposure a possible
explicit choice is `plan --mean 5.0125 --std 2.5` with the same bounds; do not
silently claim this is the current default. Decide before creating the immutable
plan; changing an existing plan requires a new --plan path. Stress/sensitivity
diagnostic suites retain their historical seed settings and are separate studies.

## Corrected fixed/Gaussian pipeline generation (2026-10-09)

The shared configure_field_slip.py now reads scalar and legacy isotropic paired
setters, but writes `(setContactSlip v)` by default. It rejects nonfinite/negative
values and records scene_setter_method in status/manifests. Backups preserve the
original syntax. Explicit legacy writing exists only for the historical contact
diagnostic arms. Installed simulator files are not changed by the owned-server
pipelines; each run edits its own local scene copy. Scalar diagnostic conversion
remains isolated and still works from an explicitly legacy generated scene.

Fixed evaluation defaults: protocol `isolated_server_scalar_slip_v3`, output
`experiments/phase2_domain_robustness/scalar_v3/`. Commands:

```bash
./run_varied_slip.sh fixed
./run_varied_slip.sh report-fixed
```

Scope remains eight controllers (including oracle/random and historical variants),
three fixed slips .025/.1/.2, five seeds, 3,000 active steps: 120 runs. Fresh servers
and same-condition oracle/random normalization are retained. Reporter follows the
new output root and rejects non-scalar manifests/old protocol receipts. Optional
legacy sampled workflow remains uniform; use variable_training for Gaussian draws.
The shared lock stays at experiments/phase2_domain_robustness/.runner.lock, not in
the new output subdirectory, to protect all launchers from concurrent server use.

Gaussian continuation/evaluation defaults: protocol `variable_slip_hidden_scalar_v2`,
plan `experiments/variable_slip_training/gaussian_scalar_v2_seed42/plan.json`.
The plan records single_argument_v1. A corrected default plan was generated
offline; original deterministic draw settings/budgets are retained.

```bash
./run_variable_training.sh plan
./run_variable_training.sh baseline
./run_variable_training.sh report --frozen-only
# Review corrected frozen results before initiating training:
./run_variable_training.sh all
./run_variable_training.sh evaluate
./run_variable_training.sh report
```

Baseline remains oracle plus frozen BC/mixed DAgger/GAIL-BC/GAIL-DAgger, five paired
Gaussian draws. Every oracle collection, DAgger rollout, GAIL rollout and evaluation
uses the corrected server preparer. Both nominal-only and varied arms are corrected;
they initialize from the same original root checkpoints, not legacy continuation
outputs. Bounds [.025,.2], parent mean .1/std .035, 63/16 observation/action contracts,
data splits, learning schedules, metrics and hidden-context design are unchanged.
Gaussian ratios remain oracle ratios, not random-normalized Scorenorm.

Both server managers verify scalar syntax, expected value and scene/config hashes
at startup. Full pipelines do not attach the passive ODE logger on every run;
the scalar oracle smoke is the live manipulation evidence. The migrated orchestration
has offline coverage, not yet a completed live batch. Old plans fail deterministic
protocol validation and old receipts cannot resume into new outputs. No artifacts,
models or historical receipts were moved/deleted/rewritten. Installed .1 scene
still has its original legacy syntax; do not mistake it for corrected run physics.
23 offline tests and ten real-config preparations passed; no simulations started.

## Single-argument field setter experiment (2026-10-09)

Command: `./run_slip_sensitivity.sh smoke --suite scalar`. Five oracle-only
runs at .025/.1/.2/1/10, seed101, 3,000 active steps each. Uses fresh owned servers,
passive ODE audit, existing evaluator/recovery and 63-observation/16-action
contracts. No server should be started beforehand. Runs automatically to completion
unless a validation/infrastructure error occurs; user initiates live experiments.

New method in contact_slip/scalar_scene.py converts ONLY the generated local
field setter from `(setContactSlip v v)` to `(setContactSlip v)`. This follows
the installed scalar native API; script-binding behaviour remains a hypothesis
until live contact evidence confirms it. The installed scene, legacy methods,
robot contact settings, models and datasets are unchanged. Normal SimSpark surface
combination remains in effect; this is not a direct solver override.

Conversion requires one matching two-value expression, requested-value agreement
and original scene checksum. It updates the local scene hash, both Ruby checksum
references, startup marker and configuration hashes. The manifest records
scene_setter_method=single_argument_v1 and the pre-conversion scene hash. Audit
still forwards contacts unchanged and records enabled solver slip/mode classes.
If every run retains identical classes, the report warns and validation remains
unresolved. No automatic inference of physical adaptation is made.

Separate artifacts: experiments/slip_sensitivity/oracle_scalar_v1/ with plan.json,
smoke/<slip>/oracle_101/, smoke_report.md/json and normal hashed run evidence.
`report-smoke --suite scalar` regenerates only the summary. Optional
`evaluate --suite scalar` is 25 runs (five settings x five task seeds), but should
not be initiated before reviewing smoke physics. Old receipts remain immutable;
changed-code attempts must use new output directories, not relaxed fingerprints.
Ten offline tests plus preparation of all five real local configurations pass;
no live scalar experiment has been launched by the assistant.

## Large-slip oracle screen (2026-10-09)

Added `--suite stress` to run_slip_sensitivity.sh. Defaults to a separate
`experiments/slip_sensitivity/oracle_stress_v1/` directory. Scene slips are
.1/.5/1/2/5/10; oracle only. `smoke --suite stress` runs seed 101 at each setting
(six runs), writes smoke_report.md/json automatically. `evaluate --suite stress`
optionally runs all five matched seeds (30 runs) and writes report.md/json.
Both retain 3,000 active steps, unchanged oracle/evaluator, passive contact logging
and owned-server safeguards. `report-smoke` regenerates the smoke summary offline.
Audit JSON now includes exact slip1/slip2/mode classes and counts. Reports flag
identical recorded classes, distinguishing contact-count changes from actual
value changes. Existing default standard plan is unchanged; earlier receipts are
not rewritten and code fingerprints remain strict. Full design, interpretation
gates and commands: contact_slip/STRESS_TEST_DESIGN.md. Seven offline tests pass;
no live stress runs yet. Run the six-run screen before deciding on confirmation.

## Passive contact-slip sensitivity pilot (2026-10-09)

Entry point: `./run_slip_sensitivity.sh {selftest,smoke,evaluate,report}`;
implementation `contact_slip/sensitivity.py`. Default interpreter is fcportugal;
override with PYTHON_BIN. Requires g++, ODE development headers/runtime, SimSpark
and existing evaluation dependencies. Do not pre-start a server: ports 3100/3200
must be free. Existing servers are refused, not killed. Each run owns a fresh
server, uses the shared experiment lock, scene/hash startup verification and
server-death watchdog, and reaps its server on completion/error/interrupt.

This is a diagnostic pilot, not Gaussian-contact training. `smoke` runs oracle
seed 101 at field slips .025/.100/.200 (three runs). `evaluate` runs oracle and
frozen root BC at all three slips with task seeds 101,202,303,404,505 (30 runs),
then generates the report. Both use 3,000 active steps; recovery remains excluded.
No interactive checkpoints. Original policies, training data and results remain
unchanged. The system scene is never modified.

The process-local ODE interposer's `audit` mode forwards the original contact
pointer unchanged. It records both solver-bound slip values and enable flags
after surface combination rather than overriding dynamics. Scope is horizontal
z=0 plane contacts within x=+-15,y=+-10, including robot/ball, startup and recovery.
This is not foot-specific or a measurement of actual sliding. Disabled slip axes
count as zero in summaries; raw values and flags remain recorded. Empty contact
traces, inconsistent flags, modified first-axis values, invalid slip numbers or
incomplete/nonfinite rollout traces prevent completion. Observation/action remains
63/16; the unchanged evaluator supplies performance and mechanical metrics.

Artifacts: `experiments/slip_sensitivity/v1/{plan.json,smoke/,runs/,report.json,report.md}`.
Each run includes scene/config copies, environment_manifest.json, server/evaluation
logs, contacts.csv, contact_audit.json, raw evaluation JSON/NPZ, and hashed
completed.json or failure.json. Resume validates inputs and artifact hashes;
interrupted attempts are archived by the common begin helper. Changed inputs need
a new --output directory. Reporting requires all 30 runs, verifies receipts and
shows paired progress differences relative to .100, total falls and both slip
ranges. Full per-run metrics and enabled-contact counts remain in JSON.

Native builds are content-addressed by source/compiler/header/runtime identity;
previous binaries are retained. `selftest` compiles and exercises standalone ODE
without SimSpark or sockets. Five Python tests and the passive/Gaussian/constant
native checks pass; live validation is pending. Weak sensitivity and policy
equivalence must not be inferred before reviewing contact evidence and results.

This repository treats the pretrained FC Portugal walking policy as an immutable
oracle. Three learned walking policies are trained and compared in this order:

1. Behavioral cloning (BC)
2. Dataset Aggregation (DAgger)
3. Generative Adversarial Imitation Learning (GAIL)

Historical defects, observed training failures, mitigations, and open research
risks are recorded in [CHALLENGES.md](CHALLENGES.md).
The status and scope of each experimental phase are recorded in
[EXPERIMENT_PHASES.md](EXPERIMENT_PHASES.md).

Phase 2 contact-slip experiments use `configure_field_slip.py` for guarded
backup/apply/status/restore operations, `run_phase2_smoke.sh` for a short oracle
feasibility check, and `run_phase2_condition.sh` for a complete condition. Every
evaluation JSON records `environment_id` and `field_contact_slip`; aggregation
rejects directories that mix environment conditions.

All policies consume the same 63-dimensional walking observation and produce the
same 16-dimensional latent walking action. A learned action is **not** a vector
of joint positions. It must be passed to `Walk.execute_action()`, which delegates
to the original FC Portugal walk environment for action smoothing, step
generation, inverse kinematics, and joint target generation.

## Simulator requirements

Collection and rollout use SimSpark's synchronous protocol. Configure these
settings before starting the server:

```ruby
# ~/.simspark/spark.rb
$agentSyncMode = true
```

```ruby
# /usr/share/rcssserver3d/rcssserver3d.rb
$enableRealTimeMode = false
```

Restart `rcssserver3d` after changing either file. Verify with:

```bash
python3 check_simulator_config.py
```

Each synchronous control cycle is:

```text
construct observation -> select action -> send command plus (syn) -> receive state
```

This ordering prevents deadlocks and stops wall-clock delays between scripts
from advancing physics without agent control.

The launchers check and, when necessary, rebuild the FC Portugal Pybind11
modules for the active Python interpreter. Activate the intended environment
before running a pipeline. Override the interpreter with
`PYTHON_BIN=/path/to/python` if necessary.

## Shared environment contract

The shared server client drains unused monitor output after each synchronous
receive. This prevents monitor-socket backpressure from crashing SimSpark during
long multi-policy evaluations.

## Artifact layout

The repository root contains stable final policies and pipeline entry points.
Training state is grouped by policy:

```text
artifacts/
  bc/                 training metrics
  dagger/             aggregate, checkpoints, rollout/training metrics
  dagger_mixed/       aggregate, checkpoints, rollout/training metrics
  gail_dagger/        active DAgger-initialized GAIL state
  gail_bc/            BC-initialized GAIL state
  gail_random/        random-initialized GAIL state
experiments/           completed and superseded experiment archives
evaluation/            current standardized evaluation and plots
```

Each pipeline copies its final deployable policy to the repository root only
after successful completion. This keeps downstream evaluation paths stable
without mixing intermediate checkpoints into the main directory.

Evaluation runs are resumable. If an interrupted run has retained its output
directory, continue it without overwriting completed policy/seed cases using:

```bash
RESET_EVALUATION=0 RESUME_EVALUATION=1 ./run_evaluation.sh
```

The primary comparison evaluates only the final BC, DAgger, and GAIL policies
by default. Evaluating every intermediate checkpoint creates many short-lived
SimSpark agent/monitor connections and is both slower and more likely to expose
the server's connection-teardown race. Run checkpoint analysis separately with:

```bash
RESET_EVALUATION=0 RESUME_EVALUATION=1 EVALUATE_CHECKPOINTS=1 ./run_evaluation.sh
```

Confidence intervals use the normal approximation across seeds. The reported
lower confidence bound for fall rate is clamped to zero because negative fall
rates are outside the metric's physical support.

Generate the evaluation plots after summarization with:

```bash
python3 plot_evaluation.py
```

This writes `evaluation/evaluation_summary.png` and a vector PDF version at
`evaluation/evaluation_summary.pdf`. The four panels report normalized return,
imitation fidelity, locomotion stability, and supervisory/environment cost. A
full-width Phase 1 table beneath the panels reports normalized return with its
95% interval, Action MSE, Wasserstein distance, forward velocity, fall rate,
oracle queries, and environment steps for every available final policy.

The written interpretation of the completed nominal-domain experiment is stored
in `evaluation/PHASE1_RESULTS_REPORT.md`.

The earlier report covering only the original BC, DAgger, and DAgger-initialized
GAIL models is preserved with its exact source results at
`experiments/archive_20260905_pre_variants/gail_dagger/evaluation/PHASE1_ORIGINAL_3_REPORT.md`.

Targets are restricted to a buffered region in the player's own half:

```text
x in [-6.5, -0.5]
y in [-6.5,  6.5]
```

Long-range targets cover this region. Short-range targets remain 1–2 metres
from the robot and are rejected if they leave it.

A fall is detected when localized head height drops below 0.28 metres or
absolute torso roll/pitch exceeds 60 degrees. Recovery
beams the player to `(-3, 0)`, commands `Zero_Bent_Knees` while suspended,
places it at the robot-specific ground height, and performs synchronized
stabilization cycles. Recovery cycles are excluded from imitation datasets.

Shared implementation:

- `behaviors/custom/Walk/Walk.py`: observation, oracle-query, and latent-action API
- `bc_data.py`: target sampling and validated atomic dataset/JSON operations
- `bc_sim.py`: synchronous fall detection and stabilization
- `check_simulator_config.py`: headless simulator preflight

Superseded prototype collectors and their dataset are retained for provenance
under `archive/legacy_imitation/`. They are not part of any supported pipeline.

## Behavioral cloning

### Technical flow

`run_pipeline.sh` alternates between `collect_long.py`, for long-distance
walking, and `collect_short.py`, for nearby targets and turns. Every useful
control cycle stores a 63-D observation and the corresponding 16-D FC Portugal
oracle action. Validated chunks are atomically appended to
`expert_walk_data.npz`. `split_expert_data.py` then assigns contiguous 250-step
blocks to a fixed seeded training/test split. The held-out test set is never
used for policy training.

`train.py` trains a `63 -> 256 -> 256 -> 16` ReLU network with mean squared
error and Adam. The train/validation split and minibatch order are seeded. The
checkpoint with the lowest validation loss is retained as a raw PyTorch
`state_dict`, allowing its actor weights to be reused later.

### Run BC

For a clean experiment:

```bash
RESET_DATASET=1 ./run_pipeline.sh
```

An existing dataset is moved to a timestamped backup. Without the reset flag,
new demonstrations are appended.

Common configuration:

```bash
N_RUNS=10 \
STEPS_PER_COLLECTOR=5000 \
EPOCHS=100 \
BATCH_SIZE=256 \
VALIDATION_FRACTION=0.1 \
SEED=42 \
./run_pipeline.sh
```

Outputs:

- `expert_walk_data.npz`: complete pure-oracle collection archive
- `expert_train_data.npz`: oracle samples available for training
- `expert_test_data.npz`: fixed held-out action-fidelity test set
- `expert_split_metadata.json`: reproducible block assignment
- `bc_walk_policy.pth`: best validation checkpoint
- `bc_training_metrics.json`: configuration and offline imitation metrics

Training metrics include validation MSE, RMSE, MAE, R-squared, and per-action
RMSE/MAE.

## DAgger

### Technical flow

`run_dagger.sh` performs this sequence:

1. `collect_seed.py` collects a pure oracle seed if no aggregate exists.
2. `train_dagger.py` trains iteration 0 as BC on that seed.
3. `rollout_dagger.py` executes an oracle/student mixture policy.
4. FC Portugal labels every visited state, independent of the executed policy.
5. New state/oracle-action pairs are atomically aggregated.
6. The student is retrained on the complete aggregate.
7. Steps 3–6 repeat, including training after the final rollout.

Default oracle-control probability (`beta`):

```text
iteration 1: 0.8
iteration 2: 0.6
iteration 3: 0.4
iteration 4: 0.2
iteration 5+: 0.0
```

This protects early rollouts while converging to the learner's state
distribution. The oracle still labels every recorded state.

### Run DAgger

Run BC first. For a clean DAgger experiment:

```bash
RESET_DAGGER_DATASET=1 ./run_dagger.sh
```

The reset is important if the existing aggregate came from an older rollout
implementation. It is backed up rather than deleted.

Common configuration:

```bash
TOTAL_ITERATIONS=7 \
SEED_STEPS=20000 \
ROLLOUT_STEPS=10000 \
EPOCHS=50 \
BATCH_SIZE=256 \
VALIDATION_FRACTION=0.1 \
SEED=42 \
./run_dagger.sh
```

Outputs:

- `aggregated_dagger_data.npz`: visited states paired with oracle actions
- `dagger_policy.pth`: latest student
- `dagger_policy_iter_N.pth`: archived iteration checkpoints
- `dagger_training_metrics_iter_N.json`: offline training metrics
- `dagger_rollout_metrics_iter_N.json`: on-policy and mechanical metrics

Rollout metrics cover oracle-action error, realized oracle-control fraction,
falls, target completions, torso roll/pitch, head height, and action smoothness.

### Mixed-seed DAgger variant

The original DAgger experiment started from 20,000 long-target oracle samples,
then added seven 10,000-step mixed/student-policy rollouts. Its final dataset had
90,000 labels—the same number as the BC training split—but its initial coverage
was much narrower and did not use the explicit short-turn collector.

`run_dagger_mixed.sh` isolates this dataset-composition hypothesis. Its default
seed contains 100,000 fresh FC Portugal samples collected in five alternating
rounds of 10,000 long-range and 10,000 short-turn steps. It then adds the same
seven 10,000-step DAgger rollouts as the original, producing 170,000 samples.
The original dilution schedule is unchanged:

```text
iteration:  1    2    3    4    5    6    7
beta:      0.8  0.6  0.4  0.2  0.0  0.0  0.0
```

Run the clean variant with:

```bash
RESET_DAGGER_DATASET=1 ./run_dagger_mixed.sh
```

Its intermediate artifacts live under `artifacts/dagger_mixed/` and do not
overwrite the original DAgger experiment. The standard evaluator includes `dagger_mixed_policy.pth`
automatically when it exists.

## GAIL

### Technical flow

All GAIL variants use the same FC Portugal expert occupancy artifact:

- `expert_train_data.npz` supplies training-only occupancy samples visited by
  FC Portugal.
- `gail_dagger` initializes its actor mean from `dagger_policy.pth`.
- `gail_bc` initializes its actor mean from `bc_walk_policy.pth`.
- `gail_random` uses the seeded PyTorch actor initialization.

The DAgger aggregate is not GAIL expert occupancy: its states were visited by
mixed/student policies and only labeled counterfactually by the oracle.

Each iteration has two phases:

1. `rollout_gail.py` samples the current Gaussian actor and stores observations,
   actions, old log probabilities, critic values, terminal masks, and a final
   bootstrap value.
2. `train_gail.py` normalizes discriminator inputs using fixed expert
   statistics, updates a regularized discriminator on balanced expert/policy
   batches, recomputes rewards, calculates generalized advantage estimates, and
   updates the actor-critic using trust-region-guarded PPO.

The discriminator uses smoothed expert/policy labels (0.9/0.1), interpolation
gradient penalty, and early stopping when both classification accuracies exceed
0.95. The non-saturating generator reward is:

```text
r(s, a) = log(D(s, a)) = log-sigmoid(discriminator_logit)
```

Fall terminals are assigned to the preceding action. Nonterminal rollout
truncations use the saved next-state value. Actor, critic, and discriminator use
separate Adam optimizers. PPO stops when full-rollout KL exceeds its target and
rolls the actor back if it exceeds the safety threshold. A small, decaying BC
anchor discourages the actor from catastrophically leaving the oracle manifold.

### Run GAIL

Train the two new initialization variants with:

```bash
RESET_GAIL=1 ./run_gail_bc.sh
RESET_GAIL=1 ./run_gail_random.sh
```

The completed DAgger-initialized policy is retained as
`gail_dagger_policy.pth`. Direct use of `run_gail.sh` defaults to this
`gail_dagger` variant.

Existing current GAIL state, per-iteration checkpoints, and GAIL rollout and
training metrics are moved into a timestamped backup directory.

Common configuration:

```bash
ITERATIONS=10 \
ROLLOUT_STEPS=4000 \
PPO_EPOCHS=3 \
DISC_EPOCHS=1 \
BATCH_SIZE=256 \
ACTOR_LEARNING_RATE=0.000003 \
CRITIC_LEARNING_RATE=0.0003 \
DISC_LEARNING_RATE=0.0001 \
ENTROPY_COEFFICIENT=0.001 \
BC_COEFFICIENT=0.05 \
GRADIENT_PENALTY_COEFFICIENT=10.0 \
TARGET_KL=0.02 \
ROLLBACK_KL=0.05 \
SEED=42 \
./run_gail.sh
```

Outputs:

- `<variant>_policy.pth`: latest actor-critic
- `<variant>_disc.pth`: latest discriminator
- `<variant>_optimizer_state.pth`: persistent optimizer state
- `<variant>_rollout_batch.npz`: latest PPO rollout
- `<variant>_policy_iter_N.pth`, `<variant>_disc_iter_N.pth`: checkpoints
- `<variant>_rollout_metrics_iter_N.json`: mechanical metrics
- `<variant>_training_metrics_iter_N.json`: discriminator/PPO diagnostics

Diagnostics include discriminator loss/accuracy, learned reward statistics,
actor/critic loss, entropy, approximate KL, PPO clip fraction, and actor-mean
error on oracle demonstrations.

## Standardized comparison protocol

`run_evaluation.sh` evaluates the uniform-random baseline, FC Portugal oracle,
BC, DAgger, and every available GAIL initialization variant under identical seeds
and target generation. It uses 3,000
active steps (60 seconds at 50 Hz) per seed by default. Fall recovery and
stabilization do not consume the active-step budget.

Run it after all training pipelines finish:

```bash
RESET_EVALUATION=1 ./run_evaluation.sh
```

Common configuration:

```bash
EVAL_STEPS=3000 \
EVAL_SEEDS="101 202 303 404 505" \
EVALUATE_CHECKPOINTS=0 \
./run_evaluation.sh
```

When checkpoint evaluation is enabled, every available DAgger and GAIL
iteration checkpoint is evaluated. This supports normalized-return convergence
curves against cumulative DAgger oracle queries and GAIL environment steps.

`evaluate_policy.py` writes an NPZ trajectory and JSON summary for every
policy/seed pair under `evaluation/raw`. Each active step records observations,
latent policy and oracle actions, 22 joint angles, position, target, target
progress, planar and torso-forward velocity, directional error, torso attitude, head height, and a
reset-separated episode identifier.

`summarize_evaluation.py` produces `evaluation/comparison.json` and
`evaluation/comparison.csv` with:

- **Normalized return:** cumulative target progress normalized per seed between
  random (0) and FC Portugal (1), without clipping values outside `[0, 1]`.
- **Resource complexity:** oracle labels/queries for BC and DAgger and cumulative
  environment steps for GAIL.
- **Action MSE:** deterministic 16-D latent-action output against FC Portugal on
  the fixed `expert_test_data.npz` partition.
- **Trajectory distance:** sliced Wasserstein-1 over standardized 22-D joint
  occupancy using 256 deterministic projections.
- **Kinematics:** target-directed velocity, torso-forward velocity, planar speed, dimensionless target
  direction error, target completions, and target-progress return.
- **Falls:** one event per recovery, normalized by active deployment minutes.
- **Jerk:** integrated squared third finite difference of joint angles in
  radians after a five-sample moving average. Reset boundaries are excluded.
  Integrated (`rad^2/s^5`) and duration-normalized (`rad^2/s^6`) costs are saved.
- **Uncertainty:** mean and normal-approximation 95% confidence intervals across
  seeds.

The random baseline samples each latent dimension uniformly between its minimum
and maximum in `expert_train_data.npz`. BC and DAgger use deterministic network
outputs; GAIL uses its deterministic actor mean.

## Recommended full experiment

Start `rcssserver3d`, verify its configuration, then run:

```bash
RESET_DATASET=1 ./run_pipeline.sh
RESET_DAGGER_DATASET=1 ./run_dagger.sh
RESET_GAIL=1 ./run_gail_bc.sh
RESET_GAIL=1 ./run_gail_random.sh
RESET_EVALUATION=1 ./run_evaluation.sh
```

Use identical simulator configuration, robot type, target bounds, reset logic,
seeds, and evaluation duration when comparing algorithms. Training metrics and
mixed-policy DAgger rollouts are diagnostics; the final scientific comparison
should use a fixed-seed evaluator that runs final BC, DAgger, and GAIL policies
without oracle control.

## Maintenance requirement

This document is part of the implementation contract. Any change to pipeline
control flow, model architecture, data format, simulator assumptions,
hyperparameters, CLI/environment variables, outputs, or metrics must update
`PIPELINES.md` in the same change.

## Interactive live demonstrations

Start `rcssserver3d` and connect RoboViz, then run:

```bash
./run_demo.sh
# Optional reproducible target/action seed (default 101):
./run_demo.sh --seed 202
```

The terminal menu selects 1: FC Portugal, 2: BC, 3: original DAgger,
4: mixed-seed DAgger, 5: GAIL-DAgger, 6: GAIL-BC, 7: GAIL-random,
8: uniform-random baseline, or 0: exit. Each invocation runs one selected agent.
Missing checkpoints are marked and cannot be selected. Ctrl+C stops the run.
`PYTHON_BIN` overrides the launcher's interpreter; it prefers the installed
`fcportugal` environment and falls back to `python3` if unavailable.

`demo_agents.py` runs 1,500 active 20-ms steps (30 simulated seconds), pacing
cycles with a monotonic clock to at most 50 Hz. Processing time is included in
each cycle; slow cycles do not trigger a burst of catch-up steps. On an overloaded
host the run can take longer than 30 wall-clock seconds. Startup stabilization
and fall recovery are also paced at 50 Hz and add time outside the active budget.
The existing synchronous-mode/disabled-server-real-time preflight still applies;
restart the server after configuration changes. The demo does not edit settings
or launch the server/viewer. Use the simulator without another agent controller.

The runner uses localhost agent/monitor ports 3100/3200, robot type 0, uniform
number 1, and team `WalkerDemo`. It sets and verifies `PlayOn` via the monitor
connection and starts from the shared recovery pose. Target generation, the
63-D observation, 16-D latent action execution, fall recovery, and deterministic
policy loading match the evaluation helpers. GAIL uses its actor mean. Only the
random baseline requires `expert_train_data.npz`, for per-action sample bounds.

Terminal progress is printed every 250 active steps with target and fall counts.
No evaluation traces, datasets, checkpoints, or benchmark summaries are written;
these paced demonstrations are not additional experimental results. The agent
connection is closed after completion or interruption using existing serialized
socket teardown.

## Human-run local-scene varied-slip pipeline (2026-09-25)

Implemented next-stage training is documented in
[variable_training/README.md](variable_training/README.md). Run
`./run_variable_training.sh plan`, then after the active sweep finishes use
`baseline`, `all`, `evaluate`, `report`. The separate package leaves the running
evaluation's fingerprinted modules untouched. Only BC, mixed DAgger, GAIL-BC and
GAIL-DAgger are trained; FC Portugal stays unchanged. Both varied-slip and nominal
continuation arms run by default. A truncated Gaussian (0.025–0.200, parent mean
0.100/std 0.035) supplies immutable paired plans. Shared simulator locking prevents
concurrent live use. Per-client server ownership, verified scenes, immutable shards,
whole-run validation and transaction receipts support stage-boundary resume.
Supervised policies warm-start at LR 1e-4; GAIL uses separate segment GAE and fixed
expert-pool normalization. Reports use oracle ratios, not random-normalized scores.
Full budgets/data contracts and quick-plan commands are in that README.

Variable-training baseline/evaluate/report now default to `--suite gaussian`:
only the five preplanned sampled draws are run. Baseline comprises 25 runs across
oracle and four frozen policies. Existing fixed-slip results are not rerun.
`--suite fixed|all` is opt-in. Gaussian-only reports leave nominal retention null
rather than requiring new nominal runs or mixing historical task seeds. The plan
and draw identities are unchanged; the selector filters scheduled evaluations.

Evaluation receipts exclude scheduler/report (`variable_training/run.py`), receipt
metadata helpers (`common.py`) and pipeline tests from source identity. Actual
client/server sources, plan, environment, models and output hashes remain checked.
Audited historical hashes of those three metadata files are accepted read-only;
unknown legacy versions are rejected. Training receipts remain fully strict.
Any future change to evaluation semantics in the scheduler must bump the protocol.

Next training specification: [VARIABLE_SLIP_TRAINING.md](VARIABLE_SLIP_TRAINING.md)
describes planned BC/DAgger/GAIL continuation with isolated servers, immutable
shards, resume receipts, whole-run splits and matched nominal controls. Proposed
Gaussian bounds are 0.025–0.200, parent mean 0.100 and std 0.035, pending review.
Its original proposed interface is superseded by the implementation commands above.
Current running evaluation is unchanged. Validation: 17 offline tests pass across
the new training package and existing isolated evaluator; live training is pending.

`run_varied_slip.sh` dispatches live stages to `run_varied_slip.py` and offline
reports/plans to `summarize_varied_slip.py`. The operational commands and complete
artifact contract are in `experiments/phase2_domain_robustness/README.md`.
The user initiates experiments. Each command executes all requested conditions or
draws without interactive prompts. Fixed runs proceed automatically from smoke
to full evaluation. Errors still stop execution and clean up the owned server.

The original root testing files remain the baseline, with reference snapshots and
hashes in `experiments/phase2_domain_robustness/baseline_testing/`. The wrapper calls
original `evaluate_policy.py` directly for smoke and every policy/seed run.
Protocol `isolated_server_slip_v2` uses a fresh server for each run, including
smoke. Simulator seed remains 0. Policy order, active-step/fall accounting,
Python/NumPy/Torch seeds, random baseline bounds, and metrics remain original.
Each run stores its own scene, scripts, server/evaluation logs, manifest, raw
summary/trace and completion receipt under `conditions/<condition>/runs/<label>_seed_<seed>`.
Smoke uses `smoke/<condition>`. Use `report-fixed` after the full sweep; the wrapper
no longer creates the original per-condition `evaluation/comparison.*` files.

The server uses `rcssserver3d --script-path <local rcssserver3d.rb>`.
Smoke runs now use 3,000 active steps, matching full evaluation runs.
Local scripts verify the scene SHA-256 inside the Ruby server process, require
a successful scene import, and print a flushed per-launch UUID marker containing
the requested slip and checksum. The wrapper requires an exact marker line,
rechecks scene slip/checksum and startup-script hashes, and records
`server_scene_import_verified` in the manifest before evaluating. This verifies
source bytes and import success, not independent runtime ODE parameter readback;
field slip is the only changed physical setting. Installed configuration remains
unchanged; final checksum/slip verification is saved and the owned server stops.
Concurrent-run locking and occupied-port checks protect the user session. Rerunning
the same command validates receipts, input/configuration fingerprints, artifact
checksums, run metadata, trace dimensions/lengths and finite values before skipping.
Incomplete attempts are renamed with `_failed_<id>` before retry. Changed completed
artifacts/inputs cause an error rather than silent reuse. A watchdog terminates an
evaluator if its server exits. There is no rollout time limit or automatic retry
loop; errors stop the command and clean up its owned processes.

Cross-condition reports validate 120 fixed results, calculate same-seed baseline
normalization, per-seed and ratio-of-means nominal retention, and save task, fall,
torso RMS, head-height and episode-aware jerk metrics. Sample plans require a
complete fixed report, human-selected bounds/rationale, and save paired uniform
draws plus distribution variance and source fingerprints. Five draws are a pilot;
near-zero nominal returns make retention unstable. No significance claims.

Offline validation: `python -m unittest test_varied_slip -v`. Live validation remains
user-run. No training occurs. The first draft's fresh-start/seeding changes were
removed before any experimental data was collected.
# Managed model/slip viewer (2026-10-15)

Implementation lives in `viewer/` (a contextual adaptation of the existing demo
loop) so viewer additions do not alter training source fingerprints. Historical
`demo_agents.py` remains unchanged.

`./run_viewer.sh` lists available historical policies and completed consolidated
models under `experiments/consolidated/v1/policies`, then asks for a model and
fixed scalar slip. `--root experiments/consolidated/smoke_v1` selects smoke models;
`--list` only lists models. `--model` accepts an exact displayed label, `--slip`
accepts a finite non-negative scalar or `original`, and `--seed` defaults to101.
Example: `./run_viewer.sh --model varied/input/seed_42/bc --slip 1`.
The original scene is preserved byte-for-byte; scalar slip uses the corrected
single-argument setter. Input models receive the selected scalar through their
existing context encoding; original-scene viewing is rejected for these models.
Hidden models do not receive slip. GAIL uses deterministic actor means.

The launcher owns and verifies its server via `consolidated.server`, takes the
shared simulator lock, rejects occupied ports, and never changes system scenes.
Do not run concurrently with training/evaluation. Open RoboViz separately on
localhost:3200 and press Enter when ready. Each demo uses the existing paced
1,500-step/30-second loop, PlayOn verification, own-half targets and fall recovery.
Ctrl+C cleans up the owned server. Scene manifests and server logs are retained
in unique `logs/viewer/demo_*` directories; these are demonstrations, not scored
evaluation artifacts. The original `run_demo.sh` interface is unchanged.

# Consolidated generation probing — 2026-10-15

`./run_consolidated_probing.sh audit|run|report` is the offline probe entrypoint
for all28 new actors plus the unchanged FC Portugal expert. No simulator starts.
Default data: `experiments/consolidated/v1/evaluation/all_all_correct`; default
output: `experiments/representation_probing/consolidated_v1`. Flags `--data`,
`--output`, `--stride` (7) must remain identical on resume. The new package does
not modify historical probe code or training source fingerprints.

Reuse30 general locomotion targets, not slip decoding, on shared receipt-verified
oracle trajectories. Grouped draws0/1 train, draw2 validates, confirmation tests:
10/5/15whole runs. Correct context is appended only for64input actors/controls.
75representations (all student h1/h2, expert, matched random networks, raw63/raw64,
phase-command controls) x4label modes =300ridge fits. Readouts use train-only
scalers, validation-selected penalties, held-out RMSE/R2 and contact metrics.
Expert action reproduction, source/checkpoint hashes and finite model weights are
checked in audit. Output reportJSON/MD and CSVs include per-replica scores,
student/expert gaps, persistence baselines, spectra, pooled/per-run/permuted CKA.
Source actors never update. No new layer-reset interventions or R1 claims.
Read `consolidated_probing/README.md` for contracts, references and limitations.

# Consolidated results synthesis — 2026-10-15

`evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md` presents the completed
1,020deployments and300multitarget probes in experimental-stage order. It includes
the actual round/seed design and metric contracts, with a dedicated DAgger section.
Read-only numerical builder:
`python evaluation/analysis/consolidated_summary.py` prints Markdown to stdout;
it reads the full-batch evaluation/probe/training JSONs without changing their
artifacts or launching experiments. Tables average balanced replicas descriptively,
retain exposure denominators and separate original versus scalar-nominal scenes.
Historical meeting summaries are not replaced by this report.

# Append scratch-initialized comparisons — 2026-10-15

Existing launchers now accept `--include-scratch`. Without the flag the historical
consolidated code/identities remain unchanged. With it they route to the isolated
`scratch_comparison` package; no completed parent files are overwritten.

```bash
./run_training.sh plan --include-scratch
./run_evaluation_all.sh audit --include-scratch
./run_training.sh run --include-scratch && ./run_evaluation_all.sh run --include-scratch
./run_consolidated_probing.sh run --include-scratch
```

Adds24 varied scratch-lineage models (BC/mixed DAgger/GAIL-BC/GAIL-DM ×hidden/input
×3seeds), plus direct random-GAIL in the original environment(seed42) and varied
hidden environment(seeds42/43/44). No input-conditioned random-GAIL. BC/DAgger
start from random weights; hidden/input paired seeds share random63input weights,
with the added context column zero. GAIL-BC/DM initialize from those new same-domain
scratch-trained parents. Direct GAIL-random has no pretrained actor, but retains
the existing BC anchor and PPO stabilization settings.

Reuse verified parent expert shards, validation/test pools, training slips/modes,
170k-labelled budgets and70klearner/PPO steps. This matches incremental budgets,
not accumulated pretraining cost. Original scratch BC/DAgger/GAIL-parent models
already exist and are not duplicated. New artifacts live in
`experiments/consolidated/v1/scratch_v1/`; `--output` selects the existing parent,
`--extension-output` optionally relocates the append-only extension. Training
`--algorithm` supports selecting one method (with new initializer dependencies).

Evaluation uses the exact saved full/correct-context parent schedule:1,020 old
deployments reused plus929new (original random-GAIL38;27varied models×33), for
1,949combined. All metrics and paired baselines retain the original contracts;
new JSON/CSV/MD include both cohorts with scratch-versus-continuation deltas.
Completed new jobs are also receipt-verified and skipped on resume. Parent code,
configurations, inputs and artifact hashes are checked; changes fail safely.

Probing reuses all300verified parent fits, oracle traces/splits/targets and controls;
only56new layers×4label modes=224fits are trained. Combined524fits/15,720target
scores include expert/geometry metrics and `initialization_comparison.csv`.
Probing uses `--batch`, `--scratch`, `--parent-probes`, `--output` overrides, rather
than the training launcher's `--output` parent convention. Quick parents support
10new training models and11new deployments; full probing requires full oracle data.
Owned servers remain automatic; no research runs launched during implementation.
Full commands/contracts and limitations: `scratch_comparison/README.md`.

