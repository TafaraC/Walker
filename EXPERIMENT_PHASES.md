# Experiment phase ledger

## 2026-10-29 — Second paper-feedback revision

Revised paper and IEEE package with direct RQ3 metrics, matched-model surface
interpretation, finite-budget/convergence qualifications, restored RoboCup and
fidelity sources, and a cautionary probe discussion. Audit corrects the nearby
64500/66500-update comparison; it does not prove exactmatching or solecausation.
Previous paper/package archived. No new experiments or slide changes. Optional
diagnostics remain unrun; user handles publication identity/URL before submission.

## 2026-10-28 — Paper integrates controls; visual evidence prepared

Revised FINAL_PAPER_DRAFT.md and generated a LaTeX/BibTeX package with six
vector figures. Latest focused deck now23slides with chart-based results and a
rendered PDF fallback. Eleven figure assets derive from saved results; source
hashes recorded. Prior prose/deck/notes archived. No new experimental evidence
or retraining. Paper compilation and final eight-body-page layout remain pending;
interpretive controls and research limits are now incorporated in the draft.

## 2026-10-28 — Submission controls complete; presentation revised

SevenBCfits,274newdeployments and25retainedcheckpoint fidelity scores completed.
Source summary/curve output hashes checked. Report:
evaluation/SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md; focusedMarpdeck now22slides,
previous deck/notes archived. Matched-updateBC reverses originalBC/DAgger MSE
ranking;200epoch varied-onlyBC narrows historygap; zero-native control makes no
usefulprogress. Longeroriginal-onlyBC also improves sampled stability/fidelity.
Convergence and broader causal claims remain unresolved; early-stopdiagnostics
and probing of new controls are not completed here. Historical data preserved.
Main final-paper draft needs these qualifications incorporated before submission.

## 2026-10-28 — Submission-control phase implemented

Added submission_controls and run_submission_controls.sh: original BC matched
to211,050updates; varied-only hidden/input BC at200epochs with three seeds;
zero native gait-action baseline; offline fidelity/resource curves from retained
original DAgger/GAIL rounds. Optional early-stopping BC and fixed-aggregate
DAgger diagnostics are separate from the default plan. Existing data/checkpoints
and evaluation results are preserved and reused with hash verification.

Default full batch: seven BC fits and274new60-active-second deployments, with
no new expert-data collection. Sources/methods/metrics are in PIPELINES.md and
submission_controls/README.md. This is a targeted control extension of the paper,
not the optional wider-slip/RL/GAIL-refinement work. Offline engineering checks
do not establish performance; research training and live evaluation remain pending.

## 2026-10-26 — Consolidated evidence framed for final-paper presentation

Slide6 additionally clarifies the historical-to-consolidated BC/DAgger comparison:
new mixed DAgger has lower action error, not a universal performance win; unequal
optimizer exposure and changed protocols prevent an isolated algorithm claim.
Presentation/notes archived before this clarification; no experimental changes.

Probing presentation subsequently focused on interpreting existing behavioural
results: fidelity differences, initialization effects and limited GAIL changes.
No new probe fits or causal claims; prior focused deck and notes archived before
revision. Full representation profiles remain in the original source report.

Created a Marp source and detailed slide companion under
evaluation/presentation/CONSOLIDATED_PAPER_*.md. Covers the standardized restart,
original and frozen-surface comparisons, varied hidden/input training, completed
scratch extension and general/expert probing. The eight-body-page paper narrative
ends with that evidence; optional wide-slip/RL and larger-budget GAIL work remain
separate. Sources are the consolidated results report including Appendix F and
FINAL_PAPER_DRAFT.md. No new experimental results or reruns; historical meeting
summaries and decks preserved. Conversion/layout review remains user-run.
Same-day revision follows the user's focused outline in19slides (0–18); prior
25-slide deck and notes fully archived. The last slide describes research omitted
from the paper but retained for curiosity-driven follow-up (stress/RL, larger-budget
GAIL, R1 transfer), distinguishing implemented pipelines from unestablished outcomes.

## Optional GAIL improvement study implemented — 2026-10-19

User requested a GAIL agent that can improve its initializer, comparing random,
BC and DAgger with more training, and confirmed original environment first.
New standalone gail_experiments pipeline uses consolidated mixed DAgger/BC sources,
350ksteps per actor,three seeds, periodic validation and held-out confirmation.
Revised unanchored noise-matched/R1/guarded-PPO recipe and same-budget legacy option
are implemented without altering completed experiments. No live runs launched.
The paper draft stays on its completed IL results; wide-slip stress is a separate
optional branch and need not precede this original-environment experiment.

## Paper drafting and separate optional stress phase — 2026-10-18

Created evaluation/FINAL_PAPER_DRAFT.md with proposed questions, design,
numerical results, conclusions, limitations and an eight-body-page outline
(references excluded). Completed paper scope remains unchanged. The optional
wide-slip/context-RL curiosity study is separate, not required for publication.
First executable stage implemented: oracle-only scalar sweep .1 through100,
three seeds, owned servers and passive ODE audit. No live results yet. Future
frozen-policy evaluation and wide-range IL/RL training are design-only, pending
identification of expert degradation/failure and plausible task feasibility.

## Scratch extension completed and interpreted — 2026-10-16

Verified 28 additional checkpoints, 929 evaluation artifact receipts and 224
new probe fits (plus 300 reused parent fits). Combined 56 policies / 1,949 runs /
15,720 probe scores. Consolidated report Appendix F now compares scratch with
pretrained continuation, including direct random-start GAIL in original/varied
environments, mechanics, fidelity, context and representations. Scratch supervised
lineages walk successfully but continuation has much lower action error; mixed
DAgger's fidelity advantage persists; direct random GAIL fails locomotion at
this budget. Context benefits remain inconsistent. Prior report archived; no
retraining or new evaluation during interpretation.

## Consolidated report reference guide — 2026-10-16

Added Appendix E to `evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md`:
15 primary paper references mapped to algorithms, metrics, dynamics/context
conditioning and representation probes. Explicitly separates published methods,
project-specific adaptations and historical FiLM work from this batch's results.
The report builder preserves this appendix. No experimental results changed.

## Consolidated standardized-budget study implemented — 2026-10-15

User reopened varied hidden/input training for BC, mixed DAgger and GAIL after
the prior context pilot; this supersedes the previous pause for this new batch.
Original four-model batch is rebuilt with BC170k expert examples, mixed
DAgger100k seed+70k learner labels and GAIL70k PPO steps (user-confirmed), with
170k static GAIL expert data and explicit inherited cost. GAIL initializers are
new BC or new mixed DAgger. Varied stage trains hidden/input versions with same
budgets, three seeds; FiLM excluded. Two launchers own server lifecycle:
run_training.sh and run_evaluation_all.sh. New original-scene, frozen-in-varied
and conditioned varied evaluations share paired baselines and recent33-round
schedule. Implementation/offline synthetic validation complete; no research
training, live smoke or evaluation launched. Details in consolidated/README.md.

## Meeting presentation emphasizes newer results — 2026-10-12

Archived previous Marp deck and refocused the 24-slide presentation on context,
FiLM, repeatability and general/expert probes, including several represented
locomotion features and their controls. Nominal/domain recap reduced to two
slides. User handles conversion. Documentation synthesis only; existing results
and research scope unchanged.

## Full-study meeting slides prepared — 2026-10-11

Added `evaluation/presentation/MEETING_SUMMARY_MARP.md`, a Marp-compatible
presentation of the consolidated nominal/domain/context/probing brief, with
speaker notes and sources. Prior deck/brief preserved. Export commands documented;
rendering unavailable in current shell. Documentation only, no new experiments.

## Full-study meeting consolidation — 2026-10-11

Updated `evaluation/MEETING_SUMMARY_2026-10-10.md` to span the original four-model
nominal comparison through legacy/corrected domain studies, grouped varied
evaluation, context/repeatability/FiLM and expert representations. Separates
unvalidated legacy slip claims from corrected evidence and retains deferred R1
scope. Prior brief archived under `evaluation/meeting_archive/`; future revisions
must archive first. Documentation synthesis only, no new experiment or audit.

## R1 transfer excluded from current scope pending progress — 2026-10-11

User reports a snag in the separate R1 work and defers that aspect until progress
resumes. Keep it as future work rather than a required deliverable or explanation
for the current results. No cause/outcome is established; do not report negative
transfer from this status. Prior transfer plans and checkpoint contracts remain
historical preparation for later use.

The active study retains NAO algorithm/performance/stability comparisons, corrected
domain experiments, context BC/FiLM findings, and broad expert/student probing.
Representation findings concern accessible locomotion properties and geometry on
NAO states, with appropriate controls. Their validity does not require R1 results;
claims about usefulness as R1 initializations remain untested. Next consolidate the
current evidence and identify necessary source-domain follow-ups. No new runs or
pipeline changes authorized or performed by this scope decision.

## Expert-anchored probing completed — 2026-10-10

Verified16 appended fits/geometry plus all parent artifacts; combined2400 scores.
Interpretation saved at nao_expert_comparison_v1/INTERPRETATION.md in representation
experiments; meeting summary updated. Expert resemblance is highest at mixed
DAgger h1 and GAIL-DAgger h2, but random/input CKA is already high. All tested
properties are more linearly decodable in larger student layers, not proof they
know more. Teacher/random comparisons highlight command orientation, phase-roll
and future pitch as candidates. Preserve architecture/causality caveats; next link
hypotheses to separate matched R1 outcomes, not another required source rerun.

## Expert representation comparison implemented — 2026-10-10

Append expert64-tanh hidden layer and matched random controls to general study,
reusing completed student/control probes. Same30 targets, not slip-centred.
Add teacher/student decodability gaps plus label-free CKA/spectral profiles to
preserve broad representation inquiry. Expert output audited against all recorded
oracle actions. No student retraining or experimental reruns. Implementation:
run_expert_probing.sh; methodology/references representation_probing/EXPERT.md.
Research fits remain user-run/pending; no R1 conclusions. Architecture differences
and distinction between retained properties, geometric resemblance and causal
transfer are explicit. Original general results preserved as parent comparison.

## General representation study completed — 2026-10-10

All64 probe fits,1920 target scores and36 source layer-reset diagnostics verified.
Interpretation at experiments/representation_probing/nao_sources_v1/INTERPRETATION.md;
meeting summary updated. Phase-conditioned roll shows the clearest trained-over-
random decoding advantage, across all15 test runs; other profiles differ without
a universal winner. BC/GAIL-BC are similar. Phase/contact block-shuffle controls
retain periodic signal, limiting their null interpretation. Source resets are
functional sensitivity, not causal transfer tests. Next compare to separate R1
matched learning outcomes; no model training/simulation or R1 changes in review.

## General NAO representations pipeline implemented — 2026-10-10

User clarified the scientific target: properties of original NAO models that
contribute to better R1 initialization, not primarily slip decoding. The original
slip-focused probe study remains a preliminary diagnostic; its proposed penalty
sweep is no longer the immediate next task. R1 learning is the utility test and
is performed separately. No transfer usefulness is inferred from source probes.

Implemented isolated representation_probing with30 gait/sensorimotor/future labels,
raw/phase-command/random controls, shared oracle states, whole-run splits and
in-memory layer-reset action sensitivity. Decoder audited against Walk Env; labels
distinguish measured sensor values from support and phase-conditioned proxies.
Existing data sufficient; no new collector or simulator launched. Audit and10
offline tests passed. Next user command ./run_representation_probing.sh run;
research fits pending. Source model reset/fine-tuning effects on R1 remain untested
by this pipeline; checkpoint/activation handoff expectations are documented.

## Frozen-policy probing completed — 2026-10-10

Verified60 fits/300 scores plus original input/source/model plan. Interpretation
saved at experiments/probing/original_policies_v1/INTERPRETATION.md; meeting
summary updated. GAIL-DAgger modestly leads slip decoding; BC/GAIL-BC closely
match. Random features already recover substantial slip/velocity signal; posture
is input-accessible. No demonstrated causal dynamics model or R1 transfer link.
Boundary-selected regularization (74/75 alpha=.01) and surviving shuffled-posture
structure limit strong rankings. Proposed follow-up: versioned regularization
sensitivity; no new research fits launched. Existing context-expansion stop holds.

## Context study closed for expansion; probing implemented — 2026-10-10

User chose not to extend input conditioning/FiLM to DAgger or GAIL: observed BC
benefits are marginal/inconsistent under the current imitation objective. This
supersedes prior automatic extension plans; it is not a result about untested
conditioned DAgger/GAIL. Expert lacks explicit slip but can react via state;
limited motivation to depart from teacher behaviour is a hypothesis, not proof.
Meeting summary: evaluation/MEETING_SUMMARY_2026-10-10.md.

R1 transfer continues in the alternate repository, untouched here. Implemented
offline probing for original BC, mixed DAgger, original-DAgger GAIL (user clarified
the mixed qualifier was a typo) and GAIL-BC. Method: Alain/Bengio linear probes
adapted to ridge regression; Hewitt/Liang motivates controls. Full references in
probing/README.md. Shared oracle states, whole-run splits, random-feature/raw-state
and shuffled-label controls, frozen post-ReLU representations. Read-only real-data
audit passed. Research fits remain user-run; synthetic tests are not findings.
Next: `./run_probing.sh run`, interpret decodability before causal/transfer claims.


## FiLM BC completion and interpretation — 2026-10-10

All 99 correct-context FiLM deployments verified against plan identities,
artifact hashes and result/trace validation. Saved controls reused as planned.
Report: experiments/context_conditioning/bc_film_v1/evaluation_append_all_v1/comparison.md.
Mean progress hidden/input/FiLM: grouped33.07294/33.06027/33.23179m;
confirmation32.29152/32.66458/32.58139m; nominal39.27243/39.17469/39.21891m.
Falls over99 runs per architecture: hidden5,input4,FiLM5. No consistent advantage.
FiLM confirmation gain over hidden is dominated by one +14.075m paired result;
remaining44 pairs average approximately -.0234m (diagnostic, not an exclusion).
FiLM joint-occupancy sliced Wasserstein is worse on both sampled suites; jerk
rates broadly similar. Held-out latent MSE approximately .001805365, only .18%
below hidden. Correct-context-only evaluation cannot isolate context use from
extra capacity. Repeatability variation remains a caveat, not a universal
significance threshold; reused confirmation draws are not untouched FiLM tests.
Next: discuss targeted context ablations/probing before broader training claims.
No pipeline changes, rerun controls or live jobs during this review.

## BC repeatability diagnostic complete — 2026-10-10

Verified45 runs: oracle and seed42 hidden/input, five repeats at each of three
fixed slip/task pairs. Reports at bc_input_v1/repeatability_v1. Progress SD per
case/controller.2405–.6181m; mean pairwise absolute variation.2990–.7426m.
Hidden/input progress differences were -.1556m high slip, +.2077m low slip,
+.0168m nominal (input minus hidden). No consistent progress benefit established.
High-slip fall totals4/2/0 for oracle/hidden/input suggest a possible stability
benefit worth testing, but low-slip totals2/0/1 do not support universal dominance.
Initial targets identical within repeated groups; initial position and head-height
velocity observations vary. This quantifies a limitation, not a startup fix or
proven causal diagnosis. Continue FiLM with those caveats and unchanged protocol
for control reuse; changing startup would require separate matched comparisons.

## Adjacent Unitree R1 RL-baseline/transfer workstream specified — 2026-10-10

User requested an agent handoff for ../unitree_rl_mjlab while Walker context-model
experiments continue independently. UNITREE_R1_TRANSFER_HANDOFF.md now specifies
native R1 scratch PPO first, then original BC/DAgger/GAIL representation/weight
initializations under matched target RL budgets, followed by probing if transfer
differences emerge. Includes verified source inventory and target configuration
inspection; no direct63→16 NAO-to-R1 action compatibility is assumed. This permits
target baseline preparation alongside context work and supersedes older strict
post-NAO sequencing for that preparation. No target implementation, training or
hardware deployment yet. Target experiment artifacts belong in the adjacent repo.

## Repeatability tooling and FiLM BC pipelines ready — 2026-10-10

Implemented optional identical-run repeatability diagnostics (45 default
deployments; unchanged startup, first-state/progress variation reports). Implemented
identity-initialized final-hidden-layer FiLM, matched offline BC training and
evaluation. User clarified that prior hidden BC tests should not be repeated for
FiLM: the comparison now appends only new FiLM checkpoints on the exact saved33
slip/task pairs, reusing verified oracle/hidden/input-correct controls. Default99
new deployments; optional FiLM ablations297. Repeatability remains a separate
optional study. Prior "confirmation" draws are now reused design-informed cases,
not untouched tests for FiLM. Previous429-run identity/results remain unchanged.
No live runs or research training launched. User-run repeatability and/or FiLM
training/smoke are next; no new scientific results yet.

## Context BC full evaluation complete: weak context sensitivity — 2026-10-10

Verified all429 completed deployments and artifacts in
experiments/context_conditioning/bc_input_v1/evaluation_all_v1; report files exist.
Correct-input minus hidden mean progress is -0.01267m grouped, +0.37306m
confirmation, -0.09775m nominal. Confirmation benefit is concentrated: one paired
seed44 comparison contributes12.77843m (~76% of total net gain), at slip.0002849946
below the training minimum, where hidden fell and correct input did not. This is
valid robustness evidence for that run, not broad proof of contextual adaptation.
Wrong/constant contexts show no consistent live-performance penalty.

Nominal correct/constant are identical-input repeat runs but differ by.46318m
mean absolute progress (9 pairs). Initial recorded observations already differ
in the three nominal_0 checkpoint pairs. Evaluation repeatability/startup variance
must be quantified before strong small-effect interpretations. Cause unresolved.
Offline same-state interventions on10k held-out expert observations show all three
input checkpoints respond to context; extreme-wrong raises mean latent expert MSE
~1.47%, constant~.41%. No clear overall locomotion advantage is established.
FiLM is still planned; no further training or live probing launched during review.

## Context BC evaluation tooling ready — 2026-10-09

Implemented isolated owned-server evaluation for all six completed hidden/input
checkpoints plus unchanged oracle. The planned paired grouped, nominal and new
confirmation schedules are supported, with correct/constant/extreme-wrong context
ablations holding physical slip fixed. Full suite has 429 deployments (33×13),
3,000 active steps each; smoke has seven deployments. Runs resume from verified
receipts and generate paired reports automatically. Original results/checkpoints
remain unchanged. This supersedes the pending live-evaluator status below, not
the requirement for user-run smoke validation. No live evaluations launched;
there are no context-conditioned locomotion findings yet. FiLM remains planned.

## Input-conditioned BC offline training completed — 2026-10-09

User completed hidden/input BC at seeds 42/43/44 using CUDA, 100 epochs, batch256,
Adam1e-4. Verified all six checkpoint loads and completion receipts/artifact hashes
under experiments/context_conditioning/bc_input_v1; summary.json includes six runs.
Mean test latent element MSE is 0.0018085948 (hidden) versus 0.0018099182 (input),
approximately 0.073% higher for input conditioning. Individual paired increases
are 0.149%, 0.035%, 0.035%. No practical fidelity improvement is evident here;
this does not establish equivalent locomotion or prove context is unused.
Next: implement controlled live context evaluation, not additional training.
FiLM remains planned; no context-conditioned simulator results exist yet.

## Input-conditioned BC offline implementation milestone — 2026-10-09

Implemented isolated hidden/input BC training with metadata-aware checkpoints,
verified reuse of the existing 100k/10k/10k oracle splits, fixed log1p context
encoding, matched budgets and three paired seeds. Source data and original
checkpoints remain unchanged. Live inference wiring/evaluation and FiLM remain
pending; conditioned DAgger/GAIL will follow the BC pilot. This supersedes only
the "not implemented" status for input-conditioned BC in the approved plan below.
The audit passed on all 40 actual oracle shards across five training slips.
No research training or live experiments started; only synthetic offline tests.
Next: implement paired owned-server evaluation and context ablations before
interpreting model conditioning as a locomotion improvement.

## Phase 4 follow-up planned: input conditioning and FiLM — 2026-10-09

Status: approved research plan, not implemented or launched. This is the current
next-stage plan; older Decision Adapter-first and provisional slip-range notes
below remain historical. Preserve all completed experiments and checkpoints.

### Motivation and questions

The completed 75-run grouped evaluation shows near-oracle mean progress from
context-hidden varied-trained policies: BC 33.292 m versus oracle 33.364 m, with
zero falls for both. Mixed DAgger, GAIL-BC and GAIL-DAgger recorded 1, 1 and 4
falls respectively. These descriptive results motivate a controlled comparison,
not a claim that context is necessary or will improve performance. The full
research interpretation of the grouped results remains to be written.

1. Does providing slip improve imitation/stability beyond varied-surface exposure?
2. Does hidden-feature modulation improve on simply appending slip to the input?

### Three architecture arms

| Arm | State/context interface | Output | Purpose |
|---|---|---|---|
| Context hidden | Existing 63-D observation | 16 latent actions | Matched varied-training control |
| Input conditioned | 63 observations plus one scaled slip value (64-D) | Same 16 actions | Test the value of explicit context |
| FiLM conditioned | 63-D state backbone plus separate scalar-slip branch | Same 16 actions | Test how context enters the network |

For FiLM, start with one hidden layer, using
`h_mod = gamma(c) * h + beta(c)` (element-wise). Initialize the conditioning
output to gamma=1 and beta=0. For input conditioning, copy the pretrained
63-input weights and initialize the extra input column to zero. Verify numerical
agreement with the common starting policy before training. Backbone and
conditioning parameters are trainable; evaluation weights remain frozen.
The downstream walking/IK controller and unchanged FC Portugal oracle remain
identical across arms. These are latent actions, not 22 direct joint commands.

### Staged comparison and controls

- Pilot BC first, then extend the validated comparison to mixed DAgger, GAIL-BC
  and GAIL-DAgger. Hypernetworks and Decision Adapters are deferred, not synonyms
  for FiLM. No per-contact randomization or new impairments in this comparison.
- Retain fixed training grid 0.025/0.1/0.2/1/10 and the long-run/short-turn
  composition. For BC, reuse verified demonstration shards and whole-trajectory
  splits, associating each sample with its verified scene slip. Match starting
  checkpoint, optimizer schedule, update budget and batches across architectures.
  Give the hidden-context control the same additional training; an old checkpoint
  alone is not a matched-budget control for newly trained conditioned policies.
- Propose three paired training seeds for the pilot, distinct from evaluation
  seeds. For DAgger/GAIL match interaction/query budgets and surface schedules,
  not learner trajectories (which necessarily differ across policies).
- Choose and record one context scaling transform before training; use it for
  both conditioned arms without fitting to evaluation outcomes. Record FiLM
  insertion point, branch width, parameter counts and inference/training cost.
  Include a capacity-matched concatenation control if added capacity is material.
- Keep the existing 15 grouped rounds as paired comparison cases: parent means
  at training values, std=0.5*mean, three draws each, truncated to (0,10]. A draw
  stays fixed throughout each 3,000-active-step run. All arms share slip/task
  seeds, using isolated owned servers and verified scalar scene configuration.
  Because these results have already informed design, reserve additional untouched
  draws/task seeds for final confirmation. Check exact nominal slip 0.1 retention.
- Evaluate correct context and predeclared constant/wrong-context ablations with
  physical slip unchanged. Log supplied context separately from physical slip.
  No retraining during these evaluation ablations. Weak sensitivity can indicate
  redundancy or ignored context; modulation values alone are not an explanation.
- Report paired progress differences, falls, tracking, joint smoothness/occupancy
  and held-out latent-action error, by actual slip and Gaussian group. Separate
  training-seed variability from evaluation-round variability. Use oracle ratios,
  not random-normalized scores, unless a matching random baseline is collected.
- Before GAIL extension, specify context for actor and value networks and an
  identical discriminator-conditioning protocol across architectural comparisons;
  pair expert/learner surface distributions so context alone cannot classify them.
- Use a new immutable experiment namespace with architecture/context metadata,
  source/checkpoint hashes and resumable receipts. Preserve historical outputs.
  Implement and update PIPELINES.md before advertising executable commands.
  The user initiates live jobs; this plan does not launch any.

### Interpretation limits

Known simulator slip is privileged context, not an estimated real-world friction
coefficient. A context input does not create online learning or a better teacher.
Near-oracle performance leaves limited improvement headroom; a null result is
informative. High-slip oracle slowdown must not be attributed solely to imitation
error. Three evaluation draws per group do not substitute for training replicas.

### References and their role

- Beukman, M., Jarvis, D., Klein, R., James, S., and Rosman, B. (2023).
  [Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e7b768198d24d883d69704eee57efb0-Paper-Conference.pdf).
  NeurIPS 36. Motivation for comparing hidden context, concatenation and structured
  conditioning under changing dynamics; introduces the distinct Decision Adapter.
  Its RL findings do not establish gains for our imitation-learning setup.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A. (2018).
  [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871).
  AAAI (preprint 2017). Source of feature-wise affine conditioning; evidence is
  from visual reasoning, not proof of improved bipedal locomotion.
- Ha, D., Dai, A., and Le, Q. V. (2016).
  [HyperNetworks](https://arxiv.org/abs/1609.09106). Background for the deferred
  alternative of generating network weights rather than modulating features.

## Grouped varied-arm evaluation completed — 2026-10-09

On-disk results now contain all 75 deployments: 15 paired Gaussian-drawn fixed-slip
rounds, each testing the unchanged oracle and four varied-trained policies (BC,
mixed DAgger, GAIL-BC and GAIL-DAgger). All 750 artifacts referenced by the 75
completion receipts passed an offline hash check. Both `report.md` and
`report.json` exist under
`experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/evaluation_grouped_v1/`;
the JSON report contains all 75 runs. This supersedes the pending-15 status below.
Next is interpretation of overall and per-group results, not another evaluation
run. Performance conclusions and nominal-arm grouped completion have not been
established by this status check. No live jobs were launched during the check.

## Grouped evaluation interrupted at final Gaussian group — 2026-10-09

60 of75 varied-arm deployments completed before first mean10-group scene
preparation failed due to decimal rounding validation. Corrected validation only;
all60 completed receipts/artifacts verified for resume and all15 scene preparations
pass offline. Remaining15 deployments are pending user rerun of the same command.
20 variable-training tests pass; no live runs launched or historical outputs changed.

## Grouped-Gaussian trained-model evaluation ready — 2026-10-09

User approved grouped evaluation: means .025/.1/.2/1/10, std half each mean,
three draws per group, positive slip capped at10. Fifteen paired rounds,75
deployments for varied-trained agents+oracle,135 for both trained arms+oracle.
New evaluation_grouped_v1 directory under the existing training experiment,
per-group reports, unchanged training plan/checkpoints. CLI grouped suite is now
default; previous five-round gaussian suite is retained explicitly.19 offline
tests pass; no live evaluation launched. User will initiate evaluation.

## Five-round trained-policy evaluation extension — 2026-10-09

Expanded Gaussian evaluation from three to five paired rounds, matching training
grid size. Preserved original training plan/checkpoint hashes and first three
draws, adding two draws via a separate evaluation schedule. Varied-only25 runs,
both arms45, oracle included.17 variable-training tests pass; no new live jobs.
All eight existing completed checkpoint hashes and training-plan references verified.

## Grid training and paired Gaussian evaluation ready — 2026-10-09

User clarified scope and asked to finish implementation: train BC/mixed DAgger/
GAIL-BC/GAIL-DAgger on current fixed settings, evaluate trained models on common
Gaussian-drawn constant-slip rounds. Implemented isolated v4 plan with fixed grid
.025/.1/.2/1/10 and evaluation mean5.0125/std2.5 bounded .025..10, three rounds.
Both nominal/grid continuation controls remain available; unchanged oracle is the
reference. Original frozen models are excluded by default; their fixed tests stay
separate. Plan created offline;26 tests pass. No live training/evaluation started.
Current code/documentation supersedes earlier cap-only Gaussian design.

## Expanded slip bounds and three-seed evaluations — 2026-10-09

User requested reducing five evaluation seeds to three and allowing slip up to 10.
Implemented new fixed scalar_wide_v4 (.025/.1/.2/1/10, three paired seeds) and
Gaussian scalar_wide_v3 (bounds .025..10, three paired evaluation draws) defaults.
No old artifacts changed or live jobs started. Gaussian mean/std choice awaits
clarification: defaults remain .1/.035, which do not meaningfully expose high slip.
Single-seed oracle survival at10 does not establish preserved teacher performance;
its measured progress fell substantially. Training budgets remain unchanged.

## Corrected pipeline generation ready — 2026-10-09

User authorized migrating fixed, Gaussian continuation and evaluation pipelines.
Default local scenes now use scalar setters, with new fixed scalar_v3 and Gaussian
gaussian_scalar_v2_seed42 outputs/protocols. Corrected default plan created offline.
Next: user-run corrected fixed and Gaussian frozen baselines; review, then train
matched nominal/varied continuations and evaluate on corrected held-out draws.
Both training arms are corrected; old data/checkpoints remain preserved separately.
No context-aware model or broader training bounds introduced. 23 offline tests and
ten real-config checks passed. No corrected production run started by assistant.

## Scalar method live validated in oracle smoke — 2026-10-09

Five runs at .025/.1/.2/1/10 completed with valid receipt hashes and enabled,
changing solver slip. Progress decreases 43.972 to 15.325 m; no falls. Report:
experiments/slip_sensitivity/oracle_scalar_v1/INTERPRETATION.md. This validates
the new manipulation in one seed, not a training range or policy ranking.
Legacy fixed/Gaussian experiments are preserved but cannot substantiate varied-
effective-slip robustness. Proposed next stage: multi-seed confirmation, then
new-version corrected fixed/Gaussian frozen baselines before learning studies.
Existing training/evaluation launchers were not migrated by this review.

## Alternative scalar scene-setting method ready — 2026-10-09

User requested a new slip modification method. Implemented separate scalar setter
suite, oracle only, five settings .025/.1/.2/1/10 with passive solver logging.
Command: ./run_slip_sensitivity.sh smoke --suite scalar. New results will go under
experiments/slip_sensitivity/oracle_scalar_v1; no prior results changed. Ten offline
tests and real-config generation checks pass. Live validation is user-run/pending;
no further training or domain robustness claim follows until effective slip varies.

## Large-slip oracle smoke completed — handoff review 2026-10-09

Six oracle runs at .1/.5/1/2/5/10 completed; all six receipt artifact hashes pass.
Every setting retains the same recorded solver-slip/mode classes (~.005/.3025),
zero falls, progress 45.109–45.349 m. Existing report:
experiments/slip_sensitivity/oracle_stress_v1/smoke_report.md. Do not expand to
the optional 30-run sweep or context training before diagnosing the configuration
path. Larger scene values did not validate changed effective slip. Historical
datasets remain preserved. No live jobs or code fixes during this review.

## Oracle large-slip stress screen designed — 2026-10-09

Following unchanged solver-slip classes at .025/.1/.2, user requested much larger
settings. Added a separate passive oracle-only screen at .1/.5/1/2/5/10, one seed,
3,000 active steps each. Optional five-seed confirmation is available after review.
Design: contact_slip/STRESS_TEST_DESIGN.md. Seven offline tests pass; no live runs
launched. This investigates the current scene-setting pathway, not a repaired
manipulation or effective-slip override. Prior results remain preserved.

## Passive smoke completed; physics validation unresolved — 2026-10-09

Three oracle smoke runs completed with zero falls and valid artifact hashes.
However, all configured slips produced the same solver-bound slip classes.
Report: experiments/slip_sensitivity/v1/SMOKE_INTERPRETATION.md. Preserve prior
experiments but withhold effective-slip robustness/adaptation conclusions. Next
is diagnosis and a corrected small smoke under a new experiment identity, not
the optional full sensitivity sweep or parameterised training. No new live jobs
or fixes were launched during this review.

## Slip-sensitivity diagnostic pilot — 2026-10-09

User selected effective-physics checks before context-aware training. Implemented
passive ODE contact logging and a separate owned-server oracle/BC pilot: three
slips x five task seeds, 3,000 active steps (30 runs), plus three oracle smoke runs.
Offline checks passed; no live experiment launched. Results will be isolated under
experiments/slip_sensitivity/v1. Prior fixed/Gaussian results remain unchanged.
Context-aware training is not implemented or started; the next decision awaits
contact evidence and paired sensitivity results.

## Cross-phase review — 2026-10-05

Reviewed original nominal, isolated fixed-slip, and per-run Gaussian evaluations.
All eight nominal/varied continuation final checkpoints and 65 Gaussian evaluation
receipts pass artifact hash checks. New retrospective report:
`experiments/variable_slip_training/gaussian_v1_seed42/CROSS_PHASE_INTERPRETATION.md`.
Frozen performance remains similar across tested slips; continued BC/mixed DAgger
improve imitation error under both nominal and varied training. No clear paired
progress advantage from varied training. Proposed next study is a small persistent-
slip context pilot plus controlled fidelity analysis, gated on effective-slip
sensitivity. Per-contact stochastic dynamics remains a separate proposed study;
no live jobs started or new protocol approved by this review.

This file is the authoritative high-level record of the FC Portugal imitation
learning experiments. Update it whenever a phase begins, finishes, changes
scope, or produces a report. Implementation detail belongs in `PIPELINES.md`;
failures and mitigations belong in `CHALLENGES.md`.

## Status summary

| Phase | Name | Status | Primary output |
|---|---|---|---|
| 0 | Pipeline construction and validation | Complete | `PIPELINES.md`, trained policies |
| 1 | Nominal-domain comparison | Complete | `evaluation/PHASE1_RESULTS_REPORT.md` |
| 2 | Parametric domain robustness | Pilot tooling ready | `experiments/phase2_domain_robustness/` |
| 3 | Joint and actuator impairment | Planned | Not started |
| 4 | Adaptation and domain-randomized training | Optional/planned | Not started |

## Phase 0 — Pipeline construction and validation

**Objective:** Build reproducible BC, DAgger, and GAIL pipelines around the
native FC Portugal Walk policy as the expert oracle.

**Completed work:**

- Exposed the 63-dimensional Walk observation and 16-dimensional latent action.
- Implemented synchronous BC collection using alternating long runs and short
  turns, own-half target constraints, fall detection, and controlled recovery.
- Corrected DAgger so learner-visited states are labelled by FC Portugal and
  retained the beta schedule 0.8, 0.6, 0.4, 0.2, 0, 0, 0.
- Stabilized GAIL with normalized discriminator inputs, gradient penalty, label
  smoothing, non-saturating reward, separate optimizers, PPO KL stopping and
  rollback, and a small BC anchor.
- Added resumable five-seed evaluation, metric aggregation, confidence
  intervals, plots, monitor-socket handling, and organized artifact storage.
- Preserved the original GAIL as `gail_dagger` and trained controlled
  `gail_bc` and `gail_random` initialization variants.
- Added `dagger_mixed`, seeded with balanced long-run and short-turn data.

**Exit condition:** Every policy could train and be evaluated through the same
Walk interface without asynchronous simulator drift. Complete.

## Phase 1 — Nominal-domain comparison

**Objective:** Compare the frozen policies in the unmodified FC Portugal domain
under identical targets, seeds, duration, fall handling, and simulator settings.

**Policies evaluated:**

- FC Portugal oracle and uniform-random action baseline
- BC
- original DAgger
- mixed-seed DAgger
- GAIL initialized from DAgger, BC, and random parameters

**Protocol:** Five seeds (101, 202, 303, 404, 505), 3,000 active steps or 60
seconds per policy/seed, synchronous simulation, own-half targets.

**Main findings:**

- BC gave the best nominal imitation fidelity and cost/performance trade-off.
- Mixed DAgger greatly improved over original DAgger on expert-test and
  on-policy action error, supporting the richer-seed hypothesis.
- Viable learned policies achieved returns statistically compatible with oracle
  parity; their intervals did not establish a return winner.
- GAIL retained the main properties of its pretrained initialization.
- Random-start GAIL remained stationary and failed to learn locomotion.
- Low jerk and zero falls can describe stationary failure and must be reported
  with forward progress.

**Outputs:**

- `evaluation/PHASE1_RESULTS_REPORT.md`
- `evaluation/comparison.json` and `evaluation/comparison.csv`
- `evaluation/evaluation_summary.png` and `.pdf`
- `evaluation/presentation/EXPERIMENT_RESULTS_SLIDES.md`: retrospective slides
  prepared on 2026-09-20 with seven reproducible PNG/PDF figures, covering the
  original comparison and completed model variants; future directions excluded.
- Legacy original-three-model report under
  `experiments/archive_20260905_pre_variants/`

**Exit condition:** All eight policy/baseline groups evaluated for all five
seeds and reported. Complete.

## Phase 2 — Parametric domain robustness

Fixed evaluation COMPLETE: all 120 isolated-server runs validated, with 40 per
condition. Report: experiments/phase2_domain_robustness/fixed_report.md;
interpretation: FIXED_RESULTS_INTERPRETATION.md in the same directory. Oracle
and selected policies show little degradation across .025/.100/.200. Retain these
bounds for a pilot; limited performance headroom is a research risk. Gaussian
baseline evaluation and variable-slip training have not yet been completed.

Current protocol revision: user approved per-policy/seed simulator isolation after
recurring shared-server crashes. Prior results archived in
`experiments/phase2_domain_robustness/archive_shared_server_v1_20261004/`.
Recollect all three conditions with `isolated_server_slip_v2`; validated resume
is per run. Original high-traction completion is historical, not part of the new
comparison. Offline checks pass; user live validation is next.

Latest execution status: high-traction fixed block completed. Nominal interrupted
by a simulator SIGABRT after BC seed 505; preserve completed results and archive
the partial nominal block before retrying that condition. Full three-condition
evaluation and report remain incomplete. See CHALLENGES.md for server evidence.

Nominal retry subsequently crashed after mixed DAgger seed 101 with the same
server exception. Evaluation remains incomplete; a consistent server-lifecycle
mitigation should be settled before another full retry.

**Objective:** Measure zero-shot robustness of frozen Phase 1 policies as
controlled physical parameters move away from the nominal simulator domain.
This phase evaluates robustness, not adaptation: no policy is retrained or
fine-tuned during the first sweep.

### Phase 2A — Ground-contact slip

SimSpark's NAO field scene defines field contact with:

```text
setContactSlip 0.1 0.1
```

in `/usr/share/rcssserver3d/rsg/agent/nao/soccer.rsg`. The NAO contact handler
also contains a smaller `setContactSlip 5e-3`. Phase 2A will initially vary only
the field value so the manipulated factor remains unambiguous. In this ODE-style
contact model, larger slip means less tangential traction.

**Pilot conditions:**

| Condition | Field slip | Purpose |
|---|---:|---|
| high traction | 0.025 | Moderate reduction from nominal slip |
| nominal | 0.100 | Phase 1 reference |
| low traction | 0.200 | Moderate increase in slip |

The pilot determines whether conditions are discriminative without making
walking uniformly impossible. If useful, expand to `0.0, 0.05, 0.10, 0.15,
0.20, 0.30`.

Before the full five-seed pilot, run a 1,000-step smoke test of FC Portugal at
each condition. This verifies that the setting is active, recovery still works,
and neither non-nominal condition makes locomotion trivially impossible. These
smoke traces are diagnostic and must not be mixed into the final aggregates.

**Controls:**

- Frozen Phase 1 checkpoints; no learning during evaluation.
- Same five seeds, 3,000 active steps, target sequence logic, robot type, fall
  recovery, synchronous mode, and real-time setting as Phase 1.
- Rerun FC Portugal and random baselines under every slip condition.
- Restart `rcssserver3d` after every server configuration change.
- Save the exact effective server configuration and its checksum with results.
- Restore and verify nominal slip after each experimental block.
- Use a unique output directory per condition; never mix traces across domains.

**Metrics:** Retain every Phase 1 metric. Report both absolute target progress
and within-condition normalized return. Also report nominal retention:

```text
Retention(policy, slip) = J(policy, slip) / J(policy, nominal)
```

The within-condition normalized score answers how close a policy remains to the
oracle under the same domain. Retention answers how much absolute performance
the policy itself loses. Both are needed because re-normalizing against a
degraded oracle can hide severe absolute degradation.

**Pilot entry condition:** Complete. `configure_field_slip.py` backs up, applies,
verifies, fingerprints, and restores the field-slip setting. Evaluation summaries
record the environment identifier and verified slip value. Smoke and full-condition
launchers keep outputs separated by condition.

**Current status:** Tooling ready; no modified-domain result has been collected.
The current verified nominal field slip is 0.1 with SHA-256
`6051bf16e67b9a6bf7449ccd80ab72a0cb2e9d7317cc52ab170e415b332d45c9`.

**Exit condition:** Complete five-seed results for each policy and pilot slip,
with configuration provenance and successful restoration of nominal settings.

### Phase 2B — Actuation and observation uncertainty

After the slip pilot, evaluate continuous severity sweeps separately:

- actuator command-strength scaling,
- actuator speed/range scaling,
- observation or command noise,
- fixed control-command latency.

Change one factor at a time before any combined-condition experiment. Prefer
client-side transformations for command scaling/noise/latency because they can
be seeded and logged per step without modifying system SimSpark files.

## Phase 3 — Joint and actuator impairment

**Objective:** Model collision or wear consequences relevant to RoboCup robots.

Planned impairments, in increasing severity:

1. Reduced range of motion for one joint.
2. Reduced maximum command/strength for one actuator.
3. Intermittent command dropout with a seeded probability.
4. A locked joint at its current or neutral position.
5. A disabled actuator receiving no command.

Begin with symmetric leg-joint categories and test left/right cases separately.
Each impairment needs an intact baseline condition, multiple severity levels,
and an explicit feasibility check using FC Portugal. If the oracle cannot walk
under a condition, learned-policy failure cannot by itself establish inferior
robustness.

## Phase 4 — Adaptation and domain-randomized training

Gaussian frozen baseline complete: 25 paired runs across oracle/four policies
already existed and were validated. Gaussian-only report generated under
experiments/variable_slip_training/gaussian_v1_seed42/. Five extra nominal runs
from the previous schedule are retained but excluded. Metadata-only resume
incompatibility was resolved without rerunning or rewriting the completed runs.

Follow-up evaluation scope corrected: frozen Gaussian baseline only by default,
using five paired draws for the oracle and four selected policies. Repetition of
the completed fixed sweep is not required. Adapted evaluation/report use the same
Gaussian-only selection; fixed/all suites are optional future commands.

Implementation milestone: added independent variable_training package with
truncated-Gaussian plans, isolated servers, per-job resume and four continuation
pipelines (BC, mixed DAgger, GAIL-BC, GAIL-DAgger). FC Portugal remains unchanged.
Default plan is provisional pending current fixed results. Paired nominal controls
and frozen/oracle evaluation are supported. Offline tests pass; no live training
has begun, and the ongoing fixed evaluation code was not altered.

Planning deliverable: VARIABLE_SLIP_TRAINING.md specifies server-controlled
BC/DAgger/GAIL continuation, matched nominal controls and held-out evaluation.
User reports isolated fixed experiments are running. Provisional truncated Gaussian
bounds [0.025, 0.200] will be reviewed after those results; parent mean 0.100 and
std 0.035 are proposed placeholders. The write-up is not executable implementation.

Execution-order update: retain phase numbers as historical identifiers, but run
the NAO varied-slip training study immediately after Phase 2A frozen evaluation,
before actuator impairments or Unitree R1 EDU transfer.

Agreed sequence:

1. Finish fixed frozen-policy evaluations at 0.025, 0.100 and 0.200.
2. Select a truncated Gaussian from those results and evaluate frozen policies
   with one draw per run held constant, paired across all policies and baselines.
   Check oracle viability. Current code still samples uniformly; updating the
   sampler and manifest validation is pending before this stage. Record parent
   mean/std, truncation bounds, effective distribution variance and exact draws;
   clipping Gaussian draws is not equivalent to sampling a truncated Gaussian.
3. Pilot BC training with slip hidden in the existing observation, collecting
   demonstrations over the selected distribution. Extend to DAgger learner-state
   labeling, then GAIL with matched expert/learner surface distributions.
4. Compare against frozen checkpoints and additional nominal-only training with
   matched budgets. Use held-out slip draws and retained nominal performance;
   record inherited supervision and multiple training seeds where feasible.
5. Run the input-conditioned and FiLM comparison specified in the 2026-10-09
   follow-up above, with a matched hidden-context control. Pilot BC before mixed
   DAgger/GAIL. Decision Adapters remain deferred. Keep this distinct from the
   completed hidden-slip training study.

The two questions are whether exposure to varied dynamics improves robustness,
and whether knowing slip adds benefit beyond that exposure. Distribution parameters
remain unset until fixed results are analysed. Unitree R1 EDU work follows the NAO
varied-slip study.

**Objective:** Only after frozen-policy robustness is known, test whether
retraining or fine-tuning across randomized domains improves robustness.

Keep these results separate from Phase 2 zero-shot evaluation. Candidate studies
include domain-randomized BC/DAgger data, GAIL rollouts across slip values, and
post-impairment fine-tuning. Compare sample cost and retained nominal performance
as well as modified-domain performance.

## General experiment rules

- Never overwrite a completed phase; archive it under `experiments/`.
- Use explicit experiment identifiers and configuration manifests.
- Evaluate random and FC Portugal baselines within every modified domain.
- Keep training seeds distinct from evaluation seeds.
- Do not interpret fall rate, jerk, or torso stability without locomotion return.
- Report marginal algorithm cost and inherited end-to-end supervision cost.
- Change one environmental factor at a time before testing interactions.

### 2026-09-25: Phase 2A human-run local-scene implementation

Scope remains frozen-policy testing, with fixed slips 0.025/0.100/0.200, oracle
smoke tests (1,000 active steps), then eight controllers × five seeds × 3,000 steps
per condition. Added workspace-local scene pipelines, user review gates, reports
and a reproducible sampled plan selected only after fixed results. Offline checks
passed; no live results or sampled bounds yet. User launches experiments manually.
See `experiments/phase2_domain_robustness/README.md` for the operational guide.
Next: NAO varied-slip learning/testing, then Unitree R1 EDU compatibility assessment,
feasible transfer tests and a target-trained RL comparison; these are not underway.

### 2026-09-25: Preserve original testing protocol

User clarified that only slip should change. Restored the original evaluator and
preserved original launchers plus baseline snapshots. Varied-slip wrappers now use
the original full evaluation launcher and condition-level server lifecycle; removed
extra startup stabilization and forced simulator seeding before live data collection.
No live experiment has run; user execution/review checkpoints remain in force.

### 2026-10-04: First live smoke startup defect

The first high-traction smoke attempt exposed an unreachable readiness marker in
the generated server configuration. SimSpark started, but the marker followed its
non-returning `run` call, so the wrapper timed out before evaluation. The marker
was moved before `run` and its ordering is now tested. No evaluation result was
produced; Phase 2A remains at the smoke-validation stage pending a user rerun.

The next retry also failed before evaluation. The marker-order explanation above
was incomplete and its non-returning `run` claim is retracted. The wrapper now
selects its configuration with the documented `--script-path` option and emits
the marker after local NAO setup. Phase 2A still awaits a successful user smoke
run; no robustness conclusions can be drawn from these startup failures.

User-requested validation extension: oracle smoke runs increased from 1,000 to
3,000 active steps per condition; full runs remain 3,000. Added server-side scene
hash/import checks and per-launch provenance validation. No new live results yet.

The user subsequently reported successful smoke validation. Phase 2A is ready
for full fixed-condition evaluation: eight controllers, five seeds, three slip
conditions, 3,000 active steps per run (120 evaluation runs). Next commands are
`./run_varied_slip.sh fixed` and, after completion, `./run_varied_slip.sh report-fixed`.
Full results have not yet been reported or reviewed.

User clarified execution workflow: human checkpoints mean manually initiating
scripts. Removed internal approval prompts for both fixed and sampled evaluation;
each invocation now processes all requested blocks automatically unless an error
occurs. Experimental settings, smoke runs and result protection are unchanged.
## 2026-10-15 — Consolidated demonstration tooling

Added managed model/slip viewer for historical and completed consolidated policies,
including input-conditioned variants. This supports visual checks, not a new
evaluation cohort or performance claim. Smoke training/evaluation completed with
12 models and20deployments; full runs remain user-launched. Live RoboViz verification
of the new viewer is pending. Commands and server contracts are in PIPELINES.md.

## 2026-10-15 — Consolidated evaluation complete; general probes ready

User reports full evaluation complete and comparison artifacts now exist in
`experiments/consolidated/v1/evaluation/all_all_correct`. Next phase probes all
28new actors (original/varied hidden/varied input), retaining replica identities,
and compares the unchanged expert. New offline entrypoint
`run_consolidated_probing.sh` reuses30locomotion labels and shared oracle traces,
not a slip-centric study. Real-input audit passed10/5/15run splits; research fits
remain user-launched. Historical four-model studies remain separate; no new
performance or causal interpretation is claimed before examining these reports.

## 2026-10-15 — Consolidated results and general probes interpreted

Completed staged report:
`evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md`. Covers experimental
design, original-scene results, frozen-policy sampled-slip tests, varied hidden/input
continuations, a dedicated DAgger analysis, resource accounting and general/expert
probes. Inputs:1,020deployments,28models,300probe fits/9,000target scores;
all300probe artifact receipts verified. Task progress is largely oracle-level;
fidelity and falls distinguish models more clearly. DAgger is a strong fidelity
baseline; explicit context has small inconsistent behavioural benefits. GAIL's
measured representations remain close to its initializer. These are descriptive
findings, not R1transfer or causal-feature claims. Historical summaries/reports
were preserved. See the new report for numeric tables and design limitations.

## 2026-10-15 — Scratch initialization comparison added

User requested from-scratch models in current evaluation/probing without losing
or repeating prior tests. Scope:24varied scratch-lineage policies (BC/mixed DAgger
and their GAIL initializers, hidden/input,3seeds), plus direct random-GAIL in original
and varied-hidden environments (1+3replicas),28new checkpoints total. Existing
28policies remain. Same expert data, incremental budgets and evaluation draws.
New owned-server training/evaluation and offline probe append routes implemented
via `--include-scratch`; live scratch experiments not launched by agent.
Expected929new deployments reuse1,020old;224new probe fits reuse300old. Outputs
under `experiments/consolidated/v1/scratch_v1/`; previous staged report remains a
parent-batch report, not evidence for the new conditions. See PIPELINES.md and
scratch_comparison/README.md. Parent data and1,020evaluation receipts audited.

