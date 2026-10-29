# Imitation-Learning Challenges and Engineering Decisions

## 2026-10-29: Near-budget curve points are not causal identification

Secondfeedback suggested the initialBCpoint matched DAgger's updatecount and
proved the wholefidelitygap was fitting alone. Artifactaudit shows66500BCupdates
versus64500DAggerupdates (MSE0.11047/0.10995). Near-equality supports optimization
sensitivity but is neither exactmatching nor causal attribution of the entiregap.
MatchedfinalBC has lower on-policyerror, but eachpolicy visits differentstates.
Updatedpaper qualifies this, centres sampledmatchedBC, and explicitly preserves
unresolvedconvergence. The directionerror metric is dimensionless unit-vector
distance, not radians or spatialtrajectoryerror; RQ3table now states this.
Existingprobe cohort is variedcontinuation, not the originalmatchedcontrolpair;
avoid claiming an observed within-probed-cohort ranking reversal. No newruns.

## 2026-10-28: Visual reporting and publication-layout validation

Finaldraft now integrates the optimizationcontrols rather than retaining the
oldalgorithm-only fidelity conclusion. Figures distinguish taskrounds from
trainingreplicas, label logarithmic/zoomedaxes and avoid invented significance
or confidenceintervals. RawJSONhashes recorded in presentation/figures_final.
NewBCcontrols remain unprobed; oldreadouts cannot explain their improvement.

NoMarp/browser orTeXcompiler was installed for this revision. A measured
MatplotlibPDFfallback renders the23-slide source; pagebounds and representative
previews checked. The LaTeXpackage has checked citations/assets but still needs
actual compilation and eightbodypage verification in the submissiontemplate.
This limitation concerns publicationlayout, not completed experimental results.

## 2026-10-28: Optimization controls change the fidelity interpretation

Completed submissioncontrols confirm the optimization confound mattered:
originalBC matched to211050updates reaches0.01024testMSE versus oldBC0.11047
and mixedDAgger0.01535. The earlier ranking cannot support an algorithm-only
advantage. Varied-only200epochBC reaches0.04210hidden/0.03775input versus
continuation0.01995/0.01986: exposure explains part, not all, of the observedgap.
Original-only matchedBC also removes recorded falls across30sampled minutes;
do not attribute all prior continuation gains to surface diversity.

Zero nativeactions do not achieve meaningful progress, addressing that specific
gait-engine confound while leaving the shared16Dgait/IK interface limitation.
All7fits exhaust budgets with selectedepochs near theend; convergence remains
unproven. Newcontrols are not represented by oldprobe results. See
evaluation/SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md and updated presentation;
historical results retained. Mainpaper interpretation update remains pending.

## 2026-10-28: Onboarding and runtime portability boundaries

The setup-workbook audit found no complete checked-in Python environment lockfile.
Documented package versions are a local snapshot, not a tested portable install.
Use reviewed Conda/pip exports from the working environment for reproduction;
do not infer a sufficient environment from a short import list. Python3.9 is a
legacy unsupported series and should not be silently upgraded within old receipts.

Launchers use a machine-specific interpreter fallback; PYTHON_BIN provides an
override. An absolute interpreter does not reproduce all activation settings:
Script.build_cpp_modules resolves python3.x by PATH. Managed consolidated server
paths are hard-coded to /usr/share/rcssserver3d even though the preflight checker
also searches /usr/local/share. These distinctions are now explicit in
docs/learning/EXPERIMENT_SETUP_WORKBOOK.{md,pdf}. No portability code, environment
installation, binary rebuild or historical identity was changed for the guide.

## 2026-10-28: Submission controls implemented, research evidence pending

New isolated submission_controls package/run_submission_controls.sh addresses
the audit below without mutating old sources or checkpoint identities. Exact
update stopping handles the partial epoch; validation selection and endpoint
weights are separately saved. Seven default BC controls match original DAgger
updates or varied-continuation lifetime epoch allocation. Optional early-stop
BC/fixed-aggregate DAgger diagnostics retain the distinction between optimizer
stopping and iterative algorithm convergence. No new GAIL training is included.

Zero actions are implemented as a checked zero-output standard policy through
the unchanged evaluator, preserving gait/IK/filter/reset semantics. Full reports
reuse verified3000-step reference runs. Smoke must not reuse those horizons:
it collects its own32-step oracle/random controls, and reporting rejects any
active-step mismatch. Old/new rows retain provenance and training-seed identities.

Mitigations are implemented, not established research outcomes. Offline tests
and real-parent data audits are available; live simulation and full training
have not been launched by the agent. New results must be reviewed before updating
the final report's conclusions. Interrupted jobs are preserved/restarted rather
than pretending best-checkpoint optimizer state is an exact mid-run continuation.

## 2026-10-28: Submission audit — BC budget stops before a demonstrated plateau

Follow-up DAgger audit: original final iteration7 selects epoch50/50, with
validation MSE0.0094085 at epoch40 and0.0089467 at50. All six scratch varied
final fits select epochs47–50; varied continuation best epochs range12–50.
Original DAgger convergence is therefore also unestablished. Early-stopping BC
alone cannot establish a comparison between two converged methods. A final
DAgger-dataset refit would address optimizer convergence on that fixed aggregate,
not convergence of the entire iterative state-collection process.

Exact update matching: BC has ceil(170000/256)=665updates per epoch. Matching
211050updates requires317full epochs plus245minibatches, not exactly317epochs.
Keep the partial final batch and validation-selection rules explicit; equal update
count is not equal total training cost because DAgger also queries/collects data.

Action audit: consolidated BC/DAgger labels and outputs are not dataset-mean/std
standardized. load_shards/load_dataset retain native16-D gait actions and the
collector passes output directly to walk.execute_action. Env.execute applies
distance-dependent scaling, factor0.7 and EMA(0.8old+0.2new), then physical
offset scales with retained gait/IK and arm swing. Zero native action is distinct
from a constant expert-mean action. Zero control must reset action-filter history
through normal reset handling; it is not zero joint commands or frozen gait.

Read-only review of consolidated BC training logs confirms continued validation
improvement at the 100-epoch budget. Original hidden seed42 validation MSE falls
from0.1107255 at epoch90 to0.0981507 at epoch100; best epoch100. Across all13BC
jobs inspected (original, six varied continuations, six varied-only), the best
epoch is99or100. This is evidence against claiming demonstrated convergence,
not proof that additional BC optimization will eliminate DAgger's fidelity gap.
Source: experiments/consolidated/v1/policies/**/bc/iteration_000/update/worker.log
and corresponding scratch_v1/policies logs/metrics.

Risk: DAgger's54million supervised exposures versus BC's17million confound an
algorithm-only interpretation. Continuation also has extra lifetime training.
Proposed mitigation (not implemented/launched): separate matched-exposure BC
control (~318total epochs on170k examples, or an explicitly matched update budget)
and varied-only BC with200total epochs against100original+100varied continuation.
Select checkpoints using validation only; preserve existing checkpoints/results.
Report both finite-budget findings and control outcomes, rather than overwrite.

Other audit qualifications: on-policy versus held-out MSE reversal is consistent
with distribution shift, not proof of its mechanism or of no drift originally.
Conservative anchored GAIL remains adversarial imitation, but is not an unrestricted
test of GAIL's potential; the original GAIL paper also uses KL-constrained updates.
Latent-action MSE and joint occupancy SWD should be reported together without
equating occupancy similarity to matched-state action-to-joint sensitivity.
A zero-latent/constant-action control is proposed to measure the gait machinery's
contribution; it has not been run. Original DAgger iterations0–7 and both GAIL
iterations1–7 checkpoint files are present for retrospective fidelity curves;
those are not return-convergence curves and would not erase unequal costs.

No training or evaluator changes during this audit. Remaining controls require
a separately recorded experiment, not silent changes to the completed batch.

## 2026-10-19: Testing GAIL adaptation rather than preservation

Current stabilized GAIL often remains close to its initializer. Plausible limits
include conservative actor updates, persistent BC anchoring and a discriminator
shortcut between deterministic expert actions and noisy policy samples. These
are hypotheses, not established causes. New isolated gail_experiments package
tests unanchored training, matched expert action noise, R1 regularization,
positive occupancy reward, exact-KL epoch retries and a350kstep budget.
Reward sign changes termination incentives; smoothing expert actions changes the
matching target. Neither modification is claimed universally correct. Exact-KL
guards constrain rollout-distribution change, not physical safety or evaluation
performance. Random starts can still fail; expert-level initializers leave little
room for task-return gains. All changes remain outside the completed paper.

Mitigation: optional same-budget legacy trainer, explicit iteration-zero baseline,
validation-only selection (initializer eligible), separate confirmation seeds,
final and selected reporting, and fidelity/stability alongside progress. CPU
synthetic updates and resume/rejection tests pass; real parent data/source-model
audit passes. Live quick/full experiments remain user-run and untested here.

## 2026-10-18: Wider slip is an exploratory intervention, not a training assumption

New optional sweep extends to100 but does not assume the task stays feasible,
that an upright expert is succeeding, or that failure is recoverable through RL.
Uses corrected scalar scene setters plus passive solver evidence; reports
absolute progress, speed, tracking, targets and falls, with paired scalar-.1
retention rather than oracle/random normalization near expert failure. No
automatic physical-failure threshold or training bounds. Contact logs include
non-foot/startup/recovery contacts and may be large. Native/interface constraints
and reward-design risk remain for later RL work. Existing paper results preserved.

## 2026-10-16: Scratch results separate locomotion, fidelity and representation

Scratch BC/GAIL-BC gain roughly 0.4m mean progress while action fidelity is about
5.6x worse than continuation. Scratch DAgger can have higher expert CKA despite
worse action error. Direct random GAIL fails useful locomotion yet admits
informative oracle-state probes; low jerk or zero original-scene falls alone
would conceal failure. Appendix F presents these metrics separately. Do not
infer causal feature utility or R1 transfer from CKA/readouts. Direct random
GAIL retains supervised anchoring and conservative 70k-step training, so failure
is configuration-specific. Incremental budgets do not match lifetime optimization;
reused controls are not new independent trials. These limitations remain unresolved.

## 2026-10-16: Literature attribution boundaries in consolidated reporting

Risk: generic or incomplete proposal citations can imply that papers validate
our exact robotics metrics or establish transfer from probe decodability.
Mitigation: the consolidated report's Appendix E now maps verified primary
papers to methods and states adaptations explicitly. In particular, sliced
joint occupancy is not exact multivariate transport; joint jerk is not torque
or energy; CKA is not percentage knowledge retained; regression probes do not
demonstrate causal use or R1 transfer. Native oracle release provenance remains
code/checkpoint-specific rather than attributed to an unverified team paper.

## 2026-10-11: R1 dependency removed from current study

User reports difficulties with the separate R1 transfer work and defers its
inclusion pending progress. Technical cause and target-learning outcome are not
provided, so do not infer failed transfer or inadequate NAO representations.
Mitigation: scope current reports around completed NAO performance/domain/context
experiments and descriptive expert/student representation analysis. Keep transfer
utility as an open future question rather than a dependency for reporting source
findings. Existing caveats on decoding, CKA and source-reset interpretation remain.
No target code changes, experiment deletion or process cancellation were requested.

## 2026-10-10: expert score gaps and high random similarity

Completed comparison gives positive student-expert R2 gaps in240/240 cases;
64-tanh versus256-ReLU differences preclude interpreting this as more knowledge
or superior control. Expert itself trails its random controls on many measurement
targets; compact task-specialized features need not reconstruct all inputs.
Raw/random CKA around.85–.90 is already high, so learned CKA .905–.936 is not
90% knowledge transfer. Dataset-conditioned constant units/participation ratios
are not dead-neuron or skill-count proofs. These caveats are in the interpretation
and meeting summary. No corrective code change or artifact rewriting performed.

## 2026-10-10: expert architecture and retention interpretation

Expert audit found one64-unit tanh hidden layer, versus students' two256-unit
ReLU layers. Added width/activation-matched expert random controls; compare each
student layer to the single expert layer without inventing expert h2. Decoding
gaps reflect architecture/capacity as well as training; error ratios are not
fractions of knowledge transferred. CKA/spectra broaden exploration but do not
enumerate all features or establish causal inheritance/R1 benefit. Probe only
expert neural state, not the entire external generator/smoothing/IK system.
Audit reproduces source oracle actions before fitting. New append-only module
preserves completed fingerprints. A synthetic empty-comparison report edge case
was caught and fixed before handoff; no historical results affected.

## 2026-10-10: periodic null controls and reset interpretation

Completed general probes show whole-trajectory shuffles preserve regular gait
phase: BC h1 shuffled phase sin/cos R2 about .819/.870. Future force-contact proxy
shuffles similarly retain signal. These are not valid chance-level nulls for
periodic labels. Phase-conditioned roll shuffles are much lower, strengthening
that particular descriptive finding. Future work could use episode-contained
phase shifts or phase-matched residual controls, in a new version, not modify
completed results. No control-code change was made during this review.

Persistence of instantaneous values is weak for averaged/oscillatory future
targets; negative R2 is not proof learned probes anticipate dynamics. Layer-reset
action RMS is unnormalized and reflects interface damage as well as knowledge:
do not infer R1 usefulness or semantic localization.89/480 true readouts still hit
minimum alpha1e-6. These limitations are recorded in the interpretation/meeting
summary before target-transfer conclusions; original models and data preserved.

## 2026-10-10: corrected probing scope and source/transfer distinction

User wanted general properties useful as R1 initialization, not a slip-centred
study. Preserved old diagnostic and built separate general representation pipeline.
Available observations support phase, force, hip-relative ankle, command and
posture labels; no extra sensor measurements are invented. Generator phase is
not measured stance, force>10N is a proxy, phase-posture products are constructed
interactions and not stability margins. Commands are displacements, not velocities.
Future labels describe oracle continuation, not candidate-policy outcomes.

Stride7 reduces aliasing with the nominal8-step half-cycle. Drop reset observations
and recovery-crossing windows. Persistence uses past completed transition, never
future input. Weaker ridge grid addresses previous lower-bound hits without editing
historical results. Random/block-shuffle and phase-command controls remain vital.
Resetting a layer on an in-memory model measures source action sensitivity only;
interface disruption can dominate. R1 matched learning/ablation outcomes, not
source readout rankings, establish initialization utility. No R1 code was changed.

## 2026-10-10: completed probes expose control and penalty limits

Review of60 verified fits found74/75 true-label readouts select the lowest tested
ridge penalty .01. Not a solver failure, but weaker regularization is untested;
small rankings and layer-depth effects need validation-selected sensitivity in
a new output version, not changes to completed artifacts or test-driven tuning.
Random networks already decode considerable slip/velocity signal. Block-label
shuffling preserves some posture signal (raw head-height R2 .318–.353), so these
controls are not zero-information nulls; shared gait/time structure is plausible,
not proven. Record these limitations before learned-dynamics/transfer claims.
No pipeline changes in this review; report: probing experiment INTERPRETATION.md.

## 2026-10-10: probing confounds and scope control

Context BC/FiLM yielded no consistent practical advantage; user stopped planned
context DAgger/GAIL expansion. The oracle lacks explicit slip but can implicitly
adapt, so do not assert that context can never improve imitation. Meeting summary
records evidence versus mechanism hypotheses. R1 is a separate ongoing study.

New probes use shared oracle trajectories to avoid conflating model representation
with on-policy occupancy. Whole-run splits and recovery-window filtering prevent
adjacent-state leakage/reset artefacts; train-only scaling and validation-only
penalty selection isolate test fitting. Raw inputs, random features and block-label
controls guard against readout capacity and trivial posture reconstruction.
Slip remains confounded with run/task/kinematics; posture is directly observable;
future velocity describes oracle continuation. Historical confirmation cases are
held out from probe fitting but were already examined for performance. Three
random seeds are controls, not trained-policy replication. Decoding is neither
causal use nor evidence of R1 transfer; differing source budgets/lineages remain.
GAIL-DAgger lineage was explicitly clarified as original DAgger, not mixed.
No silent checkpoint substitution or historical source/receipt edits were made.


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

## 2026-10-10: repeatability variation confirmed, mechanism still open

45 verified diagnostic runs show same-policy progress SD.2405–.6181m and mean
pairwise absolute differences.2990–.7426m. Initial target equality was verified
for all repeated groups; first-active positions differ up to24.12mm, and the
largest observation spread is loc_head_z_vel/2. Startup/settling state variation
is directly observed, but its cause and contribution to total outcome variation
remain unproven. These small samples (one trained seed pair, three cases) do not
provide a universal noise threshold or a statistical equivalence result. Preserve
all data and do not remove falls/outliers. Keep protocol unchanged for planned
FiLM appending; if startup is corrected, use newly matched controls rather than
silently mixing protocols. Diagnostic complete; deterministic startup unresolved.

## 2026-10-10: planned R1 transfer has incompatible policy interfaces

Inspected adjacent Unitree R1 task uses native normalized observations, joint-
position actions and512/256/128 ELU PPO networks. Original Walker actors are
63→256→256→16 ReLU, with latent ankle/foot/arm outputs decoded by stateful NAO
step generation and IK. Matching tensor length or nominal50Hz is not semantic
compatibility. The transfer handoff requires a native scratch baseline plus a
matched-architecture scratch control and explicit tensor/adapter mapping. Source
GAIL critics/exploration, ODE slip values and NAO joint metrics cannot silently
carry over to target rewards/actions/friction/morphology. Any benefits remain
unproven; preserve source/initial/intermediate/final states for later controlled
probing. See UNITREE_R1_TRANSFER_HANDOFF.md; no target changes made.

## 2026-10-10: repeatability isolation and FiLM control reuse

Added diagnostic repeats without modifying startup so the source of observed
variation is not assumed or silently removed. First-active states and progress
spread are reported; no causal diagnosis or deterministic-start fix yet. FiLM
uses a separate versioned package, preserving historical source identities.
At user request, evaluation appends only FiLM on exact existing cases, avoiding
redundant hidden/input/oracle deployments. Strict source/environment compatibility
and budget/data/initializer matching guard that reuse. Startup differences and
elapsed-time effects are still limitations of comparing noncontemporaneous runs.
If a startup fix changes the protocol, old controls cannot silently be reused.
Prior confirmation draws have informed FiLM design and are not an untouched final
test. FiLM adds1024 parameters vs256 for concatenation; this pilot is not exactly
capacity matched. These risks must accompany any improvement claim.

## 2026-10-10: identical-context evaluation variability — open research risk

Completed BC nominal runs expose a useful implicit repeatability control: correct
and constant modes both supply.1 to the same checkpoint at the same physical slip
and task seed. Their9 paired progress differences have mean absolute.46318m,
signed range -1.36790..0.67288m. First observations already differ for nominal_0
at all three training seeds (maxabs.17860,.09129,.06112; initial planar position
maxabs.00488,.00433,.00330m). Thus identical seeds/context do not ensure identical
initial states. Startup/network/simulator mechanisms are not yet diagnosed;
do not claim a proven root cause, deterministic repeats or invalidation of all runs.
Quantify same-policy repeatability and inspect startup state control before
attributing small locomotion differences to conditioning. Preserve all results.

Correct-input confirmation mean progress improves.37306m over hidden, but~76% of
the net gain comes from one seed44 low-slip hidden fall. The sensitivity mean
without that single pair is.09112m (diagnostic only; do not remove it from reports).
Wrong context does not consistently hurt live progress. Counterfactual offline
inference on fixed expert states does change actions and modestly worsens MSE,
so "the policy ignores context" is also unsupported. Keep state-distribution,
context-intervention and run-to-run variation separate in interpretation.

## Context BC replication interpretation — clarified

Three training seeds per architecture expanded the former one-checkpoint workflow
and evaluation count; this needed clearer explanation. Seeds chiefly vary batch
ordering over a shared pretrained initializer and identical demonstration splits.
They are fine-tuning repetitions, not independent from-scratch training replicas.
The new README seed section explains matching across architectures, separate task
and schedule seeds, CUDA determinism limits, and105/429-run accounting. A one-pair
45-run option is only a proposal; no automatic best-seed selection or default
change was made. Preserve all checkpoints and avoid overstating independence.

## 2026-10-09: BC context evaluation integration — implemented, live check pending

New isolated context evaluator loads the metadata-bearing BC checkpoints without
altering the legacy 63-input evaluator or training identities. Explicit supplied
context is logged separately from verified physical slip; wrong/constant tests
do not change the server scene. Owned servers, a shared pipeline lock, bounded
retry, failed-attempt archiving and strict receipt checks mitigate previous
disconnect/teardown failures. Final-action falls are checked consistently across
new controllers; this differs from older legacy results. Sliced Wasserstein and
smoothed angular jerk retain their approximation/interpretation caveats. Confirmation
draws are predeclared separately from previously inspected grouped draws. User-run
smoke remains necessary: mocked lifecycle tests do not validate live physics.

## 2026-10-09: context checkpoint/data compatibility — offline mitigation implemented

Legacy BC models require 63 observations, and reused datasets contain no appended
context. The isolated context_training package now attaches slip from verified
scalar-scene metadata, checks historical receipt integrity and whole-run splits,
and uses a distinct metadata-bearing checkpoint schema. The 64-input first layer
copies old weights plus a zero context column to preserve starting behaviour.
A matched hidden arm controls additional supervision/update budget. Context uses
a fixed log1p transform on (0,10], avoiding fitting normalization to test draws.
Existing collector/evaluator sources remain untouched to preserve old experiment
identities. Legacy evaluation cannot load these checkpoints; live integration is
explicitly pending. Held-out latent MSE must not be presented as locomotion evidence.

## 2026-10-09: context-conditioning comparison risks — planned mitigations

Input-conditioned and FiLM models are planned, not implemented. Near-oracle
context-hidden performance means gains may be small or absent. Additional training
or FiLM parameters could masquerade as a context benefit: use a matched-budget
hidden-context control, common warm start, paired training seeds, and capacity
accounting/control. Keep slip scaling identical between conditioned architectures.
Known simulator slip is privileged information, not measured physical friction;
correct/constant/wrong-context tests should hold the actual environment fixed.
Existing grouped outcomes informed this design, so reserve untouched confirmation
draws instead of presenting the reused rounds as an untouched final test.
For GAIL, balance expert/learner surface distributions to prevent discriminator
shortcuts based on context. Feature modulation alone does not establish causal
internal representations. Full protocol and references: EXPERIMENT_PHASES.md.

## 2026-10-09: grouped draw rejected due to serialization precision — resolved

group_4_draw_0 requested5.01334232429452; scene stored5.01334232429 using .12g.
Absolute difference4.52e-12 exceeded status()'s1e-12 check. Failure occurred in
preparation, before server launch, after60 completed runs; it is not a policy fall
or a slip-setter failure. Validation now recognizes the writer's exact decimal
encoding without loosening arbitrary differences. Only audited old/new validator
hashes are treated as equivalent for evaluation resume; all other provenance and
artifact checks remain strict.60 receipts verified;20 tests pass. No data deleted.

## 2026-10-09: grouped truncation and evaluation isolation

User approved five evaluation Gaussian groups centred on training settings,
std=.5*mean, three draws each, positive values capped at10. At the upper bound the
group is asymmetric and its sampled mean will be below its parent mean10. Report
labels parent moments explicitly and provides per-group comparisons rather than
only a pooled average. Three draws remain a pilot-sized sample. Grouped evaluation
uses its own directory/schedule and does not change the training plan or overwrite
prior five-round results. The scalar environment method remains modifiable;
each evaluation deliberately holds it fixed.19 offline tests pass, live run pending.

## 2026-10-09: extend evaluation without invalidating completed training

Changing the embedded evaluation list in plan.json would change checkpoint plan
provenance even though training itself is unchanged. Five-round evaluation now
uses a separately persisted schedule derived from the same seeded Gaussian stream,
with exact original-prefix verification. The immutable training plan, completed
checkpoints and existing evaluation receipts are preserved.17 offline tests pass;
no relaxation of per-run artifact hashes or retraining is required.

## 2026-10-09: distinguish fixed-grid training from Gaussian generalization tests

User clarified fixed-grid training and Gaussian-drawn fixed evaluation surfaces.
Implemented mean5.0125/std2.5 bounded .025..10 for evaluation ONLY; training stays
.025/.1/.2/1/10. Every model shares the round's slip/task seed. Default comparisons
exclude original frozen policies and retain oracle and requested trained arms.
The three seed42 draws fall between1 and10, a sparsely covered training interval:
interpret results as generalization to unseen surfaces, not matched-distribution
performance. Three paired rounds do not quantify independent training-seed variance.
Zero falls in one extreme-slip oracle run does not establish reliable teaching.
The scalar setter is modifiable and live-validated; fixed-per-run sampling is an
experimental design choice, not proof that the environment cannot be modified.

## 2026-10-09: expanded bounds versus distribution coverage

User authorized upper slip10 and three evaluation seeds. Fixed tests now explicitly
include1/10. Merely increasing a Gaussian cap from .2 to10 while keeping mean .1,
std .035 will not meaningfully sample severe slip. Asked user about broadening
mean/std; retained existing moments pending preference, with separate new output
identities. Three seeds reduce precision; zero oracle falls at10 in one run is
not evidence of unimpaired teaching (progress was only37.8% of corrected nominal).

## 2026-10-09: migrate all domain pipelines without mixing physics generations

Corrected single-argument scene writing now applies to fixed evaluation and all
Gaussian collection/training/evaluation, including nominal controls. Merely fixing
varied training would leave an invalid nominal comparison. New protocol/output
identities prevent legacy data reuse; original root initializers and old experiment
artifacts remain unchanged. Scalar status is checked at server startup; fixed
reporter rejects legacy manifests. Shared lock remains in its old common location.
23 offline tests pass, including legacy-plan/receipt rejection; ten generated
configs verified without launching a server. Full corrected batches remain user-run.

## 2026-10-09: single-value setter resolves manipulation in live smoke

Scalar oracle smoke now records solver slip classes tracking all five configured
settings, with monotonically reduced walking progress. All five receipts pass
artifact verification; passive traces retain slip enable flags. This strongly
implicates the legacy two-argument scene expression, though exact script-binding
fallback remains unproven. See oracle_scalar_v1/INTERPRETATION.md under
experiments/slip_sensitivity/. Old datasets/checkpoints remain preserved; their
varied-slip interpretation must be qualified, and old nominal physics cannot be
pooled with corrected .1. Migration of other launchers is still outstanding.
Zero falls at extreme slip do not imply unchanged locomotion or a validated range.

## 2026-10-09: alternative single-value scene setter, live check pending

After both small and 100x scene-slip sweeps retained identical solver classes,
user requested another modification method. Added a separate scalar scene-setter
suite: convert the generated `(setContactSlip v v)` to `(setContactSlip v)` while
retaining normal surface combination and passive ODE logging. Installed native
API is scalar, but rejection/handling of extra script arguments has not yet been
proven as the root cause. Ten tests and five offline real-config preparations
pass; no live success claimed. Preserve old data and withhold varied-slip claims
until the new contact evidence validates the intervention. No system-file edits.

## 2026-10-09: large-slip screen confirms unresolved manipulation

User-run stress smoke at scene .1/.5/1/2/5/10 is complete; all six artifact receipts
pass hash checks. Recorded solver-slip/mode classes remain identical (~.005/.3025)
even at 100x nominal scene slip. Zero falls and similar progress do not demonstrate
high-slip robustness. Diagnose binding/scene application/surface combination before
additional sweeps. The two-argument scene versus scalar setter remains a lead, not
a verified cause; no fix yet. See oracle_stress_v1/smoke_report.md under
experiments/slip_sensitivity/. Preserve data and historical conclusions with this
explicit limitation.

## 2026-10-09: larger settings do not automatically resolve the slip confound

User requested an oracle stress screen at much larger scene values (.1 to 10).
Kept the current configuration pathway and passive audit, with exact contact
classes/counts in reports. Unchanged classes trigger a warning; different
proportions or performance do not validate changed effective slip. Even changed
classes may reflect contact-body differences. The existing setter/binding concern
is unresolved. Prior datasets are retained; no robustness claim or new Gaussian
training range follows from this design alone. Stress screen is live-unvalidated.

## 2026-10-09: live passive audit fails to show slip variation

Three validated smoke receipts (.025/.100/.200) expose the same two solver-bound
slip classes, ~.005 and ~.3025, enabled in both axes throughout. Changing aggregate
means comes from contact counts, not changed class values. Source/import checks
therefore did not validate the intended physics manipulation. Historical artifacts
are retained; varied-domain robustness/learning claims are qualified pending a
corrected manipulation and new-version validation. Original nominal comparisons
are not invalidated by this finding. Contact classes are not yet body-identified.
The scalar native setter versus two-value scene expression is an unconfirmed
diagnostic lead. No implementation fix yet; stop expansion to context-aware
training and avoid repeating the 30-run sweep before resolving the issue.

## 2026-10-09: configured versus effective contact slip

Weak locomotion sensitivity in current results may reflect the physical
manipulation, not policy adaptation. Scene import/hash checks do not establish
combined contact parameters submitted to ODE. Added a passive contact audit with
enable flags and a crossed slip/task-seed oracle/BC pilot. Audit forwards contacts
unchanged. Logs include startup/recovery and ball contacts, so aggregate ranges
alone do not establish foot-specific physics or actual sliding. Live validation
remains pending; offline native checks and five Python tests pass.

Resolved draft build-cache mismatch: changed native build identities previously
caused strict receipt errors. Content-addressed build directories now preserve old
artifacts and create separate builds. Offline success does not establish that the
live SimSpark hook path works; empty live contact evidence is rejected.

This document records defects, experimental risks, and lessons encountered while
integrating behavioral cloning, DAgger, and GAIL with the FC Portugal SimSpark
walking environment. It is intended to prevent regressions and to distinguish a
pipeline that merely finishes from one that produces a valid scientific result.

Statuses used below:

- **Resolved:** the maintained implementation contains a correction.
- **Partially resolved:** safeguards exist, but experimental validation remains.
- **Open:** further implementation or experimentation is required.

## 1. Original behavioral-cloning implementation

### 1.1 Targets crossed the midfield boundary

**Original issue:** `collect_long.py` sampled both coordinates from `[-8, 8]`.
`collect_short.py` allowed x-coordinates in `[-6, 6]` and initialized its first
target at `[1, 0]`. Consequently, targets could lie in the opponent half even
when headless game-mode rules prevented the player from crossing midfield.

This could produce demonstrations dominated by pushing against an inaccessible
boundary rather than purposeful walking.

**Status: Resolved.** All maintained collectors sample inside:

```text
x in [-6.5, -0.5]
y in [-6.5,  6.5]
```

Short targets are rejected unless they remain 1–2 metres away and within these
bounds.

### 1.2 Unsafe asynchronous collector handoff

**Original issue:** separate long- and short-range processes introduced a
wall-clock interval during which the robot received no stabilizing commands. In
real-time/asynchronous simulation, physics continued and the unsupported robot
could fall before the next collector connected.

**Status: Resolved.** Required server settings are synchronous agent control and
disabled real-time mode. Launchers verify both settings. FC Portugal sends
`(syn)` with each command, so physics does not advance during process handoff.

### 1.3 Synchronous communication deadlock

**Original issue:** collectors performed `receive -> send`. In synchronous mode,
the agent waited for the server while the server waited for the agent's `(syn)`.
The player appeared static.

**Status: Resolved.** Maintained loops use:

```text
observe -> act -> send command plus (syn) -> receive next state
```

### 1.4 Fall reset did not stabilize the robot

**Original issue:** the collector only checked head height and teleported the
robot. It resumed walking immediately from stale simulator/controller state.
This could cause repeated falls and contaminate demonstrations with reset
transitions.

**Status: Resolved.** Shared fall recovery now suspends the robot, commands the
`Zero_Bent_Knees` pose, beams it to its robot-specific ground height, and runs
synchronous stabilization cycles. These cycles are excluded from datasets.

### 1.5 Dataset integrity and reproducibility

**Original issues:**

- Re-running the launcher silently appended to existing demonstrations.
- Dataset writes were not atomic.
- Empty, malformed, mismatched, NaN, or infinite arrays were not rejected.
- Collection and training did not consistently expose random seeds.
- There was no fixed test partition.

**Status: Resolved.** Dataset operations are validated and atomic. Reset flags
create backups instead of deleting prior data. Collection and training are
seeded. Contiguous blocks are assigned to fixed expert train/test partitions,
with split metadata saved for reproduction.

### 1.6 Training reported only training loss

**Original issue:** BC saved the final epoch with no validation partition or
independent fidelity measures, making overfitting invisible.

**Status: Resolved.** The lowest-validation-loss checkpoint is retained.
Validation MSE, RMSE, MAE, R-squared, and per-action errors are recorded.

### 1.7 Native Python extension ABI mismatch

**Observed issue:** FC Portugal `.so` modules had been compiled for Python 3.9
but were imported from other Python versions. This produced errors such as an
undefined `_PyThreadState_UncheckedGet` symbol.

**Status: Resolved.** Launchers detect the active interpreter, rebuild native
modules when its version differs, and verify imports before modifying datasets
or starting collection.

## 2. Original DAgger implementation

### 2.1 Student observations were never retrieved

**Original issue:** the rollout searched several nonexistent observation
attributes but omitted the actual walk trajectory buffer. The robot also had no
`get_observation()` fallback. `current_obs` therefore remained `None`.

**Consequence:** student inference was skipped.

**Status: Resolved.** `Walk.observe()` now explicitly returns the exact
63-dimensional observation shared by the oracle and student.

### 2.2 Student actions were never applied

**Original issue:** the rollout attempted `robot.set_joint_positions()`, which
did not exist. Even if it had existed, the policy's 16 outputs were incorrectly
treated as direct joint positions.

**Consequence:** FC Portugal controlled every rollout, so the implementation was
repeated BC on oracle-visited states rather than DAgger.

**Status: Resolved.** `Walk.execute_action()` validates and executes the 16-D
latent action through FC Portugal smoothing, step generation, inverse
kinematics, and joint-target logic.

### 2.3 Incorrect aggregation/training order

**Original issue:** each iteration trained before rollout. The final rollout was
aggregated after the final training pass, so the final checkpoint never used the
last batch.

**Status: Resolved.** Iteration 0 trains on the oracle seed. Every subsequent
iteration performs rollout, oracle labeling, aggregation, and then training.

### 2.4 No protection against an unstable initial student

**Original issue:** pure student control was used immediately. A weak initial BC
policy could fall rapidly and concentrate the aggregate on unusable states.

**Status: Resolved.** A default oracle/student mixture decays oracle execution
probability from 0.8 to zero over the first five iterations. FC Portugal labels
every state regardless of which action is executed.

### 2.5 Old aggregate was scientifically invalid

**Observed issue:** `aggregated_dagger_data.npz` produced by the original code
contained oracle-on-oracle samples while being presented as learner-state
aggregation.

**Status: Resolved operationally.** `RESET_DAGGER_DATASET=1` backs up the old
file and creates a clean experiment. Old aggregates must not be reused in final
comparisons.

### 2.6 Remaining DAgger research considerations

**Status: Open.** The seed size, rollout length, number of iterations, and beta
schedule remain experimental hyperparameters. Twenty thousand seed steps are a
reasonable starting point but are highly temporally correlated. Conclusions
should be based on coverage, multiple seeds, convergence curves, and active
oracle-query counts rather than raw sample count alone.

## 3. Original GAIL implementation

### 3.1 Adapter called an unsupported API

**Original issue:** `gail_adapter.py` passed `execute_expert=False` to
`Walk.execute()`, which had no such parameter. Rollout failed immediately with a
`TypeError`, producing an empty rollout archive.

**Status: Resolved.** The adapter uses the explicit `Walk.observe()` and
`Walk.execute_action()` interface.

### 3.2 Random, high-variance actor initialization

**Original issue:** the first rollout used a randomly initialized Gaussian actor
with unit standard deviation. Unbounded latent actions were unsafe for bipedal
locomotion and likely to cause immediate falls.

**Status: Resolved.** The actor mean initializes from the final DAgger policy and
begins with a smaller exploration standard deviation.

### 3.3 Incorrect expert occupancy data

**Original issue:** GAIL used the DAgger aggregate as expert data. Those states
were visited by mixed/student policies and only counterfactually labeled by the
oracle; they are not samples from the expert occupancy distribution.

**Status: Resolved.** GAIL uses `expert_train_data.npz`, containing states and
actions actually generated by FC Portugal. DAgger initializes the actor only.

### 3.4 Rewards came from a stale discriminator

**Original issue:** rewards were computed during rollout, then the discriminator
was updated, but PPO continued using rewards from the old discriminator.

**Status: Resolved.** Training updates the discriminator first and then
recomputes all adversarial rewards for the saved policy transitions.

### 3.5 Incorrect terminal assignment

**Original issue:** a fall was detected before selecting the current action, but
the newly selected post-reset transition was marked terminal. The terminal mask
belonged to the preceding action that produced the fallen state.

**Status: Resolved.** Fall termination is assigned to the preceding transition.

### 3.6 Truncations bootstrapped with zero

**Original issue:** every rollout appended a zero value, treating a collection
horizon as if it were a genuine terminal state and biasing GAE returns.

**Status: Resolved.** Rollouts save the next-state critic value and use it for
nonterminal truncation bootstrapping.

### 3.7 Optimizer state was discarded

**Original issue:** Adam optimizers were recreated without restoring their state
on each outer iteration.

**Status: Resolved.** Policy and discriminator optimizer states persist between
iterations.

## 4. Observed GAIL training failure after integration

The corrected pipeline completed ten iterations, but completion did not imply
successful imitation. The observed trends indicate discriminator saturation and
unstable PPO updates.

### 4.1 Discriminator saturation

By iteration 10:

```text
discriminator loss:             0.0290
expert classification accuracy: 0.99765
policy classification accuracy: 0.99850
mean learned policy reward:      0.00464
```

The discriminator could almost perfectly separate expert and policy samples.
With the current minimax reward `-log(1-D)`, the useful policy reward approached
zero.

**Status: Resolved in the September 5 clean training run.** The maintained trainer now normalizes
discriminator inputs using expert statistics, uses one discriminator epoch by
default, applies 0.9/0.1 label smoothing and interpolation gradient penalty,
stops an over-accurate discriminator, and uses non-saturating `log D(s,a)`
reward. Across ten live iterations, discriminator classification loss remained
between `1.3570` and `1.5119`; its expert/policy accuracies approached chance
instead of saturating, while learned rewards remained finite. Further candidate
remedies are only necessary if later robustness experiments destabilize it:

- Add gradient penalty, spectral normalization, or another Lipschitz constraint.
- Tune the new regularization strength or add instance noise/spectral normalization.

### 4.2 PPO trust-region failure

Observed iteration-10 diagnostics included:

```text
approximate KL: 1158.64
clip fraction:  0.9873
actor loss:     0.3252
```

Approximate KL values were already extremely large in early iterations, and
approximately 98% of samples were commonly outside the PPO clipping range. PPO
clipping limits the surrogate objective but does not itself guarantee that the
updated distribution remains close to the behavior policy.

**Status: Resolved in the September 5 clean training run.** Actor and critic now use separate learning rates,
the actor default is one hundred times smaller than the critic default, PPO performs
full-rollout target-KL early stopping, and actor updates exceeding the rollback
threshold are reverted together with their optimizer state. A controlled fixed-rollout
test at actor learning rate `3e-5` produced KL `0.3499` and correctly triggered
rollback. Reducing the default to `3e-6` completed all three PPO epochs with KL
`0.01107`, clip fraction `0.039`, and no rollback. In the subsequent ten-iteration
simulator run, accepted KL remained between `0.00838` and `0.01161`, all updates
completed three epochs, and no rollback was required. Remaining candidate remedies:

- Reduce PPO epochs or minibatch update count.
- Track the first-minibatch and first-epoch KL separately from the final average.

### 4.3 Imitation fidelity deteriorated

Actor-mean MSE on the expert dataset increased from approximately 0.47 in
iteration 1 to 4.50 in iteration 10. Rather than refining the DAgger
initialization, adversarial/PPO training moved the actor away from oracle
actions.

**Status: Resolved for training stability; final evaluation pending.** A small
decaying auxiliary BC anchor is now part of actor training, and unsafe KL updates
are rolled back. In the clean rerun, expert-dataset actor-mean MSE remained in
the narrow range `0.26782–0.26832` over all ten iterations instead of growing
catastrophically. Iteration checkpoints are retained for fixed-seed evaluation.
Automatic selection of the best mechanically validated checkpoint remains open.

### 4.4 Mechanical instability

Selected rollout observations included:

```text
iteration 3:  0 falls, 11 targets reached
iteration 8: 44 falls,  2 targets reached
iteration 9: 41 falls,  2 targets reached
iteration 10: 1 fall,   0 targets reached
```

Some torso roll/pitch statistics were also physically extreme. Iteration 3
appeared mechanically better than later checkpoints, but its PPO KL was still
invalidly high. It is therefore only a provisional candidate, not evidence of
convergence.

**Status: Substantially improved; checkpoint selection remains open.** The clean
rerun reached 6–14 targets per 4000-step rollout and recorded only one fall over
40,000 steps. Nine rollouts had normal torso RMS values near 2–5 degrees. Iteration
5 was a transient warning case (roll RMS `17.08`, pitch RMS `12.97`, 6 targets),
but recovered in iteration 6 without policy rollback. Add mechanical safety gates
and best-checkpoint selection.
Possible gates include maximum falls per rollout, target progress, torso-angle
limits, action magnitude, action-delta limits, and target KL.

### 4.5 Height-only fall detection produces false negatives

The second ten-iteration run exposed stationary, toppled states that were not
counted as falls because head height remained slightly above 0.28 metres:

```text
iteration 8: 0 reported falls, 0 targets, roll RMS 91.89 deg,
             pitch RMS 55.85 deg, mean head height 0.305 m
iteration 9: 0 reported falls, 0 targets, roll RMS 48.68 deg,
             pitch RMS 64.48 deg, mean head height 0.318 m
```

The player can lie or become mechanically trapped with its head above the height
threshold. Zero reported falls therefore does not necessarily mean stable
locomotion.

**Status: Partially resolved.** Fall recovery now triggers on head height or
absolute torso roll/pitch above 60 degrees. Sustained lack-of-progress and
foot/contact-based stuck detection remain open. Each physical event must still
be counted only once before recovery.

### 4.6 Training-rollout iteration labels are pre-update measurements

`gail_rollout_metrics_iter_N.json` is collected before training update N. Thus:

```text
rollout 1 measures the DAgger-initialized GAIL actor
rollout 2 measures gail_policy_iter_1.pth
...
rollout 10 measures gail_policy_iter_9.pth
```

There is no training-rollout measurement of `gail_policy_iter_10.pth`. Different
iterations also use different target seeds, so their raw goal/fall counts are
not controlled checkpoint comparisons.

**Status: Partially resolved.** `run_evaluation.sh` evaluates archived
checkpoints with identical seeds. Training-rollout metrics must be treated as
diagnostics, not as the final checkpoint ranking.

### 4.7 Trainer-port play-mode reset failed

Every rollout in the second run reported a refused connection to trainer port
3101, although the agent and monitor connections on ports 3100 and 3200 worked.
Collection continued under the server's existing play mode, so the requested
`PlayOn` state was not guaranteed by the rollout process.

**Status: Open.** The preflight should verify the configured trainer endpoint or
set the play mode through the established monitor connection after agent
startup. Silent continuation should be replaced by an explicit verified state
or a clearly documented opt-out.

## 5. Evaluation protocol challenges

### 5.1 Action-space terminology

The learned network predicts 16 latent walk actions, not 22 joint-velocity
commands. Current Action MSE is therefore explicitly 16-D latent-action MSE.
Calling it direct joint-command MSE would be incorrect. A future joint-command
comparison would require equivalent cloned controller state for the learner and
oracle so both latent actions can be propagated without advancing the live
controller twice.

### 5.2 High-dimensional Wasserstein estimation

Exact empirical optimal transport in 22 dimensions is computationally expensive
and sample-inefficient. The maintained evaluator uses deterministic sliced
Wasserstein-1 after standardizing joint dimensions with oracle statistics.

**Status: Partially resolved.** The projection count and estimator sensitivity
should be included in ablations.

### 5.3 Jerk amplifies simulator noise

Third finite differences strongly amplify joint-angle noise and teleportation
discontinuities. The evaluator applies a five-sample moving average and excludes
reset boundaries before calculating jerk.

**Status: Partially resolved.** The filter and sample frequency must remain fixed
across policies, and raw trajectories must be retained so alternative filters
can be audited.

### 5.4 Fair normalized-return baselines

Normalized return is meaningful only when random, learned, and expert policies
use identical seeds, active durations, target generation, simulator settings,
and fall accounting. Scores are intentionally not clipped: values below zero or
above one carry useful information.

The random latent-action policy is defined using independent uniform samples
between each action dimension's observed expert-training minimum and maximum.
This definition must remain fixed when comparing experiments.

### 5.5 Jerk can reward a failed stationary policy

The first standardized evaluation showed that late GAIL checkpoints could have
lower jerk than the oracle while making almost no target progress. A stationary,
trapped, or toppled robot produces little joint motion and can therefore obtain
an apparently favorable jerk cost.

**Status: Open analytically.** Never rank policies by jerk alone. Report jerk
alongside normalized return, target-directed velocity, torso attitude, and falls.
Consider an efficiency metric conditioned on a minimum locomotion-performance
threshold or normalized by positive target progress.

### 5.6 Single-seed results and repeated-checkpoint variation

The first evaluation run used only seed 101, so all reported confidence
intervals had `n=1` and zero estimated width. Final and last-iteration checkpoint
files were byte-identical, but their repeated simulator rollouts differed. This
shows that the evaluation seed currently controls policy/target randomness but
does not guarantee a bitwise deterministic simulator trajectory.

**Status: Open.** Final claims require multiple seeds. Simulator noise settings,
initial world state, play mode, and reset state should be explicitly recorded
and controlled. Byte-identical duplicate checkpoints should normally be
evaluated once rather than interpreted as independent policies.

### 5.7 SimSpark monitor backpressure during long evaluation runs

During multi-policy, multi-seed evaluation, SimSpark repeatedly reported
`MonitorControl send returned error 'Resource temporarily unavailable'`, then
aborted in `AgentControl::AgentThread` with a Boost mutex error. The FC Portugal
client opened a monitor connection for control commands but did not consume the
monitor frames continuously sent back by the server. The unread data eventually
filled the server's socket buffer; connection teardown then exposed a SimSpark
thread-lifecycle race.

**Status: Partially resolved; second mitigation awaiting confirmation.** Draining
pending monitor output after every synchronous receive allowed many more cases to
complete, but SimSpark eventually crashed when agent and monitor sockets were
closed together after `dagger_iter_6`. Socket shutdown is now explicit and the
monitor is closed 250 ms after the agent, serializing the two server-side thread
teardowns. The primary evaluation no longer evaluates every intermediate
checkpoint by default; checkpoint sweeps are opt-in and resumable. Evaluation
results written before a server crash remain usable.

An interrupted random seed 101 evaluation exposed a related orchestration bug:
`Server_Comm` called `exit()` without a nonzero status after a server disconnect.
Consequently, `set -e` treated the failed evaluation as successful and continued,
leaving normalization without its matching baseline. Disconnect paths now exit
with status 1, and the summarizer reports exactly which baseline seeds are
missing together with the resume command.

## 6. Next actions

Before accepting the current GAIL result:

1. Complete `run_evaluation.sh` across several seeds and inspect raw trajectories.
2. Confirm that monitor-buffer draining prevents further SimSpark crashes.
3. Add automatic mechanically informed best-checkpoint selection.
4. Report normalized performance against query/environment cost, not only the
   final iteration.

### 6.1 Controlled GAIL initialization comparison

The original successful GAIL experiment was initialized from DAgger, making it
unclear how much of its final performance came from adversarial training versus
the pretrained actor. It is now labelled `gail_dagger`. Two matched variants,
`gail_bc` and `gail_random`, use BC and seeded random actor initialization.

**Status: Both additional variants trained; random initialization failed behaviorally.** The expert occupancy, BC anchor,
training budget, PPO/discriminator settings, and evaluation protocol remain
identical across variants. Only actor initialization changes. This control is
necessary before testing whether initialization affects robustness under
friction changes, noise, collisions, or joint/motor impairment. The completed
`gail_bc` run kept accepted KL between `0.00772` and `0.00988`, required no
rollbacks, and retained expert-dataset actor MSE near `0.0071`.

The `gail_random` run was numerically stable but reached zero targets in every
4,000-step rollout. Expert-dataset actor MSE decreased only from `7.685` to
`7.653`. Its very low torso motion and zero falls describe a stationary solution,
not stable locomotion. Discriminator classification loss fell from `1.384` to
`0.882` and mean reward declined to `-1.431`, showing increasing separation from
expert occupancy. PPO KL remained only `0.0008–0.00214`; the trust-region settings
that protect pretrained actors were too conservative to bootstrap a random actor
within 40,000 environment steps.

### 6.2 Original DAgger seed coverage and dataset composition

The original DAgger policy showed much worse expert-test Action MSE than BC.
Although both final training sets contained 90,000 labels, DAgger began with only
20,000 long-target oracle demonstrations and obtained its remaining 70,000 labels
from successively diluted mixture/student rollouts. It therefore lacked BC's
explicit alternating long-run and short-turn coverage at initialization.

**Status: Experiment completed and evaluated; size/coverage confound remains.** The new
`dagger_mixed` policy starts from 100,000 fresh, evenly balanced long/short oracle
samples, then uses the original seven 10,000-step rollout budget and exact beta
schedule. It ends with 170,000 labels. The completed expanded evaluation found
a 78.5% reduction in held-out Action MSE and a 95.0% reduction in on-policy
oracle MSE relative to original DAgger. This supports the combined richer/larger
seed intervention, but cannot separate composition from sample count or isolate
the aggregation process. The stale collection-pending status was reconciled
with `evaluation/PHASE1_RESULTS_REPORT.md` during the 2026-09-20 presentation
review; the historical motivation is retained above.

### 6.3 Root-directory experiment clutter

Early pipelines wrote datasets, optimizer state, discriminators, metrics, and
every iteration checkpoint directly into the repository root. Multiple DAgger
and GAIL variants would make artifact ownership ambiguous and increase the risk
of evaluating or overwriting the wrong checkpoint.

**Status: Resolved.** Intermediate state is stored under `artifacts/<policy>/`,
completed/superseded runs live under `experiments/`, and only stable final policy
files are copied to the root for evaluation. The existing BC and original DAgger
intermediates were migrated into this layout.

### 6.4 Editing a pipeline while its shell process is active

During the artifact-layout migration, an already-running `run_gail.sh` process
continued reading the modified script and reached a new `ARTIFACT_DIR` reference
without having executed the new initialization line. With `set -u`, it stopped
with an unbound-variable error.

**Status: Resolved.** The GAIL loop reasserts its derived label and artifact
directory immediately before iteration processing. No completed `gail_random`
checkpoint existed to recover from the observed failure. Pipeline scripts should still not normally be edited while
their existing shell process is running.

### 6.5 Modified-domain configuration provenance

Manual editing of the installed SimSpark scene could silently contaminate later
nominal experiments or mix traces from different physics settings. The server
also reads the scene only at startup, so editing it while running does not safely
establish a new condition.

**Status: Mitigated; pilot validation pending.** `configure_field_slip.py`
requires an exact single field-slip expression, creates a nominal backup before
the first write, performs atomic replacement, verifies the result, and reports
the effective SHA-256. Phase 2 launchers reject unexpected slip values and store
environment manifests. The operator must still stop the server before applying
or restoring a value and restart it afterward.

## Maintenance requirement

Update this document whenever a new pipeline defect, experimental failure,
important mitigation, or unresolved research risk is discovered. When an open
issue is fixed, retain the historical description and change its status rather
than deleting it.

## 7. Live demonstration timing

### 7.1 Unpaced synchronous rollouts are too fast for a live audience

Synchronous collection/evaluation runs as quickly as the host permits, so
1,500 simulator steps do not necessarily occupy 30 wall-clock seconds.

**Status: Mitigated in the live demo runner.** `demo_agents.py` paces active
and recovery cycles to a minimum 20-ms period using a monotonic clock, including
processing time and avoiding catch-up bursts after delays. It retains the
existing synchronous simulator configuration. Startup/recovery add time beyond
the 30-second active budget; slow hardware can also extend elapsed time. The
demo sets and verifies `PlayOn` through the monitor connection, avoiding the
optional trainer endpoint issue for this runner only. Live visual validation
remains separate from the offline timing and control-loop checks.

## 8. Local-scene slip pipeline validation (2026-09-25)

System scene mutation via sudo failed because a password was required; no mutation
occurred. Sandbox socket creation was denied, and the outside-sandbox request was
interrupted. The user selected local scenes and explicitly user-run experiments.
**Status: local alternative implemented; live validation pending.** Absolute scene
imports avoid privileged edits; startup/configuration hashes provide provenance,
but do not independently read back ODE's live slip parameter. First smoke runs must
validate successful loading and locomotion before a full condition is approved.

Fresh servers exposed reliance on optional trainer port 3101 for PlayOn. The new
opt-in fresh-start evaluator uses the established monitor and stabilizes the robot
before counting falls. Rerun nominal under this same protocol; do not interpret
startup differences as slip effects. Fresh paired simulator seeds and per-run
restarts also reduce the historical socket-teardown race. Five sampled draws are
only a pilot. Bounds remain a human decision following fixed results.

### 8.1 Protocol fidelity correction

**Resolved before live experiments:** The first local-runner draft added startup
stabilization, fixed simulator seeds, thread limits and per-policy restarts. These
would confound slip comparisons. Per user instruction they were removed, and the
original evaluator restored (including original trainer/PlayOn behavior). The
wrapper now calls the original evaluation launcher and shares a server throughout
each fixed condition. Earlier section 8 mitigation describing fresh-start behavior
is historical and superseded. Any future startup fix must be validated consistently
on both baseline and varied-slip protocols. Local startup remains unvalidated live.

### 8.2 Unreachable startup marker (2026-10-04)

**Superseded diagnosis; see section 8.3.** The first live smoke attempt started SimSpark,
but the wrapper timed out waiting for `WALKER_SCENE_LOADED`. The generated Ruby
configuration had appended that marker after `run "naosoccersim.rb"`; `run` owns
the active simulator loop and therefore never reached the marker during startup.
The marker is now emitted and flushed immediately before `run`. An offline contract
test enforces this ordering. The failed attempt produced no rollout data and did
not modify the installed nominal scene.

### 8.3 Startup configuration selection (2026-10-04)

The retry also timed out. The previous claim that Ruby `run` owns a non-returning
simulation loop was unsupported and is retracted. Installed `rcssserver3d --help`
specifies `--script-path PATH` for selecting the startup configuration. The wrapper
used a bare positional filename instead. It now passes the explicit option and
emits the marker at the end of the local NAO setup script. Timeout errors include
the server log path. Offline tests cover the invocation and cleanup on a missing
marker. Live validation remains pending; no real smoke test has passed yet.

### 8.4 Stronger slip provenance checks

A static printed marker did not prove which scene bytes the server imported.
The server now hashes the local scene immediately before import and emits a
unique marker only after the import reports success. Python also checks exact
marker identity, requested slip and configuration hashes. This mitigates wrong
scene/stale-marker risks. It does not establish independent live ODE parameter
readback, and Ruby digest/import behavior still needs the user-run smoke test.

Subsequent user validation: smoke succeeded with these checks enabled. The startup
mitigation has now passed the user-run smoke test; full comparison is next.

### 8.5 Interactive workflow correction

The user intended to launch scripts manually, not approve each condition while
they run. Removed the internal `RUN` prompt from fixed and sampled blocks so
invocations proceed to completion automatically. Failure cleanup and protections
against overwriting completed/partial evaluations remain in place.

### Phase 2 live server crash after completed high-traction block

Approved mitigation implemented: isolated_server_slip_v2 starts a fresh simulator
for each policy/seed, preventing reuse of a server after agent teardown. A
watchdog aborts the client if its server dies. Completed runs are validated before
resume; failed attempts are preserved. Old fixed/smoke artifacts are archived and
all conditions will be recollected consistently. This avoids repeated connections
within a server but does not patch the underlying native crash. Live validation
is pending; offline tests cover isolation, resume, tampering and timeout cleanup.

Recurrence: the nominal retry failed after mixed DAgger seed 101, again with
boost::lock_error in AgentControl::AgentThread. The 0.25-second client shutdown
delay does not eliminate this failure. Twenty-one nominal summary files exist;
file presence alone is not a completeness audit. Repeated whole-condition retries
are not a reliable remedy. Per-policy/seed process isolation is a proposed
mitigation, not a verified fix, and requires consistently collected comparison
data because it changes simulator lifecycle/state relative to the existing series.

High traction completed, but nominal aborted after BC seed 505 at the next agent
connection. Its server.log records monitor broken pipes followed by SIGABRT and
boost::lock_error in AgentControl::AgentThread, consistent with the historical
teardown race. This is an infrastructure interruption, not a policy fall. Preserve
completed high traction and archive partial nominal before a whole-condition retry.
The existing client mitigations have not eliminated the server failure. Per-run
server restarts would change the agreed lifecycle and are not silently introduced.

### 8.6 Sample-distribution plan mismatch

Resume fingerprint defect: changing Gaussian suite selection/tests invalidated
completed baseline receipts despite unchanged evaluator, server, models, plan and
environment. All 30 existing receipts passed artifact checks, including all 25
required Gaussian runs. Evaluation identity now separates audited orchestration
metadata from rollout sources; legacy compatibility is restricted to known hashes.
Unknown legacy metadata, changed rollout inputs or corrupted artifacts still fail.
Original receipts are preserved; training fingerprints remain strict. Eleven
offline tests pass, including compatibility and rejection cases.

Redundant-evaluation correction: variable-training baseline previously scheduled
fixed and Gaussian suites together. It now defaults to Gaussian only, matching
the user's next-stage scope. Reports handle absent nominal data explicitly with
null retention instead of NaNs or requiring another fixed sweep.

Fixed-result follow-up: all 120 isolated-server runs passed validation. Selected
policies and oracle remain close to nominal progress across the tested bounds.
This supports a viable teacher pilot but creates a ceiling-effect risk for claims
of improved robustness after training. See FIXED_RESULTS_INTERPRETATION.md; wider
severities and live parameter readback would require separate validation.

Variable-training implementation now uses its own validated truncated-Gaussian
plan generator. The old Phase 2 uniform planner remains unchanged while its fixed
sweep runs. The new package resolves continuation and segment-boundary risks via
warm-start supervised training, whole-run validation, per-segment GAE and immutable
job receipts. Fixed GAIL normalization prevents input scaling from changing across
surface batches. Per-client servers and a shared lock isolate training from the
active evaluation. Original modules were deliberately left intact because root
Python files are fingerprinted by the active runner. Live validation remains
pending. New reports omit Scorenorm because random is outside the requested scope;
oracle ratios must not be presented as random-normalized scores.

Training design risks identified during pipeline review: train_policy constructs
new weights on each call (including DAgger updates), so continuation requires an
explicit warm-start interface. Existing frame-level validation must be replaced
by whole-run splits for this study. GAIL's single bootstrap scalar cannot safely
represent concatenated independent server rollouts; compute GAE per segment.
These changes are specified in VARIABLE_SLIP_TRAINING.md and remain unimplemented.
Provisional Gaussian parameters are design choices, not measured viable ranges.

The agreed follow-up now specifies truncated-Gaussian slip sampling, while
`summarize_varied_slip.py` generates and validates uniform draws. This remains
pending implementation before sampled testing; running fixed evaluations are
unaffected. Store parent mean/std, bounds, effective variance and paired draws.
Do not label current uniform artifacts Gaussian or substitute clipping for
truncation. Select parameters from completed fixed results and oracle viability.
## Consolidation notes — 2026-10-15

The standardized study distinguishes unique training examples, oracle labels,
learner environment steps and optimizer updates. BC170k and DAgger100k+70k match
label counts; GAIL70k matches learner rollout steps but also consumes170k static
expert rows and inherited initializers. They do not have equal end-to-end cost.
Validation/test add10k each outside training, and shared data are counted once in
the collection ledger. Gradient updates/exposures remain algorithm-specific.

Original installed scene still uses the legacy two-value slip expression; rewriting
it to scalar.1 would silently change the original environment. New owned-server
launcher preserves scene bytes for original runs and uses scalar setting only
for varied runs, with separate manifests and paired baselines. Existing historical
results remain untouched and are not pooled with these fresh-server evaluations.

Shell-only consolidation cannot correctly condition DAgger/GAIL: policies, labels,
critic/discriminator data and GAE segment boundaries must agree on63/64 inputs.
Isolated adapters now reuse existing networks/context encoding/PPO and test those
contracts. GAIL-mixed-DAgger explicitly uses the newly trained mixed checkpoint,
not the historical original-DAgger GAIL. Offline tests include actual synthetic
64-input BC/PPO updates; end-to-end simulator smoke remains unrun by the agent.

# Managed viewer integration — 2026-10-15

Historical demos required a manually started server and could not select corrected
scalar slip or consolidated input-conditioned models. `run_viewer.sh` now reuses
the owned, verified server lifecycle and shared simulator lock, with explicit
context routing and original-scene distinction. It must not compete with live
training for ports3100/3200. Viewer motion remains a demonstration, not robustness
evidence. Offline selection/syntax checks are available; GUI/live viewing still
requires user verification.

# Consolidated probing comparability — 2026-10-15

The historical probe loader assumes four63input policies and the older evaluation
receipt format. Pointing it at the consolidated batch would omit replicas/context
and misidentify mixed-DAgger GAIL lineage. A separate consolidated probe entrypoint
now verifies new receipts/checkpoint hashes and includes all28models, expert and
matched63/64input random/raw controls. It preserves the existing30locomotion labels
and whole-run splits. Additional context can improve decoding without proving
that the actor uses it beneficially; expert/student architectural differences and
periodic shuffle controls remain interpretation risks. These risks are documented,
not claimed resolved by a higher probe score. Historical results are preserved.

# Interpretation risks confirmed in consolidated report — 2026-10-15

The completed batch equalizes final unique expert labels, not optimization:
BC17million versus mixed-DAgger54million supervised sample exposures. Varied
continuations inherit original policies and receive extra training; no matched
nominal-only continuation control exists in this batch. Therefore improved
fidelity/fall rates cannot be attributed exclusively to aggregation or surface
diversity. Corrected scalar0.1 retention is not original-scene retention, which
was not evaluated for varied policies. Correct context alone does not establish
causal context use. Probe CKA is not a fidelity ranking:varied DAgger improves
action error while some expert-CKA scores decrease. Documented with numerical
evidence in `evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md`; these research
limitations remain open, rather than being treated as resolved implementation bugs.

# Append-only scratch comparisons and research fairness — 2026-10-15

Adding controllers directly to the old immutable schedules/plans would invalidate
receipts and could replay or relabel completed work. The scratch comparison now
uses an isolated package/output and exact parent artifact verification. Existing
models/results and300probe fits remain unchanged; combined reports reference them
and append only new results. The parent source identity is not weakened or migrated.

Scratch versus continuation matches the additional varied-domain budget but not
the continued models' inherited lifetime training cost. Report this as a pretrained
starting-point comparison, not an equal-total-compute algorithm test. Direct
random-GAIL retains the old supervised anchor, conservative actor learning rate
and70kPPO budget; this can disadvantage learning from random weights and must be
reported rather than silently changing hyperparameters. No result advantage is
assumed. User explicitly selected random-GAIL original and varied-hidden only;
no input-conditioned random-GAIL is included.

