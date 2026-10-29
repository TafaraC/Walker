# Session handoff

## Latest — publication snapshot, 2026-10-29

User authorized committing and pushing the source/documentation snapshot to
origin (TafaraC/Walker), retaining upstream history and GPLv3 attribution.
Snapshot includes research code, tests, reports, paper sources, figures and
presentation; large experiment data/checkpoints and machine files are ignored.
Eligible checkout is approximately8.5MiB across393files before this note.
Use git log/status and origin/main to inspect the resulting commit/publication
state. Research data and final academic author details remain the user's next
publication tasks; full artifact-dependent reproduction needs separate data.

## Latest — own remote and upstream credit, 2026-10-29

User supplied ownrepo https://github.com/TafaraC/Walker and requested licence/
credit. Configured origin to that URL and added upstream pointing to original
FCPCodebase; history retained. LICENSE unchanged GPLv3. README now leads with
research overview/navigation and explicit upstreamcredit; original description,
features and academic citation retained. Added ATTRIBUTION.md with modification
notice/date, upstreamcitation and researchaddition scope. REPOSITORY.md updated.
No commit/push performed. User handles data publication and academic author details.

## Latest — Git publication preparation, 2026-10-29

User requested repository initialization and sensible gitignore. Checkout already
has FC Portugal history on main and origin=https://github.com/m-abr/FCPCodebase.
Preserved history/remote; replaced broad legacy ignores with specific cache,
environment, native-build, runtime and research-artifact exclusions. experiments/
(~27GB),artifacts/,rawtraces,datasets/checkpoints ignored; code,tests,reports,
figures and required upstream .pklmodels remain trackable. Removed blanket
hiddenfolder and root/test* exclusions. Added REPOSITORY.md research entrypoints
and publication/artifact guidance. No files deleted, staged, committed or pushed.
Before publishing user must configure their ownremote. Data preparation remains
with user; author/publicationURL details remain theirs to finalize.

## Latest — second paper feedback incorporated, 2026-10-29

Updated FINAL_PAPER_DRAFT.md and IEEE LaTeX package; previous draft/package
archived under evaluation/report_archive/paper_before_second_feedback_2026-10-29_r1.
Abstract/conclusion now explicitly finite-budget and convergence-unproven.
Added six-controller original-scene RQ3 table (progress, forwardvelocity,
dimensionless unit-vector directionerror, MSE, SWD). Surface section now centres
matchedBC: on-policy0.01343 vs old0.29098; errorratios1.60 vs1.63 are descriptive.
Nearby Figure5 points are DAgger64500updates/0.10995 vsBC66500/0.11047, NOTexact
equalupdates; do not claim wholegap is causally explained by updatecount.
Added verified Beukman2024,Ingram2023,Ghasemipour2020,Hussenot2021 references with
scope qualifications. Priorprobe figure retained as warning about posthoc
explanations; it uses variedcontinuation, not originalmatchedBC. No new fits.
User explicitly will handle authors/affiliations/emails/publicURL at submission;
leave neutral localplaceholders. User's quoted informalauthorblock isn't local.
Paperzip rebuilt; source/assets checked, TeXcompilation/pagecount still pending.
Slides and explanationdoc unchanged in this revision.

## Latest — IEEE conference layout and named citations, 2026-10-28

Supervisor supplied Overleaf IEEE conference template and explicitly requested
author names rather than citation numbers. Paper generator/main.tex now use
IEEEtran conference class, IEEE author blocks/keywords and default layout;
removed article geometry/font overrides. natbib authoryear + plainnat implement
the requested exception to IEEE numeric citations. README explains citep/citet.
Generator also refreshes portable paper_latex.zip. Prior package archived under
evaluation/meeting_archive/paper_before_ieee_2026-10-28_r1. No results, slide
deck or explanation notes changed. Compilation/page count still pending in
Overleaf or a TeX installation with IEEEtran; author details remain placeholders.

## Latest documentation — regression-probe explanation, 2026-10-28

Expanded section13 of evaluation/presentation/CONSOLIDATED_PAPER_NOTES.md only
with classifier-to-regression motivation, ridge objective, actual fitter/splits,
physical/circular/binary/future target handling, RMSE/R² and controls, plus a
spoken explanation and references. Previous notes archived with suffix
2026-10-28_before_regression_explanation_r1. User explicitly requested the
explanation document, not changes to paper or slides; those remain unchanged.
No probe fitting or research code changed. Paper compilation remains next step.

## Latest — visual paper and presentation revision, 2026-10-28

FINAL_PAPER_DRAFT.md rewritten to incorporate submission controls centrally:
optimization-sensitive fidelity ranking, narrower history claim,zeroactioncontrol,
retrospective curves and unresolvedconvergence. Approximately3kwords, sixfigures.
evaluation/paper/main.tex + references.bib + sixvectorfigures generated from the
Markdown; paper_latex.zip is the portable bundle. NoTeXcompiler installed:
compilation,floatplacement and eight-body-page target need checking in Overleaf.
Author/institution/publicarchiveURL intentionally await user details.

LatestMarpdeck now23slides with10chartslides and a rendered PDF fallback:
evaluation/presentation/CONSOLIDATED_PAPER_SLIDES.pdf. Elevenfigures generated
from savedJSON with sourcehashmanifest via build_final_figures.py. PDF uses
render_final_deck.py, samecontent with a measuredMatplotliblayout, notMarpengine.
Slidebounds checked and representative renderedpages inspected. Priorpaper/deck/
notes archived with2026-10-28_before_visual_revision_r1suffix. Notes/READMEupdated.
No research artifacts or pipeline fingerprints changed. Next:reviewdraft,
compileLaTeX in desiredtemplate and confirm eightbodypages; do not claim
convergence or probe results for newBCcontrols. Optionalstudies remainseparate.

## Latest — submission controls complete and interpreted, 2026-10-28

All7BCfits and274newdeployments complete in experiments/submission_controls/v1.
Summary/curve output hashes verified; historical references reused, not rerun.
Report: evaluation/SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md. Latest focused
Marp deck/notes updated to22slides, prior versions archived before edits.
Matched-originalBC testMSE0.01024 beats oldDAgger0.01535 at211050updates;
oldBC was0.11047. This reverses the initial fidelity ranking. Frozen matchedBC
sampledMSE0.00841,0falls/30min. Varied-onlyBC200epoch hidden/inputMSE0.04210/
0.03775 narrows continuation gap (0.01995/0.01986), not eliminates it.
Zero-nativeactions achieve no useful progress;25savedcheckpoint curves complete.
Convergence not established: all newfits exhaust budgets, bestepochs nearend.
Optional early-stopdiagnostics not part of this completed batch. New BC controls
not probed. Immediate next task: incorporate addendum into final paper draft;
do not repeat earlier algorithm-only/pretraining-only interpretations. Optional
larger-budget GAIL remains separate. No new experiments launched for write-up.

## Latest documentation — setup/Python workbook, 2026-10-28

Created docs/learning/EXPERIMENT_SETUP_WORKBOOK.md and a21-page PDF, plus
render_workbook.py and README. Explains Linux/Conda/base-versus-interpreter,
system simulator/ports/synchronization, native ABI builds, GPU checks, Python
and array contracts, experiment routes/viewer, reproducibility and troubleshooting.
Six exercises progress from interpreter inspection to synthetic regression/budget
checks and a user-run live smoke. Includes answers/readiness checklist and primary
software references. Rendering uses existing Matplotlib, not a new dependency.
All Python examples parse and bash examples pass syntax checks; four synthetic
Python exercises executed successfully. Representative rendered pages inspected.

No environment installation, simulator/full research run or pipeline source was
changed. Runtime portability limitations documented in CHALLENGES. Submission
controls remain the research next step below; the guide is onboarding material,
not new experimental evidence. Editable source should be regenerated alongside
the PDF after future guide changes.

## Latest — submission-control pipelines ready, 2026-10-28

Implemented submission_controls/ and executable run_submission_controls.sh.
Defaults: original BC211,050updates, six varied-only BC fits (hidden/input,
seeds42/43/44) at200total epochs, zero native gait-action baseline, retained-round
fidelity/resource curves. Reuses saved initial weights/data and audited old
evaluations; no old fingerprinted Python source or checkpoint changed.
274newfull deployments,38savedrounds,3000active steps each. Optional --diagnostics
is a separate plan/output with standard fits plus early-stop BC/fixed-aggregate
DAgger. See submission_controls/README.md and PIPELINES.md for exact contracts.

Next user commands: ./run_submission_controls.sh plan; audit; run --quick;
then run after inspecting smoke. No manual server; no internal human checkpoints.
Training/evaluate/curves/report stages also separate. Reports under
experiments/submission_controls/v1/evaluation/summary/; curves under curves/.
Complete jobs skip via checksums; partial jobs archive/restart from job boundary.
Ten offline tests passed; real-parent full7job audit passed; tiny real-data
offline training smoke passed in /tmp. No simulator or full research training
launched. Do not claim these controls have changed the research conclusions yet.
Offline curve integration also passed: all25retained original checkpoints scored,
plot generated, final MSEs reproduce the report (BC0.1104669, DAgger0.0153452,
GAIL-BC0.1104231, GAIL-DM0.0157221). A second tiny-training invocation verified
and skipped all completed jobs. Generated diagnostic artifacts are in /tmp only.

## Latest — submission concerns audited, 2026-10-28

Follow-up checks complete: original DAgger iteration7 best epoch50/50;
validation0.0094085(epoch40)→0.0089467(epoch50). Scratch varied final DAgger fits
best47–50; varied continuations vary12–50. Neither original method has demonstrated
convergence. Exact BC matching is211050updates=317epochs+245minibatches at batch256.
Actions are native16-D gait outputs, not dataset-standardized: zero latent and
expert-mean are different controls; gait/IK/EMA remain active. Early-stopped BC
alone would not establish both-method convergence. No controls launched or code
modified; see CHALLENGES for evidence and experimental-design qualifications.

User supplied critical submission review and asked which concerns are valid and
how to address them. Read-only audit found all13consolidated BC jobs have best
validation epoch99/100; original BC validation improves0.1107255→0.0981507 from
epoch90→100. BC convergence is not established. Prior finite-budget results remain
valid but algorithm-only/pretraining-only interpretations need qualification.
See latest CHALLENGES entry. Proposed next controls: matched-exposure original BC,
matched-lifetime varied-only BC, zero/constant latent-action baseline, retrospective
MSE curves from retained original DAgger/GAIL round checkpoints. None launched or
implemented; no paper/slides/results rewritten during audit. Caution: MSE reversal
does not prove covariate shift; conservative GAIL is still adversarial imitation.
Optional larger-budget GAIL remains separate from the completed paper evidence.

## Latest documentation — consolidated-paper presentation, 2026-10-26

Slide6 now explicitly contrasts the earlier BC-led baseline assessment with
consolidated mixed DAgger's lower action MSE (7.2x in the original scene), while
BC retains slightly greater mean progress. Notes explain new model lineages and
54m versus17m optimizer exposures, not a label swap or isolated algorithm effect.
Prior deck/notes archived with suffix2026-10-26_before_bc_dagger_comparison_r3.md.

Latest refinement: slides13–15 and their notes now interpret the earlier
behavioural findings rather than survey represented features. They connect
DAgger's fidelity to selected posture readouts, limited GAIL changes to near-
initializer readouts, and continuation/scratch differences to conflicting CKA
and prediction rankings. Associations are explicitly not causal explanations;
context and mechanical findings have stated explanatory limits. The preceding
19-slide deck/notes are archived with suffix
2026-10-26_before_results_focused_probing_r2.md. Still19slides; no new experiments.

Created evaluation/presentation/CONSOLIDATED_PAPER_MARP.md (now 19 slides, 0–18) and
CONSOLIDATED_PAPER_NOTES.md (matching detailed explanations and terminology).
Narrative: standardized original-environment restart → frozen sampled-slip
evaluation → varied hidden/input continuation → scratch-initialization extension
→ general/expert probes. Numerical evidence comes from the consolidated report
through its completed Appendix F, not historical FiLM or optional later runs.
Revised to the user's focused outline; removed standalone algorithm/architecture,
counting and editorial-allocation recap slides. The final slide covers excluded
but interesting follow-on research: wide-slip stress/context-aware RL, larger-budget
GAIL and separate R1 transfer/interventions. It is not a list of background cut
for brevity. Both original 25-slide documents were fully archived under
evaluation/meeting_archive/ with suffix 2026-10-26_before_focused_outline_r1.md.
The companion matches all19slides. The deck maps the findings onto the eight-body-page final-paper draft; references
are outside the limit. Historical meeting summaries/decks remain unchanged.

Next presentation task: user-run Marp conversion and layout review. Source
structure/links checked; no PDF/PPTX generated or visual rendering validated.
No simulator, model, experiment receipt or pipeline source changed. Optional
GAIL-improvement work below remains separate; its live execution status has not
been inferred from this documentation task.

## Latest: experimental GAIL improvement ready — 2026-10-19

User asks for larger-budget GAIL with random/BC/DAgger initializers and explicitly
chooses original environment first. Implemented isolated gail_experiments/ and
run_gail_improvement.sh; uses consolidated BC and mixed DAgger, not historical
original-DAgger source. Defaults350ksteps×3initializers×3seeds=3.15m interactions.
No old Python pipeline/checkpoints changed; paper remains separate.

New recipe: no BC anchor, noisy expert actions matching current exploration,
R1 discriminator penalty, positive clipped occupancy reward, adaptive3e-5initial
actor LR, exact KL target.01/max.02 with per-epoch optimizer/actor restore/retries.
No promise of improvement. Optional --recipes legacy adaptive isolates larger
budget from changed recipe (18actors). Same initial std.2 for both recipes.
Validation every70k, baseline0 included; confirmation uses disjoint task seeds,
compares initializer/selected/final and reports fidelity plus mechanics/progress.

Next user: ./run_gail_improvement.sh run --quick (separate smoke_v1), then
./run_gail_improvement.sh run after checking smoke. plan/audit offline;
train/evaluate separable; report does no simulation. Owned servers, shared lock,
hash-verified resume. Read gail_experiments/README.md. Real parent audit passed
in /tmp;26combined offline tests passed. No live research training or evaluation
launched. Optional wide-slip sweep remains independently available.

## Latest task — paper draft and optional stress pipeline, 2026-10-18

Created evaluation/FINAL_PAPER_DRAFT.md: candidate final-paper narrative,
research questions, numerical comparisons and conclusions; eight body pages,
references excluded. Paper remains completed IL/context/initialization/probing
study; optional RL work is explicitly outside it. Meeting summaries unchanged.

New `./run_optional_robustness.sh plan|smoke|evaluate|report|report-smoke`:
oracle-only slips .1/1/5/10/20/50/100, seeds101/202/303,3000active steps.
Reuses existing passive ODE audit and fresh owned server per deployment;
installed physics/checkpoints untouched. Smoke7runs, full21separate runs,
resume verifies/skips receipts. Default experiments/optional_robustness/oracle_wide_v1.
13 offline tests pass; no simulator or research run launched. Next user: plan,
smoke, then full evaluate; inspect absolute progress AND enabled solver slip.
See optional_robustness/README.md. Wider frozen-policy evaluation and IL/RL
training remain design-only until failure/transition ranges are measured.

## Authoritative latest results — scratch comparison complete, 2026-10-16

The scratch extension is complete: 28 new checkpoints, 929 new evaluations,
224 new probe fits. Combined totals: 56 policies, 1,949 deployments, 524 fits,
131 representations and 15,720 probe scores. All new checkpoint hashes and
929 evaluation / 224 new / 300 parent probe artifact receipts verified; parent
evaluation records reused exactly and combined probe scores match metrics files.
This supersedes older pending scratch-execution instructions below.

Consolidated results report Appendix F now covers design, performance, fidelity,
mechanics, context and probes. Previous full report archived in
evaluation/report_archive/; parent sections remain identified as parent evidence.
Scratch BC/GAIL-BC gain about 0.4m sampled progress but have about 5.6x worse
held-out action MSE than continuation. Scratch mixed DAgger retains about 10.8x
better fidelity than scratch BC. Direct random GAIL fails useful locomotion
(varied mean -0.367m, score -0.0157). Context gains remain inconsistent. Scratch
DAgger CKA can increase while fidelity worsens; random-start GAIL has informative
oracle-state probes despite failed walking. Decodability is not causal use.

Next: discuss updated report/final-write-up conclusions; no rerun needed for
completed results. Lifetime compute is unmatched; controls/draws are reused;
R1 remains deferred. No live runs or model changes during this review. The report
builder includes evaluation/analysis/scratch_summary.py. Meeting brief/deck unchanged.

Latest documentation update (2026-10-16): the consolidated results report now
contains Appendix E with 15 verified paper references, a method-to-source map,
and attribution boundaries for the final write-up. Its builder includes
`evaluation/analysis/consolidated_references.md` on regeneration. Numerical
results, meeting summaries and experiment artifacts were not changed; the
scratch-comparison execution status is not inferred from this documentation update.

Read this file first when resuming work on the FC Portugal Walker imitation
learning project. Then consult `EXPERIMENT_PHASES.md`, `PIPELINES.md`, and
`CHALLENGES.md` for the full experimental, implementation, and failure history.

## Project objective

Treat the native FC Portugal Walk controller as an expert oracle. Train and
compare behavioural cloning (BC), DAgger, and GAIL policies for faithful walking
imitation, task performance, mechanical stability, and resource cost. After the
nominal comparison, test frozen policies under controlled domain changes and
then under joint/motor impairments relevant to RoboCup collisions and wear.

## Current experimental status

### Consolidated training/evaluation ready for user smoke — 2026-10-15

User requested fresh standardized batches, reopening hidden/input DAgger/GAIL
context work (supersedes prior pause for this new generation), excluding FiLM.
`run_training.sh` and `run_evaluation_all.sh` own fresh servers; no prestarted
server. BC170k training samples; mixed DAgger100k seed+70k learner labels;
GAIL170k shared static expert pool+70k PPO steps (user explicitly chose70k).
Held-out validation/test10k each extra; inherited and unique collection costs
recorded separately. Four new original models, then24varied models (four methods,
hidden/input, seeds42/43/44). GAIL uses corresponding new BC or mixed DAgger,
including context architecture; never historical original-DAgger GAIL weights.

Original environment preserves installed legacy scene bytes; varied uses scalar
.025/.1/.2/1/10. Latest33 varied rounds reused; full correct-context evaluation
1020 deployments including30original and990varied. Old experiments/launchers
preserved. Read consolidated/README.md and PIPELINES.md for exact commands.
Next user: small separate `--quick --output experiments/consolidated/smoke_v1`
training then evaluation; then full plan/run. Offline synthetic BC/PPO, lineage,
sample budgets, scene and receipt tests pass; no simulator or research fits run.
R1 remains deferred. Existing meeting brief/deck are historical, not updated here.

### Slide explanation companion added — 2026-10-12

`evaluation/presentation/SLIDE_EXPLANATIONS.md` explains all 24 slides in the
current Marp deck, including RMS/MSE, SD, ridge/R2, FiLM, feature families, CKA,
effective variance dimension, experimental controls and spoken takeaways.
Linked from presentation README. Title counts as slide 1. Deck and current
meeting brief unchanged; no research work or conversion launched.

### Meeting slides refocused on newer work — 2026-10-12

Revised `evaluation/presentation/MEETING_SUMMARY_MARP.md`: 24 slides with just
two recap slides, expanded context/FiLM/repeatability methods and results, multiple
probe feature families, expert property/CKA comparisons, spectra/reset diagnostics
and limits. Previous full deck archived in evaluation/meeting_archive before
editing. Written summary/results unchanged. User will perform Marp conversion;
only Markdown structure/local links checked. No new experiment or scope change.

### Marp meeting deck prepared — 2026-10-11

`evaluation/presentation/MEETING_SUMMARY_MARP.md` presents the consolidated
full-study brief, with Marp front matter, tables, source links and speaker notes.
Conversion commands in presentation/README.md. Historical Pandoc deck and
meeting brief unchanged. Marp CLI absent; no rendered export or live experiment
was run. Preserve deck snapshots before future revisions alongside brief history.

### Full-study meeting brief consolidated — 2026-10-11

`evaluation/MEETING_SUMMARY_2026-10-10.md` now covers the original four-policy
comparison, legacy fixed/Gaussian limitations, corrected scalar validation,
75-run grouped varied evaluation, context/repeatability/FiLM and all probing.
Stale pending-run/R1 directions removed from the current brief. Exact prior brief
preserved in `evaluation/meeting_archive/`; AGENTS.md now requires archiving before
future revisions. Earlier unrecorded versions were not reconstructed. Corrected
scalar_wide_v4 has no consolidated fixed report on disk at this review; the brief
does not substitute the legacy report or assert an audited full-grid result.
No jobs launched, policy changes or new physics audit. Research direction unchanged.

### Command reference added — 2026-10-11

`COMMANDS.md` collects training, evaluation, plotting, context/FiLM, repeatability,
probing and demo commands. It distinguishes manually started legacy/demo servers
from managed experimental servers, documents safe resume and the RoboViz menu.
Linked from README/PIPELINES. Documentation only; no experiments launched or
research scope changed. The immediate research direction below remains unchanged.

### Latest scope decision: R1 transfer deferred — 2026-10-11

User reports difficulties with R1 transfer in the alternate directory and excludes
that aspect from the current study until progress resumes. Cause is unspecified;
this is a scope decision, not evidence for negative transfer or failed source
representations. Do not make R1 outcomes a prerequisite for continuing Walker.
Earlier next-task directions to relate probes to R1 learning are deferred.

Current focus: NAO imitation performance/stability/domain experiments, context BC
findings, and broad expert/student representation analysis for BC, mixed DAgger,
original-DAgger GAIL and GAIL-BC. Completed general/expert probes remain valid as
descriptive source-domain results. Frame their contribution as which measured
properties remain accessible or are reorganized through imitation, with geometric
similarities and control limitations; avoid explaining or ranking R1 initialization.
Next direction is consolidation of this NAO evidence and warranted probe controls.
The previously discussed nonlinear regression/logistic contact readouts remain
possible follow-ups, not implemented or launched by this scope update.

Meeting summary/phase ledger and R1 handoff updated. Keep R1 material for future
work; no deletion, adjacent-repository edits or process changes were requested.

### Latest: expert comparison complete and interpreted — 2026-10-10

Full audit passed:16 new fits, geometry,65 parent receipts, current sources/models/
data and oracle action equivalence;2400 scores/240 gaps/20 profiles match artifacts.
Read experiments/representation_probing/nao_expert_comparison_v1/INTERPRETATION.md.
Mixed DAgger h1 CKA=.9355, GAIL-DAgger h2=.9314 closest pooled to expert; both
beat same-layer random controls on all15 runs. Raw/random CKA already high, so
not percent transferred knowledge. All240 student R2 gaps positive, with major
64-tanh expert versus256-ReLU student architecture confound. Expert beats its
random controls on command orientation, phase-roll interaction and future pitch;
these are teacher-related candidate properties, not causal inheritance evidence.
Expert effective variance dimension8.38; GAIL-DAgger h2 10.38; no skill-count claim.
Meeting summary/phase ledger updated. No rerun needed; next compare descriptive
hypotheses to independent R1 learning and controlled target interventions. No new
research fits, simulator/model mutations or R1 changes during this review.

### Latest: expert append-only comparison ready — 2026-10-10

User requested expert probing/retention comparison while keeping broad internal
representation discovery. Added representation_probing/expert.py and
run_expert_probing.sh, keeping original probing/run.py and general run.py unchanged
so completed results remain valid. Expert robot0 walk_R0.pkl is63->64 tanh->16,
one hidden layer (not256-ReLU). Action agreement audited over90000 source rows;
max abs error6.56e-6. Existing64 probes/36 resets reused,16 new expert/random-tanh
fits planned with same30 general labels, rows/splits/controls; no slip target.

New reports compare both student layers to expert h1 by R2/error gaps and add
pooled/per-run linear CKA, row-permutation controls and activation spectra.
These describe decodability/geometry, not percent knowledge transferred, exhaustive
representations, causal inheritance or R1 benefit. Full references/contracts:
representation_probing/EXPERT.md. Parent architecture confounds remain explicit.
Next user: ./run_expert_probing.sh audit then ./run_expert_probing.sh run.
Output experiments/representation_probing/nao_expert_comparison_v1, report automatic.
No research fits or R1/simulator work launched; old checkpoints/results preserved.

### Latest: general probing complete and interpreted — 2026-10-10

Verified64 probe receipts/1920 report scores,36 reset records, source/model/trace
hashes. Read experiments/representation_probing/nao_sources_v1/INTERPRETATION.md.
Meeting summary updated. No rerun required. Phase-conditioned roll is strongest
candidate: trained h1 R2 .586–.703 versus raw .332/random .427–.452, lower error
than each random seed on all15 test runs for every source. This is accessibility,
not causal control or R1 usefulness. BC/GAIL-BC similar; GAIL-DAgger h2 better on
gyro-x/y; mixed DAgger modestly best on future direction error. No universal rank.
Important: phase/contact signal survives block shuffles (periodic null inadequate).
Persistence is weak for future averaged/oscillatory targets; resets measure only
source action sensitivity (roughly2.5–2.8 RMS), not transfer loss.89/480 true probes
still choose minimum alpha1e-6. Preserve these caveats and completed artifacts.
Next: relate hypotheses to separate matched R1 learning/ablation results; any
stronger phase-null study is new/versioned, not a silent fix. No new fits/R1 edits.

### Latest: general representation probing ready — 2026-10-10

User corrected research focus to original NAO properties useful for R1 learning.
New isolated representation_probing/ + run_representation_probing.sh; old slip
study retained as preliminary, no longer main direction. Four originals: BC,
mixed DAgger, original-DAgger GAIL, GAIL-BC. R1 work remains in alternate directory.
Read representation_probing/README.md. Commands audit/run/report; output
experiments/representation_probing/nao_sources_v1. User launches research fits.

30 general gait/force/pose/command/phase-posture/future labels; no slip target.
64 multitarget fits with raw/phase-command/random/shuffle controls; in-memory
36 layer-reset action-sensitivity comparisons, never modifying original files.
No R1 adapters/training or transfer-utility claims. Same30 verified oracle runs,
whole-run10/5/15 split; stride7; reset/window exclusions;4270/2133/6405 rows.
Real-data audit and10 offline tests passed. No research fits/live jobs launched.
Immediate next: user runs new pipeline and reviews general property profiles;
R1 outcomes and controlled target fine-tuning are required to test usefulness.
Do not resume the older slip sweep or context DAgger/GAIL expansion by default.

### Latest: probing complete and interpreted — 2026-10-10

All60 fits/300 scores in experiments/probing/original_policies_v1 verified:
current plan inputs/source/model hashes, receipts and report-to-metrics equality.
Read INTERPRETATION.md there; meeting summary updated. Do not rerun to finish.
GAIL-DAgger has highest slip R2 (.461 h1/.427 h2); BC/GAIL-BC almost identical.
Random features already decode much slip (.404–.424 h1/.361–.396 h2), limiting
learned-dynamics claims. True slip/velocity outperform shuffled labels; posture
shuffles retain signal, so not ideal nulls.74/75 true probes hit minimum alpha.01.
Next discussion: separately versioned weaker-regularization sensitivity before
strong model rankings, and compare hypotheses against independent R1 outcomes.
No causal-use/transfer claim, new probe fitting, simulator or R1 edits in review.
Context DAgger/GAIL expansion remains stopped; user works on R1 elsewhere.

### Latest: context expansion stopped; probing pipeline ready — 2026-10-10

User opts NOT to extend conditioning to DAgger/GAIL after marginal BC/FiLM gains.
This supersedes older extension/ablation next tasks. Meeting summary created at
evaluation/MEETING_SUMMARY_2026-10-10.md with conclusions and mechanism caveats.
R1 transfer is user-run in the adjacent directory; no changes there.

New offline `probing/` + `run_probing.sh`: only root BC, mixed DAgger, original
DAgger-initialized GAIL and GAIL-BC. User confirmed "GAIL DAgger(Mixed)" was typo.
Frozen h1/h2 regression probes, shared oracle states, raw/random-network and
block-shuffled-label controls. Alain/Bengio primary reference; Hewitt/Liang control
motivation. See probing/README.md for exact methodology and interpretation limits.
Audit passed on actual30 oracle runs: 2990 train/1493 validation/4485 test rows.
Whole-run10/5/15 split; no new simulations. Research fitting has NOT been launched.
Next user commands: `./run_probing.sh audit`, `./run_probing.sh run`; report automatic
or `./run_probing.sh report`. Output experiments/probing/original_policies_v1.
Do not claim causal use, friction estimation or superior R1 transfer from decoding.
Existing models and immutable experimental sources/results unchanged.
Validation:17 offline tests passed (six probing plus context-evaluation/FiLM
regression tests); real-data audit, shell syntax and diff whitespace checks passed.
Synthetic temporary fits only; no research probing output has been generated.


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

### Repeatability diagnostic completed and verified — 2026-10-10

All45 repeatability_v1 deployments verified (receipts, artifacts, summaries/traces),
3,000 steps each, seed42 hidden/input pair plus oracle,5 repeats per three cases.
Reports under experiments/context_conditioning/bc_input_v1/repeatability_v1/.
Same-controller progress SD spans.2405–.6181m; mean pairwise absolute differences
.2990–.7426m. Hidden/input means: high17.1220/16.9664m, low43.5631/43.7708m,
nominal37.4928/37.5096m. No consistent progress advantage established.
High-slip falls: oracle4,hidden2,input0; low:oracle2,hidden0,input1; nominal all0.
These are event totals over5 runs each, not failure probabilities.

Initial targets match exactly within every repeated case/controller, but first
active planar positions differ by up to6.36–24.12mm across groups. Largest initial
observation spread is coordinate2 (loc_head_z_vel/2), followed mainly by height and
torso angles. Supports unequal starting/settling states, not a proven mechanism or
causal explanation of all return variability. This spread cautions against strong
claims from the prior small context effects; it is not a universal significance
threshold or evidence of no possible effect. Only one trained pair/three cases.
Next: proceed with FiLM append-only study if desired under unchanged protocol,
report the repeatability caveat; a startup fix would need new matched controls.
No code/protocol changed or runs launched during review. No completed FiLM summary
was found at this check; existing original controls remain preserved for reuse.

### Adjacent R1 workstream handoff prepared — 2026-10-10

User wants Walker context-model work to continue here, with original context-free
policies later compared as initialization/representation sources for target R1 RL.
Created UNITREE_R1_TRANSFER_HANDOFF.md for the agent working in the adjacent
/home/pamnt/robocup/unitree_rl_mjlab repository. First target deliverable is a native
Unitree-R1-Flat PPO baseline from random initialization; transfer and later probing
follow. This updates the earlier strict "R1 only after all NAO work" sequencing:
target baseline preparation is a separate workstream, not a replacement for context
experiments. No adjacent files changed, target training launched or models copied.

Handoff contains source checkpoint hashes/variants,63-D observation slices,
16-D latent gait/IK contract (not R1 motor commands), target inspected file paths
and PPO defaults, baseline/architecture matching, candidate middle-layer transfer,
and artifacts/controls for future probing. Original-vs-mixed DAgger and GAIL source
set needs confirmation before a sweep; gail_policy.pth absent, named variants
present. Native R1 actor uses512/256/128 ELU and different obs/actions, so direct
weight loading is not compatible. Keep Walker original files/results read-only.

### Latest: repeatability and append-only FiLM implemented — 2026-10-10

User requested repeatability then FiLM; clarified NOT to repeat old hidden BC
evaluation as part of FiLM. Optional separate diagnostic:
`./run_bc_repeatability.sh evaluate` =45 runs (five identical repeats at three
prior low/nominal/high slip-task pairs; oracle+seed42 hidden/input). Startup is
unchanged, so this measures variance rather than fixing it. `plan`/`report` offline.

FiLM training: `./run_film_training.sh audit`, then
`./run_film_training.sh train --device cuda` (match completed CUDA controls).
Default only3 new FiLM checkpoints, seeds42/43/44; originalBC initializer and
existing100k/10k/10k data/budgets. Adds1024 identity-initialized modulation parameters
at final256 hidden features; metadata checkpointfilm_bc_v1. Separate film_training/
package and bc_film_v1 output; old code/data/checkpoints unchanged.

After training: `./run_film_evaluation.sh evaluate --suite smoke` (3 new runs), then
`./run_film_evaluation.sh evaluate` (99 new FiLM runs across33 exact prior rounds).
It never launches old controls: verifies and reuses saved oracle/hidden/input-
correct results, writes separate comparison.md/json. Optional3 context modes
require297 new FiLM runs; new output if changing modes. No new Gaussian sampling.
Previous confirmation draws are reused, not untouched FiLM evidence. Strict
budget/data/initializer/device/library and baseline physics/source compatibility.
Guides: film_training/README.md and context_training/README.md. All user-run live
jobs/research training remain pending; no simulator launched during implementation.
Prior429-run evaluation identity checked unchanged. Next: user chooses optional
repeatability diagnostic and proceeds with FiLM training/smoke; interpret with
startup-variation and additional-capacity caveats. Do not rerun completed controls.
Validation:38 offline tests passed; actual oracle shards audited, repeatability
schedule constructed offline (45 runs), shell syntax/compilation checked. FiLM
updates in tests used temporary synthetic data only, not research training.

### Latest: all 429 BC context evaluations complete and reviewed — 2026-10-10

evaluation_all_v1 under experiments/context_conditioning/bc_input_v1 has all429
verified completion receipts/artifact hashes and report.md/csv/json. Do not rerun
training or evaluation merely to finish this stage. User requested interpretation.
Across three fine-tuning seeds, hidden/correct-input mean progress respectively:
grouped33.07294/33.06027m; confirmation32.29152/32.66458m;
nominal39.27243/39.17469m. Corresponding falls2/3,3/1,0/0 (45,45,9 runs per arm).
Constant/wrong context does not consistently degrade live progress. About76% of
the net confirmation progress advantage comes from one seed44 hidden failure at
slip0.0002849946 (below minimum training slip.025); retain this real result, but
do not present it as broad evidence of context-dependent adaptation.

Critical repeatability caveat: nominal correct and constant use identical models,
slip.1 and task seeds, yet paired progress differs by0.463m mean absolute across9
pairs (signed range -1.368..0.673m). First observations already differ in all three
nominal_0 pairs (maxabs differences.061..179; position differences3.3..4.9mm).
Thus matched task seeds do not reproduce identical starting states; cause not
diagnosed. Prioritize controlled repeatability checks before strong small-effect
claims or expanding experiments. No simulator changes made.

Read-only counterfactual inference on existing10k held-out states confirms context
is not ignored: changing correct to extreme-wrong increases action RMS difference
by about.0056 and mean expert MSE by~1.47%; constant-context MSE increases~.41%.
Correct-input held-out error remains essentially equal/slightly worse than hidden
BC. This supports weak context sensitivity, not a demonstrated locomotion benefit.
Next: review these caveats with user; BC FiLM pilot remains the next architectural
comparison, but quantify/control startup variation before attributing small gains.
Detailed counts/reference: context_training/README.md. No new live jobs launched.

### Documentation clarification: three seeds are fine-tuning replicas

User expected one model per technique; documented the context pilot's three
checkpoints per architecture in context_training/README.md ("What the training
seeds mean") and PIPELINES.md. Seeds42/43/44 mainly vary minibatch order, not data,
surfaces, splits or pretrained initialization. Hidden/input matching seeds share
batch order. Six checkpoints are independent deployed policies, not an ensemble;
their variability is conditional on a common initializer/data. Evaluation task
seeds and Gaussian schedule draws are separate. Run-count derivations105/429 and
the hypothetical single-pair45 are recorded; no seed-selector or default change
was requested/implemented. All models/results preserved; no jobs started.

### Latest: context BC live evaluation implemented, awaiting smoke — 2026-10-09

Added context_training/evaluate.py, eval_worker.py, test_evaluation.py and
run_context_evaluation.sh. Original training/model/collector sources unchanged;
all six trained checkpoints verified. User command:
`./run_context_evaluation.sh evaluate --suite smoke` (7×3,000 active steps).
After smoke, full comparison:
`./run_context_evaluation.sh evaluate --suite all --modes correct constant wrong`.
This runs 429 deployments: 15 historical grouped+15 new confirmation+3 nominal
rounds, each with oracle, 3 hidden models and 3 input models×3 context modes.
Default grouped/correct comparison is 105 runs. No prestarted simulator; owned
server per run, shared lock, no internal approval prompts, one retry by default.
Repeat identical command to resume; exhausted errors stop with artifacts preserved.

Results: <training-root>/evaluation_<suite>_v1/ with plan, runs and automatic
report.md/csv/json. `plan` is offline, `report` regenerates offline; same flags.
Correct context uses serialized scene slip; constant=.1; wrong=10 below physical
slip1, otherwise .025. All modes keep physics fixed. Paired metrics retain training
seed/suite; no random baseline or exact 22-D Wasserstein claim. Full contracts in
context_training/README.md and PIPELINES.md. No live evaluation launched here.
Validation: 33 offline tests passed (context worker/orchestration, training and
legacy variable pipelines); real six-checkpoint schedule construction, shell
syntax and compilation passed. Live smoke is still pending user execution.
Next: user smoke/full evaluation; diagnose if needed, then interpret before FiLM.
Older pending-evaluator notes below are superseded. Existing 75-run interpretation
write-up remains outstanding.

### Latest: input-conditioned BC training complete — 2026-10-09

User completed all six offline runs in experiments/context_conditioning/bc_input_v1:
hidden/input architectures, seeds 42/43/44, 100 epochs, batch256, Adam1e-4, CUDA.
All six completion identities/artifact checksums verified and checkpoints loaded
successfully. Summary contains six runs. Mean held-out latent element MSE:
hidden 0.0018085948; input 0.0018099182. Paired input error is higher by
0.149%, 0.035%, 0.035% respectively: essentially unchanged predictive fidelity,
not evidence of a locomotion gain or of context being ignored. Shared warm start
means these seeds measure fine-tuning variability, not independent pretraining.

Next task remains implementing live context evaluation with owned servers,
paired surfaces/task seeds and correct/constant/wrong-context ablations. No live
context evaluation exists yet; do not feed policy.pt into the legacy evaluator.
Do not retrain these completed models just to proceed to evaluation. FiLM remains
planned. No jobs launched during this completion check.

### Latest: input-conditioned BC offline pilot implemented — 2026-10-09

User asked to begin input conditioning. Added isolated `context_training/` and
`run_context_training.sh`; old Python pipelines/checkpoints/results are untouched.
`./run_context_training.sh audit` passed on actual varied-arm oracle data:
100k train, 10k validation, 10k test across .025/.1/.2/1/10 (40 trajectories).
`train` runs hidden/input arms with seeds 42/43/44, 100 epochs, batch256, Adam1e-4,
CPU default (`--device cuda` optional). No research training was launched.
Both arms start at root bc_walk_policy.pth with matched batches/budgets. Input
model is 64→256→256→16; slip encoding log1p(slip)/log(11), new input weights zero.
Source NPZ remains 63-D; metadata-aware checkpoints use context_bc_v1, policy.pt.
See context_training/README.md and PIPELINES.md for contracts, artifacts and resume.

Immediate next implementation: live context evaluation under isolated owned
servers, with paired grouped rounds, nominal retention, untouched confirmation
draws and correct/constant/wrong supplied context logged separately from physical
slip. No live evaluation support yet: do not use legacy evaluate_policy.py to
load context checkpoints. FiLM and conditioned mixed DAgger/GAIL remain planned.
Synthetic offline training tests are engineering checks, not experimental results.
Verification: 7 new context tests and 20 existing variable-training tests passed;
shell syntax and Python compilation checks passed. Real-data audit was read-only.
The full interpretation document for the previous 75 grouped runs remains pending.
The user initiates research training and live jobs.

### Next stage approved: input conditioning and FiLM — 2026-10-09

User requested a plan for both architectures. See the top of EXPERIMENT_PHASES.md
for the protocol and references (Beukman et al. 2023; Perez et al. 2018; Ha et al.
2016 for the deferred hypernetwork alternative). BC pilot compares matched-budget
context-hidden, 64-input concatenation, and 63-input backbone with a scalar-slip
FiLM branch; all output the same 16 latent actions. Extend to mixed DAgger,
GAIL-BC and GAIL-DAgger after validating the pilot. Oracle/IK unchanged.

Immediate next implementation task: design isolated context-model/data contracts
and the BC pilot, with behavior-preserving initialization, common context scaling,
three paired training seeds proposed, and correct/constant/wrong-context tests.
Retain fixed-grid training and fixed-per-run grouped evaluation; add untouched
confirmation draws and nominal retention. Existing 75 runs are complete, not
pending rerun. Their full interpretation document is still outstanding. No context
models or commands are implemented yet; this change only updates planning docs.
Do not start live jobs or overwrite existing plans/checkpoints. Update PIPELINES.md
when implementation begins. This next-stage decision supersedes the next-task
wording in the completion snapshot below.

### Latest: grouped varied-arm evaluation complete — 2026-10-09

The user has completed all **75/75 deployments**: 15 paired slip/seed rounds
across oracle, bc_varied, dagger_mixed_varied, gail_bc_varied and
gail_dagger_varied. All 75 completion receipts are present; an offline integrity
check verified all 750 referenced artifact hashes. `report.json` contains 75 runs
and `report.md` is present. The rounding blocker described below is resolved;
the earlier instruction to resume the remaining 15 runs is now superseded.

Authoritative results directory:
`experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/evaluation_grouped_v1/`

**Immediate next task:** read `report.md`, `report.json` and `schedule.json`,
interpret the overall and per-group results, and write a research interpretation
alongside the reports. Compare against prior corrected-scalar fixed-slip and
Gaussian evaluations where protocols permit; distinguish trained from frozen
policies and do not mix legacy two-argument slip results into physical robustness
claims. No further evaluation or retraining is needed to begin this analysis.
This handoff update verified completion/integrity, not performance conclusions.

Protocol reminder: training used the fixed grid .025/.1/.2/1/10. Evaluation uses
three draws for each parent Gaussian with mean at a grid value and std=0.5*mean,
truncated to (0,10]. Each sampled slip stays constant for a whole 3,000-active-step
run and is shared with the task seed across all five controllers in that round.
This is not per-contact or spatially random slip. The mean-10 parent is
upper-truncated, so its realised draws need not average 10. Oracle ratios are
not random-normalized scores; this suite has no random baseline. Only the varied
arm's grouped completion is established here, not the nominal continuation arm.
User initiates live jobs; no simulator or training was launched for this update.

### Historical: grouped rounding blocker fixed at 60/75 — 2026-10-09

User's grouped varied evaluation stopped before group_4_draw_0 oracle launch:
slip5.01334232429452 serialized to5.01334232429 (.12g), exceeding the old1e-12
validation tolerance. Fixed status() to accept exact serialized encoding, preserving
writer bytes/physics. Added narrow old/new validator-hash equivalence for evaluation
resume; unknown sources/model/plan/row changes and corrupt files still reject.
All60 completed runs verified against current inputs and artifact hashes. All15
grouped scene preparations pass offline;20 tests pass. User should rerun:
`./run_variable_training.sh evaluate --arm varied --suite grouped`
then `./run_variable_training.sh report --arm varied --suite grouped`.
Existing60 skip; incomplete preparation archives and remaining15 runs proceed.
No retraining, reset or manual deletion required. No live jobs started by assistant.

### Latest: grouped Gaussian evaluation ready — 2026-10-09

User approved means at each training slip, std=.5*mean, three draws per group,
0<slip<=10. Implemented --suite grouped as CLI evaluate/report default. Fifteen
paired rounds;75 runs varied-only+oracle,135 both training arms+oracle. Each run
holds slip fixed for3,000 active steps; fresh server, same seed/slip for all models
within the round. Original frozen policies excluded by default. No retraining.

User commands: `./run_variable_training.sh evaluate --arm varied --suite grouped`,
then `./run_variable_training.sh report --arm varied --suite grouped`.
Outputs under experiments/variable_slip_training/fixed_grid_gaussian_eval_v4_seed42/
evaluation_grouped_v1/ (schedule.json, run receipts, report.md/json). Training plan
and all completed checkpoint hashes/references preserved and verified. Previous
five-round results and root reports are unchanged; use --suite gaussian for those.
Reports include per-group results; mean10 parent distribution is upper-truncated,
and positivity permits slips below .025.19 offline tests passed; no live runs
started. User initiates simulations. Older five-round-default instructions below
are superseded, not deleted.

### Latest: evaluation extended to five rounds — 2026-10-09

User requested five Gaussian evaluation rounds to match five training slip values.
Implemented scheduler-only extension in variable_training/run.py; original immutable
training plan and eight completed checkpoints remain valid (hashes verified).
First three draws unchanged; appended4.054531697160031 and6.099415186166547 with
seeds4290003/4290004. Evaluation saves separate evaluation_schedule_5_rounds.json.
Commands unchanged: `./run_variable_training.sh evaluate --arm varied` then
`./run_variable_training.sh report --arm varied`.25 runs incl oracle; both arms45.
No retraining/plan regeneration required. Matching completed evaluation rows can
resume. Quick plans stay one round, fixed suites three seeds.17 variable-training
tests pass; no live jobs started. Older three-Gaussian-round descriptions superseded.

### Latest: fixed-grid training / Gaussian test rounds finished — 2026-10-09

User clarified protocol; fixed environment values are TRAINING settings, Gaussian
draws are EVALUATION-only. Implemented and documented (variable_training/README.md).
Plan already generated at experiments/variable_slip_training/
fixed_grid_gaussian_eval_v4_seed42/plan.json. Protocol:
fixed_grid_training_gaussian_evaluation_v4. Training grid .025/.1/.2/1/10,
round-robin long/short pairs. Matched nominal arm remains .1. Budgets unchanged;
validation/test10k each now use five surface pairs covering the grid.

Gaussian parent mean5.0125/std2.5 bounded .025..10. Three rounds draw5.774292699,
6.790566449,3.413121168; all models share the round's fixed slip/task seed, fresh
server per model,3,000 active steps. Default evaluation/report EXCLUDES frozen
originals and includes oracle+trained variants (both arms27 runs, varied-only15).
Reports show round/slip/seed and aggregates. Optional baseline/frozen-only still
exists but isn't required. Twenty-six offline tests pass; no live jobs launched.

User's frozen fixed command remains `./run_varied_slip.sh fixed` (all five values,
three seeds). Do not interrupt or alter its root pipeline while running. Once
finished/reviewed, training commands are `./run_variable_training.sh all`, then
`evaluate`, then `report`; add --arm varied consistently to exclude nominal controls.
`plan` is offline/idempotent with defaults. Shared lock prevents overlap. Do not
start server manually or launch live experiments autonomously. Prior artifacts
preserved. This replaces older pending-mean/std and Gaussian-training descriptions.

### Latest requested revision: three seeds and cap 10 — 2026-10-09

Fixed default moved to scalar_wide_v4 under phase2_domain_robustness: conditions
.025/.1/.2/1/10, seeds101/202/303, eight controllers, 120 runs. Gaussian default
plan moved to variable_slip_training/gaussian_scalar_wide_v3_seed42/plan.json;
upper bound10, three paired Gaussian evaluation draws (15 frozen baseline runs).
New protocols prevent mixing prior experiments. Corrected scalar method unchanged.
Training budgets/seeds unchanged; quick plans still one evaluation draw.

IMPORTANT pending preference: user was asked whether to broaden Gaussian mean/std
to 5.0125/2.5 or only raise its cap. Until answered, defaults retain .1/.035, so
large slips are almost never sampled. Do not describe cap10 as broad training
exposure. No new immutable wide Gaussian plan generated yet; decide distribution
before `./run_variable_training.sh plan`. No live jobs started. Existing fixed and
Gaussian narrow-range plans/results untouched. Older default paths below superseded.

### Latest pipeline migration complete — 2026-10-09

User requested pipeline updates. Shared scene writer now defaults to single scalar
slip; both fixed and variable-training server managers verify scalar syntax at
startup. Both nominal/varied collection and training, plus evaluation, are corrected.
Historical diagnostic legacy arms explicitly retain legacy syntax. Installed scene
and all original checkpoints/legacy experiment artifacts remain untouched.

NEW fixed default: experiments/phase2_domain_robustness/scalar_v3/, protocol
isolated_server_scalar_slip_v3. NEW Gaussian plan (created offline):
experiments/variable_slip_training/gaussian_scalar_v2_seed42/plan.json, protocol
variable_slip_hidden_scalar_v2. Old plans/receipts cannot resume under new protocols.
Bounds, seed42 draws, budgets, policy architectures and root initializers unchanged.
Shared server lock remains in original phase2 directory, covering all launchers.

Next user commands: `./run_varied_slip.sh fixed`, then `report-fixed`; and
`./run_variable_training.sh baseline`, then `report --frozen-only`. These are new
corrected baselines, not repeats in identical physics. Review before `all`,
`evaluate`, `report` on the variable-training launcher. No user-started live batch
has been reviewed yet. Do not launch simulations autonomously. 23 offline tests
and ten real-config preparations pass. Full pipelines verify scalar scene syntax
and hashes, not continuous ODE readback; scalar smoke provides the physical check.
This supersedes older statements that the main launchers still use legacy setters.

### Latest: scalar smoke validated — 2026-10-09

User ran scalar smoke. All five oracle runs completed 3,000 steps, zero falls;
receipt artifact hashes and passive traces passed review. Enabled solver slip now
changes: field .025/.1/.2/1/10 yields classes .015-.3125 / .0525-.35 /
.1025-.4 / .5025-.8 / 5.0025-5.3. Progress falls monotonically 43.972 -> 40.546 ->
37.591 -> 28.809 -> 15.325 m. Report:
experiments/slip_sensitivity/oracle_scalar_v1/INTERPRETATION.md.

The single-argument scene method works in these runs; old two-value syntax is
strongly implicated, but exact binding rejection/fallback is not independently
proven. Preserve all old data; corrected nominal .1 is a different physical
condition from legacy nominal. Old fixed/Gaussian robustness claims remain
unsupported. Existing fixed/variable-training launchers STILL use legacy setters.
Next decision: multi-seed oracle confirmation and a NEW corrected fixed/Gaussian
pipeline version, followed by frozen-policy baselines before any retraining or
context experiments. Do not pool old/new physics or expand training bounds merely
because the oracle stayed upright at 10. No live jobs/code fixes in this review.
This supersedes older statements that scalar live validation is pending.

### Latest next action — scalar method (2026-10-09)

User requested a new way to modify slip. Implemented an isolated single-argument
scene setter experiment, not a solver override. User should run:
`./run_slip_sensitivity.sh smoke --suite scalar` with no existing simulator.
Five oracle runs (.025/.1/.2/1/10), seed101, 3,000 active steps, passive ODE audit.
Output: experiments/slip_sensitivity/oracle_scalar_v1/smoke_report.md/json.

contact_slip/scalar_scene.py changes ONLY the generated local scene expression
from `(setContactSlip v v)` to `(setContactSlip v)` and refreshes all startup/hash
evidence. Installed scene, existing experiments, native contact combining, models
and datasets stay unchanged. Manifest records single_argument_v1. Ten offline
tests and five real-config preparation checks pass; live validation pending.
No simulator started by assistant. Check changed ENABLED solver values, not
merely changed scene text or aggregate contact proportions. Root-cause hypothesis
is still unproven until this check. Do not run optional 25-run scalar evaluation
or training before reviewing smoke. Earlier large-slip test is complete; do not
rerun it or relax old receipt hashes. This section supersedes older next tasks.

### Authoritative resume summary — 2026-10-09

The user has now run the large-slip stress smoke. All SIX completion receipts
passed artifact SHA-256 checks during this handoff update. Do not rerun it merely
because older notes below describe it as pending.

- Settings .1/.5/1/2/5/10, oracle only, seed101, 3,000 active steps each.
- All settings STILL produce the same recorded ODE slip/mode classes, with slip
  values approximately .005 and .3025 in both axes.
- Progress is 45.109–45.349 m, zero falls throughout. This does NOT establish
  robustness to increased effective slip: that manipulation remains unverified.
- Report: `experiments/slip_sensitivity/oracle_stress_v1/smoke_report.md` and JSON.
  Raw contacts/configuration/logs/hashed receipts are under the adjacent smoke/.
- Earlier .025/.1/.2 passive audit reached the same finding; report:
  `experiments/slip_sensitivity/v1/SMOKE_INTERPRETATION.md`.

**Immediate next task:** diagnose why configured scene slip does not change
solver-bound contact slip. Inspect the script binding for `setContactSlip`, scene
application and surface combination. Current scene expression uses TWO values;
installed native `ContactJointHandler::SetContactSlip(float)` takes ONE scalar.
This is a diagnostic lead, not a confirmed cause. Do not silently replace the
experiment with direct contact overrides: that would be a different intervention.
Any fix needs offline checks followed by a new-output-directory passive smoke,
initiated by the user, proving effective parameters change before larger studies.

Do NOT run the optional 30-run standard/stress sweeps, parameterised training or
per-contact Gaussian experiments yet. Preserve all original/fixed/Gaussian results
and checkpoints; qualify effective-domain robustness/learning claims pending
physics validation. Original nominal comparisons remain useful. Never overwrite
historical receipts or relax source hashes to resume changed experiments.

Implementation: `contact_slip/sensitivity.py`, `contact_slip/run.py`,
`contact_slip/contact_slip.cpp`, `run_slip_sensitivity.sh`. Passive mode forwards
contacts unchanged; logs include ball/startup/recovery, not body-labelled feet.
Seven offline tests passed before the live stress batch. Design:
`contact_slip/STRESS_TEST_DESIGN.md`. Native builds are content-addressed.
No simulator was launched or code changed during this handoff update.

The user initiates live runs; scripts own servers and require free ports 3100/3200.
No interactive checkpoints after initiation. Preserve unrelated dirty worktree
changes. Update PIPELINES/CHALLENGES/EXPERIMENT_PHASES alongside relevant changes.

### Historical status updates (superseded where inconsistent above)

NEXT (2026-10-09): user requested larger-slip oracle tests. Implemented separate
stress suite; user command `./run_slip_sensitivity.sh smoke --suite stress` runs
six oracle-only settings .1/.5/1/2/5/10, seed101, 3,000 active steps, fresh servers.
Review experiments/slip_sensitivity/oracle_stress_v1/smoke_report.md before optional
five-seed confirmation. Logs preserve exact solver-slip/mode classes; same classes
produce a warning. Seven offline tests passed; no live stress simulations started.
Read contact_slip/STRESS_TEST_DESIGN.md. Old v1 smoke/results are unchanged; setter
diagnosis and effective-slip validation remain unresolved. Do not launch live runs
autonomously or treat this as approval for training.

LATEST 2026-10-09 smoke review: all three passive oracle runs completed and receipt
hashes passed. Configured .025/.100/.200 ALL yield the same solver slip classes
~.005 and ~.3025 in both axes, enabled in all contacts. Effective variation NOT
validated. See experiments/slip_sensitivity/v1/SMOKE_INTERPRETATION.md. Preserve
prior data but suspend fixed/Gaussian terrain-robustness interpretations. Do NOT
run the optional 30-run sweep or context training yet. Next task: diagnose scene
setter/binding and surface combination, then a new-version passive smoke. Scalar
native setter versus two-argument scene call is a lead, not proven root cause.
No code fix or new live run was made during this review. This supersedes the
earlier recommendation to proceed directly from smoke to evaluate.

Immediate next task (2026-10-09): user chose effective-slip sensitivity checks.
Implemented `./run_slip_sensitivity.sh`: user runs `smoke` (oracle at three slips),
then `evaluate` (oracle+BC, three slips x five seeds, 3,000 active steps), which
automatically reports. No pre-started server; ports must be free. Passive audit
logs ODE slip/enable bits without modifying contacts. Offline native checks and
five Python tests pass; live verification is pending. Outputs go to
experiments/slip_sensitivity/v1. Review actual contact values plus paired progress
before deciding on context-aware training. Do not launch live runs autonomously.

Latest review (2026-10-05): all eight nominal/varied continuation final checkpoints
exist with matching hashes; all 65 Gaussian evaluation receipts passed artifact
hash checks (13 controllers/arms x five draws). Existing Gaussian report.md/json
is frozen-only and omits these new results. Read
`experiments/variable_slip_training/gaussian_v1_seed42/CROSS_PHASE_INTERPRETATION.md`.
No clear varied-versus-nominal progress benefit; BC/mixed DAgger fidelity improves
under both continuation arms. Proposed next decision: effective-slip sensitivity
checks, then a small persistent-slip/context-aware BC pilot and controlled fidelity
analysis rather than immediately scaling per-contact noise. This is a proposal,
not approval to run it. Draft contact_slip tooling is not live validated.
Older statements below that continuation is merely planned are superseded.

- Phase 0, pipeline construction: complete.
- Phase 1, five-seed nominal-domain evaluation: complete and reported.
- Phase 2, parametric domain robustness: all 120 isolated-server fixed runs
  completed and passed artifact/trace validation. Report and interpretation are
  in experiments/phase2_domain_robustness/. Gaussian baseline also completed;
  effective-slip interpretation is now qualified by the passive audit finding.
- Phase 3, joint/actuator impairment: planned.
- Phase 4, nominal/varied continuation: eight final checkpoints and 65 Gaussian
  evaluations completed; see CROSS_PHASE_INTERPRETATION.md in the Gaussian
  experiment. Interpretation of terrain variation awaits physics validation.

The authoritative phase ledger is `EXPERIMENT_PHASES.md`.

## Available final policies

Stable final policies are kept at the repository root:

- `bc_walk_policy.pth`
- `dagger_policy.pth`
- `dagger_mixed_policy.pth`
- `gail_dagger_policy.pth`
- `gail_bc_policy.pth`
- `gail_random_policy.pth`

Intermediate state is organized under `artifacts/<policy>/`. Completed and
superseded experiments are under `experiments/`.

## Phase 1 conclusions

- BC gave the best nominal imitation fidelity and cost/performance trade-off.
- Original DAgger reached oracle-level task performance but had much larger
  action error than BC.
- Mixed-seed DAgger used 100,000 balanced long/short oracle seed samples plus
  seven 10,000-step DAgger rollouts. It reduced offline Action MSE by about 78.5%
  and on-policy oracle MSE by about 95% relative to original DAgger, at the cost
  of 170,000 total oracle labels.
- GAIL-BC retained BC-level fidelity and stable locomotion.
- GAIL-DAgger retained original DAgger's action-error characteristics while
  maintaining task performance.
- Random-start GAIL was numerically stable but stationary: zero targets in all
  training rollouts and normalized evaluation return near the random floor.
- Low jerk and zero falls are not evidence of success without forward progress.
- The small return differences among viable policies are not significant: their
  95% confidence intervals overlap oracle parity and one another.

Current report and sources:

- `evaluation/PHASE1_RESULTS_REPORT.md`
- `evaluation/comparison.json`
- `evaluation/comparison.csv`
- `evaluation/evaluation_summary.png`
- `evaluation/evaluation_summary.pdf`

Presentation source prepared on 2026-09-20:
`evaluation/presentation/EXPERIMENT_RESULTS_SLIDES.md`, with speaker notes,
seven PNG/PDF figures, and conversion instructions in the adjacent `README.md`.
It covers original and follow-up nominal experiments, excluding future work.
Original and expanded evaluation batches remain separate. No new experiment
was run for this retrospective; the immediate experimental task below is unchanged.

The report for only the original BC, DAgger, and GAIL-DAgger experiment is at:

`experiments/archive_20260905_pre_variants/gail_dagger/evaluation/PHASE1_ORIGINAL_3_REPORT.md`

## Important implementation facts

- Policies consume a 63-dimensional Walk observation and produce a
  16-dimensional latent action. These are not direct 22-joint commands.
- Targets are restricted to the player's own half.
- Simulation must use synchronous agent mode and disabled real-time mode.
- The FC Portugal Python environment is normally:
  `/home/pamnt/miniconda3/envs/fcportugal/bin/python3`.
- Fall recovery uses head height plus torso roll/pitch and excludes recovery
  cycles from active data/evaluation steps.
- SimSpark has a monitor/agent socket teardown race. The client drains monitor
  output and serializes socket shutdown, but long optional checkpoint sweeps can
  still stress the server. Primary evaluation is resumable and excludes
  intermediate checkpoints by default.
- GAIL uses FC Portugal demonstrations as expert occupancy. BC or DAgger is only
  the actor initialization, not the GAIL expert.
- GAIL's `0` active-query count is marginal cost. Every GAIL policy still uses
  the static 90,000-sample expert dataset and may inherit initializer cost.

## Live demonstration utility

Run `./run_demo.sh` with SimSpark and RoboViz already running. The numbered menu
selects any final policy or baseline for 1,500 active steps paced at 50 Hz.
Startup and fall recovery add time to the 30-second active run. Details are in
`PIPELINES.md`; this utility does not produce new experimental results.

## Next task: user-run Phase 2A local-scene slip tests

LATEST: Gaussian baseline already completed all 25 required runs (plus five
nominal runs from the earlier all-suite schedule). A resume failure came solely
from our Gaussian selector/report and test edits changing broad source hashes.
Implemented evaluation-only compatibility for audited metadata hashes; real
rollout inputs and artifact hashes remain strict, original receipts unchanged.
Validated all 30 receipts; generated Gaussian-only report.md/json under
experiments/variable_slip_training/gaussian_v1_seed42/. No simulator launched.
Eleven training-package offline tests pass. No need to rerun baseline. Next:
review that report, then user-run quick training smoke/continuation as planned.

LATEST WORKFLOW: user requested Gaussian-only evaluation instead of repeating the
completed fixed tests. variable-training baseline/evaluate/report now default to
--suite gaussian (5 draws; baseline=25 runs, oracle + four frozen policies).
Next commands: ./run_variable_training.sh baseline and
./run_variable_training.sh report --frozen-only. Optional --suite fixed/all remains
available; Gaussian-only reports have null nominal retention. Plan draws and old
fixed results are unchanged. No live evaluations launched by this change.

LATEST RESULTS: all three fixed conditions are complete (40 runs each); loader
validated all 120 receipts/hashes/traces. See FIXED_RESULTS_INTERPRETATION.md in
the phase directory. Oracle and selected policies retain roughly 99–102% nominal
progress with few falls; current range is viable but weakly discriminative.
Keep provisional training bounds [0.025,0.200], mean .100/std .035 for the pilot.
Next user actions: separate quick training smoke, then variable-training baseline
on Gaussian/fixed draws before continuation. No live jobs started by the agent.
All statements below that fixed evaluation is still running are superseded.

NEW IMPLEMENTATION: user requested runnable training for BC, mixed DAgger,
GAIL-BC and GAIL-DAgger only, with unchanged oracle. Implemented isolated package
variable_training/ and run_variable_training.sh; old Python modules/checkpoints
were not modified. Default provisional Gaussian plan created under
experiments/variable_slip_training/gaussian_v1_seed42/plan.json. See
variable_training/README.md for commands. Both varied and nominal arms default on.
Offline CPU update/resume/GAE tests pass; no live training launched. Next: let fixed
sweep finish, run separate quick training plan, review results/teacher viability,
then baseline/all/evaluate/report. Shared simulator lock refuses concurrent runs.
The previous design-only statements below are historical; current scope excludes
original DAgger and GAIL-random. Models are continuations of existing checkpoints;
GAIL-DAgger's original initializer lineage is retained, not renamed mixed DAgger.

Latest user update: fixed-slip experiments are now running with isolated servers.
Requested next training pipeline write-up is in VARIABLE_SLIP_TRAINING.md (design
only, no new executable commands). It specifies hidden-slip continuation of BC,
DAgger and GAIL with paired nominal controls, then a separate context-aware study.
Provisional truncated Gaussian: bounds [0.025, 0.200], parent mean 0.100/std 0.035.
Bounds will be reviewed after fixed results; mean/std are explicit provisional
choices. Do not alter fingerprinted Python files during the running experiment.
Implement BC first after the frozen evaluation/teacher-viability gates. Required
changes include warm-start BC/DAgger training, whole-run splits, immutable data
shards, and per-segment GAE for GAIL. No training has been started.

AUTHORITATIVE UPDATE: user approved a fresh server per policy/seed after two
nominal crashes. Implemented isolated_server_slip_v2 with receipt-validated resume,
failed-attempt preservation and server-death watchdog. Old launches/conditions/
smoke/restoration artifacts moved to archive_shared_server_v1_20261004 under the
phase directory. No data deleted. Next user command: ./run_varied_slip.sh fixed,
then report-fixed. All three conditions need new data. The new lifecycle has
offline coverage but is not live-validated. Older status/protocol text below is
historical. Sampler remains uniform pending the separately planned Gaussian change.

MOST RECENT: nominal retry also crashed with the same boost::lock_error in
AgentControl::AgentThread, after mixed DAgger seed 101. There are now 21 nominal
summary JSONs (not yet audited for completeness). Whole-condition retries have
not resolved the fault. Discuss consistent per-run server isolation before more
live evaluation; it changes the previously requested shared-server protocol and
would require a new evaluation series including high traction. No lifecycle
change has been implemented, and existing results remain preserved.

LATEST FAILURE: high_traction has completed.json; nominal crashed after BC seed
505, while connecting the next agent. Nominal server.log shows SIGABRT with
boost::lock_error in AgentControl::AgentThread after monitor broken-pipe errors.
This matches the historical server teardown failure, not evidence of slip or
policy failure. Preserve high_traction. Nominal partial artifacts remain under
launches/nominal; archive that entire directory before rerunning the nominal
condition under the existing whole-condition protocol. Then run low_traction and
report-fixed. No server fix or automatic per-run restart has been implemented;
changing server lifecycle would require a consistent comparison protocol.

CURRENT STATUS: The user reports that smoke succeeded with the strengthened slip
checks and 3,000-step horizon. Next run `./run_varied_slip.sh fixed` to evaluate
all eight controllers over all three conditions automatically after each repeated
smoke. The user clarified that human control means initiating scripts, not prompts
within them. Fixed and sampled runs now have no interactive approval checkpoints.
Then run `./run_varied_slip.sh report-fixed`.
Full evaluation completion has not yet been reported. The startup-failure and
pending-smoke accounts below are historical and superseded by this status.

Latest user status: fixed evaluations are currently running; completion has not
been reported. Do not change their running pipeline. Agreed next sequence:
analyse fixed results; implement truncated-Gaussian sampling (current sampler is
UNIFORM); evaluate frozen policies on paired draws; train hidden-slip BC first,
then DAgger/GAIL; compare against frozen and budget-matched nominal-only training
on held-out draws; separately compare explicit slip-conditioned policies.
Select bounds/mean/std from results and oracle viability, not in advance.
Keep Unitree R1 EDU after the NAO varied-slip study. See Phase 4's execution-order
update in EXPERIMENT_PHASES.md; phase numbers are historical, not execution order.

Latest user change: smoke now runs 3,000 active steps per condition (seed 901),
matching full-run length. Server startup verifies scene SHA-256 within Ruby,
requires successful import and emits a unique per-launch marker. Python checks
the exact marker, scene slip/hash and config hashes, then records verification
in the manifest. This is source/import verification, not live ODE readback.
Five offline tests pass; live validation remains user-run and pending. Older
1,000-step descriptions below are historical and superseded.

Latest correction (2026-10-04): the retry after moving the marker ALSO failed.
The older diagnosis below that Ruby `run` never returns is retracted. Installed
`rcssserver3d --help` documents `--script-path PATH`; the wrapper now uses that
option instead of passing the configuration as a bare filename. The marker now
appears at the end of local `naosoccersim.rb`, after scene setup. Offline tests
cover the explicit command and timeout cleanup. No live verification of this
latest fix has run. Next: user retries `./run_varied_slip.sh smoke` with ports free.
The earlier four-test/marker-before-run account below records the superseded fix.

The user chose workspace-local scene overrides with a human in the loop.
**The user runs live experiments; do not launch evaluations autonomously.**
No live slip evaluations have completed. The first high-traction smoke attempt on
2026-10-04 started SimSpark but exposed an unreachable startup marker after the
server's non-returning `run` call; the marker now precedes `run` and its ordering
is covered by an offline test. The generated server log showed normal SimSpark
startup plus harmless headless OpenGL/texture warnings; it did not show a scene or
server crash. The failed attempt left
`experiments/phase2_domain_robustness/launches/high_traction/failure.json`, which
the runner now removes automatically after a successful retry. It produced no
`oracle_seed_901.json`, completed block, or experimental result. System nominal
slip is unchanged.

Latest verification: all four tests in `test_varied_slip.py` pass under the
`fcportugal` interpreter, including the new assertion that
`WALKER_SCENE_LOADED` precedes the generated Ruby `run` call. The immediate next
action is for the user—not an agent—to ensure ports 3100/3200 are free and run:

```bash
./run_varied_slip.sh smoke
```

If it fails again, inspect the newly generated
`experiments/phase2_domain_robustness/launches/<condition>/server.log` and
`failure.json` before changing the protocol. Agents must not start the live server
or evaluation autonomously.

Start from `Walker` with `./run_varied_slip.sh smoke`. Then run `fixed --condition`
for `high_traction`, `nominal`, and `low_traction`, or use `fixed` for all three.
Generate `report-fixed`, review it, and choose sampled bounds
with `plan --low ... --high ... --rationale ...`. Finally run `sampled` and
`report-sampled`. See `experiments/phase2_domain_robustness/README.md` for exact
commands, outputs, gate criteria, and cleanup behavior. Do not use the old
system-edit Phase 2 launchers for this workflow.

Fixed slips: 0.025, 0.100, 0.200. Smoke: oracle, seed 901, 1,000 active steps.
Full: six frozen policies plus oracle/random, five paired seeds, 3,000 active steps.
Sampled bounds and variance remain undecided until fixed results are reviewed.
User correction: preserve the original testing baseline and change only slip.
The wrapper now invokes unchanged `run_evaluation.sh` / `evaluate_policy.py`,
keeps original startup/fall accounting and simulator seed(0), and shares one server
across policies in each condition. Extra startup stabilization, forced simulator
seeds and per-policy restarts have been removed. Original script snapshots are in
`experiments/phase2_domain_robustness/baseline_testing/` with hashes. No live tests
have run. Full blocks are protected from overwrite; interrupted blocks require
inspection/archive and a complete rerun. No interactive approval is requested.

Next stages, in order: NAO varied-slip learning and testing; then Unitree R1 EDU
in `unitree_rl_mjlab`: assess direct transfer compatibility, test feasible transfer,
and compare with an RL baseline trained in that target environment. Neither stage
has begun. A 63-D NAO observation / 16-D Walk latent policy is not automatically
compatible with Unitree observations or joint commands.

## Phase 2 analysis requirements

For every condition report both:

1. Within-condition normalized return, using random and FC Portugal baselines
   collected under that same condition.
2. Nominal retention: `J(policy, condition) / J(policy, nominal)`.

This prevents degradation of the condition-specific oracle from hiding absolute
policy degradation. Keep each condition in its own directory under
`experiments/phase2_domain_robustness/conditions/`.

## Documentation maintenance

Whenever work continues:

- Update `EXPERIMENT_PHASES.md` when phase status/scope/results change.
- Update `PIPELINES.md` when code, commands, defaults, contracts, or artifacts
  change.
- Update `CHALLENGES.md` when a defect, failure, mitigation, or research risk is
  discovered or resolved.
- Update this handoff when the immediate next task or major project state changes.
# Viewer addition — 2026-10-15

`./run_viewer.sh` selects historical/completed consolidated models and scalar slip,
owns its server, and waits for the user to connect RoboViz before a30-second demo.
Use `--root experiments/consolidated/smoke_v1` for smoke models. No concurrent
training/evaluation permitted (shared simulator lock). Smoke training/evaluation
previously completed:12models and20deployments. User can run full training then
evaluation with `./run_training.sh run && ./run_evaluation_all.sh run`.
Viewer live GUI verification remains pending; see PIPELINES.md/COMMANDS.md.

# Current next task: consolidated probing — 2026-10-15

Full consolidated evaluation now has comparisonJSON/CSV/MD; user requests probing
new models. Implemented separate `./run_consolidated_probing.sh audit|run|report`.
Real audit loads all28new actors and expert, verifies oracle actions and receipts;
split counts4270/2132/6403rows across10/5/15runs. All30general locomotion targets
retained, no slip target. Context64input models receive correct recorded slip.
75representations,300multitarget ridge fits, expert comparisons/CKA/spectra and
matched random/raw context controls. Research fits not launched by agent.
Next user: `./run_consolidated_probing.sh run`; reports automatically written to
`experiments/representation_probing/consolidated_v1`. Historical studies and
training fingerprints unchanged. See consolidated_probing/README.md. Do not
confuse new mixed-DAgger-initialized GAIL with historical original-DAgger GAIL.

# Latest completed task: staged consolidated results report — 2026-10-15

User requested a lengthy readable numerical report following stages:original
environment, frozen policies on changed slip, varied-domain training, then probes;
also specifically DAgger on varied slip and experimental training/evaluation design.
Delivered `evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md` (~9kwords),
with source hashes, tables, per-replica/context comparisons and complete30-target
probe appendix. `evaluation/analysis/consolidated_summary.py` prints it from JSON.
Confirmed1,020evaluation runs,28final policies,9,000probe scores; verified all300
probe artifact receipts. No retraining/simulation or source-report edits.
Main caveats:170klabels does not equal compute (DAgger54m versusBC17m supervised
exposures); varied continuations have extra training without nominal-only matched
controls; correct-context only; original-scene retention absent for varied models.
Historical meeting brief/deck remain untouched. Next:review the new report with
user; do not autonomously launch further experiments or infer R1claims.

# Immediate next task: run append-only scratch comparison — 2026-10-15

User requested scratch baselines alongside current models, reusing completed
evaluation/probes. Explicit random-GAIL scope:two conditions,original and varied
hidden; no input-random GAIL. Seed convention gives1original+3varied random-GAIL
checkpoints, plus24scratch-lineage varied BC/DAgger/GAIL-BC/GAIL-DM =28newmodels.

Implemented separate `scratch_comparison/` (does not change old hashed training
or probing sources). Existing launchers route only when `--include-scratch`:
`./run_training.sh run --include-scratch && ./run_evaluation_all.sh run --include-scratch && ./run_consolidated_probing.sh run --include-scratch`.
Default old batch unchanged; newoutput`experiments/consolidated/v1/scratch_v1`.
Shared parent datasets reused. Adds929deployments with1,020old reused and224probe
fits with300old reused. Parent receipts/sources/data audit passed; synthetic tests
cover initialization,lineage,skip/tamper,and report/probe append. Live scratch
training/evaluation not run. Optional quick smoke uses existing smoke_v1 parent;
see scratch_comparison/README.md. All previous reports/meeting docs preserved.

