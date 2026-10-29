# Repository instructions

Before revising the current meeting summary, archive its complete previous
contents in `evaluation/meeting_archive/` with a unique date/revision filename.
Never overwrite archived summaries. Keep the current brief consolidated and put
superseded meeting wording in the archive, preserving experimental source reports.

At the beginning of a new session, read `SESSION_HANDOFF.md` before making
changes. It records the current project state, key conclusions, and immediate
next task.

When changing any BC, DAgger, or GAIL collector, rollout, trainer, evaluator,
model, shared imitation utility, simulator check, or launcher, update
`PIPELINES.md` in the same change. Keep its technical description, commands,
defaults, data contracts, artifacts, and metrics synchronized with the code.

When a pipeline defect, experimental failure, mitigation, or research risk is
discovered or resolved, update `CHALLENGES.md`. Preserve resolved historical
entries instead of deleting them.

When an experiment phase begins, completes, changes scope, or produces a new
report, update `EXPERIMENT_PHASES.md`. Keep it as the high-level chronological
ledger and leave implementation detail in `PIPELINES.md`.

Update `SESSION_HANDOFF.md` whenever the immediate next task or major project
state changes so a future chat can resume without relying on conversation history.
