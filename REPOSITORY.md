# Repository and publication guide

This work extends the [FC Portugal Codebase](https://github.com/m-abr/FCPCodebase).
Its upstream history, copyright notices and [GNU GPLv3 licence](LICENSE) are
retained. The original README provides the upstream academic citation.

## Reading the research

- [Paper draft](evaluation/FINAL_PAPER_DRAFT.md) and [LaTeX sources](evaluation/paper/README.md).
- [Presentation PDF](evaluation/presentation/CONSOLIDATED_PAPER_SLIDES.pdf) and
  [explanations](evaluation/presentation/CONSOLIDATED_PAPER_NOTES.md).
- [Control findings](evaluation/SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md) and
  [consolidated report](evaluation/CONSOLIDATED_RESULTS_REPORT_2026-10-15.md).
- [Setup workbook](docs/learning/README.md), [commands](COMMANDS.md),
  [technical protocols](PIPELINES.md) and [experiment phases](EXPERIMENT_PHASES.md).
- Earlier documents: `evaluation/report_archive/` and `evaluation/meeting_archive/`.
  Superseded scripts: `archive/legacy_imitation/`.

## What belongs in Git

Source, tests, reports, tables, figures and documentation remain eligible for
tracking. `.gitignore` excludes experiment directories, raw traces, datasets,
learned checkpoints, logs, native binaries, environments, machine configuration
and credentials. Upstream `.pkl` skill models remain tracked because the oracle
requires them. Tests and `.github/` configuration are eligible for tracking.
Ignoring files leaves them on disk and does not remove already tracked files.

A source clone lacks the datasets/checkpoints required to resume full experiments
or regenerate result figures. Those commands require the documented parent
artifacts. Add their download location and checksums when the data are prepared
for sharing. The workbook describes a small consolidated smoke experiment for
starting a fresh run. Historical absolute paths are provenance, not download links.

## Remote destination

This checkout retains upstream Git history. The configured remotes are:

- `origin`: `https://github.com/TafaraC/Walker`
- `upstream`: `https://github.com/m-abr/FCPCodebase`

See [ATTRIBUTION.md](ATTRIBUTION.md) for the upstream citation and modification
notice. Remote configuration is local Git metadata and is not included in a
source commit. Nothing has been pushed during this preparation.

Review `git status --short` and `git diff` before staging a publication snapshot.
Distribute large research artifacts separately through a release or research
archive. Preserve upstream licensing and clearly identify the research additions.
