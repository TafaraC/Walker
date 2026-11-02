# Walker paper — wording review revision

The current `main.tex` was revised in place to use precise scientific wording and distinguish measurements, literature statements and untested explanations. It compiled successfully with the desktop editor's built-in compiler on 31 October 2026.

## Package contents

- `main.tex`: current paper, using IEEEtran conference format and author–year citations.
- `references.bib`: supporting bibliography.
- `figures/`: six PDF figures referenced by the paper.
- `REVISION_NOTES.md`: scope of the wording revision and evidence locations.

Upload the extracted contents to Overleaf, select `main.tex` as the main document and use pdfLaTeX with BibTeX, or run `latexmk -pdf main.tex` with IEEEtran and the required packages installed. Citation style intentionally follows the supervisor's author–year requirement.

The ZIP excludes historical previews and archived drafts. The editor's compilation check does not export a standalone PDF. The existing `PRE_REVISION_preview.pdf` in the working folder is historical and is not included in the ZIP.

## Before submission

Finalize author details, repository/data-archive links and outstanding exact resource counts. Check bibliography metadata against original sources and visually review the final layout and page limit. Compilation success does not constitute a fresh independent literature or experimental audit.

Both prior working packages were archived under `paper_before_wording_review_2026-10-31_r1` in OneDrive `archive/` and repository `evaluation/report_archive/`. Earlier claim-review archives also remain preserved.
