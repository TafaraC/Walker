# Final paper: editable LaTeX package

Revision 29 October 2026 incorporates the second feedback round: matched-budget
and convergence qualifications in abstract/conclusion, an original-scene RQ3
table, a matched-model account of surface errors, expanded RoboCup/fidelity
background, and a cautionary interpretation of the prior probes. Figure5's
nearby points are64,500 DAgger updates versus66,500 BC updates, not an exactly
matched pair. The body now includes a proper math-mode normalization equation.
The user will finalize author details and publication URLs separately. Optional
early-stopping diagnostics have not been launched as part of this revision.

`main.tex`, `references.bib` and `figures/` form a self-contained paper package.
The canonical prose is [../FINAL_PAPER_DRAFT.md](../FINAL_PAPER_DRAFT.md).
The Markdown citation keys resolve through the BibTeX file. Six vector figures
replace large numerical tables; full numbers remain in the experimental reports.

## IEEE layout and supervisor-requested citations

The document now uses `\documentclass[conference]{IEEEtran}`, matching the
[IEEE conference template supplied by the supervisor](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn).
IEEEtran supplies the margins, columns, fonts, headings and author-block layout;
the previous article-class geometry and font overrides have been removed.
Overleaf supplies IEEEtran. A local TeX installation needs that class installed.

The supervisor explicitly requests author names in text instead of numbers.
This is a deliberate citation-style exception to IEEE's usual numbered format:

```latex
\usepackage[authoryear,round]{natbib}
\bibliographystyle{plainnat}
```

Use `\citep{ross2011}` for a parenthetical author–year citation, such as
“(Ross et al., 2011)”, or `\citet{ho2016}` for “Ho and Ermon (2016)” in a
sentence. The current Markdown citations generate `\citep`. Exact first-citation
author-list length follows natbib/plainnat. Do not retain the template's
`\usepackage{cite}` or substitute the numeric `IEEEtran` bibliography style:
those would conflict with this requested convention. References remain in
`references.bib`, and BibTeX formats the alphabetically ordered bibliography.

## Build and edit

From the repository root:

```bash
python3 evaluation/presentation/build_final_figures.py
python3 evaluation/paper/build_latex.py
```

These commands only read research results and write documentation artifacts.
The second command overwrites `main.tex` and rebuilds `paper_latex.zip`; edit the Markdown first to keep both
versions synchronized. To edit LaTeX independently, stop regenerating it.

Upload `main.tex`, `references.bib` and `figures/` to Overleaf, or run locally:

```bash
cd evaluation/paper
latexmk -pdf main.tex
```

Without latexmk: run pdflatex, bibtex, then pdflatex twice, all on `main`.
The portable zip `paper_latex.zip` contains only those required source files.

No LaTeX compiler was available during preparation. Citation keys, included
files and source structure were checked, but **compilation, float placement
and the final eight-body-page limit remain to be checked**. The layout now uses
the requested IEEE conference class, with the requested author–year exception.
References begin after a clear page and are excluded from that target limit.
Replace the author/institution placeholder and add the publication data URL
before submission. Do not claim the private workspace is already publicly released.

## Evidence and limits

- [Control response and full tables](../SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md)
- [Historical consolidated/scratch report](../CONSOLIDATED_RESULTS_REPORT_2026-10-15.md)
- [Figure provenance](../presentation/figures_final/manifest.json)

The revised argument distinguishes initial allocations from matched-update and
matched-exposure controls. Convergence is unproven; new BC controls were not
probed. Larger-budget GAIL, stress/RL and R1 transfer remain outside the paper.
Historical prose is archived rather than overwritten without a record.
