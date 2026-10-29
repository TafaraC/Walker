# Final paper: editable LaTeX package

`main.tex`, `references.bib` and `figures/` form a self-contained paper package.
The canonical prose is [../FINAL_PAPER_DRAFT.md](../FINAL_PAPER_DRAFT.md).
The Markdown citation keys resolve through the BibTeX file. Six vector figures
replace large numerical tables; full numbers remain in the experimental reports.

## Build and edit

From the repository root:

```bash
python3 evaluation/presentation/build_final_figures.py
python3 evaluation/paper/build_latex.py
```

These commands only read research results and write documentation artifacts.
The second command overwrites `main.tex`; edit the Markdown first to keep both
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
and the final eight-body-page limit remain to be checked**. The current
two-column article layout is a draft, not a mandated conference template.
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
