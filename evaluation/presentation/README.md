# Experimental-results presentation source

## Current consolidated-paper deck (Marp, 28 October 2026)

[CONSOLIDATED_PAPER_MARP.md](CONSOLIDATED_PAPER_MARP.md) contains 23 slides (0–22)
following the standardized restart: original environment, frozen policies on
sampled surfaces, varied-domain continuation/context, scratch initialization,
and general/expert probing. It frames the completed evidence as an eight-body-page
paper, references excluded. Later optional stress/RL and GAIL-improvement work
is explicitly outside its results. The final slide identifies that work as
curiosity-driven follow-up, not merely background omitted for presentation brevity.
Slides16–19 now report completed submission controls: matched-updateBC reverses
the original MSE ranking;200epoch varied-onlyBC narrows the pretraining gap;
zero nativeactions do not achieve useful progress; retained-round fidelity curves
qualify GAIL and resource claims. Conclusions and notes were revised accordingly.
See [feedback response](../SUBMISSION_FEEDBACK_RESPONSE_2026-10-28.md).
Previous deck and notes are preserved with suffix
`2026-10-28_before_submission_controls_r1.md` in the meeting archive.
The subsequent visual revision archives paper/deck/notes with suffix
`2026-10-28_before_visual_revision_r1.md`.

**Ready-to-view:** [PDF slide deck](CONSOLIDATED_PAPER_SLIDES.pdf).
Eleven reusable figures are in `figures_final/` as PNG and vector PDF, with
source hashes in `manifest.json`. Ten chart slides replace selected dense tables.
The paper uses six figures; full numerical evidence remains in the reports.

```bash
python3 evaluation/presentation/build_final_figures.py
python3 evaluation/presentation/render_final_deck.py
python3 evaluation/paper/build_latex.py
```

The PDF renderer is a portable fallback using the same Marp content, not the
Marp rendering engine. It checks text bounds; representative output pages were
visually inspected. Add `--preview /tmp/walker_final_deck` to save slide images.
The [LaTeX paper package](../paper/README.md) includes a buildable source bundle;
no TeX compiler was installed here, so paper compilation/page count is pending.
The original 25-slide source and companion are archived under
`evaluation/meeting_archive/` with suffix `2026-10-26_before_focused_outline_r1.md`.
Probing slides13–15 now specifically interpret DAgger fidelity, limited GAIL
change and continuation/scratch results, with causal limits made explicit.
The preceding19-slide version is archived with suffix
`2026-10-26_before_results_focused_probing_r2.md`.

[CONSOLIDATED_PAPER_NOTES.md](CONSOLIDATED_PAPER_NOTES.md) provides matching
slide-by-slide background, suggested interpretations, metric definitions,
terminology, limitations and primary references. Numerical sources are the
consolidated results report, including Appendix F, and the final-paper draft.

User-run conversion from the repository root:

```bash
marp evaluation/presentation/CONSOLIDATED_PAPER_MARP.md --pdf \
  -o evaluation/presentation/CONSOLIDATED_PAPER.pdf
```

Source structure and local links were checked. The fallbackPDF is rendered;
Marp-specific layout still depends on the user's renderer/theme. No experimental
artifacts or historical meeting summaries were changed to create this presentation.

## Historical full-study meeting deck (Marp, 12 October 2026)

For plain-language explanations of all 24 slides, including terminology, metric
interpretation and short suggested spoken takeaways, read
[SLIDE_EXPLANATIONS.md](SLIDE_EXPLANATIONS.md). Numbering includes the title slide.

`MEETING_SUMMARY_MARP.md` was revised on 2026-10-12 into 24 slides: a title,
two brief recap slides, then detailed context/FiLM, repeatability, general and
expert probing findings, conclusions and references. It includes multiple
represented feature families, quantitative readout comparisons, controls and
interpretation limits. It uses Marp YAML metadata, slide separators and
HTML-comment speaker notes. Conversion is user-run as requested; source structure
and local links were checked, but rendered layout has not been validated.
The historical Pandoc deck below is preserved unchanged.

Open it with the **Marp for VS Code** extension to preview and export, or use
an already installed Marp CLI from the repository root:

```bash
marp evaluation/presentation/MEETING_SUMMARY_MARP.md --html \
  -o evaluation/presentation/MEETING_SUMMARY.html
marp evaluation/presentation/MEETING_SUMMARY_MARP.md --pdf \
  -o evaluation/presentation/MEETING_SUMMARY.pdf
marp evaluation/presentation/MEETING_SUMMARY_MARP.md --pptx \
  -o evaluation/presentation/MEETING_SUMMARY.pptx
```

Marp CLI is not installed in the current shell; rendered exports have not been
generated or visually validated here. PDF/PPTX export also requires a compatible
browser. The source uses no external images or custom theme dependency.
PDF is suitable for sharing; PPTX slide content is image-based by default, not
ordinary editable PowerPoint text. Notes are in HTML comments in the source.

Before changing a meeting brief or its accompanying deck, preserve its complete
previous version under `evaluation/meeting_archive/` with a unique filename.
The prior deck is archived as
`MEETING_SUMMARY_MARP_before_new_work_focus_2026-10-12.md` in that directory.

## Historical nominal-results deck (Pandoc)

`EXPERIMENT_RESULTS_SLIDES.md` contains 20 slide sections plus title metadata,
including the source appendix, with Pandoc-style speaker notes. It covers the original nominal
comparison, earlier GAIL diagnostic failures and stabilization, mixed-seed
DAgger, and GAIL initialization variants. It omits future project directions.

Figures are embedded through relative paths, with PNG versions for slides and
PDF versions for vector export. Keep the Markdown and `figures/` directory
together when moving the presentation.

To convert with Pandoc, run from this directory:

```bash
pandoc EXPERIMENT_RESULTS_SLIDES.md --from=markdown --to=pptx --slide-level=1 -o EXPERIMENT_RESULTS.pptx
```

A custom PowerPoint template can be supplied with `--reference-doc=template.pptx`.
The Markdown uses one level-one heading per slide. Speaker notes contain the
source attribution, full confidence intervals, and interpretation limits.
Conversion has not been run as part of preparing this Markdown source.

To regenerate the seven figures from repository artifacts:

```bash
/home/pamnt/miniconda3/envs/fcportugal/bin/python3 evaluation/presentation/generate_figures.py
```

Run the regeneration command from the repository root. The script requires
Matplotlib and NumPy. No experiment data or policy checkpoints are modified.
