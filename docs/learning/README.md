# Setup and Python learning guide

- [PDF workbook](EXPERIMENT_SETUP_WORKBOOK.pdf):21pages, six guided exercises,
  answers, reproducibility checks and official references.
- [Editable Markdown](EXPERIMENT_SETUP_WORKBOOK.md).

Audience: a reader learning Linux/Conda/scientific Python who wants to set up and
understand the Walker experiment pipelines. Distinguishes system dependencies,
Python packages, simulator configuration and historical experiment artifacts.
The local runtime snapshot is not a complete lockfile or a claim that installation
on another platform has been tested. No simulator/full research run was launched
to create this guide; synthetic exercise examples were checked.

Regenerate using the existing Matplotlib dependency:

```bash
python docs/learning/render_workbook.py
```

Optional raster previews (keep them outside research artifact directories):

```bash
python docs/learning/render_workbook.py --preview /tmp/walker_workbook_preview
```

The source deliberately has one logical page per `<!-- PAGE -->` section. The
renderer checks code-line widths, page height and text boundaries, embeds fonts,
and writes a `.layout.json` inspection manifest. Inspect previews after edits.
Do not remove a page break without reviewing the page map and total count.
