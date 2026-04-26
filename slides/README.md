# Slides — Overfitting & Underfitting in Machine Learning

## File Overview

| File | Description |
|------|-------------|
| `generate_pptx.py` | Python script to generate the PowerPoint presentation |
| `presentation.tex` | Beamer LaTeX source (alternative format) |
| `speaker-notes.md` | Full speaker notes with timing per slide |
| `presentation.pdf` | Compiled PDF from the LaTeX source |
| `presentation_v5.pptx` | Latest PowerPoint output (recommended) |

Earlier `.pptx` versions (v1–v4) are kept for reference.

## Generate the PowerPoint

### Prerequisites

The script uses figures from `report/figures/`. Make sure those exist by running the notebooks first:

```bash
uv run jupyter nbconvert --execute --inplace notebooks/*.ipynb
```

### Dependencies

All dependencies are listed in `pyproject.toml`. Install via:

```bash
uv sync
```

Key packages used by `generate_pptx.py`:

- `python-pptx` — create and manipulate `.pptx` files
- `matplotlib` + `numpy` — generate inline diagrams (eigenvalue shift, constraint contours, L1 vs L2 geometry)
- `lxml` — XML manipulation for click-to-appear animations

### Run

```bash
uv run python slides/generate_pptx.py
```

This reads figures from `report/figures/`, generates matplotlib diagrams on the fly, and writes the `.pptx` to `slides/`.

> **Note:** If the output file is open in PowerPoint, the script will fail with a `PermissionError`. Close PowerPoint first, or change the `OUT` variable in the script to a different filename.

## Compile the PDF (LaTeX)

### Prerequisites

A TeX distribution is required (e.g., **MiKTeX** on Windows, **TeX Live** on Linux/macOS).

Required LaTeX packages (all included in standard distributions):

- `beamer` with `Madrid` theme
- `amsmath`, `amssymb`
- `graphicx`, `booktabs`, `multicol`, `xcolor`

### Compile

```bash
cd slides
pdflatex presentation.tex
pdflatex presentation.tex   # run twice for cross-references
```

Or with latexmk:

```bash
latexmk -pdf slides/presentation.tex
```
