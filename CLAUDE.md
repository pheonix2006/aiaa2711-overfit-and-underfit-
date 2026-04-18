# CLAUDE.md

## Project

AIAA 2711 course project — Overfitting & Underfitting in Machine Learning (2025 Spring, 3-person team).

## Tech Stack

- **Python 3.11+**, managed by **uv** (not pip/conda)
- Core: numpy, matplotlib, scikit-learn
- Deep Learning: torch, torchvision
- Notebooks: Jupyter
- Tests: pytest
- Report: LaTeX (MiKTeX installed locally)

## Architecture

- `src/` — all implementation logic, importable as `from src.xxx import ...`
- `notebooks/` — demonstration only, imports from `src/`, no logic implemented here
- `report/` — LaTeX report, figures auto-generated from notebooks
- `tests/` — 27 unit tests covering all src modules
- `data/` — downloaded datasets (MNIST), gitignored

## Commands

```bash
uv sync                    # Install dependencies
uv run pytest tests/ -v    # Run tests (expect 27 passed)
uv run jupyter nbconvert --execute --inplace notebooks/*.ipynb  # Run notebooks
```

## Conventions

- Use `uv run` to execute anything (not raw `python` or `pip`)
- Follow TDD: write tests first, then implement
- Git commits use conventional commit format (`feat:`, `fix:`, `docs:`, `chore:`)
- Code comments in English (matching codebase)
- Communication in Chinese

## Key Design Decisions

- True function is `sin(2*pi*x)` from Bishop's PRML — classic overfitting demo
- Bias-variance computed via Monte Carlo (200 repeats), with `np.clip` to handle high-degree polynomial numerical overflow
- Regularization uses sklearn's Ridge/Lasso with a consistent wrapper interface returning `{predictions, coefficients, model}`
- Deep learning module uses PyTorch with plain functions (no class-based training loops), matching existing module style
- MNIST is auto-downloaded via torchvision to `data/` (gitignored); notebook 03 handles download automatically
