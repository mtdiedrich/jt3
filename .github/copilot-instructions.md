# jt3

Python package (src-layout). All commands use `uv run`—never invoke `python` or `pytest` directly.

## Build and Test

```powershell
# Setup (first time)
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .[dev]

# Test
uv run pytest -v
uv run pytest --cov=jt3 --cov-report=html

# Lint / Format
uv run ruff check .
uv run ruff format .
```

## Architecture

```
src/jt3/        # Package source (importable as `jt3`)
tests/          # pytest tests; mirror src structure as the package grows
```

## Conventions

- **Python ≥ 3.11** — use modern syntax (match, `X | Y` unions, `tomllib`, etc.)
- **ruff** handles both linting and formatting; no separate black/flake8
- Dev extras live in `[project.optional-dependencies] dev`; add new dev deps there

## Notebook Conventions (Jupyter Only)

- Do not add type hints in Jupyter notebooks.
- Do not add error handling in Jupyter notebooks solely because it is a best practice.
- Add notebook error handling only when it solves a specific problem in that notebook workflow.
