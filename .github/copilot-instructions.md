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
src/jt3/              # Package root (importable as `jt3`)
    db.py             # Shared: get_connection, _validate_identifier
    models.py         # Shared dataclasses (Episode, Round, Category, Clue, etc.)
    scraping/         # Scraping pipeline
        scraper.py    # HTML parsing (parse_episode, fetch_episode)
        crawler.py    # Season/game-list crawling
        db.py         # Episode CRUD (save_episode, load_episode, etc.)
    embeddings/       # Embedding-generation pipeline
        generator.py  # Model loading, text fetching, embedding generation
        db.py         # Embedding storage (save_embeddings, get_embedding, etc.)
scripts/              # Pipeline orchestration scripts
    scraping/
        scrape.py     # Season scraping CLI
    embeddings/
        clues.py      # Generate clue embeddings
        responses.py  # Generate response embeddings
        categories.py # Generate category embeddings
        full.py         # Generate Category: Clue → Response embeddings
    show_table.py
tests/                # pytest tests; mirrors src structure
```

## Running Pipeline Scripts

```powershell
# Scrape a season
uv run python scripts/scraping/scrape.py <season> [--delay <seconds>] [--db <path>]

# Generate embeddings (each script accepts --model and --db)
uv run python scripts/embeddings/clues.py
uv run python scripts/embeddings/responses.py
uv run python scripts/embeddings/categories.py
uv run python scripts/embeddings/full.py

# Inspect a table
uv run python scripts/show_table.py
```

## Conventions

- **Python ≥ 3.11** — use modern syntax (match, `X | Y` unions, `tomllib`, etc.)
- **ruff** handles both linting and formatting; no separate black/flake8
- Dev extras live in `[project.optional-dependencies] dev`; add new dev deps there

## Notebook Conventions (Jupyter Only)

- Do not add type hints in Jupyter notebooks.
- Do not add error handling in Jupyter notebooks solely because it is a best practice.
- Add notebook error handling only when it solves a specific problem in that notebook workflow.
