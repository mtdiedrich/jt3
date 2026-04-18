# Refactor: Pipeline-Oriented Package Structure

## Goal

Restructure `jt3` into pipeline-oriented subpackages — `jt3.scraping` and `jt3.embeddings` — with shared utilities at the top level, pipeline scripts in `scripts/`, and tests mirroring the new layout.

## Current Behavior

Flat module layout under `src/jt3/`:

```
src/jt3/
    __init__.py      # Façade re-exporting everything
    models.py        # Dataclasses: Episode, Round, Category, Clue, Contestant
    scraper.py       # Single-episode fetch + parse
    crawler.py       # Batch fetching, season discovery
    db.py            # All DB: connection utils + episode CRUD + embedding CRUD
    embeddings.py    # Embedding model loading + generation
    lookup.py        # BROKEN — queries legacy schema
```

## Target Behavior

Pipeline-oriented subpackage layout:

```
src/jt3/
    __init__.py          # Slim façade
    db.py                # Shared: get_connection, DEFAULT_DB_PATH, _validate_identifier
    models.py            # Shared: dataclasses (unchanged)
    scraping/
        __init__.py      # Re-exports from scraper, crawler, db
        scraper.py       # parse_episode, fetch_episode
        crawler.py       # Batch fetching, season discovery
        db.py            # ensure_schema, save/load/delete/list_episodes
    embeddings/
        __init__.py      # Re-exports from generator, db
        generator.py     # MODELS, load_model, fetch_*_texts, generate_*_embeddings
        db.py            # save_embeddings, get_embedding

scripts/
    scrape.py                # NEW: Scraping pipeline orchestrator
    generate_embeddings.py   # Updated imports
    show_table.py            # Updated imports

tests/
    test_models.py               # Unchanged
    test_scraper.py              # Updated imports → jt3.scraping
    test_crawler.py              # Updated imports → jt3.scraping
    test_db.py                   # Shared DB util tests only
    test_scraping_db.py          # Episode CRUD tests (extracted from test_db.py)
    test_embeddings_db.py        # Embedding storage tests (extracted from test_db.py)
    test_embeddings_generator.py # Renamed from test_embeddings.py, updated imports
    test_import.py               # Updated
```

## Files to Delete

| File | Reason |
|------|--------|
| `src/jt3/lookup.py` | Broken (legacy schema), user chose to remove |
| `scripts/lookup_embeddings.py` | Uses removed `lookup.py` |
| `tests/test_lookup.py` | Tests removed module |

## Step-by-Step Instructions

### 1. Shared `src/jt3/db.py` — strip to utilities only

Keep: `_IDENTIFIER_RE`, `_validate_identifier`, `_PROJECT_ROOT`, `DEFAULT_DB_PATH`, `get_connection`.

Remove: `_SCHEMA_SQL`, `ensure_schema`, `save_episode`, `load_episode`, `delete_episode`, `list_episodes`, `save_embeddings`, `get_embedding` and their imports (`models`, `numpy`, `polars`).

`get_connection` needs to change — currently calls `ensure_schema`, but that moves to `scraping.db`. New behavior: just open connection, create parent dirs, no schema setup.

### 2. `src/jt3/scraping/__init__.py`

Re-export public API from `.scraper`, `.crawler`, `.db`.

### 3. `src/jt3/scraping/scraper.py`

Move from `src/jt3/scraper.py`. Change `from .models import ...` → `from ..models import ...`.

### 4. `src/jt3/scraping/crawler.py`

Move from `src/jt3/crawler.py`. Change:
- `from .models import Episode` → `from ..models import Episode`
- `from .scraper import fetch_episode` → `from .scraper import fetch_episode` (stays same)

### 5. `src/jt3/scraping/db.py`

Extract episode schema + CRUD from old `db.py`. Imports:
- `from ..db import get_connection, DEFAULT_DB_PATH`
- `from ..models import ...`

Contains: `_SCHEMA_SQL`, `ensure_schema`, `save_episode`, `load_episode`, `delete_episode`, `list_episodes`.

`get_connection` call should call `ensure_schema` after connecting — or each function calls `ensure_schema` on its connection. Simplest: wrap with a local helper or call `ensure_schema` inside each public function. Actually, better: provide `get_episode_connection` that calls `get_connection` then `ensure_schema`.

### 6. `src/jt3/embeddings/__init__.py`

Re-export public API from `.generator`, `.db`.

### 7. `src/jt3/embeddings/generator.py`

Move from `src/jt3/embeddings.py`. Change:
- `from .db import DEFAULT_DB_PATH, get_connection, save_embeddings` → `from ..db import DEFAULT_DB_PATH, get_connection` + `from .db import save_embeddings`

### 8. `src/jt3/embeddings/db.py`

Extract embedding CRUD from old `db.py`. Imports:
- `from ..db import get_connection, DEFAULT_DB_PATH, _validate_identifier`

Contains: `save_embeddings`, `get_embedding`.

### 9. Update `src/jt3/__init__.py`

Update all import paths to reference new subpackage locations.

### 10. Update scripts

- `scripts/generate_embeddings.py`: `from jt3.embeddings import ...` (or `from jt3.embeddings.generator import ...`)
- `scripts/show_table.py`: `from jt3.db import DEFAULT_DB_PATH, _validate_identifier` (unchanged — shared db)
- Create `scripts/scrape.py`: New scraping pipeline orchestrator

### 11. Restructure tests

- `tests/test_scraper.py`: Update `from jt3.scraper import ...` → `from jt3.scraping.scraper import ...`
- `tests/test_crawler.py`: Update `from jt3.crawler import ...` → `from jt3.scraping.crawler import ...`
- `tests/test_db.py`: Keep only shared utility tests (get_connection, ensure_schema idempotent, _validate_identifier)
- `tests/test_scraping_db.py`: Episode CRUD tests from old test_db.py
- `tests/test_embeddings_db.py`: Embedding storage tests from old test_db.py
- `tests/test_embeddings_generator.py`: Renamed from test_embeddings.py, updated imports

### 12. Delete old files

- `src/jt3/scraper.py`, `src/jt3/crawler.py`, `src/jt3/embeddings.py`, `src/jt3/lookup.py`
- `scripts/lookup_embeddings.py`
- `tests/test_lookup.py`

## Test Plan

All existing tests (94 passing) should pass with updated imports. The 3 failing lookup tests are removed along with the module. No new behavioral tests needed — this is a pure structural refactor.

## Out of Scope

- Changing any module's behavior or API signatures
- Adding new features to either pipeline
- Fixing the DuckDB data files or notebooks
- Changing notebook imports
