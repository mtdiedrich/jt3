# Spec: Embedding Generation for Clues

## Goal

Add an `embeddings` module that generates vector embeddings for clue text and answers using a HuggingFace sentence-transformer model, persists them in DuckDB tied to source clues, and provides a batch function to embed all clues in the database.

## Current Behavior

N/A — new feature. The package has no embedding or vector functionality. An unrelated `data/embeddings.npz` file exists but is not part of this feature.

## Target Behavior

After the change, a caller can:

```python
from jt3.embeddings import embed, embed_batch, embed_clues

# Embed a single string → numpy array of shape (384,)
vec = embed("This element has atomic number 1")

# Embed multiple strings at once → numpy array of shape (n, 384)
vecs = embed_batch(["text one", "text two"])

# Use a different model
vec = embed("text", model_name="sentence-transformers/all-mpnet-base-v2")

# Embed all clue texts and correct_responses in the DB
count = embed_clues(db_path="data/jt3.duckdb")
# Returns number of embedding rows written
```

Embeddings are stored in a new `embeddings` table with a composite foreign key to `clues` plus a `field` discriminator (`'text'` or `'correct_response'`).

## Database Schema

### `embeddings` table

| Column         | Type    | Constraint                                          |
|---------------|---------|-----------------------------------------------------|
| game_id        | INTEGER | NOT NULL                                            |
| round_index    | INTEGER | NOT NULL                                            |
| category_index | INTEGER | NOT NULL                                            |
| clue_id        | VARCHAR | NOT NULL                                            |
| field          | VARCHAR | NOT NULL — `'text'` or `'correct_response'`         |
| source_text    | VARCHAR | NOT NULL — the string that was embedded             |
| embedding      | FLOAT[] | NOT NULL — the vector                               |
| model_name     | VARCHAR | NOT NULL — model identifier used                    |
| PK             |         | (game_id, round_index, category_index, clue_id, field, model_name) |

The table is added to the existing `_SCHEMA_SQL` in `db.py` via `ensure_schema`.

## API

All functions live in `src/jt3/embeddings.py`.

### Constants

- `DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`

### `embed(text: str, *, model_name: str = DEFAULT_MODEL) -> numpy.ndarray`

Embed a single string. Returns a 1-D numpy array (e.g. shape `(384,)` for MiniLM).

### `embed_batch(texts: list[str], *, model_name: str = DEFAULT_MODEL) -> numpy.ndarray`

Embed multiple strings in one call. Returns a 2-D numpy array of shape `(len(texts), dim)`.

### `embed_clues(db_path: str | Path = DEFAULT_DB_PATH, *, model_name: str = DEFAULT_MODEL) -> int`

1. Opens a DuckDB connection via `get_connection(db_path)`.
2. Queries all rows from the `clues` table.
3. Collects all `text` values and all non-null `correct_response` values.
4. Calls `embed_batch` on each set.
5. Inserts (or replaces) rows into the `embeddings` table.
6. Returns the total number of embedding rows written.

Runs inside a transaction. Replaces any existing embeddings for the same `(clue PK, field, model_name)`.

## Files to Change

| File                      | Action | Summary                                                    |
|--------------------------|--------|------------------------------------------------------------|
| `pyproject.toml`          | Modify | Add `sentence-transformers>=2.0` and `numpy>=1.24` to deps |
| `src/jt3/db.py`           | Modify | Add `embeddings` table to `_SCHEMA_SQL`                    |
| `src/jt3/embeddings.py`   | Create | `embed`, `embed_batch`, `embed_clues` functions            |
| `src/jt3/__init__.py`     | Modify | Export `embed`, `embed_batch`, `embed_clues`               |
| `tests/test_embeddings.py`| Create | Tests for all three functions                              |

## Step-by-Step Instructions

1. **`pyproject.toml`** — Add `"sentence-transformers>=2.0"` and `"numpy>=1.24"` to `[project.dependencies]`.

2. **`src/jt3/db.py`** — Append to `_SCHEMA_SQL`:
   ```sql
   CREATE TABLE IF NOT EXISTS embeddings (
       game_id        INTEGER NOT NULL,
       round_index    INTEGER NOT NULL,
       category_index INTEGER NOT NULL,
       clue_id        VARCHAR NOT NULL,
       field          VARCHAR NOT NULL,
       source_text    VARCHAR NOT NULL,
       embedding      FLOAT[] NOT NULL,
       model_name     VARCHAR NOT NULL,
       PRIMARY KEY (game_id, round_index, category_index, clue_id, field, model_name)
   );
   ```

3. **`src/jt3/embeddings.py`** — Create with:
   - Import `SentenceTransformer` from `sentence_transformers`.
   - `_model_cache: dict[str, SentenceTransformer]` — module-level cache to avoid reloading models.
   - `_get_model(model_name)` — returns cached model or loads and caches it.
   - `embed(text, *, model_name)` — calls `_get_model`, encodes single string, returns 1-D ndarray.
   - `embed_batch(texts, *, model_name)` — calls `_get_model`, encodes list, returns 2-D ndarray.
   - `embed_clues(db_path, *, model_name)` — queries clues, embeds, saves to DB, returns count.

4. **`src/jt3/__init__.py`** — Add imports + `__all__` entries for `embed`, `embed_batch`, `embed_clues`.

5. **`tests/test_embeddings.py`** — Tests using `unittest.mock.patch` to mock `SentenceTransformer`.

## Test Plan

### `tests/test_embeddings.py`

All tests mock `SentenceTransformer` to avoid downloading the real model.

| # | Test                                | Expected                                              |
|---|-------------------------------------|-------------------------------------------------------|
| 1 | `embed()` returns 1-D ndarray       | Shape `(384,)`, correct dtype                         |
| 2 | `embed_batch()` returns 2-D ndarray | Shape `(n, 384)`, one row per input                   |
| 3 | `embed_clues()` on populated DB     | Returns correct count, rows in embeddings table       |
| 4 | `embed_clues()` skips null responses | No embedding row where `correct_response IS NULL`     |
| 5 | `embed_clues()` on empty DB         | Returns 0                                             |
| 6 | `embed_clues()` is idempotent       | Running twice produces same count, no duplicates      |
| 7 | `embed()` with custom model_name    | Model loaded with the custom name                     |

## Out of Scope

- Similarity search / nearest-neighbor queries
- Vector indexing (HNSW, etc.)
- CLI commands
- GPU/device configuration
- Async support
- Incremental embedding (only un-embedded clues) — always re-embeds all
