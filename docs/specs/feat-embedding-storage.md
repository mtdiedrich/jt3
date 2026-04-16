# feat-embedding-storage

## Goal

Store and retrieve embeddings in DuckDB keyed by clue text, so a user can look up the embedding vector for any previously-embedded string.

## Current behavior

Embeddings are generated in a notebook and held in a numpy array. There is no persistence or lookup mechanism.

## Target behavior

- A new `embeddings` table in DuckDB: `clue_text TEXT PRIMARY KEY, embedding FLOAT[384]`.
- `save_embeddings(texts, embeddings, db_path)` — bulk-inserts text→embedding pairs, upserting on conflict.
- `get_embedding(text, db_path) → list[float] | None` — returns the embedding for a single string, or `None` if not found.
- The table is created by `ensure_schema` alongside existing tables.
- A new notebook cell calls `save_embeddings` after generating embeddings.

## Files to change

| File | Action | Summary |
|------|--------|---------|
| `src/jt3/db.py` | Modify | Add `embeddings` table to schema; add `save_embeddings` and `get_embedding` functions |
| `src/jt3/__init__.py` | Modify | Export `save_embeddings` and `get_embedding` |
| `tests/test_db.py` | Modify | Add tests for new functions |
| `notebooks/create_embeddings.ipynb` | Modify | Add cell to save embeddings to DB |

## Step-by-step instructions

1. **`src/jt3/db.py` — schema**: Append to `_SCHEMA_SQL`:
   ```sql
   CREATE TABLE IF NOT EXISTS embeddings (
       clue_text TEXT PRIMARY KEY,
       embedding FLOAT[384] NOT NULL
   );
   ```

2. **`src/jt3/db.py` — `save_embeddings`**: New function accepting `texts: list[str]`, `embeddings: numpy.ndarray`, `db_path`. Uses `INSERT OR REPLACE INTO embeddings VALUES (?, ?)` in batches via `executemany`.

3. **`src/jt3/db.py` — `get_embedding`**: New function accepting `text: str`, `db_path`. Queries by primary key, returns `list[float] | None`.

4. **`src/jt3/__init__.py`**: Add `save_embeddings` and `get_embedding` to imports and `__all__`.

5. **`tests/test_db.py`**: Add tests:
   - `test_save_and_get_embedding_round_trip` — save one, retrieve it, compare.
   - `test_get_embedding_not_found` — returns `None`.
   - `test_save_embeddings_upserts` — saving again for the same text overwrites.

6. **Notebook**: Add a cell after embedding generation that calls `save_embeddings`.

## Test plan

| Test | Expected |
|------|----------|
| `test_save_and_get_embedding_round_trip` | Stored array matches retrieved list (within float tolerance) |
| `test_get_embedding_not_found` | Returns `None` for unknown text |
| `test_save_embeddings_upserts` | Second save overwrites; `get_embedding` returns latest values |

## Out of scope

- Batch retrieval / `get_embeddings` for multiple texts.
- Similarity search / nearest-neighbor queries.
- Changes to the embedding generation pipeline itself.
