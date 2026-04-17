# feat-embeddings-module

## Goal

Extract all embedding generation functionality from the `generate_embeddings.ipynb` notebook into a reusable `jt3.embeddings` module, and update `jt3.db` for consistent embedding table support.

## Current Behavior

- Embedding generation lives entirely in `notebooks/generate_embeddings.ipynb` and a partial `scripts/generate_embeddings.py`.
- `db.py` has `save_embeddings`/`get_embedding` hardcoded to a single `embeddings` table with `FLOAT[384]` and column `clue_text`.
- The notebook creates 4 separate tables (`clue_embeddings`, `response_embeddings`, `contextual_response_embeddings`, `prompted_response_embeddings`) with varying schemas and dynamic dimensions.

## Target Behavior

### `db.py` changes

1. **`save_embeddings`** gains keyword-only params `table` (default `"embeddings"`) and `text_column` (default `"clue_text"`). Auto-creates the target table with the correct dimension if it doesn't exist. Keeps upsert (`INSERT OR REPLACE`) semantics.
2. **`get_embedding`** gains matching `table` and `text_column` keyword-only params.
3. **New `save_contextual_embeddings`** — saves to `contextual_response_embeddings` table which has an additional `context_texts JSON` column.
4. **`_validate_identifier`** — internal helper to prevent SQL injection on table/column names.

### New `src/jt3/embeddings.py`

Public API:

| Function | Description |
|---|---|
| `MODELS` | Dict of model configurations (name → kwargs for `SentenceTransformer`). |
| `load_model(model_key)` | Returns a configured `SentenceTransformer` instance. |
| `fetch_clue_texts(db_path)` | Query distinct clue texts from the `clues` table. |
| `fetch_response_texts(db_path)` | Query distinct non-null response texts from the `clues` table. |
| `fetch_response_contexts(db_path)` | Query clue+category+response rows, return `dict[str, list[str]]` mapping response → context strings. |
| `generate_clue_embeddings(model, db_path)` | Fetch clue texts → encode → save to `clue_embeddings` table. Returns count saved. |
| `generate_response_embeddings(model, db_path)` | Fetch response texts → encode → save to `response_embeddings` table. Returns count saved. |
| `generate_contextual_response_embeddings(model, db_path)` | Fetch contexts → encode all context strings → average per response → L2-normalize → save to `contextual_response_embeddings` table. Returns count saved. |
| `generate_prompted_response_embeddings(model, db_path, prompt)` | Fetch response texts → encode with prompt → L2-normalize → save to `prompted_response_embeddings` table. Returns count saved. |

### `__init__.py` changes

Export: `load_model`, `generate_clue_embeddings`, `generate_response_embeddings`, `generate_contextual_response_embeddings`, `generate_prompted_response_embeddings`, `save_contextual_embeddings`.

### `scripts/generate_embeddings.py` changes

Update to use the new `jt3.embeddings` module instead of inline code.

## Files to Change

| File | Action | Summary |
|---|---|---|
| `src/jt3/embeddings.py` | Create | New module with model loading, text fetching, encoding, and orchestration functions. |
| `src/jt3/db.py` | Modify | Add `table`/`text_column` params to `save_embeddings`/`get_embedding`; add `save_contextual_embeddings`; add `_validate_identifier`. |
| `src/jt3/__init__.py` | Modify | Export new public symbols. |
| `scripts/generate_embeddings.py` | Modify | Use new module. |
| `tests/test_embeddings.py` | Create | Tests for the new module (mocked model). |
| `tests/test_db.py` | Modify | Add tests for new `table`/`text_column` params and `save_contextual_embeddings`. |

## Step-by-Step Instructions

### Step 1: Add `_validate_identifier` to `db.py`

Add a helper function that checks identifier names against `^[A-Za-z_][A-Za-z0-9_]*$` and raises `ValueError` on invalid names.

### Step 2: Update `save_embeddings` in `db.py`

- Add `*, table: str = "embeddings", text_column: str = "clue_text"` keyword-only params after `db_path`.
- Validate `table` and `text_column` with `_validate_identifier`.
- Before inserting, auto-create the table: `CREATE TABLE IF NOT EXISTS {table} ({text_column} TEXT PRIMARY KEY, embedding FLOAT[{dim}] NOT NULL)` where `dim = embeddings.shape[1]`.
- Use f-string with validated identifiers for the SQL.

### Step 3: Update `get_embedding` in `db.py`

- Add `*, table: str = "embeddings", text_column: str = "clue_text"` keyword-only params.
- Validate identifiers.
- Use validated identifiers in the query.

### Step 4: Add `save_contextual_embeddings` to `db.py`

```python
def save_contextual_embeddings(
    texts: list[str],
    embeddings: np.ndarray,
    context_texts: list[str],  # pre-serialized JSON strings
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
```
- Create `contextual_response_embeddings` table if not exists.
- Upsert rows with `response_text`, `context_texts`, `embedding`.

### Step 5: Create `src/jt3/embeddings.py`

Implement all functions listed in the Target Behavior section.

- `fetch_*` functions use `get_connection` from `db.py` and raw SQL.
- `generate_*` functions call the corresponding `fetch_*`, encode with the model, then call `save_embeddings`/`save_contextual_embeddings` from `db.py`.
- Contextual embeddings: build `"Category: Clue → Response"` strings, batch encode, average per response, L2-normalize with numpy.
- Prompted embeddings: encode with `prompt=` parameter, L2-normalize.

### Step 6: Update `__init__.py`

Add imports and `__all__` entries for new public symbols.

### Step 7: Update `scripts/generate_embeddings.py`

Replace inline code with calls to `jt3.embeddings` functions.

## Test Plan

### `tests/test_embeddings.py`

| Test | Expected Result |
|---|---|
| `test_load_model` | Returns a SentenceTransformer with correct model name. |
| `test_fetch_clue_texts` | Returns list of distinct clue texts from populated DB. |
| `test_fetch_clue_texts_empty` | Returns empty list from empty DB. |
| `test_fetch_response_texts` | Returns list of distinct non-null responses. |
| `test_fetch_response_texts_excludes_null` | Responses with NULL `correct_response` are excluded. |
| `test_fetch_response_contexts` | Returns dict mapping response → list of context strings. |
| `test_generate_clue_embeddings` | With mocked model, saves embeddings to `clue_embeddings` table and returns correct count. |
| `test_generate_response_embeddings` | With mocked model, saves embeddings to `response_embeddings` table and returns correct count. |
| `test_generate_contextual_response_embeddings` | With mocked model, saves averaged+normalized embeddings to `contextual_response_embeddings` table. |
| `test_generate_prompted_response_embeddings` | With mocked model, saves normalized embeddings to `prompted_response_embeddings` table. |

### `tests/test_db.py` (additions)

| Test | Expected Result |
|---|---|
| `test_save_embeddings_custom_table` | Saves to a custom table name and retrieves via `get_embedding` with same table. |
| `test_save_embeddings_custom_text_column` | Saves with custom text column and retrieves correctly. |
| `test_validate_identifier_rejects_injection` | `_validate_identifier` raises ValueError for `"foo; DROP TABLE"`. |
| `test_save_contextual_embeddings_round_trip` | Saves contextual embeddings and verifies all columns. |

## Out of Scope

- Changing the existing `embeddings` table schema or migrating data.
- Adding CLI entry points.
- Changing notebook cells (the notebook may be updated separately to use the module).
- Model training or fine-tuning.
- Performance optimization of encoding (batch size tuning, etc.).
