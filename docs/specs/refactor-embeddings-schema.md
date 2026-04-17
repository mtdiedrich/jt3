# Refactor: Embeddings Schema

## Goal

Move all embedding tables into a dedicated `embeddings` DuckDB schema with a uniform `(text TEXT PK, embedding FLOAT[])` structure, replacing the current flat tables.

## Current behavior

- Four flat tables in `main` schema: `clue_embeddings`, `response_embeddings`, `contextual_response_embeddings`, `prompted_response_embeddings`
- Each has different column names (`clue_text`, `response_text`) and structures
- `contextual_response_embeddings` averages multiple context strings per response and stores JSON context
- `prompted_response_embeddings` encodes responses with a prompt prefix
- `save_embeddings()` accepts custom `table` and `text_column` params; `save_contextual_embeddings()` is a separate function
- A legacy `embeddings` table definition exists in `_SCHEMA_SQL`

## Target behavior

- A `CREATE SCHEMA IF NOT EXISTS embeddings` schema in DuckDB
- Four tables, all with the same structure:

| Table | `text` column contains | Source |
|---|---|---|
| `embeddings.clues` | Clue text | `clues.text` |
| `embeddings.responses` | Response text | `clues.correct_response` |
| `embeddings.categories` | Category name | `categories.name` |
| `embeddings.full_context` | `"Category: Clue → Response"` string | Composed from join |

- Each table: `text TEXT PRIMARY KEY, embedding FLOAT[<dim>] NOT NULL`
- `full_context` stores **one row per context string** (not averaged per response)
- `prompted_response_embeddings` is dropped entirely
- `lookup.py` is NOT changed in this PR

## Files to change

| File | Change |
|---|---|
| `src/jt3/db.py` | Remove old `embeddings` table from `_SCHEMA_SQL`. Add `ensure_embeddings_schema(con, dim)`. Replace `save_embeddings()` to write to `embeddings.<table>`. Remove `save_contextual_embeddings()`. Update `get_embedding()`. |
| `src/jt3/embeddings.py` | Add `fetch_category_texts()`, `fetch_full_context_texts()`. Replace `generate_contextual_response_embeddings` → `generate_full_context_embeddings`. Replace `generate_prompted_response_embeddings` → `generate_category_embeddings`. Remove contextual/prompted helpers. All generators write to `embeddings.<table>`. |
| `tests/test_db.py` | Update embedding tests for new schema-qualified tables and simplified API. |
| `tests/test_embeddings.py` | Update generation tests: remove contextual/prompted, add category/full_context. |
| `scripts/generate_embeddings.py` | Call updated function names. |

## Step-by-step instructions

### db.py

1. Remove the `CREATE TABLE IF NOT EXISTS embeddings (...)` block from `_SCHEMA_SQL`.
2. Add a new function `ensure_embeddings_schema(con, dim)` that:
   - `CREATE SCHEMA IF NOT EXISTS embeddings`
   - For each of the 4 tables: `CREATE TABLE IF NOT EXISTS embeddings.<name> (text TEXT PRIMARY KEY, embedding FLOAT[<dim>] NOT NULL)`
3. Rewrite `save_embeddings(texts, embeddings, *, db_path, table)`:
   - `table` is now just the short name (`clues`, `responses`, `categories`, `full_context`)
   - Validates `table` with `_validate_identifier`
   - Calls `ensure_embeddings_schema(con, dim)`
   - Drops and recreates `embeddings.<table>`
   - Inserts via polars DataFrame
   - No more `text_column` parameter — always `text`
4. Remove `save_contextual_embeddings()` entirely.
5. Update `get_embedding(text, *, db_path, table)` — same simplification, always reads `text` column from `embeddings.<table>`.

### embeddings.py

1. Add `fetch_category_texts(*, db_path)` — `SELECT DISTINCT name FROM categories ORDER BY name`.
2. Add `fetch_full_context_texts(*, db_path)` — runs the join query producing `"Category: Clue → Response"` strings, returns `list[str]`.
3. Rewrite `generate_clue_embeddings` to call `save_embeddings(..., table="clues")`.
4. Rewrite `generate_response_embeddings` to call `save_embeddings(..., table="responses")`.
5. Add `generate_category_embeddings` — fetches category texts, encodes, saves to `table="categories"`.
6. Replace `generate_contextual_response_embeddings` → `generate_full_context_embeddings` — fetches full context texts, encodes (one embedding per string, no averaging), saves to `table="full_context"`.
7. Remove `generate_prompted_response_embeddings` and `fetch_response_contexts`.
8. Remove import of `save_contextual_embeddings`.

### tests/test_embeddings.py

1. Remove `test_generate_contextual_response_embeddings` and `test_generate_prompted_response_embeddings`.
2. Add `test_fetch_category_texts` and `test_fetch_full_context_texts`.
3. Add `test_generate_category_embeddings` and `test_generate_full_context_embeddings`.
4. Update existing generation tests to verify data lands in `embeddings.<table>`.

### tests/test_db.py

1. Update `test_save_and_get_embedding_round_trip` to use new API (no `text_column`).
2. Update `test_save_embeddings_custom_table` — use table names like `"clues"`, `"responses"`.
3. Remove `test_save_embeddings_custom_text_column` (no longer relevant).
4. Remove `test_save_contextual_embeddings_round_trip`.
5. Add test that `embeddings` schema is created.

### scripts/generate_embeddings.py

1. Replace function calls with updated names.

## Test plan

| Test | Expected result |
|---|---|
| `test_save_and_get_embedding_round_trip` | Saves to `embeddings.clues`, retrieves correctly |
| `test_save_embeddings_to_responses` | Saves to `embeddings.responses`, retrieves correctly |
| `test_get_embedding_not_found` | Returns None |
| `test_save_embeddings_upserts` | Second save replaces first |
| `test_embeddings_schema_created` | `embeddings` schema exists after save |
| `test_fetch_category_texts` | Returns distinct category names |
| `test_fetch_full_context_texts` | Returns "Category: Clue → Response" strings |
| `test_generate_clue_embeddings` | Writes to `embeddings.clues` |
| `test_generate_response_embeddings` | Writes to `embeddings.responses` |
| `test_generate_category_embeddings` | Writes to `embeddings.categories` |
| `test_generate_full_context_embeddings` | Writes to `embeddings.full_context` |

## Out of scope

- `lookup.py` changes (will be updated in a follow-up)
- Notebooks that reference old table names
- Migration of existing data in `data/jt3.duckdb`
