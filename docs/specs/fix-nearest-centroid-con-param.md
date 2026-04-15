# fix-nearest-centroid-con-param

## Goal

Allow `nearest_to_centroid` to accept an existing DuckDB connection so callers with an already-open connection (e.g. notebooks) don't hit a `ConnectionException` from conflicting configurations.

## Current behavior

`nearest_to_centroid` always calls `get_connection(db_path)` internally, which opens a **read-write** connection. If the same database file is already open with a different configuration (e.g. `read_only=True`), DuckDB raises `ConnectionException`.

## Target behavior

`nearest_to_centroid` accepts an optional `con` keyword argument (`duckdb.DuckDBPyConnection | None`, default `None`). When provided:

- The function uses the given connection instead of calling `get_connection`.
- The function does **not** close the connection when it finishes (the caller owns it).

When `con` is `None` (the default), behavior is unchanged: a new connection is opened and closed internally.

## Files to change

| File | Change |
|---|---|
| `src/jt3/embeddings.py` | Add `con` parameter to `nearest_to_centroid`; branch on whether it was supplied. |
| `tests/test_embeddings.py` | Add test that passes an existing connection. |

## Step-by-step instructions

1. **`src/jt3/embeddings.py`** — In the `nearest_to_centroid` signature, add `con: duckdb.DuckDBPyConnection | None = None` after `answers`. Add `import duckdb` to the type-checking imports. Track whether we own the connection (`_own_con = con is None`). If `con is None`, open one via `get_connection`. In the `finally` block, only close if `_own_con`.

2. **`tests/test_embeddings.py`** — Add `test_nearest_to_centroid_with_existing_connection` that opens a connection, calls `nearest_to_centroid(…, con=con)`, and asserts success. Also verify the connection is still usable afterward (not closed).

## Test plan

| Test | Expected |
|---|---|
| `test_nearest_to_centroid_with_existing_connection` | Returns a Polars DataFrame; the passed-in connection remains open and usable afterward. |

## Out of scope

- Adding `con` parameter to other functions (`embed_clues`, etc.).
- Adding `read_only` support to `get_connection`.
