# Spec: feat-lookup-module

## Goal

Extract the embedding-lookup functionality from `notebooks/lookup_embeddings.ipynb` into a reusable `jt3.lookup` module and a CLI script.

## Current behavior

The notebook `lookup_embeddings.ipynb` hardcodes a list of query strings, connects to DuckDB read-only, retrieves matching rows from the `embeddings` table by `clue_text`, and prints found/missing counts and embedding shapes. There is no reusable function; everything lives in notebook cells.

## Target behavior

A new module `jt3.lookup` exposes a single function:

```python
def lookup_embeddings(
    queries: list[str],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[dict[str, np.ndarray], list[str]]:
```

- Opens a **read-only** DuckDB connection.
- Queries the `embeddings` table for rows whose `clue_text` is in `queries`.
- Returns a tuple of:
  - `found`: `dict[str, np.ndarray]` mapping each found clue text to its embedding (float32).
  - `missing`: `list[str]` of query strings not found in the table.

A new script `scripts/lookup_embeddings.py` accepts queries as CLI positional args, calls `lookup_embeddings`, and prints the results.

### Example

```
$ uv run python scripts/lookup_embeddings.py "This element has atomic number 1" "nonexistent"
Retrieved 1 embedding(s), shape: (384,)
Not found (1): ['nonexistent']
```

## Files to change

| File | Action | Summary |
|------|--------|---------|
| `src/jt3/lookup.py` | Create | New module with `lookup_embeddings()` |
| `src/jt3/__init__.py` | Modify | Add `lookup_embeddings` import and `__all__` entry |
| `scripts/lookup_embeddings.py` | Create | CLI script |
| `tests/test_lookup.py` | Create | Tests for `lookup_embeddings()` |

## Step-by-step instructions

### 1. Create `src/jt3/lookup.py`

```python
"""Embedding lookup from DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from .db import DEFAULT_DB_PATH


def lookup_embeddings(
    queries: list[str],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Look up embeddings for a list of clue texts.

    Returns (found, missing) where found maps clue_text → embedding
    and missing lists queries not in the table.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT clue_text, embedding FROM embeddings "
            "WHERE clue_text IN (SELECT unnest(?))",
            [queries],
        ).fetchall()
    finally:
        con.close()

    found = {text: np.array(emb, dtype=np.float32) for text, emb in rows}
    missing = [q for q in queries if q not in found]
    return found, missing
```

### 2. Update `src/jt3/__init__.py`

Add import of `lookup_embeddings` from `.lookup` and add to `__all__`.

### 3. Create `scripts/lookup_embeddings.py`

```python
"""CLI script to look up embeddings for given query strings."""

import sys

from jt3.lookup import lookup_embeddings


def main() -> None:
    queries = sys.argv[1:]
    if not queries:
        print("Usage: uv run python scripts/lookup_embeddings.py <query1> [query2 ...]")
        sys.exit(1)

    found, missing = lookup_embeddings(queries)

    if found:
        first = next(iter(found.values()))
        print(f"Retrieved {len(found)} embedding(s), shape: {first.shape}")
    else:
        print("No embeddings found.")

    if missing:
        print(f"Not found ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
```

### 4. Create `tests/test_lookup.py`

Test cases:
1. `test_lookup_found` — populate DB with known embeddings, query them, assert all found.
2. `test_lookup_missing` — query for strings not in DB, assert all in missing list.
3. `test_lookup_partial` — mix of found and missing queries.
4. `test_lookup_empty_queries` — pass empty list, get empty results.

## Out of scope

- Changing the notebook itself.
- Supporting other embedding tables (response_embeddings, etc.).
- Adding search/similarity functionality.
