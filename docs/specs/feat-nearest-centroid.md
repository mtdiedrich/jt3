# feat: nearest_to_centroid

## Goal

Add a function that takes a list of answer strings, computes their embedding centroid, and returns the 10 closest answer embeddings from the DB as a Polars DataFrame.

## Current behavior

N/A — new feature.

## Target behavior

`nearest_to_centroid(["Hydrogen", "Oxygen"], db_path=...)` will:

1. Embed each input string using the configured model.
2. Compute the centroid (element-wise mean) of those embeddings.
3. Query the `embeddings` table for all `field='correct_response'` rows matching the model.
4. Rank by cosine similarity to the centroid (descending).
5. Return the top `n` (default 10) results as a Polars DataFrame with columns: `answer_text` (VARCHAR), `similarity` (FLOAT).

### Edge cases

- Single input string → centroid equals that string's embedding.
- Empty list → raise `ValueError`.
- Fewer than `n` answers in DB → return all available rows.

## Files to change

| File | Action | Summary |
|------|--------|---------|
| `src/jt3/embeddings.py` | Modify | Add `nearest_to_centroid()` function |
| `src/jt3/__init__.py` | Modify | Export `nearest_to_centroid` |
| `tests/test_embeddings.py` | Modify | Add tests for `nearest_to_centroid` |

## Step-by-step instructions

### 1. `src/jt3/embeddings.py`

Add at the end of the file:

```python
def nearest_to_centroid(
    answers: list[str],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    n: int = 10,
) -> "polars.DataFrame":
    if not answers:
        raise ValueError("answers must be a non-empty list of strings")

    vectors = embed_batch(answers, model_name=model_name, device=device)
    centroid = vectors.mean(axis=0).tolist()

    con = get_connection(db_path)
    try:
        result = con.execute(
            "SELECT source_text AS answer_text, "
            "list_cosine_similarity(embedding, $1::FLOAT[]) AS similarity "
            "FROM embeddings "
            "WHERE field = 'correct_response' AND model_name = $2 "
            "ORDER BY similarity DESC "
            "LIMIT $3",
            [centroid, model_name, n],
        ).pl()
    finally:
        con.close()
    return result
```

### 2. `src/jt3/__init__.py`

Add `nearest_to_centroid` to the import line and `__all__`.

### 3. `tests/test_embeddings.py`

Add tests:

- **test_nearest_to_centroid_returns_polars_df**: Populate DB with an episode, embed clues, call `nearest_to_centroid(["Hydrogen"])`. Assert returns a Polars DataFrame with columns `answer_text` and `similarity`.
- **test_nearest_to_centroid_respects_n**: Call with `n=1`, assert only 1 row returned.
- **test_nearest_to_centroid_empty_list_raises**: Call with `[]`, assert raises `ValueError`.

## Test plan

| # | Test | File | Expected |
|---|------|------|----------|
| 1 | `test_nearest_to_centroid_returns_polars_df` | `tests/test_embeddings.py` | Returns Polars DF with correct columns, at least 1 row |
| 2 | `test_nearest_to_centroid_respects_n` | `tests/test_embeddings.py` | Returns exactly 1 row when `n=1` |
| 3 | `test_nearest_to_centroid_empty_list_raises` | `tests/test_embeddings.py` | Raises `ValueError` |

## Out of scope

- Filtering out input answers from results.
- Adding this to the notebook (user can call it interactively).
- Caching centroid computations.
