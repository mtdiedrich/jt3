# feat-response-similarity-notebook

## Goal

Create a notebook that lets the user input a response text, looks up its embedding from the `response_embeddings` table, and returns the N most similar responses by cosine similarity along with their distances.

## Current behavior

N/A — new notebook. The existing `lookup_embeddings.ipynb` only does exact text retrieval from the old `embeddings` table and does not perform similarity search.

## Target behavior

A new notebook `notebooks/lookup_response_embeddings.ipynb` that:

1. Connects to the project DuckDB database (read-only).
2. Accepts two input parameters:
   - `RESPONSE` — the response text to look up (must exist in `response_embeddings`).
   - `N` — number of similar responses to return (default 10).
3. Queries `response_embeddings` for the target embedding, then computes `array_cosine_similarity` against all other rows.
4. Returns the top N most similar responses sorted by cosine similarity descending, displaying `response_text`, `similarity`, and `distance` (1 − similarity).
5. Displays results as a Polars DataFrame.

### Example output

| response_text          | similarity | distance |
|------------------------|-----------|----------|
| Abraham Lincoln        | 0.9234    | 0.0766   |
| George Washington      | 0.8871    | 0.1129   |
| ...                    | ...       | ...      |

## Files to change

| File | Action | Summary |
|------|--------|---------|
| `notebooks/lookup_response_embeddings.ipynb` | Create | New notebook with 4 cells |
| `docs/specs/feat-response-similarity-notebook.md` | Create | This spec |

## Step-by-step instructions

1. **Cell 1 — Connection**: Import `duckdb` and `polars`. Connect to `f:/Project/games/jt3/data/jt3.duckdb` in read-only mode.

2. **Cell 2 — Parameters**: Define `RESPONSE` (str) and `N` (int, default 10).

3. **Cell 3 — Similarity query**: Execute a SQL query that:
   - Uses a CTE to fetch the target embedding by `response_text`.
   - Joins against all other rows computing `array_cosine_similarity`.
   - Orders by similarity descending, limits to N.
   - Converts result to a Polars DataFrame and displays it.

4. **Cell 4 — Cleanup**: Close the connection.

## Test plan

No unit tests — this is a standalone notebook. Manual verification: run the notebook with a known response text and confirm results are reasonable (similarity values between 0 and 1, ordered descending, correct count).

## Out of scope

- Encoding new text that doesn't exist in the table (would require loading the model).
- Clue embedding similarity (only response embeddings).
- Changes to `db.py` or library code.
- Adding type hints or error handling (per notebook conventions).
