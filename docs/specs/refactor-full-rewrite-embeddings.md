# refactor: full-rewrite embeddings

## Goal

Change `embed_clues` from incremental (skip existing) to full-rewrite (delete all embeddings, re-embed everything) so the DB always contains exactly one model's embeddings.

## Current behavior

- `embed_clues()` queries existing `(clue, field, model_name)` tuples, skips them, and only embeds new ones.
- The `embeddings` table PK includes `model_name`, allowing multiple models to coexist.
- The `embed_clues.ipynb` notebook duplicates this incremental logic inline.
- `explore_embeddings.ipynb` joins on `embeddings` without filtering by `model_name`, which silently mixes models if more than one is present.

## Target behavior

- `embed_clues()` deletes **all** rows from the `embeddings` table, then embeds every clue and inserts fresh rows.
- The `embeddings` PK becomes `(game_id, round_index, category_index, clue_id, field)` — `model_name` stays as a regular metadata column.
- The `embed_clues.ipynb` notebook removes "check existing" logic; it builds entries for all clues, deletes old embeddings, and inserts new ones.
- `explore_embeddings.ipynb` requires no changes (there is only ever one model's data).

## Files to change

| File | Change |
|------|--------|
| `src/jt3/db.py` | Remove `model_name` from embeddings PK |
| `src/jt3/embeddings.py` | Rewrite `embed_clues()`: delete-all → embed-all → insert-all |
| `notebooks/embed_clues.ipynb` | Remove incremental filter cells; add DELETE before INSERT |
| `tests/test_embeddings.py` | Update `test_embed_clues_skips_existing` and `test_embed_clues_incremental_new_episode` |

## Step-by-step

1. **`db.py`**: Change embeddings PK from `(game_id, round_index, category_index, clue_id, field, model_name)` to `(game_id, round_index, category_index, clue_id, field)`.

2. **`embeddings.py` – `embed_clues()`**:
   - Remove the "find existing" query and `existing` set.
   - Collect **all** `(clue, field)` entries (no skip check).
   - Before inserting, execute `DELETE FROM embeddings`.
   - Insert all rows inside the same transaction.

3. **`tests/test_embeddings.py`**:
   - `test_embed_clues_skips_existing` → rename to `test_embed_clues_replaces_on_rerun`: second call returns 3 (not 0), total rows still 3.
   - `test_embed_clues_incremental_new_episode` → rename to `test_embed_clues_reembeds_all_episodes`: after adding a second episode, count should be 6 (all clues, both episodes).

4. **`embed_clues.ipynb`**:
   - Remove cell that queries existing embeddings (`existing_df`).
   - Remove filtering logic (existing set, `if key not in existing`).
   - Build entries for all clues directly.
   - Before INSERT, add `DELETE FROM embeddings`.
   - Update markdown descriptions to say "full rewrite" not "incremental".

## Test plan

| Test | Expected |
|------|----------|
| `test_embed_clues_populated_db` | count == 3, 3 rows in DB (unchanged) |
| `test_embed_clues_skips_null_responses` | 1 correct_response row (unchanged) |
| `test_embed_clues_empty_db` | count == 0 (unchanged) |
| `test_embed_clues_replaces_on_rerun` | count1 == 3, count2 == 3, total rows == 3 |
| `test_embed_clues_reembeds_all_episodes` | count1 == 3, count2 == 6, total rows == 6 |
| `test_embed_clues_stores_source_text` | unchanged |
| `test_embed_clues_stores_model_name` | unchanged |

## Out of scope

- `nearest_to_centroid()` — still filters by `model_name`, which is harmless.
- `explore_embeddings.ipynb` — no changes needed; queries are correct when there's one model.
- Dropping the `model_name` column entirely — keep it as metadata.
