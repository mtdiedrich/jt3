# Spec: DuckDB Episode Storage

## Goal

Add a persistent DuckDB-backed storage layer that provides full CRUD operations for scraped `Episode` data.

## Current Behavior

The package can scrape and parse J! Archive episodes into in-memory dataclass objects (`Episode`, `Round`, `Category`, `Clue`, `Contestant`). There is no persistence layer.

## Target Behavior

After the change, a caller can do:

```python
from jt3 import fetch_episode, save_episode, load_episode, delete_episode, list_episodes

episode = fetch_episode("https://j-archive.com/showgame.php?game_id=9418")

# Save to default DB (data/jt3.duckdb relative to CWD)
save_episode(episode)

# Save to custom path
save_episode(episode, db_path="my_data/custom.duckdb")

# Load back
ep = load_episode(9418)
assert ep.game_id == 9418

# List all stored episodes
episodes = list_episodes()
# Returns list of dicts: [{"game_id": 9418, "show_number": 9536, "air_date": date(2026, 4, 6)}, ...]

# Delete
deleted = delete_episode(9418)
assert deleted is True
```

Saving an episode with the same `game_id` again replaces (upserts) the old data.

## Database Schema (Normalized)

All tables use `game_id` as the link to the parent episode.

### `episodes`

| Column | Type | Constraint |
|--------|------|------------|
| game_id | INTEGER | PRIMARY KEY |
| show_number | INTEGER | NULLABLE |
| air_date | DATE | NULLABLE |

### `contestants`

| Column | Type | Constraint |
|--------|------|------------|
| game_id | INTEGER | NOT NULL, FK → episodes |
| name | VARCHAR | NOT NULL |
| description | VARCHAR | NOT NULL |
| player_id | INTEGER | NULLABLE |

### `rounds`

| Column | Type | Constraint |
|--------|------|------------|
| game_id | INTEGER | NOT NULL, FK → episodes |
| round_index | INTEGER | NOT NULL |
| name | VARCHAR | NOT NULL |
| PK | (game_id, round_index) | |

### `categories`

| Column | Type | Constraint |
|--------|------|------------|
| game_id | INTEGER | NOT NULL |
| round_index | INTEGER | NOT NULL |
| category_index | INTEGER | NOT NULL |
| name | VARCHAR | NOT NULL |
| comments | VARCHAR | NULLABLE |
| PK | (game_id, round_index, category_index) | |

### `clues`

| Column | Type | Constraint |
|--------|------|------------|
| game_id | INTEGER | NOT NULL |
| round_index | INTEGER | NOT NULL |
| category_index | INTEGER | NOT NULL |
| clue_id | VARCHAR | NOT NULL |
| clue_order | INTEGER | NULLABLE |
| value | INTEGER | NOT NULL |
| is_daily_double | BOOLEAN | NOT NULL |
| text | VARCHAR | NOT NULL |
| correct_response | VARCHAR | NULLABLE |
| answerer | VARCHAR | NULLABLE |
| PK | (game_id, round_index, category_index, clue_id) | |

## API

All functions live in `src/jt3/db.py`.

### `ensure_schema(con: duckdb.DuckDBPyConnection) -> None`

Creates all tables if they don't already exist. Called automatically by all CRUD functions.

### `get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection`

Opens (or creates) a DuckDB database at the given path, ensuring parent dirs exist, and calls `ensure_schema`.

### `save_episode(episode: Episode, db_path: str | Path = DEFAULT_DB_PATH) -> None`

Saves an Episode to the database. If a record with the same `game_id` exists, it is fully replaced (all child rows deleted first, then re-inserted). Runs in a single transaction.

### `load_episode(game_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> Episode | None`

Loads a full Episode from the database, reconstructing the complete dataclass hierarchy. Returns `None` if not found.

### `delete_episode(game_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> bool`

Deletes an episode and all its child data. Returns `True` if the episode existed and was deleted, `False` otherwise.

### `list_episodes(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]`

Returns a list of dicts with `game_id`, `show_number`, and `air_date` for all stored episodes, ordered by `game_id`.

### Default path

`DEFAULT_DB_PATH = Path("data/jt3.duckdb")`

## Files to Change

| File | Action | Summary |
|------|--------|---------|
| `pyproject.toml` | Modify | Add `duckdb>=1.0` to `[project.dependencies]` |
| `src/jt3/db.py` | Create | CRUD functions + schema initialization |
| `src/jt3/__init__.py` | Modify | Export `save_episode`, `load_episode`, `delete_episode`, `list_episodes` |
| `tests/test_db.py` | Create | Tests for all CRUD operations |

## Step-by-Step Instructions

1. **`pyproject.toml`** — Add `"duckdb>=1.0"` to the `dependencies` list.

2. **`src/jt3/db.py`** — Create with:
   - `DEFAULT_DB_PATH = Path("data/jt3.duckdb")`
   - `ensure_schema(con)` — executes CREATE TABLE IF NOT EXISTS for all 5 tables.
   - `get_connection(db_path)` — creates parent dirs, opens DuckDB connection, calls `ensure_schema`, returns connection.
   - `save_episode(episode, db_path)` — opens connection, deletes existing data for `game_id` (if any), inserts episode + contestants + rounds + categories + clues in a transaction.
   - `load_episode(game_id, db_path)` — opens connection, queries all tables for the game_id, reconstructs the `Episode` dataclass tree. Returns `None` if episode table has no matching row.
   - `delete_episode(game_id, db_path)` — opens connection, checks existence, deletes from all tables in dependency order, returns bool.
   - `list_episodes(db_path)` — opens connection, queries episodes table, returns list of dicts.

3. **`src/jt3/__init__.py`** — Add imports for `save_episode`, `load_episode`, `delete_episode`, `list_episodes` from `.db` and add to `__all__`.

4. **`tests/test_db.py`** — Uses `tmp_path` fixture for isolated DB files. Tests below.

## Test Plan

### `tests/test_db.py`

| # | Test | Expected |
|---|------|----------|
| 1 | `save_episode` then `load_episode` round-trips an Episode | Loaded episode matches original in all fields |
| 2 | `save_episode` twice with same game_id replaces data | Only one episode in DB, with updated data |
| 3 | `load_episode` for non-existent game_id | Returns `None` |
| 4 | `delete_episode` removes a saved episode | Returns `True`, subsequent load returns `None` |
| 5 | `delete_episode` for non-existent game_id | Returns `False` |
| 6 | `list_episodes` returns all saved episodes | Returns correct count and data |
| 7 | `list_episodes` on empty DB | Returns empty list |
| 8 | `ensure_schema` is idempotent | Calling twice does not error |
| 9 | Daily double clue round-trips correctly | `is_daily_double` preserved |
| 10 | Episode with no rounds/contestants round-trips | Empty lists preserved |
| 11 | `get_connection` creates parent directories | DB file created in nested path |

## Out of Scope

- CLI commands
- Bulk import/export
- Query/search within clues
- Migration tooling
- Connection pooling
- Async support
