# feat-season-scraper

## Goal

Add functions to list seasons, get episode game IDs for a season, and scrape an entire season — then update the notebook to use season-based scraping.

## Current behavior

- `crawler.py` has `episode_url()`, `check_robots()`, `fetch_episodes()` which work on an explicit list of `game_id` integers.
- The notebook manually defines `game_ids = range(1, 11)`.
- No awareness of J! Archive season structure.

## Target behavior

### New functions in `src/jt3/crawler.py`

1. **`season_url(season: int | str) -> str`** — Returns `https://j-archive.com/showseason.php?season={season}`.

2. **`list_seasons() -> list[dict]`** — Fetches `https://j-archive.com/listseasons.php`, parses the table, returns a list of dicts with keys `season` (str), `name` (str, e.g. "Season 1"), `game_count` (int). Each dict represents one row from the table.

3. **`get_season_game_ids(season: int | str) -> list[int]`** — Fetches the season page, extracts all `showgame.php?game_id=N` links, returns the game IDs as a list of ints in the order they appear on the page (newest first on j-archive, so reversed = chronological).

4. **`fetch_season(season: int | str, *, delay: float | None = None, user_agent: str = "*") -> Iterator[Episode]`** — Convenience that calls `get_season_game_ids(season)`, reverses to chronological order, then delegates to `fetch_episodes()`.

### HTML structures

**`listseasons.php`**: Each season is a `<tr>` containing:
```html
<tr>
  <td><a href="showseason.php?season=42">Season 42</a></td>
  <td class="left_padded">2025-09-08 to 2026-07-24</td>
  <td align="right" class="left_padded">(153 games archived)</td>
</tr>
```

**`showseason.php?season=N`**: Each episode is a link:
```html
<a href="showgame.php?game_id=7792" title="Taped 1985-01-30">&#35;193, aired&#160;1985-06-05</a>
```

### Updated notebook: `notebooks/scrape_episodes.ipynb`

Update to scrape Season 1 (first 10 episodes only, as a test) using the new season-based functions.

## Files to change

| File | Action |
|------|--------|
| `src/jt3/crawler.py` | Modify — add `season_url`, `list_seasons`, `get_season_game_ids`, `fetch_season` |
| `tests/test_crawler.py` | Modify — add tests for new functions |
| `tests/fixtures/sample_season.html` | Create — minimal fixture for season page |
| `tests/fixtures/sample_seasons_list.html` | Create — minimal fixture for seasons list |
| `src/jt3/__init__.py` | Modify — export new functions |
| `notebooks/scrape_episodes.ipynb` | Modify — use season-based scraping |
| `docs/specs/feat-season-scraper.md` | Create (this file) |

## Test plan

| # | Test | File | Expected |
|---|------|------|----------|
| 1 | `season_url(1)` returns correct URL | `test_crawler.py` | `"https://j-archive.com/showseason.php?season=1"` |
| 2 | `season_url("superjeopardy")` handles string seasons | `test_crawler.py` | `"https://j-archive.com/showseason.php?season=superjeopardy"` |
| 3 | `list_seasons` parses fixture HTML | `test_crawler.py` | Returns list of dicts with season, name, game_count |
| 4 | `get_season_game_ids` extracts game_ids from fixture | `test_crawler.py` | Returns `[101, 102, 103]` (from fixture) |
| 5 | `fetch_season` calls `get_season_game_ids` then `fetch_episodes` | `test_crawler.py` | Yields episodes in chronological order |

## Out of scope

- Caching season pages
- Discovering all game IDs across all seasons automatically
- CLI interface
- Async
