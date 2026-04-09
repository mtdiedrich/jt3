# feat-batch-scraper-notebook

## Goal

Add a `crawler` module for batch-fetching episodes (respecting robots.txt) and a Jupyter notebook that downloads the first 10 J! Archive episodes in order, saving them to DuckDB.

## Current behavior

- `scraper.fetch_episode(url)` fetches a single episode by URL.
- No batch-fetching, robots.txt checking, or rate-limiting exists.
- No notebooks exist in the project.

## Target behavior

### New module: `src/jt3/crawler.py`

Provides:

1. **`check_robots(user_agent="*") -> float | None`** — Fetches `https://j-archive.com/robots.txt`, checks if `showgame.php` is allowed for the given user-agent, and returns the `Crawl-delay` value (as seconds). Returns `None` if disallowed.

2. **`episode_url(game_id: int) -> str`** — Returns `https://j-archive.com/showgame.php?game_id={game_id}`.

3. **`fetch_episodes(game_ids: Iterable[int], *, delay: float | None = None, user_agent: str = "*") -> Iterator[Episode]`** — Generator that:
   - Calls `check_robots()` first; raises `PermissionError` if disallowed.
   - Uses the `Crawl-delay` from robots.txt if `delay` is not explicitly passed.
   - Yields `Episode` objects one at a time.
   - Sleeps `delay` seconds between requests (except before the first).
   - Skips game_ids that return HTTP errors (logs a warning).

### New notebook: `notebooks/scrape_episodes.ipynb`

Cells:
1. Markdown: title + description
2. Code: imports (`jt3`, `tqdm.notebook`)
3. Code: check robots.txt, extract crawl delay
4. Code: define `game_ids = range(1, 11)` (first 10)
5. Code: loop with `tqdm.notebook.tqdm`, fetch each episode, save to DuckDB, collect results
6. Code: summary — list episodes saved with show number & air date

### Updated files

- **`pyproject.toml`**: Add `tqdm` to runtime deps; add `tqdm`, `ipykernel` to dev deps.
- **`src/jt3/__init__.py`**: Export `check_robots`, `episode_url`, `fetch_episodes` from crawler.

## Files to change

| File | Action |
|------|--------|
| `src/jt3/crawler.py` | Create |
| `tests/test_crawler.py` | Create |
| `notebooks/scrape_episodes.ipynb` | Create |
| `pyproject.toml` | Modify — add tqdm, ipykernel |
| `src/jt3/__init__.py` | Modify — add crawler exports |

## Test plan

| # | Test | File | Expected |
|---|------|------|----------|
| 1 | `episode_url` returns correct URL | `test_crawler.py` | `"https://j-archive.com/showgame.php?game_id=42"` |
| 2 | `check_robots` returns crawl-delay when allowed | `test_crawler.py` | Returns `20.0` (mocked) |
| 3 | `check_robots` returns `None` when disallowed | `test_crawler.py` | Returns `None` (mocked) |
| 4 | `fetch_episodes` yields episodes in order | `test_crawler.py` | Yields episodes for each game_id |
| 5 | `fetch_episodes` raises `PermissionError` when robots disallows | `test_crawler.py` | `PermissionError` raised |
| 6 | `fetch_episodes` skips HTTP errors | `test_crawler.py` | Continues past errored IDs |
| 7 | `fetch_episodes` uses robots.txt delay by default | `test_crawler.py` | `time.sleep` called with `20.0` |

## Out of scope

- Async/concurrent fetching
- Resuming interrupted crawls
- Discovering game_ids from season pages (user provides IDs)
- CLI interface
- Persistent crawl state
