# Spec: Episode Scraper

## Goal
Add functionality to fetch a J! Archive episode page and parse all its data into a new `Episode` dataclass.

## Current Behavior
N/A — new feature. The package has no models or scraping logic.

## Target Behavior
After the change, a caller can do:

```python
from jt3 import fetch_episode

episode = fetch_episode("https://j-archive.com/showgame.php?game_id=9418")
episode.game_id        # 9418
episode.show_number    # 9536
episode.air_date       # datetime.date(2026, 4, 6)
episode.contestants    # [Contestant(...), ...]
episode.rounds         # [Round("Jeopardy!"), Round("Double Jeopardy!"), Round("Final Jeopardy!")]

round_j = episode.rounds[0]
round_j.categories[0].name          # "AUTHORS' LESSER-KNOWN WORKS"
round_j.categories[0].clues[0].value          # 200
round_j.categories[0].clues[0].text           # "D.H. Lawrence wrote..."
round_j.categories[0].clues[0].correct_response  # "Australia"
round_j.categories[0].clues[0].is_daily_double    # False
```

`parse_episode(html, game_id)` is also exported for callers that already have the raw HTML.

## J! Archive HTML Structure (key elements)

| Element | Selector | Contents |
|---|---|---|
| Page title | `h1` | "Show #9536 - Monday, April 6, 2026" |
| Contestants | `div#contestants_table p` | `<a>Name</a>, description` |
| J! round | `div#jeopardy_round` | Contains `table.round` |
| DJ! round | `div#double_jeopardy_round` | Contains `table.round` |
| FJ! round | `div#final_jeopardy_round` | Contains category + single clue |
| Category name | `td.category_name` | Plain text |
| Category comments | `td.category_comments` | Optional plain text |
| Clue value | `td.clue_value` | "$200" |
| DD value | `td.clue_value_daily_double` | "DD: $1,000" |
| Clue order | `td.clue_order_number` | Integer string |
| Clue text | `td#clue_J_1_1.clue_text` | Plain text |
| Answer (hidden) | `div#clue_J_1_1_r em.correct_response` | Plain text |
| Answerer | `div[onmouseover] td.right` | Contestant name |

## Data Model

### `Contestant`
| Field | Type | Description |
|---|---|---|
| `name` | `str` | Player's display name |
| `description` | `str` | Occupation and hometown |
| `player_id` | `int \| None` | J! Archive player ID |

### `Clue`
| Field | Type | Description |
|---|---|---|
| `clue_id` | `str` | Short ID, e.g. `"J_1_1"`, `"DJ_3_5"`, `"FJ"` |
| `order` | `int \| None` | Order the clue was selected (1-based) |
| `value` | `int` | Dollar value (0 for FJ) |
| `is_daily_double` | `bool` | True if the clue is a Daily Double |
| `text` | `str` | The clue itself |
| `correct_response` | `str \| None` | The correct answer |
| `answerer` | `str \| None` | Name of the contestant who answered correctly |

### `Category`
| Field | Type | Description |
|---|---|---|
| `name` | `str` | UPPERCASE category title |
| `comments` | `str \| None` | Optional narrator comment shown with category |
| `clues` | `list[Clue]` | 5 clues (or 1 for FJ) |

### `Round`
| Field | Type | Description |
|---|---|---|
| `name` | `str` | `"Jeopardy!"`, `"Double Jeopardy!"`, or `"Final Jeopardy!"` |
| `categories` | `list[Category]` | 6 categories (J!/DJ!) or 1 (FJ) |

### `Episode`
| Field | Type | Description |
|---|---|---|
| `game_id` | `int` | J! Archive game ID (from URL) |
| `show_number` | `int \| None` | Official show number |
| `air_date` | `date \| None` | Air date |
| `contestants` | `list[Contestant]` | Usually 3 contestants |
| `rounds` | `list[Round]` | 2–3 rounds (J, DJ, optionally FJ) |

## Files to Change

| File | Action | Summary |
|---|---|---|
| `pyproject.toml` | Modify | Add `requests`, `beautifulsoup4`, `lxml` to `[project.dependencies]` |
| `src/jt3/models.py` | Create | `Contestant`, `Clue`, `Category`, `Round`, `Episode` dataclasses |
| `src/jt3/scraper.py` | Create | `fetch_episode(url)`, `parse_episode(html, game_id)` + private helpers |
| `src/jt3/__init__.py` | Modify | Export `Episode`, `fetch_episode`, `parse_episode` |
| `tests/fixtures/sample_game.html` | Create | Minimal representative j-archive HTML for deterministic tests |
| `tests/test_models.py` | Create | Unit tests for data model construction |
| `tests/test_scraper.py` | Create | Parsing tests using fixture HTML |

## Step-by-Step Instructions

1. **`pyproject.toml`** — Add under `[project]`:
   ```toml
   dependencies = ["requests>=2.28", "beautifulsoup4>=4.12", "lxml>=4.9"]
   ```

2. **`src/jt3/models.py`** — Create with dataclasses for all five types (see Data Model above). All lists default to `field(default_factory=list)`. All `| None` fields default to `None`.

3. **`src/jt3/scraper.py`** — Create with:
   - `fetch_episode(url: str) -> Episode` — extracts `game_id` via regex, GETs the page, calls `parse_episode`.
   - `parse_episode(html: str, game_id: int) -> Episode` — orchestrates the sub-parsers.
   - `_parse_title(soup)` — extracts show_number and air_date from `<h1>`.
   - `_parse_contestants(soup)` — extracts contestants from `div#contestants_table`.
   - `_parse_rounds(soup)` — iterates the three round div IDs, dispatches to `_parse_regular_round` or `_parse_final_jeopardy_round`.
   - `_parse_regular_round(div, round_name)` — parses `table.round`: row 0 = categories, rows 1–5 = clues.
   - `_parse_clue_cell(cell)` — parses a single `td.clue` cell: value, order, text, DD flag, correct_response, answerer.
   - `_parse_final_jeopardy_round(div)` — special-case parser for the FJ div.
   - `_parse_dollar(text)` — strips "$" and "," and converts to int.

4. **`src/jt3/__init__.py`** — Add imports and `__all__`.

5. **`tests/fixtures/sample_game.html`** — Minimal HTML with 2 J! categories × 2 clues, 1 DJ category × 1 clue (including a DD), and 1 FJ clue, plus 3 contestants and a valid `<h1>`.

6. **`tests/test_models.py`** — Tests for direct construction of each dataclass.

7. **`tests/test_scraper.py`** — Tests for each parser function using the fixture.

## Test Plan

### `tests/test_models.py`
| # | Test | Expected |
|---|---|---|
| 1 | Construct `Contestant(name="Alice", description="teacher from Omaha, NE")` | Fields set correctly; player_id=None |
| 2 | Construct `Clue(clue_id="J_1_1", value=200, ...)` | All fields accessible |
| 3 | Construct `Category(name="SCIENCE", clues=[...])` | `len(clues) == N` |
| 4 | Construct `Round(name="Jeopardy!", categories=[...])` | `len(categories) == N` |
| 5 | Construct `Episode(game_id=9418, ...)` | All top-level fields accessible |

### `tests/test_scraper.py`
| # | Test | Expected |
|---|---|---|
| 6 | `_parse_title` with `<h1>Show #9536 - Monday, April 6, 2026</h1>` | `(9536, date(2026,4,6))` |
| 7 | `_parse_title` with no `<h1>` | `(None, None)` |
| 8 | `parse_episode(fixture_html, 9418)` | `episode.game_id == 9418` |
| 9 | `parse_episode(fixture_html, 9418)` | `episode.show_number == 1` (fixture value) |
| 10 | `parse_episode(fixture_html, 9418)` | `episode.air_date == date(2026, 1, 1)` (fixture value) |
| 11 | `parse_episode(fixture_html, 9418)` | `len(episode.contestants) == 3` |
| 12 | First contestant name and description match fixture | — |
| 13 | `len(episode.rounds) >= 1` | At least J! round parsed |
| 14 | First J! category name matches fixture value | — |
| 15 | First clue in first J! category has correct value, text, correct_response | — |
| 16 | Daily double clue has `is_daily_double == True` | — |
| 17 | Final Jeopardy round is present and has category + clue | — |
| 18 | `_parse_dollar("$1,200")` returns `1200` | — |
| 19 | `_parse_dollar("DD: $2,000")` with strip produces `2000` | — |

## Out of Scope
- Round scores / score tables
- Wagering suggestions
- Season/show listing or pagination
- Async HTTP
- Caching or rate limiting
- Tiebreaker clues
