# jt3

Fetch and parse [J! Archive](https://j-archive.com/) Jeopardy! episodes into typed Python objects.

## Usage

```python
from jt3 import fetch_episode

# Fetch a live episode from J! Archive
episode = fetch_episode("https://j-archive.com/showgame.php?game_id=9418")

print(episode.show_number)   # 9536
print(episode.air_date)      # 2026-04-06

for contestant in episode.contestants:
    print(contestant.name, "-", contestant.description)

for round_ in episode.rounds:
    print(f"\n=== {round_.name} ===")
    for category in round_.categories:
        print(f"  {category.name}")
        for clue in category.clues:
            dd = " [DD]" if clue.is_daily_double else ""
            print(f"    ${clue.value}{dd}: {clue.text}")
            print(f"      → {clue.correct_response}")
```

If you already have raw HTML (e.g. from a cache), use `parse_episode` directly:

```python
from jt3 import parse_episode

html = open("game_9418.html").read()
episode = parse_episode(html, game_id=9418)
```

## Data Model

| Class | Key fields |
|---|---|
| `Episode` | `game_id`, `show_number`, `air_date`, `contestants`, `rounds` |
| `Round` | `name` (`"Jeopardy!"` / `"Double Jeopardy!"` / `"Final Jeopardy!"`), `categories` |
| `Category` | `name`, `comments`, `clues` |
| `Clue` | `clue_id`, `order`, `value`, `is_daily_double`, `text`, `correct_response`, `answerer` |
| `Contestant` | `name`, `description`, `player_id` |

## Quick Start

```powershell
# Create virtual environment
uv venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install -e .[dev]

# Run commands
uv run pytest -v                                    # Run tests
uv run pytest --cov=jt3 --cov-report=html         # Tests with coverage
uv run ruff check .                                 # Check code quality
uv run ruff format .                                # Format code
```