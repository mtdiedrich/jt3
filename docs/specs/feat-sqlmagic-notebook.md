# Spec: SQL Magic Notebook for DuckDB Exploration

## Goal

Create a Jupyter notebook that connects to the project's DuckDB database (`data/jt3.duckdb`) using JupySQL's `%%sql` magic, allowing interactive SQL exploration of scraped Jeopardy data.

## Current Behavior

N/A — new notebook.

## Target Behavior

A notebook at `notebooks/explore_data.ipynb` that:

1. Loads the JupySQL extension (`%load_ext sql`)
2. Connects to `../data/jt3.duckdb` via a native DuckDB connection registered with jupysql
3. Provides starter `%%sql` cells querying each table (episodes, contestants, rounds, categories, clues)
4. Demonstrates the `<<` result-capture syntax for assigning query results to Python variables

Pattern follows the existing HawkeyeFootball project's approach:
```python
import duckdb
conn = duckdb.connect('../data/jt3.duckdb')
%load_ext sql
%sql conn --alias duckdb
```

## Files to Change

| File | Action | Summary |
|------|--------|---------|
| `pyproject.toml` | Modify | Add `jupysql` to dependencies |
| `notebooks/explore_data.ipynb` | Create | New notebook with sqlmagic setup + sample queries |

## Step-by-step Instructions

1. Add `"jupysql>=0.10"` to `[project.dependencies]` in `pyproject.toml`.
2. Create `notebooks/explore_data.ipynb` with cells:
   - Markdown: title & description
   - Code: `import duckdb`, connect to `../data/jt3.duckdb`, `%load_ext sql`, `%sql conn --alias duckdb`
   - Markdown: section header for episode queries
   - SQL cell: `%%sql` query against `episodes` table
   - SQL cell: `%%sql` query joining episodes + contestants
   - Markdown: section header for clue queries
   - SQL cell: `%%sql` query against `clues` with category join
   - SQL cell: `%%sql` with `<<` capture for further Python use

## Test Plan

No automated tests — this is a notebook artifact. Verification is that the notebook renders correctly and cells have valid syntax.

## Out of Scope

- Running the notebook end-to-end (requires populated DB)
- Adding visualization cells
- Modifying existing scraper or DB code
