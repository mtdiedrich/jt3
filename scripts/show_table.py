"""Print a polars DataFrame of any table in the DuckDB database.

Usage:
    uv run python scripts/show_table.py <table_name> [limit]
"""

from __future__ import annotations

import argparse

import duckdb
import polars as pl

from jt3.db import DEFAULT_DB_PATH, _validate_identifier


def main() -> None:
    parser = argparse.ArgumentParser(description="SELECT * from a DuckDB table and print as a polars DataFrame")
    parser.add_argument("table_name", help="Name of the table to query")
    parser.add_argument("limit", nargs="?", type=int, default=None, help="Optional row limit")
    args = parser.parse_args()

    table = _validate_identifier(args.table_name)

    query = f"SELECT * FROM {table}"
    if args.limit is not None:
        query += f" LIMIT {args.limit}"

    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        df = con.sql(query).pl()

    print(df)


if __name__ == "__main__":
    main()
