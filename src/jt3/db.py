"""Shared DuckDB connection utilities."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Validate a SQL identifier to prevent injection. Returns *name* if valid."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "jt3.duckdb"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating parent dirs as needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))
