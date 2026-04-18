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


def _validate_qualified_name(name: str) -> str:
    """Validate an optionally schema-qualified SQL name (e.g. ``schema.table``)."""
    parts = name.split(".")
    if len(parts) not in (1, 2) or not all(_IDENTIFIER_RE.match(p) for p in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "jt3.duckdb"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating parent dirs as needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def search_similar(
    embedding: list[float],
    *,
    n: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = "embeddings.responses",
    text_column: str = "response_text",
) -> list[tuple[str, float]]:
    """Return the *n* most similar texts by cosine similarity.

    Returns a list of ``(text, score)`` tuples sorted by descending
    similarity, or an empty list if the table does not exist.
    """
    tbl = _validate_qualified_name(table)
    col = _validate_identifier(text_column)
    parts = tbl.split(".")
    table_name = parts[-1]
    schema_name = parts[0] if len(parts) == 2 else "main"
    con = get_connection(db_path)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema_name, table_name],
        ).fetchone()[0]
        if not exists:
            return []
        rows = con.execute(
            f"SELECT {col}, list_cosine_similarity(embedding, ?::FLOAT[]) AS score "
            f"FROM {tbl} ORDER BY score DESC LIMIT ?",
            [embedding, n],
        ).fetchall()
        return [(row[0], row[1]) for row in rows]
    finally:
        con.close()
