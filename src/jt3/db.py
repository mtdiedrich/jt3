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


def _table_exists(con, schema_name: str, table_name: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = ? AND table_name = ?",
        [schema_name, table_name],
    ).fetchone()[0]


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
    schema_name = parts[0] if len(parts) == 2 else "main"
    table_name = parts[-1]
    con = get_connection(db_path)
    try:
        if not _table_exists(con, schema_name, table_name):
            return []
        rows = con.execute(
            f"SELECT {col}, list_cosine_similarity(embedding, ?::FLOAT[]) AS score "
            f"FROM {tbl} ORDER BY score DESC LIMIT ?",
            [embedding, n],
        ).fetchall()
        return [(row[0], row[1]) for row in rows]
    finally:
        con.close()


def search_by_min_similarity(
    embeddings: list[list[float]],
    *,
    n: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = "embeddings.responses",
    text_column: str = "response_text",
) -> list[tuple[str, float]]:
    """Return the *n* texts with the highest minimum similarity across all query embeddings.

    Scores each row by ``min(cosine_sim(row, q) for q in embeddings)``, so results
    must be genuinely close to *every* query rather than just their average.

    Returns a list of ``(text, score)`` tuples sorted by descending score,
    or an empty list if the table does not exist.
    """
    if not embeddings:
        return []
    tbl = _validate_qualified_name(table)
    col = _validate_identifier(text_column)
    parts = tbl.split(".")
    schema_name = parts[0] if len(parts) == 2 else "main"
    table_name = parts[-1]
    con = get_connection(db_path)
    try:
        if not _table_exists(con, schema_name, table_name):
            return []
        dim = len(embeddings[0])
        sim_exprs = ", ".join(
            f"list_cosine_similarity(embedding, ?::FLOAT[{dim}])"
            for _ in embeddings
        )
        rows = con.execute(
            f"SELECT {col}, LEAST({sim_exprs}) AS score "
            f"FROM {tbl} ORDER BY score DESC LIMIT ?",
            [*embeddings, n],
        ).fetchall()
        return [(row[0], row[1]) for row in rows]
    finally:
        con.close()
