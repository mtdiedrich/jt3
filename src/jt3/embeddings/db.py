"""DuckDB-backed storage for embedding data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from ..db import DEFAULT_DB_PATH, _validate_identifier, _validate_qualified_name, get_connection


def _ensure_schema(con, qualified_name: str) -> None:
    parts = qualified_name.split(".")
    if len(parts) == 2:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {parts[0]}")


def save_embeddings(
    texts: list[str],
    embeddings: npt.ArrayLike,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    table: str = "embeddings.clues",
    text_column: str = "text",
) -> None:
    """Bulk-save text→embedding pairs into a schema-qualified table."""
    embeddings = np.asarray(embeddings)
    tbl = _validate_qualified_name(table)
    col = _validate_identifier(text_column)
    dim = embeddings.shape[1]
    con = get_connection(db_path)
    try:
        _ensure_schema(con, tbl)
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        con.execute(
            f"CREATE TABLE {tbl} "
            f"({col} TEXT PRIMARY KEY, embedding FLOAT[{dim}] NOT NULL)"
        )
        df = pl.DataFrame({col: texts, "embedding": embeddings.tolist()}).unique(
            subset=[col], keep="first"
        )
        con.execute(f"INSERT INTO {tbl} SELECT {col}, embedding FROM df")
    finally:
        con.close()


def get_embedding(
    text: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    table: str = "embeddings.clues",
    text_column: str = "text",
) -> list[float] | None:
    """Return the embedding for *text*, or ``None`` if not found."""
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
            return None
        row = con.execute(
            f"SELECT embedding FROM {tbl} WHERE {col} = ?", [text]
        ).fetchone()
        if row is None:
            return None
        return list(row[0])
    finally:
        con.close()


def save_contextual_embeddings(
    texts: list[str],
    embeddings: npt.ArrayLike,
    context_texts: list[str],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Save contextual response embeddings with JSON context strings."""
    embeddings = np.asarray(embeddings)
    dim = embeddings.shape[1]
    con = get_connection(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS contextual_response_embeddings")
        con.execute(
            f"CREATE TABLE contextual_response_embeddings ("
            f"response_text TEXT PRIMARY KEY, "
            f"context_texts JSON NOT NULL, "
            f"embedding FLOAT[{dim}] NOT NULL)"
        )
        df = pl.DataFrame({
            "response_text": texts,
            "context_texts": context_texts,
            "embedding": embeddings.tolist(),
        })
        con.execute(
            "INSERT INTO contextual_response_embeddings "
            "SELECT response_text, context_texts, embedding FROM df"
        )
    finally:
        con.close()
