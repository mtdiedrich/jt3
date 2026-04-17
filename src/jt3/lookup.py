"""Embedding lookup from DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from .db import DEFAULT_DB_PATH


def lookup_embeddings(
    queries: list[str],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Look up embeddings for a list of clue texts.

    Returns ``(found, missing)`` where *found* maps clue_text to its
    embedding array and *missing* lists queries not in the table.
    """
    if not queries:
        return {}, []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT clue_text, embedding FROM embeddings "
            "WHERE clue_text IN (SELECT unnest(?))",
            [queries],
        ).fetchall()
    finally:
        con.close()

    found = {text: np.array(emb, dtype=np.float32) for text, emb in rows}
    missing = [q for q in queries if q not in found]
    return found, missing
