"""Embedding generation for clue and response texts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from ..db import DEFAULT_DB_PATH, get_connection, search_by_min_similarity, search_similar
from .db import save_embeddings


def compute_centroid(embeddings: npt.ArrayLike) -> list[float]:
    """Return the element-wise mean of multiple embeddings."""
    arr = np.asarray(embeddings, dtype=np.float32)
    return np.mean(arr, axis=0).tolist()


def _get_model_path(model: SentenceTransformer) -> str:
    """Extract the HuggingFace model name/path from a SentenceTransformer."""
    return model.model_card_data.base_model


# ---------------------------------------------------------------------------
# Text fetching
# ---------------------------------------------------------------------------


def fetch_clue_texts(*, db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    """Return distinct clue texts from the database."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT c.text AS clue_text "
            "FROM clues AS c "
            "ORDER BY c.game_id DESC, c.round_index, c.category_index, c.clue_order"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def fetch_category_texts(*, db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    """Return distinct category names from the database."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT cat.name AS category_name "
            "FROM categories AS cat "
            "ORDER BY cat.game_id DESC, cat.round_index, cat.category_index"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def fetch_complete_texts(
    *, db_path: str | Path = DEFAULT_DB_PATH
) -> list[str]:
    """Return distinct ``"Category: Clue → Response"`` strings."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT cat.name, c.text, c.correct_response "
            "FROM clues c "
            "JOIN categories cat ON c.game_id = cat.game_id "
            "AND c.round_index = cat.round_index "
            "AND c.category_index = cat.category_index "
            "WHERE c.correct_response IS NOT NULL "
            "ORDER BY c.game_id DESC, c.round_index, c.category_index, c.clue_order"
        ).fetchall()
        return [f"{cat}: {clue} \u2192 {resp}" for cat, clue, resp in rows]
    finally:
        con.close()


def fetch_response_texts(*, db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    """Return distinct non-null response texts from the database."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT c.correct_response AS response_text "
            "FROM clues AS c "
            "WHERE c.correct_response IS NOT NULL "
            "ORDER BY c.game_id DESC, c.round_index, c.category_index, c.clue_order"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Generation pipelines
# ---------------------------------------------------------------------------


def generate_clue_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all clue texts and save to the ``embeddings.clues`` table.

    Returns the number of embeddings saved.
    """
    clues = fetch_clue_texts(db_path=db_path)
    if not clues:
        return 0
    embeddings = model.encode(clues, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(
        clues,
        embeddings,
        db_path=db_path,
        table="embeddings.clues",
        text_column="clue_text",
        model_name=_get_model_path(model),
    )
    return len(clues)


def generate_response_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all response texts and save to the ``embeddings.responses`` table.

    Returns the number of embeddings saved.
    """
    responses = fetch_response_texts(db_path=db_path)
    if not responses:
        return 0
    embeddings = model.encode(responses, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(
        responses,
        embeddings,
        db_path=db_path,
        table="embeddings.responses",
        text_column="response_text",
        model_name=_get_model_path(model),
    )
    return len(responses)


def generate_category_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all category names and save to the ``embeddings.categories`` table.

    Returns the number of embeddings saved.
    """
    categories = fetch_category_texts(db_path=db_path)
    if not categories:
        return 0
    embeddings = model.encode(categories, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(
        categories,
        embeddings,
        db_path=db_path,
        table="embeddings.categories",
        text_column="category_name",
        model_name=_get_model_path(model),
    )
    return len(categories)


def generate_complete_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode ``"Category: Clue → Response"`` strings and save to
    the ``embeddings.complete`` table.

    Returns the number of embeddings saved.
    """
    texts = fetch_complete_texts(db_path=db_path)
    if not texts:
        return 0
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(
        texts,
        embeddings,
        db_path=db_path,
        table="embeddings.complete",
        text_column="text",
        model_name=_get_model_path(model),
    )
    return len(texts)


# ---------------------------------------------------------------------------
# Multi-table search
# ---------------------------------------------------------------------------

EMBEDDING_TABLES: list[tuple[str, str]] = [
    ("embeddings.clues", "clue_text"),
    ("embeddings.responses", "response_text"),
    ("embeddings.categories", "category_name"),
]


def search_all_tables(
    embedding: list[float],
    *,
    n: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, list[tuple[str, float]]]:
    """Search clues, responses, and categories tables for similar embeddings.

    Returns a dict keyed by table name with lists of ``(text, score)`` tuples.
    """
    return {
        table: search_similar(
            embedding, n=n, db_path=db_path, table=table, text_column=text_column
        )
        for table, text_column in EMBEDDING_TABLES
    }


def search_all_tables_by_min_sim(
    embeddings: list[list[float]],
    *,
    n: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, list[tuple[str, float]]]:
    """Search all tables scoring each row by its minimum similarity across all query embeddings.

    Results must be genuinely close to every query, not just their average.
    Returns a dict keyed by table name with lists of ``(text, score)`` tuples.
    """
    return {
        table: search_by_min_similarity(
            embeddings, n=n, db_path=db_path, table=table, text_column=text_column
        )
        for table, text_column in EMBEDDING_TABLES
    }
