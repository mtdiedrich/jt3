"""Embedding generation for clue, response, category, and full-context texts."""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer

from .db import (
    DEFAULT_DB_PATH,
    get_connection,
    save_embeddings,
)

MODELS: dict[str, dict] = {
    "all_MiniLM_L6_v2": dict(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
    ),
    "qwen3_embedding_06B": dict(
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
    ),
    "qwen3_embedding_06B_trunc_32": dict(
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        truncate_dim=32,
    ),
}


def load_model(model_key: str) -> SentenceTransformer:
    """Load a SentenceTransformer by config key."""
    return SentenceTransformer(**MODELS[model_key])


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


def fetch_category_texts(*, db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    """Return distinct category names from the database."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT name FROM categories ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def fetch_full_context_texts(*, db_path: str | Path = DEFAULT_DB_PATH) -> list[str]:
    """Return ``"Category: Clue → Response"`` strings for every clue with a response."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT cat.name, c.text, c.correct_response "
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


# ---------------------------------------------------------------------------
# Generation pipelines
# ---------------------------------------------------------------------------


def generate_clue_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all clue texts and save to ``embeddings.clues``.

    Returns the number of embeddings saved.
    """
    clues = fetch_clue_texts(db_path=db_path)
    if not clues:
        return 0
    embeddings = model.encode(clues, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(clues, embeddings, db_path=db_path, table="clues")
    return len(clues)


def generate_response_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all response texts and save to ``embeddings.responses``.

    Returns the number of embeddings saved.
    """
    responses = fetch_response_texts(db_path=db_path)
    if not responses:
        return 0
    embeddings = model.encode(responses, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(responses, embeddings, db_path=db_path, table="responses")
    return len(responses)


def generate_category_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all category names and save to ``embeddings.categories``.

    Returns the number of embeddings saved.
    """
    categories = fetch_category_texts(db_path=db_path)
    if not categories:
        return 0
    embeddings = model.encode(categories, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(categories, embeddings, db_path=db_path, table="categories")
    return len(categories)


def generate_full_context_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode ``"Category: Clue → Response"`` strings and save to ``embeddings.full_context``.

    Returns the number of embeddings saved.
    """
    contexts = fetch_full_context_texts(db_path=db_path)
    if not contexts:
        return 0
    embeddings = model.encode(contexts, batch_size=batch_size, show_progress_bar=True)
    save_embeddings(contexts, embeddings, db_path=db_path, table="full_context")
    return len(contexts)
