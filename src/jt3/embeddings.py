"""Embedding generation for clue text and answers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from .db import DEFAULT_DB_PATH, get_connection

if TYPE_CHECKING:
    import duckdb

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE: str | None = None  # None = auto-detect (prefers CUDA)

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, *, device: str | None = DEFAULT_DEVICE) -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading it on first use."""
    key = (model_name, device)
    if key not in _model_cache:
        _model_cache[key] = SentenceTransformer(model_name, device=device)
    return _model_cache[key]


def embed(
    text: str,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
) -> np.ndarray:
    """Embed a single string. Returns a 1-D numpy array."""
    model = _get_model(model_name, device=device)
    return model.encode([text])[0]


def embed_batch(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Embed multiple strings. Returns a 2-D numpy array of shape (len(texts), dim)."""
    model = _get_model(model_name, device=device)
    return model.encode(texts, show_progress_bar=show_progress_bar)


def embed_clues(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
) -> int:
    """Embed all clue texts and correct_responses in the DB.

    Replaces all existing embeddings with fresh ones for the given model.
    Returns the total number of embedding rows written.
    """
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT game_id, round_index, category_index, clue_id, "
            "text, correct_response FROM clues"
        ).fetchall()

        if not rows:
            return 0

        entries: list[tuple[int, int, int, str, str, str]] = []
        texts: list[str] = []

        for (
            game_id,
            round_index,
            category_index,
            clue_id,
            text,
            correct_response,
        ) in rows:
            entries.append(
                (game_id, round_index, category_index, clue_id, "text", text)
            )
            texts.append(text)

            if correct_response is not None:
                entries.append(
                    (
                        game_id,
                        round_index,
                        category_index,
                        clue_id,
                        "correct_response",
                        correct_response,
                    )
                )
                texts.append(correct_response)

        vectors = embed_batch(texts, model_name=model_name, device=device)

        con.begin()
        con.execute("DELETE FROM embeddings")
        for i, (
            game_id,
            round_index,
            category_index,
            clue_id,
            field,
            source_text,
        ) in enumerate(entries):
            con.execute(
                "INSERT INTO embeddings "
                "(game_id, round_index, category_index, clue_id, field, "
                "source_text, embedding, model_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    game_id,
                    round_index,
                    category_index,
                    clue_id,
                    field,
                    source_text,
                    vectors[i].tolist(),
                    model_name,
                ],
            )

        con.commit()
        return len(entries)
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def nearest_to_centroid(
    answers: list[str],
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    model_name: str = DEFAULT_MODEL,
    device: str | None = DEFAULT_DEVICE,
    n: int = 10,
    exclude_radius: float | None = None,
):
    """Find the n answer embeddings closest to the centroid of the given answers.

    Parameters
    ----------
    exclude_radius:
        If set, exclude any DB answer whose cosine similarity to **any** of the
        input answer embeddings is >= this threshold.  For example, 0.85 filters
        out answers that are near-duplicates of the inputs.

    Returns a Polars DataFrame with columns ``answer_text`` and ``similarity``.
    """
    if not answers:
        raise ValueError("answers must be a non-empty list of strings")

    vectors = embed_batch(answers, model_name=model_name, device=device)
    centroid = vectors.mean(axis=0).tolist()

    _own_con = con is None
    if _own_con:
        con = get_connection(db_path)
    try:
        if exclude_radius is not None:
            import polars as pl

            input_df = pl.DataFrame(
                {"input_embedding": [v.tolist() for v in vectors]}
            )
            con.register("_input_embeddings", input_df)

            result = con.execute(
                "SELECT source_text AS answer_text, "
                "MAX(list_cosine_similarity(embedding, $1::FLOAT[])) AS similarity "
                "FROM embeddings e "
                "WHERE e.field = 'correct_response' AND e.model_name = $2 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM _input_embeddings ie "
                "  WHERE list_cosine_similarity(e.embedding, ie.input_embedding) >= $4"
                ") "
                "GROUP BY source_text "
                "ORDER BY similarity DESC "
                "LIMIT $3",
                [centroid, model_name, n, exclude_radius],
            ).pl()

            con.unregister("_input_embeddings")
        else:
            result = con.execute(
                "SELECT source_text AS answer_text, "
                "MAX(list_cosine_similarity(embedding, $1::FLOAT[])) AS similarity "
                "FROM embeddings "
                "WHERE field = 'correct_response' AND model_name = $2 "
                "GROUP BY source_text "
                "ORDER BY similarity DESC "
                "LIMIT $3",
                [centroid, model_name, n],
            ).pl()
    finally:
        if _own_con:
            con.close()
    return result
