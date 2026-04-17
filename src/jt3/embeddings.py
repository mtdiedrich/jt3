"""Embedding generation for clue and response texts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .db import (
    DEFAULT_DB_PATH,
    get_connection,
    save_contextual_embeddings,
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


def fetch_response_contexts(
    *, db_path: str | Path = DEFAULT_DB_PATH
) -> dict[str, list[str]]:
    """Return a mapping of response text → list of context strings.

    Each context string has the form ``"Category: Clue → Response"``.
    """
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT c.correct_response, cat.name AS category, c.text AS clue "
            "FROM clues c "
            "JOIN categories cat ON c.game_id = cat.game_id "
            "AND c.round_index = cat.round_index "
            "AND c.category_index = cat.category_index "
            "WHERE c.correct_response IS NOT NULL"
        ).fetchall()
    finally:
        con.close()

    contexts: dict[str, list[str]] = {}
    for resp, category, clue in rows:
        ctx = f"{category}: {clue} \u2192 {resp}"
        contexts.setdefault(resp, []).append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Generation pipelines
# ---------------------------------------------------------------------------


def generate_clue_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all clue texts and save to the ``clue_embeddings`` table.

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
        table="clue_embeddings",
        text_column="clue_text",
    )
    return len(clues)


def generate_response_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Encode all response texts and save to the ``response_embeddings`` table.

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
        table="response_embeddings",
        text_column="response_text",
    )
    return len(responses)


def generate_contextual_response_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 128,
) -> int:
    """Build context strings, encode, average per response, L2-normalize, and save.

    Returns the number of response embeddings saved.
    """
    response_contexts = fetch_response_contexts(db_path=db_path)
    if not response_contexts:
        return 0

    # Flatten for batch encoding
    all_strings: list[str] = []
    string_to_response: list[int] = []
    response_keys = list(response_contexts.keys())

    for i, resp in enumerate(response_keys):
        for ctx in response_contexts[resp]:
            all_strings.append(ctx)
            string_to_response.append(i)

    ctx_embs = model.encode(all_strings, batch_size=batch_size, show_progress_bar=True)

    # Average embeddings per response, then L2-normalize
    ctx_avg = np.zeros((len(response_keys), ctx_embs.shape[1]))
    counts = np.zeros(len(response_keys))
    for idx, emb in zip(string_to_response, ctx_embs):
        ctx_avg[idx] += emb
        counts[idx] += 1
    ctx_avg /= counts[:, np.newaxis]
    ctx_avg /= np.linalg.norm(ctx_avg, axis=1, keepdims=True)

    context_json = [json.dumps(response_contexts[r]) for r in response_keys]
    save_contextual_embeddings(
        response_keys,
        ctx_avg.astype(np.float32),
        context_json,
        db_path=db_path,
    )
    return len(response_keys)


def generate_prompted_response_embeddings(
    model: SentenceTransformer,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    prompt: str = "Represent this trivia answer for finding topically related answers: ",
    batch_size: int = 128,
) -> int:
    """Encode responses with a prompt prefix, L2-normalize, and save.

    Returns the number of embeddings saved.
    """
    responses = fetch_response_texts(db_path=db_path)
    if not responses:
        return 0

    embeddings = model.encode(
        responses, prompt=prompt, batch_size=batch_size, show_progress_bar=True
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    save_embeddings(
        responses,
        embeddings.astype(np.float32),
        db_path=db_path,
        table="prompted_response_embeddings",
        text_column="response_text",
    )
    return len(responses)
