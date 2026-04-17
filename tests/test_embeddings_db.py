"""Tests for jt3.embeddings.db — embedding storage operations."""

from pathlib import Path

import numpy as np

from jt3.db import get_connection
from jt3.embeddings.db import get_embedding, save_embeddings


# ---------------------------------------------------------------------------
# save_embeddings / get_embedding
# ---------------------------------------------------------------------------


def test_save_and_get_embedding_round_trip(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    texts = ["This element has atomic number 1", "The powerhouse of the cell"]
    embeddings = np.random.default_rng(42).standard_normal((2, 384)).astype(np.float32)

    save_embeddings(texts, embeddings, db_path=db_path)

    result = get_embedding("This element has atomic number 1", db_path=db_path)
    assert result is not None
    np.testing.assert_allclose(result, embeddings[0], atol=1e-6)


def test_get_embedding_not_found(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    result = get_embedding("nonexistent text", db_path=db_path)
    assert result is None


def test_save_embeddings_upserts(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    rng = np.random.default_rng(42)
    texts = ["same text"]
    emb_v1 = rng.standard_normal((1, 384)).astype(np.float32)
    emb_v2 = rng.standard_normal((1, 384)).astype(np.float32)

    save_embeddings(texts, emb_v1, db_path=db_path)
    save_embeddings(texts, emb_v2, db_path=db_path)

    result = get_embedding("same text", db_path=db_path)
    assert result is not None
    np.testing.assert_allclose(result, emb_v2[0], atol=1e-6)


# ---------------------------------------------------------------------------
# save_embeddings / get_embedding — custom table and text_column
# ---------------------------------------------------------------------------


def test_save_embeddings_custom_table(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    texts = ["answer one", "answer two"]
    embeddings = np.random.default_rng(42).standard_normal((2, 16)).astype(np.float32)

    save_embeddings(
        texts,
        embeddings,
        db_path=db_path,
        table="response_embeddings",
        text_column="response_text",
    )

    result = get_embedding(
        "answer one",
        db_path=db_path,
        table="response_embeddings",
        text_column="response_text",
    )
    assert result is not None
    np.testing.assert_allclose(result, embeddings[0], atol=1e-6)


def test_save_embeddings_custom_text_column(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    texts = ["foo"]
    embeddings = np.random.default_rng(42).standard_normal((1, 32)).astype(np.float32)

    save_embeddings(
        texts,
        embeddings,
        db_path=db_path,
        table="my_table",
        text_column="my_text",
    )

    result = get_embedding(
        "foo", db_path=db_path, table="my_table", text_column="my_text"
    )
    assert result is not None
    np.testing.assert_allclose(result, embeddings[0], atol=1e-6)
