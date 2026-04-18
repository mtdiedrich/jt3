"""Tests for jt3.embeddings.generator — embedding generation module."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jt3.db import get_connection, search_similar
from jt3.embeddings.db import save_embeddings
from jt3.scraping.db import ensure_schema, save_episode
from jt3.embeddings.generator import (
    EMBEDDING_TABLES,
    MODELS,
    compute_centroid,
    fetch_category_texts,
    fetch_clue_texts,
    fetch_complete_texts,
    fetch_response_contexts,
    fetch_response_texts,
    generate_category_embeddings,
    generate_clue_embeddings,
    generate_contextual_response_embeddings,
    generate_complete_embeddings,
    generate_prompted_response_embeddings,
    generate_response_embeddings,
    load_model,
    search_all_tables,
)
from jt3.models import Category, Clue, Contestant, Episode, Round


def _make_episode(game_id: int = 9418) -> Episode:
    """Build a minimal Episode with known clues/responses for testing."""
    return Episode(
        game_id=game_id,
        show_number=9536,
        air_date=date(2026, 4, 6),
        contestants=[
            Contestant(
                name="Alice", description="teacher from Omaha, NE", player_id=101
            ),
        ],
        rounds=[
            Round(
                name="Jeopardy!",
                categories=[
                    Category(
                        name="SCIENCE",
                        comments=None,
                        clues=[
                            Clue(
                                clue_id="J_1_1",
                                order=1,
                                value=200,
                                is_daily_double=False,
                                text="This element has atomic number 1",
                                correct_response="Hydrogen",
                                answerer="Alice",
                            ),
                            Clue(
                                clue_id="J_1_2",
                                order=2,
                                value=400,
                                is_daily_double=False,
                                text="The powerhouse of the cell",
                                correct_response="Mitochondria",
                                answerer="Alice",
                            ),
                        ],
                    ),
                    Category(
                        name="HISTORY",
                        comments="All from the 20th century",
                        clues=[
                            Clue(
                                clue_id="J_2_1",
                                order=3,
                                value=600,
                                is_daily_double=False,
                                text="Year WWII ended",
                                correct_response="1945",
                                answerer="Alice",
                            ),
                            Clue(
                                clue_id="J_2_2",
                                order=4,
                                value=800,
                                is_daily_double=False,
                                text="A clue with no response",
                                correct_response=None,
                                answerer=None,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


@pytest.fixture()
def populated_db(tmp_path: Path) -> Path:
    """Create a DB with one episode and return the path."""
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)
    return db_path


@pytest.fixture()
def empty_db(tmp_path: Path) -> Path:
    """Create an empty DB with schema and return the path."""
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    ensure_schema(con)
    con.close()
    return db_path


# ---------------------------------------------------------------------------
# MODELS config
# ---------------------------------------------------------------------------


def test_models_dict_has_entries():
    assert len(MODELS) > 0
    for key, value in MODELS.items():
        assert "model_name_or_path" in value


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


@patch("jt3.embeddings.generator.SentenceTransformer")
def test_load_model(mock_st_cls):
    mock_model = MagicMock()
    mock_st_cls.return_value = mock_model

    result = load_model("all_MiniLM_L6_v2")

    assert result is mock_model
    mock_st_cls.assert_called_once()
    call_kwargs = mock_st_cls.call_args
    assert "sentence-transformers/all-MiniLM-L6-v2" in str(call_kwargs)


@patch("jt3.embeddings.generator.SentenceTransformer")
def test_load_model_invalid_key(mock_st_cls):
    with pytest.raises(KeyError):
        load_model("nonexistent_model")


# ---------------------------------------------------------------------------
# fetch_clue_texts
# ---------------------------------------------------------------------------


def test_fetch_clue_texts(populated_db: Path):
    texts = fetch_clue_texts(db_path=populated_db)
    assert len(texts) == 4
    assert "This element has atomic number 1" in texts
    assert "The powerhouse of the cell" in texts
    assert "Year WWII ended" in texts
    assert "A clue with no response" in texts


def test_fetch_clue_texts_empty(empty_db: Path):
    texts = fetch_clue_texts(db_path=empty_db)
    assert texts == []


# ---------------------------------------------------------------------------
# fetch_response_texts
# ---------------------------------------------------------------------------


def test_fetch_response_texts(populated_db: Path):
    texts = fetch_response_texts(db_path=populated_db)
    assert len(texts) == 3
    assert "Hydrogen" in texts
    assert "Mitochondria" in texts
    assert "1945" in texts


def test_fetch_response_texts_excludes_null(populated_db: Path):
    """Clue J_2_2 has correct_response=None — should not appear."""
    texts = fetch_response_texts(db_path=populated_db)
    for t in texts:
        assert t is not None


# ---------------------------------------------------------------------------
# fetch_response_contexts
# ---------------------------------------------------------------------------


def test_fetch_response_contexts(populated_db: Path):
    contexts = fetch_response_contexts(db_path=populated_db)
    assert isinstance(contexts, dict)
    assert "Hydrogen" in contexts
    assert len(contexts["Hydrogen"]) == 1
    ctx = contexts["Hydrogen"][0]
    assert "SCIENCE" in ctx
    assert "This element has atomic number 1" in ctx
    assert "Hydrogen" in ctx


# ---------------------------------------------------------------------------
# generate_clue_embeddings
# ---------------------------------------------------------------------------


def test_generate_clue_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((4, dim)).astype(np.float32)
    )

    count = generate_clue_embeddings(mock_model, db_path=populated_db)

    assert count == 4
    mock_model.encode.assert_called_once()

    # Verify data in DB
    con = get_connection(populated_db)
    rows = con.execute("SELECT * FROM embeddings.clues").fetchall()
    con.close()
    assert len(rows) == 4


# ---------------------------------------------------------------------------
# generate_response_embeddings
# ---------------------------------------------------------------------------


def test_generate_response_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((3, dim)).astype(np.float32)
    )

    count = generate_response_embeddings(mock_model, db_path=populated_db)

    assert count == 3
    mock_model.encode.assert_called_once()

    con = get_connection(populated_db)
    rows = con.execute("SELECT * FROM embeddings.responses").fetchall()
    con.close()
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# generate_contextual_response_embeddings
# ---------------------------------------------------------------------------


def test_generate_contextual_response_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    # 3 context strings (one per clue with a response)
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((3, dim)).astype(np.float32)
    )

    count = generate_contextual_response_embeddings(mock_model, db_path=populated_db)

    assert count == 3
    mock_model.encode.assert_called_once()

    con = get_connection(populated_db)
    rows = con.execute(
        "SELECT response_text, context_texts, embedding "
        "FROM contextual_response_embeddings"
    ).fetchall()
    con.close()
    assert len(rows) == 3
    # Check that context_texts is valid JSON
    import json

    for row in rows:
        contexts = json.loads(row[1])
        assert isinstance(contexts, list)


# ---------------------------------------------------------------------------
# generate_prompted_response_embeddings
# ---------------------------------------------------------------------------


def test_generate_prompted_response_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((3, dim)).astype(np.float32)
    )

    prompt = "Represent this trivia answer: "
    count = generate_prompted_response_embeddings(
        mock_model, db_path=populated_db, prompt=prompt
    )

    assert count == 3
    mock_model.encode.assert_called_once()
    call_kwargs = mock_model.encode.call_args
    assert call_kwargs.kwargs.get("prompt") == prompt

    con = get_connection(populated_db)
    rows = con.execute("SELECT * FROM prompted_response_embeddings").fetchall()
    con.close()
    assert len(rows) == 3

    # Verify L2-normalized
    emb = np.array(rows[0][1])
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_centroid
# ---------------------------------------------------------------------------


def test_compute_centroid_single_vector():
    vec = [1.0, 2.0, 3.0]
    result = compute_centroid([vec])
    np.testing.assert_allclose(result, vec)


def test_compute_centroid_multiple_vectors():
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    result = compute_centroid(vecs)
    expected = [1 / 3, 1 / 3, 1 / 3]
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# fetch_category_texts
# ---------------------------------------------------------------------------


def test_fetch_category_texts(populated_db: Path):
    texts = fetch_category_texts(db_path=populated_db)
    assert len(texts) == 2
    assert "SCIENCE" in texts
    assert "HISTORY" in texts


def test_fetch_category_texts_empty(empty_db: Path):
    texts = fetch_category_texts(db_path=empty_db)
    assert texts == []


# ---------------------------------------------------------------------------
# fetch_complete_texts
# ---------------------------------------------------------------------------


def test_fetch_complete_texts(populated_db: Path):
    texts = fetch_complete_texts(db_path=populated_db)
    # 3 clues with non-null responses
    assert len(texts) == 3
    assert any("SCIENCE" in t and "Hydrogen" in t for t in texts)
    assert any("HISTORY" in t and "1945" in t for t in texts)
    # Each should follow "Category: Clue → Response" format
    for t in texts:
        assert "→" in t


def test_fetch_complete_texts_empty(empty_db: Path):
    texts = fetch_complete_texts(db_path=empty_db)
    assert texts == []


# ---------------------------------------------------------------------------
# generate_category_embeddings
# ---------------------------------------------------------------------------


def test_generate_category_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((2, dim)).astype(np.float32)
    )

    count = generate_category_embeddings(mock_model, db_path=populated_db)

    assert count == 2
    mock_model.encode.assert_called_once()

    con = get_connection(populated_db)
    rows = con.execute("SELECT * FROM embeddings.categories").fetchall()
    con.close()
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------


def test_search_similar_returns_ranked_results(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"

    rng = np.random.default_rng(42)
    base = rng.standard_normal(384).astype(np.float32)
    similar = base + rng.standard_normal(384).astype(np.float32) * 0.1
    dissimilar = rng.standard_normal(384).astype(np.float32)

    texts = ["exact match", "close match", "far away"]
    embeddings = np.stack([base, similar, dissimilar])

    save_embeddings(
        texts,
        embeddings,
        db_path=db_path,
        table="embeddings.responses",
        text_column="response_text",
    )

    results = search_similar(base.tolist(), n=10, db_path=db_path)

    assert len(results) == 3
    assert results[0][0] == "exact match"
    assert results[0][1] > 0.99
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_similar_respects_n(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"

    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(5)]
    embeddings = rng.standard_normal((5, 384)).astype(np.float32)

    save_embeddings(
        texts,
        embeddings,
        db_path=db_path,
        table="embeddings.responses",
        text_column="response_text",
    )

    query = rng.standard_normal(384).astype(np.float32)
    results = search_similar(query.tolist(), n=2, db_path=db_path)

    assert len(results) == 2


def test_search_similar_empty_table(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    get_connection(db_path).close()

    query = np.random.default_rng(42).standard_normal(384).astype(np.float32)
    results = search_similar(query.tolist(), db_path=db_path)

    assert results == []


# ---------------------------------------------------------------------------
# search_all_tables
# ---------------------------------------------------------------------------


def test_search_all_tables(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"

    rng = np.random.default_rng(42)
    dim = 16

    save_embeddings(
        ["clue1", "clue2"],
        rng.standard_normal((2, dim)).astype(np.float32),
        db_path=db_path,
        table="embeddings.clues",
        text_column="clue_text",
    )
    save_embeddings(
        ["resp1"],
        rng.standard_normal((1, dim)).astype(np.float32),
        db_path=db_path,
        table="embeddings.responses",
        text_column="response_text",
    )
    save_embeddings(
        ["cat1", "cat2", "cat3"],
        rng.standard_normal((3, dim)).astype(np.float32),
        db_path=db_path,
        table="embeddings.categories",
        text_column="category_name",
    )

    query = rng.standard_normal(dim).astype(np.float32).tolist()
    results = search_all_tables(query, n=5, db_path=db_path)

    assert len(results) == 3
    assert "embeddings.clues" in results
    assert "embeddings.responses" in results
    assert "embeddings.categories" in results
    assert len(results["embeddings.clues"]) == 2
    assert len(results["embeddings.responses"]) == 1
    assert len(results["embeddings.categories"]) == 3


def test_search_all_tables_empty_db(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    get_connection(db_path).close()

    query = np.random.default_rng(42).standard_normal(16).astype(np.float32).tolist()
    results = search_all_tables(query, db_path=db_path)

    assert len(results) == 3
    for table, _ in EMBEDDING_TABLES:
        assert results[table] == []


# ---------------------------------------------------------------------------
# generate_complete_embeddings
# ---------------------------------------------------------------------------


def test_generate_complete_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((3, dim)).astype(np.float32)
    )

    count = generate_complete_embeddings(mock_model, db_path=populated_db)

    assert count == 3
    mock_model.encode.assert_called_once()

    con = get_connection(populated_db)
    rows = con.execute("SELECT * FROM embeddings.complete").fetchall()
    con.close()
    assert len(rows) == 3
