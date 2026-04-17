"""Tests for jt3.embeddings — embedding generation module."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jt3.db import get_connection, save_episode
from jt3.embeddings import (
    MODELS,
    fetch_category_texts,
    fetch_clue_texts,
    fetch_full_context_texts,
    fetch_response_texts,
    generate_category_embeddings,
    generate_clue_embeddings,
    generate_full_context_embeddings,
    generate_response_embeddings,
    load_model,
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


@patch("jt3.embeddings.SentenceTransformer")
def test_load_model(mock_st_cls):
    mock_model = MagicMock()
    mock_st_cls.return_value = mock_model

    result = load_model("all_MiniLM_L6_v2")

    assert result is mock_model
    mock_st_cls.assert_called_once()
    call_kwargs = mock_st_cls.call_args
    assert "sentence-transformers/all-MiniLM-L6-v2" in str(call_kwargs)


@patch("jt3.embeddings.SentenceTransformer")
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
# fetch_full_context_texts
# ---------------------------------------------------------------------------


def test_fetch_full_context_texts(populated_db: Path):
    texts = fetch_full_context_texts(db_path=populated_db)
    assert len(texts) == 3  # 3 clues with non-null responses
    # Each string has form "Category: Clue → Response"
    assert any("SCIENCE" in t and "Hydrogen" in t for t in texts)
    assert any("HISTORY" in t and "1945" in t for t in texts)


def test_fetch_full_context_texts_empty(empty_db: Path):
    texts = fetch_full_context_texts(db_path=empty_db)
    assert texts == []


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

    # Verify data in DB under embeddings schema
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
# generate_full_context_embeddings
# ---------------------------------------------------------------------------


def test_generate_full_context_embeddings(populated_db: Path):
    dim = 8
    mock_model = MagicMock()
    # 3 context strings (one per clue with a non-null response)
    mock_model.encode.return_value = (
        np.random.default_rng(42).standard_normal((3, dim)).astype(np.float32)
    )

    count = generate_full_context_embeddings(mock_model, db_path=populated_db)

    assert count == 3
    mock_model.encode.assert_called_once()

    con = get_connection(populated_db)
    rows = con.execute("SELECT * FROM embeddings.full_context").fetchall()
    con.close()
    assert len(rows) == 3
