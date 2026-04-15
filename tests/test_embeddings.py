from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from jt3.db import get_connection, save_episode
from jt3.embeddings import (
    DEFAULT_MODEL,
    embed,
    embed_batch,
    embed_clues,
    nearest_to_centroid,
)
from jt3.models import Category, Clue, Contestant, Episode, Round

FAKE_DIM = 384


def _fake_encode(texts, **kwargs):
    """Return deterministic fake embeddings based on input length."""
    if isinstance(texts, str):
        texts = [texts]
    return np.random.default_rng(42).random((len(texts), FAKE_DIM)).astype(np.float32)


def _make_mock_model():
    mock = MagicMock()
    mock.encode.side_effect = _fake_encode
    return mock


def _make_episode(game_id: int = 9418) -> Episode:
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
                        clues=[
                            Clue(
                                clue_id="J_1_1",
                                order=1,
                                value=200,
                                is_daily_double=False,
                                text="This element has atomic number 1",
                                correct_response="Hydrogen",
                            ),
                            Clue(
                                clue_id="J_1_2",
                                order=2,
                                value=400,
                                is_daily_double=False,
                                text="The powerhouse of the cell",
                                correct_response=None,  # no answer
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@patch("jt3.embeddings._get_model")
def test_embed_returns_1d_array(mock_get_model):
    mock_get_model.return_value = _make_mock_model()
    result = embed("hello world")
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape == (FAKE_DIM,)


@patch("jt3.embeddings._get_model")
def test_embed_with_custom_model(mock_get_model):
    mock_get_model.return_value = _make_mock_model()
    embed("hello", model_name="custom/model")
    mock_get_model.assert_called_with("custom/model", device=None)


# ---------------------------------------------------------------------------
# embed_batch()
# ---------------------------------------------------------------------------


@patch("jt3.embeddings._get_model")
def test_embed_batch_returns_2d_array(mock_get_model):
    mock_get_model.return_value = _make_mock_model()
    texts = ["one", "two", "three"]
    result = embed_batch(texts)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (3, FAKE_DIM)


# ---------------------------------------------------------------------------
# embed_clues()
# ---------------------------------------------------------------------------


@patch("jt3.embeddings._get_model")
def test_embed_clues_populated_db(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)

    count = embed_clues(db_path=db_path)

    # 2 clues: both have text (2 embeddings), only first has correct_response (1 embedding)
    assert count == 3

    # Verify rows exist in DB
    con = get_connection(db_path)
    rows = con.execute("SELECT * FROM embeddings").fetchall()
    con.close()
    assert len(rows) == 3


@patch("jt3.embeddings._get_model")
def test_embed_clues_skips_null_responses(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)

    embed_clues(db_path=db_path)

    con = get_connection(db_path)
    response_rows = con.execute(
        "SELECT * FROM embeddings WHERE field = 'correct_response'"
    ).fetchall()
    con.close()
    # Only J_1_1 has a correct_response
    assert len(response_rows) == 1


@patch("jt3.embeddings._get_model")
def test_embed_clues_empty_db(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    count = embed_clues(db_path=db_path)
    assert count == 0


@patch("jt3.embeddings._get_model")
def test_embed_clues_replaces_on_rerun(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)

    count1 = embed_clues(db_path=db_path)
    count2 = embed_clues(db_path=db_path)

    # Second run re-embeds everything (full rewrite)
    assert count1 == 3
    assert count2 == 3

    con = get_connection(db_path)
    rows = con.execute("SELECT * FROM embeddings").fetchall()
    con.close()
    assert len(rows) == 3


@patch("jt3.embeddings._get_model")
def test_embed_clues_reembeds_all_episodes(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(game_id=1), db_path=db_path)

    count1 = embed_clues(db_path=db_path)
    assert count1 == 3

    # Add a second episode and re-run — all clues re-embedded (full rewrite)
    save_episode(_make_episode(game_id=2), db_path=db_path)
    count2 = embed_clues(db_path=db_path)
    assert count2 == 6

    con = get_connection(db_path)
    rows = con.execute("SELECT * FROM embeddings").fetchall()
    con.close()
    assert len(rows) == 6


@patch("jt3.embeddings._get_model")
def test_embed_clues_stores_source_text(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)

    embed_clues(db_path=db_path)

    con = get_connection(db_path)
    row = con.execute(
        "SELECT source_text FROM embeddings WHERE clue_id = 'J_1_1' AND field = 'text'"
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] == "This element has atomic number 1"


@patch("jt3.embeddings._get_model")
def test_embed_clues_stores_model_name(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)

    embed_clues(db_path=db_path)

    con = get_connection(db_path)
    row = con.execute("SELECT DISTINCT model_name FROM embeddings").fetchone()
    con.close()
    assert row is not None
    assert row[0] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# nearest_to_centroid()
# ---------------------------------------------------------------------------


@patch("jt3.embeddings._get_model")
def test_nearest_to_centroid_returns_polars_df(mock_get_model, tmp_path: Path):
    import polars as pl

    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)
    embed_clues(db_path=db_path)

    result = nearest_to_centroid(["Hydrogen"], db_path=db_path)

    assert isinstance(result, pl.DataFrame)
    assert "answer_text" in result.columns
    assert "similarity" in result.columns
    assert len(result) >= 1


@patch("jt3.embeddings._get_model")
def test_nearest_to_centroid_respects_n(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)
    embed_clues(db_path=db_path)

    result = nearest_to_centroid(["Hydrogen"], n=1, db_path=db_path)

    assert len(result) == 1


def test_nearest_to_centroid_empty_list_raises(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        nearest_to_centroid([], db_path=db_path)


@patch("jt3.embeddings._get_model")
def test_nearest_to_centroid_exclude_radius(mock_get_model, tmp_path: Path):
    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)
    embed_clues(db_path=db_path)

    # With exclude_radius=0 everything is excluded (all similarities >= 0)
    result = nearest_to_centroid(["Hydrogen"], exclude_radius=0.0, db_path=db_path)
    assert len(result) == 0

    # With exclude_radius=1.0 nothing is excluded (no similarity reaches exactly 1.0 for different texts)
    result = nearest_to_centroid(["Hydrogen"], exclude_radius=1.0, db_path=db_path)
    assert len(result) >= 1


@patch("jt3.embeddings._get_model")
def test_nearest_to_centroid_with_existing_connection(mock_get_model, tmp_path: Path):
    import polars as pl

    mock_get_model.return_value = _make_mock_model()
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)
    embed_clues(db_path=db_path)

    # Open a read-only connection (like a notebook would)
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = nearest_to_centroid(["Hydrogen"], con=con)

        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 1

        # Connection should still be usable (not closed by nearest_to_centroid)
        row = con.execute("SELECT count(*) FROM embeddings").fetchone()
        assert row[0] > 0
    finally:
        con.close()
