from datetime import date
from pathlib import Path

import numpy as np
import pytest

from jt3.db import (
    delete_episode,
    ensure_schema,
    get_connection,
    get_embedding,
    list_episodes,
    load_episode,
    save_embeddings,
    save_episode,
)
import duckdb
from jt3.models import Category, Clue, Contestant, Episode, Round


def _make_episode(game_id: int = 9418, **overrides) -> Episode:
    """Build a minimal but complete Episode for testing."""
    defaults = {
        "game_id": game_id,
        "show_number": 9536,
        "air_date": date(2026, 4, 6),
        "contestants": [
            Contestant(
                name="Alice", description="teacher from Omaha, NE", player_id=101
            ),
            Contestant(
                name="Bob", description="engineer from Austin, TX", player_id=102
            ),
            Contestant(
                name="Carol", description="writer from Portland, OR", player_id=None
            ),
        ],
        "rounds": [
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
                                order=5,
                                value=400,
                                is_daily_double=False,
                                text="The powerhouse of the cell",
                                correct_response="Mitochondria",
                                answerer="Bob",
                            ),
                        ],
                    ),
                    Category(
                        name="HISTORY",
                        comments="All from the 20th century",
                        clues=[
                            Clue(
                                clue_id="J_2_1",
                                order=2,
                                value=1000,
                                is_daily_double=True,
                                text="Year WWII ended",
                                correct_response="1945",
                                answerer="Carol",
                            ),
                        ],
                    ),
                ],
            ),
            Round(
                name="Final Jeopardy!",
                categories=[
                    Category(
                        name="LITERATURE",
                        comments=None,
                        clues=[
                            Clue(
                                clue_id="FJ",
                                order=None,
                                value=0,
                                is_daily_double=False,
                                text="This author wrote 1984",
                                correct_response="George Orwell",
                                answerer=None,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    }
    defaults.update(overrides)
    return Episode(**defaults)


# ---------------------------------------------------------------------------
# get_connection / ensure_schema
# ---------------------------------------------------------------------------


def test_get_connection_creates_parent_dirs(tmp_path: Path):
    db_path = tmp_path / "nested" / "dirs" / "test.duckdb"
    con = get_connection(db_path)
    con.close()
    assert db_path.exists()


def test_ensure_schema_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    # Second call should not raise
    ensure_schema(con)
    con.close()


# ---------------------------------------------------------------------------
# save_episode + load_episode round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    assert loaded.game_id == original.game_id
    assert loaded.show_number == original.show_number
    assert loaded.air_date == original.air_date


def test_round_trip_contestants(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    assert len(loaded.contestants) == 3
    assert loaded.contestants[0].name == "Alice"
    assert loaded.contestants[0].description == "teacher from Omaha, NE"
    assert loaded.contestants[0].player_id == 101
    assert loaded.contestants[2].player_id is None


def test_round_trip_rounds(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    assert len(loaded.rounds) == 2
    assert loaded.rounds[0].name == "Jeopardy!"
    assert loaded.rounds[1].name == "Final Jeopardy!"


def test_round_trip_categories(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    j_round = loaded.rounds[0]
    assert len(j_round.categories) == 2
    assert j_round.categories[0].name == "SCIENCE"
    assert j_round.categories[0].comments is None
    assert j_round.categories[1].name == "HISTORY"
    assert j_round.categories[1].comments == "All from the 20th century"


def test_round_trip_clues(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    clues = loaded.rounds[0].categories[0].clues
    assert len(clues) == 2
    assert clues[0].clue_id == "J_1_1"
    assert clues[0].order == 1
    assert clues[0].value == 200
    assert clues[0].is_daily_double is False
    assert clues[0].text == "This element has atomic number 1"
    assert clues[0].correct_response == "Hydrogen"
    assert clues[0].answerer == "Alice"


def test_round_trip_daily_double(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    dd_clue = loaded.rounds[0].categories[1].clues[0]
    assert dd_clue.is_daily_double is True
    assert dd_clue.value == 1000


def test_round_trip_final_jeopardy(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    original = _make_episode()

    save_episode(original, db_path=db_path)
    loaded = load_episode(9418, db_path=db_path)

    assert loaded is not None
    fj_round = loaded.rounds[1]
    assert fj_round.name == "Final Jeopardy!"
    assert len(fj_round.categories) == 1
    fj_clue = fj_round.categories[0].clues[0]
    assert fj_clue.clue_id == "FJ"
    assert fj_clue.order is None
    assert fj_clue.value == 0
    assert fj_clue.answerer is None


# ---------------------------------------------------------------------------
# Upsert behavior
# ---------------------------------------------------------------------------


def test_save_episode_upserts(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    ep1 = _make_episode(show_number=100)
    ep2 = _make_episode(show_number=200)

    save_episode(ep1, db_path=db_path)
    save_episode(ep2, db_path=db_path)

    loaded = load_episode(9418, db_path=db_path)
    assert loaded is not None
    assert loaded.show_number == 200

    episodes = list_episodes(db_path=db_path)
    assert len(episodes) == 1


# ---------------------------------------------------------------------------
# load_episode — not found
# ---------------------------------------------------------------------------


def test_load_episode_not_found(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    # Ensure schema exists by saving something first, or just get_connection
    con = get_connection(db_path)
    con.close()

    result = load_episode(99999, db_path=db_path)
    assert result is None


# ---------------------------------------------------------------------------
# delete_episode
# ---------------------------------------------------------------------------


def test_delete_episode_existing(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    save_episode(_make_episode(), db_path=db_path)

    deleted = delete_episode(9418, db_path=db_path)
    assert deleted is True

    loaded = load_episode(9418, db_path=db_path)
    assert loaded is None


def test_delete_episode_not_found(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    deleted = delete_episode(99999, db_path=db_path)
    assert deleted is False


# ---------------------------------------------------------------------------
# list_episodes
# ---------------------------------------------------------------------------


def test_list_episodes_empty(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    result = list_episodes(db_path=db_path)
    assert result == []


def test_list_episodes_multiple(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    save_episode(
        _make_episode(game_id=1, show_number=10, air_date=date(2020, 1, 1)),
        db_path=db_path,
    )
    save_episode(
        _make_episode(game_id=2, show_number=20, air_date=date(2021, 6, 15)),
        db_path=db_path,
    )

    result = list_episodes(db_path=db_path)
    assert len(result) == 2
    assert result[0]["game_id"] == 1
    assert result[0]["show_number"] == 10
    assert result[0]["air_date"] == date(2020, 1, 1)
    assert result[1]["game_id"] == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_episode_no_rounds_or_contestants(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    ep = Episode(game_id=1, show_number=None, air_date=None)

    save_episode(ep, db_path=db_path)
    loaded = load_episode(1, db_path=db_path)

    assert loaded is not None
    assert loaded.contestants == []
    assert loaded.rounds == []
    assert loaded.show_number is None
    assert loaded.air_date is None


# ---------------------------------------------------------------------------
# save_embeddings / get_embedding  (embeddings schema)
# ---------------------------------------------------------------------------


def test_save_and_get_embedding_round_trip(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    texts = ["This element has atomic number 1", "The powerhouse of the cell"]
    embeddings = np.random.default_rng(42).standard_normal((2, 384)).astype(np.float32)

    save_embeddings(texts, embeddings, db_path=db_path, table="clues")

    result = get_embedding(
        "This element has atomic number 1", db_path=db_path, table="clues"
    )
    assert result is not None
    np.testing.assert_allclose(result, embeddings[0], atol=1e-6)


def test_get_embedding_not_found(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    # Save something so the table exists
    texts = ["something"]
    embeddings = np.random.default_rng(42).standard_normal((1, 16)).astype(np.float32)
    save_embeddings(texts, embeddings, db_path=db_path, table="clues")

    result = get_embedding("nonexistent text", db_path=db_path, table="clues")
    assert result is None


def test_save_embeddings_upserts(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    rng = np.random.default_rng(42)
    texts = ["same text"]
    emb_v1 = rng.standard_normal((1, 384)).astype(np.float32)
    emb_v2 = rng.standard_normal((1, 384)).astype(np.float32)

    save_embeddings(texts, emb_v1, db_path=db_path, table="clues")
    save_embeddings(texts, emb_v2, db_path=db_path, table="clues")

    result = get_embedding("same text", db_path=db_path, table="clues")
    assert result is not None
    np.testing.assert_allclose(result, emb_v2[0], atol=1e-6)


def test_save_embeddings_to_responses(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    texts = ["answer one", "answer two"]
    embeddings = np.random.default_rng(42).standard_normal((2, 16)).astype(np.float32)

    save_embeddings(texts, embeddings, db_path=db_path, table="responses")

    result = get_embedding("answer one", db_path=db_path, table="responses")
    assert result is not None
    np.testing.assert_allclose(result, embeddings[0], atol=1e-6)


def test_embeddings_schema_created(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()

    texts = ["foo"]
    embeddings = np.random.default_rng(42).standard_normal((1, 32)).astype(np.float32)
    save_embeddings(texts, embeddings, db_path=db_path, table="clues")

    con = duckdb.connect(str(db_path))
    schemas = [
        r[0]
        for r in con.execute(
            "SELECT schema_name FROM information_schema.schemata"
        ).fetchall()
    ]
    con.close()
    assert "embeddings" in schemas


def test_validate_identifier_rejects_injection(tmp_path: Path):
    from jt3.db import _validate_identifier

    with pytest.raises(ValueError):
        _validate_identifier("foo; DROP TABLE")


def test_validate_identifier_accepts_valid():
    from jt3.db import _validate_identifier

    # Should not raise
    _validate_identifier("clue_embeddings")
    _validate_identifier("response_text")
