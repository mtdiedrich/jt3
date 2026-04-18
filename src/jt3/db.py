"""DuckDB-backed storage for Episode data."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from .models import Category, Clue, Contestant, Episode, Round

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Validate a SQL identifier to prevent injection. Returns *name* if valid."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def _validate_qualified_name(name: str) -> str:
    """Validate an optionally schema-qualified SQL name (e.g. ``schema.table``)."""
    parts = name.split(".")
    if len(parts) not in (1, 2) or not all(_IDENTIFIER_RE.match(p) for p in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "jt3.duckdb"

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS episodes (
    game_id     INTEGER PRIMARY KEY,
    show_number INTEGER,
    air_date    DATE
);

CREATE TABLE IF NOT EXISTS contestants (
    game_id     INTEGER NOT NULL,
    name        VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    player_id   INTEGER
);

CREATE TABLE IF NOT EXISTS rounds (
    game_id     INTEGER NOT NULL,
    round_index INTEGER NOT NULL,
    name        VARCHAR NOT NULL,
    PRIMARY KEY (game_id, round_index)
);

CREATE TABLE IF NOT EXISTS categories (
    game_id        INTEGER NOT NULL,
    round_index    INTEGER NOT NULL,
    category_index INTEGER NOT NULL,
    name           VARCHAR NOT NULL,
    comments       VARCHAR,
    PRIMARY KEY (game_id, round_index, category_index)
);

CREATE TABLE IF NOT EXISTS clues (
    game_id        INTEGER NOT NULL,
    round_index    INTEGER NOT NULL,
    category_index INTEGER NOT NULL,
    clue_id        VARCHAR NOT NULL,
    clue_order     INTEGER,
    value          INTEGER NOT NULL,
    is_daily_double BOOLEAN NOT NULL,
    text           VARCHAR NOT NULL,
    correct_response VARCHAR,
    answerer       VARCHAR,
    PRIMARY KEY (game_id, round_index, category_index, clue_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    clue_text TEXT PRIMARY KEY,
    embedding FLOAT[384] NOT NULL
);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they don't already exist."""
    con.execute(_SCHEMA_SQL)


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating parent dirs and schema as needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    return con


def save_episode(episode: Episode, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Save an Episode to the database, replacing any existing data for the same game_id."""
    con = get_connection(db_path)
    try:
        con.begin()
        gid = episode.game_id

        # Delete existing data (child tables first).
        con.execute("DELETE FROM clues WHERE game_id = ?", [gid])
        con.execute("DELETE FROM categories WHERE game_id = ?", [gid])
        con.execute("DELETE FROM rounds WHERE game_id = ?", [gid])
        con.execute("DELETE FROM contestants WHERE game_id = ?", [gid])
        con.execute("DELETE FROM episodes WHERE game_id = ?", [gid])

        # Insert episode.
        con.execute(
            "INSERT INTO episodes (game_id, show_number, air_date) VALUES (?, ?, ?)",
            [gid, episode.show_number, episode.air_date],
        )

        # Insert contestants.
        for c in episode.contestants:
            con.execute(
                "INSERT INTO contestants (game_id, name, description, player_id) VALUES (?, ?, ?, ?)",
                [gid, c.name, c.description, c.player_id],
            )

        # Insert rounds, categories, clues.
        for ri, rnd in enumerate(episode.rounds):
            con.execute(
                "INSERT INTO rounds (game_id, round_index, name) VALUES (?, ?, ?)",
                [gid, ri, rnd.name],
            )
            for ci, cat in enumerate(rnd.categories):
                con.execute(
                    "INSERT INTO categories (game_id, round_index, category_index, name, comments) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [gid, ri, ci, cat.name, cat.comments],
                )
                for clue in cat.clues:
                    con.execute(
                        "INSERT INTO clues "
                        "(game_id, round_index, category_index, clue_id, clue_order, "
                        "value, is_daily_double, text, correct_response, answerer) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            gid,
                            ri,
                            ci,
                            clue.clue_id,
                            clue.order,
                            clue.value,
                            clue.is_daily_double,
                            clue.text,
                            clue.correct_response,
                            clue.answerer,
                        ],
                    )

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def load_episode(game_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> Episode | None:
    """Load a full Episode from the database. Returns None if not found."""
    con = get_connection(db_path)
    try:
        row = con.execute(
            "SELECT game_id, show_number, air_date FROM episodes WHERE game_id = ?",
            [game_id],
        ).fetchone()
        if row is None:
            return None

        gid, show_number, air_date = row

        # Contestants (preserve insertion order).
        contestants = [
            Contestant(name=r[0], description=r[1], player_id=r[2])
            for r in con.execute(
                "SELECT name, description, player_id FROM contestants "
                "WHERE game_id = ? ORDER BY rowid",
                [gid],
            ).fetchall()
        ]

        # Rounds → Categories → Clues.
        rounds: list[Round] = []
        round_rows = con.execute(
            "SELECT round_index, name FROM rounds WHERE game_id = ? ORDER BY round_index",
            [gid],
        ).fetchall()

        for ri, round_name in round_rows:
            cat_rows = con.execute(
                "SELECT category_index, name, comments FROM categories "
                "WHERE game_id = ? AND round_index = ? ORDER BY category_index",
                [gid, ri],
            ).fetchall()

            categories: list[Category] = []
            for ci, cat_name, cat_comments in cat_rows:
                clue_rows = con.execute(
                    "SELECT clue_id, clue_order, value, is_daily_double, text, "
                    "correct_response, answerer FROM clues "
                    "WHERE game_id = ? AND round_index = ? AND category_index = ? "
                    "ORDER BY clue_id",
                    [gid, ri, ci],
                ).fetchall()

                clues = [
                    Clue(
                        clue_id=cr[0],
                        order=cr[1],
                        value=cr[2],
                        is_daily_double=cr[3],
                        text=cr[4],
                        correct_response=cr[5],
                        answerer=cr[6],
                    )
                    for cr in clue_rows
                ]
                categories.append(
                    Category(name=cat_name, comments=cat_comments, clues=clues)
                )

            rounds.append(Round(name=round_name, categories=categories))

        return Episode(
            game_id=gid,
            show_number=show_number,
            air_date=air_date,
            contestants=contestants,
            rounds=rounds,
        )
    finally:
        con.close()


def delete_episode(game_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    """Delete an episode and all child data. Returns True if it existed."""
    con = get_connection(db_path)
    try:
        row = con.execute(
            "SELECT 1 FROM episodes WHERE game_id = ?", [game_id]
        ).fetchone()
        if row is None:
            return False

        con.begin()
        con.execute("DELETE FROM clues WHERE game_id = ?", [game_id])
        con.execute("DELETE FROM categories WHERE game_id = ?", [game_id])
        con.execute("DELETE FROM rounds WHERE game_id = ?", [game_id])
        con.execute("DELETE FROM contestants WHERE game_id = ?", [game_id])
        con.execute("DELETE FROM episodes WHERE game_id = ?", [game_id])
        con.commit()
        return True
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def list_episodes(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return a list of dicts with game_id, show_number, air_date for all episodes."""
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT game_id, show_number, air_date FROM episodes ORDER BY game_id"
        ).fetchall()
        return [{"game_id": r[0], "show_number": r[1], "air_date": r[2]} for r in rows]
    finally:
        con.close()


def save_embeddings(
    texts: list[str],
    embeddings: np.ndarray,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    table: str = "embeddings",
    text_column: str = "clue_text",
) -> None:
    """Bulk-save text→embedding pairs, upserting on conflict."""
    tbl = _validate_qualified_name(table)
    col = _validate_identifier(text_column)
    dim = embeddings.shape[1]
    con = get_connection(db_path)
    try:
        # Ensure schema exists for qualified names
        parts = tbl.split(".")
        if len(parts) == 2:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {parts[0]}")
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        con.execute(
            f"CREATE TABLE {tbl} "
            f"({col} TEXT PRIMARY KEY, embedding FLOAT[{dim}] NOT NULL)"
        )
        df = pl.DataFrame(
            {col: texts, "embedding": embeddings.tolist()}
        ).unique(subset=[col], keep="first")
        con.execute(
            f"INSERT INTO {tbl} SELECT {col}, embedding FROM df"
        )
    finally:
        con.close()


def get_embedding(
    text: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    table: str = "embeddings",
    text_column: str = "clue_text",
) -> list[float] | None:
    """Return the embedding for *text*, or ``None`` if not found."""
    tbl = _validate_qualified_name(table)
    col = _validate_identifier(text_column)
    con = get_connection(db_path)
    try:
        row = con.execute(
            f"SELECT embedding FROM {tbl} WHERE {col} = ?", [text]
        ).fetchone()
        if row is None:
            return None
        return list(row[0])
    finally:
        con.close()


def search_similar(
    embedding: list[float],
    *,
    n: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
    table: str = "embeddings.responses",
    text_column: str = "response_text",
) -> list[tuple[str, float]]:
    """Return the *n* most similar texts by cosine similarity.

    Returns a list of ``(text, score)`` tuples sorted by descending
    similarity, or an empty list if the table does not exist.
    """
    tbl = _validate_qualified_name(table)
    col = _validate_identifier(text_column)
    parts = tbl.split(".")
    table_name = parts[-1]
    schema_name = parts[0] if len(parts) == 2 else "main"
    con = get_connection(db_path)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema_name, table_name],
        ).fetchone()[0]
        if not exists:
            return []
        rows = con.execute(
            f"SELECT {col}, list_cosine_similarity(embedding, ?::FLOAT[]) AS score "
            f"FROM {tbl} ORDER BY score DESC LIMIT ?",
            [embedding, n],
        ).fetchall()
        return [(row[0], row[1]) for row in rows]
    finally:
        con.close()


def save_contextual_embeddings(
    texts: list[str],
    embeddings: np.ndarray,
    context_texts: list[str],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Save contextual response embeddings with JSON context strings."""
    dim = embeddings.shape[1]
    con = get_connection(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS contextual_response_embeddings")
        con.execute(
            f"CREATE TABLE contextual_response_embeddings ("
            f"response_text TEXT PRIMARY KEY, "
            f"context_texts JSON NOT NULL, "
            f"embedding FLOAT[{dim}] NOT NULL)"
        )
        df = pl.DataFrame({
            "response_text": texts,
            "context_texts": context_texts,
            "embedding": embeddings.tolist(),
        })
        con.execute(
            "INSERT INTO contextual_response_embeddings "
            "SELECT response_text, context_texts, embedding FROM df"
        )
    finally:
        con.close()
