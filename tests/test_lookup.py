"""Tests for jt3.lookup — embedding lookup module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jt3.db import get_connection, save_embeddings
from jt3.lookup import lookup_embeddings


@pytest.fixture()
def populated_db(tmp_path: Path) -> Path:
    """Create a DB with known embeddings and return the path."""
    db_path = tmp_path / "test.duckdb"
    texts = ["This element has atomic number 1", "The powerhouse of the cell"]
    dim = 8
    embeddings = np.random.default_rng(42).standard_normal((len(texts), dim)).astype(
        np.float32
    )
    save_embeddings(texts, embeddings, db_path=db_path)
    return db_path


@pytest.fixture()
def empty_db(tmp_path: Path) -> Path:
    """Create an empty DB with schema and return the path."""
    db_path = tmp_path / "test.duckdb"
    con = get_connection(db_path)
    con.close()
    return db_path


def test_lookup_found(populated_db: Path):
    queries = ["This element has atomic number 1", "The powerhouse of the cell"]
    found, missing = lookup_embeddings(queries, db_path=populated_db)

    assert len(found) == 2
    assert len(missing) == 0
    for q in queries:
        assert q in found
        assert found[q].dtype == np.float32
        assert found[q].shape == (8,)


def test_lookup_missing(populated_db: Path):
    queries = ["nonexistent query", "another missing one"]
    found, missing = lookup_embeddings(queries, db_path=populated_db)

    assert len(found) == 0
    assert missing == queries


def test_lookup_partial(populated_db: Path):
    queries = ["This element has atomic number 1", "nonexistent query"]
    found, missing = lookup_embeddings(queries, db_path=populated_db)

    assert len(found) == 1
    assert "This element has atomic number 1" in found
    assert missing == ["nonexistent query"]


def test_lookup_empty_queries(populated_db: Path):
    found, missing = lookup_embeddings([], db_path=populated_db)

    assert found == {}
    assert missing == []
