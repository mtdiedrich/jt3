"""Tests for jt3.db -- shared database connection utilities."""

from pathlib import Path

import pytest

from jt3.db import _validate_identifier, get_connection


# ---------------------------------------------------------------------------
# get_connection
# ---------------------------------------------------------------------------


def test_get_connection_creates_parent_dirs(tmp_path: Path):
    db_path = tmp_path / "nested" / "dirs" / "test.duckdb"
    con = get_connection(db_path)
    con.close()
    assert db_path.exists()


# ---------------------------------------------------------------------------
# _validate_identifier
# ---------------------------------------------------------------------------


def test_validate_identifier_rejects_injection():
    with pytest.raises(ValueError):
        _validate_identifier("foo; DROP TABLE")


def test_validate_identifier_accepts_valid():
    # Should not raise
    _validate_identifier("clue_embeddings")
    _validate_identifier("response_text")
