"""Tests for jt3.enrichment — DB enrichment and edge building."""

from __future__ import annotations

import numpy as np
import pytest

import duckdb

from jt3.db import ensure_schema
from jt3.enrichment import (
    build_answer_nodes,
    build_category_bridges,
    build_semantic_edges,
    get_seed_answers,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _seed_db(con: duckdb.DuckDBPyConnection) -> None:
    """Insert a small dataset with known answers and categories."""
    ensure_schema(con)

    con.execute("INSERT INTO episodes VALUES (1, 100, '2024-01-01')")
    con.execute("INSERT INTO rounds VALUES (1, 0, 'Jeopardy!')")

    # Two categories in the same round
    con.execute("INSERT INTO categories VALUES (1, 0, 0, 'History', NULL)")
    con.execute("INSERT INTO categories VALUES (1, 0, 1, 'Science', NULL)")

    # Answer "Lincoln" in History
    con.execute("""
        INSERT INTO clues VALUES
        (1, 0, 0, 'J_0_0_0', 1, 200, false, 'This president was born in 1809', 'Lincoln', NULL)
    """)
    # Answer "Darwin" in Science
    con.execute("""
        INSERT INTO clues VALUES
        (1, 0, 1, 'J_0_1_0', 2, 400, false, 'This naturalist was born in 1809', 'Darwin', NULL)
    """)
    # Answer "Lincoln" in Science too (category bridge!)
    con.execute("""
        INSERT INTO clues VALUES
        (1, 0, 1, 'J_0_1_1', 3, 600, false, 'This president patented a device', 'Lincoln', NULL)
    """)
    # Null/empty answers should be excluded
    con.execute("""
        INSERT INTO clues VALUES
        (1, 0, 0, 'J_0_0_1', 4, 200, false, 'Unrevealed clue', NULL, NULL)
    """)
    con.execute("""
        INSERT INTO clues VALUES
        (1, 0, 0, 'J_0_0_2', 5, 200, false, 'Blank answer', '', NULL)
    """)


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with seed data."""
    c = duckdb.connect(":memory:")
    _seed_db(c)
    return c


# ── build_answer_nodes ──────────────────────────────────────────────────


class TestBuildAnswerNodes:
    def test_creates_table(self, con):
        build_answer_nodes(con)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert "answer_nodes" in tables

    def test_correct_data(self, con):
        build_answer_nodes(con)
        rows = con.execute(
            "SELECT correct_response, category_span FROM answer_nodes ORDER BY correct_response"
        ).fetchall()
        answers = {r[0]: r[1] for r in rows}
        assert "Lincoln" in answers
        assert "Darwin" in answers
        # Lincoln appears in both History and Science
        assert answers["Lincoln"] == 2
        # Darwin appears only in Science
        assert answers["Darwin"] == 1

    def test_skips_null_and_empty(self, con):
        build_answer_nodes(con)
        rows = con.execute("SELECT correct_response FROM answer_nodes").fetchall()
        answers = {r[0] for r in rows}
        assert "" not in answers
        assert None not in answers


# ── build_category_bridges ──────────────────────────────────────────────


class TestBuildCategoryBridges:
    def test_returns_edges(self, con):
        build_answer_nodes(con)
        edges = build_category_bridges(con)
        # Lincoln and Darwin share the "Science" category
        lincoln_targets = {e.target for e in edges.get("Lincoln", [])}
        assert "Darwin" in lincoln_targets

    def test_no_self_edges(self, con):
        build_answer_nodes(con)
        edges = build_category_bridges(con)
        for node, edge_list in edges.items():
            for e in edge_list:
                assert e.target != node

    def test_edge_metadata(self, con):
        build_answer_nodes(con)
        edges = build_category_bridges(con)
        darwin_to_lincoln = [
            e for e in edges.get("Darwin", []) if e.target == "Lincoln"
        ]
        assert len(darwin_to_lincoln) >= 1
        assert "category" in darwin_to_lincoln[0].metadata


# ── build_semantic_edges ────────────────────────────────────────────────


class TestBuildSemanticEdges:
    def test_returns_edges_above_threshold(self):
        answers = ["A", "B", "C"]
        # A and B are similar, C is orthogonal
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Normalize
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        edges = build_semantic_edges(answers, vectors, threshold=0.8, k=5)
        a_targets = {e.target for e in edges.get("A", [])}
        assert "B" in a_targets
        # C should be too far from A
        assert "C" not in a_targets

    def test_no_self_edges(self):
        answers = ["A", "B"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        edges = build_semantic_edges(answers, vectors, threshold=0.0, k=5)
        for node, edge_list in edges.items():
            for e in edge_list:
                assert e.target != node


# ── get_seed_answers ────────────────────────────────────────────────────


class TestGetSeedAnswers:
    def test_returns_sorted(self, con):
        build_answer_nodes(con)
        seeds = get_seed_answers(con, min_span=1, limit=10)
        assert len(seeds) >= 1
        # First should be Lincoln (span 2)
        assert seeds[0][0] == "Lincoln"
        assert seeds[0][1] == 2

    def test_min_span_filter(self, con):
        build_answer_nodes(con)
        seeds = get_seed_answers(con, min_span=3, limit=10)
        assert len(seeds) == 0
