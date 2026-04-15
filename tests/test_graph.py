"""Tests for jt3.graph — data structures and cycle search."""

from __future__ import annotations

import numpy as np
import pytest

from jt3.graph import AnswerGraph, Cycle, Edge, find_cycles


# ── Edge & Cycle defaults ───────────────────────────────────────────────


class TestEdge:
    def test_defaults(self):
        e = Edge(target="Lincoln", edge_type="semantic")
        assert e.metadata == {}

    def test_with_metadata(self):
        e = Edge(
            target="Lincoln",
            edge_type="category_bridge",
            metadata={"category": "Presidents"},
        )
        assert e.metadata["category"] == "Presidents"


class TestCycle:
    def test_defaults(self):
        c = Cycle(nodes=["A", "B", "A"], edges=[])
        assert c.score == 0.0
        assert c.thesis == ""


# ── AnswerGraph ─────────────────────────────────────────────────────────


class TestAnswerGraph:
    def _make_graph(self) -> AnswerGraph:
        g = AnswerGraph()
        g.add_edges(
            {
                "A": [
                    Edge(
                        target="B",
                        edge_type="category_bridge",
                        metadata={"category": "History"},
                    ),
                    Edge(
                        target="C", edge_type="semantic", metadata={"similarity": 0.8}
                    ),
                ],
                "B": [
                    Edge(
                        target="A",
                        edge_type="shared_entity",
                        metadata={"entity": "1809"},
                    ),
                ],
            }
        )
        return g

    def test_add_edges(self):
        g = self._make_graph()
        assert len(g.adj["A"]) == 2
        assert len(g.adj["B"]) == 1

    def test_add_edges_merges(self):
        g = self._make_graph()
        g.add_edges({"A": [Edge(target="D", edge_type="semantic")]})
        assert len(g.adj["A"]) == 3

    def test_set_embeddings(self):
        g = AnswerGraph()
        answers = ["A", "B"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        g.set_embeddings(answers, vectors)
        assert "A" in g.embeddings
        assert "B" in g.embeddings

    def test_embedding_distance_identical(self):
        g = AnswerGraph()
        v = np.array([1.0, 0.0])
        g.set_embeddings(["A", "B"], np.array([v, v]))
        assert g.embedding_distance("A", "B") == pytest.approx(0.0)

    def test_embedding_distance_orthogonal(self):
        g = AnswerGraph()
        g.set_embeddings(
            ["A", "B"],
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        assert g.embedding_distance("A", "B") == pytest.approx(1.0)

    def test_embedding_distance_missing(self):
        g = AnswerGraph()
        assert g.embedding_distance("X", "Y") == pytest.approx(0.5)

    def test_neighbors_all(self):
        g = self._make_graph()
        assert len(g.neighbors("A")) == 2

    def test_neighbors_filtered(self):
        g = self._make_graph()
        bridges = g.neighbors("A", edge_type="category_bridge")
        assert len(bridges) == 1
        assert bridges[0].target == "B"

    def test_neighbors_empty(self):
        g = self._make_graph()
        assert g.neighbors("Z") == []


# ── Cycle search ────────────────────────────────────────────────────────


def _build_synthetic_graph() -> AnswerGraph:
    """
    Build a small graph where cycles are guaranteed to exist.

    Topology:
        A --(category_bridge)--> B
        B --(semantic)--> C
        C --(semantic)--> D
        D --(shared_entity)--> A   (return edge, different type from departure)

    This should yield at least one cycle: A → B → C → D → A
    """
    g = AnswerGraph()
    edges: dict[str, list[Edge]] = {
        "A": [
            Edge(
                target="B",
                edge_type="category_bridge",
                metadata={"category": "History"},
            ),
        ],
        "B": [
            Edge(target="C", edge_type="semantic", metadata={"similarity": 0.7}),
            Edge(
                target="A",
                edge_type="category_bridge",
                metadata={"category": "History"},
            ),
        ],
        "C": [
            Edge(target="D", edge_type="semantic", metadata={"similarity": 0.6}),
            Edge(target="B", edge_type="semantic", metadata={"similarity": 0.7}),
        ],
        "D": [
            Edge(target="A", edge_type="shared_entity", metadata={"entity": "1809"}),
            Edge(target="C", edge_type="semantic", metadata={"similarity": 0.6}),
        ],
    }
    g.add_edges(edges)

    # Place embeddings so that B, C, D are progressively farther from A.
    dim = 8
    rng = np.random.default_rng(42)
    va = rng.standard_normal(dim).astype(np.float32)
    va /= np.linalg.norm(va)
    vb = va + rng.standard_normal(dim).astype(np.float32) * 0.3
    vb /= np.linalg.norm(vb)
    vc = va + rng.standard_normal(dim).astype(np.float32) * 0.6
    vc /= np.linalg.norm(vc)
    vd = va + rng.standard_normal(dim).astype(np.float32) * 0.9
    vd /= np.linalg.norm(vd)

    g.set_embeddings(["A", "B", "C", "D"], np.array([va, vb, vc, vd]))
    return g


class TestFindCycles:
    def test_returns_cycles(self):
        g = _build_synthetic_graph()
        cycles = find_cycles(g, "A", n_candidates=500)
        assert len(cycles) >= 1
        for c in cycles:
            assert c.nodes[0] == "A"
            assert c.nodes[-1] == "A"
            assert c.score > 0

    def test_no_bridges_returns_empty(self):
        g = AnswerGraph()
        g.add_edges({"X": [Edge(target="Y", edge_type="semantic")]})
        assert find_cycles(g, "X") == []

    def test_dedup_by_node_set(self):
        g = _build_synthetic_graph()
        cycles = find_cycles(g, "A", n_candidates=500)
        node_sets = [frozenset(c.nodes) for c in cycles]
        assert len(node_sets) == len(set(node_sets))

    def test_respects_max_depth(self):
        g = _build_synthetic_graph()
        max_depth = 5
        cycles = find_cycles(g, "A", max_depth=max_depth, n_candidates=500)
        for c in cycles:
            # nodes includes start twice (first + last), so len <= max_depth + 1
            assert len(c.nodes) <= max_depth + 1
