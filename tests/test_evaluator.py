"""Tests for jt3.evaluator — cycle evaluation and formatting."""

from __future__ import annotations

from jt3.evaluator import TextEvaluator, evaluate_cycles, format_cycle
from jt3.graph import Cycle, Edge


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_cycle() -> Cycle:
    return Cycle(
        nodes=["Lincoln", "Booth", "Theater", "Lincoln"],
        edges=[
            Edge(
                target="Booth",
                edge_type="category_bridge",
                metadata={"category": "Presidents"},
            ),
            Edge(target="Theater", edge_type="semantic", metadata={"similarity": 0.72}),
            Edge(
                target="Lincoln",
                edge_type="shared_entity",
                metadata={"entity": "Ford's Theatre"},
            ),
        ],
        score=0.85,
    )


# ── format_cycle ────────────────────────────────────────────────────────


class TestFormatCycle:
    def test_category_bridge_hop(self):
        cycle = _make_cycle()
        text = format_cycle(cycle)
        assert "Presidents" in text
        assert "Lincoln" in text

    def test_semantic_hop(self):
        cycle = _make_cycle()
        text = format_cycle(cycle)
        assert "0.72" in text

    def test_shared_entity_hop(self):
        cycle = _make_cycle()
        text = format_cycle(cycle)
        assert "Ford's Theatre" in text

    def test_score_shown(self):
        cycle = _make_cycle()
        text = format_cycle(cycle)
        assert "0.850" in text


# ── TextEvaluator ───────────────────────────────────────────────────────


class TestTextEvaluator:
    def test_returns_formatted(self):
        cycle = _make_cycle()
        evaluator = TextEvaluator()
        result = evaluator(cycle)
        assert result == format_cycle(cycle)


# ── evaluate_cycles ─────────────────────────────────────────────────────


class TestEvaluateCycles:
    def test_filters_empty_thesis(self):
        cycles = [_make_cycle(), _make_cycle()]
        # Evaluator that rejects everything
        result = evaluate_cycles(cycles, evaluator=lambda c: "")
        assert len(result) == 0

    def test_sets_thesis(self):
        cycle = _make_cycle()
        result = evaluate_cycles([cycle], evaluator=lambda c: "A great insight")
        assert len(result) == 1
        assert result[0].thesis == "A great insight"

    def test_mixed_accept_reject(self):
        c1 = _make_cycle()
        c2 = Cycle(
            nodes=["A", "B", "A"],
            edges=[
                Edge(target="B", edge_type="semantic"),
                Edge(target="A", edge_type="semantic"),
            ],
            score=0.1,
        )
        call_count = 0

        def alternating_evaluator(c: Cycle) -> str:
            nonlocal call_count
            call_count += 1
            return "thesis" if call_count % 2 == 1 else ""

        result = evaluate_cycles([c1, c2], evaluator=alternating_evaluator)
        assert len(result) == 1
