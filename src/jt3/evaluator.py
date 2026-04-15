"""Provider-agnostic cycle evaluation."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .graph import Cycle


# ── Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class CycleEvaluator(Protocol):
    def __call__(self, cycle: Cycle) -> str:
        """Return a thesis sentence, or empty string to reject."""
        ...


# ── Formatting ──────────────────────────────────────────────────────────


def format_cycle(cycle: Cycle) -> str:
    """Format a cycle as human-readable text with hop descriptions."""
    lines = [f"Cycle (score: {cycle.score:.3f}):"]
    for i, edge in enumerate(cycle.edges):
        from_node = cycle.nodes[i]
        to_node = cycle.nodes[i + 1]
        match edge.edge_type:
            case "category_bridge":
                reason = f'both in category "{edge.metadata["category"]}"'
            case "semantic":
                sim = edge.metadata.get("similarity", "?")
                reason = (
                    f"semantically related ({sim:.2f})"
                    if isinstance(sim, float)
                    else f"semantically related ({sim})"
                )
            case "shared_entity":
                reason = f'shared entity "{edge.metadata["entity"]}"'
            case _:
                reason = "related"
        lines.append(f"  {from_node} → {to_node} ({reason})")
    return "\n".join(lines)


# ── Implementations ────────────────────────────────────────────────────


class TextEvaluator:
    """Default evaluator — returns formatted text for manual review."""

    def __call__(self, cycle: Cycle) -> str:
        return format_cycle(cycle)


class AnthropicEvaluator:
    """Evaluate cycles using the Anthropic API.

    Requires the ``anthropic`` package (optional dependency).
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        from anthropic import Anthropic

        self._client = Anthropic()
        self._model = model

    def __call__(self, cycle: Cycle) -> str:
        hop_descriptions: list[str] = []
        for i, edge in enumerate(cycle.edges):
            from_node = cycle.nodes[i]
            to_node = cycle.nodes[i + 1]
            match edge.edge_type:
                case "category_bridge":
                    reason = f'both appeared in the Jeopardy! category "{edge.metadata["category"]}"'
                case "semantic":
                    sim = edge.metadata.get("similarity", "?")
                    reason = (
                        f"semantically related (similarity: {sim:.2f})"
                        if isinstance(sim, float)
                        else f"semantically related (similarity: {sim})"
                    )
                case "shared_entity":
                    reason = f'both connected to "{edge.metadata["entity"]}"'
                case _:
                    reason = "related"
            hop_descriptions.append(f"  {from_node} → {to_node} ({reason})")

        hops_text = "\n".join(hop_descriptions)
        start = cycle.nodes[0]

        message = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Here is a cycle of connections between Jeopardy! answers, "
                        f'starting and ending at "{start}":\n\n'
                        f"{hops_text}\n\n"
                        f'In a single sentence, what does this journey reveal about "{start}" '
                        f"that wouldn't be obvious without following the chain?\n\n"
                        f"If the connections feel forced or the cycle doesn't cohere into "
                        f"an insight, respond with just: REJECT"
                    ),
                }
            ],
        )
        thesis = message.content[0].text.strip()
        if thesis.upper().startswith("REJECT"):
            return ""
        return thesis


# ── Runner ──────────────────────────────────────────────────────────────


def evaluate_cycles(
    cycles: list[Cycle],
    evaluator: CycleEvaluator | Callable[[Cycle], str] | None = None,
) -> list[Cycle]:
    """Run evaluator on each cycle. Sets thesis, filters rejects (empty string)."""
    if evaluator is None:
        evaluator = TextEvaluator()

    results: list[Cycle] = []
    for cycle in cycles:
        thesis = evaluator(cycle)
        if thesis:
            cycle.thesis = thesis
            results.append(cycle)
    return results
