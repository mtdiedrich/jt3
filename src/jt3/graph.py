"""Answer graph data structures and cycle search algorithm."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# ── Config defaults ─────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.55
SIMILARITY_NEIGHBORS = 20
CYCLE_MIN_LENGTH = 4
CYCLE_MAX_LENGTH = 8
CYCLE_CANDIDATES_PER_SEED = 200
TOP_CYCLES_PER_SEED = 10


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class Edge:
    target: str
    edge_type: str  # "category_bridge" | "semantic" | "shared_entity"
    metadata: dict = field(default_factory=dict)


@dataclass
class Cycle:
    nodes: list[str]
    edges: list[Edge]
    score: float = 0.0
    thesis: str = ""


class AnswerGraph:
    """Multi-edge-type graph over Jeopardy! answers."""

    def __init__(self) -> None:
        self.adj: dict[str, list[Edge]] = defaultdict(list)
        self.embeddings: dict[str, np.ndarray] = {}

    def add_edges(self, edges: dict[str, list[Edge]]) -> None:
        for node, edge_list in edges.items():
            self.adj[node].extend(edge_list)

    def set_embeddings(self, answers: list[str], vectors: np.ndarray) -> None:
        for a, v in zip(answers, vectors):
            self.embeddings[a] = v

    def embedding_distance(self, a: str, b: str) -> float:
        """1 - cosine similarity. Higher = further apart."""
        va, vb = self.embeddings.get(a), self.embeddings.get(b)
        if va is None or vb is None:
            return 0.5
        return 1.0 - float(np.dot(va, vb))

    def neighbors(self, node: str, edge_type: str | None = None) -> list[Edge]:
        edges = self.adj.get(node, [])
        if edge_type:
            return [e for e in edges if e.edge_type == edge_type]
        return edges


# ── Cycle search ────────────────────────────────────────────────────────


def find_cycles(
    graph: AnswerGraph,
    start: str,
    *,
    max_depth: int = CYCLE_MAX_LENGTH,
    min_depth: int = CYCLE_MIN_LENGTH,
    n_candidates: int = CYCLE_CANDIDATES_PER_SEED,
    top_k: int = TOP_CYCLES_PER_SEED,
) -> list[Cycle]:
    """Stochastic cycle search.

    1. Depart via a category bridge (mandatory pivot).
    2. Walk outward, preferring nodes far from ``start`` in embedding space.
    3. At each step, check if any neighbor connects back to ``start``
       through a *different* edge type than the departure.
    4. Collect candidates scored by total embedding drift × edge-type diversity.
    """
    bridges = graph.neighbors(start, edge_type="category_bridge")
    if not bridges:
        return []

    candidates: list[Cycle] = []

    for _ in range(n_candidates):
        bridge_edge = random.choice(bridges)
        path_nodes = [start, bridge_edge.target]
        path_edges: list[Edge] = [bridge_edge]

        for _depth in range(max_depth - 2):
            current = path_nodes[-1]
            neighbors = graph.neighbors(current)
            if not neighbors:
                break

            scored: list[tuple[float, Edge]] = []
            for edge in neighbors:
                if edge.target in path_nodes:
                    continue
                dist = graph.embedding_distance(start, edge.target)
                jitter = random.uniform(0, 0.2)
                scored.append((dist + jitter, edge))

            if not scored:
                break

            scored.sort(key=lambda x: -x[0])
            pick = scored[min(random.randint(0, 2), len(scored) - 1)]
            path_nodes.append(pick[1].target)
            path_edges.append(pick[1])

            if len(path_nodes) >= min_depth:
                current_tail = path_nodes[-1]
                return_edges = [
                    e
                    for e in graph.neighbors(current_tail)
                    if e.target == start and e.edge_type != bridge_edge.edge_type
                ]
                if return_edges:
                    return_edge = return_edges[0]
                    path_nodes.append(start)
                    path_edges.append(return_edge)

                    drift = float(
                        np.mean(
                            [
                                graph.embedding_distance(start, n)
                                for n in path_nodes[1:-1]
                            ]
                        )
                    )
                    edge_types_used = len({e.edge_type for e in path_edges})
                    score = drift * (1 + 0.2 * edge_types_used)

                    candidates.append(
                        Cycle(
                            nodes=list(path_nodes),
                            edges=list(path_edges),
                            score=score,
                        )
                    )
                    break

    # Deduplicate by node-set, keep highest-scored
    seen: dict[frozenset[str], Cycle] = {}
    for c in candidates:
        key = frozenset(c.nodes)
        if key not in seen or c.score > seen[key].score:
            seen[key] = c

    return sorted(seen.values(), key=lambda c: -c.score)[:top_k]
