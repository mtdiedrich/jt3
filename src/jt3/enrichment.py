"""Enrichment pipeline — builds answer nodes and graph edges from the clue database."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

from .graph import (
    SIMILARITY_NEIGHBORS,
    SIMILARITY_THRESHOLD,
    AnswerGraph,
    Edge,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data"


# ── Answer nodes ────────────────────────────────────────────────────────


def build_answer_nodes(con: duckdb.DuckDBPyConnection) -> None:
    """Create a table of unique answers with their category contexts."""
    con.execute("DROP TABLE IF EXISTS answer_nodes")
    con.execute("""
        CREATE TABLE answer_nodes AS
        SELECT
            cl.correct_response,
            array_agg(DISTINCT cat.name)               AS categories,
            count(DISTINCT cat.name)                    AS category_span,
            count(*)                                    AS clue_count,
            array_agg(cat.name || ': ' || cl.text)      AS contextualized_clues
        FROM clues cl
        JOIN categories cat USING (game_id, round_index, category_index)
        WHERE cl.correct_response IS NOT NULL
          AND cl.correct_response != ''
        GROUP BY cl.correct_response
    """)


# ── Category bridge edges ──────────────────────────────────────────────


def build_category_bridges(con: duckdb.DuckDBPyConnection) -> dict[str, list[Edge]]:
    """Link answers that appeared in the same Jeopardy! category."""
    rows = con.execute("""
        SELECT
            a.correct_response AS from_answer,
            b.correct_response AS to_answer,
            cat.name            AS shared_category
        FROM clues a
        JOIN clues b USING (game_id, round_index, category_index)
        JOIN categories cat USING (game_id, round_index, category_index)
        WHERE a.correct_response != b.correct_response
          AND a.correct_response IS NOT NULL
          AND b.correct_response IS NOT NULL
          AND a.correct_response != ''
          AND b.correct_response != ''
        GROUP BY a.correct_response, b.correct_response, cat.name
    """).fetchall()

    graph: dict[str, list[Edge]] = defaultdict(list)
    for from_a, to_a, cat in rows:
        graph[from_a].append(
            Edge(
                target=to_a,
                edge_type="category_bridge",
                metadata={"category": cat},
            )
        )
    return graph


# ── Semantic similarity edges ──────────────────────────────────────────


def build_embeddings(
    con: duckdb.DuckDBPyConnection,
    cache_path: str | Path | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[str], np.ndarray]:
    """Embed each answer's contextualized clues and cache to disk.

    Requires ``sentence-transformers`` (optional dependency).
    """
    if cache_path is None:
        cache_path = _DEFAULT_CACHE_DIR / "embeddings.npz"
    cache_path = Path(cache_path)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["answers"].tolist(), data["vectors"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    rows = con.execute("""
        SELECT correct_response, contextualized_clues
        FROM answer_nodes
        ORDER BY correct_response
    """).fetchall()

    answers: list[str] = []
    texts: list[str] = []
    for answer, clues in rows:
        answers.append(answer)
        combined = " | ".join(clues[:10])
        texts.append(f"{answer}: {combined}")

    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, answers=np.array(answers, dtype=object), vectors=vectors)
    return answers, vectors


def build_semantic_edges(
    answers: list[str],
    vectors: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
    k: int = SIMILARITY_NEIGHBORS,
) -> dict[str, list[Edge]]:
    """Brute-force cosine nearest neighbors on normalized vectors."""
    graph: dict[str, list[Edge]] = defaultdict(list)
    sims = vectors @ vectors.T
    effective_k = min(k, len(answers) - 1)

    for i, answer in enumerate(answers):
        row = sims[i].copy()
        row[i] = -1  # exclude self
        top_k_idx = np.argpartition(row, -effective_k)[-effective_k:]
        for j in top_k_idx:
            if row[j] >= threshold:
                graph[answer].append(
                    Edge(
                        target=answers[j],
                        edge_type="semantic",
                        metadata={"similarity": float(row[j])},
                    )
                )
    return graph


# ── Shared entity edges (NER) ──────────────────────────────────────────


def build_entity_edges(
    con: duckdb.DuckDBPyConnection,
    cache_path: str | Path | None = None,
    max_answers_per_entity: int = 50,
) -> dict[str, list[Edge]]:
    """Link answers whose clues mention the same named entity.

    Requires ``spacy`` with the ``en_core_web_sm`` model (optional dependency).
    """
    if cache_path is None:
        cache_path = _DEFAULT_CACHE_DIR / "entities.json"
    cache_path = Path(cache_path)

    if cache_path.exists():
        with open(cache_path) as f:
            entity_index: dict[str, list[str]] = json.load(f)
    else:
        import spacy

        nlp = spacy.load("en_core_web_sm")

        rows = con.execute("""
            SELECT correct_response, contextualized_clues
            FROM answer_nodes
        """).fetchall()

        entity_to_answers: dict[str, set[str]] = defaultdict(set)
        for answer, clues in rows:
            text = " ".join(clues[:20])
            doc = nlp(text[:10000])
            for ent in doc.ents:
                normalized = ent.text.strip().lower()
                if normalized == answer.lower():
                    continue
                if ent.label_ in (
                    "PERSON",
                    "ORG",
                    "GPE",
                    "EVENT",
                    "WORK_OF_ART",
                    "DATE",
                ):
                    entity_to_answers[normalized].add(answer)

        entity_index = {k: list(v) for k, v in entity_to_answers.items() if len(v) >= 2}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(entity_index, f)

    graph: dict[str, list[Edge]] = defaultdict(list)
    for entity, answer_list in entity_index.items():
        if len(answer_list) > max_answers_per_entity:
            continue
        for i, a in enumerate(answer_list):
            for b in answer_list[i + 1 :]:
                graph[a].append(
                    Edge(
                        target=b,
                        edge_type="shared_entity",
                        metadata={"entity": entity},
                    )
                )
                graph[b].append(
                    Edge(
                        target=a,
                        edge_type="shared_entity",
                        metadata={"entity": entity},
                    )
                )
    return graph


# ── Orchestrator ────────────────────────────────────────────────────────


def build_graph(
    con: duckdb.DuckDBPyConnection,
    cache_dir: str | Path | None = None,
    embed_model: str = "all-MiniLM-L6-v2",
) -> AnswerGraph:
    """Build a fully assembled answer graph from the clue database.

    Calls all enrichment steps: answer nodes, category bridges,
    embeddings + semantic edges, and entity edges.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "answer_nodes" not in existing:
        build_answer_nodes(con)

    cat_edges = build_category_bridges(con)
    answers, vectors = build_embeddings(
        con, cache_path=cache_dir / "embeddings.npz", model_name=embed_model
    )
    sem_edges = build_semantic_edges(answers, vectors)
    ent_edges = build_entity_edges(con, cache_path=cache_dir / "entities.json")

    graph = AnswerGraph()
    graph.add_edges(cat_edges)
    graph.add_edges(sem_edges)
    graph.add_edges(ent_edges)
    graph.set_embeddings(answers, vectors)
    return graph


# ── Seed selection ──────────────────────────────────────────────────────


def get_seed_answers(
    con: duckdb.DuckDBPyConnection,
    min_span: int = 5,
    limit: int = 20,
) -> list[tuple[str, int, list[str]]]:
    """Return top answers by category span (most diverse first)."""
    rows = con.execute(
        """
        SELECT correct_response, category_span, categories
        FROM answer_nodes
        WHERE category_span >= ?
        ORDER BY category_span DESC
        LIMIT ?
        """,
        [min_span, limit],
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]
