# feat-answer-graph

## Goal

Add an answer graph module that builds a multi-edge-type graph from the Jeopardy! clue database and uses stochastic cycle search to find narratively interesting loops through idea-space, optionally scored by an LLM evaluator.

## Current behavior

N/A — new feature.

## Target behavior

Three new modules under `src/jt3/`:

### `graph.py` — Data structures and cycle search

- **`Edge`** dataclass: `target: str`, `edge_type: str` (one of `"category_bridge"`, `"semantic"`, `"shared_entity"`), `metadata: dict`.
- **`Cycle`** dataclass: `nodes: list[str]`, `edges: list[Edge]`, `score: float`, `thesis: str`.
- **`AnswerGraph`** class:
  - `adj: dict[str, list[Edge]]` — adjacency list.
  - `embeddings: dict[str, ndarray]` — per-answer embedding vectors.
  - `add_edges(edges: dict[str, list[Edge]])` — merge edges into the graph.
  - `set_embeddings(answers: list[str], vectors: ndarray)` — populate embedding lookup.
  - `embedding_distance(a, b) -> float` — `1 - cosine_similarity`, defaults to 0.5 for missing.
  - `neighbors(node, edge_type=None) -> list[Edge]` — filtered adjacency lookup.
- **`find_cycles(graph, start, ...) -> list[Cycle]`** — stochastic search:
  1. Depart via category bridge.
  2. Walk outward biased toward embedding distance from start.
  3. At each step check for return edges via a different edge type.
  4. Score by average drift × edge-type diversity bonus.
  5. Deduplicate by node-set, return top-K.

### `enrichment.py` — DB enrichment and edge building

- **`build_answer_nodes(con)`** — creates `answer_nodes` table in DuckDB with `correct_response`, `categories` (array), `category_span`, `clue_count`, `contextualized_clues` (array of `"category: text"`).
- **`build_category_bridges(con) -> dict[str, list[Edge]]`** — self-join on clues sharing a category.
- **`build_embeddings(con, cache_path) -> tuple[list[str], ndarray]`** — embed contextualized clues via sentence-transformers, cache to `.npz`. Raises `ImportError` if sentence-transformers not installed.
- **`build_semantic_edges(answers, vectors, threshold, k) -> dict[str, list[Edge]]`** — brute-force cosine ANN.
- **`build_entity_edges(con, cache_path) -> dict[str, list[Edge]]`** — NER via spacy, inverted index, pairwise edges. Raises `ImportError` if spacy not installed.
- **`build_graph(con, cache_dir) -> AnswerGraph`** — orchestrator that calls all of the above and returns a fully assembled graph.
- **`get_seed_answers(con, min_span, limit) -> list[tuple[str, int, list[str]]]`** — returns top answers by category span.

### `evaluator.py` — Provider-agnostic LLM evaluation

- **`CycleEvaluator` protocol**: `def __call__(self, cycle: Cycle) -> str` — returns thesis or empty string for rejection.
- **`format_cycle(cycle) -> str`** — formats a cycle as human-readable text with hop descriptions.
- **`TextEvaluator`** class (default): returns `format_cycle()` output for manual review.
- **`AnthropicEvaluator`** class (optional): uses `anthropic` SDK to get a thesis sentence. Configurable model name.
- **`evaluate_cycles(cycles, evaluator) -> list[Cycle]`** — runs evaluator on each cycle, sets `thesis`, filters rejects.

## Files to change

| File | Action | Summary |
|------|--------|---------|
| `src/jt3/graph.py` | Create | Edge, Cycle, AnswerGraph, find_cycles |
| `src/jt3/enrichment.py` | Create | DB enrichment + edge builders |
| `src/jt3/evaluator.py` | Create | Evaluator protocol + implementations |
| `src/jt3/__init__.py` | Modify | Re-export key public names |
| `pyproject.toml` | Modify | Add `numpy` dep, `[graph]` optional extra |
| `tests/test_graph.py` | Create | Unit tests for graph + cycle search |
| `tests/test_enrichment.py` | Create | Tests for enrichment with small DuckDB |
| `tests/test_evaluator.py` | Create | Tests for evaluator formatting |

## Step-by-step instructions

### 1. `pyproject.toml`

Add `numpy>=1.24` to `dependencies`. Add optional group:
```toml
[project.optional-dependencies]
graph = ["sentence-transformers>=2.0", "spacy>=3.0", "anthropic>=0.20"]
```

### 2. `src/jt3/graph.py`

Create with:
- `Edge` dataclass (target, edge_type, metadata)
- `Cycle` dataclass (nodes, edges, score, thesis)
- `AnswerGraph` class (adj, embeddings, add_edges, set_embeddings, embedding_distance, neighbors)
- `find_cycles()` function per algorithm above
- Config constants: SIMILARITY_THRESHOLD=0.55, SIMILARITY_NEIGHBORS=20, CYCLE_MIN/MAX=4/8, CANDIDATES_PER_SEED=200, TOP_CYCLES=10

### 3. `src/jt3/enrichment.py`

Create with all enrichment functions. Use `db.get_connection()` patterns. Cache paths default to `data/` directory via `_PROJECT_ROOT`. Guard heavy imports (`sentence_transformers`, `spacy`) with try/except at call site.

### 4. `src/jt3/evaluator.py`

Create with protocol + two implementations. Guard `anthropic` import. Format cycle hops with edge-type-specific descriptions.

### 5. `src/jt3/__init__.py`

Add imports: `AnswerGraph`, `Cycle`, `Edge`, `find_cycles` from graph; `build_graph`, `build_answer_nodes`, `get_seed_answers` from enrichment; `TextEvaluator`, `evaluate_cycles` from evaluator.

### 6. Tests

See test plan below.

## Test plan

### `tests/test_graph.py`

| # | Test | Expected |
|---|------|----------|
| 1 | `test_edge_defaults` | Edge has empty dict metadata by default |
| 2 | `test_cycle_defaults` | Cycle has score=0.0 and thesis="" |
| 3 | `test_add_edges` | AnswerGraph.adj populated correctly |
| 4 | `test_set_embeddings` | embeddings dict populated |
| 5 | `test_embedding_distance_identical` | Returns 0.0 for same vector |
| 6 | `test_embedding_distance_orthogonal` | Returns 1.0 for orthogonal vectors |
| 7 | `test_embedding_distance_missing` | Returns 0.5 for unknown node |
| 8 | `test_neighbors_all` | Returns all neighbors |
| 9 | `test_neighbors_filtered` | Returns only matching edge_type |
| 10 | `test_neighbors_empty` | Returns [] for unknown node |
| 11 | `test_find_cycles_returns_cycles` | Finds cycles in a synthetic graph with 3 edge types |
| 12 | `test_find_cycles_no_bridges` | Returns [] when start has no category bridges |
| 13 | `test_find_cycles_dedup` | Deduplicates cycles with same node-set |
| 14 | `test_find_cycles_respects_max_depth` | No cycles longer than max_depth + 1 nodes |

### `tests/test_enrichment.py`

| # | Test | Expected |
|---|------|----------|
| 1 | `test_build_answer_nodes` | Creates table with correct columns and data from small DB |
| 2 | `test_build_answer_nodes_skips_empty` | Excludes null/empty correct_response |
| 3 | `test_build_category_bridges` | Returns edges linking answers from same category |
| 4 | `test_build_category_bridges_no_self` | No edge from answer to itself |
| 5 | `test_build_semantic_edges` | Returns edges above threshold |
| 6 | `test_build_semantic_edges_excludes_self` | No self-edges |
| 7 | `test_get_seed_answers` | Returns answers sorted by category_span desc |

### `tests/test_evaluator.py`

| # | Test | Expected |
|---|------|----------|
| 1 | `test_format_cycle_category_bridge` | Includes category name in output |
| 2 | `test_format_cycle_semantic` | Includes similarity score |
| 3 | `test_format_cycle_shared_entity` | Includes entity name |
| 4 | `test_text_evaluator_returns_formatted` | TextEvaluator returns format_cycle() output |
| 5 | `test_evaluate_cycles_filters_empty` | evaluate_cycles drops cycles where evaluator returns "" |
| 6 | `test_evaluate_cycles_sets_thesis` | Sets thesis field on passing cycles |

## Out of scope

- FAISS integration for large datasets (brute-force only)
- CLI entry point
- Notebook for graph exploration
- Visualization / export
- Any changes to existing scraper/crawler/db modules
