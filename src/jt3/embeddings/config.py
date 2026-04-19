"""Model configuration and loading for the embeddings pipeline."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODELS: dict[str, dict] = {
    "all_minilm_l6_v2": dict(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
    ),
    "qwen3_embedding_06b": dict(
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
    ),
    "qwen3_embedding_06b_trunc_32": dict(
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        truncate_dim=32,
    ),
}


def load_model(model_key: str) -> SentenceTransformer:
    """Load a SentenceTransformer by config key."""
    key = model_key.lower()
    if key not in MODELS:
        raise KeyError(
            f"Unknown model key {model_key!r}. "
            f"Available: {', '.join(MODELS)}"
        )
    return SentenceTransformer(**MODELS[key])


def _resolve_model_key(model_path: str) -> str:
    """Find the MODELS config key for a HuggingFace model path."""
    for key, cfg in MODELS.items():
        if cfg["model_name_or_path"] == model_path:
            return key
    raise KeyError(
        f"No config key found for model {model_path!r}. "
        f"Available: {', '.join(MODELS)}"
    )
