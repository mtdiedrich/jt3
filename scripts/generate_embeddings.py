"""Generate embeddings for all clue texts and store them in DuckDB."""

from pathlib import Path

import duckdb
import polars as pl
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "jt3.duckdb"

MODELS = {
    "all_MiniLM_L6_v2": dict(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        backend="onnx",
        model_kwargs={"provider": "CUDAExecutionProvider"},
    ),
    "qwen3_embedding_06B_trunc_32": dict(
        model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
        backend="onnx",
        model_kwargs={"provider": "CUDAExecutionProvider"},
        truncate_dim=32,
    ),
}


def main(model_key: str = "all_MiniLM_L6_v2") -> None:
    con = duckdb.connect(str(DB_PATH))

    rows = con.execute("""
        SELECT DISTINCT c.text AS clue_text
        FROM clues AS c
        ORDER BY c.game_id DESC, c.round_index, c.category_index, c.clue_order
    """).fetchall()
    clues = [r[0] for r in rows]
    print(f"Loaded {len(clues)} clues")

    model = SentenceTransformer(**MODELS[model_key])
    embeddings = model.encode(clues, batch_size=64, show_progress_bar=True)

    # dim = embeddings.shape[1]
    # df = pl.DataFrame({
    #     "clue_text": clues,
    #     "embedding": embeddings.tolist(),
    # }).unique(subset=["clue_text"], keep="first")

    # con.execute("DROP TABLE IF EXISTS embeddings")
    # con.execute(f"""CREATE TABLE embeddings (
    #     clue_text TEXT PRIMARY KEY,
    #     embedding FLOAT[{dim}] NOT NULL
    # )""")
    # con.execute("INSERT INTO embeddings SELECT * FROM df")
    # print(f"Saved {len(df)} embeddings ({len(clues) - len(df)} duplicates removed)")

    # con.close()


if __name__ == "__main__":
    main()
