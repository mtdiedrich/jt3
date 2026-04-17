"""Generate embeddings for all clue and response texts and store them in DuckDB."""

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import (
    generate_clue_embeddings,
    generate_contextual_response_embeddings,
    generate_prompted_response_embeddings,
    generate_response_embeddings,
    load_model,
)


def main(model_key: str = "all_MiniLM_L6_v2") -> None:
    model = load_model(model_key)
    db = DEFAULT_DB_PATH

    # n = generate_clue_embeddings(model, db_path=db)
    # print(f"Saved {n} clue embeddings")

    # n = generate_response_embeddings(model, db_path=db)
    # print(f"Saved {n} response embeddings")

    n = generate_contextual_response_embeddings(model, db_path=db)
    print(f"Saved {n} contextual response embeddings")

    # n = generate_prompted_response_embeddings(model, db_path=db)
    # print(f"Saved {n} prompted response embeddings")


if __name__ == "__main__":
    main()
