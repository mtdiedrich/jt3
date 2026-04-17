"""Generate embeddings for all clue, response, category, and context texts and store them in DuckDB."""

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import (
    generate_category_embeddings,
    generate_clue_embeddings,
    generate_full_context_embeddings,
    generate_response_embeddings,
    load_model,
)


def main(model_key: str = "all_MiniLM_L6_v2") -> None:
    model = load_model(model_key)
    db = DEFAULT_DB_PATH

    n = generate_clue_embeddings(model, db_path=db)
    print(f"Saved {n} clue embeddings")

    n = generate_response_embeddings(model, db_path=db)
    print(f"Saved {n} response embeddings")

    n = generate_category_embeddings(model, db_path=db)
    print(f"Saved {n} category embeddings")

    n = generate_full_context_embeddings(model, db_path=db)
    print(f"Saved {n} full context embeddings")

    # n = generate_prompted_response_embeddings(model, db_path=db)
    # print(f"Saved {n} prompted response embeddings")


if __name__ == "__main__":
    main()
