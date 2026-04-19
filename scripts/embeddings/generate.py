"""Generate clue, response, and category embeddings in a single run."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import (
    generate_category_embeddings,
    generate_clue_embeddings,
    generate_response_embeddings,
    load_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clue, response, and category embeddings"
    )
    parser.add_argument(
        "--model",
        default="all_minilm_l6_v2",
        help="Model config key (default: all_minilm_l6_v2)",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    n = generate_clue_embeddings(model, db_path=args.db)
    print(f"Saved {n} clue embeddings")

    n = generate_response_embeddings(model, db_path=args.db)
    print(f"Saved {n} response embeddings")

    n = generate_category_embeddings(model, db_path=args.db)
    print(f"Saved {n} category embeddings")


if __name__ == "__main__":
    main()
