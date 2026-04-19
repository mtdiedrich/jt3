"""Generate embeddings for all category names and store them in DuckDB."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import generate_category_embeddings, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate category name embeddings")
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
    n = generate_category_embeddings(model, db_path=args.db)
    print(f"Saved {n} category embeddings")


if __name__ == "__main__":
    main()
