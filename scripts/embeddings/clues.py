"""Generate embeddings for all clue texts and store them in DuckDB."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import generate_clue_embeddings, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clue text embeddings")
    parser.add_argument(
        "--model",
        default="all_MiniLM_L6_v2",
        help="Model config key (default: all_MiniLM_L6_v2)",
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


if __name__ == "__main__":
    main()
