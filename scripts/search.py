"""Search for similar responses by encoding a query string."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import load_model, search_similar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the most similar responses to a query string"
    )
    parser.add_argument("query", help="Text to search for similar responses")
    parser.add_argument(
        "--model",
        default="all_MiniLM_L6_v2",
        help="Model config key (default: all_MiniLM_L6_v2)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    embedding = model.encode([args.query])[0].tolist()
    results = search_similar(embedding, n=args.n, db_path=args.db)

    if not results:
        print("No results found.")
        return

    print(f"{'Rank':>4}  {'Score':>7}  Response")
    print(f"{'----':>4}  {'-------':>7}  --------")
    for i, (text, score) in enumerate(results, 1):
        print(f"{i:>4}  {score:>7.4f}  {text}")


if __name__ == "__main__":
    main()
