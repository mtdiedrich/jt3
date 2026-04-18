"""Search clues, responses, and categories by centroid of multiple query strings."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import compute_centroid, load_model, search_all_tables

TABLE_LABELS = {
    "embeddings.clues": "Clues",
    "embeddings.responses": "Responses",
    "embeddings.categories": "Categories",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the most similar clues, responses, and categories "
        "to the centroid of N query strings"
    )
    parser.add_argument(
        "queries", nargs="+", help="Query strings to embed and average"
    )
    parser.add_argument(
        "--model",
        default="all_MiniLM_L6_v2",
        help="Model config key (default: all_MiniLM_L6_v2)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of results per table (default: 10)",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    embeddings = model.encode(args.queries)
    centroid = compute_centroid(embeddings)
    all_results = search_all_tables(centroid, n=args.n, db_path=args.db)

    for table, results in all_results.items():
        label = TABLE_LABELS.get(table, table)
        print(f"\n=== {label} (top {args.n}) ===")
        if not results:
            print("  No results found.")
            continue
        print(f"{'Rank':>6}  {'Score':>7}  Text")
        print(f"{'----':>6}  {'-------':>7}  ----")
        for i, (text, score) in enumerate(results, 1):
            print(f"{i:>6}  {score:>7.4f}  {text}")


if __name__ == "__main__":
    main()
