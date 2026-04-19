"""Search for similar clues, responses, and categories by encoding a query string."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import (
    EMBEDDING_TABLES,
    _resolve_model_key,
    load_model,
    search_all_tables,
)
from jt3.embeddings.db import get_model_name

TABLE_LABELS = {
    "embeddings.clues": "Clues",
    "embeddings.responses": "Responses",
    "embeddings.categories": "Categories",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the most similar clues, responses, and categories to a query string"
    )
    parser.add_argument("query", help="Text to search for similar results")
    parser.add_argument(
        "--model",
        default=None,
        help="Model config key (auto-detected from DB when omitted)",
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

    if args.model is None:
        for table, _ in EMBEDDING_TABLES:
            model_path = get_model_name(db_path=args.db, table=table)
            if model_path:
                model_key = _resolve_model_key(model_path)
                break
        else:
            parser.error("No embeddings found in DB. Specify --model explicitly.")
    else:
        model_key = args.model

    model = load_model(model_key)
    embedding = model.encode([args.query])[0].tolist()
    all_results = search_all_tables(embedding, n=args.n, db_path=args.db)

    for table, results in all_results.items():
        label = TABLE_LABELS.get(table, table)
        print(f"\n=== {label} (top {args.n}) ===")
        if not results:
            print("  No results found.")
            continue
        print(f"{'Rank':>4}  {'Score':>7}  Text")
        print(f"{'----':>4}  {'-------':>7}  ----")
        for i, (text, score) in enumerate(results, 1):
            print(f"{i:>4}  {score:>7.4f}  {text}")


if __name__ == "__main__":
    main()
