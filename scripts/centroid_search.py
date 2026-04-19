"""Search clues, responses, and categories by minimum similarity across query strings."""

import argparse

from jt3.db import DEFAULT_DB_PATH
from jt3.embeddings import (
    EMBEDDING_TABLES,
    _resolve_model_key,
    load_model,
    search_all_tables_by_min_sim,
)
from jt3.embeddings.db import get_model_name

TABLE_LABELS = {
    "embeddings.clues": "Clues",
    "embeddings.responses": "Responses",
    "embeddings.categories": "Categories",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find clues, responses, and categories closest to all query strings "
        "(ranked by minimum similarity across queries)"
    )
    parser.add_argument(
        "queries", nargs="+", help="Query strings to embed and average"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model config key (auto-detected from DB when omitted)",
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
    query_embeddings = [emb.tolist() for emb in model.encode(args.queries)]
    all_results = search_all_tables_by_min_sim(query_embeddings, n=args.n, db_path=args.db)

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
