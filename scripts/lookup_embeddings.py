"""CLI script to look up embeddings for given query strings."""

import sys
import time

t0 = time.perf_counter()
from jt3.lookup import lookup_embeddings  # noqa: E402

t_import = time.perf_counter()


def main() -> None:
    queries = sys.argv[1:]
    if not queries:
        print("Usage: uv run python scripts/lookup_embeddings.py <query1> [query2 ...]")
        sys.exit(1)

    t1 = time.perf_counter()
    found, missing = lookup_embeddings(queries)
    t2 = time.perf_counter()

    if found:
        first = next(iter(found.values()))
        print(f"Retrieved {len(found)} embedding(s), shape: {first.shape}")
    else:
        print("No embeddings found.")

    if missing:
        print(f"Not found ({len(missing)}): {missing}")

    print(f"\n[timing] import: {t_import - t0:.2f}s | lookup: {t2 - t1:.2f}s | total: {t2 - t0:.2f}s")


if __name__ == "__main__":
    main()
