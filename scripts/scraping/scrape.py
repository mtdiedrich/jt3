"""Scrape episodes from J! Archive and store them in DuckDB."""

import argparse
import logging

from jt3.db import DEFAULT_DB_PATH
from jt3.scraping import fetch_season, save_episode

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape J! Archive episodes and store in DuckDB"
    )
    parser.add_argument("season", help="Season number or identifier to scrape")
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds between requests (default: use robots.txt Crawl-delay)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    count = 0
    for episode in fetch_season(args.season, delay=args.delay, db_path=args.db):
        save_episode(episode, db_path=args.db)
        count += 1

    print(f"\nDone — saved {count} episodes to {args.db}")


if __name__ == "__main__":
    main()
