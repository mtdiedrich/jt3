"""Batch episode fetching with robots.txt support."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Iterator
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .models import Episode
from .scraper import fetch_episode

log = logging.getLogger(__name__)

_ROBOTS_URL = "https://j-archive.com/robots.txt"
_BASE_URL = "https://j-archive.com/showgame.php"
_SEASON_URL = "https://j-archive.com/showseason.php"
_SEASONS_LIST_URL = "https://j-archive.com/listseasons.php"


def episode_url(game_id: int) -> str:
    """Return the J! Archive URL for a given *game_id*."""
    return f"{_BASE_URL}?game_id={game_id}"


def check_robots(user_agent: str = "*") -> float | None:
    """Check robots.txt and return the Crawl-delay if allowed, or ``None`` if disallowed."""
    resp = requests.get(_ROBOTS_URL, timeout=30)
    resp.raise_for_status()

    rp = RobotFileParser()
    rp.parse(resp.text.splitlines())

    if not rp.can_fetch(user_agent, _BASE_URL):
        return None

    delay = rp.crawl_delay(user_agent)
    return float(delay) if delay is not None else 0.0


def fetch_episodes(
    game_ids: Iterable[int],
    *,
    delay: float | None = None,
    user_agent: str = "*",
) -> Iterator[Episode]:
    """Yield :class:`Episode` objects for each *game_id*, respecting robots.txt.

    Parameters
    ----------
    game_ids:
        Game IDs to fetch, in order.
    delay:
        Seconds to wait between requests.  When *None*, uses the
        ``Crawl-delay`` from robots.txt.
    user_agent:
        User-agent string for the robots.txt check.

    Raises
    ------
    PermissionError
        If robots.txt disallows fetching.
    """
    crawl_delay = check_robots(user_agent)
    if crawl_delay is None:
        raise PermissionError(
            "robots.txt disallows fetching for user-agent {!r}".format(user_agent)
        )

    if delay is None:
        delay = crawl_delay

    first = True
    for gid in game_ids:
        if not first and delay > 0:
            time.sleep(delay)
        first = False

        url = episode_url(gid)
        try:
            ep = fetch_episode(url)
        except requests.HTTPError:
            log.warning("Skipping game_id=%d — HTTP error", gid)
            continue
        except Exception:
            log.warning("Skipping game_id=%d — unexpected error", gid, exc_info=True)
            continue

        yield ep


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------


def season_url(season: int | str) -> str:
    """Return the J! Archive URL for a given *season*."""
    return f"{_SEASON_URL}?season={season}"


def list_seasons() -> list[dict]:
    """Fetch the seasons listing and return metadata for each season.

    Returns a list of dicts with keys ``season`` (str), ``name`` (str),
    and ``game_count`` (int), in the order they appear on the page
    (newest first).
    """
    resp = requests.get(_SEASONS_LIST_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    seasons: list[dict] = []
    for link in soup.find_all("a", href=re.compile(r"showseason\.php\?season=")):
        href = str(link.get("href", ""))
        m = re.search(r"season=([^&]+)", href)
        if not m:
            continue

        season_id = m.group(1)
        name = link.get_text(strip=True)

        # Game count sits in a sibling <td> like "(164 games archived)"
        row = link.find_parent("tr")
        game_count = 0
        if row:
            count_match = re.search(r"\((\d+)\s+games?\s+archived\)", row.get_text())
            if count_match:
                game_count = int(count_match.group(1))

        seasons.append({"season": season_id, "name": name, "game_count": game_count})

    return seasons


def get_season_game_ids(season: int | str) -> list[int]:
    """Fetch a season page and return all episode game IDs.

    Returns game IDs in the order they appear on the page (newest first).
    """
    resp = requests.get(season_url(season), timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    game_ids: list[int] = []
    for link in soup.find_all("a", href=re.compile(r"showgame\.php\?game_id=\d+")):
        m = re.search(r"game_id=(\d+)", str(link.get("href", "")))
        if m:
            game_ids.append(int(m.group(1)))

    return game_ids


def fetch_season(
    season: int | str,
    *,
    delay: float | None = None,
    user_agent: str = "*",
) -> Iterator[Episode]:
    """Fetch all episodes for a season in chronological order.

    Retrieves game IDs from the season page, reverses them to chronological
    order, then delegates to :func:`fetch_episodes`.
    """
    game_ids = get_season_game_ids(season)
    # Season pages list newest first; reverse for chronological order.
    game_ids.reverse()
    yield from fetch_episodes(game_ids, delay=delay, user_agent=user_agent)
