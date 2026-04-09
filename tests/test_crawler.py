"""Tests for jt3.crawler — batch episode fetching with robots.txt support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jt3.crawler import (
    check_robots,
    episode_url,
    fetch_episodes,
    get_season_game_ids,
    list_seasons,
    season_url,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_game.html"
SEASON_FIXTURE = Path(__file__).parent / "fixtures" / "sample_season.html"
SEASONS_LIST_FIXTURE = Path(__file__).parent / "fixtures" / "sample_seasons_list.html"


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# episode_url
# ---------------------------------------------------------------------------


def test_episode_url_basic():
    assert episode_url(42) == "https://j-archive.com/showgame.php?game_id=42"


def test_episode_url_one():
    assert episode_url(1) == "https://j-archive.com/showgame.php?game_id=1"


# ---------------------------------------------------------------------------
# check_robots
# ---------------------------------------------------------------------------

_ROBOTS_ALLOWED = """\
User-agent: *
Crawl-delay: 20
"""

_ROBOTS_DISALLOWED = """\
User-agent: *
Disallow: /
"""


@patch("jt3.crawler.requests.get")
def test_check_robots_allowed(mock_get: MagicMock):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = _ROBOTS_ALLOWED
    mock_get.return_value = resp
    result = check_robots()
    assert result == 20.0


@patch("jt3.crawler.requests.get")
def test_check_robots_disallowed(mock_get: MagicMock):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = _ROBOTS_DISALLOWED
    mock_get.return_value = resp
    result = check_robots()
    assert result is None


# ---------------------------------------------------------------------------
# fetch_episodes
# ---------------------------------------------------------------------------


@patch("jt3.crawler.time.sleep")
@patch("jt3.crawler.check_robots", return_value=20.0)
@patch("jt3.crawler.fetch_episode")
def test_fetch_episodes_yields_in_order(
    mock_fetch: MagicMock,
    mock_robots: MagicMock,
    mock_sleep: MagicMock,
    fixture_html: str,
):
    """fetch_episodes yields one Episode per game_id, in order."""
    from jt3.scraper import parse_episode

    ep1 = parse_episode(fixture_html, game_id=1)
    ep2 = parse_episode(fixture_html, game_id=2)
    mock_fetch.side_effect = [ep1, ep2]

    results = list(fetch_episodes([1, 2], delay=0))
    assert len(results) == 2
    assert results[0].game_id == 1
    assert results[1].game_id == 2


@patch("jt3.crawler.check_robots", return_value=None)
def test_fetch_episodes_raises_when_disallowed(mock_robots: MagicMock):
    with pytest.raises(PermissionError, match="robots.txt"):
        list(fetch_episodes([1]))


@patch("jt3.crawler.time.sleep")
@patch("jt3.crawler.check_robots", return_value=20.0)
@patch("jt3.crawler.fetch_episode")
def test_fetch_episodes_skips_http_errors(
    mock_fetch: MagicMock,
    mock_robots: MagicMock,
    mock_sleep: MagicMock,
    fixture_html: str,
):
    """HTTP errors for individual episodes are skipped, not raised."""
    import requests as req

    from jt3.scraper import parse_episode

    ep = parse_episode(fixture_html, game_id=3)
    mock_fetch.side_effect = [
        req.HTTPError("404"),
        ep,
    ]

    results = list(fetch_episodes([1, 3], delay=0))
    assert len(results) == 1
    assert results[0].game_id == 3


@patch("jt3.crawler.time.sleep")
@patch("jt3.crawler.check_robots", return_value=20.0)
@patch("jt3.crawler.fetch_episode")
def test_fetch_episodes_uses_robots_delay(
    mock_fetch: MagicMock,
    mock_robots: MagicMock,
    mock_sleep: MagicMock,
    fixture_html: str,
):
    """When no explicit delay is passed, uses the robots.txt Crawl-delay."""
    from jt3.scraper import parse_episode

    ep = parse_episode(fixture_html, game_id=1)
    mock_fetch.side_effect = [ep, ep]

    list(fetch_episodes([1, 2]))
    # Sleep should have been called once (between requests, not before the first)
    mock_sleep.assert_called_once_with(20.0)


# ---------------------------------------------------------------------------
# season_url
# ---------------------------------------------------------------------------


def test_season_url_int():
    assert season_url(1) == "https://j-archive.com/showseason.php?season=1"


def test_season_url_str():
    assert season_url("superjeopardy") == "https://j-archive.com/showseason.php?season=superjeopardy"


# ---------------------------------------------------------------------------
# list_seasons
# ---------------------------------------------------------------------------


@patch("jt3.crawler.requests.get")
def test_list_seasons(mock_get: MagicMock):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = SEASONS_LIST_FIXTURE.read_text(encoding="utf-8")
    mock_get.return_value = resp

    seasons = list_seasons()
    assert len(seasons) == 3
    assert seasons[0] == {"season": "2", "name": "Season 2", "game_count": 179}
    assert seasons[1] == {"season": "1", "name": "Season 1", "game_count": 164}
    assert seasons[2] == {"season": "trebek_pilots", "name": "Trebek pilots", "game_count": 2}


# ---------------------------------------------------------------------------
# get_season_game_ids
# ---------------------------------------------------------------------------


@patch("jt3.crawler.requests.get")
def test_get_season_game_ids(mock_get: MagicMock):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = SEASON_FIXTURE.read_text(encoding="utf-8")
    mock_get.return_value = resp

    ids = get_season_game_ids(1)
    # Fixture has game_ids 103, 102, 101 (newest-first in HTML)
    assert ids == [103, 102, 101]


# ---------------------------------------------------------------------------
# fetch_season
# ---------------------------------------------------------------------------


@patch("jt3.crawler.time.sleep")
@patch("jt3.crawler.check_robots", return_value=20.0)
@patch("jt3.crawler.fetch_episode")
@patch("jt3.crawler.get_season_game_ids", return_value=[103, 102, 101])
def test_fetch_season_chronological_order(
    mock_ids: MagicMock,
    mock_fetch: MagicMock,
    mock_robots: MagicMock,
    mock_sleep: MagicMock,
    fixture_html: str,
):
    """fetch_season yields episodes in chronological order (reversed from page)."""
    from jt3.crawler import fetch_season
    from jt3.scraper import parse_episode

    ep1 = parse_episode(fixture_html, game_id=101)
    ep2 = parse_episode(fixture_html, game_id=102)
    ep3 = parse_episode(fixture_html, game_id=103)
    mock_fetch.side_effect = [ep1, ep2, ep3]

    results = list(fetch_season(1, delay=0))
    assert len(results) == 3
    assert [ep.game_id for ep in results] == [101, 102, 103]
