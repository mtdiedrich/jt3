from datetime import date
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from jt3.scraper import _parse_contestants, _parse_dollar, _parse_title, parse_episode

FIXTURE = Path(__file__).parent / "fixtures" / "sample_game.html"


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def fixture_soup(fixture_html: str) -> BeautifulSoup:
    return BeautifulSoup(fixture_html, "lxml")


# ---------------------------------------------------------------------------
# _parse_dollar
# ---------------------------------------------------------------------------

def test_parse_dollar_plain():
    from jt3.scraper import _parse_dollar
    assert _parse_dollar("$200") == 200


def test_parse_dollar_with_comma():
    from jt3.scraper import _parse_dollar
    assert _parse_dollar("$1,200") == 1200


def test_parse_dollar_large():
    from jt3.scraper import _parse_dollar
    assert _parse_dollar("$2,000") == 2000


def test_parse_dollar_invalid_returns_zero():
    from jt3.scraper import _parse_dollar
    assert _parse_dollar("") == 0


# ---------------------------------------------------------------------------
# _parse_title
# ---------------------------------------------------------------------------

def test_parse_title_valid(fixture_soup: BeautifulSoup):
    show_number, air_date = _parse_title(fixture_soup)
    assert show_number == 1
    assert air_date == date(2026, 1, 1)


def test_parse_title_no_h1():
    soup = BeautifulSoup("<html></html>", "lxml")
    show_number, air_date = _parse_title(soup)
    assert show_number is None
    assert air_date is None


# ---------------------------------------------------------------------------
# _parse_contestants
# ---------------------------------------------------------------------------

def test_parse_contestants_count(fixture_soup: BeautifulSoup):
    contestants = _parse_contestants(fixture_soup)
    assert len(contestants) == 3


def test_parse_contestants_first_name(fixture_soup: BeautifulSoup):
    contestants = _parse_contestants(fixture_soup)
    assert contestants[0].name == "Alice Smith"


def test_parse_contestants_first_description(fixture_soup: BeautifulSoup):
    contestants = _parse_contestants(fixture_soup)
    assert "teacher" in contestants[0].description


def test_parse_contestants_player_id(fixture_soup: BeautifulSoup):
    contestants = _parse_contestants(fixture_soup)
    assert contestants[0].player_id == 101
    assert contestants[1].player_id == 102
    assert contestants[2].player_id == 103


# ---------------------------------------------------------------------------
# parse_episode — top-level
# ---------------------------------------------------------------------------

def test_parse_episode_game_id(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    assert ep.game_id == 9418


def test_parse_episode_show_number(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    assert ep.show_number == 1


def test_parse_episode_air_date(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    assert ep.air_date == date(2026, 1, 1)


def test_parse_episode_contestants(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    assert len(ep.contestants) == 3


def test_parse_episode_rounds(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    assert len(ep.rounds) == 3


def test_parse_episode_round_names(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    names = [r.name for r in ep.rounds]
    assert "Jeopardy!" in names
    assert "Double Jeopardy!" in names
    assert "Final Jeopardy!" in names


# ---------------------------------------------------------------------------
# Jeopardy! round
# ---------------------------------------------------------------------------

def test_jeopardy_round_category_count(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    assert len(j_round.categories) == 6


def test_jeopardy_round_first_category_name(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    assert j_round.categories[0].name == "SCIENCE"


def test_jeopardy_round_category_with_comments(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    history = j_round.categories[1]
    assert history.name == "HISTORY"
    assert history.comments == "All answers are from the 20th century"


def test_jeopardy_round_first_clue_value(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert clue.value == 200


def test_jeopardy_round_first_clue_text(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert "atomic number 1" in clue.text


def test_jeopardy_round_first_clue_correct_response(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert clue.correct_response == "Hydrogen"


def test_jeopardy_round_first_clue_order(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert clue.order == 3


def test_jeopardy_round_first_clue_answerer(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert clue.answerer == "Alice"


def test_jeopardy_round_first_clue_not_daily_double(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert clue.is_daily_double is False


def test_daily_double_clue(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    dd_clue = j_round.categories[5].clues[0]  # POTPOURRI, first clue
    assert dd_clue.is_daily_double is True
    assert dd_clue.value == 1000
    assert dd_clue.correct_response == "Paris"


def test_clue_id_format(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    j_round = next(r for r in ep.rounds if r.name == "Jeopardy!")
    clue = j_round.categories[0].clues[0]
    assert clue.clue_id == "J_1_1"


# ---------------------------------------------------------------------------
# Final Jeopardy!
# ---------------------------------------------------------------------------

def test_final_jeopardy_category_name(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    fj_round = next(r for r in ep.rounds if r.name == "Final Jeopardy!")
    assert fj_round.categories[0].name == "WORLD CAPITALS"


def test_final_jeopardy_clue_text(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    fj_round = next(r for r in ep.rounds if r.name == "Final Jeopardy!")
    clue = fj_round.categories[0].clues[0]
    assert "northernmost" in clue.text


def test_final_jeopardy_correct_response(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    fj_round = next(r for r in ep.rounds if r.name == "Final Jeopardy!")
    clue = fj_round.categories[0].clues[0]
    assert clue.correct_response == "Reykjavik"


def test_final_jeopardy_clue_id(fixture_html: str):
    ep = parse_episode(fixture_html, 9418)
    fj_round = next(r for r in ep.rounds if r.name == "Final Jeopardy!")
    clue = fj_round.categories[0].clues[0]
    assert clue.clue_id == "FJ"


# ---------------------------------------------------------------------------
# fetch_episode (URL parsing only — no live network call)
# ---------------------------------------------------------------------------

def test_fetch_episode_invalid_url_raises():
    from jt3.scraper import fetch_episode
    with pytest.raises(ValueError, match="game_id"):
        fetch_episode("https://example.com/no-game-id")
