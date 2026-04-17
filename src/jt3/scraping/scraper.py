"""Fetch and parse J! Archive episode pages into Episode objects."""

from __future__ import annotations

import re
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup, Tag

from ..models import Category, Clue, Contestant, Episode, Round

_ROUND_DIVS = [
    ("Jeopardy!", "jeopardy_round"),
    ("Double Jeopardy!", "double_jeopardy_round"),
    ("Final Jeopardy!", "final_jeopardy_round"),
]


def fetch_episode(url: str) -> Episode:
    """Fetch a J! Archive episode page and return a parsed Episode.

    Args:
        url: Full URL, e.g. ``https://j-archive.com/showgame.php?game_id=9418``

    Raises:
        ValueError: If the URL contains no ``game_id`` parameter.
        requests.HTTPError: If the HTTP request fails.
    """
    match = re.search(r"game_id=(\d+)", url)
    if not match:
        raise ValueError(f"Cannot extract game_id from URL: {url!r}")
    game_id = int(match.group(1))
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_episode(response.text, game_id)


def parse_episode(html: str, game_id: int) -> Episode:
    """Parse a J! Archive episode HTML string into an Episode.

    Args:
        html: Raw HTML content of a showgame page.
        game_id: The J! Archive ``game_id`` (from the URL).
    """
    soup = BeautifulSoup(html, "lxml")
    show_number, air_date = _parse_title(soup)
    contestants = _parse_contestants(soup)
    rounds = _parse_rounds(soup)
    return Episode(
        game_id=game_id,
        show_number=show_number,
        air_date=air_date,
        contestants=contestants,
        rounds=rounds,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_title(soup: BeautifulSoup) -> tuple[int | None, date | None]:
    """Extract show number and air date from the page <h1>."""
    h1 = soup.find("h1")
    if not h1:
        return None, None
    text = h1.get_text()

    show_match = re.search(r"Show #(\d+)", text)
    show_number = int(show_match.group(1)) if show_match else None

    date_match = re.search(r"\w+, (\w+ \d+, \d+)", text)
    air_date: date | None = None
    if date_match:
        try:
            air_date = datetime.strptime(date_match.group(1), "%B %d, %Y").date()
        except ValueError:
            pass

    return show_number, air_date


def _parse_contestants(soup: BeautifulSoup) -> list[Contestant]:
    """Extract contestant list from div#contestants_table."""
    contestants: list[Contestant] = []
    table = soup.find(id="contestants_table")
    if not table:
        return contestants

    for p in table.find_all("p"):
        link = p.find("a", href=re.compile(r"showplayer\.php"))
        if not link:
            continue
        name = link.get_text(strip=True)

        player_id_match = re.search(r"player_id=(\d+)", str(link.get("href", "")))
        player_id = int(player_id_match.group(1)) if player_id_match else None

        # Description is the text content after the player link.
        full_text = p.get_text(" ", strip=True)
        desc = full_text.replace(name, "", 1).strip().lstrip(",").strip()
        # Remove "previous game" / "next game" navigation snippets.
        desc = re.sub(r"<<\s*previous game|next game\s*>>", "", desc).strip()

        contestants.append(Contestant(name=name, description=desc, player_id=player_id))

    return contestants


def _parse_rounds(soup: BeautifulSoup) -> list[Round]:
    """Parse all three potential round divs."""
    rounds: list[Round] = []
    for round_name, div_id in _ROUND_DIVS:
        div = soup.find(id=div_id)
        if div is None:
            continue
        if round_name == "Final Jeopardy!":
            rounds.append(_parse_final_jeopardy_round(div))
        else:
            rounds.append(_parse_regular_round(div, round_name))
    return rounds


def _parse_regular_round(div: Tag, round_name: str) -> Round:
    """Parse a Jeopardy! or Double Jeopardy! round div."""
    table = div.find("table", class_="round")
    if not table:
        return Round(name=round_name)

    rows = table.find_all("tr", recursive=False)
    if not rows:
        return Round(name=round_name)

    # Row 0 is always the category header row.
    category_cells = rows[0].find_all("td", class_="category")
    categories: list[Category] = []
    for cell in category_cells:
        name_td = cell.find("td", class_="category_name")
        name = name_td.get_text(strip=True) if name_td else ""
        comments_td = cell.find("td", class_="category_comments")
        comments = comments_td.get_text(strip=True) if comments_td else None
        categories.append(Category(name=name, comments=comments or None))

    # Rows 1+ are clue rows; each cell maps to a column index.
    for row in rows[1:]:
        clue_cells = row.find_all("td", class_="clue")
        for col_idx, cell in enumerate(clue_cells):
            if col_idx >= len(categories):
                continue
            clue = _parse_clue_cell(cell)
            if clue is not None:
                categories[col_idx].clues.append(clue)

    return Round(name=round_name, categories=categories)


def _parse_clue_cell(cell: Tag) -> Clue | None:
    """Parse a single ``td.clue`` cell into a Clue, or None if the cell is empty."""
    text_td = cell.find("td", class_="clue_text")
    if text_td is None:
        return None

    clue_id_full = str(text_td.get("id", ""))  # e.g. "clue_J_1_1"
    clue_id = clue_id_full.removeprefix("clue_")  # e.g. "J_1_1"

    # Clue value and daily-double flag.
    value_td = cell.find("td", class_="clue_value")
    is_daily_double = False
    if value_td is not None:
        value = _parse_dollar(value_td.get_text(strip=True))
    else:
        dd_td = cell.find("td", class_="clue_value_daily_double")
        is_daily_double = dd_td is not None
        raw = dd_td.get_text(strip=True) if dd_td else ""
        value = _parse_dollar(raw.replace("DD:", "").strip())

    # Order the clue was selected.
    order_td = cell.find("td", class_="clue_order_number")
    order: int | None = None
    if order_td is not None:
        try:
            order = int(order_td.get_text(strip=True))
        except ValueError:
            pass

    clue_text = text_td.get_text(strip=True)

    # Correct response lives in a hidden sibling div: id="<clue_id_full>_r".
    correct_response: str | None = None
    answerer: str | None = None
    if clue_id_full:
        reveal_div = cell.find(id=f"{clue_id_full}_r")
        if reveal_div is not None:
            em = reveal_div.find("em", class_="correct_response")
            if em is not None:
                correct_response = em.get_text(strip=True)

    # Answerer is in td.right inside the mouseover div (not the hidden reveal div).
    mouseover_div = cell.find(
        "div",
        onmouseover=re.compile(rf"toggle\('{re.escape(clue_id_full)}_r'"),
    )
    if mouseover_div is not None:
        right_td = mouseover_div.find("td", class_="right")
        if right_td is not None:
            answerer = right_td.get_text(strip=True)

    return Clue(
        clue_id=clue_id,
        order=order,
        value=value,
        is_daily_double=is_daily_double,
        text=clue_text,
        correct_response=correct_response,
        answerer=answerer,
    )


def _parse_final_jeopardy_round(div: Tag) -> Round:
    """Parse the Final Jeopardy! div into a Round with one Category and one Clue."""
    cat_name_td = div.find("td", class_="category_name")
    cat_name = cat_name_td.get_text(strip=True) if cat_name_td else "Final Jeopardy!"

    cat_comments_td = div.find("td", class_="category_comments")
    cat_comments = cat_comments_td.get_text(strip=True) if cat_comments_td else None

    text_td = div.find("td", class_="clue_text")
    clue_text = text_td.get_text(strip=True) if text_td else ""

    clue_id_full = str(text_td.get("id", "clue_FJ")) if text_td else "clue_FJ"
    clue_id = clue_id_full.removeprefix("clue_")

    correct_response: str | None = None
    answerer: str | None = None
    if clue_id_full:
        reveal_div = div.find(id=f"{clue_id_full}_r")
        if reveal_div is not None:
            em = reveal_div.find("em", class_="correct_response")
            if em is not None:
                correct_response = em.get_text(strip=True)

        mouseover_div = div.find(
            "div",
            onmouseover=re.compile(rf"toggle\('{re.escape(clue_id_full)}_r'"),
        )
        if mouseover_div is not None:
            right_td = mouseover_div.find("td", class_="right")
            if right_td is not None:
                answerer = right_td.get_text(strip=True)

    clue = Clue(
        clue_id=clue_id,
        order=None,
        value=0,
        is_daily_double=False,
        text=clue_text,
        correct_response=correct_response,
        answerer=answerer,
    )
    category = Category(name=cat_name, comments=cat_comments or None, clues=[clue])
    return Round(name="Final Jeopardy!", categories=[category])


def _parse_dollar(text: str) -> int:
    """Convert a dollar string like ``"$1,200"`` to an integer."""
    cleaned = text.replace("$", "").replace(",", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return 0
