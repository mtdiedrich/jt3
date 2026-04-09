from dataclasses import dataclass, field
from datetime import date


@dataclass
class Contestant:
    name: str
    description: str
    player_id: int | None = None


@dataclass
class Clue:
    clue_id: str
    order: int | None
    value: int
    is_daily_double: bool
    text: str
    correct_response: str | None = None
    answerer: str | None = None


@dataclass
class Category:
    name: str
    comments: str | None = None
    clues: list[Clue] = field(default_factory=list)


@dataclass
class Round:
    name: str
    categories: list[Category] = field(default_factory=list)


@dataclass
class Episode:
    game_id: int
    show_number: int | None
    air_date: date | None
    contestants: list[Contestant] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)
