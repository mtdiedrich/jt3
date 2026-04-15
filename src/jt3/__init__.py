__version__ = "0.1.0"

from .crawler import (
    check_robots,
    episode_url,
    fetch_episodes,
    fetch_season,
    get_season_game_ids,
    list_seasons,
    season_url,
)
from .db import delete_episode, list_episodes, load_episode, save_episode
from .embeddings import embed, embed_batch, embed_clues, nearest_to_centroid
from .models import Category, Clue, Contestant, Episode, Round
from .scraper import fetch_episode, parse_episode

__all__ = [
    "Category",
    "Clue",
    "Contestant",
    "Episode",
    "Round",
    "check_robots",
    "delete_episode",
    "embed",
    "embed_batch",
    "embed_clues",
    "nearest_to_centroid",
    "episode_url",
    "fetch_episode",
    "fetch_episodes",
    "fetch_season",
    "get_season_game_ids",
    "list_episodes",
    "list_seasons",
    "load_episode",
    "parse_episode",
    "save_episode",
    "season_url",
]
