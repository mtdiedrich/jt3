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
from .db import (
    delete_episode,
    get_embedding,
    list_episodes,
    load_episode,
    save_contextual_embeddings,
    save_embeddings,
    save_episode,
)
from .embeddings import (
    MODELS,
    fetch_clue_texts,
    fetch_response_contexts,
    fetch_response_texts,
    generate_clue_embeddings,
    generate_contextual_response_embeddings,
    generate_prompted_response_embeddings,
    generate_response_embeddings,
    load_model,
)
from .models import Category, Clue, Contestant, Episode, Round
from .scraper import fetch_episode, parse_episode

__all__ = [
    "Category",
    "Clue",
    "Contestant",
    "Episode",
    "MODELS",
    "Round",
    "check_robots",
    "delete_episode",
    "episode_url",
    "fetch_clue_texts",
    "fetch_episode",
    "fetch_episodes",
    "fetch_response_contexts",
    "fetch_response_texts",
    "fetch_season",
    "generate_clue_embeddings",
    "generate_contextual_response_embeddings",
    "generate_prompted_response_embeddings",
    "generate_response_embeddings",
    "get_embedding",
    "get_season_game_ids",
    "list_episodes",
    "list_seasons",
    "load_episode",
    "load_model",
    "parse_episode",
    "save_contextual_embeddings",
    "save_embeddings",
    "save_episode",
    "season_url",
]
