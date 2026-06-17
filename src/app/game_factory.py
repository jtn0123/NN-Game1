"""Game construction helpers shared by runtime modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Config
from src.game import (
    BaseGame,
    BaseVecGame,
    GameConstructor,
    get_game,
    get_vec_game,
    list_games,
)


@dataclass
class GameEnvironment:
    """Concrete game environment selected for a runtime mode."""

    game: BaseGame
    game_class: GameConstructor
    vec_env: Optional[BaseVecGame] = None
    num_envs: int = 1


def available_game_message(game_name: str) -> str:
    """Return a clear unknown-game error message."""
    return f"Unknown game: {game_name}. Available games: {', '.join(list_games())}"


def resolve_game_class(game_name: str) -> GameConstructor:
    """Resolve a registered game class or raise a clear ValueError."""
    game_class = get_game(game_name)
    if game_class is None:
        raise ValueError(available_game_message(game_name))
    return game_class


def create_single_game(
    game_name: str,
    config: Config,
    *,
    headless: Optional[bool] = None,
) -> GameEnvironment:
    """Create one registered game instance."""
    game_class = resolve_game_class(game_name)
    if headless is None:
        game = game_class(config)
    else:
        game = game_class(config, headless=headless)
    return GameEnvironment(game=game, game_class=game_class)


def create_training_environment(
    game_name: str,
    config: Config,
    *,
    num_envs: int = 1,
    headless: bool = True,
) -> GameEnvironment:
    """Create either a single or vectorized game environment."""
    game_class = resolve_game_class(game_name)

    if num_envs > 1:
        vec_game_class = get_vec_game(game_name)
        if vec_game_class is not None:
            vec_env = vec_game_class(num_envs, config, headless=headless)
            return GameEnvironment(
                game=vec_env.envs[0],
                game_class=game_class,
                vec_env=vec_env,
                num_envs=num_envs,
            )

    game = game_class(config, headless=headless)
    return GameEnvironment(game=game, game_class=game_class, num_envs=1)
