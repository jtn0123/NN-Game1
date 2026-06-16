"""
Game Module
===========

Contains game implementations that the AI can learn to play.

Classes:
    Breakout - Classic Atari Breakout game
    BaseGame - Abstract base class for creating new games
    ParticleSystem - Visual effects system

Game Registry:
    Use get_game(name) to get a game class by name
    Use list_games() to get all available games
    Use get_game_info(name) to get metadata about a game
"""

from typing import Any, Dict, List, Optional, Type, cast

from .breakout import Breakout, VecBreakout
from .space_invaders import SpaceInvaders, VecSpaceInvaders
from .pong import Pong, VecPong
from .snake import Snake, VecSnake
from .asteroids import Asteroids, VecAsteroids
from .base_game import (
    BaseGame,
    BaseVecGame,
    ControlDisplayProvider,
    HumanActionProvider,
    HumanStepProvider,
)
from .particles import ParticleSystem, TrailRenderer
from .menu import GameMenu

# =============================================================================
# GAME REGISTRY
# =============================================================================
# Maps game names to their classes and metadata.
# To add a new game:
#   1. Create the game class inheriting from BaseGame
#   2. Add an entry to GAME_REGISTRY below
#   3. The game will automatically appear in menus and CLI

GAME_REGISTRY: Dict[str, Dict[str, Any]] = {
    "breakout": {
        "class": Breakout,
        "vec_class": VecBreakout,
        "name": "Breakout",
        "description": "Classic brick-breaking arcade game",
        "actions": ["LEFT", "STAY", "RIGHT"],
        "controls": ["LEFT/RIGHT arrows: Move paddle"],
        "difficulty": "Medium",
        "color": (52, 152, 219),  # Blue theme
        "icon": "🧱",
    },
    "space_invaders": {
        "class": SpaceInvaders,
        "vec_class": VecSpaceInvaders,
        "name": "Space Invaders",
        "description": "Defend Earth from alien invasion",
        "actions": ["LEFT", "STAY", "RIGHT", "SHOOT"],
        "controls": ["LEFT/RIGHT arrows: Move ship", "SPACE: Shoot"],
        "difficulty": "Medium-Hard",
        "color": (0, 255, 100),  # Green CRT theme
        "icon": "👾",
    },
    "pong": {
        "class": Pong,
        "vec_class": VecPong,
        "name": "Pong",
        "description": "Classic paddle vs AI opponent",
        "actions": ["UP", "STAY", "DOWN"],
        "controls": ["UP/DOWN arrows (or W/S): Move paddle"],
        "difficulty": "Easy",
        "color": (255, 255, 255),  # White retro theme
        "icon": "🏓",
    },
    "snake": {
        "class": Snake,
        "vec_class": VecSnake,
        "name": "Snake",
        "description": "Grow the snake by eating food",
        "actions": ["UP", "DOWN", "LEFT", "RIGHT"],
        "controls": ["Arrow keys (or WASD): Change direction"],
        "difficulty": "Medium",
        "color": (100, 255, 100),  # Green modern theme
        "icon": "🐍",
    },
    "asteroids": {
        "class": Asteroids,
        "vec_class": VecAsteroids,
        "name": "Asteroids",
        "description": "Destroy asteroids with your spaceship",
        "actions": ["ROTATE_LEFT", "ROTATE_RIGHT", "THRUST", "SHOOT", "NOTHING"],
        "controls": [
            "LEFT/RIGHT arrows: Rotate ship",
            "UP arrow: Thrust",
            "SPACE: Shoot",
            "(Multiple keys can be pressed simultaneously)",
        ],
        "difficulty": "Hard",
        "color": (200, 200, 200),  # Vector gray theme
        "icon": "🚀",
    },
}


def get_game(name: str) -> Optional[Type[BaseGame]]:
    """
    Get a game class by name.

    Args:
        name: Game identifier (e.g., 'breakout', 'space_invaders')

    Returns:
        The game class, or None if not found

    Example:
        >>> GameClass = get_game('breakout')
        >>> game = GameClass(config)
    """
    entry = GAME_REGISTRY.get(name.lower())
    if entry:
        return cast(Type[BaseGame], entry["class"])
    return None


def get_vec_game(name: str) -> Optional[Type[BaseVecGame]]:
    """
    Get a vectorized game environment class by name.

    Args:
        name: Game identifier (e.g., 'breakout', 'space_invaders')

    Returns:
        The vectorized environment class, or None if not supported
    """
    entry = GAME_REGISTRY.get(name.lower())
    if entry and entry.get("vec_class"):
        return cast(Type[BaseVecGame], entry["vec_class"])
    return None


def list_games() -> List[str]:
    """
    Get a list of all available game names.

    Returns:
        List of game identifiers

    Example:
        >>> games = list_games()
        >>> print(games)  # ['breakout', 'space_invaders']
    """
    return list(GAME_REGISTRY.keys())


def get_game_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata about a game.

    Args:
        name: Game identifier

    Returns:
        Dictionary with game info, or None if not found

    Info includes:
        - name: Display name
        - description: Short description
        - actions: List of action names
        - controls: Human-readable keyboard controls
        - difficulty: Difficulty rating
        - color: Theme color (RGB tuple)
        - icon: Emoji icon
    """
    entry = GAME_REGISTRY.get(name.lower())
    if entry:
        return {k: v for k, v in entry.items() if k not in {"class", "vec_class"}}
    return None


def get_all_game_info() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all registered games.

    Returns:
        Dictionary mapping game IDs to their info
    """
    result: Dict[str, Dict[str, Any]] = {}
    for game_id in GAME_REGISTRY.keys():
        info = get_game_info(game_id)
        if info is not None:
            result[game_id] = info
    return result


__all__ = [
    # Classes
    "Breakout",
    "VecBreakout",
    "SpaceInvaders",
    "VecSpaceInvaders",
    "Pong",
    "VecPong",
    "Snake",
    "VecSnake",
    "Asteroids",
    "VecAsteroids",
    "BaseGame",
    "BaseVecGame",
    "ControlDisplayProvider",
    "HumanActionProvider",
    "HumanStepProvider",
    "ParticleSystem",
    "TrailRenderer",
    "GameMenu",
    # Registry functions
    "GAME_REGISTRY",
    "get_game",
    "get_vec_game",
    "list_games",
    "get_game_info",
    "get_all_game_info",
]
