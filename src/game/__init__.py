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

from typing import Dict, List, Type, Optional, Any
from .breakout import Breakout
from .space_invaders import SpaceInvaders
from .base_game import BaseGame
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
    'breakout': {
        'class': Breakout,
        'name': 'Breakout',
        'description': 'Classic brick-breaking arcade game',
        'actions': ['LEFT', 'STAY', 'RIGHT'],
        'difficulty': 'Medium',
        'color': (52, 152, 219),  # Blue theme
        'icon': '🧱',
    },
    'space_invaders': {
        'class': SpaceInvaders,
        'name': 'Space Invaders',
        'description': 'Defend Earth from alien invasion',
        'actions': ['LEFT', 'STAY', 'RIGHT', 'SHOOT'],
        'difficulty': 'Medium-Hard',
        'color': (0, 255, 100),  # Green CRT theme
        'icon': '👾',
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
        return entry['class']
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
        - difficulty: Difficulty rating
        - color: Theme color (RGB tuple)
        - icon: Emoji icon
    """
    entry = GAME_REGISTRY.get(name.lower())
    if entry:
        return {k: v for k, v in entry.items() if k != 'class'}
    return None


def get_all_game_info() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all registered games.
    
    Returns:
        Dictionary mapping game IDs to their info
    """
    return {
        game_id: get_game_info(game_id)
        for game_id in GAME_REGISTRY.keys()
    }


__all__ = [
    # Classes
    'Breakout',
    'SpaceInvaders',
    'BaseGame', 
    'ParticleSystem', 
    'TrailRenderer',
    'GameMenu',
    # Registry functions
    'GAME_REGISTRY',
    'get_game',
    'list_games',
    'get_game_info',
    'get_all_game_info',
]
