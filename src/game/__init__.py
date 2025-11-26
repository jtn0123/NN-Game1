"""
Game Module
===========

Contains game implementations that the AI can learn to play.

Classes:
    Breakout - Classic Atari Breakout game
    BaseGame - Abstract base class for creating new games
    ParticleSystem - Visual effects system
"""

from .breakout import Breakout
from .base_game import BaseGame
from .particles import ParticleSystem, TrailRenderer

__all__ = ['Breakout', 'BaseGame', 'ParticleSystem', 'TrailRenderer']

