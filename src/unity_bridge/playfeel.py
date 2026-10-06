"""Classic human-play tuning for the Unity game, independent of training defaults."""

from __future__ import annotations

from src.game.crystal_caves import CrystalCaves

from .classic_levels import apply_classic_caves


def apply_classic_controls(game: CrystalCaves) -> None:
    apply_classic_caves(game)
    # About 4.4 tiles/second for a more deliberate arcade pace.
    # Keep simulation and rendering at 60 Hz.
    game.MOVE_SPEED = 140 / 60
    game.AIR_SPEED = 140 / 60
    # Fixed-height jump: about 80 pixels / 2.5 tiles, in a little over a second.
    game.JUMP_SPEED = 5.3
    game.GRAVITY = 0.17
    # The engine clamps both ascent and descent with this value.
    game.MAX_FALL_SPEED = 5.3
    # The original arcade movement stops when the direction is released.
    game.FRICTION = 0.0
    # Human expeditions do not end because an AI training budget runs out.
    # Keep positive denominators for observations and fit the Unity int protocol.
    game.MAX_STEPS = 2_000_000_000
    game.MAX_STEPS_WITHOUT_PROGRESS = 2_000_000_000
