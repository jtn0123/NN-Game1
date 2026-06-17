"""Pure Space Invaders scoring and difficulty rules."""

from __future__ import annotations


def alien_pressure_ratio(remaining_aliens: int, total_aliens: int) -> float:
    """Return the alive-alien ratio used for pressure scaling."""
    if total_aliens <= 0:
        return 1.0
    return max(0.0, min(1.0, remaining_aliens / total_aliens))


def alien_pulse_speed(remaining_aliens: int, total_aliens: int) -> float:
    """Return the alien visual pulse speed for the current wave pressure."""
    ratio = alien_pressure_ratio(remaining_aliens, total_aliens)
    return 0.5 + (1.0 - ratio) * 3.0


def alien_shoot_chance(base_chance: float, remaining_aliens: int, total_aliens: int) -> float:
    """Return per-alien fire chance after pressure scaling."""
    ratio = alien_pressure_ratio(remaining_aliens, total_aliens)
    return base_chance * (1.0 + 0.5 * (1.0 - ratio))


def alien_points(alien_type: int) -> int:
    """Return classic score for a Space Invaders alien type."""
    return 30 - alien_type * 10


def alien_speed_after_kill(
    base_speed: float,
    total_aliens: int,
    remaining_aliens: int,
) -> float:
    """Return alien horizontal speed after kills in the current wave."""
    killed_count = max(0, total_aliens - remaining_aliens)
    return base_speed * (1.0 + killed_count * 0.015)


def level_speed(base_speed: float, level: int) -> float:
    """Return base alien speed for a level."""
    level_index = max(0, level - 1)
    return base_speed * (1.0 + 0.15 * level_index)


def level_y_offset(level: int) -> int:
    """Return how many pixels lower a wave starts at a level."""
    level_index = max(0, level - 1)
    return min(level_index * 20, 100)


def invasion_reached(alien_y: int, alien_height: int, ground_y: int) -> bool:
    """Return whether an alien has reached the defended ground line."""
    return alien_y + alien_height >= ground_y
