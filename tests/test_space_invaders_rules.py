"""Tests for pure Space Invaders rule helpers."""

import pytest

from src.game.space_invaders_rules import (
    alien_points,
    alien_pressure_ratio,
    alien_pulse_speed,
    alien_shoot_chance,
    alien_speed_after_kill,
    invasion_reached,
    level_speed,
    level_y_offset,
)


def test_alien_pressure_ratio_handles_empty_and_clamps() -> None:
    assert alien_pressure_ratio(5, 10) == 0.5
    assert alien_pressure_ratio(0, 0) == 1.0
    assert alien_pressure_ratio(-1, 10) == 0.0
    assert alien_pressure_ratio(11, 10) == 1.0


@pytest.mark.parametrize(
    ("alien_type", "points"),
    [
        (0, 30),
        (1, 20),
        (2, 10),
    ],
)
def test_alien_points_match_classic_rows(alien_type: int, points: int) -> None:
    assert alien_points(alien_type) == points


def test_alien_pressure_increases_pulse_and_shooting() -> None:
    assert alien_pulse_speed(10, 10) == pytest.approx(0.5)
    assert alien_pulse_speed(0, 10) == pytest.approx(3.5)
    assert alien_shoot_chance(0.001, 10, 10) == pytest.approx(0.001)
    assert alien_shoot_chance(0.001, 0, 10) == pytest.approx(0.0015)


def test_alien_speed_after_kill_uses_current_wave_progress() -> None:
    assert alien_speed_after_kill(2.0, total_aliens=10, remaining_aliens=10) == pytest.approx(2.0)
    assert alien_speed_after_kill(2.0, total_aliens=10, remaining_aliens=5) == pytest.approx(2.15)


def test_level_difficulty_rules() -> None:
    assert level_speed(1.0, 1) == pytest.approx(1.0)
    assert level_speed(1.0, 3) == pytest.approx(1.3)
    assert level_y_offset(1) == 0
    assert level_y_offset(4) == 60
    assert level_y_offset(20) == 100


def test_invasion_reached_at_ground_line() -> None:
    assert not invasion_reached(alien_y=70, alien_height=20, ground_y=100)
    assert invasion_reached(alien_y=80, alien_height=20, ground_y=100)
    assert invasion_reached(alien_y=81, alien_height=20, ground_y=100)
