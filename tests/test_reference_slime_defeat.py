"""Slime-only defeat presentation preserves real combat and training outcomes."""

import numpy as np
import pytest

from src.game.crystal_caves_entities import Bullet
from src.unity_bridge.session import CaveSession


@pytest.mark.parametrize("powered", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_fatal_slime_hit_emits_signed_pulse_with_unchanged_rewards(powered, direction):
    session = CaveSession()
    game = session.game
    enemy = next(e for e in game.enemies if e.appearance == "eye_flyer")
    game.freeze_timer = 100
    game.bullets.append(Bullet(enemy.x + 8, enemy.y + 15, direction * 8, 80, powered))
    reward = game._update_enemies()
    assert not enemy.alive and enemy.health == 0 and not game.bullets
    assert reward == 4.0 and game.score == (250 if powered else 200)
    event = game.visual_events[-1]
    assert event.kind == "slime_pulse" and event.ttl == event.max_ttl == 36
    assert event.text == f"+{game.score}" and event.facing == direction
    game.facing *= -1
    effect = session.snapshot()["effects"][-1]
    assert effect["kind"] == "slime_pulse" and effect["facing"] == direction
    observation = game.get_state().copy()
    for _ in range(35):
        game._update_visual_events()
    assert game.visual_events[-1].ttl == 1
    game._update_visual_events()
    assert not game.visual_events and not enemy.alive
    np.testing.assert_array_equal(game.get_state(), observation)


def test_real_climb_and_shot_kills_slime_with_original_score_health_and_ammo():
    session = CaveSession(1)
    # Meet the two-axis eye at its current patrol height. The older 115-tick
    # climb assumed a horizontal-only target and now correctly misses it.
    # Walking off this higher ladder position also collects one 100-point gem.
    for action in [0] * 12 + [2] * 14 + [3] * 121 + [6] + [2] * 10:
        snapshot = session.handle({"op": "step", "actions": [action]})
    assert snapshot["steps"] == 158 and snapshot["health"] == 3
    assert not session.game.enemies[2].alive
    assert any(
        effect["kind"] == "slime_pulse" and effect["text"] == "+200"
        for effect in snapshot["effects"]
    )
    assert snapshot["crystals"] == snapshot["initial_crystals"] - 1
    assert session.game.score == 300 and session.game.ammo == 4


def test_headless_and_training_keep_their_feedback_contract():
    for training in (False, True):
        game = CaveSession(classic_controls=not training).game
        target = game.enemies[1]
        if not training:
            assert target.appearance == "eye_flyer"
            game.headless = True
        game.freeze_timer = 100
        game.bullets.append(Bullet(target.x + 8, target.y + 15, 8, 80, False))
        game._update_enemies()
        assert not target.alive and game.score == 200
        if training:
            assert game.visual_events[-1].kind == "poof"
        else:
            assert not game.visual_events
