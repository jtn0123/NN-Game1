"""Reference combat feedback follows real hits without changing combat rules."""

import numpy as np
import pygame
import pytest

from src.game.crystal_caves_entities import Bullet
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_creatures import creature_sprites
from src.unity_bridge.visual_materials import INK, WHITE


def hit(game, enemy, powered=False):
    game.freeze_timer = 100
    game.bullets.append(Bullet(enemy.x + 8, enemy.y + 15, 0, 80, powered))
    game._update_enemies()


def test_surviving_hit_flashes_for_half_a_second_and_repeat_hit_refreshes_expiry():
    session = CaveSession()
    game = session.game
    enemy = next(e for e in game.enemies if e.appearance == "dinosaur_enemy")
    assert getattr(enemy, "hit_until", 0) == 0
    hit(game, enemy)
    assert enemy.alive and enemy.health == 4 and game.score == 0
    assert getattr(enemy, "hit_until", 0) == game.steps + 30
    game.steps += 12
    hit(game, enemy)
    assert enemy.alive and enemy.health == 3 and game.score == 0
    assert enemy.hit_until == game.steps + 30
    before = game.get_state().copy()
    before_snapshot = session.snapshot()
    assert before_snapshot["player"]["x"] == game.player_x
    assert next(e for e in before_snapshot["entities"] if e["sprite"] == "dinosaur_enemy")["hit"]
    np.testing.assert_array_equal(game.get_state(), before)
    # Presentation expiry is derived from the existing simulation clock.
    game.steps = enemy.hit_until - 1
    assert enemy.hit_until > game.steps
    assert next(e for e in session.snapshot()["entities"] if e["sprite"] == "dinosaur_enemy")["hit"]
    game.steps += 1
    assert not enemy.hit_until > game.steps
    assert not next(e for e in session.snapshot()["entities"] if e["sprite"] == "dinosaur_enemy")[
        "hit"
    ]
    assert enemy.alive and enemy.health == 3
    session.reset(0)
    assert all(e.hit_until == 0 for e in session.game.enemies)


def test_armor_and_powered_fatal_hits_do_not_flash_a_surviving_creature():
    game = CaveSession().game
    rock = next(e for e in game.enemies if e.appearance == "walking_rock")
    hit(game, rock)
    assert rock.alive and rock.health == 1 and getattr(rock, "hit_until", 0) == 0
    enemy = next(e for e in game.enemies if e.appearance == "dinosaur_enemy")
    hit(game, enemy, powered=True)
    assert not enemy.alive and enemy.health == 0 and getattr(enemy, "hit_until", 0) == 0
    assert game.score == 250


def test_hit_deadline_is_presentation_only_and_training_enemies_remain_unchanged():
    game = CaveSession().game
    enemy = next(e for e in game.enemies if e.appearance == "dinosaur_enemy")
    assert hasattr(enemy, "hit_until")
    before = game.get_state().copy()
    enemy.hit_until = game.steps + 30
    np.testing.assert_array_equal(game.get_state(), before)
    enemy.hit_until = 0
    np.testing.assert_array_equal(game.get_state(), before)
    training = CaveSession(classic_controls=False).game
    assert all(not hasattr(e, "hit_until") for e in training.enemies)
    target = training.enemies[-1]
    hit(training, target)
    assert not target.alive and training.score == 200


def test_white_hit_poses_keep_the_exact_green_creature_alpha_and_dimensions():
    sprites = creature_sprites()
    raw_frames = []
    for frame in range(4):
        original = sprites[f"dinosaur_enemy_{frame}"]
        hit_pose = sprites[f"dinosaur_enemy_hit_{frame}"]
        assert hit_pose.get_size() == original.get_size() == (24, 64)
        original_alpha = pygame.image.tobytes(original, "RGBA")[3::4]
        pixels = pygame.image.tobytes(hit_pose, "RGBA")
        assert pixels[3::4] == original_alpha
        colors = {tuple(pixels[n : n + 3]) for n in range(0, len(pixels), 4) if pixels[n + 3]}
        assert colors == {INK, WHITE}
        raw_frames.append(pixels)
    assert len(set(raw_frames)) == 4


@pytest.mark.parametrize("powered", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
def test_green_creature_death_emits_bones_without_changing_combat_rewards(powered, direction):
    session = CaveSession()
    game = session.game
    enemy = next(e for e in game.enemies if e.appearance == "dinosaur_enemy")
    game.freeze_timer = 100
    for _ in range(1 if powered else 5):
        game.bullets.append(Bullet(enemy.x + 8, enemy.y + 15, direction * 8, 80, powered))
        reward = game._update_enemies()
    assert not enemy.alive and enemy.health == 0 and not game.bullets
    assert reward == 4.0 and game.score == (250 if powered else 200)
    deaths = [event for event in game.visual_events if event.kind in {"bones", "poof"}]
    assert len(deaths) == 1
    event = deaths[0]
    assert event.kind == "bones" and event.ttl == event.max_ttl == 72
    assert event.text == f"+{game.score}" and getattr(event, "facing", 0) == direction
    exported = next(effect for effect in session.snapshot()["effects"] if effect["kind"] == "bones")
    assert exported["facing"] == direction
    game.facing *= -1
    assert event.facing == direction
    exported = next(effect for effect in session.snapshot()["effects"] if effect["kind"] == "bones")
    assert exported["facing"] == direction


def test_bones_expire_after_the_final_frame_without_affecting_observations_or_reset():
    session = CaveSession()
    game = session.game
    enemy = next(e for e in game.enemies if e.appearance == "dinosaur_enemy")
    hit(game, enemy, powered=True)
    assert game.visual_events[-1].kind == "bones"
    before = game.get_state().copy()
    for _ in range(71):
        game._update_visual_events()
    assert len(game.visual_events) == 1 and game.visual_events[0].ttl == 1
    np.testing.assert_array_equal(game.get_state(), before)
    game._update_visual_events()
    assert not game.visual_events and not enemy.alive and game.score == 250
    np.testing.assert_array_equal(game.get_state(), before)
    session.reset(0)
    assert not game.visual_events and all(e.alive for e in game.enemies) and game.score == 0


def test_other_species_training_and_headless_combat_keep_their_previous_feedback():
    game = CaveSession().game
    slug = next(e for e in game.enemies if e.appearance == "slug_enemy")
    hit(game, slug)
    assert not slug.alive and game.score == 200
    assert game.visual_events[-1].kind == "poof" and game.visual_events[-1].max_ttl == 36
    training = CaveSession(classic_controls=False).game
    target = training.enemies[-1]
    hit(training, target)
    assert not target.alive and training.score == 200
    assert training.visual_events[-1].kind == "poof"
    assert training.visual_events[-1].max_ttl == 36
    hidden = CaveSession().game
    hidden.headless = True
    target = next(e for e in hidden.enemies if e.appearance == "dinosaur_enemy")
    hit(hidden, target, powered=True)
    assert not target.alive and hidden.score == 250 and not hidden.visual_events


def test_bone_poses_use_four_bounded_native_pixel_frames():
    sprites = creature_sprites()
    frames = []
    for frame in range(4):
        image = sprites[f"defeat_bones_{frame}"]
        assert image.get_size() == (32, 32)
        pixels = pygame.image.tobytes(image, "RGBA")
        assert set(pixels[3::4]) == {0, 255}
        assert any(tuple(pixels[n : n + 3]) == WHITE for n in range(0, len(pixels), 4))
        frames.append(pixels)
    assert len(set(frames)) == 4
