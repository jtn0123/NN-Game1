"""Capsule artwork must preserve the original projectile's center and physics."""

import numpy as np
import pygame
import pytest
from PIL import Image

from src.game.crystal_caves_entities import CaveSpec
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visuals import art_sprites


def test_capsule_pngs_have_discrete_gray_poses_and_keep_the_legacy_alias(tmp_path):
    art = art_sprites()
    assert art["bullet"].get_size() == (16, 8)
    poses = []
    for frame in range(4):
        path = tmp_path / f"bullet_{frame}.png"
        pygame.image.save(art[f"bullet_{frame}"], path)
        with Image.open(path) as opened:
            assert opened.size == (16, 8)
            pixels = np.asarray(opened.convert("RGBA"))
        assert set(np.unique(pixels[:, :, 3])) == {0, 255}
        assert np.count_nonzero(pixels[:, :, 3]) >= 65
        visible = pixels[:, :, :3][pixels[:, :, 3] > 0].astype(int)
        assert np.all(visible.max(axis=1) - visible.min(axis=1) <= 65)
        poses.append(pixels)
    assert len({pose.tobytes() for pose in poses}) == 4
    np.testing.assert_array_equal(
        pygame.surfarray.array3d(art["bullet"]), pygame.surfarray.array3d(art["bullet_0"])
    )


def projectile_arena(classic_controls):
    session = CaveSession(classic_controls=classic_controls)
    rows = ["#" * 40] + ["#" + "." * 38 + "#" for _ in range(12)]
    floor = list("#" + "." * 38 + "#")
    floor[20], floor[30], floor[36] = "P", "*", "E"
    rows += ["".join(floor), "#" * 40, "#" * 40]
    session.game.CAVES = (CaveSpec("projectile check", tuple(rows), (0, 0, 0), (0, 255, 0)),)
    session.game.use_eval_levels(1)
    session.reset(0)
    session.handle({"op": "step", "actions": [0] * 4})
    return session


@pytest.mark.parametrize("classic_controls", [True, False])
@pytest.mark.parametrize("facing", [-1, 1])
def test_capsule_rendering_keeps_actual_shot_speed_lifetime_bounds_and_observations(
    classic_controls, facing
):
    session = projectile_arena(classic_controls)
    reference = projectile_arena(classic_controls)
    for game in (session.game, reference.game):
        game.facing = facing
    ammo = session.game.ammo
    for frame in range(6):
        action = session.game.SHOOT if frame == 0 else 0
        state, reward, done, _ = reference.game.step(action)
        snapshot = session.handle({"op": "step", "actions": [action]})
        np.testing.assert_array_equal(session.game.get_state(), state)
        assert snapshot["last_reward"] == pytest.approx(reward)
        assert snapshot["done"] is done
        actual = session.game.bullets[0]
        expected = reference.game.bullets[0]
        assert (actual.x, actual.y, actual.vx, actual.ttl, actual.rect) == (
            expected.x,
            expected.y,
            expected.vx,
            expected.ttl,
            expected.rect,
        )
        assert actual.vx == facing * session.game.BULLET_SPEED
        assert actual.ttl == 79 - frame and actual.rect.size == (10, 4)
        entity = next(item for item in snapshot["entities"] if item["id"] == "bullet_0")
        assert entity["sprite"] == "bullet"
        assert entity["flip"] is (facing < 0)
        assert (entity["x"], entity["y"], entity["w"], entity["h"]) == (
            actual.x,
            actual.y,
            10,
            4,
        )
    assert session.game.ammo == ammo - 1


@pytest.mark.parametrize("classic_controls", [True, False])
def test_muzzle_spark_is_hidden_while_real_wall_impacts_remain(classic_controls):
    session = projectile_arena(classic_controls)
    snapshot = session.handle({"op": "step", "actions": [session.game.SHOOT]})
    assert not snapshot["effects"]
    assert len(session.game.visual_events) == 1
    assert session.game.visual_events[0].kind == "spark"
    # Put the real shot just before the boundary so the next engine frame
    # produces an actual wall impact while its muzzle cue is still alive.
    bullet = session.game.bullets[0]
    bullet.x = (session.game.level_cols - 1) * 32 - 9
    snapshot = session.handle({"op": "step", "actions": [0]})
    assert not session.game.bullets
    assert len(snapshot["effects"]) == 1
    assert snapshot["effects"][0]["kind"] == "spark"
    assert snapshot["effects"][0]["max_ttl"] == 12
    before = tuple(session.game.visual_events)
    session.snapshot()
    assert tuple(session.game.visual_events) == before
    # A fresh episode and the separate main mine must not retain filter IDs.
    session.reset(0)
    snapshot = session.handle({"op": "step", "actions": [session.game.SHOOT]})
    assert not snapshot["effects"]
    session.handle({"op": "mine", "cleared": []})
    snapshot = session.handle({"op": "step", "actions": [session.game.SHOOT]})
    assert not snapshot["effects"] and snapshot["sounds"] == ["shoot"]
