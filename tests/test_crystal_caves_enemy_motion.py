"""Enemy-motion perception (CRYSTAL_CAVES_ENEMY_MOTION) is field-of-view limited.

The feature block exposes [present, dx, dy, vx, is_flyer] for the nearest enemies
INSIDE the perception window only. These tests pin the three contract points: the
state grows by exactly ENEMY_MOTION_FEATURES; a visible enemy's motion (position
sign + velocity sign + kind) is reported; an enemy OUTSIDE the window contributes
nothing (no map-wide radar).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_entities import CaveSpec, Enemy  # noqa: E402

# One flat corridor, no authored enemies — tests inject enemies directly.
_FLAT = CaveSpec(
    name="motion-test",
    layout=(
        "########################################",
        "#......................................#",
        "#P..............................*....E.#",
        "########################################",
    ),
    background=(0, 0, 0),
    accent=(255, 255, 255),
)


def _game(enemy_motion: bool) -> CrystalCaves:
    cfg = Config()
    cfg.CRYSTAL_CAVES_ENEMY_MOTION = enemy_motion
    game = CrystalCaves(cfg, headless=True)
    game.CAVES = (_FLAT,)
    game.CAVE_DRESSING = {0: ()}
    game._eval_caves = (_FLAT,)
    game._randomize_levels = False
    game.reset()
    return game


def _motion_block(game: CrystalCaves) -> list:
    """The enemy-motion slice of the current state (after base meta + compass)."""
    state = game.get_state()
    start = (
        game.WINDOW_ROWS * game.WINDOW_COLS
        + game.GLOBAL_MAP_ROWS * game.GLOBAL_MAP_COLS
        + game.BASE_METADATA_SIZE
        + game._geo_compass_size
    )
    return list(state[start : start + game.ENEMY_MOTION_FEATURES])


def test_state_size_grows_by_feature_block() -> None:
    off = _game(enemy_motion=False)
    on = _game(enemy_motion=True)
    assert on.state_size == off.state_size + CrystalCaves.ENEMY_MOTION_FEATURES


def test_visible_enemy_motion_is_reported() -> None:
    game = _game(enemy_motion=True)
    game.enemies = [
        # 3 tiles right of the player, walking LEFT (toward the player)
        Enemy(x=game.player_x + 3 * game.TILE_SIZE, y=game.player_y, vx=-1.1, kind="crawler")
    ]
    block = _motion_block(game)
    present, dx, dy, vx, is_flyer = block[:5]
    assert present == 1.0
    assert dx > 0.5, "enemy is to the RIGHT of the player"
    assert vx < 0.5, "enemy is moving LEFT (toward the player)"
    assert is_flyer == 0.0
    # only one enemy: the second slot is an empty pad
    assert block[5] == 0.0


def test_offscreen_enemy_is_invisible() -> None:
    game = _game(enemy_motion=True)
    far_x = game.player_x + (game.WINDOW_COLS // 2 + 4) * game.TILE_SIZE
    game.enemies = [Enemy(x=far_x, y=game.player_y, vx=1.6, kind="flyer")]
    block = _motion_block(game)
    assert block == [0.0, 0.5, 0.5, 0.5, 0.0] * CrystalCaves.ENEMY_MOTION_MAX_TRACKED, (
        "an enemy outside the perception window must contribute nothing"
    )


def test_nearest_visible_enemies_win_the_slots() -> None:
    game = _game(enemy_motion=True)
    ts = game.TILE_SIZE
    game.enemies = [
        Enemy(x=game.player_x + 6 * ts, y=game.player_y, vx=1.1, kind="crawler"),
        Enemy(x=game.player_x + 2 * ts, y=game.player_y, vx=1.6, kind="flyer"),
        Enemy(x=game.player_x + 4 * ts, y=game.player_y, vx=-1.1, kind="crawler"),
        Enemy(x=game.player_x + 8 * ts, y=game.player_y, vx=-1.6, kind="flyer"),
    ]
    block = _motion_block(game)
    # 3 tracked slots, all present; nearest (the flyer at 2 tiles) first
    assert [block[0], block[5], block[10]] == [1.0, 1.0, 1.0]
    assert block[4] == 1.0, "nearest enemy is the flyer"
    # slots are ordered by distance: dx increases across slots
    assert block[1] < block[6] < block[11]
