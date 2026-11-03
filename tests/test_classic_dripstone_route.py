"""An optional upper route varies Dripstone's long chain ascents."""

import pytest

from src.unity_bridge.human_controls import HumanControls
from src.unity_bridge.session import CaveSession


def test_dripstone_upper_branch_earns_reward_and_returns_by_existing_chain():
    game = CaveSession(level=1).game
    # Regional checkpoint, with every authored actor/trap and normal damage on.
    # Access from spawn and a complete cave win are separate acceptance work.
    game.player_x, game.player_y = 19 * 32 + 4, 7 * 32 - game.PLAYER_HEIGHT
    game.vx = game.vy = 0.0
    game.grounded = True
    initial_crystals = len(game.crystals)
    assert initial_crystals == 30

    def walk(col):
        for _ in range(100):
            delta = col * 32 + 4 - game.player_x
            if abs(delta) < 3:
                return
            game.step_human(HumanControls(1 if delta > 0 else -1, False, False, False))
        pytest.fail("branch walk did not reach its return checkpoint")

    def hop(col):
        game.step(game.RIGHT_JUMP if game.player_x < col * 32 + 4 else game.LEFT_JUMP)
        for _ in range(90):
            delta = col * 32 + 4 - game.player_x
            game.step(game.IDLE if abs(delta) < 3 else game.RIGHT if delta > 0 else game.LEFT)
            if game.grounded or game._is_on_ladder():
                return
        pytest.fail("branch jump has no landing")

    hop(21)
    assert game._player_rect().bottom == 5 * 32
    assert (21, 4) not in game.treasures
    hop(18)
    assert game._player_rect().bottom == 4 * 32
    walk(15)
    for _ in range(80):
        # Leave a little clearance when stepping sideways off the chain.
        if game._player_rect().bottom >= 7 * 32 - 2:
            break
        game.step(game.INTERACT)
    else:
        pytest.fail("existing chain did not return to the lower shelf")
    walk(19)
    assert game._player_rect().bottom == 7 * 32
    assert game.health == 3 and not game.game_over
    assert len(game.enemies) == 6 and len(game.crystals) <= initial_crystals
