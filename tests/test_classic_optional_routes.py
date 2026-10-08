"""Bonus routes use real jumps and safe returns with authored threats active."""

import pytest

from src.unity_bridge.session import CaveSession


def walk(game, col):
    for _ in range(120):
        delta = col * 32 + 4 - game.player_x
        if abs(delta) < 3:
            return
        game.step(1 if delta < 0 else 2)
    pytest.fail("route walk did not reach its checkpoint")


def hop(game, col, delay=0):
    delta = col * 32 + 4 - game.player_x
    game.step(3 if delay else 4 if delta < 0 else 5)
    for frame in range(1, 91):
        delta = col * 32 + 4 - game.player_x
        game.step(0 if frame < delay or abs(delta) < 3 else 1 if delta < 0 else 2)
        if game.grounded or game._is_on_ladder():
            return
    pytest.fail("route jump did not reach a landing")


def fixture(level):
    game = CaveSession(level=level).game
    game.player_x, game.player_y = 12 * 32 + 4, 18 * 32 - game.PLAYER_HEIGHT
    game.vx = game.vy = 0.0
    game.grounded = True
    return game


def test_twin_vaults_shortcut_earns_chest_and_returns_to_central_chain():
    game = fixture(8)
    hop(game, 15)
    assert game._player_rect().bottom == 17 * 32
    assert (15, 16) not in game.treasures
    hop(game, 19)
    assert game._is_on_ladder() and game._player_tile() == (18, 14)
    assert game.health == 3 and not game.game_over


def test_sunken_grotto_branch_has_trap_wait_reward_and_safe_return():
    game = fixture(13)
    # Step into the visible falling-trap column, retreat, then wait for it to break.
    for action, frames in ((2, 4), (1, 4), (0, 60)):
        for _ in range(frames):
            game.step(action)
    trap = next(t for t in game.stalactites if (t.col, t.row) == (13, 15))
    assert not trap.alive and game.health == 3
    hop(game, 15)
    assert game._player_rect().bottom == 17 * 32
    walk(game, 16)
    hop(game, 17)
    assert game._player_rect().bottom == 16 * 32
    walk(game, 17)
    hop(game, 16, delay=19)
    assert game._player_rect().bottom == 14 * 32
    assert (17, 15) not in game.treasures
    walk(game, 9)
    for _ in range(50):
        game.step(game.INTERACT)
    assert game._player_tile() == (9, 16)
    assert game.health == 3 and not game.game_over
