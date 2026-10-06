"""Reference ceiling traps: visibility, trigger columns, falling, reset and isolation."""

from src.unity_bridge.session import CaveSession


def trap_session():
    session = CaveSession()
    session.reset(4)
    game = session.game
    game.enemies.clear()
    game.hazards.clear()
    trap = next(trap for trap in game.stalactites if (trap.col, trap.row) == (12, 17))
    return session, game, trap


def test_ceiling_trap_waits_then_drops_in_discrete_poses_with_one_source_cue():
    session, game, trap = trap_session()
    start = trap.y
    for _ in range(20):
        game._check_player_danger()
    assert trap.y == start and not trap.falling
    game.player_x, game.player_y = 389, 674
    for _ in range(3):
        game._check_player_danger()
    assert trap.falling and trap.y == start
    game._check_player_danger()
    assert trap.y == start + 16
    assert game.audio.events.count("stalactite") == 1
    entity = next(e for e in session.snapshot()["entities"] if e["id"] == "stalactite_0")
    assert entity["sprite"] == "stalactite" and entity["y"] == trap.y
    game.player_x -= 96
    for _ in range(4):
        game._check_player_danger()
    assert trap.y == start + 32  # Once released it keeps falling after Mylo leaves.


def test_detection_stops_at_stone_and_falling_trap_cannot_hit_through_it():
    _, game, trap = trap_session()
    game.grid[19][12] = "#"
    game.player_x, game.player_y = 389, 674
    for _ in range(60):
        game._check_player_danger()
    assert not trap.falling and game.health == 3
    game.player_y = 576
    game._check_player_danger()
    assert trap.falling
    game.player_y = 674
    for _ in range(30):
        game._check_player_danger()
    assert not trap.alive and game.health == 3


def test_fall_sweeps_the_visible_tip_and_breaks_once_on_the_floor():
    session, game, trap = trap_session()
    game.player_x, game.player_y = 389, 674
    for _ in range(60):
        game._check_player_danger()
    assert game.health == 2 and not trap.alive
    assert not any(e["id"] == "stalactite_0" for e in session.snapshot()["entities"])
    session.reset(4)
    trap = session.game.stalactites[0]
    assert trap.alive and not trap.falling and trap.y == 17 * 32
    assert not hasattr(CaveSession(classic_controls=False).game, "stalactites")


def test_ceiling_trap_route_uses_real_input_and_leaves_a_safe_way_to_dodge():
    session = CaveSession()
    session.reset(4)
    game = session.game
    trap = game.stalactites[0]
    for step in range(134):
        game.step(0 if step < 10 else 5 if step < 25 else 2)
    assert trap.falling and game.health == 3 and game._player_rect().right > 384
    for _ in range(20):
        game.step(1)
    for _ in range(40):
        game.step(0)
    assert not trap.alive and game.health == 3 and not game.game_over
    assert not game._rect_collides_solid(game._player_rect())


def test_authored_traps_preserve_objects_and_warn_observations_where_they_are():
    session = CaveSession()
    count = 0
    for level in range(16):
        session.reset(level)
        game = session.game
        for trap in game.stalactites:
            count += 1
            assert game.level.layout[trap.row - 1][trap.col] == "#"
            assert game.grid[trap.row][trap.col] == "."
            assert game._tile_code(trap.col, trap.row) == game.TILE_CODES[game.SPIKE]
            trap.falling = True
            game.player_x = 100
            for _ in range(8):
                game._check_player_danger()
            row = trap.rect.centery // 32
            assert game._code_grid()[row, trap.col] == game.TILE_CODES[game.SPIKE]
            assert game._tile_code(trap.col, row) == game.TILE_CODES[game.SPIKE]
            assert not game._rect_collides_solid(trap.rect)
            assert trap.rect.width == 12
    assert count == 6
