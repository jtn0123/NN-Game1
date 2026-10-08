"""Normal-spawn Twin Vaults routes preserve the cache, gates and gun economy.

The original trace is retained below as historical input evidence. Covered-rest
and armed routes adapt it to the repaired floor. These are simulated human
buttons, not physical keyboard/controller or first-time clue-recognition proof.
"""

import pytest

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_scenery import placement_cells
from src.unity_bridge.visuals import dressing_placements

# (ticks, input): N neutral, L/R move, U jump/climb, D use/descend;
# LJ/RJ are moving jumps and LS/RS are moving shots. No state mutations.
ROUTE_RLE = (
    (54, "L"),
    (1, "N"),
    (1, "D"),
    (14, "L"),
    (1, "N"),
    (80, "R"),
    (1, "N"),
    (42, "U"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (12, "U"),
    (1, "D"),
    (1, "N"),
    (28, "R"),
    (1, "N"),
    (26, "L"),
    (1, "N"),
    (32, "U"),
    (1, "D"),
    (1, "N"),
    (14, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (12, "U"),
    (1, "D"),
    (1, "N"),
    (28, "R"),
    (1, "N"),
    (26, "L"),
    (1, "N"),
    (32, "U"),
    (1, "N"),
    (14, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (11, "U"),
    (1, "N"),
    (28, "R"),
    (1, "N"),
    (26, "L"),
    (1, "N"),
    (32, "U"),
    (1, "N"),
    (14, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (1, "U"),
    (1, "N"),
    (42, "R"),
    (1, "N"),
    (40, "L"),
    (1, "N"),
    (60, "U"),
    (1, "N"),
    (27, "L"),
    (1, "N"),
    (25, "R"),
    (1, "N"),
    (1, "U"),
    (1, "N"),
    (8, "U"),
    (55, "R"),
    (1, "N"),
    (53, "L"),
    (1, "N"),
    (1, "U"),
    (1, "N"),
    (8, "U"),
    (82, "L"),
    (1, "N"),
    (55, "L"),
    (1, "N"),
    (81, "R"),
    (1, "N"),
    (1, "RJ"),
    (49, "R"),
    (102, "D"),
    (1, "N"),
    (49, "L"),
    (1, "N"),
    (1, "LJ"),
    (64, "L"),
    (3, "N"),
    (55, "L"),
    (1, "N"),
    (1, "U"),
    (70, "N"),
    (53, "R"),
    (1, "N"),
    (1, "RJ"),
    (40, "R"),
    (14, "N"),
    (1, "RJ"),
    (50, "R"),
    (111, "D"),
    (1, "N"),
    (72, "R"),
    (1, "N"),
    (1, "D"),
    (28, "R"),
    (1, "N"),
    (1, "RJ"),
    (136, "R"),
    (1, "N"),
    (42, "U"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (42, "U"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (42, "U"),
    (1, "N"),
    (26, "L"),
    (1, "N"),
    (26, "R"),
    (1, "N"),
    (42, "U"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (255, "D"),
    (1, "N"),
    (108, "L"),
    (1, "N"),
    (1, "LJ"),
    (54, "L"),
    (7, "N"),
    (69, "L"),
    (1, "N"),
    (82, "L"),
    (1, "N"),
    (20, "L"),
    (1, "N"),
    (1, "LJ"),
    (75, "L"),
    (1, "N"),
    (14, "L"),
    (1, "N"),
    (1, "LS"),
    (20, "N"),
    (26, "L"),
    (1, "N"),
    (27, "L"),
    (1, "N"),
    (25, "R"),
    (1, "N"),
    (42, "U"),
    (1, "N"),
    (14, "R"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (21, "U"),
    (1, "N"),
    (14, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (11, "U"),
    (98, "N"),
    (11, "U"),
    (1, "D"),
    (1, "N"),
    (1, "RS"),
    (10, "N"),
    (13, "R"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (31, "U"),
    (301, "N"),
    (11, "U"),
    (1, "D"),
    (1, "N"),
    (1, "RS"),
    (10, "N"),
    (25, "R"),
    (1, "N"),
    (53, "L"),
    (1, "N"),
    (47, "R"),
    (1, "N"),
    (83, "U"),
    (1, "N"),
    (27, "R"),
    (1, "N"),
    (26, "L"),
    (1, "N"),
    (1, "U"),
    (1, "D"),
    (1, "N"),
    (27, "L"),
    (1, "N"),
    (47, "R"),
    (1, "N"),
    (73, "U"),
    (1, "N"),
    (12, "L"),
    (1, "N"),
    (12, "R"),
    (1, "N"),
    (312, "D"),
    (1, "N"),
    (27, "R"),
    (1, "N"),
    (50, "R"),
    (1, "N"),
    (1, "RJ"),
    (45, "R"),
    (1, "N"),
    (96, "R"),
    (1, "N"),
    (197, "U"),
    (1, "N"),
    (6, "R"),
)


def controls(symbol):
    return {
        "move": -1 if symbol.startswith("L") else 1 if symbol.startswith("R") else 0,
        "jump": symbol == "U" or symbol.endswith("J"),
        "shoot": symbol.endswith("S"),
        "interact": symbol == "D",
    }


def covered_wait_route(rest_delay=0):
    route = list(ROUTE_RLE)
    # The repaired top floor supports the left landing: return directly to
    # the chain, then stop descending above the lower floor's solid shoulder.
    route[207] = (27, "R")
    route[215] = (306, "D")
    route[-1] = (5, "R")
    # Descend below the overhead floor, wait on the chain-side ledge, then
    # return to the original firing height. Total duration remains 301 ticks.
    route[184:185] = [(16, "D"), (16, "R"), (242 + rest_delay, "N"), (16, "L"), (11, "U")]
    return tuple(route)


def armed_vaults_route(rest_delay=0):
    route = list(covered_wait_route(rest_delay))
    # Face left from the chain and fire five separate shots. The dinosaur
    # keeps its five health; each ordinary bullet must hit it.
    route[209:209] = [(count, symbol) for _ in range(5) for count, symbol in ((1, "S"), (20, "N"))]
    route[-1] = (4, "R")
    return tuple(route)


def replay_twin_vaults_route(route):
    session = CaveSession(level=8, seed=0)
    game = session.game
    assert game.health == 3 and game.ammo == 5
    assert len(game.enemies) == 6 and all(enemy.alive for enemy in game.enemies)
    assert len(game.crystals) == game.initial_crystals == 30 and not game.open_colors
    damage = []
    revealed = False
    shots = 0
    for count, symbol in route:
        value = controls(symbol)
        for _ in range(count):
            health = game.health
            state = session.handle({"op": "human_step", "controls": [value]})
            shots += state["sounds"].count("shoot")
            revealed |= not game.hidden_crystals and (8, 17) in game.crystals
            if game.health < health:
                damage.append((game.steps, game._last_damage_source))
    return session, revealed, shots, damage


def test_twin_vaults_covered_wait_route_preserves_an_additional_heart():
    session, revealed, shots, damage = replay_twin_vaults_route(covered_wait_route())
    game = session.game
    assert game.won and game.steps == 4971 and not game.crystals and not game.hidden_crystals
    assert revealed and game.open_colors == {"red", "blue"}
    assert game.health == 2 and game.ammo == 7 and shots == 3
    assert damage == [(4087, "enemy")]
    assert session.snapshot()["level"] == 8 and session.snapshot()["human_only"]


def test_twin_vaults_covered_rest_is_safe_while_the_bat_keeps_dropping_eggs():
    session = CaveSession(level=8, seed=0)
    game = session.game
    for count, symbol in (*ROUTE_RLE[:184], (16, "D"), (16, "R")):
        for _ in range(count):
            session.handle({"op": "human_step", "controls": [controls(symbol)]})
    assert 128 <= game.player_x <= 145 and 352 <= game.player_y <= 355
    first_egg = game._next_egg
    for _ in range(1000):
        session.handle({"op": "human_step", "controls": [controls("N")]})
        assert game.health == 3 and not game.game_over
    assert game.grounded and game.enemies[1].alive and game._next_egg > first_egg
    assert game.level.layout[10][4] == game.level.layout[12][4] == "#"
    for count, symbol in ((16, "L"), (11, "U")):
        for _ in range(count):
            session.handle({"op": "human_step", "controls": [controls(symbol)]})
    assert 98 <= game.player_x <= 104 and 319 <= game.player_y <= 322
    assert game._is_on_ladder() and game.health == 3


def test_twin_vaults_dinosaur_has_a_patrol_instead_of_one_pixel_of_ledge():
    game = CaveSession(level=8, seed=0).game
    dinosaur = game.enemies[0]
    assert dinosaur.appearance == "dinosaur_enemy" and dinosaur.health == 5
    positions = []
    for _ in range(180):
        game.step(game.IDLE)
        positions.append(dinosaur.x)
        assert not game._rect_collides_solid(dinosaur.rect)
    assert max(positions) - min(positions) >= 30
    assert game.level.layout[6][3] == "H" and dinosaur.alive


def test_twin_vaults_existing_ammo_can_be_collected_on_the_central_route():
    session = CaveSession(level=8, seed=0)
    game = session.game
    assert game.ammo_pickups == {(21, 13)}
    assert sum(row.count("A") for row in game.level.layout) == 1
    for count, symbol in ROUTE_RLE:
        for _ in range(count):
            session.handle({"op": "human_step", "controls": [controls(symbol)]})
            if game.steps == 400:
                assert game.ammo == 10 and not game.ammo_pickups and game.health == 3
                return
    raise AssertionError("the normal route did not reach the ammo encounter")


@pytest.mark.parametrize("rest_delay", [0, 4, 8, 12, 16])
def test_twin_vaults_eight_shot_route_wins_with_three_hearts_and_two_ammo(rest_delay):
    trace = armed_vaults_route(rest_delay)
    session, revealed, shots, damage = replay_twin_vaults_route(trace)
    game = session.game
    assert game.won and game.steps == sum(count for count, _ in trace) == 5075 + rest_delay
    assert game.health == 3 and game.ammo == 2 and shots == 8 and not damage
    assert revealed and not game.hidden_crystals and not game.crystals
    assert game.open_colors == {"red", "blue"} and game._solid_at(8, 14)
    assert not game.enemies[0].alive and game.enemies[0].health == 0
    assert [enemy.alive for enemy in game.enemies] == [False, False, True, False, False, True]
    assert (10, 21) in game.hazards and session.episode == 1
    assert session.snapshot()["level"] == 8 and session.snapshot()["human_only"]


def test_twin_vaults_upper_chamber_warning_has_clear_space():
    layout = CaveSession(level=8).game.level.layout
    placements = dressing_placements(layout, 8)
    warnings = [
        item
        for item in placements
        if item["sprite"] == "danger_sign" and item["row"] <= 3 and 1 <= item["col"] <= 5
    ]
    assert len(warnings) == 1
    footprint = placement_cells(warnings[0])
    assert all(layout[row][col] == "." for col, row in footprint)
    for item in placements:
        if item is not warnings[0]:
            assert footprint.isdisjoint(placement_cells(item))


def test_twin_vaults_normal_spawn_cache_both_gates_and_all_thirty_crystals():
    session = CaveSession(level=8, seed=0)
    game = session.game
    assert (game.player_x, game.player_y) == (581, 673)
    assert game.level.name == "Twin Vaults"
    assert game.initial_crystals == len(game.crystals) == 30
    assert game.health == 3 and game.ammo == 5
    assert game.hidden_crystals == {(8, 17)}
    assert len(game.enemies) == 6 and all(enemy.alive for enemy in game.enemies)
    assert (10, 21) in game.hazards
    assert not game.open_colors and session.episode == 1

    cache_revealed_before_collection = False
    shots = 0
    trace = covered_wait_route()
    for ticks, symbol in trace:
        value = controls(symbol)
        while ticks:
            count = min(ticks, 8)
            state = session.handle({"op": "human_step", "controls": [value] * count})
            shots += state["sounds"].count("shoot")
            if not game.hidden_crystals and (8, 17) in game.crystals:
                cache_revealed_before_collection = True
            assert game.health >= 1, f"route died at simulation tick {game.steps}"
            assert session.episode == 1
            ticks -= count

    assert cache_revealed_before_collection
    assert game._solid_at(8, 14)  # The cache's supporting platform remains intact.
    assert game.used_switches == {(14, 21), (24, 21)}
    assert game.open_colors == {"red", "blue"}
    assert not game._solid_at(6, 21) and not game._solid_at(33, 21)
    assert not game.crystals and not game.hidden_crystals and game.exit_unlocked
    assert game.game_over and game.won and game._end_reason == "won"
    assert game.steps == sum(ticks for ticks, _ in trace) == 4971
    assert shots == 3 and game.ammo == 7 and game._damage_taken == 1
    assert [enemy.alive for enemy in game.enemies] == [True, False, True, False, False, True]
    assert (10, 21) in game.hazards and session.mode == "human"
    final = session.snapshot()
    assert final["human_only"]
    assert final["level"] == 8 and final["level_name"] == "Twin Vaults"
