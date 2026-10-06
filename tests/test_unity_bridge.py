"""The Unity transport must preserve the original game's transitions."""

import json
import socket
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config import Config
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_entities import CaveSpec
from src.unity_bridge.classic_audio import (
    SAMPLE_RATE,
    SOURCE,
    TONE_RATE,
    SpeakerSound,
    read_programs,
    render,
    sound_bank,
)
from src.unity_bridge.classic_game import ClassicCaves
from src.unity_bridge.playfeel import apply_classic_controls
from src.unity_bridge.server import BridgeServer
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_fidelity import service_pipe_cells
from src.unity_bridge.visual_scenery import placement_cells
from src.unity_bridge.visuals import TILE


def reference_game(level=0, classic_controls=True):
    game_type = ClassicCaves if classic_controls else CrystalCaves
    game = game_type(Config(GAME_NAME="crystal_caves", CRYSTAL_CAVES_IMPORTED=True), headless=True)
    if classic_controls:
        apply_classic_controls(game)
    game.use_eval_levels(len(game.CAVES))
    game._eval_cursor = level
    game.seed(0)
    game.reset()
    return game


def open_arena_session(classic_controls=True):
    session = CaveSession(classic_controls=classic_controls)
    rows = ["#" * 40] + ["#" + "." * 38 + "#" for _ in range(12)]
    floor = list("#" + "." * 38 + "#")
    floor[3], floor[30], floor[36] = "P", "*", "E"
    rows += ["".join(floor), "#" * 40, "#" * 40]
    session.game.CAVES = (CaveSpec("control test", tuple(rows), (0, 0, 0), (0, 255, 0)),)
    session.game.use_eval_levels(1)
    session.reset(0)
    # Spawn is one pixel above the floor. Settle through real engine steps
    # before measuring a grounded jump or the distance covered in one second.
    session.handle({"op": "step", "actions": [0] * 4})
    assert session.game.grounded
    return session


@pytest.mark.parametrize("boundary", ["time", "idle"])
def test_classic_expedition_has_no_training_cutoff(boundary):
    for classic in (False, True):
        session = open_arena_session(classic)
        if boundary == "time":
            session.game.steps = CrystalCaves.MAX_STEPS - 1
        else:
            session.game.steps_since_progress = CrystalCaves.MAX_STEPS_WITHOUT_PROGRESS - 1
        result = session.handle({"op": "step", "actions": [0]})
        assert result["done"] is (not classic)
        assert result["training_limits"] is (not classic)


def test_classic_walk_has_a_deliberate_pace_and_stops_on_release():
    session = open_arena_session()
    start = session.game.player_x
    for _ in range(60):
        session.handle({"op": "step", "actions": [2]})
    distance = session.game.player_x - start
    assert 135 <= distance <= 145
    stopped = session.game.player_x
    session.handle({"op": "step", "actions": [0] * 8})
    assert session.game.player_x == pytest.approx(stopped)


def test_classic_jump_is_lower_and_has_a_slower_complete_arc():
    session = open_arena_session()
    start_y = session.game.player_y
    positions = []
    for frame in range(100):
        result = session.handle({"op": "step", "actions": [3 if frame == 0 else 0]})
        positions.append(result["player"]["y"])
        if frame > 0 and result["player"]["grounded"]:
            break
    assert 76 <= start_y - min(positions) <= 82
    assert 60 <= len(positions) <= 72
    assert positions[-1] == pytest.approx(start_y, abs=1)


def test_last_crystal_opens_exit_and_restart_clears_the_ready_state():
    session = open_arena_session()
    game = session.game
    col, row = next(iter(game.crystals))
    game.player_x, game.player_y = col * 32 + 4, row * 32 + 2
    result = session.handle({"op": "step", "actions": [0]})
    assert result["crystals"] == 0 and result["exit_unlocked"] and not result["done"]
    assert "gem" in result["sounds"] and "win" not in result["sounds"]
    col, row = game.exit_pos
    game.player_x, game.player_y = col * 32 + 4, row * 32 + 2
    won = session.handle({"op": "step", "actions": [0]})
    assert won["won"] and won["sounds"] == ["win"]
    again = session.handle({"op": "reset", "level": 0})
    assert again["crystals"] == 1 and not again["exit_unlocked"]


@pytest.mark.parametrize("level", [3, 9])
def test_lower_jump_can_reach_repaired_cave_objectives(level, monkeypatch):
    from experiments.cc_status import level_reach as reach

    classic = CaveSession().game
    from src.unity_bridge.classic_layouts import CLASSIC_LEVELS

    for field, value in {
        "MOVE_SPEED": classic.MOVE_SPEED,
        "AIR_SPEED": classic.AIR_SPEED,
        "JUMP_SPEED": classic.JUMP_SPEED,
        "GRAVITY": classic.GRAVITY,
        "MAX_FALL": classic.MAX_FALL_SPEED,
        "FRICTION": classic.FRICTION,
    }.items():
        monkeypatch.setattr(reach, field, value)
    macros = [
        [
            (frames + 170 if frames >= reach.MAX_FRAMES - 13 else frames, direction, jump)
            for frames, direction, jump in macro
        ]
        for macro in reach._MACROS
    ]
    monkeypatch.setattr(reach, "MAX_FRAMES", 240)
    monkeypatch.setattr(reach, "_MACROS", macros)
    before = reach.analyze_gated(CLASSIC_LEVELS[level].layout)
    assert not before["gated_winnable"]
    after = reach.analyze_gated(classic.CAVES[level].layout)
    assert after["gated_winnable"] and not after["search_truncated"]
    for old_row, new_row in zip(CLASSIC_LEVELS[level].layout, classic.CAVES[level].layout):
        assert len(old_row) == len(new_row)
        for old, new in zip(old_row, new_row):
            assert old == new or old in ".#" and new == "H"


@pytest.mark.parametrize("classic_controls", [True, False])
def test_actions_match_authoritative_simulation_frame_by_frame(classic_controls):
    session = CaveSession(classic_controls=classic_controls)
    reference = reference_game(classic_controls=classic_controls)
    actions = [2] * 18 + [5] * 24 + [1] * 8 + [6] * 14 + [9] * 3
    total = 0.0
    for action in actions:
        state, reward, done, _ = reference.step(action)
        result = session.handle({"op": "step", "actions": [action]})
        total += reward
        np.testing.assert_array_equal(session.game.get_state(), state)
        assert result["total_reward"] == pytest.approx(total)
        assert result["done"] is done
        assert result["player"]["x"] == reference.player_x
        assert result["player"]["y"] == reference.player_y


@pytest.mark.parametrize("actions", [[2, 10], [2, -1], [2, 1.2], [2, True], [], [0] * 9])
def test_invalid_batch_never_partially_advances(actions):
    session = CaveSession()
    before = session.game.get_state().copy()
    with pytest.raises(ValueError):
        session.handle({"op": "step", "actions": actions})
    assert session.game.steps == 0
    np.testing.assert_array_equal(session.game.get_state(), before)


def test_level_switch_and_restart_are_pinned_and_disable_training_curriculum():
    session = CaveSession()
    for level in (3, 0, 15, 3):
        result = session.handle({"op": "reset", "level": level})
        reference = reference_game(level)
        assert result["level"] == level
        assert result["level_name"] == reference.level.name
        np.testing.assert_array_equal(session.game.get_state(), reference.get_state())
        session.handle({"op": "step", "actions": [2] * 8})
    for invalid in (-1, 16, True, 1.5):
        with pytest.raises(ValueError):
            session.handle({"op": "reset", "level": invalid})
    assert session.game.steps == 8


def test_ai_requires_an_explicit_loaded_policy():
    session = CaveSession()
    with pytest.raises(ValueError, match="checkpoint"):
        session.handle({"op": "mode", "mode": "ai"})
    assert session.mode == "human"
    assert not session.snapshot()["ai_available"]


def test_ai_uses_policy_values_and_takeover_uses_human_actions():
    values = np.arange(10, dtype=np.float32)
    session = CaveSession(policy=lambda state: values, policy_name="test policy")
    assert session.snapshot()["human_only"]
    reference = reference_game()
    session.handle({"op": "mode", "mode": "ai"})
    result = session.handle({"op": "step", "actions": [2, 2]})
    for _ in range(2):
        reference.step(9)
    np.testing.assert_array_equal(session.game.get_state(), reference.get_state())
    assert result["action"] == 9
    assert result["q_values"] == values.tolist()
    assert result["mode"] == "ai"
    assert not result["human_only"]
    session.handle({"op": "mode", "mode": "human"})
    session.handle({"op": "step", "actions": [2]})
    reference.step(2)
    np.testing.assert_array_equal(session.game.get_state(), reference.get_state())
    assert not session.snapshot()["human_only"]
    assert session.handle({"op": "reset", "level": 0})["human_only"]


def test_effects_and_audio_do_not_change_the_original_transition():
    session = CaveSession()
    reference = reference_game()
    state, reward, _, _ = reference.step(6)
    result = session.handle({"op": "step", "actions": [6]})
    np.testing.assert_array_equal(session.game.get_state(), state)
    assert result["last_reward"] == reward
    assert "shoot" in result["sounds"]
    assert any(event.kind == "spark" for event in session.game.visual_events)
    assert not result["effects"]
    assert session.handle({"op": "snapshot"})["sounds"] == []


def test_classic_crystal_uses_the_original_tone_sequence_and_priority():
    programs = read_programs()
    assert len(programs) == 36
    crystal = programs[7]
    assert crystal.frequencies == (
        2381,
        2381,
        2381,
        2381,
        0,
        0,
        0,
        0,
        2786,
        0,
        0,
        2894,
        2894,
        2894,
        2894,
    )
    assert (crystal.priority, crystal.vibrate) == (10, 2)
    assert len(render(crystal)) / SAMPLE_RATE == pytest.approx(0.10715, abs=0.0001)


def test_speaker_renderer_preserves_pauses_and_rapid_gating():
    samples = render(SpeakerSound((1000, 1000, 0, 1000), 5, 2))
    first_end = round(SAMPLE_RATE / TONE_RATE)
    assert np.any(samples[:first_end])
    assert np.all(samples[first_end:] == 0)
    assert set(np.unique(samples)) == {-0.23, 0, 0.23}


def test_truncated_original_sound_file_is_rejected(tmp_path):
    (tmp_path / "CC1-1.SND").write_bytes((SOURCE / "CC1-1.SND").read_bytes()[:-1])
    with pytest.raises(ValueError, match="610-byte"):
        read_programs(tmp_path)


def test_exported_pcm_matches_classic_programs_and_rejected_music_is_removed():
    resources = Path(__file__).resolve().parents[1] / "unity/Assets/Resources"
    clips, metadata = sound_bank()
    catalog = json.loads((resources / "ClassicAudio.json").read_text())
    assert {item["name"]: item["priority"] for item in catalog["sounds"]} == {
        name: info["priority"] for name, info in metadata.items()
    }
    for name, samples in clips.items():
        with wave.open(str(resources / "Audio" / f"{name}.wav")) as clip:
            assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (
                1,
                2,
                SAMPLE_RATE,
            )
            pcm = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(pcm, (samples * 32767).astype("<i2"))
    assert not (resources / "Audio/music.wav").exists()


@pytest.mark.parametrize(
    "kind, expected", [("ammo", "ammo"), ("treasure", "treasure"), ("freeze", "freeze")]
)
def test_classic_pickups_have_distinct_cues(kind, expected):
    session = open_arena_session()
    game = session.game
    tile = (3, 13)
    if kind == "ammo":
        game.ammo_pickups.add(tile)
    elif kind == "treasure":
        game.treasures.add(tile)
    else:
        game.powerups[tile] = game.FREEZE_POWER
    result = session.handle({"op": "step", "actions": [0]})
    assert result["sounds"] == [expected]


def test_powered_shot_uses_its_own_classic_cue():
    session = open_arena_session()
    session.game.super_timer = 100
    result = session.handle({"op": "step", "actions": [6]})
    assert "power_shoot" in result["sounds"] and "shoot" not in result["sounds"]


def test_every_cave_has_correctly_sized_terrain_and_complete_sprite_assets():
    resources = Path(__file__).resolve().parents[1] / "unity/Assets/Resources"
    catalog = json.loads((resources / "CaveCatalog.json").read_text())["caves"]
    session = CaveSession()
    for level in range(len(session.game.CAVES)):
        snapshot = session.handle({"op": "reset", "level": level})
        assert {key: catalog[level][key] for key in ("name", "crystals", "cols", "rows")} == {
            "name": snapshot["level_name"],
            "crystals": snapshot["initial_crystals"],
            "cols": snapshot["cols"],
            "rows": snapshot["rows"],
        }
        with Image.open(resources / f"Terrain/level_{level}.png") as image:
            assert image.size == (snapshot["cols"] * TILE, snapshot["rows"] * TILE)
        for entity in snapshot["entities"]:
            assert (resources / "Sprites" / f"{entity['sprite']}.png").is_file()
        reference = reference_game(level)
        result = session.handle({"op": "step", "actions": [2] * 8})
        reward = 0.0
        for _ in range(8):
            state, step_reward, _, _ = reference.step(2)
            reward += step_reward
        np.testing.assert_array_equal(session.game.get_state(), state)
        assert result["total_reward"] == pytest.approx(reward)


def test_socket_protocol_recovers_after_bad_json_without_mutating_game():
    with BridgeServer(("127.0.0.1", 0), lambda: CaveSession()) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with socket.create_connection(server.server_address, timeout=3) as connection:
                stream = connection.makefile("rwb")
                for packet, expected in (
                    (b"{broken}\n", "error"),
                    (b'{"op":"snapshot"}\n', "player"),
                    (b'{"op":"step","actions":[2,2,2]}\n', "player"),
                ):
                    stream.write(packet)
                    stream.flush()
                    response = json.loads(stream.readline())
                    assert response["protocol"] == 1
                    assert expected in response
                assert response["steps"] == 3
                stream.close()
        finally:
            server.shutdown()
            worker.join(timeout=3)


def test_new_terrain_and_dressing_preserve_the_authoritative_grid():
    resources = Path(__file__).resolve().parents[1] / "unity/Assets/Resources"
    catalog = json.loads((resources / "CaveCatalog.json").read_text())["caves"]
    session = CaveSession()
    live_scenery = set()
    for level, info in enumerate(catalog):
        snapshot = session.handle({"op": "reset", "level": level})
        with Image.open(resources / f"Terrain/level_{level}.png") as image:
            alpha = np.asarray(image)[:, :, 3]
        for row, cells in enumerate(snapshot["layout"]):
            for col, symbol in enumerate(cells):
                tile = alpha[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE]
                if symbol == "#":
                    assert np.all(tile == 255), (level, col, row)
                elif symbol not in ("H", "^", "~"):
                    assert not np.any(tile), (level, col, row)
        with Image.open(resources / f"Environment/dressing_{level}.png") as image:
            assert image.size == (snapshot["cols"] * TILE, snapshot["rows"] * TILE)
            decoration = np.asarray(image)
        for placement in info["decorations"]:
            name = placement["sprite"]
            if not name.startswith(
                ("vine_", "silver_column_", "flared_column_", "purple_mushroom")
            ) and name not in (
                "danger_sign",
                "reverse_gravity_sign",
                "barrel",
                "ventilation_grille",
            ):
                continue
            live_scenery.add(
                name
                if name in ("purple_mushroom", "ventilation_grille")
                else name.rsplit("_", 1)[0]
            )
            footprint = placement_cells(placement)
            # Test the full multi-tile silhouette, not just its upper-left cell.
            for col, row in footprint:
                assert all(session.game.level.layout[row][col + dx] == "." for dx in (-1, 0, 1))
            assert footprint.isdisjoint(service_pipe_cells(session.game.level.layout))
            for other in info["decorations"]:
                if other is not placement:
                    assert footprint.isdisjoint(placement_cells(other)), (level, name, other)
            with Image.open(resources / f"Sprites/{name}.png") as prop:
                pixels = np.asarray(prop)
                x, y = placement["col"] * TILE, placement["row"] * TILE
                width = max(col for col, _ in footprint) - placement["col"] + 1
                height = max(row for _, row in footprint) - placement["row"] + 1
                assert prop.size == (width * TILE, height * TILE)
                # Metadata must describe the art actually composited into Unity's
                # scenery layer; pipes and neighboring fixtures cannot cover it.
                assert np.array_equal(decoration[y : y + prop.height, x : x + prop.width], pixels)
        for light in info["lights"]:
            col, row = light["x"] // 32, light["y"] // 32
            assert session.game.level.layout[row][col] == "."
            assert session.game.level.layout[row - 1][col] == "#"
    assert live_scenery == {
        "vine",
        "silver_column",
        "flared_column",
        "purple_mushroom",
        "danger",
        "reverse_gravity",
        "barrel",
        "ventilation_grille",
    }
    for name in ("mylo_walk_3", "mylo_walk_4", "bat_enemy_3", "slug_enemy_1", "crystal_blue_glint"):
        assert (resources / "Sprites" / f"{name}.png").is_file()


def test_air_art_expands_into_clear_headroom_without_moving_the_gameplay_base():
    resources = Path(__file__).resolve().parents[1] / "unity/Assets/Resources/Sprites"
    session = CaveSession()
    forms = set()
    for level in range(len(session.game.CAVES)):
        session.reset(level)
        sites = session.game.air_tanks.copy()
        observation = session.game.get_state().copy()
        snapshot = session.snapshot()
        vessels = [entity for entity in snapshot["entities"] if entity["id"].startswith("air_")]
        assert len(vessels) == len(sites)
        for vessel in vessels:
            _, col, row = vessel["id"].split("_")
            col, row = int(col), int(row)
            assert (col, row) in sites
            assert vessel["x"] == col * TILE
            assert vessel["y"] + vessel["h"] == (row + 1) * TILE
            if vessel["h"] == TILE * 2:
                assert session.game.level.layout[row - 1][col] == "."
                for frame in range(2):
                    assert (resources / f"air_tank_tall_{frame}.png").is_file()
            else:
                assert vessel["h"] == TILE
                assert session.game.level.layout[row - 1][col] != "."
            with Image.open(resources / f"{vessel['sprite']}.png") as art:
                assert art.size == (vessel["w"], vessel["h"])
            forms.add(vessel["sprite"])
        assert session.game.air_tanks == sites
        np.testing.assert_array_equal(session.game.get_state(), observation)
    assert forms == {"air_tank", "air_tank_tall"}


def test_creature_identities_survive_patrols_and_tall_art_clears_the_entire_route():
    session = CaveSession()
    resources = Path(__file__).resolve().parents[1] / "unity/Assets/Resources"
    with Image.open(resources / "Sprites/dinosaur_enemy.png") as image:
        art_width, art_height = image.size
    roster = set()
    for level in range(len(session.game.CAVES)):
        initial = session.handle({"op": "reset", "level": level})
        identities = {
            entity["id"]: entity["sprite"]
            for entity in initial["entities"]
            if entity["id"].startswith("enemy_")
        }
        roster.update(identities.values())
        for _ in range(60):
            snapshot = session.handle({"op": "step", "actions": [0] * 8})
            for entity in snapshot["entities"]:
                if entity["id"] not in identities:
                    continue
                assert entity["sprite"] == identities[entity["id"]]
                if entity["sprite"] == "dinosaur_enemy":
                    # The taller picture ends at the patrol's feet. Its full
                    # head/body footprint must remain clear while it moves.
                    layout = session.game.level.layout
                    art_x = entity["x"] + (entity["w"] - art_width) / 2
                    left = int(art_x // 32)
                    right = int((art_x + art_width - 1) // 32)
                    top = int((entity["y"] + entity["h"] - art_height) // 32)
                    bottom = int((entity["y"] + entity["h"] - 1) // 32)
                    assert all(
                        layout[row][col] != "#"
                        for row in range(top, bottom + 1)
                        for col in range(left, right + 1)
                    ), (level, entity)
        again = session.handle({"op": "reset", "level": level})
        assert identities == {
            entity["id"]: entity["sprite"]
            for entity in again["entities"]
            if entity["id"].startswith("enemy_")
        }
    assert roster == {"bat_enemy", "eye_flyer", "slug_enemy", "walking_rock", "dinosaur_enemy"}


def test_training_rebalances_do_not_move_the_accepted_classic_game():
    from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS
    from src.unity_bridge.classic_layouts import CLASSIC_LEVELS

    training = CaveSession(classic_controls=False).game
    classic = CaveSession().game
    assert training.CAVES is HANDCRAFTED_LEVELS
    assert classic.CAVES is not HANDCRAFTED_LEVELS
    assert CLASSIC_LEVELS[0].layout[21][8] == "^"
    assert classic.CAVES[0].layout[21][8] == "t"
    assert CLASSIC_LEVELS[0].layout != HANDCRAFTED_LEVELS[0].layout
    # Native terrain/catalog are exported from this fixed human-game profile.
    assert sum(row.count("*") for row in classic.CAVES[0].layout) == 32
