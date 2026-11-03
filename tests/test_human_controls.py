"""Human button combinations must preserve physics and legacy demo contracts."""

import json
import socket
import threading

import numpy as np
import pytest

from config import Config
from src.game.crystal_caves_entities import CaveSpec
from src.unity_bridge.server import BridgeServer
from src.unity_bridge.session import CaveSession


def controls(move=0, jump=False, shoot=False, interact=False):
    return {"move": move, "jump": jump, "shoot": shoot, "interact": interact}


def arena(record_dir=None, ladders=True, history=False):
    config = Config(GAME_NAME="crystal_caves", CRYSTAL_CAVES_IMPORTED=True)
    config.CRYSTAL_CAVES_HISTORY_STATE = history
    session = CaveSession(record_dir=record_dir, config=config)
    rows = ["#" * 40] + ["#" + "." * 38 + "#" for _ in range(12)]
    floor = list("#" + "." * 38 + "#")
    floor[3], floor[30], floor[36] = "P", "*", "E"
    rows += ["".join(floor), "#" * 40, "#" * 40]
    for row in range(5, 14) if ladders else ():
        line = list(rows[row])
        line[5] = "H"
        rows[row] = "".join(line)
    session.game.CAVES = (CaveSpec("human controls", tuple(rows), (0, 0, 0), (0, 255, 0)),)
    session.game.use_eval_levels(1)
    session.reset(0)
    session.handle({"op": "step", "actions": [0] * 4})
    assert session.game.grounded
    return session


def finish_cave(session):
    game = session.game
    game.crystals.clear()
    game.exit_unlocked = True
    col, row = game.exit_pos
    game.player_x, game.player_y = col * 32 + 4, row * 32 + 2
    return session.handle({"op": "human_step", "controls": [controls()]})


def test_jump_and_fire_start_on_the_same_grounded_frame():
    session = arena()
    x, y = session.game.player_x, session.game.player_y
    result = session.handle(
        {"op": "human_step", "controls": [controls(move=1, jump=True, shoot=True)]}
    )
    assert result["player"]["x"] > x and result["player"]["y"] < y
    assert result["player"]["vy"] < 0 and not result["player"]["grounded"]
    assert result["ammo"] == 4 and len(session.game.bullets) == 1
    assert "jump" in result["sounds"] and "shoot" in result["sounds"]
    assert result["action"] == -1


def test_firing_does_not_block_a_jump_pressed_later():
    session = arena()
    session.handle({"op": "human_step", "controls": [controls(shoot=True)]})
    before = session.game.player_y
    result = session.handle({"op": "human_step", "controls": [controls(jump=True, shoot=True)]})
    assert result["player"]["y"] < before and "jump" in result["sounds"]
    assert result["ammo"] == 4  # Cooldown remains authoritative.


def test_interact_and_fire_do_not_cancel_horizontal_input():
    session = arena()
    game = session.game
    game.switches.add(game._player_tile())
    game.switch_color[game._player_tile()] = "red"
    before = game.player_x
    result = session.handle(
        {"op": "human_step", "controls": [controls(move=1, shoot=True, interact=True)]}
    )
    assert result["player"]["x"] > before
    assert game.used_switches and "red" in game.open_colors
    assert result["ammo"] == 4 and "shoot" in result["sounds"]


@pytest.mark.parametrize("realm", ["cave", "mine"])
def test_neutral_climbing_holds_and_buttons_climb_and_descend(realm):
    session = arena()
    if realm == "mine":
        session.handle({"op": "mine"})
    game = session.game
    col = 4 if realm == "mine" else 5
    game.player_x, game.player_y, game.vy = col * 32 + 4, 8 * 32 + 2, 0
    assert game._is_on_ladder()
    before = game.player_y
    session.handle({"op": "human_step", "controls": [controls()] * 8})
    assert game.player_y == before and game.vy == 0
    session.handle({"op": "human_step", "controls": [controls(jump=True, shoot=True)]})
    assert game.player_y < before and game.ammo == 4
    up = game.player_y
    session.handle({"op": "human_step", "controls": [controls(interact=True)]})
    assert game.player_y > up


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [controls()] * 9,
        [controls(), controls(move=True)],
        [controls(), controls(move=2)],
        [controls(), controls(move=0.5)],
        [controls(), controls(jump=1)],
        [controls(), controls(shoot=None)],
        [controls(), {"move": 0}],
        [controls(), {**controls(), "unknown": False}],
        [controls(), None],
    ],
)
def test_human_batch_is_fully_validated_before_any_mutation(invalid):
    session = arena()
    before = session.snapshot()
    state = session.game.get_state().copy()
    with pytest.raises(ValueError):
        session.handle({"op": "human_step", "controls": invalid})
    assert session.snapshot() == before
    np.testing.assert_array_equal(session.game.get_state(), state)


def test_human_controls_are_rejected_in_ai_mode_without_advancing():
    session = CaveSession(policy=lambda state: np.arange(10, dtype=np.float32))
    session.handle({"op": "mode", "mode": "ai"})
    before = session.snapshot()
    with pytest.raises(ValueError, match="human mode"):
        session.handle({"op": "human_step", "controls": [controls()]})
    assert session.snapshot() == before


def test_compound_buttons_keep_clear_credit_but_never_save_a_false_legacy_demo(tmp_path):
    session = arena(str(tmp_path))
    result = session.handle({"op": "human_step", "controls": [controls(jump=True, shoot=True)]})
    assert result["human_only"] and not result["recording"]
    won = finish_cave(session)
    assert won["won"] and won["human_only"] and won["demos_saved"] == 0
    assert not list(tmp_path.iterdir())
    reset = session.handle({"op": "reset", "level": 0})
    assert reset["human_only"] and reset["recording"]


def test_replayable_human_buttons_keep_the_existing_demo_format(tmp_path):
    session = arena(str(tmp_path))
    result = session.handle({"op": "human_step", "controls": [controls(move=1)]})
    assert result["recording"] and result["human_only"]
    won = finish_cave(session)
    assert won["demos_saved"] == 1
    saved = json.loads(next(tmp_path.iterdir()).read_text())
    assert saved["actions"] == [0, 0, 0, 0, 2, 0]
    assert saved["steps"] == won["steps"] and saved["won"]


def test_cave_ladder_hold_is_not_mislabeled_as_a_replayable_idle_demo(tmp_path):
    session = arena(str(tmp_path))
    session.game.player_x, session.game.player_y = 5 * 32 + 4, 8 * 32 + 2
    result = session.handle({"op": "human_step", "controls": [controls()]})
    assert result["human_only"] and not result["recording"]
    assert finish_cave(session)["demos_saved"] == 0


def test_ai_takeover_still_disqualifies_cave_clear_credit_after_human_return():
    session = CaveSession(policy=lambda state: np.arange(10, dtype=np.float32))
    session.handle({"op": "mode", "mode": "ai"})
    session.handle({"op": "mode", "mode": "human"})
    result = session.handle({"op": "human_step", "controls": [controls(jump=True)]})
    assert not result["human_only"]
    assert session.handle({"op": "reset", "level": 0})["human_only"]


@pytest.mark.parametrize("history", [False, True])
def test_replayable_controls_match_legacy_actions_frame_by_frame(history):
    human, legacy = arena(ladders=False, history=history), arena(ladders=False, history=history)
    sequence = [controls(move=1)] * 10 + [controls(move=1, jump=True)]
    sequence += [controls(move=1)] * 5 + [controls(move=-1, shoot=True)] + [controls()] * 8
    for buttons in sequence:
        result = human.handle({"op": "human_step", "controls": [buttons]})
        legacy_result = legacy.handle({"op": "step", "actions": [result["action"]]})
        np.testing.assert_array_equal(human.game.get_state(), legacy.game.get_state())
        for field in ("player", "ammo", "health", "score", "last_reward", "total_reward"):
            assert result[field] == legacy_result[field]
    assert human.game.action_size == legacy.game.action_size == 10


def test_compound_input_history_stays_truthful_with_the_existing_checkpoint_shape():
    session = arena(ladders=False, history=True)
    before_size = session.game.state_size
    result = session.handle(
        {"op": "human_step", "controls": [controls(move=1, jump=True, shoot=True)]}
    )
    assert session.game._history_metadata()[-7:-1] == [0, 0, 1, 1, 1, 0]
    assert result["state_size"] == before_size and session.game.action_size == 10
    # A later legacy AI frame retains the truthful compound human history.
    session.policy = lambda state: np.arange(10, dtype=np.float32)
    session.handle({"op": "mode", "mode": "ai"})
    session.handle({"op": "step", "actions": [0]})
    assert session.game._history_metadata()[-14:-8] == [0, 0, 1, 1, 1, 0]
    assert session.game._history_metadata()[-7:-1] == [0, 0, 0, 0, 0, 1]


def test_training_profile_rejects_independent_controls_without_mutation():
    session = CaveSession(classic_controls=False)
    before = session.snapshot()
    with pytest.raises(ValueError, match="classic human profile"):
        session.handle({"op": "human_step", "controls": [controls(jump=True, shoot=True)]})
    assert session.snapshot() == before
    assert session.game.action_size == 10


def test_legacy_steps_after_human_controls_keep_their_original_cave_climb_behavior():
    session = arena()
    game = session.game
    game.player_x, game.player_y = 5 * 32 + 4, 8 * 32 + 2
    session.handle({"op": "human_step", "controls": [controls()]})
    before = game.player_y
    session.handle({"op": "step", "actions": [0]})
    assert game.player_y == before + game.LADDER_DESCEND_SPEED


def test_compound_frames_after_a_terminal_frame_do_not_disqualify_saved_demo(tmp_path):
    session = arena(str(tmp_path))
    game = session.game
    game.crystals.clear()
    game.exit_unlocked = True
    col, row = game.exit_pos
    game.player_x, game.player_y = col * 32 + 4, row * 32 + 2
    result = session.handle(
        {"op": "human_step", "controls": [controls(), controls(jump=True, shoot=True)]}
    )
    assert result["won"] and result["human_only"] and result["recording"]
    assert result["demos_saved"] == 1
    assert json.loads(next(tmp_path.iterdir()).read_text())["actions"] == [0] * 5


def test_loopback_transport_accepts_compound_controls_and_recovers_from_a_bad_batch():
    with BridgeServer(("127.0.0.1", 0), lambda: arena(ladders=False)) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with socket.create_connection(server.server_address, timeout=3) as connection:
                with connection.makefile("rwb") as stream:

                    def send(request):
                        stream.write((json.dumps(request) + "\n").encode())
                        stream.flush()
                        return json.loads(stream.readline())

                    before = send({"op": "snapshot"})
                    bad = send(
                        {
                            "op": "human_step",
                            "controls": [controls(move=1), controls(jump="true")],
                        }
                    )
                    assert "booleans" in bad["error"]
                    assert send({"op": "snapshot"}) == before
                    result = send(
                        {
                            "op": "human_step",
                            "controls": [controls(move=1, jump=True, shoot=True)],
                            "actions": None,
                            "level": 0,
                            "mode": "",
                        }
                    )
                    assert result["steps"] == before["steps"] + 1
                    assert result["player"]["vy"] < 0 and result["ammo"] == 4
                    assert result["human_only"] and result["action"] == -1
        finally:
            server.shutdown()
            worker.join(timeout=3)
