"""The human demo recorder captures exactly the actions an episode fed to
``game.step``, survives mid-episode resets, and writes replay-compatible JSON."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.app.demo_recorder import HumanDemoRecorder  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402


class _FakeGame:
    """Minimal stand-in exposing the attributes the recorder reads."""

    class _Level:
        name = "Ore Shaft"  # matches HANDCRAFTED_LEVELS[0]

    def __init__(self):
        self.steps = 0
        self.level = self._Level()
        self.level_index = 0
        self.won = False


def test_records_actions_and_saves_on_done(tmp_path):
    rec = HumanDemoRecorder(tmp_path)
    game = _FakeGame()
    for i, action in enumerate([2, 2, 5, 0]):
        game.steps = i + 1
        done = i == 3
        info = {"won": False, "end_reason": "killed"} if done else {}
        rec.after_step(game, action, done, info)
    assert len(rec.saved) == 1
    data = json.loads(rec.saved[0].read_text())
    assert data["actions"] == [2, 2, 5, 0]
    assert data["won"] is False
    assert data["end_reason"] == "killed"
    assert data["level_index"] == 0  # resolved by name against HANDCRAFTED_LEVELS
    assert data["level_name"] == "Ore Shaft"
    assert "lost" in rec.saved[0].name


def test_mid_episode_reset_discards_aborted_attempt(tmp_path):
    rec = HumanDemoRecorder(tmp_path)
    game = _FakeGame()
    # three steps into an episode, then the player hits R (game.steps rewinds to 1)
    for i, action in enumerate([1, 1, 1]):
        game.steps = i + 1
        rec.after_step(game, action, False, {})
    game.steps = 1
    rec.after_step(game, 2, False, {})
    game.steps = 2
    rec.after_step(game, 2, True, {"won": True, "end_reason": "won"})
    data = json.loads(rec.saved[0].read_text())
    assert data["actions"] == [2, 2]  # the aborted [1, 1, 1] prefix is gone
    assert data["won"] is True
    assert "won" in rec.saved[0].name


def test_unknown_level_gets_no_index(tmp_path):
    rec = HumanDemoRecorder(tmp_path)
    game = _FakeGame()
    game.level.name = "Not A Real Level"
    game.steps = 1
    rec.after_step(game, 0, True, {"won": False, "end_reason": "timeout"})
    data = json.loads(rec.saved[0].read_text())
    assert data["level_index"] is None
    assert rec.saved[0].name.startswith("Lxx_")


def test_live_engine_episode_is_replayable(tmp_path):
    """Record a short live episode on a pinned hand-crafted level, then replay the
    stored actions on a fresh engine and confirm the trajectories match."""
    spec = HANDCRAFTED_LEVELS[0]
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True

    def pinned_game() -> CrystalCaves:
        game = CrystalCaves(cfg, headless=True)
        game.CAVES = (spec,)
        game._eval_caves = (spec,)
        game._randomize_levels = False
        game.use_eval_levels(1)
        game.reset_eval_cursor()
        game.reset()
        return game

    rec = HumanDemoRecorder(tmp_path)
    game = pinned_game()
    actions = [2, 2, 5, 5, 2, 1, 3, 2] * 3
    end_x = None
    for action in actions:
        _s, _r, done, info = game.step(action)
        rec.after_step(game, action, done, info)
        end_x = game.player_x
        if done:
            break
    assert rec._actions  # episode still running: buffer holds the prefix
    assert [int(a) for a in rec._actions] == actions[: len(rec._actions)]

    replay = pinned_game()
    for action in rec._actions:
        replay.step(int(action))
    assert replay.player_x == end_x  # deterministic engine: identical trajectory
