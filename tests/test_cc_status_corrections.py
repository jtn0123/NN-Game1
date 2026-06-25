"""Tests for policy-visited correction dataset helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import experiments.cc_status.corrections as corrections


class _FakeAgent:
    epsilon = 0.0

    def __init__(self):
        self.policy_net = _FakePolicyNet()

    def select_action(self, state, training=False):  # noqa: ARG002
        return 1

    def get_q_values(self, state):  # noqa: ARG002
        return np.array([0.0, 2.0, 1.0, -1.0], dtype=np.float32)


class _FakePolicyNet:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeCrystalCaves:
    TILE_SIZE = 16
    ACTION_LABELS = ["IDLE", "LEFT", "RIGHT", "JUMP"]

    def __init__(self, config, headless=True):  # noqa: ARG002
        self.state_size = 3
        self.action_size = 4
        self.initial_crystals = 1
        self.exit_unlocked = False
        self._game_index = -1
        self._step = 0

    def use_eval_levels(self, games):
        self._games = games

    def reset_eval_cursor(self):
        self._game_index = -1

    def reset(self):
        self._game_index += 1
        self._step = 0
        return np.array([self._game_index, self._step, 0.0], dtype=np.float32)

    def _info(self):
        return {
            "level": self._game_index,
            "level_name": f"fake-{self._game_index}",
            "end_reason": "running",
            "crystals_remaining": 1,
            "exit_unlocked": False,
            "progress": 0.0,
            "steps_since_progress": 100,
        }

    def _current_target(self):
        return ("crystal", 2, 2), 2 * self.TILE_SIZE

    def _player_tile(self):
        return (1, 1)

    def _player_center(self):
        return (self.TILE_SIZE, self.TILE_SIZE)

    def step(self, action):  # noqa: ARG002
        self._step += 1
        done = self._step >= 2
        info = self._info()
        if done:
            info["end_reason"] = "timeout"
        return np.array([self._game_index, self._step, 0.0], dtype=np.float32), 0.0, done, info

    def close(self):
        pass


def test_correction_trigger_reasons_marks_close_stale_and_loop_states():
    reasons = corrections.correction_trigger_reasons(
        target_distance_tiles=2.5,
        steps_since_progress=120,
        tile_visits=9,
        close_zone_distance_tiles=3.0,
        stale_steps=90,
        loop_tile_visits=8,
    )

    assert reasons == ("close_zone", "stale", "loop")
    assert corrections.correction_reason_mask(reasons) == 7


def test_correction_trigger_reasons_ignores_non_trigger_state():
    assert (
        corrections.correction_trigger_reasons(
            target_distance_tiles=8.0,
            steps_since_progress=10,
            tile_visits=1,
        )
        == ()
    )


def test_collect_policy_correction_dataset_writes_npz_and_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(corrections, "CrystalCaves", _FakeCrystalCaves)
    monkeypatch.setattr(
        corrections,
        "choose_correction_action",
        lambda game, reasons, stale_steps: (2, {"label_source": "fake_oracle"}),
    )

    summary = corrections.collect_policy_correction_dataset(
        object(),
        _FakeAgent(),
        out_dir=tmp_path,
        label="fake",
        games=2,
        max_steps=4,
        max_examples=3,
        sample_every=1,
        max_examples_per_game=2,
    )

    assert summary["dataset_version"] == corrections.CORRECTION_DATASET_VERSION
    assert summary["kept_examples"] == 3
    assert summary["candidate_states"] == 3
    assert summary["label_action_counts"] == {"RIGHT": 3}
    assert summary["policy_action_counts"] == {"LEFT": 3}
    assert summary["trigger_counts"]["close_zone"] == 3
    assert summary["candidate_trigger_counts"]["close_zone"] == 3

    states, actions = corrections.load_correction_action_dataset(
        tmp_path / "corrections" / "fake" / "correction_examples.npz"
    )

    assert states.shape == (3, 3)
    assert actions.tolist() == [2, 2, 2]
    rows = (tmp_path / "corrections" / "fake" / "correction_examples.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"policy_label_disagreement": true' in rows


def test_load_correction_action_dataset_rejects_mismatched_arrays(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        states=np.zeros((2, 3), dtype=np.float32),
        actions=np.zeros((1,), dtype=np.int64),
    )

    try:
        corrections.load_correction_action_dataset(path)
    except ValueError as exc:
        assert "matching states" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected mismatched correction arrays to fail")


def test_correction_finetune_rejects_empty_dataset(tmp_path, monkeypatch):
    path = tmp_path / "empty.npz"
    np.savez_compressed(
        path,
        states=np.zeros((0, 3), dtype=np.float32),
        actions=np.zeros((0,), dtype=np.int64),
    )
    monkeypatch.setattr(
        corrections,
        "load_selected_weight_snapshot",
        lambda checkpoint_path: {"weights": {}, "episode": 0, "source_eval": {}},
    )
    monkeypatch.setattr(
        corrections,
        "config_from_selected_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(MAX_EPISODES=0, EVAL_EVERY=0, EVAL_EPISODES=0),
    )
    monkeypatch.setattr(
        corrections,
        "prepare_trainer",
        lambda *args, **kwargs: SimpleNamespace(agent=SimpleNamespace(state_size=3, action_size=4)),
    )
    monkeypatch.setattr(corrections, "_validate_checkpoint_shape", lambda *args, **kwargs: None)

    try:
        corrections.run_correction_finetune(
            tmp_path,
            checkpoint_path=tmp_path / "checkpoint.pth",
            correction_dataset_path=path,
            episodes=1,
            seed=0,
            eval_games=1,
            train_eval_games=0,
            eval_every=0,
            log_every=1,
            report_seconds=1.0,
            heartbeat_seconds=0.0,
            vec_envs=1,
            save_checkpoints=False,
            correction_action_weight=0.02,
            correction_action_margin=0.6,
            correction_action_batch_size=2,
        )
    except ValueError as exc:
        assert "at least one correction transition" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected empty correction fine-tune dataset to fail")
