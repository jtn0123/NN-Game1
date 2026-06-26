"""Tests for policy-visited correction dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import experiments.cc_status.corrections as corrections
from experiments.cc_status.contact_label_audit import (  # noqa: E402
    run_contact_label_audit,
    run_contact_label_filter,
)
from experiments.cc_status.correction_calibration import (  # noqa: E402
    combine_correction_action_datasets,
    parse_correction_dataset_paths,
    stratified_calibration_split,
)


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


def test_collect_policy_correction_dataset_can_roll_out_custom_selector(tmp_path, monkeypatch):
    monkeypatch.setattr(corrections, "CrystalCaves", _FakeCrystalCaves)
    monkeypatch.setattr(
        corrections,
        "choose_correction_action",
        lambda game, reasons, stale_steps: (3, {"label_source": "fake_oracle"}),
    )

    def selector(agent, state, game, info, step, action_labels):  # noqa: ANN001, ANN202
        del agent, state, game, info, step, action_labels
        return 2, {"source": "contact_action_head", "confidence": 0.91}

    summary = corrections.collect_policy_correction_dataset(
        object(),
        _FakeAgent(),
        out_dir=tmp_path,
        label="selector",
        games=1,
        max_steps=2,
        max_examples=1,
        sample_every=1,
        rollout_action_selector=selector,
    )

    assert summary["rollout_action_selector"] is True
    assert summary["policy_action_counts"] == {"RIGHT": 1}
    rows = (tmp_path / "corrections" / "selector" / "correction_examples.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"policy_action_source": "contact_action_head"' in rows
    assert '"confidence": 0.91' in rows


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


def test_parse_correction_dataset_paths_splits_comma_list():
    paths = parse_correction_dataset_paths("a.npz, b.npz")

    assert [path.name for path in paths] == ["a.npz", "b.npz"]


def test_combine_correction_action_datasets_keeps_source_indices(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez_compressed(
        first,
        states=np.ones((2, 3), dtype=np.float32),
        actions=np.array([1, 2], dtype=np.int64),
    )
    np.savez_compressed(
        second,
        states=np.zeros((3, 3), dtype=np.float32),
        actions=np.array([2, 3, 3], dtype=np.int64),
    )

    combined = combine_correction_action_datasets((first, second))

    assert combined.states.shape == (5, 3)
    assert combined.actions.tolist() == [1, 2, 2, 3, 3]
    assert combined.source_dataset_indices.tolist() == [0, 0, 1, 1, 1]
    assert combined.source_example_indices.tolist() == [0, 1, 0, 1, 2]


def test_combine_correction_action_datasets_accepts_single_filtered_source(tmp_path):
    path = tmp_path / "filtered.npz"
    np.savez_compressed(
        path,
        states=np.ones((2, 3), dtype=np.float32),
        actions=np.array([1, 2], dtype=np.int64),
    )

    combined = combine_correction_action_datasets((path,))

    assert combined.states.shape == (2, 3)
    assert combined.source_dataset_indices.tolist() == [0, 0]
    assert combined.source_example_indices.tolist() == [0, 1]


def test_stratified_calibration_split_preserves_train_examples_per_class():
    actions = np.array([1] * 6 + [2] * 4 + [3] * 2, dtype=np.int64)

    train_indices, calibration_indices = stratified_calibration_split(
        actions,
        calibration_frac=0.25,
        seed=0,
    )

    assert len(set(train_indices.tolist()) & set(calibration_indices.tolist())) == 0
    assert len(train_indices) + len(calibration_indices) == len(actions)
    assert set(actions[calibration_indices].tolist()) == {1, 2, 3}
    assert set(actions[train_indices].tolist()) == {1, 2, 3}


def test_contact_label_audit_reports_conflicts_and_flips(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "correction_examples.npz"
    second_path = second_dir / "correction_examples.npz"
    shared_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.savez_compressed(
        first_path,
        states=np.stack([shared_state, np.array([1.0, 2.0, 4.0], dtype=np.float32)]),
        actions=np.array([1, 4], dtype=np.int64),
    )
    np.savez_compressed(
        second_path,
        states=np.stack([shared_state, np.array([5.0, 6.0, 7.0], dtype=np.float32)]),
        actions=np.array([4, 2], dtype=np.int64),
    )
    first_rows = [
        {
            "game_index": 0,
            "step": 10,
            "label_action_label": "LEFT",
            "policy_action_label": "IDLE",
            "tile": [5, 5],
            "objective": {
                "player_tile": [5, 5],
                "target_tile": [4, 5],
                "target_distance_tiles": 1.0,
                "target_kind": "crystal",
            },
        },
        {
            "game_index": 0,
            "step": 11,
            "label_action_label": "LEFT_JUMP",
            "policy_action_label": "IDLE",
            "tile": [5, 5],
            "objective": {
                "player_tile": [5, 5],
                "target_tile": [4, 5],
                "target_distance_tiles": 1.0,
                "target_kind": "crystal",
            },
        },
    ]
    second_rows = [
        {
            "game_index": 0,
            "step": 5,
            "label_action_label": "LEFT_JUMP",
            "policy_action_label": "IDLE",
            "tile": [5, 5],
            "objective": {
                "player_tile": [5, 5],
                "target_tile": [4, 5],
                "target_distance_tiles": 1.0,
                "target_kind": "crystal",
            },
        },
        {
            "game_index": 1,
            "step": 8,
            "label_action_label": "RIGHT",
            "policy_action_label": "IDLE",
            "tile": [3, 5],
            "objective": {
                "player_tile": [3, 5],
                "target_tile": [2, 5],
                "target_distance_tiles": 1.5,
                "target_kind": "crystal",
            },
        },
    ]
    (first_dir / "correction_examples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in first_rows) + "\n",
        encoding="utf-8",
    )
    (second_dir / "correction_examples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in second_rows) + "\n",
        encoding="utf-8",
    )

    run = run_contact_label_audit(
        tmp_path / "out",
        correction_dataset_paths=(first_path, second_path),
        state_round_decimals=3,
        adjacent_step_window=2,
        top_groups=5,
        label="audit",
    )
    audit = run["contact_label_audit"]

    assert audit["state_conflicts"]["conflict_groups"] == 1
    assert audit["semantic_ambiguity"]["ambiguous_groups"] >= 1
    assert audit["adjacent_label_flips"]["flips"] == 1
    assert audit["direction_alignment"]["mismatch_counts"] == {"RIGHT": 1}
    assert Path(run["correction_dataset"]["states_path"]).exists()


def test_contact_label_filter_keeps_stable_majority_nonflip_rows(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "correction_examples.npz"
    second_path = second_dir / "correction_examples.npz"
    np.savez_compressed(
        first_path,
        states=np.ones((2, 3), dtype=np.float32),
        actions=np.array([1, 4], dtype=np.int64),
    )
    np.savez_compressed(
        second_path,
        states=np.zeros((2, 3), dtype=np.float32),
        actions=np.array([4, 2], dtype=np.int64),
    )
    shared_objective = {
        "player_tile": [5, 5],
        "target_tile": [4, 5],
        "target_distance_tiles": 1.0,
        "target_kind": "crystal",
    }
    first_rows = [
        {
            "game_index": 0,
            "step": 10,
            "label_action_label": "LEFT",
            "policy_action_label": "IDLE",
            "tile": [5, 5],
            "objective": shared_objective,
        },
        {
            "game_index": 0,
            "step": 11,
            "label_action_label": "LEFT_JUMP",
            "policy_action_label": "IDLE",
            "tile": [5, 5],
            "objective": shared_objective,
        },
    ]
    second_rows = [
        {
            "game_index": 0,
            "step": 5,
            "label_action_label": "LEFT_JUMP",
            "policy_action_label": "IDLE",
            "tile": [5, 5],
            "objective": shared_objective,
        },
        {
            "game_index": 1,
            "step": 8,
            "label_action_label": "RIGHT",
            "policy_action_label": "IDLE",
            "tile": [3, 5],
            "objective": {
                "player_tile": [3, 5],
                "target_tile": [2, 5],
                "target_distance_tiles": 1.5,
                "target_kind": "crystal",
            },
        },
    ]
    (first_dir / "correction_examples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in first_rows) + "\n",
        encoding="utf-8",
    )
    (second_dir / "correction_examples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in second_rows) + "\n",
        encoding="utf-8",
    )

    run = run_contact_label_filter(
        tmp_path / "out",
        correction_dataset_paths=(first_path, second_path),
        semantic_majority_threshold=0.6,
        adjacent_step_window=2,
        label="filter",
    )

    summary = run["contact_label_filter"]
    assert summary["kept_examples"] == 2
    assert summary["drop_reason_counts"]["adjacent_label_flip"] == 2
    assert run["correction_dataset"]["label_action_counts"] == {"LEFT_JUMP": 1, "RIGHT": 1}
    assert Path(run["correction_dataset"]["states_path"]).exists()


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
