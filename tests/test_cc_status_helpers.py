"""Focused tests for Crystal Caves status-session helper contracts."""

import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.cc_status.demo_collect as demo_collect  # noqa: E402
import experiments.cc_status.evals as cc_evals  # noqa: E402
import experiments.cc_status.runs_demo as runs_demo  # noqa: E402


class _DemoRunAgent:
    state_size = 4
    action_size = 3


class _DemoRunTrainer:
    def __init__(self) -> None:
        self.agent = _DemoRunAgent()
        self.current_episode = 9


def test_demo_source_snapshot_training_selects_best_training_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    trainer = _DemoRunTrainer()
    initial_snapshot = {
        "episode": 0,
        "source_eval": {"win_rate": 0.1, "mean_crystal_frac": 0.2},
    }
    initial_weights = {"initial": {"w": torch.tensor([1.0])}}
    training_snapshot = {
        "episode": 5,
        "source_eval": {
            "win_rate": 0.2,
            "mean_crystal_frac": 0.4,
            "mean_depth_frac": 0.3,
            "mean_score": 12.0,
        },
    }
    training_weights = {"trained": {"w": torch.tensor([2.0])}}

    def fake_training(*args, **kwargs):
        assert kwargs["label"] == "demo"
        return (
            1.5,
            [{"episode": 5, "source_eval": training_snapshot["source_eval"]}],
            training_snapshot,
            training_weights,
        )

    monkeypatch.setattr(runs_demo, "run_training_with_source_snapshots", fake_training)

    result = runs_demo._run_source_snapshot_training(
        trainer,
        object(),
        run_dir=tmp_path,
        label="demo",
        total_episodes=10,
        heartbeat_seconds=0.0,
        source_eval_every=5,
        eval_games=2,
        initial_snapshot=initial_snapshot,
        initial_weights=initial_weights,
    )

    assert result.train_seconds == 1.5
    assert result.best_snapshot is training_snapshot
    assert result.best_weights is training_weights
    assert result.eval_payload is training_snapshot["source_eval"]
    assert result.history == [{"episode": 5, "source_eval": training_snapshot["source_eval"]}]


def test_selected_checkpoint_eval_saves_evidence_and_restores_final_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    trainer = _DemoRunTrainer()
    config = SimpleNamespace(EVAL_MAX_STEPS=123, GAME_MODEL_DIR=str(tmp_path / "models"))
    best_weights = {"best": {"w": torch.tensor([2.0])}}
    final_weights = {"final": {"w": torch.tensor([3.0])}}
    loaded_snapshots: list[object] = []
    checkpoint_path = tmp_path / "selected.pth"

    monkeypatch.setattr(runs_demo, "capture_weight_snapshot", lambda agent: final_weights)
    monkeypatch.setattr(
        runs_demo,
        "load_weight_snapshot",
        lambda agent, snapshot: loaded_snapshots.append(snapshot),
    )
    monkeypatch.setattr(runs_demo, "config_snapshot", lambda cfg: {"seed": 0})
    monkeypatch.setattr(
        runs_demo,
        "save_selected_weight_snapshot",
        lambda path, **kwargs: str(path),
    )
    monkeypatch.setattr(
        runs_demo,
        "final_eval",
        lambda *args, **kwargs: {"label": kwargs["label"], "games": kwargs["games"]},
    )
    monkeypatch.setattr(
        runs_demo,
        "trace_heldout_failures",
        lambda *args, **kwargs: {"label": kwargs["label"], "games": kwargs["games"]},
    )
    monkeypatch.setattr(
        runs_demo,
        "first_objective_near_miss_eval",
        lambda *args, **kwargs: {
            "label": kwargs["label"],
            "episode": kwargs["episode"],
            "max_steps": kwargs["max_steps"],
        },
    )

    result = runs_demo._evaluate_selected_checkpoint(
        config,
        trainer,
        run_dir=tmp_path,
        label="demo",
        best_snapshot={"episode": 7, "source_eval": {"win_rate": 0.5}},
        best_weights=best_weights,
        selected_eval_games=3,
        trace_games=2,
        trace_max_steps=50,
        trace_sample_every=5,
        trace_tail_steps=10,
        checkpoint_path=checkpoint_path,
    )

    assert loaded_snapshots[0] is best_weights
    assert loaded_snapshots[1] is final_weights
    assert result.checkpoint_path == str(checkpoint_path)
    assert result.eval_payload == {"label": "demo_selected_ep7", "games": 3}
    assert result.diagnostics == {"label": "demo_selected_ep7_heldout", "games": 2}
    assert result.near_miss_eval == {
        "label": "demo_selected_ep7",
        "episode": 7,
        "max_steps": 123,
    }


def test_selected_checkpoint_eval_restores_final_weights_when_eval_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    trainer = _DemoRunTrainer()
    config = SimpleNamespace(EVAL_MAX_STEPS=123, GAME_MODEL_DIR=str(tmp_path / "models"))
    best_weights = {"best": {"w": torch.tensor([2.0])}}
    final_weights = {"final": {"w": torch.tensor([3.0])}}
    loaded_snapshots: list[object] = []

    monkeypatch.setattr(runs_demo, "capture_weight_snapshot", lambda agent: final_weights)
    monkeypatch.setattr(
        runs_demo,
        "load_weight_snapshot",
        lambda agent, snapshot: loaded_snapshots.append(snapshot),
    )

    def fail_eval(*args, **kwargs):
        raise RuntimeError("eval failed")

    monkeypatch.setattr(runs_demo, "final_eval", fail_eval)

    with pytest.raises(RuntimeError, match="eval failed"):
        runs_demo._evaluate_selected_checkpoint(
            config,
            trainer,
            run_dir=tmp_path,
            label="demo",
            best_snapshot={"episode": 7, "source_eval": {"win_rate": 0.5}},
            best_weights=best_weights,
            selected_eval_games=3,
            trace_games=2,
            trace_max_steps=50,
            trace_sample_every=5,
            trace_tail_steps=10,
        )

    assert loaded_snapshots[0] is best_weights
    assert loaded_snapshots[1] is final_weights


def test_eval_agent_state_helpers_restore_epsilon_and_training_mode():
    class FakePolicyNet:
        def __init__(self) -> None:
            self.training = True
            self.eval_calls = 0
            self.train_calls = 0

        def eval(self) -> None:
            self.eval_calls += 1
            self.training = False

        def train(self) -> None:
            self.train_calls += 1
            self.training = True

    class FakeAgent:
        def __init__(self) -> None:
            self.epsilon = 0.42
            self.policy_net = FakePolicyNet()

    agent = FakeAgent()

    state = cc_evals._enter_greedy_agent_eval(agent)
    assert agent.epsilon == 0.0
    assert agent.policy_net.eval_calls == 1
    assert agent.policy_net.training is False

    cc_evals._restore_greedy_agent_eval(agent, state)
    assert agent.epsilon == 0.42
    assert agent.policy_net.train_calls == 1
    assert agent.policy_net.training is True


def test_eval_end_reason_helper_normalizes_running_rows():
    assert (
        cc_evals._resolved_end_reason({"end_reason": "running", "won": True}, steps=2, max_steps=5)
        == "won"
    )
    assert (
        cc_evals._resolved_end_reason(
            {"end_reason": "running", "won": False},
            steps=5,
            max_steps=5,
        )
        == "timeout"
    )
    assert (
        cc_evals._resolved_end_reason(
            {"end_reason": "running", "won": False},
            steps=3,
            max_steps=5,
        )
        == "ended"
    )
    assert (
        cc_evals._resolved_end_reason({"end_reason": "stalled"}, steps=1, max_steps=5) == "stalled"
    )


def test_route_demo_arg_validation_rejects_bad_controller_variant():
    with pytest.raises(ValueError, match="unknown route demo controller variants"):
        demo_collect._validate_route_demo_collection_args(
            max_levels=1,
            max_steps=10,
            close_zone_distance_tiles=3.0,
            controller_variants=("direct", "bad"),
            oracle_close_zone_stride=1,
            oracle_close_zone_max_per_trajectory=1,
        )


def test_route_demo_summary_uses_zero_when_failed_distances_are_absent():
    summary = demo_collect._route_demo_summary(
        levels=2,
        max_steps=10,
        close_zone_distance_tiles=3.0,
        controller_variants=("direct",),
        oracle_close_zone_labels=False,
        oracle_close_zone_stride=4,
        oracle_close_zone_max_per_trajectory=8,
        trajectories=[],
        close_zone_trajectories=[],
        oracle_close_zone_trajectories=[],
        kept_rows=[],
        rows=[
            {
                "won": False,
                "variant": "direct",
                "failure_modes": ["timeout_navigation"],
                "min_target_distance_tiles": None,
                "target_distance_best_delta_tiles": 0.0,
            }
        ],
    )

    assert summary["wins"] == 0
    assert summary["failure_mode_counts"] == {"timeout_navigation": 1}
    assert summary["mean_failed_min_target_distance_tiles"] == 0.0
    assert summary["mean_failed_best_delta_tiles"] == 0.0


def test_route_demo_oracle_label_helper_records_relabels(monkeypatch: pytest.MonkeyPatch):
    class FakeGame:
        ACTION_LABELS = ["IDLE", "RIGHT"]

    def fake_oracle_action(game, *, stale_steps):
        return 1, {"score": 12.5}

    monkeypatch.setattr(demo_collect, "close_zone_oracle_action", fake_oracle_action)

    trajectory = []
    action_counts: Counter[str] = Counter()
    oracle_scores = []
    relabeled = demo_collect._maybe_oracle_close_zone_label(
        FakeGame(),
        state=np.array([1.0, 2.0], dtype=np.float32),
        action=0,
        stale_steps=3,
        should_label=True,
        trajectory=trajectory,
        action_counts=action_counts,
        oracle_scores=oracle_scores,
    )

    assert relabeled == 1
    assert action_counts == Counter({"RIGHT": 1})
    assert oracle_scores == [12.5]
    assert len(trajectory) == 1
    assert trajectory[0][1] == 1
