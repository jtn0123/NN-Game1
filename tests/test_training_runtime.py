import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from config import Config
from src.ai.agent import Agent
from src.app.training_runtime import (
    NNSnapshot,
    build_nn_snapshot,
    emit_nn_snapshot_to_dashboard,
    is_new_best_score,
    request_save_and_stop,
    resolve_model_path,
    should_emit_episode_metrics,
)


@pytest.mark.parametrize(
    ("score", "best_score", "expected"),
    [(11, 10, True), (10, 10, False), (9, 10, False)],
)
def test_is_new_best_score_compares_against_previous_best(score, best_score, expected):
    assert is_new_best_score(score, best_score) is expected


@pytest.mark.parametrize(
    ("episode", "is_new_best", "expected"),
    [(1, False, True), (10, False, True), (11, True, True), (15, False, True), (16, False, False)],
)
def test_should_emit_episode_metrics_matches_dashboard_throttle(episode, is_new_best, expected):
    assert should_emit_episode_metrics(episode, is_new_best) is expected


def test_resolve_model_path_uses_latest_compatible_checkpoint(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    older = model_dir / "older.pth"
    newer = model_dir / "newer.pth"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    older_mtime = 1_700_000_000
    newer_mtime = older_mtime + 10
    older.touch()
    newer.touch()
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    def inspect_model(path, **kwargs):
        return {"state_size": 4, "action_size": 2}

    resolved = resolve_model_path(
        explicit_path=None,
        state_size=4,
        action_size=2,
        config=config,
        inspect_model=inspect_model,
    )

    assert resolved == str(newer)


def test_resolve_model_path_skips_newer_incompatible_checkpoint(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    older = model_dir / "older.pth"
    newer = model_dir / "newer.pth"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_010, 1_700_000_010))

    def inspect_model(path, **kwargs):
        if os.path.basename(path) == "newer.pth":
            return {"state_size": 99, "action_size": 2}
        return {"state_size": 4, "action_size": 2}

    resolved = resolve_model_path(
        explicit_path=None,
        state_size=4,
        action_size=2,
        config=config,
        inspect_model=inspect_model,
    )

    assert resolved == str(older)


def test_resolve_model_path_handles_explicit_checkpoint_compatibility(tmp_path, capsys):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "explicit.pth"
    checkpoint.write_bytes(b"model")

    resolved = resolve_model_path(
        explicit_path=str(checkpoint),
        state_size=4,
        action_size=2,
        config=config,
        inspect_model=lambda _path, **_kwargs: {"state_size": 4, "action_size": 2},
    )

    assert resolved == str(checkpoint)

    rejected = resolve_model_path(
        explicit_path=str(checkpoint),
        state_size=4,
        action_size=2,
        config=config,
        inspect_model=lambda _path, **_kwargs: {"state_size": 99, "action_size": 2},
    )

    assert rejected is None
    assert "Specified model incompatible" in capsys.readouterr().out


@pytest.mark.parametrize("save_success", [True, False])
def test_request_save_and_stop_logs_result(monkeypatch, capsys, save_success):
    logs = []
    running = []
    saved = []
    dashboard = SimpleNamespace(log=lambda message, level: logs.append((message, level)))

    monkeypatch.setattr("src.app.training_runtime.time.sleep", lambda _seconds: None)

    request_save_and_stop(
        game_name="breakout",
        save_model=lambda filename, reason: saved.append((filename, reason)) or save_success,
        set_running=running.append,
        dashboard=dashboard,
    )

    assert saved == [("breakout_final.pth", "shutdown")]
    assert running == [False]
    assert logs[0][1] == "warning"
    output = capsys.readouterr().out
    if save_success:
        assert "Model saved. Exiting" in output
        assert logs[-1][1] == "success"
    else:
        assert "Save may have failed. Exiting" in output
        assert logs[-1][1] == "warning"


def test_emit_nn_snapshot_to_dashboard_forwards_full_payload():
    emitted = []
    dashboard = SimpleNamespace(
        emit_nn_visualization=lambda **payload: emitted.append(payload),
    )
    snapshot = NNSnapshot(
        layer_info=[{"name": "input", "neurons": 2, "type": "input"}],
        activations={"layer_0": [0.1, 0.2]},
        q_values=[1.0, 2.0],
        weights=[[[0.1, 0.2]]],
        action_labels=["LEFT", "RIGHT"],
        input_state=[0.0, 1.0],
        analysis_activations={"layer_0": [0.1, 0.2, 0.3]},
        analysis_weights=[[[0.1, 0.2, 0.3]]],
    )

    emit_nn_snapshot_to_dashboard(dashboard, snapshot, selected_action=1, step=42)

    assert emitted == [
        {
            "layer_info": snapshot.layer_info,
            "activations": snapshot.activations,
            "q_values": snapshot.q_values,
            "selected_action": 1,
            "weights": snapshot.weights,
            "step": 42,
            "action_labels": snapshot.action_labels,
            "input_state": snapshot.input_state,
            "analysis_activations": snapshot.analysis_activations,
            "analysis_weights": snapshot.analysis_weights,
        }
    ]


def test_build_nn_snapshot_creates_sampled_and_full_payloads():
    config = Config()
    config.HIDDEN_LAYERS = [32, 16]
    config.USE_DUELING = False
    config.USE_N_STEP_RETURNS = False
    agent = Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config,
    )
    game = SimpleNamespace(get_action_labels=lambda: ["LEFT", "RIGHT"])
    state = np.zeros(config.STATE_SIZE, dtype=np.float32)

    snapshot = build_nn_snapshot(agent, game, state)

    assert snapshot.action_labels == ["LEFT", "RIGHT"]
    assert snapshot.input_state == [0.0] * config.STATE_SIZE
    assert len(snapshot.q_values) == config.ACTION_SIZE
    assert snapshot.weights
    assert snapshot.analysis_weights
    assert len(snapshot.analysis_weights[0][0]) == config.STATE_SIZE
    assert len(snapshot.weights[0]) <= 15


@pytest.mark.parametrize(
    ("use_dueling", "use_noisy"),
    [(False, False), (True, False), (True, True)],
)
def test_build_nn_snapshot_contract_for_network_variants(use_dueling, use_noisy):
    """Dashboard NN snapshots should stay structurally valid across network variants."""
    config = Config()
    config.HIDDEN_LAYERS = [24, 12]
    config.USE_DUELING = use_dueling
    config.USE_NOISY_NETWORKS = use_noisy
    config.USE_N_STEP_RETURNS = False
    config.USE_PRIORITIZED_REPLAY = False
    agent = Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config,
    )
    game = SimpleNamespace(get_action_labels=lambda: ["LEFT", "STAY", "RIGHT"])
    state = np.linspace(0, 1, config.STATE_SIZE, dtype=np.float32)

    snapshot = build_nn_snapshot(agent, game, state)

    assert snapshot.layer_info[0]["type"] == "input"
    assert snapshot.layer_info[-1]["type"] == "output"
    assert len(snapshot.q_values) == config.ACTION_SIZE
    assert len(snapshot.action_labels) == config.ACTION_SIZE
    assert len(snapshot.input_state) == config.STATE_SIZE
    assert np.isfinite(snapshot.q_values).all()
    assert snapshot.activations
    assert snapshot.analysis_activations
    assert snapshot.weights
    assert snapshot.analysis_weights
    assert all(len(layer) <= 15 for layer in snapshot.weights)
    if use_dueling:
        layer_types = {layer["type"] for layer in snapshot.layer_info}
        assert "value_stream" in layer_types
        assert "advantage_stream" in layer_types


def test_build_nn_snapshot_smoke_performance_budget():
    """A small NN snapshot should stay comfortably below an interactive budget."""
    config = Config()
    config.HIDDEN_LAYERS = [32, 16]
    config.USE_DUELING = False
    config.USE_NOISY_NETWORKS = False
    config.USE_N_STEP_RETURNS = False
    config.USE_PRIORITIZED_REPLAY = False
    agent = Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config,
    )
    game = SimpleNamespace(get_action_labels=lambda: ["LEFT", "STAY", "RIGHT"])
    state = np.zeros(config.STATE_SIZE, dtype=np.float32)

    start = time.perf_counter()
    for _ in range(5):
        build_nn_snapshot(agent, game, state)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
