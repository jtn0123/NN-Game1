"""Tests for application lifecycle helpers in main.py."""

from types import SimpleNamespace

import numpy as np
import pytest

import main


class FakeDashboard:
    def __init__(self):
        self.logs = []

    def log(self, message, level="info", data=None):
        self.logs.append((message, level, data))


def test_game_app_save_and_quit_requests_loop_shutdown(monkeypatch):
    """Visual save-and-quit should stop the loop without hard-exiting the process."""
    app = main.GameApp.__new__(main.GameApp)
    app.config = SimpleNamespace(GAME_NAME="breakout")
    app.web_dashboard = FakeDashboard()
    app.running = True
    app._save_model = lambda *args, **kwargs: True
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    app._save_and_quit()

    assert app.running is False
    assert any(
        "Shutting down" in message for message, _level, _data in app.web_dashboard.logs
    )


def test_headless_save_and_quit_requests_loop_shutdown(monkeypatch):
    """Headless save-and-quit should stop training loops without os._exit()."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(GAME_NAME="breakout")
    trainer.web_dashboard = FakeDashboard()
    trainer.running = True
    trainer._save_model = lambda *args, **kwargs: True
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    trainer._save_and_quit()

    assert trainer.running is False
    assert any(
        "Shutting down" in message
        for message, _level, _data in trainer.web_dashboard.logs
    )


@pytest.mark.parametrize("runtime_cls", [main.GameApp, main.HeadlessTrainer])
def test_runtime_nn_visualization_uses_shared_snapshot_builder(
    runtime_cls, monkeypatch
):
    """Both app modes should emit the same shared NN snapshot contract."""
    emitted = []

    class FakeWebDashboard:
        def emit_nn_visualization(self, **payload):
            emitted.append(payload)

    snapshot = SimpleNamespace(
        layer_info=[{"name": "Input", "neurons": 2, "type": "input"}],
        activations={"layer_0": [0.1, 0.2]},
        q_values=[0.3, 0.4],
        weights=[[[0.5, 0.6]]],
        action_labels=["LEFT", "RIGHT"],
        input_state=[1.0, 0.0],
        analysis_activations={"layer_0": [0.1, 0.2]},
        analysis_weights=[[[0.5, 0.6]]],
    )
    builder_calls = []

    def fake_build_nn_snapshot(agent, game, state):
        builder_calls.append((agent, game, state.copy()))
        return snapshot

    monkeypatch.setattr(main, "build_nn_snapshot", fake_build_nn_snapshot)

    runtime = runtime_cls.__new__(runtime_cls)
    runtime.web_dashboard = FakeWebDashboard()
    runtime.agent = SimpleNamespace(steps=42)
    runtime.game = SimpleNamespace()
    state = np.array([1.0, 0.0], dtype=np.float32)

    runtime._emit_nn_visualization(state, selected_action=1)

    assert len(builder_calls) == 1
    assert builder_calls[0][0] is runtime.agent
    assert builder_calls[0][1] is runtime.game
    assert builder_calls[0][2].tolist() == [1.0, 0.0]
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
