"""Tests for application lifecycle helpers in main.py."""

import importlib
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import main
from src.app import training_runtime


class FakeDashboard:
    def __init__(self):
        self.logs = []

    def log(self, message, level="info", data=None):
        self.logs.append((message, level, data))


class FakePublisher:
    def __init__(self):
        self.state = SimpleNamespace()
        self.eval_payloads = []

    def record_eval(self, **payload):
        self.eval_payloads.append(payload)


class FakeMetricsDashboard(FakeDashboard):
    def __init__(self):
        super().__init__()
        self.metric_payloads = []
        self.publisher = FakePublisher()

    def emit_metrics(self, **payload):
        self.metric_payloads.append(payload)


class FakeVectorAgent:
    def __init__(self):
        self.epsilon = 0.4
        self.steps = 25
        self.memory = [object(), object()]
        self.decay_calls = []
        self.scheduler_steps = 0

    def get_average_loss(self, _window):
        return 0.25

    def get_q_values(self, _state):
        return np.array([1.0, 3.0], dtype=np.float32)

    def decay_epsilon(self, episode):
        self.decay_calls.append(episode)

    def step_scheduler(self):
        self.scheduler_steps += 1


class FakeEvaluator:
    def __init__(self, result, *, evals_since_improvement=0, best_eval_score=0.0, plateau=False):
        self.result = result
        self.evals_since_improvement = evals_since_improvement
        self.best_eval_score = best_eval_score
        self.best_eval_win_rate = 0.0
        self.plateau = plateau
        self.evaluate_calls = []
        self.logged_results = []

    def evaluate(self, **kwargs):
        self.evaluate_calls.append(kwargs)
        return self.result

    def log_results(self, result):
        self.logged_results.append(result)

    def is_plateau(self):
        return self.plateau


def test_game_app_save_and_quit_requests_loop_shutdown(monkeypatch):
    """Visual save-and-quit should stop the loop without hard-exiting the process."""
    app = main.GameApp.__new__(main.GameApp)
    app.config = SimpleNamespace(GAME_NAME="breakout")
    app.web_dashboard = FakeDashboard()
    app.running = True
    app._save_model = lambda *args, **kwargs: True
    monkeypatch.setattr(training_runtime.time, "sleep", lambda _seconds: None)

    app._save_and_quit()

    assert app.running is False
    assert any("Shutting down" in message for message, _level, _data in app.web_dashboard.logs)


def test_headless_save_and_quit_requests_loop_shutdown(monkeypatch):
    """Headless save-and-quit should stop training loops without os._exit()."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(GAME_NAME="breakout")
    trainer.web_dashboard = FakeDashboard()
    trainer.running = True
    trainer._save_model = lambda *args, **kwargs: True
    monkeypatch.setattr(training_runtime.time, "sleep", lambda _seconds: None)

    trainer._save_and_quit()

    assert trainer.running is False
    assert any("Shutting down" in message for message, _level, _data in trainer.web_dashboard.logs)


def test_headless_vectorized_episode_completion_records_metrics_and_checkpoints():
    """Vector episode completion should update training, dashboard, and checkpoint state."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(
        GAME_NAME="crystal_caves",
        TARGET_UPDATE=10,
        SAVE_EVERY=2,
        EVAL_EVERY=0,
        LEARN_EVERY=8,
        GRADIENT_STEPS=2,
        BATCH_SIZE=128,
        CRYSTAL_CAVES_DIFFICULTY="tutorial",
        BRICK_ROWS=1,
        BRICK_COLS=1,
    )
    trainer.agent = FakeVectorAgent()
    trainer.web_dashboard = FakeMetricsDashboard()
    trainer.best_score = 5
    trainer.current_episode = 1
    trainer.scores = []
    trainer.wins = []
    trainer.levels = []
    trainer.q_values = []
    trainer.losses = []
    trainer.epsilons = []
    trainer.rewards = []
    trainer.progresses = []
    trainer.end_reasons = []
    trainer.progress_parts = []
    trainer.total_steps = 8
    trainer.exploration_actions = 6
    trainer.exploitation_actions = 2
    trainer.target_updates = 0
    trainer.last_target_update_step = 0
    trainer.evaluator = None
    saved = []
    cleanups = []
    nn_emits = []
    trainer._save_model = lambda filename, **kwargs: saved.append((filename, kwargs))
    trainer._cleanup_old_periodic_saves = lambda keep_last: cleanups.append(keep_last)
    trainer._emit_nn_visualization = lambda state, action: nn_emits.append((state.tolist(), action))

    states = np.array([[1.0, 2.0]], dtype=np.float32)
    actions = np.array([3], dtype=np.int64)
    rewards = np.array([7.5], dtype=np.float64)
    steps = np.array([4], dtype=np.int64)
    info = {
        "score": 10,
        "won": True,
        "level": 3,
        "progress": 0.75,
        "end_reason": "won",
        "progress_parts": {"crystal_frac": 1.0},
        "bricks": 4,
    }

    last_score = trainer._complete_vectorized_episode(
        0,
        info,
        states,
        actions,
        rewards,
        steps,
    )

    assert last_score == 10
    assert trainer.current_episode == 2
    assert trainer.scores == [10]
    assert trainer.wins == [True]
    assert trainer.levels == [3]
    assert trainer.q_values == [2.0]
    assert trainer.losses == [0.25]
    assert trainer.epsilons == [0.4]
    assert trainer.rewards == [7.5]
    assert trainer.progresses == [0.75]
    assert trainer.progress_parts == [{"crystal_frac": 1.0}]
    assert trainer.best_score == 10
    assert trainer.target_updates == 1
    assert trainer.last_target_update_step == 25
    assert rewards.tolist() == [0.0]
    assert steps.tolist() == [0]
    assert saved == [
        ("crystal_caves_best.pth", {"save_reason": "best", "quiet": True}),
        (
            "crystal_caves_ep2.pth",
            {"save_reason": "periodic", "save_replay_buffer": False},
        ),
    ]
    assert cleanups == [5]
    assert trainer.web_dashboard.metric_payloads[0]["episode"] == 1
    assert trainer.web_dashboard.metric_payloads[0]["reward"] == 7.5
    assert trainer.web_dashboard.metric_payloads[0]["episode_length"] == 4
    assert trainer.web_dashboard.metric_payloads[0]["cc_info"] is info
    assert trainer.web_dashboard.publisher.state.learn_every == 8
    assert nn_emits == [([1.0, 2.0], 3)]


def test_headless_vectorized_eval_can_stop_and_restore_on_plateau(tmp_path):
    """Periodic vector eval should record dashboard metrics and early-stop on plateau."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(
        GAME_NAME="crystal_caves",
        GAME_MODEL_DIR=str(tmp_path),
        EVAL_EVERY=2,
        EVAL_EPISODES=5,
        EVAL_MAX_STEPS=120,
        EARLY_STOP_ON_PLATEAU=True,
        EARLY_STOP_PATIENCE=3,
        EVAL_PLATEAU_BOOST_EPISODES=4,
        EVAL_PLATEAU_EPSILON_BOOST=0.7,
        DISABLE_EXPLORATION_BOOST=False,
        EVAL_BOOST_WIN_REGRESSION_FRAC=0.7,
    )
    trainer.current_episode = 2
    trainer.running = True
    trainer.agent = FakeVectorAgent()
    trainer.web_dashboard = FakeMetricsDashboard()
    trainer._exploration_boost_active = False
    trainer._exploration_boost_end_episode = 0
    restored = []
    trainer._restore_eval_best = lambda: restored.append(True) or True
    eval_result = SimpleNamespace(
        mean_score=20.0,
        std_score=1.5,
        median_score=19.0,
        win_rate=0.0,
        num_games=5,
        mean_crystal_frac=0.4,
        mean_switch_rate=0.2,
        mean_depth_frac=0.6,
        end_reason_counts={"timeout": 5},
        max_level=1,
    )
    trainer.evaluator = FakeEvaluator(
        eval_result,
        evals_since_improvement=3,
        best_eval_score=99.0,
        plateau=True,
    )

    trainer._maybe_run_vectorized_eval()

    assert trainer.evaluator.evaluate_calls == [
        {"num_episodes": 5, "max_steps": 120, "episode_num": 2}
    ]
    assert trainer.evaluator.logged_results == [eval_result]
    assert trainer.web_dashboard.publisher.eval_payloads[0]["episode"] == 2
    assert trainer.web_dashboard.publisher.eval_payloads[0]["mean_score"] == 20.0
    assert trainer.running is False
    assert restored == [True]
    assert any("Early stop" in message for message, _level, _data in trainer.web_dashboard.logs)


def test_headless_vectorized_post_episode_update_ends_exploration_boost():
    """Boost shutdown should reset epsilon, clear evaluator plateau count, and resume decay."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(EPSILON_END=0.05, LR_DECAY=False, MAX_EPISODES=10)
    trainer.agent = FakeVectorAgent()
    trainer.web_dashboard = FakeDashboard()
    trainer.current_episode = 5
    trainer.epsilon_episode_offset = 0
    trainer._exploration_boost_active = True
    trainer._exploration_boost_end_episode = 5
    trainer.evaluator = SimpleNamespace(evals_since_improvement=4)
    trainer._apply_lr_decay = lambda start_episode, current_episode: None

    trainer._update_vectorized_after_completed_episodes(start_episode=0)

    assert trainer._exploration_boost_active is False
    assert trainer.agent.epsilon == 0.05
    assert trainer.agent.decay_calls == [5]
    assert trainer.agent.scheduler_steps == 1
    assert trainer.evaluator.evals_since_improvement == 0


def test_headless_apply_config_rejects_invalid_learning_rate():
    """Headless dashboard config changes should not inject NaN into the optimizer."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(LEARNING_RATE=0.001)
    trainer.agent = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]))
    trainer.web_dashboard = FakeDashboard()

    trainer._apply_config({"learning_rate": float("nan")})

    assert trainer.config.LEARNING_RATE == 0.001
    assert trainer.agent.optimizer.param_groups[0]["lr"] == 0.001
    assert trainer.web_dashboard.logs == []


@pytest.mark.parametrize("runtime_cls", [main.GameApp, main.HeadlessTrainer])
def test_runtime_performance_mode_uses_shared_preset(runtime_cls):
    """Visual and headless modes should apply the same shared preset values."""
    runtime = runtime_cls.__new__(runtime_cls)
    runtime.config = SimpleNamespace(LEARN_EVERY=1, BATCH_SIZE=64, GRADIENT_STEPS=1)
    runtime.web_dashboard = None

    runtime._set_performance_mode("ultra")

    assert runtime.config.LEARN_EVERY == 32
    assert runtime.config.BATCH_SIZE == 128
    assert runtime.config.GRADIENT_STEPS == 2


def test_game_app_set_speed_rejects_non_numeric_values():
    """Speed changes should ignore malformed values instead of raising."""
    app = main.GameApp.__new__(main.GameApp)
    app.game_speed = 5.0
    app.web_dashboard = None
    app._last_logged_speed = 5.0

    app._set_speed("fast")

    assert app.game_speed == 5.0


def test_game_app_new_best_dashboard_log_handles_deque_scores():
    """New-best dashboard logging should average deque history without slicing it."""
    scores = deque([10, 20, 30], maxlen=1000)

    assert main.GameApp._average_recent_scores(scores) == 20.0


@pytest.mark.parametrize("runtime_cls", [main.GameApp, main.HeadlessTrainer])
def test_runtime_nn_visualization_uses_shared_snapshot_builder(runtime_cls, monkeypatch):
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

    runtime_module = importlib.import_module(runtime_cls.__module__)
    monkeypatch.setattr(runtime_module, "build_nn_snapshot", fake_build_nn_snapshot)

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
