"""
Tests for the web dashboard server.

Tests cover:
- Utility functions
- Data structures
- MetricsPublisher functionality
- Training state management
"""

import re
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

# Try to import web server components
try:
    from src.web.server import (
        FLASK_AVAILABLE,
        LogMessage,
        MetricsPublisher,
        TrainingState,
        _make_json_safe,
    )

    WEB_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_AVAILABLE = False


# Skip all tests if Flask is not available
pytestmark = pytest.mark.skipif(not WEB_AVAILABLE, reason="Flask/SocketIO not installed")


class TestMakeJsonSafe:
    """Tests for the _make_json_safe utility function."""

    def test_native_types_unchanged(self):
        """Native Python types should pass through unchanged."""
        assert _make_json_safe(42) == 42
        assert _make_json_safe(3.14) == 3.14
        assert _make_json_safe("hello") == "hello"
        assert _make_json_safe(True) is True
        assert _make_json_safe(None) is None

    def test_numpy_int_converted(self):
        """NumPy integer types should be converted to native Python int."""
        import numpy as np

        assert _make_json_safe(np.int64(42)) == 42
        assert isinstance(_make_json_safe(np.int64(42)), int)

    def test_numpy_float_converted(self):
        """NumPy float types should be converted to native Python float."""
        import numpy as np

        assert _make_json_safe(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_make_json_safe(np.float64(3.14)), float)

    def test_numpy_arrays_converted(self):
        """NumPy arrays should be converted to lists."""
        import numpy as np

        arr = np.array([1, 2, 3])
        result = _make_json_safe(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict_converted(self):
        """Nested dictionaries with NumPy values should be converted."""
        import numpy as np

        data = {"score": np.float64(100.0), "nested": {"values": np.array([1, 2, 3])}}
        result = _make_json_safe(data)
        assert result["score"] == 100.0
        assert result["nested"]["values"] == [1, 2, 3]

    def test_list_with_numpy_converted(self):
        """Lists containing NumPy values should be converted."""
        import numpy as np

        data = [np.int64(1), np.float64(2.0), "three"]
        result = _make_json_safe(data)
        assert result[0] == 1
        assert result[1] == 2.0
        assert result[2] == "three"


class TestTrainingState:
    """Tests for the TrainingState dataclass."""

    def test_default_values(self):
        """TrainingState should have sensible defaults."""
        state = TrainingState()
        assert state.episode == 0
        assert state.score == 0
        assert state.best_score == 0
        assert state.epsilon == 1.0
        assert state.is_paused is False
        assert state.is_running is False

    def test_update_from_kwargs(self):
        """TrainingState fields should be assignable."""
        state = TrainingState()
        state.episode = 100
        state.score = 50
        state.epsilon = 0.5
        assert state.episode == 100
        assert state.score == 50
        assert state.epsilon == 0.5

    def test_device_default(self):
        """TrainingState should have cpu as default device."""
        state = TrainingState()
        assert state.device == "cpu"

    def test_performance_fields(self):
        """TrainingState should have performance tracking fields."""
        state = TrainingState()
        assert state.learn_every == 1
        assert state.gradient_steps == 1
        assert state.steps_per_second == 0.0


class TestLogMessage:
    """Tests for the LogMessage class."""

    def test_creation(self):
        """LogMessage should store message and level."""
        msg = LogMessage(timestamp=datetime.now().isoformat(), message="Test message", level="info")
        assert msg.message == "Test message"
        assert msg.level == "info"

    def test_to_dict(self):
        """LogMessage should convert to dictionary correctly."""
        msg = LogMessage(timestamp="2024-01-01T00:00:00", message="Test", level="warning")
        data = msg.to_dict()
        assert data["message"] == "Test"
        assert data["level"] == "warning"
        assert data["timestamp"] == "2024-01-01T00:00:00"

    def test_data_field(self):
        """LogMessage should support optional data field."""
        msg = LogMessage(
            timestamp="2024-01-01T00:00:00",
            message="With data",
            level="info",
            data={"key": "value"},
        )
        data = msg.to_dict()
        assert data["data"] == {"key": "value"}


class TestMetricsPublisher:
    """Tests for the MetricsPublisher class."""

    def test_initialization(self):
        """MetricsPublisher should initialize with default state."""
        publisher = MetricsPublisher(history_length=100)
        assert publisher.state.episode == 0
        assert len(publisher.scores) == 0
        assert len(publisher.console_logs) == 0

    def test_update_state(self):
        """MetricsPublisher.update should update state correctly."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=10, score=100, epsilon=0.5, loss=0.01)

        assert publisher.state.episode == 10
        assert publisher.state.score == 100
        assert publisher.state.epsilon == 0.5
        assert publisher.state.loss == 0.01

    def test_crystal_caves_info_populates_panel_state(self):
        """cc_info should drive the Crystal Caves dashboard fields."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(
            episode=5,
            score=350,
            epsilon=0.1,
            loss=0.01,
            won=False,
            game_name="crystal_caves",
            cc_info={
                "progress": 0.45,
                "progress_parts": {
                    "crystal_frac": 0.33,
                    "switch_done": 1.0,
                    "depth_frac": 0.6,
                    "won": 0.0,
                },
                "crystals_remaining": 2,
                "initial_crystals": 3,
                "level_name": "Cave 1",
                "end_reason": "killed",
            },
        )

        state = publisher.state
        assert state.game_name == "crystal_caves"
        assert state.cc_active is True
        assert state.cc_progress == 0.45
        assert state.cc_best_progress == 0.45
        assert state.cc_crystal_frac == 0.33
        assert state.cc_switch_done == 1.0
        assert state.cc_depth_frac == 0.6
        assert state.cc_crystals_remaining == 2
        assert state.cc_initial_crystals == 3
        assert state.cc_level_name == "Cave 1"
        assert state.cc_end_reason == "killed"
        assert state.cc_end_reason_counts == {"killed": 1}

    def test_crystal_caves_best_progress_is_monotonic(self):
        """cc_best_progress should only ever rise, even as live progress dips."""
        publisher = MetricsPublisher(history_length=100)

        for prog in (0.30, 0.55, 0.20):
            publisher.update(
                episode=1,
                score=0,
                epsilon=0.1,
                loss=0.0,
                game_name="crystal_caves",
                cc_info={"progress": prog, "end_reason": "running"},
            )

        # "running" episodes are in-progress, so they must NOT be counted as outcomes.
        assert publisher.state.cc_best_progress == 0.55
        assert publisher.state.cc_progress == 0.20
        assert publisher.state.cc_end_reason_counts == {}

    def test_non_crystal_game_leaves_panel_inactive(self):
        """Other games never pass cc_info, so the panel stays hidden."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=1, score=10, epsilon=0.5, loss=0.1, game_name="breakout")

        assert publisher.state.game_name == "breakout"
        assert publisher.state.cc_active is False

    def test_record_eval_drives_the_held_out_panel(self):
        """record_eval should populate the held-out eval state + sparkline history,
        with a monotonic best — this is the trustworthy generalization measure."""
        publisher = MetricsPublisher(history_length=100)
        assert publisher.state.eval_ran is False

        for ep, mean in [(150, 31), (300, 28), (450, 42), (600, 66)]:
            publisher.record_eval(
                episode=ep,
                mean_score=mean,
                std_score=100.0,
                median_score=0.0,
                win_rate=0.0,
                num_games=20,
            )

        st = publisher.state
        assert st.eval_ran is True
        assert st.eval_episode == 600
        assert st.eval_mean_score == 66.0
        assert st.eval_num_games == 20
        # Best is monotonic even though the mean dipped 31 -> 28 along the way.
        assert st.eval_best_mean == 66.0
        # History feeds the sparkline in trajectory order.
        assert st.eval_history == [31.0, 28.0, 42.0, 66.0]

    def test_record_eval_pushes_an_update(self):
        """An eval must push to the dashboard immediately, not wait for the next
        per-episode metric emit."""
        publisher = MetricsPublisher(history_length=100)
        received = []
        publisher.on_update(lambda snapshot: received.append(snapshot))

        publisher.record_eval(
            episode=150, mean_score=31, std_score=108, median_score=0, win_rate=0.0, num_games=20
        )

        assert len(received) == 1
        assert received[0]["state"]["eval_mean_score"] == 31.0

    def test_history_tracking(self):
        """MetricsPublisher should track history."""
        publisher = MetricsPublisher(history_length=100)

        for i in range(5):
            publisher.update(episode=i, score=i * 10, epsilon=1.0 - i * 0.1, loss=1.0 / (i + 1))

        assert len(publisher.scores) == 5
        assert len(publisher.losses) == 5
        assert len(publisher.epsilons) == 5

    def test_best_score_tracking(self):
        """MetricsPublisher should track best score."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=1, score=50, epsilon=0.9, loss=0.1)
        assert publisher.state.best_score == 50

        publisher.update(episode=2, score=30, epsilon=0.8, loss=0.1)
        assert publisher.state.best_score == 50  # Should not decrease

        publisher.update(episode=3, score=75, epsilon=0.7, loss=0.1)
        assert publisher.state.best_score == 75

    def test_log_message(self):
        """MetricsPublisher.log should add messages to console."""
        publisher = MetricsPublisher(history_length=100)

        publisher.log("Test message", level="info")

        assert len(publisher.console_logs) == 1
        assert publisher.console_logs[0].message == "Test message"

    def test_log_timestamp_uses_milliseconds(self):
        """Log timestamps should keep three fractional-second digits."""
        publisher = MetricsPublisher(history_length=100)

        publisher.log("Test message", level="info")

        assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}$", publisher.console_logs[0].timestamp)

    def test_log_level_parsing(self):
        """MetricsPublisher.log should handle different log levels."""
        publisher = MetricsPublisher(history_length=100)

        publisher.log("Debug", level="debug")
        publisher.log("Info", level="info")
        publisher.log("Warning", level="warning")
        publisher.log("Error", level="error")

        assert publisher.console_logs[0].level == "debug"
        assert publisher.console_logs[1].level == "info"
        assert publisher.console_logs[2].level == "warning"
        assert publisher.console_logs[3].level == "error"

    def test_console_log_limit(self):
        """MetricsPublisher should limit console log size."""
        publisher = MetricsPublisher(history_length=100)

        # Add more than the limit (500)
        for i in range(600):
            publisher.log(f"Message {i}")

        # Should be capped at 500
        assert len(publisher.console_logs) <= 500

    def test_get_snapshot(self):
        """MetricsPublisher.get_snapshot should return current state."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=5, score=25, epsilon=0.8, loss=0.05)
        snapshot = publisher.get_snapshot()

        assert "state" in snapshot
        assert snapshot["state"]["episode"] == 5
        assert snapshot["state"]["score"] == 25

    def test_get_snapshot_can_limit_history_payload(self):
        """Initial dashboard payloads should be able to send only recent history."""
        publisher = MetricsPublisher(history_length=100)

        for episode in range(5):
            publisher.update(
                episode=episode,
                score=episode * 10,
                epsilon=0.8,
                loss=0.05,
                reward=float(episode),
                avg_q_value=float(episode) / 10,
                episode_length=episode + 1,
            )

        snapshot = publisher.get_snapshot(history_limit=2)

        assert snapshot["history"]["scores"] == [30, 40]
        assert snapshot["history"]["losses"] == [0.05, 0.05]
        assert snapshot["history"]["rewards"] == [3.0, 4.0]
        assert snapshot["history"]["q_values"] == [0.3, 0.4]
        assert snapshot["history"]["episode_lengths"] == [4, 5]

    def test_update_log_and_save_callbacks_receive_payloads(self, monkeypatch):
        """Registered callbacks should receive immutable payload snapshots."""
        from src.web import metrics_publisher as metrics_module

        publisher = MetricsPublisher(history_length=100)
        updates = []
        logs = []
        saves = []
        publisher.on_update(updates.append)
        publisher.on_log(logs.append)
        publisher.on_save(saves.append)

        now = iter([100.0, 101.0, 102.0])
        monkeypatch.setattr(metrics_module.time, "time", lambda: next(now))

        publisher.update(
            episode=1,
            score=20,
            epsilon=0.5,
            loss=0.01,
            total_steps=10,
            won=True,
            reward=3.0,
            memory_size=50,
            avg_q_value=1.5,
            exploration_actions=2,
            exploitation_actions=8,
            target_updates=1,
            bricks_broken=4,
            episode_length=30,
            q_value_left=0.1,
            q_value_stay=0.2,
            q_value_right=0.3,
            selected_action=2,
        )
        publisher.log("Saved checkpoint", level="success", data={"episode": 1})
        publisher.record_save("best.pth", "best_score", episode=1, best_score=20)

        assert updates[0]["state"]["episode"] == 1
        assert updates[0]["state"]["win_rate"] == 1.0
        assert updates[0]["history"]["scores"] == [20]
        assert publisher.action_frequency["right"] == 1
        assert publisher.action_frequency["exploration"] == 2
        assert publisher.action_frequency["exploitation"] == 8
        assert logs[0].message == "Saved checkpoint"
        assert saves[0]["last_save_filename"] == "best.pth"
        assert saves[0]["saves_this_session"] == 1
        assert saves[0]["time_since_save_str"] == "1s ago"

    def test_state_mutators_and_console_log_limit_readback(self):
        publisher = MetricsPublisher(history_length=100)

        publisher.set_paused(True)
        publisher.set_speed(2.5)
        publisher.update_config(
            {
                "learning_rate": 0.002,
                "batch_size": 64,
                "learn_every": 4,
                "gradient_steps": 2,
            }
        )
        publisher.set_performance_mode("fast")
        publisher.set_system_info("cpu", torch_compiled=True, target_episodes=500, headless=True)
        publisher.log("first")
        publisher.log("second")

        assert publisher.state.is_paused is True
        assert publisher.state.game_speed == 2.5
        assert publisher.state.learning_rate == 0.002
        assert publisher.state.batch_size == 64
        assert publisher.state.learn_every == 4
        assert publisher.state.gradient_steps == 2
        assert publisher.state.performance_mode == "fast"
        assert publisher.state.device == "cpu"
        assert publisher.state.torch_compiled is True
        assert publisher.state.target_episodes == 500
        assert publisher.state.headless is True
        assert [entry["message"] for entry in publisher.get_console_logs(limit=1)] == ["second"]

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "Never"), (59, "59s ago"), (120, "2m ago"), (3660, "1h 1m ago")],
    )
    def test_format_time_ago(self, seconds, expected):
        assert MetricsPublisher()._format_time_ago(seconds) == expected

    def test_screenshot_handling(self):
        """MetricsPublisher should handle screenshot retrieval."""
        publisher = MetricsPublisher(history_length=100)

        # get_screenshot should return None initially
        assert publisher.get_screenshot() is None

    def test_set_screenshot_encodes_png_or_clears_on_error(self, monkeypatch):
        class FakeSurface:
            def get_size(self):
                return (1, 1)

            def copy(self):
                return self

        fake_image = SimpleNamespace(
            tostring=lambda _surface, _fmt: b"\x00\x00\x00",
            save=lambda _surface, buffer: buffer.write(b"png"),
        )
        monkeypatch.setitem(sys.modules, "pygame", SimpleNamespace(image=fake_image))

        publisher = MetricsPublisher(history_length=100)
        publisher.set_screenshot(FakeSurface())

        assert publisher.get_screenshot()

        fake_image.tostring = lambda _surface, _fmt: (_ for _ in ()).throw(RuntimeError("boom"))
        publisher.set_screenshot(FakeSurface())

        assert publisher.get_screenshot() is None

    def test_set_running(self):
        """MetricsPublisher.set_running should update is_running flag."""
        publisher = MetricsPublisher(history_length=100)

        assert publisher.state.is_running is False
        publisher.set_running(True)
        assert publisher.state.is_running is True
        publisher.set_running(False)
        assert publisher.state.is_running is False

    def test_action_frequency_tracking(self):
        """MetricsPublisher should track action frequencies."""
        publisher = MetricsPublisher(history_length=100)

        # Action frequency dict should exist
        assert "left" in publisher.action_frequency
        assert "stay" in publisher.action_frequency
        assert "right" in publisher.action_frequency

    def test_layer_analysis_handles_empty_arrays(self):
        """Layer analysis should not crash on empty activation, weight, or gradient arrays."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="empty",
            neuron_count=0,
            activations=np.array([], dtype=np.float32),
            weights=np.array([], dtype=np.float32),
            gradients=np.array([], dtype=np.float32),
        )

        analysis = publisher.get_layer_analysis(0)
        assert analysis["avg_activation"] == 0.0
        assert analysis["activation_histogram"] == [0] * 20
        assert analysis["weight_histogram"] == [0] * 20
        assert analysis["gradient_max_magnitude"] == 0.0

    def test_all_layer_analysis_is_sorted(self):
        """All layer analysis should return layers in index order."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update_layer_analysis(2, "layer_2", 1, np.array([0.2], dtype=np.float32))
        publisher.update_layer_analysis(1, "layer_1", 1, np.array([0.1], dtype=np.float32))

        layers = publisher.get_all_layer_analysis()
        assert [layer["layer_idx"] for layer in layers] == [1, 2]

    def test_nn_visualization_callbacks_and_analysis_details(self, monkeypatch):
        from src.web import metrics_publisher as metrics_module

        publisher = MetricsPublisher(history_length=100)
        nn_updates = []
        publisher.on_nn_update(nn_updates.append)
        monkeypatch.setattr(metrics_module.time, "time", lambda: 10.0)

        publisher.update_nn_visualization(
            layer_info=[{"name": "hidden", "neurons": 2, "type": "dense"}],
            activations={"hidden": [0.1, 0.2]},
            q_values=[1.0, 2.0],
            selected_action=1,
            weights=[[[0.1, 0.2], [0.3, 0.4]]],
            step=7,
            action_labels=["LEFT", "RIGHT"],
        )

        assert nn_updates[0]["step"] == 7
        assert nn_updates[0]["action_labels"] == ["LEFT", "RIGHT"]
        assert nn_updates[0]["weights"] == []
        assert publisher.get_nn_visualization(include_weights=True)["weights"]
        assert publisher.should_update_nn_visualization(current_time=10.0) is False

        publisher.update_neuron_inspection(
            layer_idx=1,
            neuron_idx=2,
            layer_name="hidden",
            current_activation=0.75,
            activation_history=list(range(600)),
            incoming_weights=[0.1, 0.2, 0.3],
            outgoing_weights=[0.4, 0.5],
            q_contributions={"LEFT": 0.2},
        )
        neuron = publisher.get_neuron_details(1, 2)
        assert neuron["current_activation"] == 0.75
        assert len(neuron["activation_history"]) == 100
        assert neuron["incoming_weight_stats"]["mean"] == pytest.approx(0.2)
        assert neuron["outgoing_weight_stats"]["max"] == pytest.approx(0.5)
        assert publisher.get_neuron_details(9, 9) == {"error": "Neuron not found"}

        publisher.update_layer_analysis(
            layer_idx=3,
            layer_name="dense",
            neuron_count=4,
            activations=np.array([0.0, 0.5, 0.96, -0.97], dtype=np.float32),
            weights=np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
            gradients=np.array([0.1, -0.3], dtype=np.float32),
        )
        layer = publisher.get_layer_analysis(3)
        assert layer["dead_neuron_count"] == 1
        assert layer["saturated_neuron_count"] == 2
        assert layer["weight_max"] == pytest.approx(0.4)
        assert layer["gradient_max_magnitude"] == pytest.approx(0.3)
        assert publisher.get_layer_analysis(99) == {"error": "Layer not found"}

    def test_reset_all_state_clears_runtime_payloads(self):
        publisher = MetricsPublisher(history_length=100)
        publisher.update(episode=3, score=50, epsilon=0.4, loss=0.1)
        publisher.log("before reset")
        publisher.record_save("best.pth", "best", episode=3, best_score=50)
        publisher._screenshot_data = "encoded"
        publisher.update_nn_visualization(
            layer_info=[{"name": "hidden", "neurons": 1, "type": "dense"}],
            activations={"hidden": [1.0]},
            q_values=[1.0],
            selected_action=0,
            weights=[],
            step=1,
        )

        publisher.reset_all_state()

        assert publisher.state.episode == 0
        assert publisher.state.epsilon == 1.0
        assert list(publisher.scores) == []
        assert list(publisher.console_logs) == []
        assert publisher.get_screenshot() is None
        assert publisher.get_save_status()["saves_this_session"] == 0
        assert publisher.get_nn_visualization()["step"] == 0
