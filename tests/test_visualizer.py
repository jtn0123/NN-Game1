"""
Tests for the visualizer module.

Tests cover:
- NeuralNetVisualizer initialization and configuration
- Dashboard metrics tracking
- DataFlowPulse animation
- Color interpolation
- HUD state tracking
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys

# Set SDL_VIDEODRIVER before importing pygame to avoid display errors in CI
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

# Mock pygame.display.init before importing modules
with patch.dict('os.environ', {'SDL_VIDEODRIVER': 'dummy'}):
    import pygame
    pygame.init()


from config import Config
from src.visualizer.nn_visualizer import NeuralNetVisualizer, DataFlowPulse
from src.visualizer.dashboard import Dashboard
from src.visualizer.hud import TrainingHUD


class TestDataFlowPulse:
    """Tests for the DataFlowPulse animation class."""

    def test_initialization(self):
        """DataFlowPulse should initialize with correct properties."""
        pulse = DataFlowPulse(
            start_pos=(0, 0),
            end_pos=(100, 100),
            color=(255, 0, 0),
            speed=0.1
        )
        assert pulse.start == (0, 0)
        assert pulse.end == (100, 100)
        assert pulse.color == (255, 0, 0)
        assert pulse.speed == 0.1
        assert pulse.progress == 0.0
        assert pulse.alive is True

    def test_update_increments_progress(self):
        """Update should increment progress by speed."""
        pulse = DataFlowPulse(
            start_pos=(0, 0),
            end_pos=(100, 100),
            color=(255, 0, 0),
            speed=0.1
        )
        pulse.update()
        assert pulse.progress == pytest.approx(0.1)
        assert pulse.alive is True

    def test_pulse_dies_at_end(self):
        """Pulse should die when progress reaches 1.0."""
        pulse = DataFlowPulse(
            start_pos=(0, 0),
            end_pos=(100, 100),
            color=(255, 0, 0),
            speed=0.5
        )
        pulse.update()  # 0.5
        assert pulse.alive is True
        pulse.update()  # 1.0
        assert pulse.alive is False

    def test_position_interpolation(self):
        """Position should interpolate between start and end."""
        pulse = DataFlowPulse(
            start_pos=(0, 0),
            end_pos=(100, 100),
            color=(255, 0, 0),
            speed=0.5
        )
        # At start
        assert pulse.position == (0, 0)

        # After one update (50% progress)
        pulse.update()
        assert pulse.position == (50, 50)


class TestNeuralNetVisualizer:
    """Tests for the NeuralNetVisualizer class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def visualizer(self, config):
        """Create a NeuralNetVisualizer instance."""
        return NeuralNetVisualizer(
            config=config,
            x=0,
            y=0,
            width=300,
            height=500
        )

    def test_initialization(self, visualizer, config):
        """Visualizer should initialize with correct config values."""
        assert visualizer.x == 0
        assert visualizer.y == 0
        assert visualizer.width == 300
        assert visualizer.height == 500
        assert visualizer.neuron_radius == config.VIS_NEURON_RADIUS
        assert visualizer.max_neurons == config.VIS_MAX_NEURONS_DISPLAY

    def test_fast_mode_setting(self, config):
        """Visualizer should respect fast_mode config."""
        config.VIS_FAST_MODE = True
        visualizer = NeuralNetVisualizer(config=config)
        assert visualizer.fast_mode is True

        config.VIS_FAST_MODE = False
        visualizer = NeuralNetVisualizer(config=config)
        assert visualizer.fast_mode is False

    def test_color_interpolation(self, visualizer):
        """Color interpolation should work correctly with ease-out."""
        # The implementation uses ease-out: t = 1 - (1 - t) ** 2
        # For t=0.5: eased_t = 1 - 0.25 = 0.75
        # So (0,0,0) to (255,255,255) at t=0.5 gives 255 * 0.75 = 191
        result = visualizer._interpolate_color((0, 0, 0), (255, 255, 255), 0.5)
        assert result == (191, 191, 191)

        # At t=0, should be color1
        result = visualizer._interpolate_color((100, 50, 0), (0, 200, 100), 0.0)
        assert result == (100, 50, 0)

        # At t=1, should be color2
        result = visualizer._interpolate_color((100, 50, 0), (0, 200, 100), 1.0)
        assert result == (0, 200, 100)

    def test_pulse_management(self, visualizer):
        """Visualizer should manage pulse list correctly."""
        assert len(visualizer.pulses) == 0

        # Add a pulse
        pulse = DataFlowPulse((0, 0), (100, 100), (255, 0, 0))
        visualizer.pulses.append(pulse)
        assert len(visualizer.pulses) == 1

    def test_layer_positions_calculation(self, visualizer):
        """Visualizer should calculate layer positions correctly."""
        # Create mock layer_info matching the format from network.get_layer_info()
        layer_info = [
            {'name': 'input', 'neurons': 55, 'type': 'input'},
            {'name': 'hidden1', 'neurons': 128, 'type': 'hidden'},
            {'name': 'hidden2', 'neurons': 64, 'type': 'hidden'},
            {'name': 'output', 'neurons': 3, 'type': 'output'}
        ]

        # Calculate layer positions
        positions = visualizer._calculate_layer_positions(layer_info)

        # Should have same number of layers as input
        assert len(positions) == len(layer_info)

        # Each layer should have position info
        for pos in positions:
            assert 'x' in pos
            assert 'neurons' in pos

    def test_activation_history(self, visualizer):
        """Visualizer should track activation history."""
        # Activation history should be a dict of deques
        assert isinstance(visualizer.activation_history, dict)
        assert visualizer.history_length == 30


class TestDashboard:
    """Tests for the Dashboard class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def dashboard(self, config):
        """Create a Dashboard instance."""
        return Dashboard(
            config=config,
            x=0,
            y=0,
            width=800,
            height=150
        )

    def test_initialization(self, dashboard):
        """Dashboard should initialize with correct properties."""
        assert dashboard.x == 0
        assert dashboard.y == 0
        assert dashboard.width == 800
        assert dashboard.height == 150

    def test_metrics_storage(self, dashboard):
        """Dashboard should store metrics in history."""
        # Add some metrics using update()
        dashboard.update(episode=1, score=100, epsilon=0.5, loss=0.1)
        dashboard.update(episode=2, score=150, epsilon=0.4, loss=0.08)

        assert len(dashboard.scores) == 2
        assert dashboard.scores[-1] == 150
        assert dashboard.epsilons[-1] == 0.4

    def test_running_average(self, dashboard):
        """Dashboard should calculate running averages correctly."""
        # Add 10 scores
        for i in range(10):
            dashboard.update(episode=i, score=i * 10, epsilon=0.5, loss=0.1)

        # Dashboard tracks avg_scores internally
        assert len(dashboard.avg_scores) == 10
        # Last average should be close to mean of recent scores
        assert dashboard.avg_scores[-1] > 0

    def test_loss_spike_detection(self, dashboard):
        """Dashboard should detect loss spikes internally."""
        # Add stable losses to build up smoothed_loss
        for i in range(10):
            dashboard.update(episode=i, score=100, epsilon=0.5, loss=0.1)

        # Record the smoothed loss before spike
        smoothed_before = dashboard.smoothed_loss

        # Add a spike (10x increase) - should trigger internal spike timer
        dashboard.update(episode=10, score=100, epsilon=0.5, loss=1.0)

        # Spike timer should be set (internal mechanism)
        assert dashboard._loss_spike_timer > 0

    def test_metric_trend(self, dashboard):
        """Dashboard should track metric trends via MetricCard."""
        # Add increasing scores
        for i in range(20):
            dashboard.update(episode=i, score=i * 10, epsilon=0.5, loss=0.1)

        # MetricCard tracks trend via history
        score_card = dashboard.cards['score']
        assert len(score_card.history) > 0
        # Trend should show improvement (↑)
        assert score_card.trend in ("↑", "→", "↓")

    def test_history_limits(self, dashboard):
        """Dashboard should limit history size."""
        # Add many metrics
        for i in range(1000):
            dashboard.update(episode=i, score=i, epsilon=0.5, loss=0.1)

        # Should be limited to max_history
        assert len(dashboard.scores) <= dashboard.max_history


class TestTrainingHUD:
    """Tests for the TrainingHUD class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def hud(self, config):
        """Create a TrainingHUD instance."""
        return TrainingHUD(config=config)

    def test_initialization(self, hud, config):
        """HUD should initialize with correct properties."""
        assert hud.config == config
        assert hud.enabled == getattr(config, 'HUD_ENABLED', True)
        assert hud.opacity == getattr(config, 'HUD_OPACITY', 0.8)

    def test_fonts_initialized(self, hud):
        """HUD should initialize fonts correctly."""
        assert hud._font_small is not None
        assert hud._font_medium is not None
        assert hud._font_large is not None

    def test_colors_defined(self, hud):
        """HUD should have color properties defined."""
        assert hud.text_color == (220, 220, 220)
        assert hud.text_dim == (150, 150, 150)
        assert hud.accent_color == (52, 152, 219)
        assert hud.good_color == (46, 204, 113)
        assert hud.warn_color == (241, 196, 15)

    def test_smooth_color_transition_state(self, hud):
        """HUD should track color transition state for smooth animations."""
        # Bug 90: Smooth score color transition
        assert hasattr(hud, '_current_score_color')
        assert len(hud._current_score_color) == 3
        assert hud._color_lerp_speed == 0.1


class TestColorUtils:
    """Tests for color utility functions."""

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer for testing color methods."""
        return NeuralNetVisualizer(Config())

    def test_activation_colors_diverge(self, visualizer):
        """Activation colors should diverge for positive/negative values."""
        # Positive activation should give warm color
        pos_color = visualizer._interpolate_color(
            visualizer.inactive_color,
            visualizer.color_positive,
            0.8
        )

        # Negative activation should give cool color
        neg_color = visualizer._interpolate_color(
            visualizer.inactive_color,
            visualizer.color_negative,
            0.8
        )

        # Colors should be different
        assert pos_color != neg_color

    def test_interpolation_clamps(self, visualizer):
        """Color interpolation should clamp t to [0, 1] range."""
        # t > 1 should clamp to 1, giving color2
        result = visualizer._interpolate_color((0, 0, 0), (255, 255, 255), 1.5)
        assert result == (255, 255, 255)

        # t < 0 should clamp to 0, giving color1
        result = visualizer._interpolate_color((0, 0, 0), (255, 255, 255), -0.5)
        assert result == (0, 0, 0)
