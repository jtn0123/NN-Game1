"""
Tests for Phase 1: Foundation & Performance improvements

Tests verify:
1.1 Enhanced Metrics Collection - Q-value tracking, action frequency
1.2 Adaptive Visualization Update Rate - Backend rate calculation
1.3 Selective Neural Network Data Transmission - Conditional weight sending
1.4 Frontend Rendering Optimization - Render throttling (integration tested)
"""

import pytest
import time
from src.web.server import (
    MetricsPublisher,
    TrainingState,
    NNVisualizationData,
)


class TestPhase1EnhancedMetrics:
    """Test 1.1: Enhanced Metrics Collection"""

    def test_q_value_history_tracking(self):
        """Verify per-action Q-values are tracked in history"""
        publisher = MetricsPublisher()

        # Record some Q-value data
        for i in range(10):
            publisher.update(
                episode=i,
                score=100,
                epsilon=0.5,
                loss=0.1,
                q_value_left=0.5 + i * 0.01,
                q_value_stay=0.6 + i * 0.01,
                q_value_right=0.55 + i * 0.01,
                selected_action=1,  # STAY
            )

        # Verify per-action Q-values are stored
        assert len(publisher.q_values_left) == 10
        assert len(publisher.q_values_stay) == 10
        assert len(publisher.q_values_right) == 10

        # Verify values are correct
        assert abs(publisher.q_values_left[-1] - 0.59) < 0.01
        assert abs(publisher.q_values_stay[-1] - 0.69) < 0.01
        assert abs(publisher.q_values_right[-1] - 0.64) < 0.01

    def test_action_frequency_tracking(self):
        """Verify action frequency is tracked correctly"""
        publisher = MetricsPublisher()

        # Record multiple actions
        actions = [0, 1, 2, 0, 1, 0, 0, 2, 1, 1]  # Actions taken
        for i, action in enumerate(actions):
            publisher.update(
                episode=i,
                score=100,
                epsilon=0.5,
                loss=0.1,
                selected_action=action,
            )

        # Verify counts
        assert publisher.action_frequency['left'] == 4
        assert publisher.action_frequency['stay'] == 4
        assert publisher.action_frequency['right'] == 2

        # Verify state is updated
        assert publisher.state.action_count_left == 4
        assert publisher.state.action_count_stay == 4
        assert publisher.state.action_count_right == 2

    def test_q_value_deque_maxlen(self):
        """Verify Q-value history maintains rolling window (1000 items)"""
        publisher = MetricsPublisher()

        # Record more than maxlen (1000) entries
        for i in range(1500):
            publisher.update(
                episode=i,
                score=100,
                epsilon=0.5,
                loss=0.1,
                q_value_left=0.5 + i * 0.001,
                q_value_stay=0.6 + i * 0.001,
                q_value_right=0.55 + i * 0.001,
            )

        # Verify deques maintain maxlen
        assert len(publisher.q_values_left) == 1000
        assert len(publisher.q_values_stay) == 1000
        assert len(publisher.q_values_right) == 1000

        # Verify oldest entries are dropped
        # Last entry should be from episode 1499
        assert abs(publisher.q_values_left[-1] - (0.5 + 1499 * 0.001)) < 0.01


class TestPhase1AdaptiveUpdateRate:
    """Test 1.2: Adaptive Visualization Update Rate (Backend)"""

    def test_adaptive_update_rate_very_high_speed(self):
        """Test update rate for very high-speed training (>2000 steps/sec)"""
        publisher = MetricsPublisher()

        # Simulate high-speed training
        publisher._calculate_adaptive_update_rate(2500)

        # Should reduce to 10 FPS (100ms)
        assert publisher._nn_update_interval == 0.1

    def test_adaptive_update_rate_high_speed(self):
        """Test update rate for high-speed training (1000-2000 steps/sec)"""
        publisher = MetricsPublisher()

        publisher._calculate_adaptive_update_rate(1500)

        # Should render at ~15Hz (67ms)
        assert abs(publisher._nn_update_interval - 0.067) < 0.005

    def test_adaptive_update_rate_medium_speed(self):
        """Test update rate for medium-speed training (500-1000 steps/sec)"""
        publisher = MetricsPublisher()

        publisher._calculate_adaptive_update_rate(750)

        # Should render at ~30Hz (33ms)
        assert publisher._nn_update_interval == 0.033

    def test_adaptive_update_rate_slow_speed(self):
        """Test update rate for slow training (<500 steps/sec)"""
        publisher = MetricsPublisher()

        publisher._calculate_adaptive_update_rate(300)

        # Should render at ~60Hz (16ms)
        assert publisher._nn_update_interval == 0.016

    def test_adaptive_update_disabled(self):
        """Test that adaptive updates can be disabled"""
        publisher = MetricsPublisher()
        publisher._adaptive_update_enabled = False

        original_interval = publisher._nn_update_interval
        publisher._calculate_adaptive_update_rate(5000)

        # Should not change when disabled
        assert publisher._nn_update_interval == original_interval


class TestPhase1SelectiveWeightTransmission:
    """Test 1.3: Selective Neural Network Data Transmission"""

    def test_weights_included_on_periodic_update(self):
        """Verify weights are included every 100 steps"""
        nn_data = NNVisualizationData()
        nn_data.step = 150  # Past the 100 step threshold
        nn_data.weights = [[0.1, 0.2], [0.3, 0.4]]
        nn_data._last_weights_step = 0

        # Update at step 150 with last update at 0 - should include weights
        result = nn_data.to_dict(include_weights=False)
        assert 'weights' in result
        assert len(result['weights']) > 0

    def test_weights_empty_when_not_requested(self):
        """Verify weights are empty between periodic updates"""
        nn_data = NNVisualizationData()
        nn_data.step = 50  # Between updates
        nn_data.weights = [[0.1, 0.2], [0.3, 0.4]]
        nn_data._last_weights_step = 0

        # Mid-interval update - should have empty weights
        result = nn_data.to_dict(include_weights=False)
        assert 'weights' in result
        assert len(result['weights']) == 0

    def test_explicit_weight_request(self):
        """Verify weights are included when explicitly requested"""
        nn_data = NNVisualizationData()
        nn_data.step = 50
        nn_data.weights = [[0.1, 0.2], [0.3, 0.4]]

        # Explicit request for weights
        result = nn_data.to_dict(include_weights=True)
        assert 'weights' in result
        assert len(result['weights']) == 2

    def test_nn_data_excludes_internal_fields(self):
        """Verify to_dict() doesn't expose internal implementation details"""
        nn_data = NNVisualizationData()
        result = nn_data.to_dict()

        # Should not expose internal fields
        assert '_last_weights_step' not in result
        assert 'include_weights' not in result

        # Should include public fields
        assert 'layer_info' in result
        assert 'activations' in result
        assert 'q_values' in result


class TestPhase1Integration:
    """Integration tests for Phase 1 improvements"""

    def test_metrics_update_includes_all_fields(self):
        """Verify update() accepts all new Phase 1 parameters"""
        publisher = MetricsPublisher()

        # This should not raise an exception
        publisher.update(
            episode=1,
            score=100,
            epsilon=0.5,
            loss=0.1,
            total_steps=1000,
            won=False,
            reward=0.0,
            memory_size=50000,
            avg_q_value=0.5,
            exploration_actions=100,
            exploitation_actions=900,
            target_updates=1,
            bricks_broken=5,
            episode_length=500,
            q_value_left=0.5,      # Phase 1 new
            q_value_stay=0.6,      # Phase 1 new
            q_value_right=0.55,    # Phase 1 new
            selected_action=1,     # Phase 1 new
        )

        # Verify state contains new fields
        assert hasattr(publisher.state, 'q_value_left')
        assert hasattr(publisher.state, 'q_value_stay')
        assert hasattr(publisher.state, 'q_value_right')
        assert hasattr(publisher.state, 'action_count_left')
        assert hasattr(publisher.state, 'action_count_stay')
        assert hasattr(publisher.state, 'action_count_right')

    def test_adaptive_rate_updates_during_training_simulation(self):
        """Simulate training at different speeds and verify rate adaptation"""
        publisher = MetricsPublisher()

        # Start slow training
        publisher._calculate_adaptive_update_rate(300)
        slow_interval = publisher._nn_update_interval
        assert slow_interval == 0.016  # ~60Hz

        # Training speeds up
        publisher._calculate_adaptive_update_rate(1500)
        fast_interval = publisher._nn_update_interval
        assert fast_interval == 0.067  # ~15Hz
        assert fast_interval > slow_interval  # Slower updates for faster training


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
