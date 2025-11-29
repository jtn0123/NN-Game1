"""
Tests for Phase 2: Interactive Exploration - Neuron Inspection

Tests verify:
- Neuron inspection data collection and retrieval
- Layer analysis statistics
- API endpoint functionality
- Integration with MetricsPublisher
"""

import pytest
import numpy as np
from src.web.server import (
    MetricsPublisher,
    NeuronInspectionData,
    LayerAnalysisData,
)


class TestNeuronInspectionData:
    """Test NeuronInspectionData dataclass"""

    def test_neuron_inspection_creation(self):
        """Verify neuron inspection data can be created"""
        data = NeuronInspectionData(
            layer_idx=0,
            neuron_idx=5,
            layer_name="hidden_0",
        )

        assert data.layer_idx == 0
        assert data.neuron_idx == 5
        assert data.layer_name == "hidden_0"
        assert data.current_activation == 0.0

    def test_neuron_inspection_to_dict(self):
        """Verify neuron inspection data converts to dict correctly"""
        data = NeuronInspectionData(
            layer_idx=1,
            neuron_idx=10,
            layer_name="hidden_1",
            current_activation=0.5,
            activation_history=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        # Manually set weights to test the stats computation
        data.incoming_weights = [0.1, -0.2, 0.15]
        data.incoming_weight_stats = {
            'mean': 0.016666,
            'std': 0.162,
            'min': -0.2,
            'max': 0.15,
        }
        data.outgoing_weights = [0.2, -0.1]
        data.outgoing_weight_stats = {
            'mean': 0.05,
            'std': 0.15,
            'min': -0.1,
            'max': 0.2,
        }
        data.q_value_contributions = {'left': 0.1, 'stay': 0.3, 'right': 0.05}

        result = data.to_dict()

        assert result['layer_idx'] == 1
        assert result['neuron_idx'] == 10
        assert result['layer_name'] == "hidden_1"
        assert result['current_activation'] == 0.5
        assert len(result['activation_history']) == 5
        assert result['incoming_weight_stats']['mean'] == pytest.approx(0.016666, abs=0.001)
        assert result['q_value_contributions']['stay'] == 0.3


class TestLayerAnalysisData:
    """Test LayerAnalysisData dataclass"""

    def test_layer_analysis_creation(self):
        """Verify layer analysis data can be created"""
        data = LayerAnalysisData(
            layer_idx=0,
            layer_name="input",
            neuron_count=55,
        )

        assert data.layer_idx == 0
        assert data.layer_name == "input"
        assert data.neuron_count == 55

    def test_layer_analysis_to_dict(self):
        """Verify layer analysis data converts to dict correctly"""
        data = LayerAnalysisData(
            layer_idx=1,
            layer_name="hidden_0",
            neuron_count=256,
            avg_activation=0.3,
            activation_std=0.2,
            dead_neuron_count=5,
            saturated_neuron_count=10,
        )

        result = data.to_dict()

        assert result['layer_idx'] == 1
        assert result['neuron_count'] == 256
        assert result['avg_activation'] == 0.3
        assert result['dead_neuron_percent'] == pytest.approx(1.953, abs=0.01)
        assert result['saturated_percent'] == pytest.approx(3.906, abs=0.01)


class TestNeuronInspectionStorage:
    """Test neuron inspection storage in MetricsPublisher"""

    def test_update_neuron_inspection(self):
        """Verify neuron inspection data is stored and retrieved"""
        publisher = MetricsPublisher()

        # Update neuron inspection data
        publisher.update_neuron_inspection(
            layer_idx=0,
            neuron_idx=5,
            layer_name="hidden_0",
            current_activation=0.45,
            activation_history=[0.1, 0.2, 0.3, 0.4, 0.45],
            incoming_weights=np.array([0.1, -0.2, 0.15]),
            outgoing_weights=np.array([0.2, -0.1, 0.05, 0.08]),
            q_contributions={'left': 0.1, 'stay': 0.3, 'right': 0.05},
        )

        # Retrieve neuron details
        details = publisher.get_neuron_details(0, 5)

        assert details['layer_idx'] == 0
        assert details['neuron_idx'] == 5
        assert details['current_activation'] == 0.45
        assert details['q_value_contributions']['stay'] == 0.3

    def test_neuron_not_found(self):
        """Verify appropriate error when neuron not found"""
        publisher = MetricsPublisher()

        details = publisher.get_neuron_details(99, 99)

        assert 'error' in details

    def test_activation_history_rolling_window(self):
        """Verify activation history maintains rolling window (500 steps)"""
        publisher = MetricsPublisher()

        # Create a long activation history
        long_history = list(np.linspace(0.1, 0.9, 1000))

        publisher.update_neuron_inspection(
            layer_idx=0,
            neuron_idx=0,
            layer_name="hidden_0",
            current_activation=0.9,
            activation_history=long_history,
        )

        details = publisher.get_neuron_details(0, 0)

        # Should keep last 500 for storage, last 100 for transmission
        assert len(details['activation_history']) == 100  # Last 100


class TestLayerAnalysisStatistics:
    """Test layer analysis statistics computation"""

    def test_update_layer_analysis_activations(self):
        """Verify layer analysis computes activation statistics"""
        publisher = MetricsPublisher()

        # Create sample activations
        activations = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="hidden_0",
            neuron_count=10,
            activations=activations,
        )

        analysis = publisher.get_layer_analysis(0)

        assert analysis['neuron_count'] == 10
        assert abs(analysis['avg_activation'] - 0.45) < 0.01
        assert analysis['activation_min'] == 0.0
        assert analysis['activation_max'] == 0.9

    def test_dead_neuron_detection(self):
        """Verify dead neuron detection (activation < 0.01)"""
        publisher = MetricsPublisher()

        # Create activations with some dead neurons
        activations = np.array([0.001, 0.001, 0.5, 0.6, 0.7, 0.0, 0.0, 0.8, 0.9, 0.95])

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="hidden_0",
            neuron_count=10,
            activations=activations,
        )

        analysis = publisher.get_layer_analysis(0)

        assert analysis['dead_neuron_count'] == 4  # 0.001, 0.001, 0.0, 0.0
        assert analysis['dead_neuron_percent'] == pytest.approx(40.0)

    def test_saturated_neuron_detection(self):
        """Verify saturated neuron detection (activation > 0.95)"""
        publisher = MetricsPublisher()

        # Create activations with some saturated neurons
        activations = np.array([0.96, 0.97, 0.98, 0.5, 0.6, 0.7, 0.8, 0.0, 0.1, 0.2])

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="hidden_0",
            neuron_count=10,
            activations=activations,
        )

        analysis = publisher.get_layer_analysis(0)

        assert analysis['saturated_neuron_count'] == 3
        assert analysis['saturated_percent'] == pytest.approx(30.0)

    def test_weight_statistics(self):
        """Verify weight statistics computation"""
        publisher = MetricsPublisher()

        activations = np.ones(10) * 0.5
        weights = np.array([[0.1, -0.2], [0.15, -0.05], [0.3, -0.1]])

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="hidden_0",
            neuron_count=10,
            activations=activations,
            weights=weights,
        )

        analysis = publisher.get_layer_analysis(0)

        assert 'weight_mean' in analysis
        assert 'weight_std' in analysis
        assert 'weight_histogram' in analysis
        assert len(analysis['weight_histogram']) == 20

    def test_gradient_statistics(self):
        """Verify gradient statistics computation"""
        publisher = MetricsPublisher()

        activations = np.ones(10) * 0.5
        gradients = np.array([[0.01, -0.02], [0.015, -0.005]])

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="hidden_0",
            neuron_count=10,
            activations=activations,
            gradients=gradients,
        )

        analysis = publisher.get_layer_analysis(0)

        assert analysis['gradient_mean'] > 0
        assert analysis['gradient_max_magnitude'] > 0


class TestMultipleLayers:
    """Test analysis of multiple layers"""

    def test_get_all_layer_analysis(self):
        """Verify all layer analysis can be retrieved"""
        publisher = MetricsPublisher()

        # Add analysis for multiple layers
        for layer_idx in range(3):
            activations = np.random.rand(256)
            publisher.update_layer_analysis(
                layer_idx=layer_idx,
                layer_name=f"hidden_{layer_idx}",
                neuron_count=256,
                activations=activations,
            )

        layers = publisher.get_all_layer_analysis()

        assert len(layers) == 3
        assert layers[0]['layer_idx'] == 0
        assert layers[1]['layer_idx'] == 1
        assert layers[2]['layer_idx'] == 2

    def test_layer_analysis_sorted_by_index(self):
        """Verify layers are sorted by index"""
        publisher = MetricsPublisher()

        # Add layers out of order
        for layer_idx in [2, 0, 1]:
            activations = np.ones(256) * 0.5
            publisher.update_layer_analysis(
                layer_idx=layer_idx,
                layer_name=f"hidden_{layer_idx}",
                neuron_count=256,
                activations=activations,
            )

        layers = publisher.get_all_layer_analysis()

        # Should be sorted
        assert layers[0]['layer_idx'] == 0
        assert layers[1]['layer_idx'] == 1
        assert layers[2]['layer_idx'] == 2


class TestPhase2Integration:
    """Integration tests for Phase 2 features"""

    def test_neuron_and_layer_together(self):
        """Verify neuron and layer analysis work together"""
        publisher = MetricsPublisher()

        # Update layer analysis
        activations = np.random.rand(256)
        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="hidden_0",
            neuron_count=256,
            activations=activations,
        )

        # Update neuron inspection for a specific neuron
        publisher.update_neuron_inspection(
            layer_idx=0,
            neuron_idx=42,
            layer_name="hidden_0",
            current_activation=activations[42],
        )

        # Both should be retrievable
        layer_analysis = publisher.get_layer_analysis(0)
        neuron_details = publisher.get_neuron_details(0, 42)

        assert layer_analysis['neuron_count'] == 256
        assert neuron_details['layer_idx'] == 0
        assert neuron_details['neuron_idx'] == 42

    def test_callback_registration(self):
        """Verify callback registration works"""
        publisher = MetricsPublisher()

        callbacks_called = {'neuron': False, 'layer': False}

        def on_neuron(data):
            callbacks_called['neuron'] = True

        def on_layer(data):
            callbacks_called['layer'] = True

        publisher.on_neuron_select(on_neuron)
        publisher.on_layer_analysis(on_layer)

        # Callbacks are registered (actual calling happens in WebDashboard)
        assert len(publisher._on_neuron_select_callbacks) == 1
        assert len(publisher._on_layer_analysis_callbacks) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
