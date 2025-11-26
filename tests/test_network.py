"""
Tests for the DQN Neural Network.

These tests verify:
    - Network architecture
    - Forward pass
    - Activation capture
    - Weight access
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ai.network import DQN


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def network(config):
    """Create network instance."""
    return DQN(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config
    )


class TestNetworkInitialization:
    """Test network initialization."""
    
    def test_network_creates_successfully(self, config):
        """Network should initialize without errors."""
        net = DQN(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        assert net is not None
    
    def test_correct_input_size(self, network, config):
        """Network should have correct input size."""
        assert network.state_size == config.STATE_SIZE
    
    def test_correct_output_size(self, network, config):
        """Network should have correct output size."""
        assert network.action_size == config.ACTION_SIZE
    
    def test_layers_created(self, network, config):
        """All layers should be created."""
        expected_layers = len(config.HIDDEN_LAYERS) + 1  # Hidden + output
        assert len(network.layers) == expected_layers


class TestForwardPass:
    """Test forward pass."""
    
    def test_forward_single_input(self, network, config):
        """Forward pass with single input."""
        x = torch.randn(1, config.STATE_SIZE)
        output = network(x)
        assert output.shape == (1, config.ACTION_SIZE)
    
    def test_forward_batch_input(self, network, config):
        """Forward pass with batch input."""
        batch_size = 32
        x = torch.randn(batch_size, config.STATE_SIZE)
        output = network(x)
        assert output.shape == (batch_size, config.ACTION_SIZE)
    
    def test_output_not_all_same(self, network, config):
        """Different inputs should produce different outputs."""
        x1 = torch.randn(1, config.STATE_SIZE)
        x2 = torch.randn(1, config.STATE_SIZE)
        
        y1 = network(x1)
        y2 = network(x2)
        
        assert not torch.allclose(y1, y2)
    
    def test_deterministic_output(self, network, config):
        """Same input should produce same output."""
        network.eval()
        x = torch.randn(1, config.STATE_SIZE)
        
        y1 = network(x)
        y2 = network(x)
        
        assert torch.allclose(y1, y2)


class TestActivationCapture:
    """Test activation capture for visualization."""
    
    def test_activations_captured(self, network, config):
        """Activations should be captured after forward pass."""
        x = torch.randn(1, config.STATE_SIZE)
        _ = network(x)
        
        activations = network.get_activations()
        assert len(activations) > 0
    
    def test_activation_shapes(self, network, config):
        """Captured activations should have correct shapes."""
        x = torch.randn(1, config.STATE_SIZE)
        _ = network(x)
        
        activations = network.get_activations()
        
        # Check each hidden layer activation
        for i, hidden_size in enumerate(config.HIDDEN_LAYERS):
            key = f'layer_{i}'
            if key in activations:
                assert activations[key].shape[1] == hidden_size


class TestWeights:
    """Test weight access."""
    
    def test_get_weights(self, network, config):
        """Should be able to get weight matrices."""
        weights = network.get_weights()
        assert len(weights) == len(config.HIDDEN_LAYERS) + 1
    
    def test_weight_shapes(self, network, config):
        """Weight matrices should have correct shapes."""
        weights = network.get_weights()
        
        # First layer: input -> hidden1
        assert weights[0].shape == (config.HIDDEN_LAYERS[0], config.STATE_SIZE)
        
        # Output layer: last_hidden -> output
        assert weights[-1].shape == (config.ACTION_SIZE, config.HIDDEN_LAYERS[-1])


class TestLayerInfo:
    """Test layer info for visualization."""
    
    def test_get_layer_info(self, network):
        """Should return layer information."""
        info = network.get_layer_info()
        assert len(info) > 0
    
    def test_layer_info_format(self, network):
        """Layer info should have expected format."""
        info = network.get_layer_info()
        
        for layer in info:
            assert 'name' in layer
            assert 'neurons' in layer
            assert 'type' in layer
    
    def test_input_and_output_layers(self, network, config):
        """Should have input and output layer info."""
        info = network.get_layer_info()
        
        assert info[0]['type'] == 'input'
        assert info[0]['neurons'] == config.STATE_SIZE
        
        assert info[-1]['type'] == 'output'
        assert info[-1]['neurons'] == config.ACTION_SIZE


class TestParameterCount:
    """Test parameter counting."""
    
    def test_count_parameters(self, network):
        """Should count all trainable parameters."""
        count = network.count_parameters()
        assert count > 0
    
    def test_parameter_count_matches(self, network):
        """Count should match PyTorch's count."""
        counted = network.count_parameters()
        actual = sum(p.numel() for p in network.parameters() if p.requires_grad)
        assert counted == actual


class TestCustomArchitecture:
    """Test custom network architectures."""
    
    def test_custom_hidden_layers(self, config):
        """Should work with custom hidden layer sizes."""
        custom_hidden = [512, 256, 128, 64]
        net = DQN(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config,
            hidden_layers=custom_hidden
        )
        
        assert net.hidden_sizes == custom_hidden
        
        # Verify with forward pass
        x = torch.randn(1, config.STATE_SIZE)
        output = net(x)
        assert output.shape == (1, config.ACTION_SIZE)
    
    def test_single_hidden_layer(self, config):
        """Should work with single hidden layer."""
        net = DQN(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config,
            hidden_layers=[64]
        )
        
        x = torch.randn(1, config.STATE_SIZE)
        output = net(x)
        assert output.shape == (1, config.ACTION_SIZE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

