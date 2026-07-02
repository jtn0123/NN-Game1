"""
Tests for the DQN Neural Network.

These tests verify:
    - Network architecture
    - Forward pass
    - Activation capture
    - Weight access
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ai.network import DQN, DuelingDQN, NoisyLinear, SpatialDQN


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def network(config):
    """Create network instance."""
    return DQN(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)


class TestNetworkInitialization:
    """Test network initialization."""

    def test_network_creates_successfully(self, config):
        """Network should initialize without errors."""
        net = DQN(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
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
        """Activations should be captured after forward pass when enabled."""
        network.capture_activations = True  # Enable capture
        x = torch.randn(1, config.STATE_SIZE)
        _ = network(x)

        activations = network.get_activations()
        assert len(activations) > 0

    def test_activation_shapes(self, network, config):
        """Captured activations should have correct shapes."""
        network.capture_activations = True  # Enable capture
        x = torch.randn(1, config.STATE_SIZE)
        _ = network(x)

        activations = network.get_activations()

        # Check each hidden layer activation
        for i, hidden_size in enumerate(config.HIDDEN_LAYERS):
            key = f"layer_{i}"
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
            assert "name" in layer
            assert "neurons" in layer
            assert "type" in layer

    def test_input_and_output_layers(self, network, config):
        """Should have input and output layer info."""
        info = network.get_layer_info()

        assert info[0]["type"] == "input"
        assert info[0]["neurons"] == config.STATE_SIZE

        assert info[-1]["type"] == "output"
        assert info[-1]["neurons"] == config.ACTION_SIZE


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
            hidden_layers=custom_hidden,
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
            hidden_layers=[64],
        )

        x = torch.randn(1, config.STATE_SIZE)
        output = net(x)
        assert output.shape == (1, config.ACTION_SIZE)


class TestDuelingDQN:
    """Test Dueling DQN architecture."""

    @pytest.fixture
    def dueling_network(self, config):
        """Create DuelingDQN instance."""
        return DuelingDQN(
            state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config
        )

    def test_dueling_dqn_creates_successfully(self, config):
        """DuelingDQN should initialize without errors."""
        net = DuelingDQN(
            state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config
        )
        assert net is not None

    def test_dueling_dqn_forward_shape(self, dueling_network, config):
        """Forward pass should return correct shape."""
        batch_size = 16
        x = torch.randn(batch_size, config.STATE_SIZE)
        output = dueling_network(x)
        assert output.shape == (batch_size, config.ACTION_SIZE)

    def test_dueling_dqn_advantage_mean_subtraction(self, config):
        """Q = V + (A - mean(A)) should hold."""
        net = DuelingDQN(
            state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config
        )
        net.eval()

        x = torch.randn(1, config.STATE_SIZE)

        # Get Q-values through normal forward pass
        q_values = net(x)

        # Compute manually using internal layers
        features = x
        for layer in net.feature_layers:
            features = torch.relu(layer(features))

        value = torch.relu(net.value_hidden(features))
        value = net.value_output(value)

        advantage = torch.relu(net.advantage_hidden(features))
        advantage = net.advantage_output(advantage)

        # Manual Q-value computation
        expected_q = value + (advantage - advantage.mean(dim=1, keepdim=True))

        assert torch.allclose(q_values, expected_q, atol=1e-5)

    def test_dueling_dqn_value_stream_shape(self, dueling_network, config):
        """Value stream should output (batch, 1)."""
        batch_size = 8
        x = torch.randn(batch_size, config.STATE_SIZE)

        # Pass through feature layers
        features = x
        for layer in dueling_network.feature_layers:
            features = torch.relu(layer(features))

        value_hidden = torch.relu(dueling_network.value_hidden(features))
        value = dueling_network.value_output(value_hidden)

        assert value.shape == (batch_size, 1)

    def test_dueling_dqn_gradient_flow(self, config):
        """Gradients should flow through both value and advantage streams."""
        # Disable NoisyNets for this test to ensure standard Linear layers
        config.USE_NOISY_NETWORKS = False
        net = DuelingDQN(
            state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config
        )

        x = torch.randn(4, config.STATE_SIZE, requires_grad=True)
        output = net(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for both streams
        assert net.value_hidden.weight.grad is not None
        assert net.advantage_hidden.weight.grad is not None
        # Output layers should have gradients
        assert net.value_output.weight.grad is not None
        assert net.advantage_output.weight.grad is not None

    def test_dueling_dqn_get_weights_includes_output_streams(self, dueling_network, config):
        """Dueling visualization weights should include stream output layers."""
        weights = dueling_network.get_weights()
        expected_count = len(dueling_network.feature_layers) + 4

        assert len(weights) == expected_count
        assert weights[-2].shape == (1, config.HIDDEN_LAYERS[-1])
        assert weights[-1].shape == (config.ACTION_SIZE, config.HIDDEN_LAYERS[-1])

    def test_dueling_dqn_captures_stream_output_activations(self, dueling_network, config):
        """Dueling output stream activations should be available for inspection."""
        dueling_network.capture_activations = True
        _ = dueling_network(torch.randn(1, config.STATE_SIZE))

        activations = dueling_network.get_activations()
        assert "value_output" in activations
        assert "advantage_output" in activations
        assert activations["value_output"].shape == (1, 1)
        assert activations["advantage_output"].shape == (1, config.ACTION_SIZE)


class TestNoisyLinear:
    """Test NoisyLinear layer for exploration."""

    @pytest.fixture
    def noisy_layer(self):
        """Create NoisyLinear layer."""
        return NoisyLinear(in_features=64, out_features=32)

    def test_noisy_linear_creates_successfully(self):
        """NoisyLinear should initialize without errors."""
        layer = NoisyLinear(in_features=10, out_features=5)
        assert layer is not None

    def test_noisy_linear_forward_shape(self, noisy_layer):
        """Forward pass should return correct shape."""
        batch_size = 8
        x = torch.randn(batch_size, 64)
        output = noisy_layer(x)
        assert output.shape == (batch_size, 32)

    def test_noisy_linear_reset_noise_changes_epsilon(self, noisy_layer):
        """reset_noise should generate new noise values."""
        old_weight_epsilon = noisy_layer.weight_epsilon.clone()
        old_bias_epsilon = noisy_layer.bias_epsilon.clone()

        noisy_layer.reset_noise()

        # Noise should be different after reset
        assert not torch.allclose(noisy_layer.weight_epsilon, old_weight_epsilon)
        assert not torch.allclose(noisy_layer.bias_epsilon, old_bias_epsilon)

    def test_noisy_linear_training_vs_eval_mode(self, noisy_layer):
        """Training mode should use noise, eval mode should not."""
        x = torch.randn(4, 64)

        # In training mode, outputs should vary due to noise
        noisy_layer.train()
        noisy_layer.reset_noise()
        train_out1 = noisy_layer(x).clone()
        noisy_layer.reset_noise()
        train_out2 = noisy_layer(x).clone()

        # Due to noise reset, outputs should differ
        assert not torch.allclose(train_out1, train_out2)

        # In eval mode, same input should give same output (no noise)
        noisy_layer.eval()
        eval_out1 = noisy_layer(x).clone()
        eval_out2 = noisy_layer(x).clone()

        assert torch.allclose(eval_out1, eval_out2)

    def test_noisy_linear_weight_formula(self, noisy_layer):
        """Weight should equal mu + sigma * epsilon in training."""
        noisy_layer.train()
        noisy_layer.reset_noise()

        # Compute expected weight using the noisy formula
        expected_weight = (
            noisy_layer.weight_mu + noisy_layer.weight_sigma * noisy_layer.weight_epsilon
        )
        expected_bias = noisy_layer.bias_mu + noisy_layer.bias_sigma * noisy_layer.bias_epsilon

        # Verify the formula by computing output manually vs using forward pass
        x = torch.randn(1, 64)

        # Compute expected output using the noisy weight formula
        expected_output = torch.nn.functional.linear(x, expected_weight, expected_bias)

        # Get actual output from forward pass (uses the same formula internally)
        actual_output = noisy_layer(x)

        # Outputs should match since we computed with the same noise values
        assert torch.allclose(
            actual_output, expected_output, atol=1e-5
        ), f"Output mismatch: expected {expected_output}, got {actual_output}"

        # Also verify shape consistency
        assert noisy_layer.weight_mu.shape == noisy_layer.weight_sigma.shape
        assert noisy_layer.weight_mu.shape == noisy_layer.weight_epsilon.shape

    def test_noisy_linear_parameter_initialization(self):
        """Parameters should be initialized within expected ranges."""
        in_features = 100
        out_features = 50
        layer = NoisyLinear(in_features, out_features, std_init=0.5)

        # mu should be initialized uniformly in [-1/sqrt(in), 1/sqrt(in)]
        mu_range = 1 / (in_features**0.5)
        assert layer.weight_mu.min() >= -mu_range - 0.01
        assert layer.weight_mu.max() <= mu_range + 0.01

        # sigma should be initialized to std_init / sqrt(in_features)
        expected_sigma = 0.5 / (in_features**0.5)
        assert torch.allclose(
            layer.weight_sigma, torch.full_like(layer.weight_sigma, expected_sigma)
        )


class TestSpatialDQN:
    """The convolutional Q-network for grid-structured (spatial) observations."""

    def _layout(self):
        # window 11x19 (=209) + gmap 6x11 (=66) + meta 20 = 295 (the rich state)
        return {"window": (11, 19), "gmap": (6, 11), "meta": 20}

    def test_forward_shape_and_split(self):
        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        net = SpatialDQN(size, 10, Config(), layout)
        out = net(torch.zeros(4, size))
        assert out.shape == (4, 10)

    def test_handles_legacy_layout_without_gmap(self):
        layout = {"window": (9, 11), "gmap": (0, 0), "meta": 20}
        size = 9 * 11 + 0 + 20
        net = SpatialDQN(size, 10, Config(), layout)
        out = net(torch.zeros(2, size))
        assert out.shape == (2, 10)

    def test_handles_expanded_history_metadata(self):
        layout = {"window": (11, 19), "gmap": (6, 11), "meta": 48}
        size = 11 * 19 + 6 * 11 + 48
        net = SpatialDQN(size, 10, Config(), layout)
        out = net(torch.zeros(3, size))
        assert out.shape == (3, 10)

    def test_route_auxiliary_head_predicts_direction_bins_when_enabled(self):
        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        cfg = Config()
        cfg.CRYSTAL_CAVES_ROUTE_AUX_LOSS = True
        net = SpatialDQN(size, 10, cfg, layout)

        out = net(torch.zeros(4, size))
        logits = net.route_aux_logits(torch.zeros(4, size))

        assert out.shape == (4, 10)
        assert logits.shape == (4, 9)

    def test_contact_action_head_predicts_actions_when_enabled(self):
        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        cfg = Config()
        cfg.CRYSTAL_CAVES_CONTACT_ACTION_HEAD = True
        net = SpatialDQN(size, 10, cfg, layout)

        logits = net.contact_action_logits(torch.zeros(4, size))

        assert logits.shape == (4, 10)

    def test_global_average_pool_option(self) -> None:
        """GAP collapses the conv map to one value per filter (translation-invariant):
        fc input = conv_channels(32) + gmap + meta, independent of window size; default
        flatten keeps the full spatial map. Both still produce valid Q-values."""
        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20

        flat = SpatialDQN(size, 10, Config(), layout)
        assert flat.global_pool is False

        cfg = Config()
        cfg.CRYSTAL_CAVES_CNN_GLOBAL_POOL = True
        gap = SpatialDQN(size, 10, cfg, layout)
        assert gap.global_pool is True
        # GAP fc takes 32 conv channels + 66 gmap + 20 meta = 118, far fewer than flatten.
        assert gap.fc.in_features == 32 + 66 + 20
        assert gap.fc.in_features < flat.fc.in_features
        assert gap(torch.zeros(4, size)).shape == (4, 10)

    def test_contact_action_head_detaches_shared_features(self):
        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        cfg = Config()
        cfg.CRYSTAL_CAVES_CONTACT_ACTION_HEAD = True
        net = SpatialDQN(size, 10, cfg, layout)
        loss = net.contact_action_logits(torch.zeros(4, size)).sum()

        loss.backward()

        assert net.contact_action_head.weight.grad is not None
        assert net.fc.weight.grad is None

    def test_agent_uses_cnn_when_enabled(self):
        from src.ai.agent import Agent

        cfg = Config()
        cfg.USE_CNN_STATE = True
        cfg.STATE_LAYOUT = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        agent = Agent(state_size=size, action_size=10, config=cfg)
        assert isinstance(agent.policy_net, SpatialDQN)

    def test_exposes_visualizer_introspection(self):
        """The dashboard NN panel needs get_layer_info/get_weights/activations;
        without them build_nn_snapshot fails and the panel sticks on 'waiting'."""
        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        net = SpatialDQN(size, 3, Config(), layout)

        info = net.get_layer_info()
        assert len(info) >= 3
        assert info[0]["type"] == "input"
        assert info[-1]["type"] == "output"
        assert info[-1]["neurons"] == 3

        weights = net.get_weights()
        assert len(weights) >= 2
        assert all(w.ndim == 2 for w in weights)

        # Activations are only captured when the flag is set (skipped in training).
        assert net.get_activations() == {}
        net.capture_activations = True
        net(torch.zeros(1, size))
        acts = net.get_activations()
        assert "layer_0" in acts and "layer_1" in acts

    def test_noisy_nets_drive_exploration(self):
        """With NoisyNets on, the conv head's output layers carry learned noise so
        resampling changes the Q-values — that IS the exploration (no longer reliant
        on a low epsilon tuned for the MLP). With them off it must be deterministic."""
        from src.ai.network import NoisyLinear

        layout = self._layout()
        size = 11 * 19 + 6 * 11 + 20
        x = torch.zeros(1, size)

        cfg = Config()
        cfg.USE_NOISY_NETWORKS = True
        noisy = SpatialDQN(size, 10, cfg, layout)
        noisy.train()
        assert isinstance(noisy.adv, NoisyLinear)
        noisy.reset_noise()
        a = noisy(x)
        noisy.reset_noise()
        b = noisy(x)
        assert torch.abs(a - b).max().item() > 1e-4  # noise actually perturbs Q-values

        cfg2 = Config()
        cfg2.USE_NOISY_NETWORKS = False
        plain = SpatialDQN(size, 10, cfg2, layout)
        plain.eval()
        plain.reset_noise()  # no-op
        assert torch.allclose(plain(x), plain(x))


class TestDistributionalDQN:
    """C51-style heads should expose distributions while forward() returns Q-values."""

    def _config(self):
        cfg = Config()
        cfg.USE_DISTRIBUTIONAL_DQN = True
        cfg.C51_NUM_ATOMS = 7
        cfg.C51_V_MIN = -3.0
        cfg.C51_V_MAX = 9.0
        cfg.USE_NOISY_NETWORKS = False
        return cfg

    def _assert_distributional_forward(self, net, x, action_size: int, num_atoms: int):
        q_values = net(x)
        logits = net.distributional_logits(x)
        probs = net.distributional_probs(x)

        assert q_values.shape == (x.shape[0], action_size)
        assert logits.shape == (x.shape[0], action_size, num_atoms)
        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=2), torch.ones(x.shape[0], action_size))
        expected_q = (probs * net.support.view(1, 1, num_atoms)).sum(dim=2)
        assert torch.allclose(q_values, expected_q, atol=1e-5)

    def test_plain_dqn_distributional_head_returns_expected_q_values(self):
        cfg = self._config()
        net = DQN(state_size=12, action_size=4, config=cfg, hidden_layers=[16])
        self._assert_distributional_forward(net, torch.zeros(3, 12), 4, cfg.C51_NUM_ATOMS)

    def test_dueling_dqn_distributional_head_returns_expected_q_values(self):
        cfg = self._config()
        net = DuelingDQN(state_size=12, action_size=4, config=cfg, hidden_layers=[16, 8])
        self._assert_distributional_forward(net, torch.zeros(3, 12), 4, cfg.C51_NUM_ATOMS)

    def test_spatial_dqn_distributional_head_returns_expected_q_values(self):
        cfg = self._config()
        layout = {"window": (11, 19), "gmap": (6, 11), "meta": 20}
        size = 11 * 19 + 6 * 11 + 20
        net = SpatialDQN(size, 10, cfg, layout)
        self._assert_distributional_forward(net, torch.zeros(2, size), 10, cfg.C51_NUM_ATOMS)

    def test_distributional_support_is_checkpoint_compatible_buffer(self):
        cfg = self._config()
        layout = {"window": (11, 19), "gmap": (6, 11), "meta": 20}
        size = 11 * 19 + 6 * 11 + 20
        net = SpatialDQN(size, 10, cfg, layout)

        assert hasattr(net, "support")
        assert "support" not in net.state_dict()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
