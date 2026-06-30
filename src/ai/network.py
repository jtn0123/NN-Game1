"""
Deep Q-Network (DQN) Architecture
=================================

The neural network that approximates Q-values for state-action pairs.

Theory:
    Q-Learning aims to learn Q(s, a) = expected future reward
    We use a neural network to approximate this function

    Input:  State vector (ball pos, paddle pos, brick states)
    Output: Q-value for each possible action

The network learns by minimizing TD (Temporal Difference) error:
    Loss = (Q(s,a) - (r + γ * max_a' Q(s', a')))²

Key Features:
    - Configurable hidden layers
    - Support for different activation functions
    - Weight initialization for stable training
    - Forward hooks for visualization
"""

import math
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def _distributional_enabled(config: Config) -> bool:
    return bool(getattr(config, "USE_DISTRIBUTIONAL_DQN", False))


def _distributional_num_atoms(config: Config) -> int:
    return int(getattr(config, "C51_NUM_ATOMS", 51))


def _distributional_support(config: Config) -> torch.Tensor:
    return torch.linspace(
        float(getattr(config, "C51_V_MIN", -20.0)),
        float(getattr(config, "C51_V_MAX", 120.0)),
        _distributional_num_atoms(config),
    )


def _expected_q_from_logits(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=2)
    return torch.sum(probs * support.view(1, 1, -1), dim=2)


def _distributional_action_weight(
    layer: nn.Module,
    *,
    action_size: int,
    num_atoms: int,
) -> np.ndarray:
    if isinstance(layer, NoisyLinear):
        weight = layer.visualization_weight()
    elif isinstance(layer, nn.Linear):
        weight = layer.weight.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported layer type for visualization: {type(layer)!r}")
    if weight.shape[0] == action_size * num_atoms:
        return weight.reshape(action_size, num_atoms, -1).mean(axis=1)
    return weight


class NoisyLinear(nn.Module):
    """
    Linear layer with learnable parameter noise for exploration.

    Replaces epsilon-greedy exploration with learned, state-dependent exploration.
    The noise parameters are learned during training, allowing the network
    to learn when and how much to explore.

    Reference:
        Fortunato et al., 2017 - "Noisy Networks for Exploration"
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable mean parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise parameters (sigma)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learned, regenerated each forward pass)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Sample new noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features, self.weight_epsilon.device)
        epsilon_out = self._scale_noise(self.out_features, self.bias_epsilon.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        """Factorized Gaussian noise (more efficient than independent)."""
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def visualization_weight(self) -> np.ndarray:
        """Return the deterministic weight matrix used for visualization."""
        return self.weight_mu.detach().cpu().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights during training."""
        if self.training:
            # Use noisy weights (noise is reset before forward pass by caller)
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon  # type: ignore[operator]
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon  # type: ignore[operator]
        else:
            # Use mean parameters during evaluation (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQN(nn.Module):
    """
    Deep Q-Network for reinforcement learning.

    Architecture:
        Input Layer → Hidden Layers → Output Layer

    The network outputs Q-values for all possible actions,
    and the agent selects the action with the highest Q-value.

    Attributes:
        layers (nn.ModuleList): All neural network layers
        activations (Dict): Stores activations for visualization

    Example:
        >>> config = Config()
        >>> net = DQN(state_size=55, action_size=3, config=config)
        >>> state = torch.randn(1, 55)
        >>> q_values = net(state)  # Shape: (1, 3)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[Config] = None,
        hidden_layers: Optional[List[int]] = None,
    ):
        """
        Initialize the DQN.

        Args:
            state_size: Dimension of state input
            action_size: Number of possible actions (output dimension)
            config: Configuration object
            hidden_layers: Override config's hidden layer sizes
        """
        super(DQN, self).__init__()

        self.config = config or Config()
        self.state_size = state_size
        self.action_size = action_size
        # Op 2 (learnable route): the last `route_label_dims` of the state are a TRAILING
        # supervision label (the geodesic route direction), NOT policy input — slice them off
        # so the trunk reads only `core_in` features. route_label=0 (the default) leaves the
        # network byte-identical to before.
        layout = getattr(self.config, "STATE_LAYOUT", None)
        self.route_label_dims = int(layout.get("route_label", 0)) if layout else 0
        self.core_in = self.state_size - self.route_label_dims
        self.route_aux_enabled = bool(getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_LOSS", False))
        self.distributional = _distributional_enabled(self.config)
        self.num_atoms = _distributional_num_atoms(self.config)
        self.register_buffer("support", _distributional_support(self.config), persistent=False)

        # Use provided hidden layers or config
        self.hidden_sizes = hidden_layers or self.config.HIDDEN_LAYERS

        # Store activations for visualization
        self.activations: Dict[str, torch.Tensor] = {}

        # Flag to enable/disable activation capture (disable during training for speed)
        self.capture_activations = False

        # Cache activation function (avoids dict lookup every forward pass)
        self._activation_fn = self._get_activation_fn()

        # Build network layers
        self.layers = nn.ModuleList()
        self._build_network()

        # Initialize weights
        self._init_weights()

        # Register hooks for capturing activations
        self._register_hooks()

    def _build_network(self) -> None:
        """Construct the neural network layers."""
        output_size = self.action_size * self.num_atoms if self.distributional else self.action_size
        layer_sizes = [self.core_in] + self.hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

    def _init_weights(self) -> None:
        """
        Initialize weights using Xavier/Glorot initialization.
        This helps with training stability.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def _get_activation_fn(self) -> Callable[..., Any]:
        """Get the activation function based on config."""
        activation_map: Dict[str, Callable[..., Any]] = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "tanh": torch.tanh,
            "elu": F.elu,
        }
        result = activation_map.get(self.config.ACTIVATION, F.relu)
        return cast(Callable[..., Any], result)

    def _register_hooks(self) -> None:
        """Register forward hooks to capture activations (only when enabled)."""

        def get_activation(name):
            def hook(module, input, output):
                # Only capture when flag is set (skip during training for speed)
                if self.capture_activations:
                    self.activations[name] = output.detach()

            return hook

        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(get_activation(f"layer_{i}"))

    def _raw_output(self, state: torch.Tensor) -> torch.Tensor:
        x = state[:, : self.core_in] if self.route_label_dims else state

        # Apply hidden layers with activation (use cached function for speed)
        for i, layer in enumerate(self.layers[:-1]):
            x = self._activation_fn(layer(x))

        # Output layer (no activation - raw Q-values)
        return self.layers[-1](x)

    def distributional_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw C51 logits shaped (batch, actions, atoms)."""
        if not self.distributional:
            raise RuntimeError("distributional DQN head is disabled")
        return self._raw_output(state).view(-1, self.action_size, self.num_atoms)

    def distributional_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized C51 probabilities shaped (batch, actions, atoms)."""
        return F.softmax(self.distributional_logits(state), dim=2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, state_size)

        Returns:
            Expected Q-values tensor of shape (batch_size, action_size)
        """
        raw = self._raw_output(state)
        if self.distributional:
            logits = raw.view(-1, self.action_size, self.num_atoms)
            return _expected_q_from_logits(logits, cast(torch.Tensor, self.support))
        return raw

    def get_layer_info(self) -> List[Dict]:
        """
        Get information about each layer for visualization.

        Returns:
            List of dicts with layer metadata
        """
        info = []

        # Input layer (virtual - no actual layer)
        info.append({"name": "Input", "neurons": self.state_size, "type": "input"})

        # Hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            info.append(
                {
                    "name": f"Hidden {i + 1}",
                    "neurons": layer.out_features,
                    "type": "hidden",
                }
            )

        output_name = "Output (C51 Q)" if self.distributional else "Output"
        info.append({"name": output_name, "neurons": self.action_size, "type": "output"})

        return info

    def get_weights(self) -> List[np.ndarray]:
        """
        Get all weight matrices as numpy arrays.

        Returns:
            List of weight matrices (for visualization)
        """
        weights = []
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if self.distributional and index == len(self.layers) - 1:
                    weights.append(
                        _distributional_action_weight(
                            layer,
                            action_size=self.action_size,
                            num_atoms=self.num_atoms,
                        )
                    )
                else:
                    weights.append(layer.weight.detach().cpu().numpy())
        return weights

    def get_activations(self) -> Dict[str, np.ndarray]:
        """
        Get the most recent activations from all layers.

        Returns:
            Dict mapping layer names to activation arrays
        """
        return {name: act.cpu().numpy() for name, act in self.activations.items()}

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture.

    Separates the Q-value estimation into two streams:
        - Value stream V(s): How good is this state overall?
        - Advantage stream A(s,a): How much better is each action vs average?

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))

    This helps the network learn:
        - State values independent of actions (useful when actions don't matter much)
        - Relative action advantages (useful when action choice is critical)

    Reference:
        Wang et al., 2016 - "Dueling Network Architectures for Deep Reinforcement Learning"
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[Config] = None,
        hidden_layers: Optional[List[int]] = None,
    ):
        """
        Initialize the Dueling DQN.

        Args:
            state_size: Dimension of state input
            action_size: Number of possible actions (output dimension)
            config: Configuration object
            hidden_layers: Override config's hidden layer sizes
        """
        super(DuelingDQN, self).__init__()

        self.config = config or Config()
        self.state_size = state_size
        self.action_size = action_size
        # Op 2 (learnable route): the last `route_label_dims` of the state are a TRAILING
        # supervision label (the geodesic route direction), NOT policy input — slice them off
        # so the trunk reads only `core_in` features. route_label=0 (the default) leaves the
        # network byte-identical to before.
        layout = getattr(self.config, "STATE_LAYOUT", None)
        self.route_label_dims = int(layout.get("route_label", 0)) if layout else 0
        self.core_in = self.state_size - self.route_label_dims
        self.route_aux_enabled = bool(getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_LOSS", False))
        self.distributional = _distributional_enabled(self.config)
        self.num_atoms = _distributional_num_atoms(self.config)
        self.register_buffer("support", _distributional_support(self.config), persistent=False)

        # Use provided hidden layers or config
        self.hidden_sizes = hidden_layers or self.config.HIDDEN_LAYERS

        # Store activations for visualization
        self.activations: Dict[str, torch.Tensor] = {}

        # Flag to enable/disable activation capture (disable during training for speed)
        self.capture_activations = False

        # Cache activation function (avoids dict lookup every forward pass)
        self._activation_fn = self._get_activation_fn()

        # Build network layers
        self._build_network()

        # Initialize weights
        self._init_weights()

        # Register hooks for capturing activations
        self._register_hooks()

    def _build_network(self) -> None:
        """Construct the dueling network architecture."""
        # Shared feature layers (all but the last hidden layer)
        self.feature_layers = nn.ModuleList()

        # Build shared feature extraction layers
        layer_sizes = [self.core_in] + self.hidden_sizes[:-1]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.feature_layers.append(layer)

        # Get the size of the last shared layer (or core_in if no hidden layers)
        if len(self.hidden_sizes) > 1:
            shared_output_size = self.hidden_sizes[-2]
        else:
            shared_output_size = self.core_in

        # Final hidden layer size for streams
        stream_hidden_size = self.hidden_sizes[-1]

        # Value stream: shared -> hidden -> 1 (state value)
        self.value_hidden = nn.Linear(shared_output_size, stream_hidden_size)

        # Advantage stream: shared -> hidden -> action_size (per-action advantage)
        self.advantage_hidden = nn.Linear(shared_output_size, stream_hidden_size)

        # Final output layers - use NoisyLinear if enabled
        use_noisy = getattr(self.config, "USE_NOISY_NETWORKS", False)
        value_output_size = self.num_atoms if self.distributional else 1
        advantage_output_size = (
            self.action_size * self.num_atoms if self.distributional else self.action_size
        )
        if use_noisy:
            noisy_std = getattr(self.config, "NOISY_STD_INIT", 0.5)
            self.value_output = NoisyLinear(stream_hidden_size, value_output_size, noisy_std)
            self.advantage_output = NoisyLinear(
                stream_hidden_size, advantage_output_size, noisy_std
            )
        else:
            self.value_output = nn.Linear(stream_hidden_size, value_output_size)  # type: ignore[assignment]
            self.advantage_output = nn.Linear(stream_hidden_size, advantage_output_size)  # type: ignore[assignment]

        # Op 2 (learnable route): supervised head predicting the 9-way geodesic route
        # direction from the SHARED trunk (the route label is sliced off the input, so this
        # must be learned). Trained only as an auxiliary loss; unused at action time.
        if self.route_aux_enabled:
            self.route_aux_head = nn.Linear(shared_output_size, 9)

        # For compatibility with DQN interface, create a layers list
        self.layers = nn.ModuleList(
            list(self.feature_layers)
            + [
                self.value_hidden,
                self.value_output,
                self.advantage_hidden,
                self.advantage_output,
            ]
        )

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def _get_activation_fn(self) -> Callable[..., Any]:
        """Get the activation function based on config."""
        activation_map: Dict[str, Callable[..., Any]] = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "tanh": torch.tanh,
            "elu": F.elu,
        }
        result = activation_map.get(self.config.ACTIVATION, F.relu)
        return cast(Callable[..., Any], result)

    def _register_hooks(self) -> None:
        """Register forward hooks to capture activations (only when enabled)."""

        def get_activation(name):
            def hook(module, input, output):
                # Only capture when flag is set (skip during training for speed)
                if self.capture_activations:
                    self.activations[name] = output.detach()

            return hook

        # Register hooks on feature layers
        for i, layer in enumerate(self.feature_layers):
            layer.register_forward_hook(get_activation(f"layer_{i}"))

        # Register hooks on stream layers
        self.value_hidden.register_forward_hook(get_activation("value_hidden"))
        self.advantage_hidden.register_forward_hook(get_activation("advantage_hidden"))
        self.value_output.register_forward_hook(get_activation("value_output"))
        self.advantage_output.register_forward_hook(get_activation("advantage_output"))

    def _shared_features(self, state: torch.Tensor) -> torch.Tensor:
        # Slice off the trailing route-aux label (Op 2) so the policy never reads it.
        x = state[:, : self.core_in] if self.route_label_dims else state

        # Pass through shared feature layers (use cached activation fn for speed)
        for layer in self.feature_layers:
            x = self._activation_fn(layer(x))
        return x

    def route_aux_logits(self, state: torch.Tensor) -> torch.Tensor:
        """9-way geodesic route-direction logits from the shared trunk (Op 2 aux head)."""
        if not self.route_aux_enabled:
            raise RuntimeError("route-aux head is disabled")
        return self.route_aux_head(self._shared_features(state))

    def _dueling_outputs(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._shared_features(state)

        # Value stream
        value = self._activation_fn(self.value_hidden(x))
        value = self.value_output(value)  # Shape: (batch, 1)

        # Advantage stream
        advantage = self._activation_fn(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage)  # Shape: (batch, action_size)

        return value, advantage

    def distributional_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw C51 logits shaped (batch, actions, atoms)."""
        if not self.distributional:
            raise RuntimeError("distributional DQN head is disabled")
        value, advantage = self._dueling_outputs(state)
        value = value.view(-1, 1, self.num_atoms)
        advantage = advantage.view(-1, self.action_size, self.num_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def distributional_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized C51 probabilities shaped (batch, actions, atoms)."""
        return F.softmax(self.distributional_logits(state), dim=2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.

        Args:
            state: Input state tensor of shape (batch_size, state_size)

        Returns:
            Expected Q-values tensor of shape (batch_size, action_size)
        """
        if self.distributional:
            return _expected_q_from_logits(
                self.distributional_logits(state),
                cast(torch.Tensor, self.support),
            )

        value, advantage = self._dueling_outputs(state)
        # Combine streams: Q = V + (A - mean(A))
        # Subtracting mean ensures identifiability and stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def reset_noise(self):
        """Reset noise for all noisy layers."""
        if isinstance(self.value_output, NoisyLinear):
            self.value_output.reset_noise()
        if isinstance(self.advantage_output, NoisyLinear):
            self.advantage_output.reset_noise()

    def get_layer_info(self) -> List[Dict]:
        """
        Get information about each layer for visualization.

        Returns:
            List of dicts with layer metadata
        """
        info = []

        # Input layer (virtual - no actual layer)
        info.append({"name": "Input", "neurons": self.state_size, "type": "input"})

        # Shared feature layers
        for i, layer in enumerate(self.feature_layers):
            info.append(
                {
                    "name": f"Shared {i + 1}",
                    "neurons": layer.out_features,
                    "type": "hidden",
                }
            )

        # Value stream
        info.append(
            {
                "name": "Value",
                "neurons": self.value_hidden.out_features,
                "type": "value_stream",
            }
        )

        # Advantage stream
        info.append(
            {
                "name": "Advantage",
                "neurons": self.advantage_hidden.out_features,
                "type": "advantage_stream",
            }
        )

        # Output layer
        output_name = "Output (C51 Q)" if self.distributional else "Output (Q)"
        info.append({"name": output_name, "neurons": self.action_size, "type": "output"})

        return info

    def get_weights(self) -> List[np.ndarray]:
        """
        Get all weight matrices as numpy arrays.

        Returns:
            List of weight matrices (for visualization)
        """

        def layer_weight(layer: nn.Module, *, distributional_output: bool = False) -> np.ndarray:
            if isinstance(layer, NoisyLinear):
                if distributional_output:
                    return _distributional_action_weight(
                        layer,
                        action_size=self.action_size,
                        num_atoms=self.num_atoms,
                    )
                return layer.visualization_weight()
            if isinstance(layer, nn.Linear):
                if distributional_output:
                    return _distributional_action_weight(
                        layer,
                        action_size=self.action_size,
                        num_atoms=self.num_atoms,
                    )
                return layer.weight.detach().cpu().numpy()
            raise TypeError(f"Unsupported layer type for visualization: {type(layer)!r}")

        weights = []
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer_weight(layer))
        weights.append(layer_weight(self.value_hidden))
        weights.append(layer_weight(self.advantage_hidden))
        weights.append(layer_weight(self.value_output))
        weights.append(
            layer_weight(self.advantage_output, distributional_output=self.distributional)
        )
        return weights

    def get_activations(self) -> Dict[str, np.ndarray]:
        """
        Get the most recent activations from all layers.

        Returns:
            Dict mapping layer names to activation arrays
        """
        return {name: act.cpu().numpy() for name, act in self.activations.items()}

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpatialDQN(nn.Module):
    """Convolutional Q-network for grid-structured observations. The flat state is
    split into a 2D perception window (convolved), an optional coarse 2D global
    map, and metadata scalars; conv features are concatenated with the map +
    metadata and fed to a (dueling) value/advantage head. This exploits the
    spatial locality a plain MLP discards — the right architecture for the rich
    Crystal Caves state (a 19x11 window + an objective map). Exploration is
    epsilon-greedy (no NoisyNets in the conv head)."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[Config] = None,
        layout: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.config = config or Config()
        self.state_size = state_size
        self.action_size = action_size
        self.activations: Dict[str, torch.Tensor] = {}
        self.capture_activations = False

        layout = layout or getattr(self.config, "STATE_LAYOUT", None)
        if layout is None:
            raise ValueError("SpatialDQN requires a STATE_LAYOUT (window/gmap/meta dims)")
        wr, wc = layout["window"]
        gr, gc = layout.get("gmap", (0, 0))
        self.win_rows, self.win_cols = wr, wc
        self.win_size = wr * wc
        self.gmap_size = gr * gc
        self.meta_size = layout["meta"]
        self.dueling = bool(getattr(self.config, "USE_DUELING", True))
        self.route_aux_enabled = bool(getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_LOSS", False))
        self.contact_action_head_enabled = bool(
            getattr(self.config, "CRYSTAL_CAVES_CONTACT_ACTION_HEAD", False)
        )
        self.distributional = _distributional_enabled(self.config)
        self.num_atoms = _distributional_num_atoms(self.config)
        self.register_buffer("support", _distributional_support(self.config), persistent=False)

        # Structured exploration: like DuelingDQN, put NoisyNets on the OUTPUT
        # layers (the Rainbow placement) so the conv net actually explores via
        # learned noise instead of relying on a low epsilon tuned for the MLP.
        self.use_noisy = bool(getattr(self.config, "USE_NOISY_NETWORKS", False))
        noisy_std = getattr(self.config, "NOISY_STD_INIT", 0.5)

        def out_layer(in_f: int, out_f: int) -> nn.Module:
            return NoisyLinear(in_f, out_f, noisy_std) if self.use_noisy else nn.Linear(in_f, out_f)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # Global average pooling over the conv feature map (translation-invariant) vs
        # the default flatten (which preserves absolute position and tends to memorize
        # layouts). Off by default to keep existing checkpoints/behavior; flag it on to
        # test whether translation invariance closes the train/test generalization gap.
        self.global_pool = bool(getattr(self.config, "CRYSTAL_CAVES_CNN_GLOBAL_POOL", False))
        conv_out = 32 if self.global_pool else 32 * max(1, wr // 2) * max(1, wc // 2)
        hidden = (self.config.HIDDEN_LAYERS or [256, 128])[0]
        merged = conv_out + self.gmap_size + self.meta_size
        self.fc = nn.Linear(merged, hidden)
        if self.dueling:
            value_output_size = self.num_atoms if self.distributional else 1
            advantage_output_size = (
                action_size * self.num_atoms if self.distributional else action_size
            )
            self.value = out_layer(hidden, value_output_size)
            self.adv = out_layer(hidden, advantage_output_size)
        else:
            output_size = action_size * self.num_atoms if self.distributional else action_size
            self.head = out_layer(hidden, output_size)
        if self.route_aux_enabled:
            self.route_aux_head = nn.Linear(hidden, 9)
        if self.contact_action_head_enabled:
            self.contact_action_head = nn.Linear(hidden, action_size)
        # Xavier-init the plain conv/linear layers; NoisyLinear self-inits.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _features(self, state: torch.Tensor) -> torch.Tensor:
        win = state[:, : self.win_size].view(-1, 1, self.win_rows, self.win_cols)
        rest = state[:, self.win_size :]
        gmap = rest[:, : self.gmap_size]
        # Bound the meta slice so any trailing route-aux label (Op 2) is excluded from input.
        meta = rest[:, self.gmap_size : self.gmap_size + self.meta_size]
        x = F.relu(self.conv1(win))
        x = self.pool(F.relu(self.conv2(x)))
        if self.capture_activations:
            # one value per conv filter (mean over the spatial grid) for the viz
            self.activations["layer_0"] = x.mean(dim=(2, 3)).detach()
        x = x.mean(dim=(2, 3)) if self.global_pool else torch.flatten(x, 1)
        x = torch.cat([x, gmap, meta], dim=1)
        x = F.relu(self.fc(x))
        if self.capture_activations:
            self.activations["layer_1"] = x.detach()
        return x

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self._features(state)
        if self.dueling:
            v = self.value(x)
            a = self.adv(x)
            if self.distributional:
                v = v.view(-1, 1, self.num_atoms)
                a = a.view(-1, self.action_size, self.num_atoms)
                logits = v + (a - a.mean(dim=1, keepdim=True))
                q = _expected_q_from_logits(logits, cast(torch.Tensor, self.support))
            else:
                q = v + (a - a.mean(dim=1, keepdim=True))
        else:
            raw = self.head(x)
            if self.distributional:
                logits = raw.view(-1, self.action_size, self.num_atoms)
                q = _expected_q_from_logits(logits, cast(torch.Tensor, self.support))
            else:
                q = raw
        if self.capture_activations:
            self.activations["layer_2"] = q.detach()
        return q

    def distributional_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw C51 logits shaped (batch, actions, atoms)."""
        if not self.distributional:
            raise RuntimeError("distributional DQN head is disabled")
        x = self._features(state)
        if self.dueling:
            v = self.value(x).view(-1, 1, self.num_atoms)
            a = self.adv(x).view(-1, self.action_size, self.num_atoms)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(x).view(-1, self.action_size, self.num_atoms)

    def distributional_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Return normalized C51 probabilities shaped (batch, actions, atoms)."""
        return F.softmax(self.distributional_logits(state), dim=2)

    def route_aux_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Predict one of 9 coarse objective directions from shared features."""
        if not self.route_aux_enabled:
            raise RuntimeError("route auxiliary head is disabled")
        return self.route_aux_head(self._features(state))

    def contact_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Predict a close-zone action without updating the shared route trunk."""
        if not self.contact_action_head_enabled:
            raise RuntimeError("contact action head is disabled")
        return self.contact_action_head(self._features(state).detach())

    def reset_noise(self) -> None:
        """Resample exploration noise on the output layers (no-op when NoisyNets
        are disabled — exploration then falls back to epsilon-greedy)."""
        if not self.use_noisy:
            return
        for layer in (
            getattr(self, "value", None),
            getattr(self, "adv", None),
            getattr(self, "head", None),
        ):
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def get_activations(self) -> Dict[str, np.ndarray]:
        return {name: act.cpu().numpy() for name, act in self.activations.items()}

    def get_layer_info(self) -> List[Dict]:
        """Layer structure for the dashboard NN visualizer. The conv stack is
        summarised as a single 'Conv' column (one node per filter) so the spatial
        network renders on the same node-graph the MLP variants use."""
        hidden = self.fc.out_features
        return [
            {"name": "Input", "neurons": self.state_size, "type": "input"},
            {"name": "Conv", "neurons": self.conv2.out_channels, "type": "hidden"},
            {"name": "Dense", "neurons": hidden, "type": "hidden"},
            {
                "name": "Output (C51 Q)" if self.distributional else "Output (Q)",
                "neurons": self.action_size,
                "type": "output",
            },
        ]

    def get_weights(self) -> List[np.ndarray]:
        """Sampled 2D weight matrices for the visualizer's connection lines — one
        per layer transition (input->conv, conv->dense, dense->output)."""

        def matrix(layer: nn.Module) -> np.ndarray:
            if isinstance(layer, NoisyLinear):
                return layer.visualization_weight()
            if isinstance(layer, nn.Linear):
                return layer.weight.detach().cpu().numpy()
            raise TypeError(f"Unsupported layer type for visualization: {type(layer)!r}")

        conv_w = self.conv2.weight.detach().cpu().numpy()
        conv_w = conv_w.reshape(conv_w.shape[0], -1)
        out_layer = self.adv if self.dueling else self.head
        if self.distributional:
            output_weight = _distributional_action_weight(
                out_layer,
                action_size=self.action_size,
                num_atoms=self.num_atoms,
            )
        else:
            output_weight = matrix(out_layer)
        return [conv_w, matrix(self.fc), output_weight]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing
if __name__ == "__main__":
    config = Config()

    # Create network
    net = DQN(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)

    print("=" * 60)
    print("DQN Network Architecture")
    print("=" * 60)

    # Print layer info
    for i, info in enumerate(net.get_layer_info()):
        print(f"Layer {i}: {info['name']} - {info['neurons']} neurons ({info['type']})")

    print(f"\nTotal parameters: {net.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    state = torch.randn(batch_size, config.STATE_SIZE)
    q_values = net(state)

    print("\nTest forward pass:")
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {q_values.shape}")

    # Check activations
    print("\nActivations captured:")
    for name, act in net.get_activations().items():
        print(f"  {name}: {act.shape}")

    print("=" * 60)
