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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Dict, Any, cast
import numpy as np
import math

import sys
sys.path.append('../..')
from config import Config


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
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

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
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Factorized Gaussian noise (more efficient than independent)."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights during training."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
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
        hidden_layers: Optional[List[int]] = None
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
        layer_sizes = [self.state_size] + self.hidden_sizes + [self.action_size]
        
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
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'elu': F.elu,
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
            layer.register_forward_hook(get_activation(f'layer_{i}'))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_size)
            
        Returns:
            Q-values tensor of shape (batch_size, action_size)
        """
        x = state
        
        # Apply hidden layers with activation (use cached function for speed)
        for i, layer in enumerate(self.layers[:-1]):
            x = self._activation_fn(layer(x))
        
        # Output layer (no activation - raw Q-values)
        x = self.layers[-1](x)
        
        return x
    
    def get_layer_info(self) -> List[Dict]:
        """
        Get information about each layer for visualization.
        
        Returns:
            List of dicts with layer metadata
        """
        info = []
        
        # Input layer (virtual - no actual layer)
        info.append({
            'name': 'Input',
            'neurons': self.state_size,
            'type': 'input'
        })
        
        # Hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            info.append({
                'name': f'Hidden {i + 1}',
                'neurons': layer.out_features,
                'type': 'hidden'
            })
        
        # Output layer
        info.append({
            'name': 'Output',
            'neurons': self.action_size,
            'type': 'output'
        })
        
        return info
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get all weight matrices as numpy arrays.
        
        Returns:
            List of weight matrices (for visualization)
        """
        weights = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().cpu().numpy())
        return weights
    
    def get_activations(self) -> Dict[str, np.ndarray]:
        """
        Get the most recent activations from all layers.
        
        Returns:
            Dict mapping layer names to activation arrays
        """
        return {
            name: act.cpu().numpy() 
            for name, act in self.activations.items()
        }
    
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
        hidden_layers: Optional[List[int]] = None
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
        layer_sizes = [self.state_size] + self.hidden_sizes[:-1]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.feature_layers.append(layer)
        
        # Get the size of the last shared layer (or state_size if no hidden layers)
        if len(self.hidden_sizes) > 1:
            shared_output_size = self.hidden_sizes[-2]
        else:
            shared_output_size = self.state_size
        
        # Final hidden layer size for streams
        stream_hidden_size = self.hidden_sizes[-1]
        
        # Value stream: shared -> hidden -> 1 (state value)
        self.value_hidden = nn.Linear(shared_output_size, stream_hidden_size)

        # Advantage stream: shared -> hidden -> action_size (per-action advantage)
        self.advantage_hidden = nn.Linear(shared_output_size, stream_hidden_size)

        # Final output layers - use NoisyLinear if enabled
        use_noisy = getattr(self.config, 'USE_NOISY_NETWORKS', False)
        if use_noisy:
            noisy_std = getattr(self.config, 'NOISY_STD_INIT', 0.5)
            self.value_output = NoisyLinear(stream_hidden_size, 1, noisy_std)
            self.advantage_output = NoisyLinear(stream_hidden_size, self.action_size, noisy_std)
        else:
            self.value_output = nn.Linear(stream_hidden_size, 1)
            self.advantage_output = nn.Linear(stream_hidden_size, self.action_size)
        
        # For compatibility with DQN interface, create a layers list
        self.layers = nn.ModuleList(list(self.feature_layers) + [
            self.value_hidden, self.value_output,
            self.advantage_hidden, self.advantage_output
        ])
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def _get_activation_fn(self) -> Callable[..., Any]:
        """Get the activation function based on config."""
        activation_map: Dict[str, Callable[..., Any]] = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'elu': F.elu,
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
            layer.register_forward_hook(get_activation(f'layer_{i}'))
        
        # Register hooks on stream layers
        self.value_hidden.register_forward_hook(get_activation('value_hidden'))
        self.advantage_hidden.register_forward_hook(get_activation('advantage_hidden'))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_size)
            
        Returns:
            Q-values tensor of shape (batch_size, action_size)
        """
        x = state
        
        # Pass through shared feature layers (use cached activation fn for speed)
        for layer in self.feature_layers:
            x = self._activation_fn(layer(x))
        
        # Value stream
        value = self._activation_fn(self.value_hidden(x))
        value = self.value_output(value)  # Shape: (batch, 1)
        
        # Advantage stream
        advantage = self._activation_fn(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage)  # Shape: (batch, action_size)
        
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
        info.append({
            'name': 'Input',
            'neurons': self.state_size,
            'type': 'input'
        })
        
        # Shared feature layers
        for i, layer in enumerate(self.feature_layers):
            info.append({
                'name': f'Shared {i + 1}',
                'neurons': layer.out_features,
                'type': 'hidden'
            })
        
        # Value stream
        info.append({
            'name': 'Value',
            'neurons': self.value_hidden.out_features,
            'type': 'value_stream'
        })
        
        # Advantage stream
        info.append({
            'name': 'Advantage',
            'neurons': self.advantage_hidden.out_features,
            'type': 'advantage_stream'
        })
        
        # Output layer
        info.append({
            'name': 'Output (Q)',
            'neurons': self.action_size,
            'type': 'output'
        })
        
        return info
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get all weight matrices as numpy arrays.
        
        Returns:
            List of weight matrices (for visualization)
        """
        weights = []
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().cpu().numpy())
        weights.append(self.value_hidden.weight.detach().cpu().numpy())
        weights.append(self.advantage_hidden.weight.detach().cpu().numpy())
        return weights
    
    def get_activations(self) -> Dict[str, np.ndarray]:
        """
        Get the most recent activations from all layers.
        
        Returns:
            Dict mapping layer names to activation arrays
        """
        return {
            name: act.cpu().numpy() 
            for name, act in self.activations.items()
        }
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing
if __name__ == "__main__":
    config = Config()
    
    # Create network
    net = DQN(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config
    )
    
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
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {q_values.shape}")
    
    # Check activations
    print(f"\nActivations captured:")
    for name, act in net.get_activations().items():
        print(f"  {name}: {act.shape}")
    
    print("=" * 60)

