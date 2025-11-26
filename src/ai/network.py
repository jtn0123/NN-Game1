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
from typing import List, Optional, Callable, Dict
import numpy as np

import sys
sys.path.append('../..')
from config import Config


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
    
    def _get_activation_fn(self) -> Callable:
        """Get the activation function based on config."""
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'elu': F.elu,
        }
        return activation_map.get(self.config.ACTIVATION, F.relu)
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
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
        activation_fn = self._get_activation_fn()
        
        # Apply hidden layers with activation
        for i, layer in enumerate(self.layers[:-1]):
            x = activation_fn(layer(x))
        
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

