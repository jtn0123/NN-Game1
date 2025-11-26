"""
DQN Agent
=========

The AI agent that learns to play games using Deep Q-Learning.

Key Components:
    1. Policy Network  - Used for action selection
    2. Target Network  - Used for stable Q-value estimation
    3. Replay Buffer   - Stores experiences for training
    4. Epsilon-Greedy  - Balances exploration vs exploitation

Training Algorithm (DQN):
    1. Observe state s
    2. Choose action a (epsilon-greedy)
    3. Execute action, observe reward r and next state s'
    4. Store (s, a, r, s', done) in replay buffer
    5. Sample mini-batch from replay buffer
    6. Calculate target: y = r + γ * max_a' Q_target(s', a')
    7. Update policy network: minimize (Q(s,a) - y)²
    8. Periodically sync target network with policy network

References:
    Mnih et al., 2015 - "Human-level control through deep reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List
import random
import os

from .network import DQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import sys
sys.path.append('../..')
from config import Config


class Agent:
    """
    DQN Agent for reinforcement learning.
    
    The agent maintains two networks:
        - policy_net: Updated every training step
        - target_net: Updated periodically for stability
    
    Action Selection:
        - With probability epsilon: random action (exploration)
        - With probability (1-epsilon): best Q-value action (exploitation)
    
    Attributes:
        policy_net: Network used for action selection
        target_net: Network used for computing targets
        memory: Experience replay buffer
        epsilon: Current exploration rate
        
    Example:
        >>> agent = Agent(state_size=55, action_size=3)
        >>> action = agent.select_action(state)
        >>> agent.remember(state, action, reward, next_state, done)
        >>> loss = agent.learn()
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[Config] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            config: Configuration object
        """
        self.config = config or Config()
        self.state_size = state_size
        self.action_size = action_size
        self.device = self.config.DEVICE
        
        # Networks
        self.policy_net = DQN(state_size, action_size, config).to(self.device)
        self.target_net = DQN(state_size, action_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Replay buffer (standard or prioritized based on config)
        self.use_per = self.config.USE_PRIORITIZED_REPLAY
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=self.config.MEMORY_SIZE,
                alpha=self.config.PER_ALPHA,
                beta_start=self.config.PER_BETA_START,
                beta_decay=self.config.PER_BETA_DECAY
            )
        else:
            self.memory = ReplayBuffer(capacity=self.config.MEMORY_SIZE)
        
        # Exploration
        self.epsilon = self.config.EPSILON_START
        self.episode_count = 0  # Track episodes for warmup and decay strategies
        
        # Training step counter (for target network updates)
        self.steps = 0
        
        # Training metrics
        self.losses: List[float] = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: best Q-value action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions (useful for visualization).
        
        Args:
            state: Current game state
            
        Returns:
            Array of Q-values for each action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        """
        Perform one training step.
        
        Supports both standard and prioritized experience replay.
        When using PER, applies importance sampling weights to the loss.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Don't learn until we have enough experiences
        if len(self.memory) < self.config.MEMORY_MIN:
            return None
        
        if not self.memory.is_ready(self.config.BATCH_SIZE):
            return None
        
        # Sample batch (different return format for PER vs standard)
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
                self.config.BATCH_SIZE
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.config.BATCH_SIZE
            )
            weights = None
            indices = None
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Double DQN
        # Key insight: Use policy network to SELECT actions, target network to EVALUATE them
        # This reduces overestimation of Q-values that occurs in standard DQN
        with torch.no_grad():
            # Policy network selects the best actions for next states
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target network evaluates those actions
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.config.GAMMA * next_q
        
        # Compute TD errors (for priority updates)
        td_errors = (current_q - target_q).detach()
        
        # Compute loss with importance sampling weights (if using PER)
        if self.use_per and weights is not None:
            # Weighted MSE loss - important samples have higher weight
            element_wise_loss = (current_q - target_q) ** 2
            loss = (weights * element_wise_loss).mean()
        else:
            loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config.GRAD_CLIP
            )
        
        self.optimizer.step()
        
        # Update priorities in PER buffer
        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        # Update step counter and target network
        self.steps += 1
        if self.steps % self.config.TARGET_UPDATE == 0:
            self.update_target_network()
        
        # Store loss for metrics
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self) -> None:
        """
        Decay exploration rate based on configured strategy.
        
        Strategies:
            - exponential: epsilon *= decay_rate (default, classic DQN)
            - linear: epsilon decreases linearly over episodes
            - cosine: smooth cosine annealing (often better for longer training)
        """
        self.episode_count += 1
        
        # Skip decay during warmup period
        if self.episode_count < self.config.EPSILON_WARMUP:
            return
        
        effective_episode = self.episode_count - self.config.EPSILON_WARMUP
        strategy = self.config.EXPLORATION_STRATEGY
        
        if strategy == 'exponential':
            # Classic exponential decay
            self.epsilon = max(
                self.config.EPSILON_END,
                self.epsilon * self.config.EPSILON_DECAY
            )
        
        elif strategy == 'linear':
            # Linear decay over episodes
            # Calculate how many episodes to decay over based on decay rate
            decay_episodes = int(-np.log(self.config.EPSILON_END / self.config.EPSILON_START) 
                                / np.log(self.config.EPSILON_DECAY))
            progress = min(1.0, effective_episode / decay_episodes)
            self.epsilon = self.config.EPSILON_START + \
                          (self.config.EPSILON_END - self.config.EPSILON_START) * progress
        
        elif strategy == 'cosine':
            # Cosine annealing - smooth transition
            decay_episodes = int(-np.log(self.config.EPSILON_END / self.config.EPSILON_START) 
                                / np.log(self.config.EPSILON_DECAY))
            progress = min(1.0, effective_episode / decay_episodes)
            # Cosine decay from start to end
            self.epsilon = self.config.EPSILON_END + \
                          0.5 * (self.config.EPSILON_START - self.config.EPSILON_END) * \
                          (1 + np.cos(np.pi * progress))
        
        else:
            # Fallback to exponential
            self.epsilon = max(
                self.config.EPSILON_END,
                self.epsilon * self.config.EPSILON_DECAY
            )
    
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        print(f"📂 Agent loaded from {filepath}")
        print(f"   Steps: {self.steps}, Epsilon: {self.epsilon:.4f}")
    
    def get_network_activations(self) -> dict:
        """Get current network activations for visualization."""
        return self.policy_net.get_activations()
    
    def get_average_loss(self, n: int = 100) -> float:
        """Get average of last n losses."""
        if not self.losses:
            return 0.0
        return np.mean(self.losses[-n:])


# Testing
if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    config = Config()
    agent = Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config
    )
    
    print(f"\n📊 Agent Configuration:")
    print(f"   State size: {agent.state_size}")
    print(f"   Action size: {agent.action_size}")
    print(f"   Device: {agent.device}")
    print(f"   Epsilon: {agent.epsilon}")
    
    # Test action selection
    state = np.random.randn(config.STATE_SIZE).astype(np.float32)
    action = agent.select_action(state)
    print(f"\n🎮 Test action selection:")
    print(f"   State shape: {state.shape}")
    print(f"   Selected action: {action}")
    
    # Test Q-value computation
    q_values = agent.get_q_values(state)
    print(f"   Q-values: {q_values}")
    
    # Test experience storage
    for _ in range(100):
        s = np.random.randn(config.STATE_SIZE).astype(np.float32)
        a = np.random.randint(0, config.ACTION_SIZE)
        r = np.random.randn()
        s2 = np.random.randn(config.STATE_SIZE).astype(np.float32)
        d = False
        agent.remember(s, a, r, s2, d)
    
    print(f"\n📦 Memory buffer size: {len(agent.memory)}")
    
    print("\n✓ Agent tests passed!")

