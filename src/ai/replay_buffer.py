"""
Experience Replay Buffer
========================

A memory buffer that stores experiences for training the DQN.

Why Experience Replay?
    1. Breaks correlation between consecutive experiences
       (Neural networks learn poorly from correlated data)
    
    2. Improves sample efficiency
       (Each experience can be used for multiple training steps)
    
    3. Stabilizes training
       (Random sampling provides more diverse gradients)

How it works:
    1. Agent plays game, stores (state, action, reward, next_state, done) tuples
    2. During training, we sample random batches from the buffer
    3. Old experiences are discarded when buffer is full (FIFO)

References:
    Mnih et al., 2015 - "Human-level control through deep reinforcement learning"
"""

import numpy as np
from collections import deque
from typing import Tuple, List
import random


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    Experience tuple: (state, action, reward, next_state, done)
        - state: Current game state (np.ndarray)
        - action: Action taken (int)
        - reward: Reward received (float)
        - next_state: Resulting state (np.ndarray)
        - done: Whether episode ended (bool)
    
    Attributes:
        capacity: Maximum number of experiences to store
        buffer: Internal storage (deque for O(1) operations)
        
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=64)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience to the buffer.
        
        When buffer is full, oldest experience is automatically removed.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Sample random experiences
        batch = random.sample(self.buffer, batch_size)
        
        # Separate into arrays
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.
    
    Experiences with higher TD-error are sampled more frequently,
    as they provide more learning signal.
    
    This is an ADVANCED feature and is OPTIONAL.
    Start with regular ReplayBuffer, then upgrade to this for better performance.
    
    Reference:
        Schaul et al., 2016 - "Prioritized Experience Replay"
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_decay: float = 0.001
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_decay: Beta increase per sampling
        """
        super().__init__(capacity)
        
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_decay = beta_decay
        
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience with maximum priority (will be trained on soon)."""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch with probability proportional to priority.
        
        Returns:
            Tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get experiences
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
        
        # Anneal beta
        self.beta = min(self.beta_end, self.beta + self.beta_decay)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of sampled experiences
            td_errors: Absolute TD errors from training
        """
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6  # Small epsilon to avoid zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


# Testing
if __name__ == "__main__":
    print("Testing ReplayBuffer...")
    
    # Create buffer
    buffer = ReplayBuffer(capacity=100)
    
    # Add some experiences
    state_size = 10
    for i in range(50):
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.random() > 0.9
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Is ready for batch of 32: {buffer.is_ready(32)}")
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    
    print(f"\nSampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Dones: {dones.shape}")
    
    print("\n✓ ReplayBuffer tests passed!")

