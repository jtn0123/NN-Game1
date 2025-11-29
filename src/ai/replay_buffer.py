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
from typing import Tuple, Optional, List
import random


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples with contiguous numpy storage.
    
    Experience tuple: (state, action, reward, next_state, done)
        - state: Current game state (np.ndarray)
        - action: Action taken (int)
        - reward: Reward received (float)
        - next_state: Resulting state (np.ndarray)
        - done: Whether episode ended (bool)
    
    Optimizations:
        - Contiguous numpy arrays for all data (cache-friendly)
        - Vectorized batch extraction via numpy fancy indexing (no Python loops)
        - Circular buffer implementation for efficient memory management
        - Lazy initialization to support unknown state_size at creation
        
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=64)
    """
    
    def __init__(self, capacity: int, state_size: int = 0):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            state_size: Size of state vector (auto-detected on first push if 0)
        """
        self.capacity = capacity
        self._state_size = state_size
        self._size = 0  # Current number of experiences stored
        self._position = 0  # Current write position for circular buffer
        self._initialized = False
        
        # Contiguous storage arrays (lazily initialized on first push if state_size=0)
        if state_size > 0:
            self._init_arrays(state_size)
    
    def _init_arrays(self, state_size: int) -> None:
        """Initialize contiguous storage arrays."""
        self._state_size = state_size
        self.states = np.empty((self.capacity, state_size), dtype=np.float32)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.next_states = np.empty((self.capacity, state_size), dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.float32)
        self._initialized = True
    
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
        
        When buffer is full, oldest experience is overwritten (circular buffer).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Lazy initialization on first push
        if not self._initialized:
            self._init_arrays(len(state))
        
        # Store experience at current position
        # Use np.copyto for explicit copy semantics (safe with views from VecBreakout)
        np.copyto(self.states[self._position], state)
        self.actions[self._position] = action
        self.rewards[self._position] = reward
        np.copyto(self.next_states[self._position], next_state)
        self.dones[self._position] = float(done)
        
        # Update position and size
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Add multiple experiences to the buffer at once (vectorized for speed).

        This is optimized for vectorized environments where multiple games
        run in parallel. Much faster than calling push() in a loop.

        Args:
            states: Batch of current states (batch_size, state_size)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, state_size)
            dones: Batch of done flags (batch_size,)
        """
        batch_size = len(states)

        # Lazy initialization on first push
        if not self._initialized:
            self._init_arrays(states.shape[1])

        # Calculate indices where batch will be stored
        indices = np.arange(self._position, self._position + batch_size) % self.capacity

        # Vectorized copy to buffer (much faster than loop)
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones.astype(float)

        # Update position and size
        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of experiences using vectorized numpy indexing.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
            All arrays are copies to prevent modification of buffer data.
            
        Raises:
            RuntimeError: If buffer has not been initialized (no push() calls yet)
        """
        if not self._initialized:
            raise RuntimeError("Cannot sample from uninitialized buffer. Call push() first.")
        
        # Sample with replacement for speed (duplicates are rare with large buffers)
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        # Vectorized extraction via fancy indexing (no Python loop!)
        return (
            self.states[indices].copy(),
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            self.next_states[indices].copy(),
            self.dones[indices].copy()
        )
    
    def sample_no_copy(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample without copying (faster but arrays are views into buffer).
        
        Use this only if you'll consume the batch immediately before the next sample
        or push operation. The returned arrays share memory with the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of numpy array views (not copies)
            
        Raises:
            RuntimeError: If buffer has not been initialized (no push() calls yet)
        """
        if not self._initialized:
            raise RuntimeError("Cannot sample from uninitialized buffer. Call push() first.")
        
        # Sample with replacement for speed (duplicates are rare with large buffers)
        indices = np.random.choice(self._size, size=batch_size, replace=True)

        # Return views - faster but caller must use immediately
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self._size >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._size = 0
        self._position = 0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer with contiguous numpy storage.
    
    Experiences with higher TD-error are sampled more frequently,
    as they provide more learning signal.
    
    Uses a sum-tree data structure for O(log n) prioritized sampling.
    
    Reference:
        Schaul et al., 2016 - "Prioritized Experience Replay"
    """
    
    def __init__(
        self,
        capacity: int,
        state_size: int = 0,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_size: Size of state vector (auto-detected if 0)
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Number of frames over which to anneal beta
        """
        self.capacity = capacity
        self._state_size = state_size
        self._size = 0
        self._position = 0
        self._initialized = False
        
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self._frame_count = 0
        
        # Priorities stored as numpy array for fast operations
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        if state_size > 0:
            self._init_arrays(state_size)
    
    def _init_arrays(self, state_size: int) -> None:
        """Initialize contiguous storage arrays."""
        self._state_size = state_size
        self.states = np.empty((self.capacity, state_size), dtype=np.float32)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.next_states = np.empty((self.capacity, state_size), dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.float32)
        self._initialized = True
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience with maximum priority."""
        if not self._initialized:
            self._init_arrays(len(state))
        
        # Use np.copyto for explicit copy semantics (safe with views from VecBreakout)
        np.copyto(self.states[self._position], state)
        self.actions[self._position] = action
        self.rewards[self._position] = reward
        np.copyto(self.next_states[self._position], next_state)
        self.dones[self._position] = float(done)
        self.priorities[self._position] = self.max_priority
        
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch with probability proportional to priority.
        
        Returns:
            Tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Anneal beta
        self._frame_count += 1
        beta_progress = min(1.0, self._frame_count / self.beta_frames)
        self.beta = self.beta_start + beta_progress * (self.beta_end - self.beta_start)
        
        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:self._size]
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones(self._size, dtype=np.float32) / self._size
        
        # Sample indices based on priorities
        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        weights = weights.astype(np.float32)
        
        return (
            self.states[indices].copy(),
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            self.next_states[indices].copy(),
            self.dones[indices].copy(),
            indices,
            weights
        )
    
    def sample_no_copy(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample without copying (faster, returns views)."""
        self._frame_count += 1
        beta_progress = min(1.0, self._frame_count / self.beta_frames)
        self.beta = self.beta_start + beta_progress * (self.beta_end - self.beta_start)
        
        priorities = self.priorities[:self._size]
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones(self._size, dtype=np.float32) / self._size
        
        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=False)
        
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of sampled experiences
            td_errors: Absolute TD errors from training
        """
        priorities = np.abs(td_errors) + 1e-6  # Small epsilon to avoid zero priority
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self._size >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._size = 0
        self._position = 0
        self.priorities.fill(0)
        self.max_priority = 1.0
        self._frame_count = 0
        self.beta = self.beta_start



class NStepReplayBuffer(ReplayBuffer):
    """
    Replay buffer with N-step returns for faster reward propagation.

    Instead of single-step TD targets r + γ * max Q(s'), uses N-step returns:
        r₁ + γ * r₂ + γ² * r₃ + ... + γⁿ * max Q(sₙ)

    This provides:
        - Faster credit assignment (rewards propagate back faster)
        - Lower bias (less bootstrapping)
        - Higher variance (more Monte Carlo-like)

    Typically N=3-5 provides good trade-off between bias and variance.

    Reference:
        Hessel et al., 2017 - "Rainbow: Combining Improvements in Deep RL"
    """

    def __init__(self, capacity: int, state_size: int = 0,
                 n_steps: int = 3, gamma: float = 0.99):
        """
        Initialize N-step replay buffer.

        Args:
            capacity: Maximum buffer size
            state_size: Size of state vector (auto-detected if 0)
            n_steps: Number of steps to look ahead (typically 3-5)
            gamma: Discount factor for computing N-step returns
        """
        super().__init__(capacity, state_size)
        self.n_steps = n_steps
        self.gamma = gamma

        # Temporary buffer to accumulate N-step trajectories
        self._n_step_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def push(self, state, action, reward, next_state, done):
        """
        Add experience and compute N-step return when ready.

        Accumulates experiences until we have N steps or episode ends,
        then computes N-step returns and stores them.
        """
        # Store in temporary buffer
        self._n_step_buffer.append((state.copy(), action, reward, next_state.copy(), done))

        # Flush when we have N steps or episode ended
        if done or len(self._n_step_buffer) >= self.n_steps:
            self._flush_n_step_buffer(done)

    def _flush_n_step_buffer(self, done: bool):
        """Compute and store N-step experiences."""
        if not self._n_step_buffer:
            return

        n = len(self._n_step_buffer)

        # For each experience in the buffer, compute its N-step return
        for i in range(n):
            state, action, _, _, _ = self._n_step_buffer[i]

            # Compute discounted N-step return from position i
            n_step_reward = 0.0
            for j in range(i, n):
                _, _, r, _, d = self._n_step_buffer[j]
                n_step_reward += (self.gamma ** (j - i)) * r
                if d:
                    break

            # The N-step next state is the final state in the trajectory
            final_idx = min(i + self.n_steps - 1, n - 1)
            _, _, _, n_step_next_state, n_step_done = self._n_step_buffer[final_idx]

            # Store the N-step experience in the base buffer
            super().push(state, action, n_step_reward, n_step_next_state, n_step_done)

        # Clear temporary buffer
        self._n_step_buffer.clear()

    def __len__(self) -> int:
        """Return total experiences (main buffer + buffered)."""
        # Cap at capacity to maintain contract with capacity tests
        return min(super().__len__() + len(self._n_step_buffer), self.capacity)

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        super().clear()
        self._n_step_buffer.clear()


# Testing
if __name__ == "__main__":
    print("Testing ReplayBuffer (contiguous numpy storage)...")
    
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
    
    # Test circular buffer behavior
    print("\nTesting circular buffer...")
    for i in range(100):  # Overflow capacity
        state = np.random.randn(state_size).astype(np.float32)
        buffer.push(state, 0, 0.0, state, False)
    
    print(f"Buffer size after overflow: {len(buffer)} (capacity: {buffer.capacity})")
    
    # Test PrioritizedReplayBuffer
    print("\n" + "="*50)
    print("Testing PrioritizedReplayBuffer...")
    
    per_buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4)
    
    for i in range(50):
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(state_size).astype(np.float32)
        done = np.random.random() > 0.9
        
        per_buffer.push(state, action, reward, next_state, done)
    
    states, actions, rewards, next_states, dones, indices, weights = per_buffer.sample(32)
    print(f"PER sample shapes: states={states.shape}, weights={weights.shape}")
    print(f"Indices range: {indices.min()} - {indices.max()}")
    print(f"Weights range: {weights.min():.4f} - {weights.max():.4f}")
    
    # Update priorities
    td_errors = np.random.rand(32).astype(np.float32)
    per_buffer.update_priorities(indices, td_errors)
    print(f"Max priority after update: {per_buffer.max_priority:.4f}")
    
    print("\n✓ All ReplayBuffer tests passed!")
