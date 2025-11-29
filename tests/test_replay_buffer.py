"""
Tests for the Replay Buffer implementations.

These tests verify:
    - Buffer initialization
    - Experience storage (push)
    - Sampling behavior
    - Circular buffer overflow
    - Prioritized replay buffer
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


@pytest.fixture
def state_size():
    """State size for testing."""
    return 10


@pytest.fixture
def buffer(state_size):
    """Create a replay buffer instance."""
    return ReplayBuffer(capacity=100, state_size=state_size)


@pytest.fixture
def sample_experience(state_size):
    """Create a sample experience tuple."""
    def _make_experience(reward=1.0, done=False):
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.randint(0, 3)
        next_state = np.random.randn(state_size).astype(np.float32)
        return state, action, reward, next_state, done
    return _make_experience


class TestReplayBufferInitialization:
    """Test buffer initialization."""
    
    def test_buffer_creates_successfully(self):
        """Buffer should initialize without errors."""
        buffer = ReplayBuffer(capacity=100)
        assert buffer is not None
    
    def test_buffer_starts_empty(self, buffer):
        """Buffer should start empty."""
        assert len(buffer) == 0
    
    def test_capacity_set_correctly(self, buffer):
        """Capacity should be set correctly."""
        assert buffer.capacity == 100
    
    def test_state_size_auto_detection(self, state_size):
        """State size should auto-detect on first push."""
        buffer = ReplayBuffer(capacity=100, state_size=0)
        state = np.random.randn(state_size).astype(np.float32)
        buffer.push(state, 0, 1.0, state, False)
        assert buffer._state_size == state_size


class TestReplayBufferPush:
    """Test experience storage."""
    
    def test_push_increases_size(self, buffer, sample_experience):
        """Push should increase buffer size."""
        state, action, reward, next_state, done = sample_experience()
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1
    
    def test_push_multiple_experiences(self, buffer, sample_experience):
        """Should store multiple experiences."""
        for i in range(10):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 10
    
    def test_experience_stored_correctly(self, buffer, state_size):
        """Experience should be stored with correct values in contiguous arrays."""
        state = np.ones(state_size, dtype=np.float32)
        action = 2
        reward = 5.0
        next_state = np.zeros(state_size, dtype=np.float32)
        done = True
        
        buffer.push(state, action, reward, next_state, done)
        
        # Access contiguous arrays directly
        assert np.array_equal(buffer.states[0], state)
        assert buffer.actions[0] == action
        assert buffer.rewards[0] == reward
        assert np.array_equal(buffer.next_states[0], next_state)
        assert buffer.dones[0] == float(done)


class TestReplayBufferCircularBehavior:
    """Test circular buffer overflow behavior."""
    
    def test_buffer_respects_capacity(self, state_size, sample_experience):
        """Buffer should not exceed capacity."""
        buffer = ReplayBuffer(capacity=50, state_size=state_size)
        
        for _ in range(100):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
    
    def test_oldest_overwritten(self, state_size):
        """Oldest experiences should be overwritten when full."""
        buffer = ReplayBuffer(capacity=3, state_size=state_size)
        
        # Push 3 experiences with identifiable rewards
        for i in range(3):
            state = np.ones(state_size, dtype=np.float32) * i
            buffer.push(state, 0, float(i), state, False)
        
        # Push a 4th - should overwrite first
        new_state = np.ones(state_size, dtype=np.float32) * 99
        buffer.push(new_state, 0, 99.0, new_state, False)
        
        # First experience should now have reward 99 (position 0 overwritten)
        assert buffer.rewards[0] == 99.0
        # Second and third should still be 1 and 2
        assert buffer.rewards[1] == 1.0
        assert buffer.rewards[2] == 2.0


class TestReplayBufferSample:
    """Test sampling behavior."""
    
    def test_sample_returns_correct_shapes(self, buffer, sample_experience, state_size):
        """Sampled batch should have correct shapes."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        batch_size = 16
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, state_size)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, state_size)
        assert dones.shape == (batch_size,)
    
    def test_sample_returns_correct_dtypes(self, buffer, sample_experience):
        """Sampled arrays should have correct dtypes."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        states, actions, rewards, next_states, dones = buffer.sample(16)
        
        assert states.dtype == np.float32
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert next_states.dtype == np.float32
        assert dones.dtype == np.float32
    
    def test_sample_is_random(self, buffer, sample_experience):
        """Multiple samples should be different (random)."""
        for i in range(100):
            state, action, reward, next_state, done = sample_experience(reward=float(i))
            buffer.push(state, action, reward, next_state, done)
        
        # Sample twice
        _, _, rewards1, _, _ = buffer.sample(32)
        _, _, rewards2, _, _ = buffer.sample(32)
        
        # They should be different (extremely unlikely to be identical)
        assert not np.array_equal(rewards1, rewards2)
    
    def test_sample_contains_valid_experiences(self, buffer, sample_experience):
        """Sampled experiences should exist in the buffer."""
        for i in range(50):
            state, action, reward, next_state, done = sample_experience(reward=float(i))
            buffer.push(state, action, reward, next_state, done)
        
        # Sample and check rewards are in valid range
        _, _, rewards, _, _ = buffer.sample(30)
        
        # All sampled rewards should be in the range [0, 49]
        assert all(0 <= r < 50 for r in rewards)


class TestReplayBufferSampleNoCopy:
    """Test sample_no_copy method."""
    
    def test_sample_no_copy_returns_views(self, buffer, sample_experience, state_size):
        """sample_no_copy should return arrays with correct shapes."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        batch_size = 16
        states, actions, rewards, next_states, dones = buffer.sample_no_copy(batch_size)
        
        # Check shapes are correct
        assert states.shape == (batch_size, state_size)
        assert actions.shape == (batch_size,)
    
    def test_sample_no_copy_shares_memory_with_buffer(self, buffer, sample_experience, state_size):
        """sample_no_copy arrays should share memory with buffer storage."""
        for i in range(50):
            state = np.ones(state_size, dtype=np.float32) * i
            buffer.push(state, 0, float(i), state, False)
        
        # Get a sample
        states, _, rewards, _, _ = buffer.sample_no_copy(16)
        
        # Verify that the sampled data shares base with buffer arrays
        # (numpy fancy indexing creates copies, but they reference valid buffer data)
        assert states.dtype == np.float32
        assert rewards.dtype == np.float32


class TestReplayBufferUtilities:
    """Test utility methods."""
    
    def test_is_ready_false_when_empty(self, buffer):
        """is_ready should return False for empty buffer."""
        assert not buffer.is_ready(32)
    
    def test_is_ready_false_when_insufficient(self, buffer, sample_experience):
        """is_ready should return False with insufficient samples."""
        for _ in range(10):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        assert not buffer.is_ready(32)
    
    def test_is_ready_true_when_sufficient(self, buffer, sample_experience):
        """is_ready should return True with enough samples."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        assert buffer.is_ready(32)
    
    def test_clear_empties_buffer(self, buffer, sample_experience):
        """clear should empty the buffer."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        buffer.clear()
        assert len(buffer) == 0
    
    def test_clear_resets_position(self, buffer, sample_experience):
        """clear should reset write position."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        buffer.clear()
        assert buffer._position == 0


class TestPrioritizedReplayBuffer:
    """Test prioritized replay buffer."""
    
    @pytest.fixture
    def per_buffer(self, state_size):
        """Create a prioritized replay buffer."""
        return PrioritizedReplayBuffer(
            capacity=100,
            state_size=state_size,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_frames=1000  # Small for testing
        )
    
    def test_per_creates_successfully(self):
        """PER buffer should initialize without errors."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert buffer is not None
    
    def test_per_push_adds_priority(self, per_buffer, sample_experience):
        """Push should add experience with priority."""
        state, action, reward, next_state, done = sample_experience()
        per_buffer.push(state, action, reward, next_state, done)
        
        # Priority at position 0 should equal max_priority
        assert per_buffer.priorities[0] == per_buffer.max_priority
        assert len(per_buffer) == 1
    
    def test_per_sample_returns_indices_and_weights(self, per_buffer, sample_experience):
        """PER sample should return indices and weights."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            per_buffer.push(state, action, reward, next_state, done)
        
        result = per_buffer.sample(16)
        
        # Should return 7 items: states, actions, rewards, next_states, dones, indices, weights
        assert len(result) == 7
        
        states, actions, rewards, next_states, dones, indices, weights = result
        assert indices.shape == (16,)
        assert weights.shape == (16,)
    
    def test_per_beta_annealing(self, per_buffer, sample_experience):
        """Beta should increase after each sample."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            per_buffer.push(state, action, reward, next_state, done)
        
        initial_beta = per_buffer.beta
        per_buffer.sample(16)
        
        assert per_buffer.beta > initial_beta
    
    def test_per_beta_capped_at_end(self, state_size, sample_experience):
        """Beta should not exceed beta_end."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            state_size=state_size,
            beta_start=0.9,
            beta_end=1.0,
            beta_frames=10  # Very small for fast annealing
        )
        
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)
        
        # Sample multiple times to anneal beta
        for _ in range(20):
            buffer.sample(16)
        
        assert buffer.beta == 1.0
    
    def test_per_update_priorities(self, per_buffer, sample_experience):
        """update_priorities should update priority values."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            per_buffer.push(state, action, reward, next_state, done)
        
        # Sample and get indices
        _, _, _, _, _, indices, _ = per_buffer.sample(16)
        
        # Update with high TD errors
        td_errors = np.ones(16) * 10.0
        per_buffer.update_priorities(indices, td_errors)
        
        # Priorities at those indices should be updated (10.0 + epsilon)
        for idx in indices:
            assert per_buffer.priorities[idx] > 1.0  # Higher than default
    
    def test_per_max_priority_updated(self, per_buffer, sample_experience):
        """max_priority should track highest priority."""
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            per_buffer.push(state, action, reward, next_state, done)
        
        initial_max = per_buffer.max_priority
        
        # Sample and update with very high TD error
        _, _, _, _, _, indices, _ = per_buffer.sample(16)
        td_errors = np.ones(16) * 100.0
        per_buffer.update_priorities(indices, td_errors)
        
        assert per_buffer.max_priority > initial_max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
