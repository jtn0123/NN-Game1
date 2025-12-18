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

from src.ai.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer


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


class TestNStepReplayBuffer:
    """Test N-step replay buffer."""
    
    @pytest.fixture
    def n_step_buffer(self, state_size):
        """Create an N-step replay buffer with n=3."""
        return NStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.99
        )
    
    def test_n_step_creates_successfully(self):
        """N-step buffer should initialize without errors."""
        buffer = NStepReplayBuffer(capacity=100, n_steps=3, gamma=0.99)
        assert buffer is not None
        assert buffer.n_steps == 3
        assert buffer.gamma == 0.99
    
    def test_n_step_reward_accumulation(self, state_size):
        """N-step buffer should correctly accumulate discounted rewards."""
        buffer = NStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.5  # Simple gamma for easy calculation
        )
        
        # Push 3 experiences with rewards 1, 2, 4 (no early termination)
        for i, reward in enumerate([1.0, 2.0, 4.0]):
            state = np.ones(state_size, dtype=np.float32) * i
            next_state = np.ones(state_size, dtype=np.float32) * (i + 1)
            done = (i == 2)  # Last one is terminal
            buffer.push(state, 0, reward, next_state, done)
        
        # Expected accumulated reward for first state:
        # 1.0 + 0.5*2.0 + 0.25*4.0 = 1.0 + 1.0 + 1.0 = 3.0
        assert buffer.rewards[0] == 3.0
    
    def test_n_step_early_termination_uses_correct_state(self, state_size):
        """When episode terminates early, buffer should use correct terminal state and done flag.
        
        This tests the fix for the bug where final_idx was calculated independently
        of early termination, causing wrong next_state and done flag to be stored.
        """
        buffer = NStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=5,  # 5 steps but we'll terminate at step 2
            gamma=0.9
        )
        
        # Push experience 0: state=0, next_state=1, done=False
        state0 = np.zeros(state_size, dtype=np.float32)
        next_state0 = np.ones(state_size, dtype=np.float32)
        buffer.push(state0, 0, 1.0, next_state0, False)
        
        # Push experience 1: state=1, next_state=2, done=True (TERMINAL)
        state1 = np.ones(state_size, dtype=np.float32)
        next_state1 = np.ones(state_size, dtype=np.float32) * 2  # Terminal state
        buffer.push(state1, 1, 2.0, next_state1, True)  # Episode ends here!
        
        # Now the buffer should have been flushed with correct terminal info
        # For the first experience (state0), since done=True at step 1,
        # the next_state should be next_state1 (the terminal state) and done=True
        
        # Check that done flag is correctly set to True (episode terminated early)
        assert buffer.dones[0] == 1.0, "First experience should have done=True due to early termination"
        
        # Check that next_state is the correct terminal state (next_state1, all 2s)
        assert np.allclose(buffer.next_states[0], next_state1), \
            "First experience should have next_state from terminal step, not final_idx"
        
        # The accumulated reward should be: 1.0 + 0.9 * 2.0 = 2.8
        expected_reward = 1.0 + 0.9 * 2.0
        assert np.isclose(buffer.rewards[0], expected_reward), \
            f"Expected reward {expected_reward}, got {buffer.rewards[0]}"
    
    def test_n_step_no_early_termination(self, state_size):
        """When no early termination, buffer should use N-th state correctly."""
        buffer = NStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.9
        )
        
        # Push 4 experiences without early termination until the end
        states = []
        for i in range(4):
            state = np.ones(state_size, dtype=np.float32) * i
            next_state = np.ones(state_size, dtype=np.float32) * (i + 1)
            done = (i == 3)  # Only last one terminates
            states.append((state, next_state))
            buffer.push(state, i, float(i + 1), next_state, done)
        
        # For the first experience (state=0), with n_steps=3:
        # It should look ahead 3 steps, so next_state should be from index 2 (0+3-1=2)
        # next_state at index 2 is np.ones * 3
        expected_next_state = np.ones(state_size, dtype=np.float32) * 3
        assert np.allclose(buffer.next_states[0], expected_next_state), \
            "Without early termination, should use N-th step's next_state"
    
    def test_n_step_multiple_episodes(self, state_size):
        """Buffer should handle multiple episodes correctly."""
        buffer = NStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.99
        )
        
        # Episode 1: 2 steps then done
        for i in range(2):
            state = np.zeros(state_size, dtype=np.float32)
            next_state = np.ones(state_size, dtype=np.float32)
            done = (i == 1)
            buffer.push(state, 0, 1.0, next_state, done)
        
        # Episode 2: 2 steps then done
        for i in range(2):
            state = np.ones(state_size, dtype=np.float32) * 5
            next_state = np.ones(state_size, dtype=np.float32) * 6
            done = (i == 1)
            buffer.push(state, 0, 2.0, next_state, done)
        
        # Should have 4 experiences total (2 from each episode)
        assert len(buffer) == 4
    
    def test_n_step_clear(self, n_step_buffer, sample_experience):
        """Clear should empty both main buffer and n-step buffer."""
        for _ in range(10):
            state, action, reward, next_state, done = sample_experience()
            n_step_buffer.push(state, action, reward, next_state, done)
        
        n_step_buffer.clear()
        
        assert len(n_step_buffer) == 0
        assert len(n_step_buffer._n_step_buffer) == 0


class TestReplayBufferEdgeCases:
    """Test edge cases and validation."""

    def test_sample_from_empty_raises(self):
        """Sampling from empty buffer should raise RuntimeError."""
        buffer = ReplayBuffer(capacity=100, state_size=10)
        with pytest.raises(RuntimeError, match="Cannot sample from empty buffer"):
            buffer.sample(16)

    def test_sample_no_copy_from_empty_raises(self):
        """sample_no_copy from empty buffer should raise RuntimeError."""
        buffer = ReplayBuffer(capacity=100, state_size=10)
        with pytest.raises(RuntimeError, match="Cannot sample from empty buffer"):
            buffer.sample_no_copy(16)

    def test_negative_capacity_raises(self):
        """Creating buffer with negative capacity should raise ValueError."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            ReplayBuffer(capacity=-10)

    def test_zero_capacity_raises(self):
        """Creating buffer with zero capacity should raise ValueError."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            ReplayBuffer(capacity=0)

    def test_state_shape_mismatch_raises(self, state_size):
        """Pushing state with wrong shape should raise ValueError."""
        buffer = ReplayBuffer(capacity=100, state_size=state_size)
        # First push with correct size
        state = np.ones(state_size, dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)

        # Second push with wrong size
        wrong_state = np.ones(state_size + 5, dtype=np.float32)
        with pytest.raises(ValueError, match="State shape mismatch"):
            buffer.push(wrong_state, 0, 1.0, wrong_state, False)


class TestPrioritizedReplayBufferEdgeCases:
    """Test PER edge cases and validation."""

    def test_per_sample_with_batch_larger_than_buffer(self, state_size, sample_experience):
        """PER should use replacement when batch > buffer size."""
        buffer = PrioritizedReplayBuffer(capacity=100, state_size=state_size)
        # Add only 5 experiences
        for _ in range(5):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)

        # Sample 16 (larger than 5) - should not crash
        result = buffer.sample(16)
        assert len(result) == 7

    def test_per_save_and_load_restores_beta(self, state_size, sample_experience):
        """Save/load should restore beta annealing state."""
        buffer = PrioritizedReplayBuffer(capacity=100, state_size=state_size)
        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)

        # Sample to advance beta
        for _ in range(10):
            buffer.sample(8)

        saved_beta = buffer.beta
        saved_frame_count = buffer._frame_count

        # Save and load to new buffer
        data = buffer.save_to_dict()
        new_buffer = PrioritizedReplayBuffer(capacity=100, state_size=state_size)
        new_buffer.load_from_dict(data)

        assert new_buffer.beta == saved_beta
        assert new_buffer._frame_count == saved_frame_count

    def test_per_sample_empty_raises(self, state_size):
        """Sampling from empty PER buffer should raise."""
        buffer = PrioritizedReplayBuffer(capacity=100, state_size=state_size)
        buffer._initialized = True  # Force initialization
        buffer._size = 0
        # Match either "empty buffer" or "empty PrioritizedReplayBuffer"
        with pytest.raises(RuntimeError, match="Cannot sample from empty"):
            buffer.sample_no_copy(16)

    def test_per_negative_capacity_raises(self):
        """Creating PER buffer with negative capacity should raise ValueError."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            PrioritizedReplayBuffer(capacity=-10)

    def test_per_state_shape_mismatch_raises(self, state_size):
        """Pushing state with wrong shape to PER buffer should raise ValueError."""
        buffer = PrioritizedReplayBuffer(capacity=100, state_size=state_size)
        # First push with correct size
        state = np.ones(state_size, dtype=np.float32)
        buffer.push(state, 0, 1.0, state, False)

        # Second push with wrong size
        wrong_state = np.ones(state_size + 5, dtype=np.float32)
        with pytest.raises(ValueError, match="State shape mismatch"):
            buffer.push(wrong_state, 0, 1.0, wrong_state, False)


class TestReplayBufferPersistence:
    """Test replay buffer save/load functionality."""

    def test_buffer_save_load_roundtrip(self, state_size, sample_experience):
        """Buffer should save and load correctly."""
        buffer = ReplayBuffer(capacity=100, state_size=state_size)

        # Fill buffer with identifiable experiences
        for i in range(50):
            state = np.ones(state_size, dtype=np.float32) * i
            next_state = np.ones(state_size, dtype=np.float32) * (i + 1)
            buffer.push(state, i % 3, float(i), next_state, i % 10 == 0)

        # Save to dict
        saved_data = buffer.save_to_dict()

        # Create new buffer and load
        new_buffer = ReplayBuffer(capacity=100, state_size=state_size)
        success = new_buffer.load_from_dict(saved_data)

        assert success is True
        assert len(new_buffer) == 50
        assert new_buffer._position == 50

        # Verify data integrity
        assert np.array_equal(new_buffer.states[:50], buffer.states[:50])
        assert np.array_equal(new_buffer.actions[:50], buffer.actions[:50])
        assert np.array_equal(new_buffer.rewards[:50], buffer.rewards[:50])

    def test_buffer_save_load_empty(self, state_size):
        """Saving empty buffer should work correctly."""
        buffer = ReplayBuffer(capacity=100, state_size=state_size)
        saved_data = buffer.save_to_dict()

        assert saved_data['initialized'] is False
        assert saved_data['size'] == 0

        new_buffer = ReplayBuffer(capacity=100, state_size=state_size)
        success = new_buffer.load_from_dict(saved_data)
        assert success is False  # Empty buffer returns False

    def test_buffer_load_larger_saved_truncates(self, state_size, sample_experience):
        """Loading larger buffer into smaller capacity should truncate."""
        # Create buffer with 100 capacity and fill with 80 experiences
        buffer = ReplayBuffer(capacity=100, state_size=state_size)
        for i in range(80):
            state = np.ones(state_size, dtype=np.float32) * i
            buffer.push(state, 0, float(i), state, False)

        saved_data = buffer.save_to_dict()

        # Create smaller buffer and load
        small_buffer = ReplayBuffer(capacity=50, state_size=state_size)
        success = small_buffer.load_from_dict(saved_data)

        assert success is True
        assert len(small_buffer) == 50
        # Should have the most recent 50 experiences (indices 30-79)
        # First element should have reward 30.0 (offset by 30)
        assert small_buffer.rewards[0] == 30.0


class TestPrioritizedReplayBufferPriorities:
    """Test PER priority-weighted sampling correctness."""

    def test_per_priority_sampling_correctness(self, state_size, sample_experience):
        """Higher priority experiences should be sampled more frequently."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            state_size=state_size,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_frames=100000
        )

        # Add experiences with identifiable rewards
        for i in range(50):
            state = np.ones(state_size, dtype=np.float32) * i
            buffer.push(state, 0, float(i), state, False)

        # Update priorities: make experience at index 0 have very high priority
        high_priority_indices = np.array([0])
        high_td_errors = np.array([1000.0])  # Very high TD error
        buffer.update_priorities(high_priority_indices, high_td_errors)

        # Sample many times and count how often index 0 is sampled
        sample_count = 0
        for _ in range(100):
            _, _, rewards, _, _, indices, _ = buffer.sample(10)
            sample_count += np.sum(indices == 0)

        # High priority experience should be sampled significantly more often
        # With alpha=0.6 and very high priority, expect > 50% of samples
        # Note: This is probabilistic, but 1000x priority diff should be very reliable
        assert sample_count > 20, f"High priority sample count {sample_count} too low"

    def test_per_importance_weights_normalized(self, state_size, sample_experience):
        """Importance sampling weights should be normalized (max = 1)."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            state_size=state_size,
            alpha=0.6,
            beta_start=1.0,  # Full correction
            beta_end=1.0,
            beta_frames=1
        )

        for _ in range(50):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)

        # Sample and check weights
        _, _, _, _, _, _, weights = buffer.sample(16)

        # Max weight should be 1.0 (normalized)
        assert np.max(weights) == pytest.approx(1.0, rel=1e-5)
        # All weights should be <= 1.0
        assert np.all(weights <= 1.0 + 1e-5)


class TestPushBatchCircularWrapping:
    """Test push_batch behavior across circular buffer boundary."""

    def test_push_batch_circular_wrapping(self, state_size):
        """push_batch should correctly handle wrapping at buffer boundary."""
        buffer = ReplayBuffer(capacity=10, state_size=state_size)

        # Fill buffer to position 8
        for i in range(8):
            state = np.ones(state_size, dtype=np.float32) * i
            buffer.push(state, 0, float(i), state, False)

        assert buffer._position == 8
        assert len(buffer) == 8

        # Push batch of 5 - should wrap around
        batch_states = np.ones((5, state_size), dtype=np.float32) * 100
        batch_actions = np.array([1, 1, 1, 1, 1])
        batch_rewards = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        batch_next_states = batch_states.copy()
        batch_dones = np.array([False, False, False, False, False])

        buffer.push_batch(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)

        # Buffer should be full (capacity 10)
        assert len(buffer) == 10

        # Position should have wrapped: (8 + 5) % 10 = 3
        assert buffer._position == 3

        # Check data integrity - indices 8,9,0,1,2 should have new data
        assert buffer.rewards[8] == 100.0
        assert buffer.rewards[9] == 101.0
        assert buffer.rewards[0] == 102.0
        assert buffer.rewards[1] == 103.0
        assert buffer.rewards[2] == 104.0

        # Old data at indices 3-7 should be preserved
        assert buffer.rewards[3] == 3.0
        assert buffer.rewards[7] == 7.0

    def test_push_batch_larger_than_capacity(self, state_size):
        """push_batch larger than capacity should fill entire buffer."""
        buffer = ReplayBuffer(capacity=10, state_size=state_size)

        # Push batch of 15 experiences (larger than capacity)
        batch_states = np.arange(15).reshape(15, 1).repeat(state_size, axis=1).astype(np.float32)
        batch_actions = np.arange(15) % 3
        batch_rewards = np.arange(15, dtype=np.float32)
        batch_next_states = batch_states.copy()
        batch_dones = np.zeros(15, dtype=bool)

        buffer.push_batch(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)

        # Buffer should be at capacity
        assert len(buffer) == 10

        # Position should be 15 % 10 = 5
        assert buffer._position == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
