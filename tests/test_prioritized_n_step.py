"""Tests for prioritized n-step replay."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.prioritized_n_step import PrioritizedNStepReplayBuffer


@pytest.fixture
def state_size():
    return 10


@pytest.fixture
def sample_experience(state_size):
    def _make_experience(reward=1.0, done=False):
        state = np.random.randn(state_size).astype(np.float32)
        action = np.random.randint(0, 3)
        next_state = np.random.randn(state_size).astype(np.float32)
        return state, action, reward, next_state, done

    return _make_experience


class TestPrioritizedNStepReplayBuffer:
    """Test N-step replay with prioritized sampling."""

    def test_prioritized_n_step_accumulates_returns_and_priorities(self, state_size):
        """Prioritized N-step should store computed N-step transitions, not raw rewards."""
        buffer = PrioritizedNStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.5,
        )

        for i, reward in enumerate([1.0, 2.0, 4.0]):
            state = np.ones(state_size, dtype=np.float32) * i
            next_state = np.ones(state_size, dtype=np.float32) * (i + 1)
            buffer.push(state, i, reward, next_state, False)

        assert buffer._size == 3
        assert buffer.rewards[0] == pytest.approx(3.0)
        assert np.all(buffer.priorities[:3] == 1.0)

        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(5)
        assert states.shape == (5, state_size)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, state_size)
        assert dones.shape == (5,)
        assert indices.shape == (5,)
        assert weights.shape == (5,)
        assert np.all(weights <= 1.0 + 1e-5)

    def test_prioritized_n_step_push_batch_tracks_envs_independently(self, state_size):
        """Vectorized prioritized N-step must not interleave parallel env trajectories."""
        buffer = PrioritizedNStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.9,
        )

        for step in range(3):
            states = np.stack(
                [
                    np.ones(state_size, dtype=np.float32) * step,
                    np.ones(state_size, dtype=np.float32) * (100 + step),
                ]
            )
            buffer.push_batch(
                states,
                np.array([0, 1]),
                np.array([1.0, 10.0]),
                states + 1,
                np.array([False, False]),
            )

        assert buffer._size == 6
        expected = np.sort([1.0, 1.9, 2.71, 10.0, 19.0, 27.1])
        assert np.allclose(np.sort(buffer.rewards[:6]), expected)
        assert np.all(buffer.priorities[:6] == 1.0)

    def test_prioritized_n_step_updates_priorities(self, state_size):
        """Priority updates should work for computed N-step transitions."""
        buffer = PrioritizedNStepReplayBuffer(capacity=100, state_size=state_size, n_steps=2)
        state = np.zeros(state_size, dtype=np.float32)
        for _ in range(2):
            buffer.push(state, 0, 1.0, state, False)

        buffer.update_priorities(np.array([0, 1]), np.array([0.25, 3.0]))

        assert buffer.priorities[0] == pytest.approx(0.250001)
        assert buffer.priorities[1] == pytest.approx(3.000001)
        assert buffer.max_priority == pytest.approx(3.000001)

    def test_prioritized_n_step_save_and_load_restores_per_state(
        self, state_size, sample_experience
    ):
        """Save/load should preserve priorities and beta annealing state."""
        buffer = PrioritizedNStepReplayBuffer(capacity=100, state_size=state_size, n_steps=3)
        for _ in range(10):
            state, action, reward, next_state, done = sample_experience()
            buffer.push(state, action, reward, next_state, done)

        buffer.update_priorities(np.arange(buffer._size), np.linspace(0.1, 1.0, buffer._size))
        buffer.sample(4)
        data = buffer.save_to_dict()

        restored = PrioritizedNStepReplayBuffer(capacity=100, state_size=state_size, n_steps=3)
        assert restored.load_from_dict(data) is True
        assert restored._size == buffer._size
        assert restored.beta == buffer.beta
        assert restored._frame_count == buffer._frame_count
        assert np.allclose(restored.priorities[: buffer._size], buffer.priorities[: buffer._size])
