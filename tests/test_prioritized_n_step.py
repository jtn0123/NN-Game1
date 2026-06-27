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

        states, actions, rewards, next_states, dones, indices, weights, n_step_lengths = (
            buffer.sample(5)
        )
        assert states.shape == (5, state_size)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, state_size)
        assert dones.shape == (5,)
        assert indices.shape == (5,)
        assert weights.shape == (5,)
        assert n_step_lengths.shape == (5,)
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
        assert np.allclose(np.sort(buffer.n_step_lengths[:6]), np.sort([3, 2, 1, 3, 2, 1]))
        assert np.all(buffer.priorities[:6] == 1.0)

    def test_prioritized_n_step_records_actual_bootstrap_span(self, state_size):
        """Prioritized N-step stores the actual bootstrap span for each transition."""
        buffer = PrioritizedNStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.9,
        )

        for i in range(3):
            state = np.ones(state_size, dtype=np.float32) * i
            next_state = np.ones(state_size, dtype=np.float32) * (i + 1)
            buffer.push(state, i, 1.0, next_state, False)

        assert buffer._size == 3
        assert buffer.n_step_lengths[:3].tolist() == [3, 2, 1]

    def test_truncation_bootstrap_stores_non_terminal_and_flushes(self, state_size: int) -> None:
        """A truncated env stores done=False (bootstrap) but still flushes its trajectory."""
        buffer = PrioritizedNStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.5,
        )

        # Two ongoing steps, then a step that ends the episode by TIMEOUT (truncated).
        for step in range(2):
            states = np.ones((1, state_size), dtype=np.float32) * step
            buffer.push_batch(
                states,
                np.array([0]),
                np.array([1.0]),
                states + 1,
                np.array([False]),
                truncateds=np.array([False]),
            )
        # The truncated terminal step: dones=True but truncateds=True.
        last = np.ones((1, state_size), dtype=np.float32) * 2
        buffer.push_batch(
            last,
            np.array([0]),
            np.array([4.0]),
            last + 1,
            np.array([True]),
            truncateds=np.array([True]),
        )

        # All three transitions flushed (env reset boundary), so nothing bridges forward.
        assert buffer._size == 3
        assert len(buffer._env_buffers[0]) == 0
        # None are stored as terminal — every transition bootstraps its final state.
        assert np.all(buffer.dones[:3] == 0.0)
        # The first transition accumulated the full 3-step return because no step in the
        # trajectory was treated as terminal: 1.0 + 0.5*1.0 + 0.25*4.0 = 2.5.
        assert buffer.rewards[0] == pytest.approx(2.5)
        assert buffer.n_step_lengths[0] == 3

    def test_real_terminal_still_stops_bootstrap(self, state_size: int) -> None:
        """Without truncated, a real terminal keeps done=True and cuts the n-step return."""
        buffer = PrioritizedNStepReplayBuffer(
            capacity=100,
            state_size=state_size,
            n_steps=3,
            gamma=0.5,
        )

        for step in range(2):
            states = np.ones((1, state_size), dtype=np.float32) * step
            buffer.push_batch(
                states,
                np.array([0]),
                np.array([1.0]),
                states + 1,
                np.array([False]),
            )
        last = np.ones((1, state_size), dtype=np.float32) * 2
        buffer.push_batch(
            last,
            np.array([0]),
            np.array([4.0]),
            last + 1,
            np.array([True]),
        )

        assert buffer._size == 3
        # Every transition's n-step return ends at the real terminal, so all bootstrap
        # to a terminal (done=True) — the exact opposite of the truncated case above,
        # where the same trajectory stores done=False so the value still bootstraps.
        assert np.all(buffer.dones[:3] == 1.0)

    def test_push_batch_rejects_misshaped_truncateds(self, state_size: int) -> None:
        """A truncated mask whose length does not match the batch must raise, never
        silently misalign env terminal flags."""
        buffer = PrioritizedNStepReplayBuffer(capacity=100, state_size=state_size, n_steps=3)
        states = np.ones((2, state_size), dtype=np.float32)
        with pytest.raises(ValueError, match="truncateds length mismatch"):
            buffer.push_batch(
                states,
                np.array([0, 1]),
                np.array([1.0, 1.0]),
                states + 1,
                np.array([False, False]),
                truncateds=np.array([False]),  # too short
            )

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
