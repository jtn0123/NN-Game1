"""Lightweight performance regression coverage for hot local training paths."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from src.ai.replay_buffer import ReplayBuffer


def test_replay_buffer_sample_no_copy_stays_within_loose_budget() -> None:
    """Catch catastrophic replay sampling regressions without a brittle threshold."""
    rng = np.random.default_rng(123)
    capacity = 8192
    state_size = 32
    batch_size = 128
    sample_count = 100
    buffer = ReplayBuffer(capacity=capacity, state_size=state_size)

    states = rng.normal(size=(capacity, state_size)).astype(np.float32)
    next_states = rng.normal(size=(capacity, state_size)).astype(np.float32)
    actions = rng.integers(0, 4, size=capacity, dtype=np.int64)
    rewards = rng.normal(size=capacity).astype(np.float32)
    dones = rng.choice([False, True], size=capacity)
    buffer.push_batch(states, actions, rewards, next_states, dones)

    started_at = perf_counter()
    for _ in range(sample_count):
        sampled = buffer.sample_no_copy(batch_size)
        assert sampled[0].shape == (batch_size, state_size)
    elapsed = perf_counter() - started_at

    assert elapsed < 2.0
