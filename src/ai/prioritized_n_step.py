"""Prioritized n-step replay buffer.

The base replay buffer module owns storage and basic PER. This module combines
PER with n-step trajectory accumulation so the larger implementation does not
inflate ``replay_buffer.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    """N-step replay buffer with prioritized sampling."""

    def __init__(
        self,
        capacity: int,
        state_size: int = 0,
        n_steps: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
    ):
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if not np.isfinite(gamma) or gamma <= 0 or gamma > 1:
            raise ValueError(f"gamma must be finite and in (0, 1], got {gamma}")
        super().__init__(
            capacity=capacity,
            state_size=state_size,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
        )
        self.n_steps = n_steps
        self.gamma = gamma
        self._n_step_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._env_buffers: List[List[Tuple[np.ndarray, int, float, np.ndarray, bool]]] = []

    def push(self, state, action, reward, next_state, done):
        """Accumulate one trajectory and store computed N-step transitions."""
        self._n_step_buffer.append((state.copy(), action, reward, next_state.copy(), done))
        if done or len(self._n_step_buffer) >= self.n_steps:
            self._flush_buffer(self._n_step_buffer)

    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        truncateds: np.ndarray | None = None,
    ) -> None:
        """Accumulate N-step returns per parallel environment.

        ``truncateds`` (optional) marks envs that ended on a time/no-progress cutoff
        rather than a real terminal. A truncated env still flushes its trajectory, but
        the stored done flag is kept False so the N-step return bootstraps the value of
        the final state (truncation-aware bootstrapping, Pardo et al. 2018).
        """
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones)
        batch_size = len(states)
        if batch_size <= 0:
            raise ValueError("push_batch requires at least one experience")
        if states.ndim != 2 or next_states.shape != states.shape:
            raise ValueError("states and next_states must be matching 2D arrays")
        truncateds = ReplayBuffer._validate_truncateds(truncateds, batch_size)

        if len(self._env_buffers) != batch_size:
            for buf in self._env_buffers:
                self._flush_buffer(buf)
            self._env_buffers = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            ended = bool(dones[i])
            truncated = bool(truncateds[i]) if truncateds is not None else False
            buf = self._env_buffers[i]
            buf.append(
                (
                    np.asarray(states[i]).copy(),
                    int(actions[i]),
                    float(rewards[i]),
                    np.asarray(next_states[i]).copy(),
                    ended and not truncated,
                )
            )
            if ended or len(buf) >= self.n_steps:
                self._flush_buffer(buf)

    def _flush_buffer(self, buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        """Compute N-step returns for one trajectory and store them with priority."""
        if not buffer:
            return

        n = len(buffer)
        for i in range(n):
            state, action, _, _, _ = buffer[i]
            n_step_reward = 0.0
            actual_final_idx = i
            for j in range(i, min(i + self.n_steps, n)):
                _, _, r, _, d = buffer[j]
                n_step_reward += (self.gamma ** (j - i)) * r
                actual_final_idx = j
                if d:
                    break

            _, _, _, n_step_next_state, n_step_done = buffer[actual_final_idx]
            n_step_length = actual_final_idx - i + 1
            PrioritizedReplayBuffer.push(
                self,
                state,
                action,
                n_step_reward,
                n_step_next_state,
                n_step_done,
                n_step_length,
            )

        buffer.clear()

    def sample(self, batch_size: int):
        """Sample prioritized N-step transitions with actual bootstrap spans."""
        result = super().sample(batch_size)
        indices = result[5]
        return (*result, self.n_step_lengths[indices].copy())

    def sample_no_copy(self, batch_size: int):
        """Sample prioritized N-step transitions without copying transition arrays."""
        result = super().sample_no_copy(batch_size)
        indices = result[5]
        return (*result, self.n_step_lengths[indices])

    def __len__(self) -> int:
        """Return stored plus pending experiences, capped at capacity."""
        pending = len(self._n_step_buffer) + sum(len(b) for b in self._env_buffers)
        return min(super().__len__() + pending, self.capacity)

    def clear(self) -> None:
        super().clear()
        self._n_step_buffer.clear()
        for buf in self._env_buffers:
            buf.clear()
        self._env_buffers = []
