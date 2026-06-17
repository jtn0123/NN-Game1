"""Vectorized Asteroids environment."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import Config
from src.game.asteroids import Asteroids
from src.game.base_game import step_vector_env_no_copy, validate_action_batch


class VecAsteroids:
    """
    Vectorized Asteroids environment for parallel game execution.

    Runs N independent game instances simultaneously for faster training.
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """Initialize vectorized environment."""
        self.num_envs = num_envs
        self.config = config
        self.headless = headless

        # Create N independent game instances
        self.envs = [Asteroids(config, headless=headless) for _ in range(num_envs)]

        # Pre-allocate arrays
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size

        self._states = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(num_envs, dtype=np.float32)
        self._dones = np.empty(num_envs, dtype=np.bool_)

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments with batched actions."""
        actions = validate_action_batch(actions, self.num_envs, self.action_size, "VecAsteroids")
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)

            if done:
                env.reset()

        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        # Update state array for done episodes
        for i, done_flag in enumerate(self._dones):
            if bool(done_flag):
                self._states[i] = self.envs[i].get_state()

        return states_to_return, rewards_to_return, dones_to_return, infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments and return reusable internal buffers."""
        return step_vector_env_no_copy(self.envs, self._states, self._rewards, self._dones, actions)

    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]) -> None:
        """Set random seeds for each environment."""
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)
