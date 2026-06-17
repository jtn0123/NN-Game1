"""Vectorized Space Invaders environment."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import Config
from src.game.base_game import validate_action_batch
from src.game.space_invaders import SpaceInvaders


class VecSpaceInvaders:
    """
    Vectorized Space Invaders environment for parallel game execution.

    Runs N independent game instances simultaneously, allowing batched
    action selection and experience collection. This amortizes Python/PyTorch
    overhead across multiple environments.

    Example:
        >>> vec_env = VecSpaceInvaders(num_envs=8, config=config)
        >>> states = vec_env.reset()  # Shape: (8, state_size)
        >>> actions = agent.select_actions_batch(states)  # Shape: (8,)
        >>> next_states, rewards, dones, infos = vec_env.step(actions)
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            config: Game configuration
            headless: Whether to run in headless mode (no rendering)
        """
        self.num_envs = num_envs
        self.config = config
        self.headless = headless

        # Create N independent game instances
        self.envs = [SpaceInvaders(config, headless=headless) for _ in range(num_envs)]

        # Pre-allocate arrays for batched returns (avoid allocation each step)
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size

        self._states = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(num_envs, dtype=np.float32)
        self._dones = np.empty(num_envs, dtype=np.bool_)

    def reset(self) -> np.ndarray:
        """
        Reset all environments.

        Returns:
            Batched initial states of shape (num_envs, state_size)
        """
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def reset_single(self, env_idx: int) -> np.ndarray:
        """
        Reset a single environment.

        Args:
            env_idx: Index of environment to reset

        Returns:
            Initial state for that environment
        """
        return self.envs[env_idx].reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with batched actions.

        Args:
            actions: Array of actions, shape (num_envs,)

        Returns:
            Tuple of (next_states, rewards, dones, infos)
            - next_states: shape (num_envs, state_size)
            - rewards: shape (num_envs,)
            - dones: shape (num_envs,)
            - infos: list of info dicts
        """
        actions = validate_action_batch(
            actions, self.num_envs, self.action_size, "VecSpaceInvaders"
        )
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)

            # Reset internally but DON'T overwrite state array yet
            if done:
                env.reset()

        # Return terminal states for done episodes
        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        # NOW update state array for next iteration
        for i, done_flag in enumerate(self._dones):
            if bool(done_flag):
                self._states[i] = self.envs[i].get_state()

        return states_to_return, rewards_to_return, dones_to_return, infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step without copying arrays (faster but caller must use immediately).

        Args:
            actions: Array of actions, shape (num_envs,)

        Returns:
            Tuple of (next_states, rewards, dones, infos) - arrays are views
        """
        actions = validate_action_batch(
            actions, self.num_envs, self.action_size, "VecSpaceInvaders"
        )
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)

            # Auto-reset environments that are done
            if done:
                self._states[i] = env.reset()

        return self._states, self._rewards, self._dones, infos

    def get_states(self) -> np.ndarray:
        """Get current states of all environments."""
        for i, env in enumerate(self.envs):
            self._states[i] = env.get_state()
        return self._states.copy()

    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def seed(self, seeds: List[int]) -> None:
        """Set random seeds for each environment."""
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)
