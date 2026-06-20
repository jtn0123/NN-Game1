"""Vectorized Crystal Caves environment for parallel training."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from config import Config

from .crystal_caves import CrystalCaves


class VecCrystalCaves:
    """
    Vectorized Crystal Caves environment for parallel training.
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        self.num_envs = num_envs
        self.config = config
        self.headless = headless
        self.envs = [CrystalCaves(config, headless=headless) for _ in range(num_envs)]
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        self._states: np.ndarray = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards: np.ndarray = np.empty(num_envs, dtype=np.float32)
        self._dones: np.ndarray = np.empty(num_envs, dtype=np.bool_)
        self._pending_resets: np.ndarray = np.zeros(num_envs, dtype=np.bool_)
        self._last_infos: List[dict] = []

    def reset(self) -> np.ndarray:
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        self._step_into_buffers(actions)
        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        for i, done in enumerate(self._dones):
            if done:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        return states_to_return, rewards_to_return, dones_to_return, self._last_infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        for i in range(self.num_envs):
            if self._pending_resets[i]:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        self._step_into_buffers(actions)
        return self._states, self._rewards, self._dones, self._last_infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def _step_into_buffers(self, actions: np.ndarray) -> None:
        infos = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))
            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)
            if done:
                env.reset()
                self._pending_resets[i] = True
        self._last_infos = infos
