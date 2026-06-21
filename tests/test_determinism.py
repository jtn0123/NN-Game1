"""Determinism / reproducibility guards.

These lock in that the greedy (epsilon=0, eval-mode) policy is deterministic and
that same-seed construction yields identical behavior — the baseline RL debugging
relies on. Skipped if torch is unavailable.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config import Config  # noqa: E402
from src.ai.agent import Agent  # noqa: E402


@pytest.mark.torch
class TestDeterminism:
    def _agent(self, config):
        return Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)

    def test_greedy_action_is_stable_across_calls(self):
        config = Config()
        config.FORCE_CPU = True
        torch.manual_seed(0)
        agent = self._agent(config)
        agent.epsilon = 0.0
        state = np.random.RandomState(0).randn(config.STATE_SIZE).astype(np.float32)

        actions = [agent.select_action(state, training=False) for _ in range(8)]
        assert len(set(actions)) == 1  # greedy is deterministic

    def test_same_seed_agents_select_same_action(self):
        config = Config()
        config.FORCE_CPU = True
        state = np.random.RandomState(1).randn(config.STATE_SIZE).astype(np.float32)

        torch.manual_seed(123)
        agent_a = self._agent(config)
        agent_a.epsilon = 0.0
        action_a = agent_a.select_action(state, training=False)

        torch.manual_seed(123)
        agent_b = self._agent(config)
        agent_b.epsilon = 0.0
        action_b = agent_b.select_action(state, training=False)

        assert action_a == action_b
