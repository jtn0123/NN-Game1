"""Agent checkpoint round-trip contract tests."""

import numpy as np
import pytest
import torch

from config import Config
from src.ai.agent import Agent, TrainingHistory


def _small_config() -> Config:
    config = Config()
    config.GAME_NAME = "breakout"
    config.HIDDEN_LAYERS = [16]
    config.MEMORY_SIZE = 32
    config.MEMORY_MIN = 4
    config.BATCH_SIZE = 4
    config.LEARN_EVERY = 1
    config.GRADIENT_STEPS = 1
    config.USE_DUELING = False
    config.USE_NOISY_NETWORKS = False
    config.USE_PRIORITIZED_REPLAY = False
    config.USE_N_STEP_RETURNS = False
    config.USE_LR_SCHEDULER = False
    return config


def test_agent_save_load_round_trip_preserves_metadata_history_and_q_values(tmp_path):
    """Saved checkpoints should restore model state, metadata, and dashboard history."""
    config = _small_config()
    config.MODEL_DIR = str(tmp_path / "models")
    agent = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    agent.epsilon = 0.42
    agent.steps = 7
    state = np.linspace(0, 1, config.STATE_SIZE, dtype=np.float32)
    next_state = np.roll(state, 1).astype(np.float32)
    for action in range(5):
        agent.remember(state, action % config.ACTION_SIZE, 0.5, next_state, False)

    history = TrainingHistory(
        scores=[1, 2, 3],
        rewards=[0.1, 0.2, 0.3],
        steps=[4, 5, 6],
        epsilons=[0.5, 0.45, 0.42],
        bricks=[1, 2, 3],
        wins=[False, False, True],
        losses=[0.9, 0.8],
        q_values=[0.2, 0.3],
        exploration_actions=4,
        exploitation_actions=5,
        target_updates=1,
        best_score=3,
    )
    before_q = agent.get_q_values(state)
    path = tmp_path / "models" / "breakout" / "agent_roundtrip.pth"

    metadata = agent.save(
        str(path),
        save_reason="manual",
        episode=3,
        best_score=3,
        avg_score_last_100=2.0,
        win_rate=0.25,
        training_history=history,
        save_replay_buffer=True,
        quiet=True,
    )
    restored = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    with pytest.warns(RuntimeWarning, match="unrestricted checkpoint load"):
        restored_metadata, restored_history = restored.load(str(path), quiet=True)
    after_q = restored.get_q_values(state)

    assert metadata is not None
    assert restored_metadata is not None
    assert restored_history is not None
    assert restored_metadata.episode == 3
    assert restored_metadata.best_score == 3
    assert restored_metadata.total_steps == 7
    assert restored.epsilon == 0.42
    assert restored.steps == 7
    assert restored_history.scores == [1, 2, 3]
    assert restored_history.best_score == 3
    assert len(restored.memory) == 5
    assert np.allclose(before_q, after_q)


def test_agent_load_rejects_incompatible_checkpoint_without_mutating_agent(tmp_path):
    """Wrong state/action sizes should return None and leave the live agent untouched."""
    config = _small_config()
    agent = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    agent.epsilon = 0.33
    path = tmp_path / "bad_architecture.pth"
    torch.save(
        {
            "policy_net_state_dict": {},
            "target_net_state_dict": {},
            "optimizer_state_dict": {},
            "epsilon": 0.9,
            "steps": 99,
            "state_size": config.STATE_SIZE + 1,
            "action_size": config.ACTION_SIZE,
        },
        path,
    )

    metadata, history = agent.load(str(path), quiet=True)

    assert metadata is None
    assert history is None
    assert agent.epsilon == 0.33
    assert agent.steps == 0
