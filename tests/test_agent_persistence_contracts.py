"""Agent checkpoint round-trip contract tests."""

import numpy as np
import pytest
import torch

from config import Config
from src.ai.agent import Agent, TrainingHistory
from src.utils.checkpoint_loader import load_checkpoint


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


def test_agent_load_legacy_checkpoint_defaults_missing_counters(tmp_path):
    """Old checkpoints without newer counters should still resume predictably."""
    config = _small_config()
    source = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    source.epsilon = 0.22
    source.steps = 11
    path = tmp_path / "legacy_checkpoint.pth"
    torch.save(
        {
            "policy_net_state_dict": source.policy_net.state_dict(),
            "target_net_state_dict": source.target_net.state_dict(),
            "optimizer_state_dict": source.optimizer.state_dict(),
            "epsilon": source.epsilon,
            "steps": source.steps,
            "state_size": config.STATE_SIZE,
            "action_size": config.ACTION_SIZE,
        },
        path,
    )

    restored = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    metadata, history = restored.load(str(path), quiet=True)

    assert metadata is None
    assert history is None
    assert restored.epsilon == 0.22
    assert restored.steps == 11
    assert restored._learn_step == 0
    assert restored._next_target_update == 11 + config.TARGET_UPDATE


def test_agent_load_ignores_bad_optional_metadata_history_and_replay(tmp_path):
    """Optional checkpoint sections should not block core model/counter restore."""
    config = _small_config()
    source = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    source.epsilon = 0.18
    source.steps = 19
    state = np.linspace(0, 1, config.STATE_SIZE, dtype=np.float32)
    before_q = source.get_q_values(state)
    path = tmp_path / "bad_optional_sections.pth"
    torch.save(
        {
            "policy_net_state_dict": source.policy_net.state_dict(),
            "target_net_state_dict": source.target_net.state_dict(),
            "optimizer_state_dict": source.optimizer.state_dict(),
            "epsilon": source.epsilon,
            "steps": source.steps,
            "_learn_step": 5,
            "_next_target_update": 25,
            "state_size": config.STATE_SIZE,
            "action_size": config.ACTION_SIZE,
            "metadata": {"timestamp": "missing required fields"},
            "training_history": "not-a-history-dict",
            "replay_buffer": {
                "initialized": True,
                "size": 1,
                "state_size": config.STATE_SIZE,
            },
        },
        path,
    )

    restored = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    metadata, history = restored.load(str(path), quiet=True)

    assert metadata is None
    assert history is None
    assert restored.epsilon == 0.18
    assert restored.steps == 19
    assert restored._learn_step == 5
    assert restored._next_target_update == 25
    assert len(restored.memory) == 0
    assert np.allclose(before_q, restored.get_q_values(state))


def test_agent_save_core_checkpoint_schema_omits_optional_sections_by_default(tmp_path):
    """Default saves should stay lightweight while preserving resume counters."""
    config = _small_config()
    agent = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    agent.epsilon = 0.31
    agent.steps = 23
    agent._learn_step = 7
    agent._next_target_update = 111
    path = tmp_path / "schema_default.pth"

    metadata = agent.save(
        str(path),
        save_reason="periodic",
        episode=4,
        best_score=9,
        avg_score_last_100=3.5,
        win_rate=0.2,
        quiet=True,
    )
    checkpoint = load_checkpoint(str(path), map_location="cpu")

    assert metadata is not None
    assert set(checkpoint) >= {
        "policy_net_state_dict",
        "target_net_state_dict",
        "optimizer_state_dict",
        "epsilon",
        "steps",
        "_learn_step",
        "_next_target_update",
        "state_size",
        "action_size",
        "metadata",
    }
    assert "training_history" not in checkpoint
    assert "replay_buffer" not in checkpoint
    assert checkpoint["epsilon"] == 0.31
    assert checkpoint["steps"] == 23
    assert checkpoint["_learn_step"] == 7
    assert checkpoint["_next_target_update"] == 111
    assert checkpoint["metadata"]["save_reason"] == "periodic"
    assert checkpoint["metadata"]["episode"] == 4
    assert checkpoint["metadata"]["best_score"] == 9
    assert checkpoint["metadata"]["memory_buffer_size"] == 0


def test_agent_save_attaches_history_and_replay_only_when_requested(tmp_path):
    """Explicit full saves should include dashboard history and replay state."""
    config = _small_config()
    agent = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    state = np.linspace(0, 1, config.STATE_SIZE, dtype=np.float32)
    next_state = np.roll(state, 1).astype(np.float32)
    agent.remember(state, 1, 0.5, next_state, False)
    history = TrainingHistory(
        scores=[4],
        rewards=[0.5],
        steps=[8],
        epsilons=[agent.epsilon],
        bricks=[0],
        wins=[False],
        losses=[0.1],
        q_values=[0.2],
        best_score=4,
    )
    path = tmp_path / "full_schema.pth"

    metadata = agent.save(
        str(path),
        training_history=history,
        save_replay_buffer=True,
        quiet=True,
    )
    with pytest.warns(RuntimeWarning, match="unrestricted checkpoint load"):
        checkpoint = load_checkpoint(
            str(path),
            map_location="cpu",
            trusted_dirs=[str(tmp_path)],
            allow_unsafe_fallback=True,
        )

    assert metadata is not None
    assert checkpoint["training_history"]["scores"] == [4]
    assert checkpoint["training_history"]["best_score"] == 4
    assert checkpoint["replay_buffer"]["initialized"] is True
    assert checkpoint["replay_buffer"]["size"] == 1
    assert checkpoint["metadata"]["memory_buffer_size"] == 1


def test_load_weights_only_rejects_architecture_mismatch_without_rewinding_state(tmp_path):
    """Weight-only load should reject wrong sizes without changing training position."""
    config = _small_config()
    source = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    path = tmp_path / "bad_weight_architecture.pth"
    torch.save(
        {
            "policy_net_state_dict": source.policy_net.state_dict(),
            "target_net_state_dict": source.target_net.state_dict(),
            "optimizer_state_dict": source.optimizer.state_dict(),
            "epsilon": 0.99,
            "steps": 999,
            "state_size": config.STATE_SIZE + 1,
            "action_size": config.ACTION_SIZE,
        },
        path,
    )

    live = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    live.epsilon = 0.12
    live.steps = 34

    assert live.load_weights_only(str(path), quiet=True) is False
    assert live.epsilon == 0.12
    assert live.steps == 34


def test_load_weights_only_rejects_bad_state_dict_without_rewinding_state(tmp_path):
    """Malformed weight payloads should fail cleanly and preserve live counters."""
    config = _small_config()
    path = tmp_path / "bad_weight_payload.pth"
    torch.save(
        {
            "policy_net_state_dict": {},
            "target_net_state_dict": {},
            "optimizer_state_dict": {},
            "epsilon": 0.88,
            "steps": 888,
            "state_size": config.STATE_SIZE,
            "action_size": config.ACTION_SIZE,
        },
        path,
    )

    live = Agent(config.STATE_SIZE, config.ACTION_SIZE, config=config)
    live.epsilon = 0.21
    live.steps = 55

    assert live.load_weights_only(str(path), quiet=True) is False
    assert live.epsilon == 0.21
    assert live.steps == 55
