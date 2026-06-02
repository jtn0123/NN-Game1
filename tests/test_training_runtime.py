import os
from types import SimpleNamespace

import numpy as np

from src.app.training_runtime import build_nn_snapshot, resolve_model_path
from src.ai.agent import Agent
from config import Config


def test_resolve_model_path_uses_latest_compatible_checkpoint(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    older = model_dir / "older.pth"
    newer = model_dir / "newer.pth"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    older_mtime = 1_700_000_000
    newer_mtime = older_mtime + 10
    older.touch()
    newer.touch()
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))

    def inspect_model(path, **kwargs):
        return {"state_size": 4, "action_size": 2}

    resolved = resolve_model_path(
        explicit_path=None,
        state_size=4,
        action_size=2,
        config=config,
        inspect_model=inspect_model,
    )

    assert resolved == str(newer)


def test_build_nn_snapshot_creates_sampled_and_full_payloads():
    config = Config()
    config.HIDDEN_LAYERS = [32, 16]
    config.USE_DUELING = False
    config.USE_N_STEP_RETURNS = False
    agent = Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config,
    )
    game = SimpleNamespace(get_action_labels=lambda: ["LEFT", "RIGHT"])
    state = np.zeros(config.STATE_SIZE, dtype=np.float32)

    snapshot = build_nn_snapshot(agent, game, state)

    assert snapshot.action_labels == ["LEFT", "RIGHT"]
    assert snapshot.input_state == [0.0] * config.STATE_SIZE
    assert len(snapshot.q_values) == config.ACTION_SIZE
    assert snapshot.weights
    assert snapshot.analysis_weights
    assert len(snapshot.analysis_weights[0][0]) == config.STATE_SIZE
    assert len(snapshot.weights[0]) <= 15
