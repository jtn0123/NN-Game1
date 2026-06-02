"""Tests for shared model persistence helpers."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from config import Config
from src.ai.agent import Agent
from src.app import ModelService

pytestmark = pytest.mark.torch


@pytest.fixture
def service(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    return ModelService(config)


def test_normalize_checkpoint_filename_strips_paths_and_suffix(service):
    """Custom save names should not become arbitrary paths."""
    assert service.normalize_checkpoint_filename("../bad/name!.pth") == "name.pth"
    assert service.normalize_checkpoint_filename(" spaced save ") == "spacedsave.pth"
    assert service.normalize_checkpoint_filename("!!!") == "custom_save.pth"


def test_checkpoint_path_uses_game_model_dir(service):
    """Checkpoints should land in the active game's model directory."""
    path = service.checkpoint_path("custom")

    assert path.endswith("models/breakout/custom.pth")


def test_resolve_model_path_rejects_incompatible_explicit_model(service, tmp_path, monkeypatch):
    """Explicit model paths should be checked before use."""
    model_path = tmp_path / "bad.pth"
    model_path.write_text("checkpoint", encoding="utf-8")
    messages = []
    monkeypatch.setattr(
        Agent,
        "inspect_model",
        staticmethod(lambda path: {"state_size": 99, "action_size": 3}),
    )

    result = service.resolve_model_path(str(model_path), 10, 3, log=messages.append)

    assert result is None
    assert any("incompatible" in message for message in messages)


def test_resolve_model_path_selects_newest_game_checkpoint(service):
    """Auto-load should use the newest game-specific checkpoint."""
    model_dir = service.ensure_model_dir()
    older = service.checkpoint_path("breakout_ep1")
    newer = service.checkpoint_path("breakout_ep2")
    with open(older, "w", encoding="utf-8") as f:
        f.write("older")
    with open(newer, "w", encoding="utf-8") as f:
        f.write("newer")

    result = service.resolve_model_path(None, 10, 3, log=lambda message: None)

    assert result == newer
    assert model_dir in result


def test_build_visual_history_maps_episode_metrics(service):
    """Visual episode history should serialize through the shared TrainingHistory path."""
    episodes = [
        SimpleNamespace(score=10, reward=1.5, steps=20, epsilon=0.8, bricks_hit=3, won=False),
        SimpleNamespace(score=30, reward=2.5, steps=25, epsilon=0.7, bricks_hit=8, won=True),
    ]

    history = service.build_visual_history(episodes, losses=[0.1, 0.05])

    assert history.scores == [10, 30]
    assert history.rewards == [1.5, 2.5]
    assert history.steps == [20, 25]
    assert history.bricks == [3, 8]
    assert history.wins == [False, True]
    assert history.losses == [0.1, 0.05]


def test_cleanup_old_periodic_saves_removes_sidecars(service):
    """Periodic cleanup should prune checkpoint files and their JSON sidecars."""
    kept = []
    for episode in range(1, 5):
        path = service.checkpoint_path(f"breakout_ep{episode}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("checkpoint")
        sidecar = Agent.metadata_sidecar_path(path)
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write("{}")
        kept.append((path, sidecar))

    service.cleanup_old_periodic_saves(keep_last=2)

    assert not os.path.exists(kept[0][0])
    assert not os.path.exists(kept[0][1])
    assert os.path.exists(kept[-1][0])
    assert os.path.exists(kept[-1][1])
