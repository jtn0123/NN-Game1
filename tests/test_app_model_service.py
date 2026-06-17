import os
from types import SimpleNamespace

from config import Config
from src.ai.agent import Agent
from src.app.model_service import ModelService


def test_app_model_service_skips_newer_incompatible_checkpoint(tmp_path, monkeypatch):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    older = model_dir / "older.pth"
    newer = model_dir / "newer.pth"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_010, 1_700_000_010))

    def fake_inspect_model(path, **kwargs):
        if os.path.basename(path) == "newer.pth":
            return {"state_size": 99, "action_size": 2}
        return {"state_size": 4, "action_size": 2}

    monkeypatch.setattr(Agent, "inspect_model", staticmethod(fake_inspect_model))

    resolved = ModelService(config).resolve_model_path(
        explicit_path=None,
        state_size=4,
        action_size=2,
        log=lambda _message: None,
    )

    assert resolved == str(older)


def test_app_model_service_resolves_explicit_checkpoint_compatibility(tmp_path, monkeypatch):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "explicit.pth"
    checkpoint.write_bytes(b"model")
    seen_logs = []

    monkeypatch.setattr(
        Agent,
        "inspect_model",
        staticmethod(lambda _path, **_kwargs: {"state_size": 4, "action_size": 2}),
    )

    service = ModelService(config)
    assert service.resolve_model_path(str(checkpoint), 4, 2, log=seen_logs.append) == str(
        checkpoint
    )

    monkeypatch.setattr(
        Agent,
        "inspect_model",
        staticmethod(lambda _path, **_kwargs: {"state_size": 99, "action_size": 2}),
    )

    assert service.resolve_model_path(str(checkpoint), 4, 2, log=seen_logs.append) is None
    assert any("Specified model incompatible" in message for message in seen_logs)


def test_app_model_service_returns_none_without_model_dir_or_candidates(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    service = ModelService(config)

    assert service.resolve_model_path(None, 4, 2, log=lambda _message: None) is None

    os.makedirs(config.GAME_MODEL_DIR)
    assert service.resolve_model_path(None, 4, 2, log=lambda _message: None) is None


def test_app_model_service_returns_none_when_all_candidates_incompatible(tmp_path, monkeypatch):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    (model_dir / "bad.pth").write_bytes(b"model")
    logs = []
    monkeypatch.setattr(
        Agent,
        "inspect_model",
        staticmethod(lambda _path, **_kwargs: {"state_size": 99, "action_size": 2}),
    )

    resolved = ModelService(config).resolve_model_path(None, 4, 2, log=logs.append)

    assert resolved is None
    assert any("No compatible saved model found" in message for message in logs)


def test_cleanup_old_periodic_saves_removes_checkpoints_and_sidecars(tmp_path):
    """Periodic checkpoint cleanup should not depend on Agent internals."""
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)

    for episode in range(6):
        checkpoint = model_dir / f"breakout_ep{episode}.pth"
        checkpoint.write_bytes(b"x" * 1200)
        sidecar = model_dir / f"breakout_ep{episode}.pth.json"
        sidecar.write_text("{}", encoding="utf-8")

    ModelService(config).cleanup_old_periodic_saves(keep_last=5)

    assert not (model_dir / "breakout_ep0.pth").exists()
    assert not (model_dir / "breakout_ep0.pth.json").exists()
    assert (model_dir / "breakout_ep1.pth").exists()


def test_app_model_service_paths_and_history_payloads(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    service = ModelService(config)

    checkpoint_path = service.checkpoint_path("../unsafe name")

    assert checkpoint_path == os.path.join(config.GAME_MODEL_DIR, "unsafename.pth")
    assert os.path.isdir(config.GAME_MODEL_DIR)
    assert service.metadata_sidecar_path(checkpoint_path) == f"{checkpoint_path}.json"

    visual_history = service.build_visual_history(
        [
            SimpleNamespace(score=10, reward=1.5, steps=20, epsilon=0.8, bricks_hit=3, won=False),
            SimpleNamespace(score=30, reward=2.5, steps=40, epsilon=0.6, bricks_hit=8, won=True),
        ],
        losses=[0.9, 0.4, 0.1],
        limit=1,
    )

    assert visual_history.scores == [30]
    assert visual_history.rewards == [2.5]
    assert visual_history.steps == [40]
    assert visual_history.epsilons == [0.6]
    assert visual_history.bricks == [8]
    assert visual_history.wins == [True]
    assert visual_history.losses == [0.9, 0.4, 0.1]

    headless_history = service.build_headless_history(
        scores=[1, 2, 3],
        rewards=[0.1, 0.2, 0.3],
        epsilons=[0.9, 0.8, 0.7],
        wins=[False, True, True],
        losses=[0.5, 0.4, 0.3],
        q_values=[1.1, 1.2, 1.3],
        exploration_actions=12,
        exploitation_actions=34,
        target_updates=5,
        best_score=99,
        limit=2,
    )

    assert headless_history.scores == [2, 3]
    assert headless_history.rewards == [0.2, 0.3]
    assert headless_history.epsilons == [0.8, 0.7]
    assert headless_history.wins == [True, True]
    assert headless_history.losses == [0.4, 0.3]
    assert headless_history.q_values == [1.2, 1.3]
    assert headless_history.exploration_actions == 12
    assert headless_history.exploitation_actions == 34
    assert headless_history.target_updates == 5
    assert headless_history.best_score == 99


def test_app_model_service_stat_helpers():
    assert ModelService.average_last_100([]) == 0.0
    assert ModelService.average_last_100(range(101)) == 50.5
    assert ModelService.win_rate_last_100([]) == 0.0
    assert ModelService.win_rate_last_100([True, False, True]) == 2 / 3
    assert ModelService.max_recent_level([]) == 1
    assert ModelService.max_recent_level([1, 4, 2]) == 4


def test_cleanup_old_periodic_saves_reports_os_errors(tmp_path, monkeypatch, capsys):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    for episode in range(3):
        (model_dir / f"breakout_ep{episode}.pth").write_bytes(b"x")

    def fail_remove(path):
        raise OSError("permission denied")

    monkeypatch.setattr(os, "remove", fail_remove)

    deleted = ModelService(config).cleanup_old_periodic_saves(keep_last=1)

    assert deleted == []
    assert "Could not delete old checkpoint" in capsys.readouterr().out


def test_cleanup_old_periodic_saves_keeps_short_history(tmp_path):
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = str(tmp_path / "models")
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    checkpoints = [model_dir / f"breakout_ep{episode}.pth" for episode in range(2)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"x")

    deleted = ModelService(config).cleanup_old_periodic_saves(keep_last=5)

    assert deleted == []
    assert all(checkpoint.exists() for checkpoint in checkpoints)
