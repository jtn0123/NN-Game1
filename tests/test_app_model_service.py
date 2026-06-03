import os

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
