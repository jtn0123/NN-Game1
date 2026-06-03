"""Tests for dashboard game statistics aggregation."""

from config import Config
from src.web.game_stats_service import build_game_stats


def test_build_game_stats_uses_restricted_metadata_scan(tmp_path):
    """Game stats should not enable unsafe checkpoint fallback while scanning."""
    config = Config()
    config.MODEL_DIR = str(tmp_path / "models")
    breakout_dir = tmp_path / "models" / "breakout"
    breakout_dir.mkdir(parents=True)
    (breakout_dir / "demo.pth").write_bytes(b"checkpoint")
    calls = []

    def fake_load_checkpoint(*args, **kwargs):
        calls.append(kwargs.get("allow_unsafe_fallback"))
        return {"metadata": {"best_score": 10, "episode": 2}}

    stats = build_game_stats(config, checkpoint_loader=fake_load_checkpoint)

    assert stats["breakout"]["model_count"] == 1
    assert stats["breakout"]["best_score"] == 10
    assert calls
    assert set(calls) == {False}


def test_build_game_stats_ignores_unreadable_checkpoints(tmp_path):
    """Bad checkpoint files should not break the whole comparison panel."""
    config = Config()
    config.MODEL_DIR = str(tmp_path / "models")
    breakout_dir = tmp_path / "models" / "breakout"
    breakout_dir.mkdir(parents=True)
    (breakout_dir / "bad.pth").write_bytes(b"not a checkpoint")

    def failing_loader(*args, **kwargs):
        raise RuntimeError("bad checkpoint")

    stats = build_game_stats(config, checkpoint_loader=failing_loader)

    assert stats["breakout"]["model_count"] == 1
    assert stats["breakout"]["best_score"] == 0
    assert stats["breakout"]["best_model"] is None
