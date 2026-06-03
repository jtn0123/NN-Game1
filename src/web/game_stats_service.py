"""Game statistics helpers for the web dashboard."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict

from src.game import get_game_info, list_games
from src.utils.checkpoint_loader import load_checkpoint
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def build_game_stats(
    config: Any,
    *,
    checkpoint_loader: Callable[..., Dict[str, Any]] = load_checkpoint,
) -> Dict[str, Dict[str, Any]]:
    """Build dashboard comparison statistics for all registered games."""
    stats = {}
    for game_id in list_games():
        game_info = get_game_info(game_id)
        game_model_dir = os.path.join(config.MODEL_DIR, game_id)

        game_stats: Dict[str, Any] = {
            "name": (
                game_info.get("name", game_id.title()) if game_info else game_id.title()
            ),
            "icon": game_info.get("icon", "🎮") if game_info else "🎮",
            "color": (
                game_info.get("color", (100, 100, 100))
                if game_info
                else (100, 100, 100)
            ),
            "best_score": 0,
            "total_episodes": 0,
            "total_training_time": 0,
            "model_count": 0,
            "best_model": None,
        }

        if os.path.exists(game_model_dir):
            for filename in os.listdir(game_model_dir):
                if not filename.endswith(".pth"):
                    continue

                game_stats["model_count"] += 1
                path = os.path.join(game_model_dir, filename)
                try:
                    checkpoint = checkpoint_loader(
                        path,
                        map_location="cpu",
                        trusted_dirs=[game_model_dir],
                        allow_unsafe_fallback=False,
                    )
                except Exception as exc:
                    _logger.debug(f"Could not load stats from {filename}: {exc}")
                    continue

                metadata = checkpoint.get("metadata")
                if not metadata:
                    continue

                if metadata.get("best_score", 0) > game_stats["best_score"]:
                    game_stats["best_score"] = metadata["best_score"]
                    game_stats["best_model"] = filename
                if metadata.get("episode", 0) > game_stats["total_episodes"]:
                    game_stats["total_episodes"] = metadata["episode"]
                game_stats["total_training_time"] += metadata.get(
                    "total_training_time_seconds", 0
                )

        stats[game_id] = game_stats

    return stats
