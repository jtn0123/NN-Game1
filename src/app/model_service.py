"""Shared model persistence helpers for app entrypoints."""

import glob
import os
import re
from typing import Any, Callable, Iterable, Optional

import numpy as np

from config import Config
from src.ai.agent import Agent, TrainingHistory
from src.app.checkpoint_catalog import iter_checkpoint_candidates
from src.app.model_paths import normalize_checkpoint_filename


class ModelService:
    """Owns model filenames, paths, discovery, and history payload construction."""

    def __init__(self, config: Config):
        self.config = config

    def ensure_model_dir(self) -> str:
        """Create and return the active game-specific model directory."""
        model_dir = self.config.GAME_MODEL_DIR
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def normalize_checkpoint_filename(self, filename: str, default: str = "custom_save") -> str:
        """Return a safe checkpoint filename with a `.pth` suffix."""
        return normalize_checkpoint_filename(filename, default=default)

    def checkpoint_path(self, filename: str) -> str:
        """Return the absolute path for a normalized checkpoint filename."""
        return os.path.join(self.ensure_model_dir(), self.normalize_checkpoint_filename(filename))

    def resolve_model_path(
        self,
        explicit_path: Optional[str],
        state_size: int,
        action_size: int,
        log: Callable[[str], None] = print,
    ) -> Optional[str]:
        """Resolve the explicit compatible checkpoint or newest game-specific save."""
        trusted_dirs = [self.config.MODEL_DIR, self.config.GAME_MODEL_DIR]
        if explicit_path and os.path.exists(explicit_path):
            info = Agent.inspect_model(
                explicit_path,
                trusted_dirs=trusted_dirs,
                allow_unsafe_fallback=True,
            )
            if (
                info
                and info.get("state_size") == state_size
                and info.get("action_size") == action_size
            ):
                return explicit_path

            log(f"⚠️  Specified model incompatible: {os.path.basename(explicit_path)}")
            log(f"   Expected: state_size={state_size}, action_size={action_size}")
            if info:
                log(
                    f"   Model has: state_size={info.get('state_size')}, action_size={info.get('action_size')}"
                )
            return None

        model_dir = self.config.GAME_MODEL_DIR
        if not os.path.exists(model_dir):
            return None

        model_files = iter_checkpoint_candidates([(model_dir, self.config.GAME_NAME)])
        if not model_files:
            return None

        for candidate in model_files:
            model_path = candidate.path
            info = Agent.inspect_model(
                model_path,
                trusted_dirs=trusted_dirs,
                allow_unsafe_fallback=True,
            )
            if (
                info
                and info.get("state_size") == state_size
                and info.get("action_size") == action_size
            ):
                log(f"📂 Auto-loading most recent compatible save: {os.path.basename(model_path)}")
                return model_path
            log(f"⚠️  Skipping incompatible save: {os.path.basename(model_path)}")

        log("⚠️  No compatible saved model found for this game")
        return None

    def build_visual_history(
        self,
        episode_history: Iterable[Any],
        losses: Iterable[float],
        limit: int = 100000,
    ) -> TrainingHistory:
        """Build TrainingHistory from visual-mode EpisodeMetrics entries."""
        episodes = list(episode_history)[-limit:]
        return TrainingHistory(
            scores=[ep.score for ep in episodes],
            rewards=[ep.reward for ep in episodes],
            steps=[ep.steps for ep in episodes],
            epsilons=[ep.epsilon for ep in episodes],
            bricks=[ep.bricks_hit for ep in episodes],
            wins=[ep.won for ep in episodes],
            losses=list(losses)[-1000:],
            q_values=[],
        )

    def build_headless_history(
        self,
        scores: list[int],
        rewards: list[float],
        epsilons: list[float],
        wins: list[bool],
        losses: list[float],
        q_values: list[float],
        exploration_actions: int,
        exploitation_actions: int,
        target_updates: int,
        best_score: int,
        limit: int = 100000,
    ) -> TrainingHistory:
        """Build TrainingHistory from headless-mode metric lists."""
        return TrainingHistory(
            scores=scores[-limit:],
            rewards=rewards[-limit:],
            steps=[],
            epsilons=epsilons[-limit:],
            bricks=[],
            wins=wins[-limit:],
            losses=losses[-limit:],
            q_values=q_values[-limit:],
            exploration_actions=exploration_actions,
            exploitation_actions=exploitation_actions,
            target_updates=target_updates,
            best_score=best_score,
        )

    @staticmethod
    def average_last_100(values: Iterable[float]) -> float:
        """Average the last 100 values, returning 0.0 for empty input."""
        recent = list(values)[-100:]
        return float(np.mean(recent)) if recent else 0.0

    @staticmethod
    def win_rate_last_100(wins: list[bool]) -> float:
        """Return the recent win rate."""
        recent = wins[-100:]
        return sum(recent) / len(recent) if recent else 0.0

    @staticmethod
    def max_recent_level(levels: list[int]) -> int:
        """Return recent max level, defaulting to 1 when no level history exists."""
        return max(levels[-100:]) if levels else 1

    @staticmethod
    def metadata_sidecar_path(checkpoint_path: str) -> str:
        """Return the JSON metadata sidecar path for a checkpoint file."""
        return f"{checkpoint_path}.json"

    def cleanup_old_periodic_saves(self, keep_last: int = 5) -> list[str]:
        """Delete old periodic checkpoints and metadata sidecars, returning removed paths."""
        pattern = os.path.join(self.config.GAME_MODEL_DIR, f"{self.config.GAME_NAME}_ep*.pth")
        periodic_saves = glob.glob(pattern)
        if len(periodic_saves) <= keep_last:
            return []

        def get_episode_num(path: str) -> int:
            match = re.search(r"_ep(\d+)\.pth$", path)
            return int(match.group(1)) if match else 0

        periodic_saves.sort(key=get_episode_num)
        deleted: list[str] = []
        for filepath in periodic_saves[:-keep_last]:
            try:
                os.remove(filepath)
                deleted.append(filepath)
                sidecar_path = self.metadata_sidecar_path(filepath)
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
            except OSError as exc:
                print(f"⚠️ Could not delete old checkpoint {filepath}: {exc}")
        return deleted
