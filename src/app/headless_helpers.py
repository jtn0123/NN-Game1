"""Helper methods for the headless trainer runtime."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


class HeadlessRuntimeHelpersMixin:
    def _eval_win_rate_regressed(self: Any, eval_results: Any) -> bool:
        """Return whether held-out win rate has fallen well below its best."""

        if self.evaluator is None or self.evaluator.best_eval_win_rate <= 0.0:
            return False
        frac = getattr(self.config, "EVAL_BOOST_WIN_REGRESSION_FRAC", 0.7)
        return eval_results.win_rate < frac * self.evaluator.best_eval_win_rate

    def _restore_eval_best(self: Any) -> bool:
        """Reload the eval-best checkpoint's weights into the live policy."""

        eval_best_path = os.path.join(
            self.config.GAME_MODEL_DIR, f"{self.config.GAME_NAME}_eval_best.pth"
        )
        restored = self.agent.load_weights_only(eval_best_path)
        if restored and self.web_dashboard:
            self.web_dashboard.log("↩️ Rolled live policy back to eval-best", "info")
        return restored

    def _apply_lr_decay(self: Any, start_episode: int, current_episode: int) -> None:
        """Cosine-decay the learning rate over this run's episode span."""

        if not getattr(self.config, "LR_DECAY", False):
            return
        target = self.config.MAX_EPISODES
        if target <= 0:
            return
        lr0 = self.config.LEARNING_RATE
        lr_min = getattr(self.config, "LR_MIN", 1e-5)
        span = max(1, target - start_episode)
        frac = min(1.0, max(0.0, (current_episode - start_episode) / span))
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * frac))
        self.agent.set_learning_rate(lr)

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        """Human-friendly time-remaining, e.g. '3m', '1h12m', '45s'."""

        s = int(max(0, seconds))
        if s >= 3600:
            return f"{s // 3600}h{(s % 3600) // 60:02d}m"
        if s >= 60:
            return f"{s // 60}m"
        return f"{s}s"

    def _print_progress_breakdown(self: Any, window: int = 100) -> None:
        """Print recent end-reason and progress-component breakdowns."""

        if not self.end_reasons and not self.progress_parts:
            return

        recent_reasons = self.end_reasons[-window:]
        if recent_reasons:
            counts: dict[str, int] = {}
            for reason in recent_reasons:
                counts[reason] = counts.get(reason, 0) + 1
            total = len(recent_reasons)
            mix = ", ".join(
                f"{name} {count / total * 100:.0f}%"
                for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
            )
            print(f"   End reasons:      {mix}")

        recent_parts = self.progress_parts[-window:]
        if recent_parts:
            keys = ("crystal_frac", "switch_done", "depth_frac", "won")
            means = {
                key: float(np.mean([float(p.get(key, 0.0)) for p in recent_parts])) for key in keys
            }
            print(
                "   Phi@death parts:  "
                f"crystals {means['crystal_frac']:.2f} | "
                f"switch {means['switch_done']:.2f} | "
                f"depth {means['depth_frac']:.2f} | "
                f"won {means['won']:.2f}"
            )
