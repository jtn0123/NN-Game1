"""
Deterministic Model Evaluator
=============================

Runs periodic evaluation with ε=0 to measure TRUE model performance,
separate from noisy training metrics.

This prevents the situation where training metrics show improvement
but the model has actually plateaued.

Usage:
    evaluator = Evaluator(game, agent, config)
    results = evaluator.evaluate(num_episodes=50)

    # During training loop:
    if episode % EVAL_EVERY == 0:
        eval_results = evaluator.evaluate()
        evaluator.log_results(eval_results, episode)
"""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


@dataclass
class EvalResults:
    """Results from a deterministic evaluation run."""

    timestamp: str
    episode: int
    num_games: int

    # Score metrics
    mean_score: float
    median_score: float
    std_score: float
    min_score: int
    max_score: int
    q25_score: float
    q75_score: float

    # Level metrics
    mean_level: float
    max_level: int
    level_distribution: Dict[int, int]

    # Win metrics
    wins: int
    win_rate: float

    # Survival metrics
    mean_steps: float
    max_steps: int

    # Crystal Caves diagnostics. These stay zero/empty for games that do not
    # report Crystal Caves progress_parts/end_reason telemetry.
    mean_crystal_frac: float = 0.0
    mean_switch_rate: float = 0.0
    mean_depth_frac: float = 0.0
    mean_target_distance_progress: float = 0.0
    mean_exit_unlocked_rate: float = 0.0
    end_reason_counts: Dict[str, int] = field(default_factory=dict)

    # Continuous "keep-best" score (set by Evaluator.evaluate). Drives the eval-best
    # checkpoint and the plateau / early-stop signal so a high-score but low-progress
    # policy does not get kept over a better chain-progress policy.
    selection_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Evaluator:
    """
    Runs deterministic evaluation to measure true model performance.

    Key features:
    - Runs with ε=0 (no exploration) for deterministic results
    - Tracks score, level, win rate, and survival time
    - Logs results to JSON for historical comparison
    - Detects plateau (no improvement over N evals)
    """

    def __init__(
        self,
        game,
        agent,
        config,
        log_dir: str = "eval_logs",
        plateau_threshold: int = 5,
    ):
        """
        Initialize the evaluator.

        Args:
            game: Game instance
            agent: Agent instance
            config: Config object
            log_dir: Directory to save evaluation logs
            plateau_threshold: Number of evals without improvement to trigger plateau warning
        """
        self.game = game
        self.agent = agent
        self.config = config
        self.log_dir = log_dir
        self.plateau_threshold = plateau_threshold

        # History tracking
        self.eval_history: List[EvalResults] = []
        self.best_eval_score: float = 0.0
        self.evals_since_improvement: int = 0
        # Keep-best / win-regression signals (see EVAL_SELECTION_* config).
        self.best_eval_selection: float = 0.0
        self.best_eval_win_rate: float = 0.0

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

    def evaluate(
        self, num_episodes: int = 30, max_steps: int = 5000, episode_num: int = 0
    ) -> EvalResults:
        """
        Run deterministic evaluation.

        Args:
            num_episodes: Number of evaluation games to run
            max_steps: Maximum steps per game
            episode_num: Current training episode (for logging)

        Returns:
            EvalResults with all metrics
        """
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        # Save current epsilon and set to 0 for deterministic play
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        # Evaluate on a fixed HELD-OUT set of distinct caves (one per game), unseen
        # during training, identical across evals. Without this the evaluator
        # replayed a single fixed level N times (Score ± 0), which is neither
        # diverse nor a generalisation measure. No-op for authored caves.
        if hasattr(self.game, "use_eval_levels"):
            self.game.use_eval_levels(num_episodes)
        if hasattr(self.game, "reset_eval_cursor"):
            self.game.reset_eval_cursor()

        scores = []
        levels = []
        steps_list = []
        crystal_fracs = []
        switch_rates = []
        depth_fracs = []
        target_distance_progress = []
        exit_unlocked_rates = []
        end_reasons = []
        wins = 0

        try:
            for _ in range(num_episodes):
                state = self.game.reset()
                done = False
                steps = 0
                info = {"score": 0, "level": 1, "won": False}
                target_distances = []
                initial_target_distance = self._target_distance_tiles()
                if initial_target_distance is not None:
                    target_distances.append(initial_target_distance)

                while not done and steps < max_steps:
                    action = self.agent.select_action(state, training=False)
                    state, _, done, info = self.game.step(action)
                    steps += 1
                    target_distance = self._target_distance_tiles()
                    if target_distance is not None:
                        target_distances.append(target_distance)

                score = info.get("score", 0)
                level = info.get("level", 1)
                won = info.get("won", False)
                parts: Any = info.get("progress_parts") or {}
                if isinstance(parts, dict):
                    crystal_fracs.append(float(parts.get("crystal_frac", 0.0) or 0.0))
                    switch_rates.append(float(parts.get("switch_done", 0.0) or 0.0))
                    depth_fracs.append(float(parts.get("depth_frac", 0.0) or 0.0))
                exit_unlocked_rates.append(1.0 if info.get("exit_unlocked", False) else 0.0)
                if target_distances:
                    initial_distance = target_distances[0]
                    min_distance = min(target_distances)
                    if initial_distance > 1e-6:
                        progress = 1.0 - min_distance / initial_distance
                        target_distance_progress.append(float(np.clip(progress, 0.0, 1.0)))
                    else:
                        target_distance_progress.append(1.0 if min_distance <= 1e-6 else 0.0)
                reason = str(info.get("end_reason", "") or "")
                if not reason or reason == "running":
                    reason = "won" if won else ("timeout" if steps >= max_steps else "ended")
                end_reasons.append(reason)

                scores.append(score)
                levels.append(level)
                steps_list.append(steps)
                if won:
                    wins += 1
        finally:
            # Restore epsilon even if the game or agent raises.
            self.agent.epsilon = original_epsilon

        # Calculate statistics
        scores_arr = np.array(scores)
        levels_arr = np.array(levels)
        steps_arr = np.array(steps_list)

        # Level distribution
        level_dist = {}
        max_level_seen = max(10, int(np.max(levels_arr)))
        for lvl in range(1, max_level_seen + 1):
            level_dist[lvl] = int((levels_arr == lvl).sum())

        results = EvalResults(
            timestamp=datetime.now().isoformat(),
            episode=episode_num,
            num_games=num_episodes,
            mean_score=float(np.mean(scores_arr)),
            median_score=float(np.median(scores_arr)),
            std_score=float(np.std(scores_arr)),
            min_score=int(np.min(scores_arr)),
            max_score=int(np.max(scores_arr)),
            q25_score=float(np.percentile(scores_arr, 25)),
            q75_score=float(np.percentile(scores_arr, 75)),
            mean_level=float(np.mean(levels_arr)),
            max_level=int(np.max(levels_arr)),
            level_distribution=level_dist,
            wins=wins,
            win_rate=wins / num_episodes,
            mean_steps=float(np.mean(steps_arr)),
            max_steps=int(np.max(steps_arr)),
            mean_crystal_frac=float(np.mean(crystal_fracs)) if crystal_fracs else 0.0,
            mean_switch_rate=float(np.mean(switch_rates)) if switch_rates else 0.0,
            mean_depth_frac=float(np.mean(depth_fracs)) if depth_fracs else 0.0,
            mean_target_distance_progress=(
                float(np.mean(target_distance_progress)) if target_distance_progress else 0.0
            ),
            mean_exit_unlocked_rate=(
                float(np.mean(exit_unlocked_rates)) if exit_unlocked_rates else 0.0
            ),
            end_reason_counts=dict(Counter(end_reasons)),
        )

        results.selection_score = self._selection_score(results)

        # Update history and check for plateau
        self._update_history(results)

        return results

    def _selection_score(self, results: EvalResults) -> float:
        """Continuous keep-best score (see EVAL_SELECTION_* config).

        Crystal Caves uses dense chain-progress signals so keep-best / plateau can
        see progress before wins appear. Non-Crystal-Caves evaluations keep the
        older win/score behavior.
        """
        has_chain_progress = (
            results.mean_crystal_frac > 0.0
            or results.mean_depth_frac > 0.0
            or results.mean_target_distance_progress > 0.0
            or results.mean_exit_unlocked_rate > 0.0
        )
        if has_chain_progress:
            w_crystal = getattr(self.config, "EVAL_SELECTION_W_CRYSTAL", 0.5)
            w_depth = getattr(self.config, "EVAL_SELECTION_W_DEPTH", 0.3)
            w_target = getattr(self.config, "EVAL_SELECTION_W_TARGET_DISTANCE", 0.2)
            w_exit = getattr(self.config, "EVAL_SELECTION_W_EXIT_UNLOCKED", 1.0)
            return (
                results.mean_crystal_frac * w_crystal
                + results.mean_depth_frac * w_depth
                + results.mean_target_distance_progress * w_target
                + results.mean_exit_unlocked_rate * w_exit
            )

        w_win = getattr(self.config, "EVAL_SELECTION_W_WIN", 1.0)
        w_score = getattr(self.config, "EVAL_SELECTION_W_SCORE", 0.0001)
        return results.win_rate * w_win + results.mean_score * w_score

    def _target_distance_tiles(self) -> float | None:
        current_target = getattr(self.game, "_current_target", None)
        if not callable(current_target):
            return None
        try:
            _, distance = current_target()
        except Exception:
            return None
        if not np.isfinite(distance):
            return None
        tile_size = float(getattr(self.game, "TILE_SIZE", 1.0) or 1.0)
        return float(distance) / max(1.0, tile_size)

    def _update_history(self, results: EvalResults) -> None:
        """Update evaluation history and check for plateau.

        Improvement is measured on selection_score, not raw mean_score, so a
        high-score/low-progress policy does not reset the plateau counter or get
        treated as a new best. best_eval_score still tracks the mean score of the
        current selection-best eval for display/back-compat.
        """
        self.eval_history.append(results)
        self.best_eval_win_rate = max(self.best_eval_win_rate, results.win_rate)

        if results.selection_score > self.best_eval_selection:
            self.best_eval_selection = results.selection_score
            self.best_eval_score = results.mean_score
            self.evals_since_improvement = 0
        else:
            self.evals_since_improvement += 1

    def is_plateau(self) -> bool:
        """Check if model has plateaued (no improvement in N evals)."""
        return self.evals_since_improvement >= self.plateau_threshold

    def log_results(self, results: EvalResults, verbose: bool = True) -> None:
        """
        Log evaluation results to file and console.

        Args:
            results: EvalResults to log
            verbose: Print to console if True
        """
        # Save to JSON log
        log_file = os.path.join(self.log_dir, f"{self.config.GAME_NAME}_eval_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(results.to_dict()) + "\n")

        if verbose:
            self._print_results(results)

    def _print_results(self, results: EvalResults) -> None:
        """Print formatted evaluation results."""
        plateau_warning = " ⚠️ PLATEAU" if self.is_plateau() else ""

        print()
        print("=" * 60)
        print(f"📊 EVAL @ Episode {results.episode}{plateau_warning}")
        print("=" * 60)
        print(
            f"   Score:  {results.mean_score:.0f} ± {results.std_score:.0f} "
            f"(median: {results.median_score:.0f})"
        )
        print(f"   Range:  {results.min_score} - {results.max_score}")
        print(f"   Level:  avg {results.mean_level:.1f}, max {results.max_level}")
        print(f"   Wins:   {results.wins}/{results.num_games} ({results.win_rate*100:.1f}%)")
        if results.mean_crystal_frac or results.mean_switch_rate or results.end_reason_counts:
            print(
                f"   Caves:  crystals {results.mean_crystal_frac*100:.0f}%, "
                f"switch {results.mean_switch_rate*100:.0f}%, "
                f"depth {results.mean_depth_frac*100:.0f}%"
            )
            print(f"   Ends:   {results.end_reason_counts}")
        print(
            f"   Best eval ever: {self.best_eval_score:.0f} "
            f"(no improvement for {self.evals_since_improvement} evals)"
        )
        print("=" * 60)
        print()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.eval_history:
            return {}

        scores = [r.mean_score for r in self.eval_history]
        episodes = [r.episode for r in self.eval_history]

        return {
            "num_evals": len(self.eval_history),
            "best_eval_score": self.best_eval_score,
            "latest_eval_score": scores[-1] if scores else 0,
            "is_plateau": self.is_plateau(),
            "evals_since_improvement": self.evals_since_improvement,
            "score_trend": self._calculate_trend(episodes, scores),
        }

    def _calculate_trend(self, x: List[int], y: List[float]) -> float:
        """Calculate linear trend (slope) of scores over episodes."""
        if len(x) < 2:
            return 0.0

        x_arr = np.array(x)
        y_arr = np.array(y)

        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)

        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sum((x_arr - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        # Return slope per 1000 episodes
        return (numerator / denominator) * 1000


def compare_checkpoints(
    checkpoint_paths: List[str], game_class, agent_class, config, num_episodes: int = 30
) -> List[Dict[str, Any]]:
    """
    Compare multiple checkpoints with deterministic evaluation.

    Args:
        checkpoint_paths: List of paths to checkpoint files
        game_class: Game class to instantiate
        agent_class: Agent class to instantiate
        config: Config object
        num_episodes: Episodes per checkpoint

    Returns:
        List of evaluation results for each checkpoint
    """
    results = []

    for path in checkpoint_paths:
        game = game_class(config)
        agent = agent_class(game.state_size, game.action_size, config)
        agent.load(path)

        evaluator = Evaluator(game, agent, config)
        eval_results = evaluator.evaluate(num_episodes)

        results.append({"checkpoint": os.path.basename(path), "results": eval_results.to_dict()})

    return results
