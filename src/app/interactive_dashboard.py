"""Dashboard, model, and save helpers for the interactive runtime."""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from src.ai.agent import Agent, TrainingHistory
from src.app.lifecycle_types import EpisodeMetrics
from src.app.performance_modes import apply_performance_mode
from src.app.training_runtime import build_nn_snapshot, emit_nn_snapshot_to_dashboard
from src.utils.checkpoint_loader import load_checkpoint


class InteractiveDashboardMixin:
    def _restore_training_history(self: Any, filepath: str) -> None:
        """Restore training history from a saved model (called after dashboard is ready)."""
        try:
            checkpoint = load_checkpoint(
                filepath,
                map_location=self.config.DEVICE,
                trusted_dirs=[self.config.MODEL_DIR, self.config.GAME_MODEL_DIR],
                allow_unsafe_fallback=True,
            )

            if "training_history" in checkpoint:
                history_data = checkpoint["training_history"]
                training_history = TrainingHistory.from_dict(history_data)

                if len(training_history.scores) > 0:
                    # Restore internal tracking from TrainingHistory
                    self.episode_history.clear()
                    for i in range(len(training_history.scores)):
                        self.episode_history.append(
                            EpisodeMetrics(
                                score=training_history.scores[i],
                                reward=(
                                    training_history.rewards[i]
                                    if i < len(training_history.rewards)
                                    else 0.0
                                ),
                                steps=(
                                    training_history.steps[i]
                                    if i < len(training_history.steps)
                                    else 0
                                ),
                                epsilon=(
                                    training_history.epsilons[i]
                                    if i < len(training_history.epsilons)
                                    else 1.0
                                ),
                                bricks_hit=(
                                    training_history.bricks[i]
                                    if i < len(training_history.bricks)
                                    else 0
                                ),
                                won=(
                                    training_history.wins[i]
                                    if i < len(training_history.wins)
                                    else False
                                ),
                            )
                        )
                    self.recent_scores = deque(training_history.scores[-1000:], maxlen=1000)

                    # Restore dashboard with historical data
                    for i, score in enumerate(training_history.scores):
                        eps = (
                            training_history.epsilons[i]
                            if i < len(training_history.epsilons)
                            else 0.5
                        )
                        reward = (
                            training_history.rewards[i]
                            if i < len(training_history.rewards)
                            else 0.0
                        )
                        bricks = (
                            training_history.bricks[i] if i < len(training_history.bricks) else 0
                        )
                        won = training_history.wins[i] if i < len(training_history.wins) else False
                        loss = (
                            training_history.losses[i] if i < len(training_history.losses) else 0.0
                        )

                        self.dashboard.update(
                            episode=i,
                            score=score,
                            epsilon=eps,
                            loss=loss,
                            bricks_broken=bricks,
                            won=won,
                            reward=reward,
                        )

                    print(
                        f"📊 Restored {len(training_history.scores)} episodes of training history"
                    )

            # Restore episode counter and best score from metadata
            if "metadata" in checkpoint:
                metadata = checkpoint["metadata"]
                self.episode = metadata.get("episode", len(self.episode_history))
                self.best_score_ever = metadata.get("best_score", 0)
            elif self.episode_history:
                self.episode = len(self.episode_history)
                self.best_score_ever = max(ep.score for ep in self.episode_history)

        except Exception as e:
            print(f"⚠️ Could not restore training history: {e}")

    def _log_startup_info(self: Any) -> None:
        """Log startup configuration to web dashboard."""
        if not self.web_dashboard:
            return

        self.web_dashboard.log("🚀 Training session started", "success")
        self.web_dashboard.log(f"Device: {self.config.DEVICE}", "info")
        self.web_dashboard.log(
            f"State size: {self.game.state_size}, Action size: {self.game.action_size}",
            "info",
        )
        self.web_dashboard.log(
            f"Network: {self.config.HIDDEN_LAYERS}",
            "info",
            {
                "hidden_layers": self.config.HIDDEN_LAYERS,
                "activation": self.config.ACTIVATION,
            },
        )
        self.web_dashboard.log(
            f"Learning rate: {self.config.LEARNING_RATE}",
            "info",
            {
                "lr": self.config.LEARNING_RATE,
                "gamma": self.config.GAMMA,
                "batch_size": self.config.BATCH_SIZE,
            },
        )
        self.web_dashboard.log(
            f"Epsilon: {self.config.EPSILON_START} → {self.config.EPSILON_END}",
            "info",
            {
                "start": self.config.EPSILON_START,
                "end": self.config.EPSILON_END,
                "decay": self.config.EPSILON_DECAY,
            },
        )
        target_str = "Unlimited" if self.config.MAX_EPISODES == 0 else str(self.config.MAX_EPISODES)
        self.web_dashboard.log(f"Target episodes: {target_str}", "info")
        self.web_dashboard.log("Ready to train! Use controls to manage training.", "info")

    def _start_fresh(self: Any) -> None:
        """Start fresh training - reset agent, clear memory, reset all training state."""

        if self.web_dashboard:
            # Clear logs first, then log the fresh start
            self.web_dashboard.publisher.console_logs.clear()
            self.web_dashboard.publisher.reset_all_state()
            self.web_dashboard.log(
                "🔄 Starting fresh training - resetting agent and clearing memory",
                "warning",
            )

        # Create a new agent (fresh neural network)
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=self.config,
        )

        # Clear replay buffer
        self.agent.memory.clear()

        # Reset all training state
        self.episode = 0
        self.total_reward = 0.0
        self.steps = 0
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.target_updates = 0
        self.last_target_update_step = 0
        self.training_start_time = time.time()
        self.best_score_ever = 0

        # Clear training history
        self.recent_scores.clear()
        self.episode_history.clear()

        # Reset game state
        self.state = self.game.reset()
        self.selected_action: Optional[int] = None

        if self.web_dashboard:
            # Update publisher state with fresh values
            publisher = self.web_dashboard.publisher
            publisher.state.episode = 0
            publisher.state.score = 0
            publisher.state.best_score = 0
            publisher.state.total_steps = 0
            publisher.state.epsilon = self.config.EPSILON_START
            publisher.state.memory_size = 0
            publisher.state.loss = 0.0
            publisher.state.win_rate = 0.0
            publisher.state.avg_q_value = 0.0
            publisher.state.exploration_actions = 0
            publisher.state.exploitation_actions = 0
            publisher.state.target_updates = 0
            publisher.state.total_reward = 0.0
            publisher.state.bricks_broken_total = 0
            publisher.state.episodes_per_second = 0.0
            publisher.state.steps_per_second = 0.0
            publisher.state.training_start_time = time.time()

            # Emit reset event to frontend to clear charts
            self.web_dashboard.socketio.emit(
                "training_reset", {"message": "Training reset - starting fresh"}
            )

            # Emit cleared logs
            self.web_dashboard.socketio.emit("console_logs", {"logs": []})

            # Emit updated state with empty history
            self.web_dashboard.socketio.emit("state_update", publisher.get_snapshot())

            self.web_dashboard.log(
                "✅ Fresh training started - agent reset, memory cleared, all charts and logs reset",
                "success",
            )

        print("✅ Fresh training started - agent reset, memory cleared")

    def _load_model(self: Any, filepath: str) -> None:
        """Load a model from file and restore training history."""
        try:
            metadata, training_history = self.agent.load(filepath, quiet=True)

            # If load returned None, model is incompatible (architecture mismatch)
            if metadata is None and training_history is None:
                if self.web_dashboard:
                    self.web_dashboard.log(
                        f"⚠️  Model incompatible: {os.path.basename(filepath)} - starting fresh training",
                        "warning",
                    )
                return  # Skip restoration, start fresh

            # Restore training history if available
            if training_history and len(training_history.scores) > 0:
                self.episode_history.clear()
                for i in range(len(training_history.scores)):
                    self.episode_history.append(
                        EpisodeMetrics(
                            score=training_history.scores[i],
                            reward=(
                                training_history.rewards[i]
                                if i < len(training_history.rewards)
                                else 0.0
                            ),
                            steps=(
                                training_history.steps[i] if i < len(training_history.steps) else 0
                            ),
                            epsilon=(
                                training_history.epsilons[i]
                                if i < len(training_history.epsilons)
                                else 1.0
                            ),
                            bricks_hit=(
                                training_history.bricks[i]
                                if i < len(training_history.bricks)
                                else 0
                            ),
                            won=(
                                training_history.wins[i]
                                if i < len(training_history.wins)
                                else False
                            ),
                        )
                    )
                self.recent_scores = deque(training_history.scores[-1000:], maxlen=1000)

                # Restore dashboard with historical data
                for i, score in enumerate(training_history.scores):
                    eps = (
                        training_history.epsilons[i] if i < len(training_history.epsilons) else 0.5
                    )
                    reward = (
                        training_history.rewards[i] if i < len(training_history.rewards) else 0.0
                    )
                    bricks = training_history.bricks[i] if i < len(training_history.bricks) else 0
                    won = training_history.wins[i] if i < len(training_history.wins) else False
                    loss = training_history.losses[i] if i < len(training_history.losses) else 0.0

                    self.dashboard.update(
                        episode=i,
                        score=score,
                        epsilon=eps,
                        loss=loss,
                        bricks_broken=bricks,
                        won=won,
                        reward=reward,
                    )

                # Restore episode counter and best score from metadata
                if metadata:
                    self.episode = metadata.episode
                    self.best_score_ever = metadata.best_score
                    self.steps = metadata.total_steps
                else:
                    self.episode = len(training_history.scores)
                    self.best_score_ever = (
                        max(training_history.scores) if training_history.scores else 0
                    )

                history_msg = f" ({len(training_history.scores)} episodes restored)"

                # Sync web dashboard with restored history
                if self.web_dashboard:
                    self._sync_web_dashboard_history(training_history, metadata)
            else:
                history_msg = " (no history)"
                # Still restore episode counter from metadata if available
                if metadata:
                    self.episode = metadata.episode
                    self.best_score_ever = metadata.best_score
                    self.steps = metadata.total_steps

            if self.web_dashboard:
                self.web_dashboard.log(
                    f"📂 Loaded model: {os.path.basename(filepath)}{history_msg}",
                    "success",
                    {
                        "path": filepath,
                        "epsilon": self.agent.epsilon,
                        "steps": self.agent.steps,
                        "episode": self.episode,
                        "best_score": self.best_score_ever,
                    },
                )
                # Update save status to reflect loaded model
                self.web_dashboard.publisher.record_save(
                    filename=os.path.basename(filepath),
                    reason="loaded",
                    episode=self.episode,
                    best_score=self.best_score_ever,
                )
        except Exception as e:
            if self.web_dashboard:
                self.web_dashboard.log(f"❌ Failed to load model: {str(e)}", "error")
            import traceback

            traceback.print_exc()

    def _sync_web_dashboard_history(
        self: Any, training_history: TrainingHistory, metadata: Optional[Any]
    ) -> None:
        """Sync training history to web dashboard metrics publisher."""
        if not self.web_dashboard:
            return

        publisher = self.web_dashboard.publisher

        # Clear existing history in publisher
        publisher.scores.clear()
        publisher.losses.clear()
        publisher.epsilons.clear()
        publisher.rewards.clear()
        publisher.q_values.clear()
        publisher.episode_lengths.clear()

        # Restore scores to publisher for chart display
        for i, score in enumerate(training_history.scores):
            publisher.scores.append(score)

            # Add losses if available
            if i < len(training_history.losses):
                publisher.losses.append(training_history.losses[i])
            else:
                publisher.losses.append(0.0)

            # Add epsilons
            if i < len(training_history.epsilons):
                publisher.epsilons.append(training_history.epsilons[i])
            else:
                publisher.epsilons.append(0.5)

            # Add rewards
            if i < len(training_history.rewards):
                publisher.rewards.append(training_history.rewards[i])
            else:
                publisher.rewards.append(0.0)

            # Add steps as episode lengths
            if i < len(training_history.steps):
                publisher.episode_lengths.append(training_history.steps[i])
            else:
                publisher.episode_lengths.append(0)

            # Q-values aren't stored in history, so use placeholder
            publisher.q_values.append(0.0)

        # Update publisher state from metadata
        if metadata:
            publisher.state.episode = metadata.episode
            publisher.state.best_score = metadata.best_score
            publisher.state.epsilon = metadata.epsilon
            publisher.state.total_steps = metadata.total_steps
            publisher.state.memory_size = metadata.memory_buffer_size
            publisher.state.loss = metadata.avg_loss
        else:
            publisher.state.episode = self.episode
            publisher.state.best_score = self.best_score_ever
            publisher.state.epsilon = self.agent.epsilon
            publisher.state.total_steps = self.agent.steps

        # Calculate win rate from history
        if training_history.wins:
            recent_wins = training_history.wins[-100:]
            publisher.state.win_rate = (
                sum(1 for w in recent_wins if w) / len(recent_wins) if recent_wins else 0.0
            )

        # Emit an update to connected clients
        self.web_dashboard.socketio.emit("state_update", publisher.get_snapshot())

    def _apply_config(self: Any, config_data: dict) -> None:
        """Apply configuration changes from web dashboard."""
        if not isinstance(config_data, dict):
            print("⚠️  Ignoring invalid config payload")
            return
        changes = []

        if "learning_rate" in config_data:
            try:
                lr = float(config_data["learning_rate"])
                # Bug 80 fix: Add reasonable range validation for learning rate
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError("Learning rate must be finite and positive")
                if lr > 10.0:
                    raise ValueError(f"Learning rate {lr} is unreasonably large (max 10.0)")
                if lr < 1e-10:
                    raise ValueError(f"Learning rate {lr} is too small (min 1e-10)")
                old_lr = self.config.LEARNING_RATE
                self.config.LEARNING_RATE = lr
                # Update optimizer learning rate
                for param_group in self.agent.optimizer.param_groups:
                    param_group["lr"] = lr
                changes.append(f"LR: {old_lr} → {lr}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid learning_rate value: {config_data['learning_rate']} - {e}")

        if "epsilon" in config_data:
            try:
                eps = float(config_data["epsilon"])
                if not math.isfinite(eps):
                    raise ValueError("Epsilon must be finite (not NaN or Inf)")
                old_eps = self.agent.epsilon
                # Clamp epsilon to valid range with feedback
                clamped_eps = max(self.config.EPSILON_END, min(self.config.EPSILON_START, eps))
                if clamped_eps != eps:
                    print(
                        f"⚠️  Epsilon {eps:.4f} clamped to valid range [{self.config.EPSILON_END}, {self.config.EPSILON_START}]"
                    )
                self.agent.epsilon = clamped_eps
                changes.append(f"Epsilon: {old_eps:.4f} → {self.agent.epsilon:.4f}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid epsilon value: {config_data['epsilon']} - {e}")

        if "epsilon_decay" in config_data:
            try:
                decay = float(config_data["epsilon_decay"])
                if not math.isfinite(decay) or decay <= 0 or decay > 1:
                    raise ValueError("Epsilon decay must be finite and in (0, 1]")
                self.config.EPSILON_DECAY = decay
                changes.append(f"Decay: {decay}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid epsilon_decay value: {config_data['epsilon_decay']} - {e}")

        if "gamma" in config_data:
            try:
                gamma = float(config_data["gamma"])
                if not math.isfinite(gamma) or gamma < 0 or gamma > 1:
                    raise ValueError("Gamma must be finite and in [0, 1]")
                self.config.GAMMA = gamma
                changes.append(f"Gamma: {gamma}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid gamma value: {config_data['gamma']} - {e}")

        if "batch_size" in config_data:
            try:
                batch_size = int(config_data["batch_size"])
                if batch_size <= 0:
                    raise ValueError("Batch size must be positive")
                if batch_size > self.config.MEMORY_SIZE:
                    raise ValueError(
                        f"Batch size ({batch_size}) cannot exceed memory size ({self.config.MEMORY_SIZE})"
                    )
                self.config.BATCH_SIZE = batch_size
                changes.append(f"Batch: {batch_size}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid batch_size value: {config_data['batch_size']} - {e}")

        if "learn_every" in config_data:
            try:
                learn_every = int(config_data["learn_every"])
                if learn_every <= 0:
                    raise ValueError("Learn every must be positive")
                self.config.LEARN_EVERY = learn_every
                changes.append(f"LearnEvery: {learn_every}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid learn_every value: {config_data['learn_every']} - {e}")

        if "gradient_steps" in config_data:
            try:
                grad_steps = int(config_data["gradient_steps"])
                if grad_steps <= 0:
                    raise ValueError("Gradient steps must be positive")
                self.config.GRADIENT_STEPS = grad_steps
                changes.append(f"GradSteps: {grad_steps}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid gradient_steps value: {config_data['gradient_steps']} - {e}")

        if self.web_dashboard and changes:
            self.web_dashboard.log(
                f"⚙️ Config updated: {', '.join(changes)}", "action", config_data
            )

    def _send_system_info(self: Any) -> None:
        """Send system information to web dashboard."""
        if not self.web_dashboard:
            return

        # Check if torch.compile was used
        torch_compiled = getattr(self.agent, "_compiled", False)
        device_str = str(self.config.DEVICE)

        self.web_dashboard.publisher.set_system_info(
            device=device_str,
            torch_compiled=torch_compiled,
            target_episodes=self.config.MAX_EPISODES,
        )

    def _set_performance_mode(self: Any, mode: str) -> None:
        """Set performance mode from web dashboard."""
        try:
            preset = apply_performance_mode(self.config, mode)
        except KeyError:
            print(f"⚠️  Unknown performance mode: {mode}")
            return

        if self.web_dashboard:
            self.web_dashboard.publisher.set_performance_mode(mode)
            self.web_dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
            self.web_dashboard.publisher.state.batch_size = self.config.BATCH_SIZE
            self.web_dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS
            self.web_dashboard.log(
                f"⚡ Performance mode: {mode.upper()} (learn_every={preset.learn_every}, batch={preset.batch_size}, grad_steps={preset.gradient_steps})",
                "action",
            )
        print(f"⚡ Performance mode: {mode.upper()}")

    def _emit_nn_visualization(self: Any, state: np.ndarray, selected_action: int) -> None:
        """
        Extract and emit neural network visualization data to web dashboard.

        Args:
            state: Current game state
            selected_action: Currently selected action
        """
        if not self.web_dashboard:
            return

        try:
            runtime_module = sys.modules.get(self.__class__.__module__)
            snapshot_builder = getattr(runtime_module, "build_nn_snapshot", build_nn_snapshot)
            snapshot = snapshot_builder(self.agent, self.game, state)
            emit_nn_snapshot_to_dashboard(
                self.web_dashboard,
                snapshot,
                selected_action=selected_action,
                step=self.agent.steps,
            )
        except Exception as exc:
            # Don't crash training on visualization errors, but surface the first failure.
            if not getattr(self, "_nn_visualization_error_reported", False):
                self._nn_visualization_error_reported = True
                message = f"⚠️ Neural network visualization update failed: {type(exc).__name__}"
                if self.web_dashboard:
                    self.web_dashboard.log(message, "warning")
                else:
                    print(message)

    def _save_model(
        self: Any, filename: str, save_reason: str = "manual", quiet: bool = False
    ) -> bool:
        """Save the current model with rich metadata.

        Returns:
            True if save succeeded, False otherwise
        """
        # Ensure game-specific model directory exists
        os.makedirs(self.config.GAME_MODEL_DIR, exist_ok=True)
        filepath = os.path.join(self.config.GAME_MODEL_DIR, filename)

        # Calculate metrics for metadata
        avg_score = self._average_recent_scores(self.recent_scores)
        win_rate = self.dashboard.get_win_rate() if hasattr(self.dashboard, "get_win_rate") else 0.0

        # Build training history for dashboard restoration from episode_history
        training_history = TrainingHistory(
            scores=[ep.score for ep in self.episode_history],
            rewards=[ep.reward for ep in self.episode_history],
            steps=[ep.steps for ep in self.episode_history],
            epsilons=[ep.epsilon for ep in self.episode_history],
            bricks=[ep.bricks_hit for ep in self.episode_history],
            wins=[ep.won for ep in self.episode_history],
            losses=list(self.agent.losses)[-1000:] if self.agent.losses else [],
            q_values=[],  # Q-values not tracked per-episode currently
        )

        result = self.agent.save(
            filepath=filepath,
            save_reason=save_reason,
            episode=self.episode,
            best_score=self.best_score_ever,
            avg_score_last_100=float(avg_score),
            win_rate=win_rate,
            training_start_time=self.training_start_time,
            training_history=training_history,
            quiet=quiet,
        )

        # Only record save to web dashboard if save succeeded
        if result is not None and self.web_dashboard:
            self.web_dashboard.publisher.record_save(
                filename=filename,
                reason=save_reason,
                episode=self.episode,
                best_score=self.best_score_ever,
            )

        return result is not None

    def _save_model_as(self: Any, filename: str) -> None:
        """Save model with a custom filename (from web dashboard)."""
        # Remove .pth extension if present (we'll add it back later)
        if filename.endswith(".pth"):
            filename = filename[:-4]

        # Sanitize filename (only alphanumeric, underscore, hyphen)
        filename = "".join(c for c in filename if c.isalnum() or c in "_-").strip()

        # Ensure we have a valid base name (not empty or just dots)
        if not filename or filename.replace(".", "") == "":
            filename = "custom_save"

        # Add .pth extension
        filename = filename + ".pth"

        self._save_model(filename, save_reason="manual")
        if self.web_dashboard:
            self.web_dashboard.log(f"💾 Saved as: {filename}", "success")

    def _cleanup_old_periodic_saves(self: Any, keep_last: int = 5) -> None:
        """
        Delete old periodic checkpoint saves, keeping only the most recent ones.
        """
        for filepath in self.model_service.cleanup_old_periodic_saves(keep_last=keep_last):
            if self.web_dashboard:
                self.web_dashboard.log(
                    f"🗑️ Cleaned up old checkpoint: {os.path.basename(filepath)}",
                    "info",
                )
