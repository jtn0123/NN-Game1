"""Dashboard, configuration, and save helpers for headless training."""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Optional

import numpy as np

from src.ai.agent import Agent, TrainingHistory
from src.app.dashboard_bindings import DashboardCallbacks, bind_dashboard_callbacks
from src.app.performance_modes import apply_performance_mode
from src.app.process_control import restart_with_game
from src.app.training_runtime import (
    build_nn_snapshot,
    build_runtime_nn_snapshot,
    emit_nn_snapshot_to_dashboard,
    request_save_and_stop,
)
from src.utils.checkpoint_loader import load_checkpoint


class HeadlessDashboardMixin:
    def _sync_history_to_dashboard_after_load(self: Any, filepath: str) -> None:
        """
        Load training history from checkpoint and sync to dashboard.

        This is called AFTER the web dashboard is initialized, since the initial
        _load_model happens before the dashboard exists.
        """
        if not self.web_dashboard:
            return

        try:
            checkpoint = load_checkpoint(
                filepath,
                map_location=self.config.DEVICE,
                trusted_dirs=[self.config.MODEL_DIR, self.config.GAME_MODEL_DIR],
                allow_unsafe_fallback=True,
            )

            if "training_history" in checkpoint:
                training_history = TrainingHistory.from_dict(checkpoint["training_history"])
                metadata = None
                if "metadata" in checkpoint:
                    from src.ai.agent import SaveMetadata

                    metadata = SaveMetadata.from_dict(checkpoint["metadata"])

                if len(training_history.scores) > 0:
                    self._sync_web_dashboard_history(training_history, metadata)
                    print(f"📊 Dashboard charts restored ({len(training_history.scores)} episodes)")
        except Exception as e:
            print(f"⚠️ Could not restore dashboard history: {e}")

    def _resolve_model_path(
        self: Any, explicit_path: Optional[str], state_size: int, action_size: int
    ) -> Optional[str]:
        """Resolve which model to load on startup."""
        return self.model_service.resolve_model_path(
            explicit_path,
            state_size=state_size,
            action_size=action_size,
        )

    def _setup_web_callbacks(self: Any) -> None:
        """Set up web dashboard control callbacks."""
        if not self.web_dashboard:
            return

        bind_dashboard_callbacks(
            self.web_dashboard,
            DashboardCallbacks(
                pause=self._toggle_pause,
                save=lambda: self._save_model(
                    f"{self.config.GAME_NAME}_web_save.pth", save_reason="manual"
                ),
                save_as=self._save_model_as,
                reset=self._reset_episode,
                start_fresh=self._start_fresh,
                load_model=self._load_model,
                config_change=self._apply_config,
                performance_mode=self._set_performance_mode,
                restart_with_game=lambda game: restart_with_game(game, self.args),
                save_and_quit=self._save_and_quit,
            ),
        )

    def _send_system_info(self: Any) -> None:
        """Send system information to web dashboard."""
        if not self.web_dashboard:
            return

        torch_compiled = getattr(self.agent, "_compiled", False)
        device_str = str(self.config.DEVICE)

        self.web_dashboard.publisher.set_system_info(
            device=device_str,
            torch_compiled=torch_compiled,
            target_episodes=self.config.MAX_EPISODES,
            headless=True,  # No pygame, no screenshots
        )

        # ADD: Set number of parallel environments
        self.web_dashboard.publisher.state.num_envs = self.num_envs

        # Set performance mode based on turbo flag
        if self.args.turbo:
            self.web_dashboard.publisher.set_performance_mode("turbo")
        else:
            self.web_dashboard.publisher.set_performance_mode("normal")

    def _log_startup_info(self: Any) -> None:
        """Log startup configuration to web dashboard."""
        if not self.web_dashboard:
            return

        self.web_dashboard.log("🚀 Headless training started", "success")
        self.web_dashboard.log(f"Device: {self.config.DEVICE}", "info")
        self.web_dashboard.log(
            f"State size: {self.game.state_size}, Action size: {self.game.action_size}",
            "info",
        )
        self.web_dashboard.log(f"Network: {self.config.HIDDEN_LAYERS}", "info")
        self.web_dashboard.log(
            f"Learn every: {self.config.LEARN_EVERY}, Grad steps: {self.config.GRADIENT_STEPS}",
            "info",
        )
        self.web_dashboard.log(f"Batch size: {self.config.BATCH_SIZE}", "info")
        target_str = "Unlimited" if self.config.MAX_EPISODES == 0 else str(self.config.MAX_EPISODES)
        self.web_dashboard.log(f"Target episodes: {target_str}", "info")

    def _toggle_pause(self: Any) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
        if self.web_dashboard:
            self.web_dashboard.publisher.set_paused(self.paused)
            status = "⏸️ Training paused" if self.paused else "▶️ Training resumed"
            self.web_dashboard.log(status, "action")
        print("⏸️  Paused" if self.paused else "▶️  Resumed")

    def _reset_episode(self: Any) -> None:
        """Reset is handled at episode boundary in headless mode."""
        if self.web_dashboard:
            self.web_dashboard.log("🔄 Episode will reset at next boundary", "action")
        print("🔄 Episode reset requested")

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
        self.current_episode = 0
        self.best_score = 0
        self.scores.clear()
        self.wins.clear()
        self.levels.clear()
        self.q_values.clear()
        self.losses.clear()
        self.epsilons.clear()
        self.rewards.clear()
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.target_updates = 0
        self.total_steps = 0
        self.training_start_time = time.time()
        self.last_target_update_step = 0

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
        """Load a model from file and sync history to dashboard."""
        try:
            metadata, training_history = self.agent.load(filepath)

            # If load returned None, model is incompatible (architecture mismatch)
            if metadata is None and training_history is None:
                if self.web_dashboard:
                    self.web_dashboard.log(
                        f"⚠️  Model incompatible: {os.path.basename(filepath)} - starting fresh training",
                        "warning",
                    )
                return  # Skip restoration, start fresh

            # Restore tracking state from metadata
            if metadata:
                self.best_score = metadata.best_score
                self.current_episode = metadata.episode
                self.total_steps = metadata.total_steps

            # Restore local score/win history from training history (no limit - keep full history)
            if training_history and len(training_history.scores) > 0:
                self.scores = training_history.scores.copy()
                self.wins = training_history.wins.copy() if training_history.wins else []
                self.q_values = (
                    training_history.q_values.copy() if training_history.q_values else []
                )
                self.losses = training_history.losses.copy() if training_history.losses else []
                self.epsilons = (
                    training_history.epsilons.copy() if training_history.epsilons else []
                )
                self.rewards = training_history.rewards.copy() if training_history.rewards else []

                # Restore dashboard counters
                self.exploration_actions = training_history.exploration_actions
                self.exploitation_actions = training_history.exploitation_actions
                self.target_updates = training_history.target_updates
                if training_history.best_score > self.best_score:
                    self.best_score = training_history.best_score

            if self.web_dashboard:
                history_msg = ""
                if training_history and len(training_history.scores) > 0:
                    history_msg = f" ({len(training_history.scores)} episodes restored)"
                    # Sync training history to dashboard charts
                    self._sync_web_dashboard_history(training_history, metadata)

                self.web_dashboard.log(
                    f"📂 Loaded model: {os.path.basename(filepath)}{history_msg}",
                    "success",
                    {
                        "path": filepath,
                        "epsilon": self.agent.epsilon,
                        "episode": self.current_episode,
                    },
                )
                # Update save status to reflect loaded model
                self.web_dashboard.publisher.record_save(
                    filename=os.path.basename(filepath),
                    reason="loaded",
                    episode=self.current_episode,
                    best_score=self.best_score,
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

            # Add Q-values (now stored in history)
            if i < len(training_history.q_values):
                publisher.q_values.append(training_history.q_values[i])
            else:
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
            publisher.state.episode = self.current_episode
            publisher.state.best_score = self.best_score
            publisher.state.epsilon = self.agent.epsilon
            publisher.state.total_steps = self.total_steps

        # Restore dashboard counters from training history
        publisher.state.exploration_actions = training_history.exploration_actions
        publisher.state.exploitation_actions = training_history.exploitation_actions
        publisher.state.target_updates = training_history.target_updates

        # Calculate win rate from history
        if training_history.wins:
            recent_wins = training_history.wins[-100:]
            publisher.state.win_rate = (
                sum(1 for w in recent_wins if w) / len(recent_wins) if recent_wins else 0.0
            )

        # Emit an update to connected clients
        self.web_dashboard.socketio.emit("state_update", publisher.get_snapshot())

    def _save_model_as(self: Any, filename: str) -> bool:
        """Save model with custom filename."""
        if not filename.endswith(".pth"):
            filename += ".pth"
        success = self._save_model(filename, save_reason="manual")
        if success and self.web_dashboard:
            self.web_dashboard.log(f"💾 Saved as: {filename}", "success")
        return success

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
            snapshot = build_runtime_nn_snapshot(
                self.agent,
                self.game,
                state,
                step=self.agent.steps,
                snapshot_builder=snapshot_builder,
            )
            emit_nn_snapshot_to_dashboard(
                self.web_dashboard,
                snapshot,
                selected_action=selected_action,
                step=self.agent.steps,
            )
        except Exception as exc:
            if not getattr(self, "_nn_visualization_error_reported", False):
                self._nn_visualization_error_reported = True
                message = f"⚠️ Neural network visualization update failed: {type(exc).__name__}"
                if self.web_dashboard:
                    self.web_dashboard.log(message, "warning")
                else:
                    print(message)

    def _apply_config(self: Any, config_data: dict) -> None:
        """Apply configuration changes from web dashboard."""
        if not isinstance(config_data, dict):
            print("⚠️  Ignoring invalid config payload")
            return
        changes = []

        if "learning_rate" in config_data:
            try:
                lr = float(config_data["learning_rate"])
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError("Learning rate must be finite and positive")
                if lr > 10.0:
                    raise ValueError(f"Learning rate {lr} is unreasonably large (max 10.0)")
                if lr < 1e-10:
                    raise ValueError(f"Learning rate {lr} is too small (min 1e-10)")
                old_lr = self.config.LEARNING_RATE
                self.config.LEARNING_RATE = lr
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

    def _set_performance_mode(self: Any, mode: str) -> None:
        """Set performance mode preset."""
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

    def _save_and_quit(self: Any) -> None:
        """Save the model and exit the application gracefully."""
        request_save_and_stop(
            game_name=self.config.GAME_NAME,
            save_model=lambda filename, reason: self._save_model(filename, save_reason=reason),
            set_running=self._set_running,
            dashboard=self.web_dashboard,
        )

    def _set_running(self: Any, running: bool) -> None:
        self.running = running

    def _save_model(
        self: Any,
        filename: str,
        save_reason: str = "manual",
        quiet: bool = False,
        save_replay_buffer: bool = True,
    ) -> bool:
        """Save the current model with rich metadata.

        Args:
            filename: Name of file to save
            save_reason: Why this save is happening
            quiet: Suppress output if True
            save_replay_buffer: Include replay buffer for cross-session persistence

        Returns:
            True if save succeeded, False otherwise
        """
        # Ensure game-specific model directory exists
        os.makedirs(self.config.GAME_MODEL_DIR, exist_ok=True)
        filepath = os.path.join(self.config.GAME_MODEL_DIR, filename)

        # Calculate metrics for metadata
        avg_score = np.mean(self.scores[-100:]) if self.scores else 0.0
        recent_wins = self.wins[-100:]
        win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0.0
        max_level = max(self.levels[-100:]) if self.levels else 1

        # Build training history for dashboard restoration
        # Keep last 100000 episodes for full chart history
        history_limit = 100000
        training_history = TrainingHistory(
            scores=self.scores[-history_limit:],
            rewards=self.rewards[-history_limit:],
            steps=[],  # Not tracked per-episode in vectorized mode
            epsilons=self.epsilons[-history_limit:],
            bricks=[],  # Not tracked in Space Invaders
            wins=self.wins[-history_limit:],
            losses=self.losses[-history_limit:],
            q_values=self.q_values[-history_limit:],
            exploration_actions=self.exploration_actions,
            exploitation_actions=self.exploitation_actions,
            target_updates=self.target_updates,
            best_score=self.best_score,
        )

        result = self.agent.save(
            filepath=filepath,
            save_reason=save_reason,
            episode=self.current_episode,
            best_score=self.best_score,
            avg_score_last_100=float(avg_score),
            win_rate=win_rate,
            max_level=max_level,
            training_start_time=self.training_start_time,
            training_history=training_history,
            save_replay_buffer=save_replay_buffer,
            quiet=quiet,
        )

        # Notify web dashboard about save
        if result is not None and self.web_dashboard:
            self.web_dashboard.publisher.record_save(
                filename=filename,
                reason=save_reason,
                episode=self.current_episode,
                best_score=self.best_score,
            )
            if not quiet:
                self.web_dashboard.log(f"💾 Saved: {filename} ({save_reason})", "success")

        return result is not None

    def _cleanup_old_periodic_saves(self: Any, keep_last: int = 5) -> None:
        """
        Delete old periodic checkpoint saves, keeping only the most recent ones.

        This prevents disk space bloat from accumulating ep100.pth, ep200.pth, etc.
        Important saves (best, final, interrupted) are NOT deleted.

        Args:
            keep_last: Number of recent periodic saves to keep
        """
        for filepath in self.model_service.cleanup_old_periodic_saves(keep_last=keep_last):
            if self.web_dashboard:
                self.web_dashboard.log(
                    f"🗑️ Cleaned up old checkpoint: {os.path.basename(filepath)}",
                    "info",
                )
