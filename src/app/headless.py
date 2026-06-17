"""Headless training runtime for high-throughput local training."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Optional, cast

import numpy as np

from config import Config
from src.ai.agent import Agent, TrainingHistory
from src.ai.evaluator import Evaluator
from src.ai.trainer import Trainer, calculate_progress_count
from src.app.game_factory import create_training_environment
from src.app.model_service import ModelService as AppModelService
from src.app.performance_modes import apply_performance_mode
from src.app.process_control import restart_with_game
from src.app.training_runtime import (
    build_nn_snapshot,
    emit_nn_snapshot_to_dashboard,
    request_save_and_stop,
)
from src.game import BaseGame, BaseVecGame, list_games
from src.utils.checkpoint_loader import load_checkpoint

WEB_AVAILABLE: bool
WebDashboard: Optional[type[Any]]
try:
    from src.web import WebDashboard as _WebDashboard

    WebDashboard = _WebDashboard
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    WebDashboard = None


class HeadlessTrainer:
    """
    Lightweight headless trainer that skips pygame entirely.

    This provides maximum training throughput by:
        - No pygame initialization
        - No visualization overhead
        - Optimized training loop with configurable learning frequency
        - Progress reporting via terminal
        - Optional web dashboard for remote monitoring

    Usage:
        python main.py --headless --turbo --episodes 5000
        python main.py --headless --turbo --web --port 5001  # With web dashboard
    """

    def __init__(
        self,
        config: Config,
        args: argparse.Namespace,
        existing_dashboard: Optional[Any] = None,
    ):
        """
        Initialize headless trainer.

        Args:
            config: Configuration object
            args: Command line arguments
            existing_dashboard: Optional existing WebDashboard to reuse (for launcher mode)
        """
        self.config = config
        self.args = args
        self._existing_dashboard = existing_dashboard
        self.running = True
        self.model_service = AppModelService(config)

        # Apply CLI overrides to config
        if args.lr:
            config.LEARNING_RATE = args.lr
        if args.episodes:
            config.MAX_EPISODES = args.episodes
        if args.learn_every:
            config.LEARN_EVERY = args.learn_every
        if args.gradient_steps:
            config.GRADIENT_STEPS = args.gradient_steps
        if args.batch_size:
            config.BATCH_SIZE = args.batch_size
        if args.torch_compile:
            config.USE_TORCH_COMPILE = True

        # Apply turbo preset (overrides individual settings)
        # Optimized for M4 CPU based on benchmarks
        if args.turbo:
            config.LEARN_EVERY = 8
            config.BATCH_SIZE = 128
            config.GRADIENT_STEPS = 2
            config.USE_TORCH_COMPILE = False  # No benefit for small models on CPU
            config.FORCE_CPU = True  # CPU is faster for this model size
            print("🚀 Turbo mode: CPU, B=128, LE=8, GS=2 (~5000 steps/sec on M4)")

        # Vectorized environment support
        self.num_envs = getattr(args, "vec_envs", 1)

        # Get game environment from registry
        try:
            game_environment = create_training_environment(
                config.GAME_NAME,
                config,
                num_envs=self.num_envs,
                headless=True,
            )
        except ValueError:
            print(f"❌ Unknown game: {config.GAME_NAME}")
            print(f"   Available games: {', '.join(list_games())}")
            sys.exit(1)

        self.vec_env: Optional[BaseVecGame] = game_environment.vec_env
        self.game: BaseGame = game_environment.game
        GameClass = game_environment.game_class
        if self.num_envs > 1 and self.vec_env is None:
            print(f"⚠️ Vectorized environments not yet supported for {config.GAME_NAME}")
            print(f"   Falling back to single environment")
        self.num_envs = game_environment.num_envs
        if self.vec_env is not None:
            print(f"🎮 Vectorized: {self.num_envs} parallel environments")

        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config,
        )

        # Load model if specified (headless mode - just restore agent state)
        # Note: Compatibility check happens in _resolve_model_path, so we skip explicit load here
        # and let _resolve_model_path handle it to avoid duplicate loading

        # Create game-specific model directory
        os.makedirs(config.GAME_MODEL_DIR, exist_ok=True)

        # Tracking for loaded model info
        self.best_score = 0
        self.current_episode = 0
        self.scores: list[int] = []
        self.wins: list[bool] = []
        self.levels: list[int] = []  # Track level reached per episode
        self.q_values: list[float] = []  # Track Q-values for chart persistence
        self.losses: list[float] = []  # Track losses for chart persistence
        self.epsilons: list[float] = []  # Track epsilon for chart persistence
        self.rewards: list[float] = []  # Track rewards for chart persistence
        self.total_steps = 0
        self.training_start_time = time.time()

        # Extended metrics tracking (previously missing from headless mode)
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.target_updates = 0
        self.last_target_update_step = 0

        # Web dashboard (initialize early so _load_model can use it)
        self.web_dashboard: Optional[Any] = None

        # Auto-load most recent model if no explicit model specified
        initial_model_path = self._resolve_model_path(
            args.model,
            state_size=self.game.state_size,
            action_size=self.game.action_size,
        )
        if initial_model_path:
            self._load_model(initial_model_path)

        # Setup web dashboard if enabled
        self.paused = False
        if self._existing_dashboard is not None:
            # Reuse existing dashboard from launcher mode (already running on the port)
            self.web_dashboard = self._existing_dashboard
            self._setup_web_callbacks()
            # Dashboard is already started, just send system info
            self._send_system_info()
            self._log_startup_info()
            if initial_model_path:
                self.web_dashboard.log(
                    f"📂 Auto-loaded: {os.path.basename(initial_model_path)}", "success"
                )
                # Sync history to dashboard NOW that dashboard is ready
                self._sync_history_to_dashboard_after_load(initial_model_path)
        elif hasattr(args, "web") and args.web and WEB_AVAILABLE and WebDashboard is not None:
            self.web_dashboard = WebDashboard(
                config, port=args.port, host=getattr(args, "host", "127.0.0.1")
            )
            self._setup_web_callbacks()
            self.web_dashboard.start()

            # Show URL prominently
            print("\n" + "=" * 60)
            print(f"🌐 WEB DASHBOARD: {self.web_dashboard.dashboard_url()}")
            print("=" * 60 + "\n")

            self._send_system_info()
            self._log_startup_info()
            if initial_model_path:
                self.web_dashboard.log(
                    f"📂 Auto-loaded: {os.path.basename(initial_model_path)}", "success"
                )
                # Sync history to dashboard NOW that dashboard is ready
                self._sync_history_to_dashboard_after_load(initial_model_path)

        # Initialize evaluator for deterministic performance tracking
        # Uses a separate single-game instance (not vectorized) for consistent eval
        self.evaluator: Optional[Evaluator] = None
        self._exploration_boost_active: bool = False
        self._exploration_boost_end_episode: int = 0
        if config.EVAL_EVERY > 0:
            eval_game = GameClass(config, headless=True)  # type: ignore[call-arg]
            self.evaluator = Evaluator(
                game=eval_game,
                agent=self.agent,
                config=config,
                log_dir=os.path.join(config.LOG_DIR, "eval"),
                plateau_threshold=config.EVAL_PLATEAU_THRESHOLD,
            )

    def _sync_history_to_dashboard_after_load(self, filepath: str) -> None:
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
        self, explicit_path: Optional[str], state_size: int, action_size: int
    ) -> Optional[str]:
        """Resolve which model to load on startup."""
        return self.model_service.resolve_model_path(
            explicit_path,
            state_size=state_size,
            action_size=action_size,
        )

    def _setup_web_callbacks(self) -> None:
        """Set up web dashboard control callbacks."""
        if not self.web_dashboard:
            return

        self.web_dashboard.on_pause_callback = self._toggle_pause
        self.web_dashboard.on_save_callback = lambda: self._save_model(
            f"{self.config.GAME_NAME}_web_save.pth", save_reason="manual"
        )
        self.web_dashboard.on_save_as_callback = self._save_model_as
        self.web_dashboard.on_reset_callback = self._reset_episode
        self.web_dashboard.on_start_fresh_callback = self._start_fresh
        self.web_dashboard.on_load_model_callback = self._load_model
        self.web_dashboard.on_config_change_callback = self._apply_config
        self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
        self.web_dashboard.on_restart_with_game_callback = lambda game: restart_with_game(
            game, self.args
        )
        self.web_dashboard.on_save_and_quit_callback = self._save_and_quit
        # Speed control doesn't apply to headless (no frame timing)
        self.web_dashboard.on_speed_callback = lambda x: None

    def _send_system_info(self) -> None:
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

    def _log_startup_info(self) -> None:
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

    def _toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
        if self.web_dashboard:
            self.web_dashboard.publisher.set_paused(self.paused)
            status = "⏸️ Training paused" if self.paused else "▶️ Training resumed"
            self.web_dashboard.log(status, "action")
        print("⏸️  Paused" if self.paused else "▶️  Resumed")

    def _reset_episode(self) -> None:
        """Reset is handled at episode boundary in headless mode."""
        if self.web_dashboard:
            self.web_dashboard.log("🔄 Episode will reset at next boundary", "action")
        print("🔄 Episode reset requested")

    def _start_fresh(self) -> None:
        """Start fresh training - reset agent, clear memory, reset all training state."""
        from src.ai.agent import Agent

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

    def _load_model(self, filepath: str) -> None:
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
        self, training_history: TrainingHistory, metadata: Optional[Any]
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

    def _save_model_as(self, filename: str) -> bool:
        """Save model with custom filename."""
        if not filename.endswith(".pth"):
            filename += ".pth"
        success = self._save_model(filename, save_reason="manual")
        if success and self.web_dashboard:
            self.web_dashboard.log(f"💾 Saved as: {filename}", "success")
        return success

    def _emit_nn_visualization(self, state: np.ndarray, selected_action: int) -> None:
        """
        Extract and emit neural network visualization data to web dashboard.

        Args:
            state: Current game state
            selected_action: Currently selected action
        """
        if not self.web_dashboard:
            return

        try:
            snapshot = build_nn_snapshot(self.agent, self.game, state)
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

    def _apply_config(self, config_data: dict) -> None:
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

    def _set_performance_mode(self, mode: str) -> None:
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

    def _save_and_quit(self) -> None:
        """Save the model and exit the application gracefully."""
        request_save_and_stop(
            game_name=self.config.GAME_NAME,
            save_model=lambda filename, reason: self._save_model(filename, save_reason=reason),
            set_running=self._set_running,
            dashboard=self.web_dashboard,
        )

    def _set_running(self, running: bool) -> None:
        self.running = running

    def train(self) -> None:
        """Run headless training loop with optimized throughput."""
        # Dispatch to vectorized training if using multiple environments
        if self.num_envs > 1:
            self.train_vectorized()
            return

        config = self.config

        # Calculate starting episode (may be resuming from loaded model)
        start_episode = self.current_episode

        print("\n" + "=" * 70)
        print("🚀 HEADLESS TRAINING - Maximum Performance Mode")
        if self.web_dashboard:
            print("🌐 Web dashboard enabled")
        print("=" * 70)
        eps_str = "∞ (Unlimited)" if config.MAX_EPISODES == 0 else str(config.MAX_EPISODES)
        print(f"   Episodes:        {start_episode} → {eps_str}")
        print(f"   Device:          {config.DEVICE}")
        print(f"   Batch size:      {config.BATCH_SIZE}")
        print(f"   Learn every:     {config.LEARN_EVERY} steps")
        print(f"   Gradient steps:  {config.GRADIENT_STEPS}")
        print(f"   torch.compile:   {config.USE_TORCH_COMPILE}")
        if self.best_score > 0:
            print(f"   Resumed best:    {self.best_score}")
        print("=" * 70 + "\n")

        # Training timing
        self.training_start_time = time.time()
        last_report_time = self.training_start_time
        steps_since_report = 0
        last_logged_episode = start_episode - 1  # Track last logged episode to prevent duplicates

        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        episode = start_episode
        while self.running and (config.MAX_EPISODES == 0 or episode < config.MAX_EPISODES):
            self.current_episode = episode

            # Handle pause (only if web dashboard is active)
            while self.running and self.paused:
                time.sleep(0.1)
            if not self.running:
                break

            state = self.game.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            info = {"score": 0, "won": False}

            while not done and episode_steps < config.MAX_STEPS_PER_EPISODE:
                # Handle pause during episode
                while self.paused:
                    time.sleep(0.1)

                # Select action
                action = self.agent.select_action(state, training=True)

                # Track exploration vs exploitation
                if self.agent._last_action_explored:
                    self.exploration_actions += 1
                else:
                    self.exploitation_actions += 1

                # Execute action
                next_state, reward, done, info = self.game.step(action)

                # Store experience
                self.agent.remember(state, action, reward, next_state, done)

                # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
                self.agent.learn()

                # Track target network updates
                if (
                    self.agent.steps % config.TARGET_UPDATE == 0
                    and self.agent.steps != self.last_target_update_step
                ):
                    self.target_updates += 1
                    self.last_target_update_step = self.agent.steps
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"🎯 Target network updated (#{self.target_updates})",
                            "metric",
                            {
                                "step": self.agent.steps,
                                "update_number": self.target_updates,
                            },
                        )

                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                steps_since_report += 1

            # Episode complete
            self.agent.decay_epsilon(episode)
            self.agent.step_scheduler()  # Step learning rate scheduler
            self.scores.append(info["score"])

            # Track wins (all bricks cleared)
            won = bool(info.get("won", False))
            self.wins.append(won)

            bricks_broken = calculate_progress_count(info, config)

            # Update web dashboard metrics (throttled to every 5 episodes for performance)
            # Always emit on: first 10 episodes, new best score, or every 5th episode
            is_new_best = info["score"] > getattr(self, "best_score", 0)
            dashboard = self.web_dashboard
            if dashboard is not None and (episode <= 10 or is_new_best or episode % 5 == 0):
                avg_loss = self.agent.get_average_loss(100)

                # Calculate average Q-value for current state (was missing from headless)
                q_values = self.agent.get_q_values(state)
                avg_q_value = float(np.mean(q_values))

                dashboard.emit_metrics(
                    episode=episode,
                    score=info["score"],
                    epsilon=self.agent.epsilon,
                    loss=avg_loss,
                    total_steps=self.total_steps,
                    won=won,
                    reward=episode_reward,
                    memory_size=len(self.agent.memory),
                    avg_q_value=avg_q_value,
                    exploration_actions=self.exploration_actions,
                    exploitation_actions=self.exploitation_actions,
                    target_updates=self.target_updates,
                    bricks_broken=bricks_broken,
                    episode_length=episode_steps,
                )
                # Update performance settings in dashboard state
                dashboard.publisher.state.learn_every = config.LEARN_EVERY
                dashboard.publisher.state.gradient_steps = config.GRADIENT_STEPS
                dashboard.publisher.state.batch_size = config.BATCH_SIZE

                # Emit NN visualization data (throttled by server to ~10 FPS)
                self._emit_nn_visualization(state, action)

            # Progress reporting (terminal) - only log when episodes complete
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time

            # Handle fresh start: if current_episode was reset (e.g., by _start_fresh)
            if self.current_episode < episode:
                episode = self.current_episode
                last_logged_episode = -1  # Reset so logging can resume
                start_episode = 0
                last_report_time = current_time
                steps_since_report = 0

            # Log every LOG_EVERY episodes OR if REPORT_INTERVAL_SECONDS has passed since last log
            # Only log if this is a new episode (prevent duplicate logs)
            should_log_by_episode = (episode - last_logged_episode) >= config.LOG_EVERY
            should_log_by_time = elapsed_since_report >= config.REPORT_INTERVAL_SECONDS
            is_new_episode = episode > last_logged_episode

            if is_new_episode and (should_log_by_episode or should_log_by_time):
                elapsed_total = current_time - self.training_start_time
                steps_per_sec = (
                    steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                )
                eps_per_hour = (
                    (episode - start_episode) / elapsed_total * 3600 if elapsed_total > 0 else 0
                )
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0

                progress_msg = (
                    f"Ep {episode:5d} | "
                    f"Score: {info['score']:4d} | "
                    f"Avg: {avg_score:6.1f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"⚡ {steps_per_sec:,.0f} steps/s | "
                    f"📊 {eps_per_hour:,.0f} ep/hr"
                )

                print(progress_msg)

                # Also log to web dashboard console
                if self.web_dashboard:
                    self.web_dashboard.log(progress_msg, "metric")

                last_logged_episode = episode
                last_report_time = current_time
                steps_since_report = 0

            # Save checkpoints
            if info["score"] > self.best_score:
                self.best_score = info["score"]
                self._save_model(
                    f"{self.config.GAME_NAME}_best.pth", save_reason="best", quiet=True
                )
                if self.web_dashboard:
                    self.web_dashboard.log(f"🏆 New best score: {self.best_score}", "success")

            if episode % config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(
                    f"{self.config.GAME_NAME}_ep{episode}.pth",
                    save_reason="periodic",
                    save_replay_buffer=False,  # Periodic saves are lightweight
                )
                self._cleanup_old_periodic_saves(keep_last=5)

            # Increment episode counter (was implicit in for loop, now explicit for while loop)
            episode += 1

        # Final save
        self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="final")

        # Summary
        total_time = time.time() - self.training_start_time
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Total episodes:   {self.current_episode - start_episode}")
        print(f"   Total steps:      {self.total_steps:,}")
        print(f"   Total time:       {total_time/60:.1f} minutes")
        print(f"   Avg steps/sec:    {self.total_steps/total_time:,.0f}")
        print(f"   Best score:       {self.best_score}")
        print(f"   Final avg score:  {np.mean(self.scores[-100:]):.1f}")
        recent_wins = self.wins[-100:]
        win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0
        print(f"   Win rate (100):   {win_rate*100:.1f}%")
        print("=" * 70)

        if self.web_dashboard:
            self.web_dashboard.log("✅ Training complete!", "success")

    def train_vectorized(self) -> None:
        """
        Run headless training with vectorized environments for parallel execution.

        This method runs N games simultaneously, collecting N experiences per step
        and performing batched action selection for improved throughput.
        """
        # This method is only called when num_envs > 1, so vec_env is always set
        assert self.vec_env is not None, "train_vectorized requires vec_env to be initialized"

        config = self.config
        num_envs = self.num_envs

        # Calculate starting episode (may be resuming from loaded model)
        start_episode = self.current_episode

        print("\n" + "=" * 70)
        print("🚀 VECTORIZED HEADLESS TRAINING - Maximum Performance Mode")
        if self.web_dashboard:
            print("🌐 Web dashboard enabled")
        print("=" * 70)
        print(f"   Environments:    {num_envs} parallel games")
        eps_str = "∞ (Unlimited)" if config.MAX_EPISODES == 0 else str(config.MAX_EPISODES)
        print(f"   Episodes:        {start_episode} → {eps_str}")
        print(f"   Device:          {config.DEVICE}")
        print(f"   Batch size:      {config.BATCH_SIZE}")
        print(f"   Learn every:     {config.LEARN_EVERY} steps")
        print(f"   Gradient steps:  {config.GRADIENT_STEPS}")
        print(f"   torch.compile:   {config.USE_TORCH_COMPILE}")
        if self.best_score > 0:
            print(f"   Resumed best:    {self.best_score}")
        print("=" * 70 + "\n")

        # Training timing
        self.training_start_time = time.time()
        last_report_time = self.training_start_time
        steps_since_report = 0

        # Per-environment episode tracking
        env_episode_rewards = np.zeros(
            num_envs, dtype=np.float64
        )  # Use float64 to prevent precision loss
        env_episode_steps = np.zeros(num_envs, dtype=np.int64)
        episodes_completed = 0

        # Initialize all environments
        states = (
            self.vec_env.reset().copy()
        )  # Shape: (num_envs, state_size) - copy to avoid aliasing

        # Track last completed episode info for reporting
        last_score = 0
        last_info: dict = {}
        last_logged_episode = start_episode - 1  # Track last logged episode to prevent duplicates

        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        while self.running and (
            config.MAX_EPISODES == 0 or self.current_episode < config.MAX_EPISODES
        ):
            # Handle pause (only if web dashboard is active)
            while self.running and self.paused:
                time.sleep(0.1)
            if not self.running:
                break

            # Batch action selection for all environments
            actions, num_explored, num_exploited = self.agent.select_actions_batch(
                states, training=True
            )
            self.exploration_actions += num_explored
            self.exploitation_actions += num_exploited

            # Step all environments simultaneously
            next_states, rewards, dones, infos = self.vec_env.step_no_copy(actions)

            # Store experiences from all environments
            self.agent.remember_batch(states, actions, rewards, next_states, dones)

            # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
            self.agent.learn()

            # Update per-environment tracking
            env_episode_rewards += rewards
            env_episode_steps += 1
            self.total_steps += num_envs
            steps_since_report += num_envs

            # Handle completed episodes
            for i, done in enumerate(dones):
                if done:
                    # Episode completed for environment i
                    score = infos[i].get("score", 0)
                    self.scores.append(score)

                    won = infos[i].get("won", False)
                    self.wins.append(won)

                    level = infos[i].get("level", 1)
                    self.levels.append(level)

                    # Track metrics for persistence (used by save)
                    avg_loss = self.agent.get_average_loss(100)
                    q_values_arr = self.agent.get_q_values(states[i])
                    avg_q_value = float(np.mean(q_values_arr))
                    self.q_values.append(avg_q_value)
                    self.losses.append(avg_loss)
                    self.epsilons.append(self.agent.epsilon)
                    self.rewards.append(float(env_episode_rewards[i]))

                    # Track best score
                    if score > self.best_score:
                        self.best_score = score
                        self._save_model(
                            f"{self.config.GAME_NAME}_best.pth",
                            save_reason="best",
                            quiet=True,
                        )
                        if self.web_dashboard:
                            self.web_dashboard.log(
                                f"🏆 New best score: {self.best_score}", "success"
                            )

                    # Track target updates (for persistence, independent of dashboard)
                    if self.agent.steps > self.last_target_update_step + config.TARGET_UPDATE:
                        self.target_updates += 1
                        self.last_target_update_step = self.agent.steps

                    # Update web dashboard metrics (throttled to every 5 episodes for performance)
                    # Always emit on: first 10 episodes, new best score, or every 5th episode
                    is_new_best = score > self.best_score
                    dashboard = self.web_dashboard
                    if dashboard is not None and (
                        self.current_episode <= 10 or is_new_best or self.current_episode % 5 == 0
                    ):
                        bricks_broken = calculate_progress_count(infos[i], config)

                        dashboard.emit_metrics(
                            episode=self.current_episode,
                            score=score,
                            epsilon=self.agent.epsilon,
                            loss=avg_loss,
                            total_steps=self.total_steps,
                            won=won,
                            reward=float(env_episode_rewards[i]),
                            memory_size=len(self.agent.memory),
                            avg_q_value=avg_q_value,
                            exploration_actions=self.exploration_actions,
                            exploitation_actions=self.exploitation_actions,
                            target_updates=self.target_updates,
                            bricks_broken=bricks_broken,
                            episode_length=int(env_episode_steps[i]),
                        )
                        # Update performance settings in dashboard state
                        dashboard.publisher.state.learn_every = config.LEARN_EVERY
                        dashboard.publisher.state.gradient_steps = config.GRADIENT_STEPS
                        dashboard.publisher.state.batch_size = config.BATCH_SIZE

                        # Emit NN visualization data (throttled by server to ~10 FPS)
                        # Convert numpy int64 to Python int for JSON serialization
                        self._emit_nn_visualization(states[i], int(actions[i]))

                    # Store for reporting
                    last_score = score
                    last_info = infos[i]

                    # Reset per-environment tracking
                    env_episode_rewards[i] = 0.0
                    env_episode_steps[i] = 0

                    # Increment episode counter
                    episodes_completed += 1
                    self.current_episode += 1

                    # Save checkpoints (no replay buffer for periodic saves - saves disk space)
                    if self.current_episode % config.SAVE_EVERY == 0 and self.current_episode > 0:
                        self._save_model(
                            f"{self.config.GAME_NAME}_ep{self.current_episode}.pth",
                            save_reason="periodic",
                            save_replay_buffer=False,  # Periodic saves are lightweight
                        )
                        self._cleanup_old_periodic_saves(keep_last=5)

                    # Run deterministic evaluation periodically
                    if (
                        self.evaluator is not None
                        and config.EVAL_EVERY > 0
                        and self.current_episode % config.EVAL_EVERY == 0
                        and self.current_episode > 0
                    ):
                        eval_results = self.evaluator.evaluate(
                            num_episodes=config.EVAL_EPISODES,
                            max_steps=config.EVAL_MAX_STEPS,
                            episode_num=self.current_episode,
                        )
                        self.evaluator.log_results(eval_results)

                        # Auto-exploration boost: when plateau detected, increase epsilon
                        if self.evaluator.is_plateau() and not self._exploration_boost_active:
                            self._exploration_boost_active = True
                            self._exploration_boost_end_episode = (
                                self.current_episode + config.EVAL_PLATEAU_BOOST_EPISODES
                            )
                            old_epsilon = self.agent.epsilon
                            self.agent.epsilon = config.EVAL_PLATEAU_EPSILON_BOOST
                            print(
                                f"\n🚀 PLATEAU DETECTED! Boosting exploration: "
                                f"ε {old_epsilon:.3f} → {self.agent.epsilon:.3f} "
                                f"for {config.EVAL_PLATEAU_BOOST_EPISODES} episodes\n"
                            )
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"🚀 Exploration boost activated! ε → {self.agent.epsilon:.2f}",
                                    "warning",
                                )

                        # Log to web dashboard if available
                        if self.web_dashboard:
                            plateau_str = (
                                " ⚠️ PLATEAU DETECTED" if self.evaluator.is_plateau() else ""
                            )
                            self.web_dashboard.log(
                                f"📊 EVAL: {eval_results.mean_score:.0f} avg, "
                                f"max level {eval_results.max_level}, "
                                f"{eval_results.win_rate*100:.0f}% wins{plateau_str}",
                                ("info" if not self.evaluator.is_plateau() else "warning"),
                            )

            # Decay epsilon once per step if any episodes completed
            # (NOT per environment - that would decay too fast with many parallel envs)
            if np.any(dones):
                # Check if exploration boost period has ended
                if (
                    self._exploration_boost_active
                    and self.current_episode >= self._exploration_boost_end_episode
                ):
                    self._exploration_boost_active = False
                    # Reset epsilon to minimum and let it decay normally
                    self.agent.epsilon = config.EPSILON_END
                    print(
                        f"\n✓ Exploration boost ended. Resuming normal ε={self.agent.epsilon:.3f}\n"
                    )
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"✓ Exploration boost ended, ε → {self.agent.epsilon:.3f}",
                            "info",
                        )
                    # Reset plateau counter so we can detect new plateaus
                    if self.evaluator:
                        self.evaluator.evals_since_improvement = 0

                # Only decay epsilon if not in boost mode
                if not self._exploration_boost_active:
                    self.agent.decay_epsilon(self.current_episode)
                self.agent.step_scheduler()  # Step learning rate scheduler

            # Update states for next iteration (vector envs auto-reset completed games)
            states = next_states.copy()

            # Progress reporting (terminal) - only log when new episodes complete
            # Check if we should log: either LOG_EVERY episodes completed OR time interval passed
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time

            # Handle fresh start: if current_episode < last_logged_episode, a reset occurred
            if last_logged_episode > self.current_episode:
                last_logged_episode = -1  # Reset so logging can resume
                last_report_time = current_time
                steps_since_report = 0
                episodes_completed = 0  # Reset episodes count for accurate ep/hr

            should_log_by_episode = (self.current_episode - last_logged_episode) >= config.LOG_EVERY
            should_log_by_time = elapsed_since_report >= config.REPORT_INTERVAL_SECONDS

            # Only log if we have new episodes AND (LOG_EVERY condition OR time interval)
            if self.current_episode > last_logged_episode and (
                should_log_by_episode or should_log_by_time
            ):
                elapsed_total = current_time - self.training_start_time
                steps_per_sec = (
                    steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                )
                eps_per_hour = episodes_completed / elapsed_total * 3600 if elapsed_total > 0 else 0
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0
                avg_loss = self.agent.get_average_loss(100)
                avg_q = np.mean(self.q_values[-100:]) if self.q_values else 0.0

                # Get level reached from last completed episode
                level_reached = last_info.get("level", 1) if last_info else 1

                progress_msg = (
                    f"Ep {self.current_episode:5d} | "
                    f"Score: {last_score:4d} | "
                    f"Avg: {avg_score:6.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Q: {avg_q:.1f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"⚡ {steps_per_sec:,.0f} steps/s"
                )

                print(progress_msg)

                # Also log to web dashboard console
                if self.web_dashboard:
                    self.web_dashboard.log(progress_msg, "metric")

                last_logged_episode = self.current_episode
                last_report_time = current_time
                steps_since_report = 0

        # Final save
        self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="final")

        # Summary
        total_time = time.time() - self.training_start_time
        print("\n" + "=" * 70)
        print("✅ VECTORIZED TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Parallel envs:    {num_envs}")
        print(f"   Total episodes:   {self.current_episode - start_episode}")
        print(f"   Total steps:      {self.total_steps:,}")
        print(f"   Total time:       {total_time/60:.1f} minutes")
        print(f"   Avg steps/sec:    {self.total_steps/total_time:,.0f}")
        print(f"   Best score:       {self.best_score}")
        print(f"   Final avg score:  {np.mean(self.scores[-100:]):.1f}")
        recent_wins = self.wins[-100:]
        win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0
        print(f"   Win rate (100):   {win_rate*100:.1f}%")
        print("=" * 70)

        if self.web_dashboard:
            self.web_dashboard.log("✅ Vectorized training complete!", "success")

    def _save_model(
        self,
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

    def _cleanup_old_periodic_saves(self, keep_last: int = 5) -> None:
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
