"""Interactive pygame runtime for visual training and play modes."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from typing import Any, Callable, Iterable, List, Optional, cast

import numpy as np
import pygame

from config import Config
from src.ai.agent import Agent, TrainingHistory
from src.ai.evaluator import Evaluator
from src.ai.trainer import Trainer, calculate_progress_count
from src.app.game_factory import create_single_game
from src.app.lifecycle_types import EpisodeMetrics
from src.app.model_service import ModelService as AppModelService
from src.app.performance_modes import apply_performance_mode
from src.app.process_control import restart_with_game
from src.app.training_runtime import (
    build_nn_snapshot,
    emit_nn_snapshot_to_dashboard,
    request_save_and_stop,
)
from src.game import (
    BaseGame,
    ControlDisplayProvider,
    GameMenu,
    HumanActionProvider,
    HumanStepProvider,
    get_game_info,
    list_games,
)
from src.utils.checkpoint_loader import load_checkpoint
from src.visualizer.dashboard import Dashboard
from src.visualizer.hud import TrainingHUD
from src.visualizer.pause_menu import PauseMenu

WEB_AVAILABLE: bool
WebDashboard: Optional[type[Any]]
try:
    from src.web import WebDashboard as _WebDashboard

    WebDashboard = _WebDashboard
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    WebDashboard = None


class GameApp:
    """
    Main application that runs the game with AI training and visualization.

    This class manages:
        - Pygame window and rendering
        - Game instance
        - AI agent and training
        - Visualizations (neural network + dashboard)
        - User input handling
    """

    @staticmethod
    def _average_recent_scores(scores: Iterable[int], limit: int = 100) -> float:
        """Average the most recent scores from any iterable score history."""
        recent_scores = list(scores)[-limit:]
        return float(np.mean(recent_scores)) if recent_scores else 0.0

    def __init__(
        self,
        config: Config,
        args: argparse.Namespace,
        existing_dashboard: Optional[Any] = None,
    ):
        """
        Initialize the application.

        Args:
            config: Configuration object
            args: Command line arguments
            existing_dashboard: Optional pre-initialized WebDashboard instance
        """
        self.config = config
        self.args = args
        self.model_service = AppModelService(config)

        # Update config from args
        if args.lr:
            config.LEARNING_RATE = args.lr
        if args.episodes:
            config.MAX_EPISODES = args.episodes

        # Initialize Pygame
        pygame.init()

        # Cache fonts to avoid recreating them every frame
        self._pause_font = pygame.font.Font(None, 72)
        self._speed_font = pygame.font.Font(None, 24)
        self._help_font = pygame.font.Font(None, 28)
        self._help_title_font = pygame.font.Font(None, 36)
        self._notification_font = pygame.font.Font(None, 28)

        # Notification system for visual feedback
        self._notifications: List[dict] = []  # [{text, color, start_time, duration}]

        # Get game info for display
        game_info = get_game_info(config.GAME_NAME)
        game_display_name = (
            game_info.get("name", config.GAME_NAME.title())
            if game_info
            else config.GAME_NAME.title()
        )
        pygame.display.set_caption(f"🧠 Neural Network AI - {game_display_name}")

        # Calculate window size
        # Layout: Game only - all training info is on the web dashboard
        self.game_width = config.SCREEN_WIDTH  # 800 - for reference
        self.game_height = config.SCREEN_HEIGHT  # 600 - for reference

        # Window size: game only (training stats and NN viz are on web dashboard)
        self.window_width = config.SCREEN_WIDTH + 20
        self.window_height = config.SCREEN_HEIGHT + 20

        # Minimum window dimensions
        self.min_window_width = config.SCREEN_WIDTH + 10
        self.min_window_height = config.SCREEN_HEIGHT + 10

        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        # Create a fixed-size surface for game rendering (will be scaled to window)
        self.game_surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

        # Scaling state
        self.scale_factor = 1.0
        self.game_offset_x = 0
        self.game_offset_y = 0
        self._update_scale()

        # Create game from registry
        try:
            game_environment = create_single_game(config.GAME_NAME, config)
        except ValueError:
            print(f"❌ Unknown game: {config.GAME_NAME}")
            print(f"   Available games: {', '.join(list_games())}")
            sys.exit(1)
        self.game: BaseGame = game_environment.game

        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config,
        )

        # Load model if specified, or auto-load most recent save
        self._initial_model_path = self._resolve_model_path(
            args.model,
            state_size=self.game.state_size,
            action_size=self.game.action_size,
        )

        # Create dashboard for internal tracking (rendering moved to web dashboard)
        self.dashboard = Dashboard(
            config=config,
            x=0,
            y=0,
            width=400,
            height=100,  # Position doesn't matter, not rendered
        )

        # Create HUD and PauseMenu
        self.hud = TrainingHUD(config=config)
        self.pause_menu = PauseMenu(
            screen_width=config.SCREEN_WIDTH, screen_height=config.SCREEN_HEIGHT
        )

        # Training state
        self.episode = 0
        self.total_reward = 0.0
        self.steps = 0
        self.paused = False
        self.running = True
        self.return_to_menu = False  # Flag to return to game selector
        self.game_speed = 1.0  # Speed multiplier
        self.show_help_legend = False  # Toggle with H key

        # Extended training metrics
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.episode_start_time = time.time()
        self.target_updates = 0
        self.last_target_update_step = 0

        # Training start time for total training time tracking
        self.training_start_time = time.time()

        # Track recent scores for save metadata (bounded deque)
        self.recent_scores: deque[int] = deque(maxlen=1000)
        self.best_score_ever = 0

        # Full training history for save/restore (allows dashboard restoration)
        self.episode_history: deque[EpisodeMetrics] = deque(maxlen=100000)

        # Current state
        self.state = self.game.reset()
        self.selected_action: Optional[int] = None

        # FPS for rendering
        self.render_fps = 60
        self.train_fps = 0  # Unlimited for headless

        # Frame counter for screenshot timing (independent of training steps)
        self.frame_count = 0

        # Web dashboard (if enabled)
        self.web_dashboard: Optional[Any] = None
        if existing_dashboard:
            # Use pre-initialized dashboard from main()
            self.web_dashboard = existing_dashboard
            self.web_dashboard.launcher_mode = False  # Switch out of launcher mode

            # Setup callbacks
            self.web_dashboard.on_pause_callback = self._toggle_pause
            self.web_dashboard.on_save_callback = lambda: self._save_model(
                f"{config.GAME_NAME}_web_save.pth", save_reason="manual"
            )
            self.web_dashboard.on_save_as_callback = self._save_model_as
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.on_reset_callback = self._reset_episode
            self.web_dashboard.on_start_fresh_callback = self._start_fresh
            self.web_dashboard.on_load_model_callback = self._load_model
            self.web_dashboard.on_config_change_callback = self._apply_config
            self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
            self.web_dashboard.on_restart_with_game_callback = lambda game: restart_with_game(
                game, args
            )
            self.web_dashboard.on_save_and_quit_callback = self._save_and_quit

            # Send system info to dashboard
            self._send_system_info()

            # Log startup info
            self._log_startup_info()
        elif hasattr(args, "web") and args.web and WEB_AVAILABLE and WebDashboard is not None:
            # Create new dashboard
            self.web_dashboard = WebDashboard(
                config, port=args.port, host=getattr(args, "host", "127.0.0.1")
            )
            self.web_dashboard.on_pause_callback = self._toggle_pause
            self.web_dashboard.on_save_callback = lambda: self._save_model(
                f"{config.GAME_NAME}_web_save.pth", save_reason="manual"
            )
            self.web_dashboard.on_save_as_callback = self._save_model_as
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.on_reset_callback = self._reset_episode
            self.web_dashboard.on_start_fresh_callback = self._start_fresh
            self.web_dashboard.on_load_model_callback = self._load_model
            self.web_dashboard.on_config_change_callback = self._apply_config
            self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
            self.web_dashboard.on_restart_with_game_callback = lambda game: restart_with_game(
                game, args
            )
            self.web_dashboard.on_save_and_quit_callback = self._save_and_quit
            self.web_dashboard.start()

            # Show URL prominently
            print("\n" + "=" * 60)
            print(f"🌐 WEB DASHBOARD: {self.web_dashboard.dashboard_url()}")
            print("=" * 60 + "\n")

            # Send system info to dashboard
            self._send_system_info()

            # Log startup info
            self._log_startup_info()

        # Load the initial model (now that dashboard is ready)
        if self._initial_model_path:
            self._load_model(self._initial_model_path)
            if self.web_dashboard:
                self.web_dashboard.log(
                    f"📂 Auto-loaded: {os.path.basename(self._initial_model_path)}",
                    "success",
                )

    def _resolve_model_path(
        self, explicit_path: Optional[str], state_size: int, action_size: int
    ) -> Optional[str]:
        """Resolve which model to load on startup."""
        return self.model_service.resolve_model_path(
            explicit_path,
            state_size=state_size,
            action_size=action_size,
        )

    def _restore_training_history(self, filepath: str) -> None:
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

    def _log_startup_info(self) -> None:
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

    def _toggle_pause(self) -> None:
        """Toggle pause state (for web dashboard control)."""
        self.paused = not self.paused
        # Bug 85: Reset pause menu state when entering pause to clear stale dialogs
        if self.paused and hasattr(self, "pause_menu") and self.pause_menu:
            self.pause_menu.reset_state()
        if self.web_dashboard:
            self.web_dashboard.publisher.set_paused(self.paused)
            status = "⏸️ Training paused" if self.paused else "▶️ Training resumed"
            self.web_dashboard.log(status, "action")
        print("⏸️  Paused" if self.paused else "▶️  Resumed")
        # Bug 83: Show on-screen notification for pause/resume
        if hasattr(self, "_show_notification"):
            msg = "Paused" if self.paused else "Resumed"
            color = (255, 200, 100) if self.paused else (100, 255, 100)
            self._show_notification(msg, color, 1.0)

    def _reset_episode(self) -> None:
        """Reset the current episode."""
        self.state = self.game.reset()
        self.total_reward = 0.0
        if self.web_dashboard:
            self.web_dashboard.log("🔄 Episode reset", "action")
        print("🔄 Episode reset")

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
        self.selected_action = None

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

    def _load_model(self, filepath: str) -> None:
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

    def _apply_config(self, config_data: dict) -> None:
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

    def _send_system_info(self) -> None:
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

    def _set_performance_mode(self, mode: str) -> None:
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

    def _update_layout(self, new_width: int, new_height: int) -> None:
        """Update component positions based on new window size."""
        # Enforce minimum window size
        new_width = max(new_width, self.min_window_width)
        new_height = max(new_height, self.min_window_height)

        self.window_width = new_width
        self.window_height = new_height

        # The game has a FIXED render size (config.SCREEN_WIDTH x config.SCREEN_HEIGHT)
        # We position other elements around it
        # Game display area stays fixed (training stats and NN viz are on web dashboard)
        self.game_width = self.config.SCREEN_WIDTH  # 800
        self.game_height = self.config.SCREEN_HEIGHT  # 600

        # Update scaling for the new window size
        self._update_scale()

        # Update pause menu positions
        if hasattr(self, "pause_menu") and self.pause_menu:
            self.pause_menu.handle_resize(new_width, new_height)

    def _update_scale(self) -> None:
        """Calculate scaling factor to fit game in window while maintaining aspect ratio."""
        # Guard against zero dimensions during window minimize/restore
        if self.window_width <= 0 or self.window_height <= 0:
            return

        game_aspect = self.config.SCREEN_WIDTH / self.config.SCREEN_HEIGHT
        window_aspect = self.window_width / self.window_height

        if window_aspect > game_aspect:
            # Window is wider than game - scale by height
            self.scale_factor = self.window_height / self.config.SCREEN_HEIGHT
            scaled_width = int(self.config.SCREEN_WIDTH * self.scale_factor)
            self.game_offset_x = (self.window_width - scaled_width) // 2
            self.game_offset_y = 0
        else:
            # Window is taller than game - scale by width
            self.scale_factor = self.window_width / self.config.SCREEN_WIDTH
            scaled_height = int(self.config.SCREEN_HEIGHT * self.scale_factor)
            self.game_offset_x = 0
            self.game_offset_y = (self.window_height - scaled_height) // 2

        # Ensure minimum scale factor to prevent division by zero or rendering issues
        self.scale_factor = max(0.1, self.scale_factor)

    def _show_notification(
        self, text: str, color: tuple = (100, 200, 255), duration: float = 2.0
    ) -> None:
        """
        Show a notification on screen.

        Args:
            text: Notification text
            color: Text color (RGB tuple)
            duration: How long to show the notification in seconds
        """
        import time

        self._notifications.append(
            {
                "text": text,
                "color": color,
                "start_time": time.time(),
                "duration": duration,
            }
        )

    def _update_notifications(self) -> None:
        """Remove expired notifications."""
        import time

        current_time = time.time()
        self._notifications = [
            n for n in self._notifications if current_time - n["start_time"] < n["duration"]
        ]

    def _render_notifications(self, surface: pygame.Surface) -> None:
        """Render all active notifications."""
        import time

        if not self._notifications:
            return

        current_time = time.time()
        y_offset = 10

        for notification in self._notifications:
            elapsed = current_time - notification["start_time"]
            # Fade out in the last 0.5 seconds
            alpha = 255
            if elapsed > notification["duration"] - 0.5:
                alpha = int(255 * (notification["duration"] - elapsed) / 0.5)
            alpha = max(0, min(255, alpha))

            text_surface = self._notification_font.render(
                notification["text"], True, notification["color"]
            )

            # Create background with alpha
            bg_width = text_surface.get_width() + 20
            bg_height = text_surface.get_height() + 10
            bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
            pygame.draw.rect(
                bg_surface,
                (0, 0, 0, int(alpha * 0.7)),
                bg_surface.get_rect(),
                border_radius=5,
            )

            # Position at top-center
            x = (surface.get_width() - bg_width) // 2
            y = y_offset

            # Apply alpha to text
            text_surface.set_alpha(alpha)

            surface.blit(bg_surface, (x, y))
            surface.blit(text_surface, (x + 10, y + 5))

            y_offset += bg_height + 5

    # Speed presets for clean stepping
    # Smoother geometric progression for speed control
    SPEED_PRESETS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    _last_logged_speed: float = 0.0  # Track last logged speed to avoid spam

    def _set_speed(self, speed: float, force_log: bool = False) -> None:
        """Set game speed (for web dashboard control)."""
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            print(f"⚠️  Invalid speed value: {speed}")
            return
        if not math.isfinite(speed):
            print(f"⚠️  Invalid speed value: {speed}")
            return
        new_speed = max(1.0, min(1000.0, speed))
        old_speed = self.game_speed
        self.game_speed = new_speed

        if self.web_dashboard:
            self.web_dashboard.publisher.set_speed(self.game_speed)

        # Only log if speed changed significantly (avoid spam when dragging slider)
        # Log when: forced, or speed is a preset value, or changed by >10%
        speed_changed_significantly = (
            abs(new_speed - self._last_logged_speed) / max(1, self._last_logged_speed) > 0.1
        )
        is_preset = int(new_speed) in self.SPEED_PRESETS

        if force_log or (speed_changed_significantly and is_preset):
            if self.web_dashboard:
                self.web_dashboard.log(f"⏩ Speed set to {int(self.game_speed)}x", "action")
            print(f"⏩ Speed: {int(self.game_speed)}x")
            self._last_logged_speed = new_speed

    def _speed_up(self) -> None:
        """Increase speed to next preset."""
        for preset in self.SPEED_PRESETS:
            if preset > self.game_speed + 0.01:  # Epsilon comparison for float precision
                self._set_speed(preset, force_log=True)
                return
        # Already at max
        self._set_speed(self.SPEED_PRESETS[-1], force_log=True)

    def _speed_down(self) -> None:
        """Decrease speed to previous preset."""
        for preset in reversed(self.SPEED_PRESETS):
            if preset < self.game_speed - 0.01:  # Epsilon comparison for float precision
                self._set_speed(preset, force_log=True)
                return
        # Already at min
        self._set_speed(self.SPEED_PRESETS[0], force_log=True)

    def run_human_mode(self) -> None:
        """Run in human play mode for testing the game."""
        print("\n" + "=" * 60)
        print("  HUMAN PLAY MODE")
        print("=" * 60)

        # Show game-specific controls
        game_name = self.config.GAME_NAME
        game_info = get_game_info(game_name) or {}
        for control in game_info.get("controls", ["Use arrow keys to control"]):
            print(f"   {control}")

        print("\n   R: Reset game")
        print("   Q/ESC: Quit")
        print("   F: Toggle fullscreen")
        print("=" * 60 + "\n")

        # Enable controls display for games that support it
        if hasattr(self.game, "show_controls"):
            cast(ControlDisplayProvider, self.game).show_controls = True

        state = self.game.reset()

        while self.running:
            self._handle_events()

            # Get keyboard input as dict for games that use get_human_action
            pressed = pygame.key.get_pressed()
            keys_dict = {key: pressed[key] for key in range(len(pressed))}
            human_action_provider = (
                cast(HumanActionProvider, self.game)
                if hasattr(self.game, "get_human_action")
                else None
            )
            human_step_provider = (
                cast(HumanStepProvider, self.game) if hasattr(self.game, "step_human") else None
            )

            # Handle game-specific controls
            if game_name == "asteroids" and human_step_provider and human_action_provider:
                # Asteroids supports simultaneous actions via step_human
                state, reward, done, info = human_step_provider.step_human(keys_dict)
                action = human_action_provider.get_human_action(keys_dict)  # For display purposes
            elif human_action_provider:
                # Use game's built-in human action helper
                action = human_action_provider.get_human_action(keys_dict)
                state, reward, done, info = self.game.step(action)
            else:
                # Fallback: generic control mapping
                action = 1  # STAY / IDLE
                if pressed[pygame.K_LEFT]:
                    action = 0  # LEFT
                elif pressed[pygame.K_RIGHT]:
                    action = 2  # RIGHT
                elif pressed[pygame.K_SPACE] and game_name == "space_invaders":
                    action = 3  # SHOOT
                state, reward, done, info = self.game.step(action)

            if done:
                score = info.get("score", 0)
                print(f"   Game Over! Score: {score}")
                pygame.time.wait(1500)  # Brief pause before reset
                state = self.game.reset()

            # Render
            self._render_frame(state, action, info)
            self.clock.tick(self.config.FPS)

        pygame.quit()

    def run_play_mode(self) -> None:
        """Run trained agent without training (demonstration mode)."""
        print("\n🤖 AI Play Mode (No Training)")
        print("   Watching trained agent play...")
        print("   Press Q to quit\n")

        self.agent.epsilon = 0  # No exploration
        state = self.game.reset()
        episode_reward = 0.0
        total_bricks = self.config.BRICK_ROWS * self.config.BRICK_COLS
        info: dict = {
            "score": 0,
            "bricks_remaining": total_bricks,
        }  # Default info for paused state

        while self.running:
            self._handle_events()

            if not self.paused:
                # Agent selects action
                action = self.agent.select_action(state, training=False)

                # Step game
                state, reward, done, info = self.game.step(action)
                episode_reward += float(reward)

                if done:
                    print(
                        f"   Episode complete! Score: {info['score']}, Reward: {episode_reward:.1f}"
                    )
                    state = self.game.reset()
                    episode_reward = 0.0
                    self.episode += 1
                    self.dashboard.update(
                        self.episode,
                        info["score"],
                        0,
                        0,
                        bricks_broken=total_bricks - info.get("bricks_remaining", total_bricks),
                    )

                self.selected_action = action

            # Render
            self._render_frame(
                state,
                self.selected_action if self.selected_action is not None else 1,
                {"score": info.get("score", 0)},
            )
            self.clock.tick(int(self.config.FPS * self.game_speed))

        pygame.quit()

    def run_training(self) -> None:
        """Run training loop with visualization."""
        print("\n" + "=" * 60)
        print("🧠 Starting AI Training with Live Visualization")
        print("=" * 60)
        eps_str = "Unlimited" if self.config.MAX_EPISODES == 0 else str(self.config.MAX_EPISODES)
        print(f"   Episodes:       {eps_str}")
        print(f"   Learning Rate:  {self.config.LEARNING_RATE}")
        print(f"   Device:         {self.config.DEVICE}")
        print(f"   Batch Size:     {self.config.BATCH_SIZE}")
        print(f"   Learn Every:    {self.config.LEARN_EVERY} steps")
        print(f"   Gradient Steps: {self.config.GRADIENT_STEPS}")
        print(f"\n   Controls:")
        print(f"   - P: Pause/Resume")
        print(f"   - S: Save model")
        print(f"   - +/-: Speed up/down")
        print(f"   - F: Toggle fullscreen")
        print(f"   - Q/ESC: Quit")
        print("=" * 60 + "\n")

        state = self.game.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_start_time = time.time()
        episode_bricks_broken = 0
        total_bricks = self.config.BRICK_ROWS * self.config.BRICK_COLS
        info: dict = {}

        # Speed slider directly controls training intensity
        # At 1x: 1 training step per render (real-time, ~60 steps/sec)
        # At 100x: 100 training steps per render (~6000 steps/sec)
        # At 1000x: 1000 training steps per render (~60000 steps/sec!)

        # Track step time for performance logging (bounded deque)
        avg_step_time = 0.001
        step_time_samples: deque[float] = deque(maxlen=100)

        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        while self.running and (
            self.config.MAX_EPISODES == 0 or self.episode < self.config.MAX_EPISODES
        ):
            frame_start = time.time()
            self._handle_events()

            if not self.paused:
                # Speed directly controls steps per render cycle
                # At 1x: 1 step then render
                # At 10x: 10 steps then render
                # At 100x: 100 steps then render
                # At 1000x: 1000 steps then render (blazing fast training!)
                steps_per_frame = max(1, int(self.game_speed))

                steps_this_frame = 0
                training_start = time.time()

                # Run the calculated number of training steps
                while steps_this_frame < steps_per_frame:

                    if self.paused or not self.running:
                        break

                    # Agent selects action
                    action = self.agent.select_action(state, training=True)
                    self.selected_action = action

                    # Track exploration vs exploitation (using actual selection, not separate RNG)
                    if self.agent._last_action_explored:
                        self.exploration_actions += 1
                    else:
                        self.exploitation_actions += 1

                    # Execute action
                    next_state, reward, done, info = self.game.step(action)

                    # Track bricks broken this episode
                    if reward > 0.5:  # Brick hit reward
                        episode_bricks_broken += 1

                    # Store experience
                    self.agent.remember(state, action, reward, next_state, done)

                    # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
                    loss = self.agent.learn()

                    # Track target network updates
                    if (
                        self.agent.steps % self.config.TARGET_UPDATE == 0
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
                    self.steps += 1
                    steps_this_frame += 1

                    if done:
                        # Episode complete
                        episode_duration = time.time() - episode_start_time

                        # Decay epsilon (pass episode for warmup check)
                        old_epsilon = self.agent.epsilon
                        self.agent.decay_epsilon(self.episode)
                        self.agent.step_scheduler()  # Step learning rate scheduler

                        # Calculate average Q-value for current state
                        q_values = self.agent.get_q_values(state)
                        avg_q_value = float(np.mean(q_values))

                        # Update dashboard
                        self.dashboard.update(
                            self.episode,
                            info["score"],
                            self.agent.epsilon,
                            self.agent.get_average_loss(100),
                            bricks_broken=total_bricks - info.get("bricks_remaining", total_bricks),
                            won=info.get("won", False),
                            reward=episode_reward,
                        )

                        # Update web dashboard if enabled (throttled to every 5 episodes for performance)
                        # Always emit on: first 10 episodes, new best score, or every 5th episode
                        is_new_best = info["score"] > self.best_score_ever
                        dashboard = self.web_dashboard
                        if dashboard is not None and (
                            self.episode <= 10 or is_new_best or self.episode % 5 == 0
                        ):
                            dashboard.emit_metrics(
                                episode=self.episode,
                                score=info["score"],
                                epsilon=self.agent.epsilon,
                                loss=self.agent.get_average_loss(100),
                                total_steps=self.steps,
                                won=info.get("won", False),
                                reward=episode_reward,
                                memory_size=len(self.agent.memory),
                                avg_q_value=avg_q_value,
                                exploration_actions=self.exploration_actions,
                                exploitation_actions=self.exploitation_actions,
                                target_updates=self.target_updates,
                                bricks_broken=episode_bricks_broken,
                                episode_length=episode_steps,
                            )
                            # Update performance settings in dashboard state
                            dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
                            dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS
                            dashboard.publisher.state.batch_size = self.config.BATCH_SIZE

                            # Log episode completion
                            self._log_episode_complete(
                                info,
                                episode_reward,
                                episode_steps,
                                episode_duration,
                                episode_bricks_broken,
                                avg_q_value,
                                old_epsilon,
                            )

                        # Terminal log
                        if self.episode % self.config.LOG_EVERY == 0:
                            avg_score = (
                                np.mean(list(self.dashboard.scores)[-100:])
                                if self.dashboard.scores
                                else 0
                            )
                            avg_loss = self.agent.get_average_loss(100)
                            print(
                                f"Episode {self.episode:5d} | "
                                f"Score: {info['score']:4d} | "
                                f"Avg: {avg_score:6.1f} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"Q: {avg_q_value:.1f} | "
                                f"ε: {self.agent.epsilon:.3f}"
                            )

                        # Track scores for metadata (deque auto-trims to maxlen=1000)
                        self.recent_scores.append(info["score"])

                        # Track full training history for save/restore
                        self.episode_history.append(
                            EpisodeMetrics(
                                score=info["score"],
                                reward=episode_reward,
                                steps=episode_steps,
                                epsilon=self.agent.epsilon,
                                bricks_hit=episode_bricks_broken,
                                won=info.get("won", False),
                            )
                        )

                        # Save checkpoint (no replay buffer for periodic - saves disk space)
                        if self.episode % self.config.SAVE_EVERY == 0 and self.episode > 0:
                            self._save_model(
                                f"{self.config.GAME_NAME}_ep{self.episode}.pth",
                                save_reason="periodic",
                            )
                            self._cleanup_old_periodic_saves(keep_last=5)
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"💾 Checkpoint saved: {self.config.GAME_NAME}_ep{self.episode}.pth",
                                    "success",
                                )

                        if info["score"] > self.best_score_ever:
                            self.best_score_ever = info["score"]
                            self._save_model(
                                f"{self.config.GAME_NAME}_best.pth",
                                save_reason="best",
                                quiet=False,
                            )
                            # Bug 97: Show on-screen notification for new best score
                            self._show_notification(
                                f"New Best: {info['score']}!", (255, 215, 0), 2.0
                            )
                            if self.web_dashboard:
                                avg_score = self._average_recent_scores(self.recent_scores)
                                self.web_dashboard.log(
                                    f"🏆 New best score: {info['score']}! Model saved.",
                                    "success",
                                    {
                                        "score": info["score"],
                                        "episode": self.episode,
                                        "avg_100": round(avg_score, 1),
                                    },
                                )

                        # Reset for next episode
                        self.episode += 1
                        episode_reward = 0.0
                        episode_steps = 0
                        episode_bricks_broken = 0
                        episode_start_time = time.time()
                        state = self.game.reset()

                        # Check if we should stop training
                        if self.episode >= self.config.MAX_EPISODES:
                            break

                # Update average step time for dynamic adjustment
                if steps_this_frame > 0:
                    frame_training_time = time.time() - training_start
                    measured_step_time = frame_training_time / steps_this_frame
                    step_time_samples.append(measured_step_time)  # deque auto-trims to maxlen=100
                    avg_step_time = sum(step_time_samples) / len(step_time_samples)

                    # Log performance occasionally
                    if self.steps % 500 == 0 and self.web_dashboard:
                        frame_time = time.time() - frame_start
                        steps_per_sec = steps_this_frame / frame_time if frame_time > 0 else 0
                        self.web_dashboard.log(
                            f"⚡ {int(self.game_speed)}x: {steps_this_frame} steps/render, {steps_per_sec:.0f} steps/sec",
                            "debug",
                        )

            # Render the current state
            if not self.args.headless and not self.paused:
                render_action = self.selected_action if self.selected_action is not None else 1
                self._render_frame(state, render_action, info if info else {})

                # Cap frame rate to prevent GPU spikes
                # At 1x: 60 FPS for smooth real-time feel
                # At higher speeds: 120 FPS max to prevent GPU thrashing
                target_fps = 60 if self.game_speed <= 1 else 120
                self.clock.tick(target_fps)

        # Training complete
        self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="final")
        if self.web_dashboard:
            self.web_dashboard.log(
                "🎉 Training complete!",
                "success",
                {
                    "total_episodes": self.episode,
                    "best_score": self.best_score_ever,
                    "total_steps": self.steps,
                },
            )

        print("\n✅ Training complete!")
        print(f"   Total episodes: {self.episode}")
        print(f"   Best score: {self.best_score_ever}")

        pygame.quit()

    def _log_episode_complete(
        self,
        info: dict,
        episode_reward: float,
        episode_steps: int,
        episode_duration: float,
        bricks_broken: int,
        avg_q_value: float,
        old_epsilon: float,
    ) -> None:
        """Log detailed episode completion to web dashboard."""
        if not self.web_dashboard:
            return

        # Determine episode outcome
        won = info.get("won", False)
        score = info["score"]

        # Log episode metrics
        level = "success" if won else "metric"
        outcome = "🏆 WIN!" if won else ""

        self.web_dashboard.log(
            f"Episode {self.episode} complete {outcome}",
            level,
            {
                "episode": self.episode,
                "score": score,
                "reward": round(episode_reward, 2),
                "steps": episode_steps,
                "duration": round(episode_duration, 2),
                "bricks": bricks_broken,
                "won": won,
            },
        )

        # Log detailed metrics every few episodes
        if self.episode % 5 == 0:
            loss = self.agent.get_average_loss(100)
            explore_ratio = self.exploration_actions / max(
                1, self.exploration_actions + self.exploitation_actions
            )

            self.web_dashboard.log(
                f"Training metrics: Loss={loss:.4f}, Q={avg_q_value:.2f}, Explore={explore_ratio:.1%}",
                "metric",
                {
                    "loss": round(loss, 6),
                    "avg_q_value": round(avg_q_value, 4),
                    "explore_ratio": round(explore_ratio, 4),
                    "memory_size": len(self.agent.memory),
                    "total_steps": self.steps,
                },
            )

        # Log epsilon decay
        if abs(old_epsilon - self.agent.epsilon) > 0.001:
            self.web_dashboard.log(
                f"Epsilon decayed: {old_epsilon:.4f} → {self.agent.epsilon:.4f}",
                "debug",
                {"old": old_epsilon, "new": self.agent.epsilon},
            )

    def run_headless_training(self) -> None:
        """Run training without visualization (faster). Use HeadlessTrainer for max speed."""
        print("\n" + "=" * 60)
        print("🚀 Starting Headless Training (No Visualization)")
        print("=" * 60)
        eps_str = "Unlimited" if self.config.MAX_EPISODES == 0 else str(self.config.MAX_EPISODES)
        print(f"   Episodes:       {eps_str}")
        print(f"   Device:         {self.config.DEVICE}")
        print(f"   Batch Size:     {self.config.BATCH_SIZE}")
        print(f"   Learn Every:    {self.config.LEARN_EVERY} steps")
        print(f"   Gradient Steps: {self.config.GRADIENT_STEPS}")
        print("=" * 60 + "\n")

        state = self.game.reset()
        episode_reward = 0.0
        episode_steps = 0
        total_steps = 0

        scores: list[int] = []
        best_score = 0
        start_time = time.time()
        last_report_time = start_time
        steps_since_report = 0

        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        episode = 0
        while self.config.MAX_EPISODES == 0 or episode < self.config.MAX_EPISODES:
            done = False
            state = self.game.reset()
            episode_reward = 0.0
            episode_steps = 0
            info = {"score": 0, "won": False}

            while not done and episode_steps < self.config.MAX_STEPS_PER_EPISODE:
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.game.step(action)

                self.agent.remember(state, action, reward, next_state, done)

                # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
                self.agent.learn()

                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                steps_since_report += 1

            self.agent.decay_epsilon(episode)
            self.agent.step_scheduler()  # Step learning rate scheduler
            scores.append(info["score"])

            # Time-based progress reporting
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time

            if (
                elapsed_since_report >= self.config.REPORT_INTERVAL_SECONDS
                or episode % self.config.LOG_EVERY == 0
            ):
                elapsed_total = current_time - start_time
                steps_per_sec = (
                    steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                )
                avg_score = np.mean(scores[-100:]) if scores else 0
                avg_loss = self.agent.get_average_loss(100)
                q_values = self.agent.get_q_values(state)
                avg_q = float(np.mean(q_values))

                print(
                    f"Episode {episode:5d} | "
                    f"Score: {info['score']:4d} | "
                    f"Avg: {avg_score:6.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Q: {avg_q:.1f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"⚡ {steps_per_sec:,.0f} steps/s"
                )

                last_report_time = current_time
                steps_since_report = 0

            # Save checkpoints
            if episode % self.config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(f"{self.config.GAME_NAME}_ep{episode}.pth")

            if info["score"] > best_score:
                best_score = info["score"]
                self._save_model(f"{self.config.GAME_NAME}_best.pth", quiet=True)

            # Increment episode counter
            episode += 1

        self._save_model(f"{self.config.GAME_NAME}_final.pth")

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"   Total time:      {total_time/60:.1f} minutes")
        print(f"   Total steps:     {total_steps:,}")
        print(f"   Avg steps/sec:   {total_steps/total_time:,.0f}")
        print(f"   Final avg score: {np.mean(scores[-100:]):.1f}")
        print(f"   Best score:      {max(scores) if scores else 0}")
        print("=" * 60)

    def _handle_events(self) -> None:
        """Handle pygame events and keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                new_width = max(event.w, self.min_window_width)
                new_height = max(event.h, self.min_window_height)
                self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                self._update_layout(new_width, new_height)

            # Handle pause menu interactions when paused
            if self.paused:
                action = self.pause_menu.handle_event(event)
                if action == "resume":
                    self._toggle_pause()
                elif action == "save":
                    self._save_model(
                        f"{self.config.GAME_NAME}_manual_save.pth", save_reason="manual"
                    )
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"💾 Manual save: {self.config.GAME_NAME}_manual_save.pth",
                            "success",
                        )
                elif action == "menu":
                    # Save current progress before returning to menu
                    self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="menu_exit")
                    if self.web_dashboard:
                        self.web_dashboard.log("🏠 Returning to game selector...", "warning")
                        self.web_dashboard.launcher_mode = True
                        self.web_dashboard.socketio.emit(
                            "redirect_to_launcher",
                            {"message": "Returning to game selector..."},
                        )
                    self.return_to_menu = True
                    self.running = False
                elif action == "quit":
                    self.running = False
                continue  # Skip other event handling when paused

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False

                elif event.key == pygame.K_p:
                    self._toggle_pause()

                elif event.key == pygame.K_s:
                    success = self._save_model(
                        f"{self.config.GAME_NAME}_manual_save.pth", save_reason="manual"
                    )
                    if success:
                        self._show_notification("Model Saved", (100, 255, 100), 1.5)
                    else:
                        self._show_notification("Save Failed", (255, 100, 100), 2.0)
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"💾 Manual save: {self.config.GAME_NAME}_manual_save.pth",
                            "success",
                        )

                elif event.key == pygame.K_r:
                    self._reset_episode()
                    self._show_notification("Episode Reset", (255, 200, 100), 1.0)

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._speed_up()
                    self._show_notification(f"Speed: {self.game_speed:.0f}x", (100, 200, 255), 1.0)

                elif event.key == pygame.K_MINUS:
                    self._speed_down()
                    self._show_notification(f"Speed: {self.game_speed:.0f}x", (100, 200, 255), 1.0)

                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    if self.screen.get_flags() & pygame.FULLSCREEN:
                        self.screen = pygame.display.set_mode(
                            (self.window_width, self.window_height), pygame.RESIZABLE
                        )
                        # Bug 95: Show notification for fullscreen toggle
                        self._show_notification("Windowed", (100, 200, 255), 1.0)
                    else:
                        self.screen = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN | pygame.RESIZABLE
                        )
                        # Update layout to new screen size
                        display_info = pygame.display.Info()
                        self._update_layout(display_info.current_w, display_info.current_h)
                        # Bug 95: Show notification for fullscreen toggle
                        self._show_notification("Fullscreen", (100, 200, 255), 1.0)

                elif event.key == pygame.K_h:
                    # Toggle help legend
                    self.show_help_legend = not self.show_help_legend
                    # Bug 96: Show notification for help legend toggle
                    msg = "Help: ON" if self.show_help_legend else "Help: OFF"
                    self._show_notification(msg, (150, 150, 200), 1.0)

    def _render_frame(self, state: np.ndarray, action: int, info: dict) -> None:
        """Render one frame of the visualization with proper scaling."""
        # Clear the game surface (fixed size)
        self.game_surface.fill((10, 10, 15))

        # Render game to the fixed-size game surface
        self.game.render(self.game_surface)

        # Render HUD (training stats overlay) if not paused
        if not self.paused:
            # Get action labels from game
            action_labels = []
            if hasattr(self.game, "get_action_labels"):
                action_labels = self.game.get_action_labels()
            else:
                # Fallback generic labels
                action_labels = [f"Action {i}" for i in range(self.game.action_size)]

            self.hud.render(
                surface=self.game_surface,
                episode=self.episode,
                score=info.get("score", 0),
                best_score=self.best_score_ever,
                epsilon=self.agent.epsilon if hasattr(self, "agent") else 0.0,
                loss=0.0,  # Could track recent loss
                speed=self.game_speed,
                max_episodes=self.config.MAX_EPISODES,
                selected_action=action,
                action_labels=action_labels,
            )

        # Render pause menu with context (centered on the game surface)
        if self.paused:
            # Build context for pause menu
            pause_context = {
                "episode": self.episode,
                "score": info.get("score", 0),
                "best_score": self.best_score_ever,
                "epsilon": self.agent.epsilon if hasattr(self, "agent") else 0.0,
                "training_time": time.time() - self.training_start_time,
                "memory_size": len(self.agent.memory) if hasattr(self, "agent") else 0,
                "memory_capacity": (
                    self.config.MEMORY_SIZE if hasattr(self.config, "MEMORY_SIZE") else 0
                ),
            }
            self.pause_menu.render(self.game_surface, pause_context)

        # Render help legend (bottom left of game area, toggle with H)
        if self.show_help_legend:
            controls = [
                ("P", "Pause/Resume"),
                ("S", "Save Model"),
                ("R", "Reset Episode"),
                ("+/-", "Speed Up/Down"),
                ("F", "Fullscreen"),
                ("H", "Hide Help"),
                ("ESC", "Quit"),
            ]
            padding = 15
            line_height = 24
            legend_width = 180
            legend_height = len(controls) * line_height + padding * 2 + 30

            # Position at bottom-left
            legend_x = 10
            legend_y = self.config.SCREEN_HEIGHT - legend_height - 10

            # Draw background
            legend_bg = pygame.Surface((legend_width, legend_height), pygame.SRCALPHA)
            pygame.draw.rect(legend_bg, (0, 0, 0, 200), legend_bg.get_rect(), border_radius=8)
            self.game_surface.blit(legend_bg, (legend_x, legend_y))
            pygame.draw.rect(
                self.game_surface,
                (100, 100, 100),
                (legend_x, legend_y, legend_width, legend_height),
                1,
                border_radius=8,
            )

            # Draw title
            title = self._help_title_font.render("Controls", True, (255, 255, 255))
            self.game_surface.blit(title, (legend_x + padding, legend_y + padding))

            # Draw controls
            y = legend_y + padding + 30
            for key, desc in controls:
                key_surface = self._help_font.render(key, True, (100, 200, 255))
                desc_surface = self._help_font.render(f" - {desc}", True, (180, 180, 180))
                self.game_surface.blit(key_surface, (legend_x + padding, y))
                self.game_surface.blit(
                    desc_surface, (legend_x + padding + key_surface.get_width(), y)
                )
                y += line_height
        else:
            # Show hint to display help
            hint_text = self._speed_font.render("Press H for controls", True, (80, 80, 80))
            self.game_surface.blit(hint_text, (10, self.config.SCREEN_HEIGHT - 25))

        # Capture screenshot for web dashboard (before scaling, every 10 frames)
        self.frame_count = (self.frame_count + 1) % 10000  # Keep bounded to avoid overflow
        if self.web_dashboard and self.frame_count % 10 == 0:
            self.web_dashboard.capture_screenshot(self.game_surface)

        # Emit NN visualization data to web dashboard (throttled by server)
        if self.web_dashboard:
            self._emit_nn_visualization(state, action)

        # Clear main screen and scale game surface to fit window
        self.screen.fill((0, 0, 0))

        # Scale the game surface (optimize: skip scaling at 1:1 ratio)
        if abs(self.scale_factor - 1.0) < 0.001:  # Effectively 1.0
            # No scaling needed - blit directly
            self.screen.blit(self.game_surface, (self.game_offset_x, self.game_offset_y))
        else:
            # Calculate scaled size
            scaled_width = int(self.config.SCREEN_WIDTH * self.scale_factor)
            scaled_height = int(self.config.SCREEN_HEIGHT * self.scale_factor)
            # Use smoothscale for better quality
            scaled_surface = pygame.transform.smoothscale(
                self.game_surface, (scaled_width, scaled_height)
            )
            # Blit centered on screen
            self.screen.blit(scaled_surface, (self.game_offset_x, self.game_offset_y))

        # Update and render notifications (on top of scaled game)
        self._update_notifications()
        self._render_notifications(self.screen)

        pygame.display.flip()

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
            # Don't crash training on visualization errors, but surface the first failure.
            if not getattr(self, "_nn_visualization_error_reported", False):
                self._nn_visualization_error_reported = True
                message = f"⚠️ Neural network visualization update failed: {type(exc).__name__}"
                if self.web_dashboard:
                    self.web_dashboard.log(message, "warning")
                else:
                    print(message)

    def _save_model(self, filename: str, save_reason: str = "manual", quiet: bool = False) -> bool:
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

    def _save_model_as(self, filename: str) -> None:
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

    def _cleanup_old_periodic_saves(self, keep_last: int = 5) -> None:
        """
        Delete old periodic checkpoint saves, keeping only the most recent ones.
        """
        for filepath in self.model_service.cleanup_old_periodic_saves(keep_last=keep_last):
            if self.web_dashboard:
                self.web_dashboard.log(
                    f"🗑️ Cleaned up old checkpoint: {os.path.basename(filepath)}",
                    "info",
                )
