"""Interactive pygame runtime for visual training and play modes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Any, Iterable, List, Optional, cast

import numpy as np
import pygame

from config import Config
from src.ai.agent import Agent
from src.app.game_factory import create_single_game
from src.app.interactive_dashboard import InteractiveDashboardMixin
from src.app.interactive_rendering import InteractiveRenderingMixin
from src.app.lifecycle_types import EpisodeMetrics
from src.app.model_service import ModelService as AppModelService
from src.app.process_control import restart_with_game
from src.app.training_runtime import (
    build_nn_snapshot,
    emit_nn_snapshot_to_dashboard,
    request_save_and_stop,
)
from src.game import (
    BaseGame,
    ControlDisplayProvider,
    HumanActionProvider,
    HumanStepProvider,
    get_game_info,
    list_games,
)
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

__all__ = ["GameApp", "build_nn_snapshot", "emit_nn_snapshot_to_dashboard"]


class GameApp(InteractiveDashboardMixin, InteractiveRenderingMixin):
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

    # Speed presets for clean stepping
    # Smoother geometric progression for speed control
    SPEED_PRESETS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    _last_logged_speed: float = 0.0  # Track last logged speed to avoid spam

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
        print("\n   Controls:")
        print("   - P: Pause/Resume")
        print("   - S: Save model")
        print("   - +/-: Speed up/down")
        print("   - F: Toggle fullscreen")
        print("   - Q/ESC: Quit")
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
                    self.agent.learn()

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
