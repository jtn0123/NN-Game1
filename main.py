#!/usr/bin/env python3
"""
Neural Network Game AI - Main Entry Point
==========================================

This is the main script that runs the AI training with live visualization.

Usage:
    # Train with visualization (default)
    python main.py
    
    # Train without visualization (faster)
    python main.py --headless
    
    # TURBO MODE: Maximum speed training (~5000 steps/sec on M4!)
    python main.py --headless --turbo --episodes 5000
    
    # TURBO + Web Dashboard: Best of both worlds
    python main.py --headless --turbo --web --port 5001
    
    # VECTORIZED: 8 parallel games for ~3x additional speedup
    python main.py --headless --turbo --vec-envs 8
    
    # Custom performance tuning
    python main.py --headless --learn-every 4 --batch-size 256
    
    # Play with a trained model
    python main.py --play --model models/breakout_best.pth
    
    # Human play mode (for testing game)
    python main.py --human
    
    # Custom training parameters
    python main.py --episodes 5000 --lr 0.0001

Performance Options:
    --headless        Skip pygame entirely for max throughput
    --turbo           Preset: learn-every=8, batch=128, grad-steps=2 (~5000 steps/sec)
    --learn-every N   Learn every N steps (default: 1, try 4 for ~4x speedup)
    --batch-size N    Training batch size (default: 128)
    --torch-compile   Enable torch.compile() for ~20-50% extra speedup

The visualization shows:
    - Left side: The game (Breakout)
    - Right side: Neural network with live activations
    - Bottom: Training metrics dashboard

Press:
    - ESC or Q: Quit
    - P: Pause/Resume training
    - S: Save current model
    - R: Reset episode
    - +/-: Adjust game speed
    - F: Toggle fullscreen
"""

# Suppress pygame's pkg_resources deprecation warning (pygame issue #4557)
# This is a known issue - pygame hasn't migrated to importlib.resources yet
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import pygame
import numpy as np
import argparse
import sys
import os
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Any, Type, Union, List
from enum import Enum, auto

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.game import get_game, list_games, get_game_info, BaseGame, GameMenu
from src.game.breakout import Breakout, VecBreakout
from src.game.space_invaders import VecSpaceInvaders
from src.ai.agent import Agent, TrainingHistory
from src.ai.trainer import Trainer
from src.ai.evaluator import Evaluator  # Deterministic performance tracking
from src.visualizer.dashboard import Dashboard  # Still used for internal tracking
from src.visualizer.hud import TrainingHUD
from src.visualizer.pause_menu import PauseMenu

# Optional web dashboard
WEB_AVAILABLE: bool
WebDashboard: Optional[Type[Any]]
try:
    from src.web import WebDashboard as _WebDashboard
    WebDashboard = _WebDashboard
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    WebDashboard = None


class GameState(Enum):
    """Application state machine."""
    MENU = auto()
    TRAINING = auto()
    PAUSED = auto()
    PLAY_MODE = auto()
    HUMAN_MODE = auto()


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    score: int
    reward: float
    steps: int
    epsilon: float
    bricks_hit: int
    won: bool


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
    
    def __init__(self, config: Config, args: argparse.Namespace, existing_dashboard: Optional[Any] = None):
        """
        Initialize the application.

        Args:
            config: Configuration object
            args: Command line arguments
            existing_dashboard: Optional pre-initialized WebDashboard instance
        """
        self.config = config
        self.args = args
        
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
        game_display_name = game_info.get('name', config.GAME_NAME.title()) if game_info else config.GAME_NAME.title()
        pygame.display.set_caption(f"🧠 Neural Network AI - {game_display_name}")
        
        # Calculate window size
        # Layout: Game only - all training info is on the web dashboard
        self.game_width = config.SCREEN_WIDTH   # 800 - for reference
        self.game_height = config.SCREEN_HEIGHT  # 600 - for reference
        
        # Window size: game only (training stats and NN viz are on web dashboard)
        self.window_width = config.SCREEN_WIDTH + 20
        self.window_height = config.SCREEN_HEIGHT + 20
        
        # Minimum window dimensions
        self.min_window_width = config.SCREEN_WIDTH + 10
        self.min_window_height = config.SCREEN_HEIGHT + 10
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE
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
        GameClass = get_game(config.GAME_NAME)
        if GameClass is None:
            print(f"❌ Unknown game: {config.GAME_NAME}")
            print(f"   Available games: {', '.join(list_games())}")
            sys.exit(1)
        # Concrete game classes accept (config, headless) but BaseGame has no __init__ params
        self.game: BaseGame = GameClass(config)  # type: ignore[call-arg]
        
        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config
        )
        
        # Load model if specified, or auto-load most recent save
        self._initial_model_path = self._resolve_model_path(
            args.model, 
            state_size=self.game.state_size,
            action_size=self.game.action_size
        )
        
        # Create dashboard for internal tracking (rendering moved to web dashboard)
        self.dashboard = Dashboard(
            config=config,
            x=0, y=0, width=400, height=100  # Position doesn't matter, not rendered
        )

        # Create HUD and PauseMenu
        self.hud = TrainingHUD(config=config)
        self.pause_menu = PauseMenu(screen_width=config.SCREEN_WIDTH, screen_height=config.SCREEN_HEIGHT)

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
            self.web_dashboard.on_save_callback = lambda: self._save_model(f"{config.GAME_NAME}_web_save.pth", save_reason="manual")
            self.web_dashboard.on_save_as_callback = self._save_model_as
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.on_reset_callback = self._reset_episode
            self.web_dashboard.on_start_fresh_callback = self._start_fresh
            self.web_dashboard.on_load_model_callback = self._load_model
            self.web_dashboard.on_config_change_callback = self._apply_config
            self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
            self.web_dashboard.on_restart_with_game_callback = lambda game: restart_with_game(game, args)
            self.web_dashboard.on_save_and_quit_callback = self._save_and_quit

            # Send system info to dashboard
            self._send_system_info()

            # Log startup info
            self._log_startup_info()
        elif hasattr(args, 'web') and args.web and WEB_AVAILABLE and WebDashboard is not None:
            # Create new dashboard
            self.web_dashboard = WebDashboard(config, port=args.port)
            self.web_dashboard.on_pause_callback = self._toggle_pause
            self.web_dashboard.on_save_callback = lambda: self._save_model(f"{config.GAME_NAME}_web_save.pth", save_reason="manual")
            self.web_dashboard.on_save_as_callback = self._save_model_as
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.on_reset_callback = self._reset_episode
            self.web_dashboard.on_start_fresh_callback = self._start_fresh
            self.web_dashboard.on_load_model_callback = self._load_model
            self.web_dashboard.on_config_change_callback = self._apply_config
            self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
            self.web_dashboard.on_restart_with_game_callback = lambda game: restart_with_game(game, args)
            self.web_dashboard.on_save_and_quit_callback = self._save_and_quit
            self.web_dashboard.start()

            # Show URL prominently
            print("\n" + "=" * 60)
            print(f"🌐 WEB DASHBOARD: http://localhost:{args.port}")
            print("=" * 60 + "\n")

            # Send system info to dashboard
            self._send_system_info()

            # Log startup info
            self._log_startup_info()
        
        # Load the initial model (now that dashboard is ready)
        if self._initial_model_path:
            self._load_model(self._initial_model_path)
            if self.web_dashboard:
                self.web_dashboard.log(f"📂 Auto-loaded: {os.path.basename(self._initial_model_path)}", "success")
    
    def _resolve_model_path(self, explicit_path: Optional[str], state_size: int, action_size: int) -> Optional[str]:
        """
        Resolve which model to load on startup.
        
        Priority:
        1. Explicitly specified --model path (if compatible)
        2. Most recently modified .pth file in game-specific directory
        
        Args:
            explicit_path: Path from --model argument, or None
            state_size: Expected state size for compatibility check
            action_size: Expected action size for compatibility check
            
        Returns:
            Path to model file, or None if no model found
        """
        from src.ai.agent import Agent
        
        # If explicit path specified, check compatibility
        if explicit_path and os.path.exists(explicit_path):
            info = Agent.inspect_model(explicit_path)
            if info and info.get('state_size') == state_size and info.get('action_size') == action_size:
                return explicit_path
            else:
                print(f"⚠️  Specified model incompatible: {os.path.basename(explicit_path)}")
                print(f"   Expected: state_size={state_size}, action_size={action_size}")
                if info:
                    print(f"   Model has: state_size={info.get('state_size')}, action_size={info.get('action_size')}")
                return None
        
        # Only auto-load from game-specific directory (e.g., models/space_invaders/)
        model_dir = self.config.GAME_MODEL_DIR
        if not os.path.exists(model_dir):
            return None
        
        model_files = []
        for f in os.listdir(model_dir):
            if f.endswith('.pth'):
                filepath = os.path.join(model_dir, f)
                mtime = os.path.getmtime(filepath)
                model_files.append((filepath, mtime))
        
        if not model_files:
            return None
        
        # Sort by modification time, newest first
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        most_recent = model_files[0][0]
        print(f"📂 Auto-loading most recent save: {os.path.basename(most_recent)}")
        return most_recent
    
    def _restore_training_history(self, filepath: str) -> None:
        """Restore training history from a saved model (called after dashboard is ready)."""
        try:
            import torch
            checkpoint = torch.load(filepath, map_location=self.config.DEVICE, weights_only=False)
            
            if 'training_history' in checkpoint:
                history_data = checkpoint['training_history']
                training_history = TrainingHistory.from_dict(history_data)
                
                if len(training_history.scores) > 0:
                    # Restore internal tracking from TrainingHistory
                    self.episode_history.clear()
                    for i in range(len(training_history.scores)):
                        self.episode_history.append(EpisodeMetrics(
                            score=training_history.scores[i],
                            reward=training_history.rewards[i] if i < len(training_history.rewards) else 0.0,
                            steps=training_history.steps[i] if i < len(training_history.steps) else 0,
                            epsilon=training_history.epsilons[i] if i < len(training_history.epsilons) else 1.0,
                            bricks_hit=training_history.bricks[i] if i < len(training_history.bricks) else 0,
                            won=training_history.wins[i] if i < len(training_history.wins) else False,
                        ))
                    self.recent_scores = deque(training_history.scores[-1000:], maxlen=1000)
                    
                    # Restore dashboard with historical data
                    for i, score in enumerate(training_history.scores):
                        eps = training_history.epsilons[i] if i < len(training_history.epsilons) else 0.5
                        reward = training_history.rewards[i] if i < len(training_history.rewards) else 0.0
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
                            reward=reward
                        )
                    
                    print(f"📊 Restored {len(training_history.scores)} episodes of training history")
            
            # Restore episode counter and best score from metadata
            if 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                self.episode = metadata.get('episode', len(self.episode_history))
                self.best_score_ever = metadata.get('best_score', 0)
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
        self.web_dashboard.log(f"State size: {self.game.state_size}, Action size: {self.game.action_size}", "info")
        self.web_dashboard.log(f"Network: {self.config.HIDDEN_LAYERS}", "info", {
            'hidden_layers': self.config.HIDDEN_LAYERS,
            'activation': self.config.ACTIVATION
        })
        self.web_dashboard.log(f"Learning rate: {self.config.LEARNING_RATE}", "info", {
            'lr': self.config.LEARNING_RATE,
            'gamma': self.config.GAMMA,
            'batch_size': self.config.BATCH_SIZE
        })
        self.web_dashboard.log(f"Epsilon: {self.config.EPSILON_START} → {self.config.EPSILON_END}", "info", {
            'start': self.config.EPSILON_START,
            'end': self.config.EPSILON_END,
            'decay': self.config.EPSILON_DECAY
        })
        target_str = "Unlimited" if self.config.MAX_EPISODES == 0 else str(self.config.MAX_EPISODES)
        self.web_dashboard.log(f"Target episodes: {target_str}", "info")
        self.web_dashboard.log("Ready to train! Use controls to manage training.", "info")
    
    def _toggle_pause(self) -> None:
        """Toggle pause state (for web dashboard control)."""
        self.paused = not self.paused
        if self.web_dashboard:
            self.web_dashboard.publisher.set_paused(self.paused)
            status = "⏸️ Training paused" if self.paused else "▶️ Training resumed"
            self.web_dashboard.log(status, "action")
        print("⏸️  Paused" if self.paused else "▶️  Resumed")
    
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
            self.web_dashboard.log("🔄 Starting fresh training - resetting agent and clearing memory", "warning")
        
        # Create a new agent (fresh neural network)
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=self.config
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
            self.web_dashboard.socketio.emit('training_reset', {
                'message': 'Training reset - starting fresh'
            })
            
            # Emit cleared logs
            self.web_dashboard.socketio.emit('console_logs', {'logs': []})
            
            # Emit updated state with empty history
            self.web_dashboard.socketio.emit('state_update', publisher.get_snapshot())
            
            self.web_dashboard.log("✅ Fresh training started - agent reset, memory cleared, all charts and logs reset", "success")
        
        print("✅ Fresh training started - agent reset, memory cleared")
    
    def _save_and_quit(self) -> None:
        """Save the model and exit the application gracefully."""
        if self.web_dashboard:
            self.web_dashboard.log("💾 Saving model before shutdown...", "warning")
        
        # Save the model and verify it completed
        save_success = self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="shutdown")
        
        if save_success:
            if self.web_dashboard:
                self.web_dashboard.log("✅ Model saved. Shutting down...", "success")
            print("\n👋 Save & Quit requested. Model saved. Exiting...")
        else:
            if self.web_dashboard:
                self.web_dashboard.log("⚠️ Save may have failed. Shutting down...", "warning")
            print("\n⚠️ Save & Quit requested. Save may have failed. Exiting...")
        
        # Flush output buffers to ensure messages are written
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Give time for the save event to propagate to clients
        time.sleep(0.5)
        
        # Exit gracefully - use os._exit() because this is called from SocketIO thread
        self.running = False
        pygame.quit()
        os._exit(0)  # Terminates process from any thread
    
    def _load_model(self, filepath: str) -> None:
        """Load a model from file and restore training history."""
        try:
            metadata, training_history = self.agent.load(filepath, quiet=True)
            
            # If load returned None, model is incompatible (architecture mismatch)
            if metadata is None and training_history is None:
                if self.web_dashboard:
                    self.web_dashboard.log(f"⚠️  Model incompatible: {os.path.basename(filepath)} - starting fresh training", "warning")
                return  # Skip restoration, start fresh
            
            # Restore training history if available
            if training_history and len(training_history.scores) > 0:
                self.episode_history.clear()
                for i in range(len(training_history.scores)):
                    self.episode_history.append(EpisodeMetrics(
                        score=training_history.scores[i],
                        reward=training_history.rewards[i] if i < len(training_history.rewards) else 0.0,
                        steps=training_history.steps[i] if i < len(training_history.steps) else 0,
                        epsilon=training_history.epsilons[i] if i < len(training_history.epsilons) else 1.0,
                        bricks_hit=training_history.bricks[i] if i < len(training_history.bricks) else 0,
                        won=training_history.wins[i] if i < len(training_history.wins) else False,
                    ))
                self.recent_scores = deque(training_history.scores[-1000:], maxlen=1000)
                
                # Restore dashboard with historical data
                for i, score in enumerate(training_history.scores):
                    eps = training_history.epsilons[i] if i < len(training_history.epsilons) else 0.5
                    reward = training_history.rewards[i] if i < len(training_history.rewards) else 0.0
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
                        reward=reward
                    )
                
                # Restore episode counter and best score from metadata
                if metadata:
                    self.episode = metadata.episode
                    self.best_score_ever = metadata.best_score
                    self.steps = metadata.total_steps
                else:
                    self.episode = len(training_history.scores)
                    self.best_score_ever = max(training_history.scores) if training_history.scores else 0
                
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
                self.web_dashboard.log(f"📂 Loaded model: {os.path.basename(filepath)}{history_msg}", "success", {
                    'path': filepath,
                    'epsilon': self.agent.epsilon,
                    'steps': self.agent.steps,
                    'episode': self.episode,
                    'best_score': self.best_score_ever
                })
                # Update save status to reflect loaded model
                self.web_dashboard.publisher.record_save(
                    filename=os.path.basename(filepath),
                    reason="loaded",
                    episode=self.episode,
                    best_score=self.best_score_ever
                )
        except Exception as e:
            if self.web_dashboard:
                self.web_dashboard.log(f"❌ Failed to load model: {str(e)}", "error")
            import traceback
            traceback.print_exc()
    
    def _sync_web_dashboard_history(self, training_history: TrainingHistory, metadata: Optional[Any]) -> None:
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
            publisher.state.win_rate = sum(1 for w in recent_wins if w) / len(recent_wins) if recent_wins else 0.0
        
        # Emit an update to connected clients
        self.web_dashboard.socketio.emit('state_update', publisher.get_snapshot())
    
    def _apply_config(self, config_data: dict) -> None:
        """Apply configuration changes from web dashboard."""
        changes = []

        if 'learning_rate' in config_data:
            try:
                lr = float(config_data['learning_rate'])
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
                    param_group['lr'] = lr
                changes.append(f"LR: {old_lr} → {lr}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid learning_rate value: {config_data['learning_rate']} - {e}")
        
        if 'epsilon' in config_data:
            try:
                eps = float(config_data['epsilon'])
                if not math.isfinite(eps):
                    raise ValueError("Epsilon must be finite (not NaN or Inf)")
                old_eps = self.agent.epsilon
                # Clamp epsilon to valid range with feedback
                clamped_eps = max(self.config.EPSILON_END, min(self.config.EPSILON_START, eps))
                if clamped_eps != eps:
                    print(f"⚠️  Epsilon {eps:.4f} clamped to valid range [{self.config.EPSILON_END}, {self.config.EPSILON_START}]")
                self.agent.epsilon = clamped_eps
                changes.append(f"Epsilon: {old_eps:.4f} → {self.agent.epsilon:.4f}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid epsilon value: {config_data['epsilon']} - {e}")
        
        if 'epsilon_decay' in config_data:
            try:
                decay = float(config_data['epsilon_decay'])
                if not math.isfinite(decay) or decay <= 0 or decay > 1:
                    raise ValueError("Epsilon decay must be finite and in (0, 1]")
                self.config.EPSILON_DECAY = decay
                changes.append(f"Decay: {decay}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid epsilon_decay value: {config_data['epsilon_decay']} - {e}")

        if 'gamma' in config_data:
            try:
                gamma = float(config_data['gamma'])
                if not math.isfinite(gamma) or gamma < 0 or gamma > 1:
                    raise ValueError("Gamma must be finite and in [0, 1]")
                self.config.GAMMA = gamma
                changes.append(f"Gamma: {gamma}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid gamma value: {config_data['gamma']} - {e}")

        if 'batch_size' in config_data:
            try:
                batch_size = int(config_data['batch_size'])
                if batch_size <= 0:
                    raise ValueError("Batch size must be positive")
                if batch_size > self.config.MEMORY_SIZE:
                    raise ValueError(f"Batch size ({batch_size}) cannot exceed memory size ({self.config.MEMORY_SIZE})")
                self.config.BATCH_SIZE = batch_size
                changes.append(f"Batch: {batch_size}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid batch_size value: {config_data['batch_size']} - {e}")

        if 'learn_every' in config_data:
            try:
                learn_every = int(config_data['learn_every'])
                if learn_every <= 0:
                    raise ValueError("Learn every must be positive")
                self.config.LEARN_EVERY = learn_every
                changes.append(f"LearnEvery: {learn_every}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid learn_every value: {config_data['learn_every']} - {e}")

        if 'gradient_steps' in config_data:
            try:
                grad_steps = int(config_data['gradient_steps'])
                if grad_steps <= 0:
                    raise ValueError("Gradient steps must be positive")
                self.config.GRADIENT_STEPS = grad_steps
                changes.append(f"GradSteps: {grad_steps}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid gradient_steps value: {config_data['gradient_steps']} - {e}")
        
        if self.web_dashboard and changes:
            self.web_dashboard.log(f"⚙️ Config updated: {', '.join(changes)}", "action", config_data)
    
    def _send_system_info(self) -> None:
        """Send system information to web dashboard."""
        if not self.web_dashboard:
            return
        
        # Check if torch.compile was used
        torch_compiled = getattr(self.agent, '_compiled', False)
        device_str = str(self.config.DEVICE)
        
        self.web_dashboard.publisher.set_system_info(
            device=device_str,
            torch_compiled=torch_compiled,
            target_episodes=self.config.MAX_EPISODES
        )
    
    def _set_performance_mode(self, mode: str) -> None:
        """Set performance mode from web dashboard."""
        if mode == 'normal':
            self.config.LEARN_EVERY = 1
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 1
        elif mode == 'fast':
            self.config.LEARN_EVERY = 4
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 1
        elif mode == 'turbo':
            # Match CLI turbo preset - optimized for M4 CPU based on benchmarks
            self.config.LEARN_EVERY = 8
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 2
        elif mode == 'ultra':
            # Maximum throughput: less frequent learning, same batch size as turbo
            # learn_every=32 means learning 4x less often than turbo
            self.config.LEARN_EVERY = 32
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 2
        else:
            print(f"⚠️  Unknown performance mode: {mode}")
            return

        if self.web_dashboard:
            self.web_dashboard.publisher.set_performance_mode(mode)
            self.web_dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
            self.web_dashboard.publisher.state.batch_size = self.config.BATCH_SIZE
            self.web_dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS
            self.web_dashboard.log(
                f"⚡ Performance mode: {mode.upper()} (learn_every={self.config.LEARN_EVERY}, batch={self.config.BATCH_SIZE}, grad_steps={self.config.GRADIENT_STEPS})",
                "action"
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
        self.game_width = self.config.SCREEN_WIDTH   # 800
        self.game_height = self.config.SCREEN_HEIGHT  # 600

        # Update scaling for the new window size
        self._update_scale()

        # Update pause menu positions
        if hasattr(self, 'pause_menu') and self.pause_menu:
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

    def _show_notification(self, text: str, color: tuple = (100, 200, 255), duration: float = 2.0) -> None:
        """
        Show a notification on screen.

        Args:
            text: Notification text
            color: Text color (RGB tuple)
            duration: How long to show the notification in seconds
        """
        import time
        self._notifications.append({
            'text': text,
            'color': color,
            'start_time': time.time(),
            'duration': duration
        })

    def _update_notifications(self) -> None:
        """Remove expired notifications."""
        import time
        current_time = time.time()
        self._notifications = [
            n for n in self._notifications
            if current_time - n['start_time'] < n['duration']
        ]

    def _render_notifications(self, surface: pygame.Surface) -> None:
        """Render all active notifications."""
        import time
        if not self._notifications:
            return

        current_time = time.time()
        y_offset = 10

        for notification in self._notifications:
            elapsed = current_time - notification['start_time']
            # Fade out in the last 0.5 seconds
            alpha = 255
            if elapsed > notification['duration'] - 0.5:
                alpha = int(255 * (notification['duration'] - elapsed) / 0.5)
            alpha = max(0, min(255, alpha))

            text_surface = self._notification_font.render(notification['text'], True, notification['color'])

            # Create background with alpha
            bg_width = text_surface.get_width() + 20
            bg_height = text_surface.get_height() + 10
            bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
            pygame.draw.rect(bg_surface, (0, 0, 0, int(alpha * 0.7)), bg_surface.get_rect(), border_radius=5)

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
        new_speed = max(1.0, min(1000.0, speed))
        old_speed = self.game_speed
        self.game_speed = new_speed
        
        if self.web_dashboard:
            self.web_dashboard.publisher.set_speed(self.game_speed)
        
        # Only log if speed changed significantly (avoid spam when dragging slider)
        # Log when: forced, or speed is a preset value, or changed by >10%
        speed_changed_significantly = abs(new_speed - self._last_logged_speed) / max(1, self._last_logged_speed) > 0.1
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
        if game_name == 'space_invaders':
            print("   LEFT/RIGHT arrows: Move ship")
            print("   SPACE: Shoot")
        elif game_name == 'breakout':
            print("   LEFT/RIGHT arrows: Move paddle")
        elif game_name == 'pong':
            print("   UP/DOWN arrows (or W/S): Move paddle")
        elif game_name == 'snake':
            print("   Arrow keys (or WASD): Change direction")
        elif game_name == 'asteroids':
            print("   LEFT/RIGHT arrows: Rotate ship")
            print("   UP arrow: Thrust")
            print("   SPACE: Shoot")
            print("   (Multiple keys can be pressed simultaneously)")
        else:
            print("   Use arrow keys to control")

        print("\n   R: Reset game")
        print("   Q/ESC: Quit")
        print("   F: Toggle fullscreen")
        print("=" * 60 + "\n")

        # Enable controls display for games that support it
        if hasattr(self.game, 'show_controls'):
            self.game.show_controls = True

        state = self.game.reset()

        while self.running:
            self._handle_events()

            # Get keyboard input as dict for games that use get_human_action
            pressed = pygame.key.get_pressed()
            keys_dict = {key: pressed[key] for key in range(len(pressed))}

            # Handle game-specific controls
            if game_name == 'asteroids' and hasattr(self.game, 'step_human'):
                # Asteroids supports simultaneous actions via step_human
                state, reward, done, info = self.game.step_human(keys_dict)
                action = self.game.get_human_action(keys_dict)  # For display purposes
            elif hasattr(self.game, 'get_human_action'):
                # Use game's built-in human action helper
                action = self.game.get_human_action(keys_dict)
                state, reward, done, info = self.game.step(action)
            else:
                # Fallback: generic control mapping
                action = 1  # STAY / IDLE
                if pressed[pygame.K_LEFT]:
                    action = 0  # LEFT
                elif pressed[pygame.K_RIGHT]:
                    action = 2  # RIGHT
                elif pressed[pygame.K_SPACE] and game_name == 'space_invaders':
                    action = 3  # SHOOT
                state, reward, done, info = self.game.step(action)

            if done:
                score = info.get('score', 0)
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
        info: dict = {'score': 0, 'bricks_remaining': total_bricks}  # Default info for paused state
        
        while self.running:
            self._handle_events()
            
            if not self.paused:
                # Agent selects action
                action = self.agent.select_action(state, training=False)
                
                # Step game
                state, reward, done, info = self.game.step(action)
                episode_reward += float(reward)
                
                if done:
                    print(f"   Episode complete! Score: {info['score']}, Reward: {episode_reward:.1f}")
                    state = self.game.reset()
                    episode_reward = 0.0
                    self.episode += 1
                    self.dashboard.update(
                        self.episode, info['score'], 0, 0,
                        bricks_broken=total_bricks-info.get('bricks_remaining', total_bricks)
                    )
                
                self.selected_action = action
            
            # Render
            self._render_frame(state, self.selected_action if self.selected_action is not None else 1, {'score': info.get('score', 0)})
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
        while self.running and (self.config.MAX_EPISODES == 0 or self.episode < self.config.MAX_EPISODES):
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
                    if self.agent.steps % self.config.TARGET_UPDATE == 0 and self.agent.steps != self.last_target_update_step:
                        self.target_updates += 1
                        self.last_target_update_step = self.agent.steps
                        if self.web_dashboard:
                            self.web_dashboard.log(
                                f"🎯 Target network updated (#{self.target_updates})", 
                                "metric",
                                {'step': self.agent.steps, 'update_number': self.target_updates}
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
                            info['score'],
                            self.agent.epsilon,
                            self.agent.get_average_loss(100),
                            bricks_broken=total_bricks-info.get('bricks_remaining', total_bricks),
                            won=info.get('won', False),
                            reward=episode_reward
                        )
                        
                        # Update web dashboard if enabled (throttled to every 5 episodes for performance)
                        # Always emit on: first 10 episodes, new best score, or every 5th episode
                        is_new_best = info['score'] > self.best_score_ever
                        should_emit = (
                            self.web_dashboard and
                            (self.episode <= 10 or is_new_best or self.episode % 5 == 0)
                        )
                        if should_emit:
                            self.web_dashboard.emit_metrics(
                                episode=self.episode,
                                score=info['score'],
                                epsilon=self.agent.epsilon,
                                loss=self.agent.get_average_loss(100),
                                total_steps=self.steps,
                                won=info.get('won', False),
                                reward=episode_reward,
                                memory_size=len(self.agent.memory),
                                avg_q_value=avg_q_value,
                                exploration_actions=self.exploration_actions,
                                exploitation_actions=self.exploitation_actions,
                                target_updates=self.target_updates,
                                bricks_broken=episode_bricks_broken,
                                episode_length=episode_steps
                            )
                            # Update performance settings in dashboard state
                            self.web_dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
                            self.web_dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS
                            self.web_dashboard.publisher.state.batch_size = self.config.BATCH_SIZE
                            
                            # Log episode completion
                            self._log_episode_complete(
                                info, episode_reward, episode_steps, episode_duration,
                                episode_bricks_broken, avg_q_value, old_epsilon
                            )
                        
                        # Terminal log
                        if self.episode % self.config.LOG_EVERY == 0:
                            avg_score = np.mean(list(self.dashboard.scores)[-100:]) if self.dashboard.scores else 0
                            avg_loss = self.agent.get_average_loss(100)
                            print(f"Episode {self.episode:5d} | "
                                  f"Score: {info['score']:4d} | "
                                  f"Avg: {avg_score:6.1f} | "
                                  f"Loss: {avg_loss:.4f} | "
                                  f"Q: {avg_q_value:.1f} | "
                                  f"ε: {self.agent.epsilon:.3f}")
                        
                        # Track scores for metadata (deque auto-trims to maxlen=1000)
                        self.recent_scores.append(info['score'])

                        # Track full training history for save/restore
                        self.episode_history.append(EpisodeMetrics(
                            score=info['score'],
                            reward=episode_reward,
                            steps=episode_steps,
                            epsilon=self.agent.epsilon,
                            bricks_hit=episode_bricks_broken,
                            won=info.get('won', False),
                        ))
                        
                        # Save checkpoint (no replay buffer for periodic - saves disk space)
                        if self.episode % self.config.SAVE_EVERY == 0 and self.episode > 0:
                            self._save_model(f"{self.config.GAME_NAME}_ep{self.episode}.pth", save_reason="periodic")
                            self._cleanup_old_periodic_saves(keep_last=5)
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"💾 Checkpoint saved: {self.config.GAME_NAME}_ep{self.episode}.pth",
                                    "success"
                                )
                        
                        if info['score'] > self.best_score_ever:
                            self.best_score_ever = info['score']
                            self._save_model(f"{self.config.GAME_NAME}_best.pth", save_reason="best", quiet=False)
                            if self.web_dashboard:
                                avg_score = np.mean(self.recent_scores[-100:]) if self.recent_scores else 0.0
                                self.web_dashboard.log(
                                    f"🏆 New best score: {info['score']}! Model saved.",
                                    "success",
                                    {'score': info['score'], 'episode': self.episode, 'avg_100': round(avg_score, 1)}
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
                            "debug"
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
            self.web_dashboard.log("🎉 Training complete!", "success", {
                'total_episodes': self.episode,
                'best_score': self.best_score_ever,
                'total_steps': self.steps
            })
        
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
        old_epsilon: float
    ) -> None:
        """Log detailed episode completion to web dashboard."""
        if not self.web_dashboard:
            return
        
        # Determine episode outcome
        won = info.get('won', False)
        score = info['score']
        
        # Log episode metrics
        level = "success" if won else "metric"
        outcome = "🏆 WIN!" if won else ""
        
        self.web_dashboard.log(
            f"Episode {self.episode} complete {outcome}",
            level,
            {
                'episode': self.episode,
                'score': score,
                'reward': round(episode_reward, 2),
                'steps': episode_steps,
                'duration': round(episode_duration, 2),
                'bricks': bricks_broken,
                'won': won
            }
        )
        
        # Log detailed metrics every few episodes
        if self.episode % 5 == 0:
            loss = self.agent.get_average_loss(100)
            explore_ratio = self.exploration_actions / max(1, self.exploration_actions + self.exploitation_actions)
            
            self.web_dashboard.log(
                f"Training metrics: Loss={loss:.4f}, Q={avg_q_value:.2f}, Explore={explore_ratio:.1%}",
                "metric",
                {
                    'loss': round(loss, 6),
                    'avg_q_value': round(avg_q_value, 4),
                    'explore_ratio': round(explore_ratio, 4),
                    'memory_size': len(self.agent.memory),
                    'total_steps': self.steps
                }
            )
        
        # Log epsilon decay
        if abs(old_epsilon - self.agent.epsilon) > 0.001:
            self.web_dashboard.log(
                f"Epsilon decayed: {old_epsilon:.4f} → {self.agent.epsilon:.4f}",
                "debug",
                {'old': old_epsilon, 'new': self.agent.epsilon}
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
            
            while not done:
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
            scores.append(info['score'])
            
            # Time-based progress reporting
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time
            
            if elapsed_since_report >= self.config.REPORT_INTERVAL_SECONDS or episode % self.config.LOG_EVERY == 0:
                elapsed_total = current_time - start_time
                steps_per_sec = steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                avg_score = np.mean(scores[-100:]) if scores else 0
                avg_loss = self.agent.get_average_loss(100)
                q_values = self.agent.get_q_values(state)
                avg_q = float(np.mean(q_values))
                
                print(f"Episode {episode:5d} | "
                      f"Score: {info['score']:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Q: {avg_q:.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"⚡ {steps_per_sec:,.0f} steps/s")
                
                last_report_time = current_time
                steps_since_report = 0
            
            # Save checkpoints
            if episode % self.config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(f"{self.config.GAME_NAME}_ep{episode}.pth")
            
            if info['score'] > best_score:
                best_score = info['score']
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
                self.screen = pygame.display.set_mode(
                    (new_width, new_height),
                    pygame.RESIZABLE
                )
                self._update_layout(new_width, new_height)

            # Handle pause menu interactions when paused
            if self.paused:
                action = self.pause_menu.handle_event(event)
                if action == "resume":
                    self._toggle_pause()
                elif action == "save":
                    self._save_model(f"{self.config.GAME_NAME}_manual_save.pth", save_reason="manual")
                    if self.web_dashboard:
                        self.web_dashboard.log(f"💾 Manual save: {self.config.GAME_NAME}_manual_save.pth", "success")
                elif action == "menu":
                    # Save current progress before returning to menu
                    self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="menu_exit")
                    if self.web_dashboard:
                        self.web_dashboard.log("🏠 Returning to game selector...", "warning")
                        self.web_dashboard.launcher_mode = True
                        self.web_dashboard.socketio.emit('redirect_to_launcher', {
                            'message': 'Returning to game selector...'
                        })
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
                    success = self._save_model(f"{self.config.GAME_NAME}_manual_save.pth", save_reason="manual")
                    if success:
                        self._show_notification("Model Saved", (100, 255, 100), 1.5)
                    else:
                        self._show_notification("Save Failed", (255, 100, 100), 2.0)
                    if self.web_dashboard:
                        self.web_dashboard.log(f"💾 Manual save: {self.config.GAME_NAME}_manual_save.pth", "success")

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
                            (self.window_width, self.window_height),
                            pygame.RESIZABLE
                        )
                    else:
                        self.screen = pygame.display.set_mode(
                            (0, 0),
                            pygame.FULLSCREEN | pygame.RESIZABLE
                        )
                        # Update layout to new screen size
                        display_info = pygame.display.Info()
                        self._update_layout(display_info.current_w, display_info.current_h)

                elif event.key == pygame.K_h:
                    # Toggle help legend
                    self.show_help_legend = not self.show_help_legend
    
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
            if hasattr(self.game, 'get_action_labels'):
                action_labels = self.game.get_action_labels()
            else:
                # Fallback generic labels
                action_labels = [f"Action {i}" for i in range(self.game.action_size)]

            self.hud.render(
                surface=self.game_surface,
                episode=self.episode,
                score=info.get('score', 0),
                best_score=self.best_score_ever,
                epsilon=self.agent.epsilon if hasattr(self, 'agent') else 0.0,
                loss=0.0,  # Could track recent loss
                speed=self.game_speed,
                max_episodes=self.config.MAX_EPISODES,
                selected_action=action,
                action_labels=action_labels
            )

        # Render pause menu with context (centered on the game surface)
        if self.paused:
            # Build context for pause menu
            pause_context = {
                'episode': self.episode,
                'score': info.get('score', 0),
                'best_score': self.best_score_ever,
                'epsilon': self.agent.epsilon if hasattr(self, 'agent') else 0.0,
                'training_time': time.time() - self.training_start_time,
                'memory_size': len(self.agent.memory) if hasattr(self, 'agent') else 0,
                'memory_capacity': self.config.MEMORY_SIZE if hasattr(self.config, 'MEMORY_SIZE') else 0,
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
            pygame.draw.rect(self.game_surface, (100, 100, 100), (legend_x, legend_y, legend_width, legend_height), 1, border_radius=8)

            # Draw title
            title = self._help_title_font.render("Controls", True, (255, 255, 255))
            self.game_surface.blit(title, (legend_x + padding, legend_y + padding))

            # Draw controls
            y = legend_y + padding + 30
            for key, desc in controls:
                key_surface = self._help_font.render(key, True, (100, 200, 255))
                desc_surface = self._help_font.render(f" - {desc}", True, (180, 180, 180))
                self.game_surface.blit(key_surface, (legend_x + padding, y))
                self.game_surface.blit(desc_surface, (legend_x + padding + key_surface.get_width(), y))
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
            scaled_surface = pygame.transform.smoothscale(self.game_surface, (scaled_width, scaled_height))
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
            # Enable activation capture temporarily
            self.agent.policy_net.capture_activations = True
            
            # Get layer info
            layer_info = self.agent.policy_net.get_layer_info()
            
            # Get Q-values (this forward pass captures activations)
            q_values = self.agent.get_q_values(state)
            
            # Get activations
            raw_activations = self.agent.policy_net.get_activations()
            
            # Get weights (sampled for performance)
            raw_weights = self.agent.policy_net.get_weights()
            
            # Disable activation capture
            self.agent.policy_net.capture_activations = False
            
            # Format activations for JSON (convert numpy arrays to lists, limit neurons)
            max_neurons = 15  # Match the pygame visualizer limit
            activations: dict[str, list[float]] = {}
            for key, act in raw_activations.items():
                if len(act.shape) > 1:
                    act = act[0]  # Take first batch item
                # Normalize and limit to max_neurons
                act_list = act[:min(max_neurons, len(act))].tolist()
                activations[key] = act_list
            
            # Format weights for JSON (sample connections for performance)
            weights: list[list[list[float]]] = []
            for i, w in enumerate(raw_weights):
                if w is not None:
                    # Sample weights: take first 15 rows and first 15 columns
                    sampled_w = w[:min(15, w.shape[0]), :min(15, w.shape[1])]
                    weights.append(sampled_w.tolist())
            
            # Get action labels from game if available
            action_labels = ["LEFT", "STAY", "RIGHT"]  # Default for Breakout
            if hasattr(self.game, 'get_action_labels'):
                action_labels = self.game.get_action_labels()
            
            # Emit to web dashboard (throttling handled by publisher)
            self.web_dashboard.emit_nn_visualization(
                layer_info=layer_info,
                activations=activations,
                q_values=q_values.tolist(),
                selected_action=selected_action,
                weights=weights,
                step=self.agent.steps,
                action_labels=action_labels
            )
        except Exception:
            # Don't crash training on visualization errors - silently ignore
            pass
    
    def _save_model(
        self,
        filename: str,
        save_reason: str = "manual",
        quiet: bool = False
    ) -> bool:
        """Save the current model with rich metadata.
        
        Returns:
            True if save succeeded, False otherwise
        """
        # Ensure game-specific model directory exists
        os.makedirs(self.config.GAME_MODEL_DIR, exist_ok=True)
        filepath = os.path.join(self.config.GAME_MODEL_DIR, filename)
        
        # Calculate metrics for metadata
        avg_score = np.mean(list(self.recent_scores)[-100:]) if self.recent_scores else 0.0
        win_rate = self.dashboard.get_win_rate() if hasattr(self.dashboard, 'get_win_rate') else 0.0

        # Build training history for dashboard restoration from episode_history
        training_history = TrainingHistory(
            scores=[ep.score for ep in self.episode_history],
            rewards=[ep.reward for ep in self.episode_history],
            steps=[ep.steps for ep in self.episode_history],
            epsilons=[ep.epsilon for ep in self.episode_history],
            bricks=[ep.bricks_hit for ep in self.episode_history],
            wins=[ep.won for ep in self.episode_history],
            losses=list(self.agent.losses)[-1000:] if self.agent.losses else [],
            q_values=[]  # Q-values not tracked per-episode currently
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
            quiet=quiet
        )
        
        # Only record save to web dashboard if save succeeded
        if result is not None and self.web_dashboard:
            self.web_dashboard.publisher.record_save(
                filename=filename,
                reason=save_reason,
                episode=self.episode,
                best_score=self.best_score_ever
            )
        
        return result is not None
    
    def _save_model_as(self, filename: str) -> None:
        """Save model with a custom filename (from web dashboard)."""
        # Remove .pth extension if present (we'll add it back later)
        if filename.endswith('.pth'):
            filename = filename[:-4]
        
        # Sanitize filename (only alphanumeric, underscore, hyphen)
        filename = "".join(c for c in filename if c.isalnum() or c in '_-').strip()
        
        # Ensure we have a valid base name (not empty or just dots)
        if not filename or filename.replace('.', '') == '':
            filename = "custom_save"
        
        # Add .pth extension
        filename = filename + '.pth'
        
        self._save_model(filename, save_reason="manual")
        if self.web_dashboard:
            self.web_dashboard.log(f"💾 Saved as: {filename}", "success")
    
    def _cleanup_old_periodic_saves(self, keep_last: int = 5) -> None:
        """
        Delete old periodic checkpoint saves, keeping only the most recent ones.
        """
        import glob
        import re
        
        model_dir = self.config.GAME_MODEL_DIR
        game_name = self.config.GAME_NAME
        
        pattern = os.path.join(model_dir, f"{game_name}_ep*.pth")
        periodic_saves = glob.glob(pattern)
        
        if len(periodic_saves) <= keep_last:
            return
        
        def get_episode_num(path: str) -> int:
            match = re.search(r'_ep(\d+)\.pth$', path)
            return int(match.group(1)) if match else 0
        
        periodic_saves.sort(key=get_episode_num)
        to_delete = periodic_saves[:-keep_last]
        
        for filepath in to_delete:
            try:
                os.remove(filepath)
            except Exception:
                pass


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
    
    def __init__(self, config: Config, args: argparse.Namespace, existing_dashboard: Optional[Any] = None):
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
        self.num_envs = getattr(args, 'vec_envs', 1)
        
        # Get game class from registry
        GameClass = get_game(config.GAME_NAME)
        if GameClass is None:
            print(f"❌ Unknown game: {config.GAME_NAME}")
            print(f"   Available games: {', '.join(list_games())}")
            sys.exit(1)
        
        # Type annotations for game and vec_env
        self.vec_env: Optional[Union[VecBreakout, VecSpaceInvaders]] = None
        self.game: BaseGame

        if self.num_envs > 1:
            # Create vectorized environment for parallel game execution
            if config.GAME_NAME == 'breakout':
                self.vec_env = VecBreakout(self.num_envs, config, headless=True)
                self.game = self.vec_env.envs[0]  # Reference for state/action size
                print(f"🎮 Vectorized: {self.num_envs} parallel environments")
            elif config.GAME_NAME == 'space_invaders':
                self.vec_env = VecSpaceInvaders(self.num_envs, config, headless=True)
                self.game = self.vec_env.envs[0]  # Reference for state/action size
                print(f"🎮 Vectorized: {self.num_envs} parallel environments")
            else:
                print(f"⚠️ Vectorized environments not yet supported for {config.GAME_NAME}")
                print(f"   Falling back to single environment")
                # Concrete game classes accept (config, headless) but BaseGame has no __init__ params
                self.game = GameClass(config, headless=True)  # type: ignore[call-arg]
                self.num_envs = 1
        else:
            # Single game mode (original behavior)
            # Concrete game classes accept (config, headless) but BaseGame has no __init__ params
            self.game = GameClass(config, headless=True)  # type: ignore[call-arg]
        
        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config
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
            action_size=self.game.action_size
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
                self.web_dashboard.log(f"📂 Auto-loaded: {os.path.basename(initial_model_path)}", "success")
                # Sync history to dashboard NOW that dashboard is ready
                self._sync_history_to_dashboard_after_load(initial_model_path)
        elif hasattr(args, 'web') and args.web and WEB_AVAILABLE and WebDashboard is not None:
            self.web_dashboard = WebDashboard(config, port=args.port)
            self._setup_web_callbacks()
            self.web_dashboard.start()

            # Show URL prominently
            print("\n" + "=" * 60)
            print(f"🌐 WEB DASHBOARD: http://localhost:{args.port}")
            print("=" * 60 + "\n")

            self._send_system_info()
            self._log_startup_info()
            if initial_model_path:
                self.web_dashboard.log(f"📂 Auto-loaded: {os.path.basename(initial_model_path)}", "success")
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
                log_dir=os.path.join(config.LOG_DIR, 'eval'),
                plateau_threshold=config.EVAL_PLATEAU_THRESHOLD
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
            import torch
            checkpoint = torch.load(filepath, map_location=self.config.DEVICE, weights_only=False)
            
            if 'training_history' in checkpoint:
                training_history = TrainingHistory.from_dict(checkpoint['training_history'])
                metadata = None
                if 'metadata' in checkpoint:
                    from src.ai.agent import SaveMetadata
                    metadata = SaveMetadata.from_dict(checkpoint['metadata'])
                
                if len(training_history.scores) > 0:
                    self._sync_web_dashboard_history(training_history, metadata)
                    print(f"📊 Dashboard charts restored ({len(training_history.scores)} episodes)")
        except Exception as e:
            print(f"⚠️ Could not restore dashboard history: {e}")
    
    def _resolve_model_path(self, explicit_path: Optional[str], state_size: int, action_size: int) -> Optional[str]:
        """
        Resolve which model to load on startup.
        
        Priority:
        1. Explicitly specified --model path (if compatible)
        2. Most recently modified .pth file in game-specific directory
        
        Args:
            explicit_path: Path from --model argument, or None
            state_size: Expected state size for compatibility check
            action_size: Expected action size for compatibility check
            
        Returns:
            Path to model file, or None if no model found
        """
        from src.ai.agent import Agent
        
        # If explicit path specified, check compatibility
        if explicit_path and os.path.exists(explicit_path):
            info = Agent.inspect_model(explicit_path)
            if info and info.get('state_size') == state_size and info.get('action_size') == action_size:
                return explicit_path
            else:
                print(f"⚠️  Specified model incompatible: {os.path.basename(explicit_path)}")
                print(f"   Expected: state_size={state_size}, action_size={action_size}")
                if info:
                    print(f"   Model has: state_size={info.get('state_size')}, action_size={info.get('action_size')}")
                return None
        
        # Only auto-load from game-specific directory (e.g., models/space_invaders/)
        model_dir = self.config.GAME_MODEL_DIR
        if not os.path.exists(model_dir):
            return None
        
        model_files = []
        for f in os.listdir(model_dir):
            if f.endswith('.pth'):
                filepath = os.path.join(model_dir, f)
                mtime = os.path.getmtime(filepath)
                model_files.append((filepath, mtime))
        
        if not model_files:
            return None
        
        # Sort by modification time, newest first
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        most_recent = model_files[0][0]
        print(f"📂 Auto-loading most recent save: {os.path.basename(most_recent)}")
        return most_recent
    
    def _setup_web_callbacks(self) -> None:
        """Set up web dashboard control callbacks."""
        if not self.web_dashboard:
            return
        
        self.web_dashboard.on_pause_callback = self._toggle_pause
        self.web_dashboard.on_save_callback = lambda: self._save_model(f"{self.config.GAME_NAME}_web_save.pth", save_reason="manual")
        self.web_dashboard.on_save_as_callback = self._save_model_as
        self.web_dashboard.on_reset_callback = self._reset_episode
        self.web_dashboard.on_start_fresh_callback = self._start_fresh
        self.web_dashboard.on_load_model_callback = self._load_model
        self.web_dashboard.on_config_change_callback = self._apply_config
        self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
        self.web_dashboard.on_restart_with_game_callback = lambda game: restart_with_game(game, self.args)
        self.web_dashboard.on_save_and_quit_callback = self._save_and_quit
        # Speed control doesn't apply to headless (no frame timing)
        self.web_dashboard.on_speed_callback = lambda x: None
    
    def _send_system_info(self) -> None:
        """Send system information to web dashboard."""
        if not self.web_dashboard:
            return
        
        torch_compiled = getattr(self.agent, '_compiled', False)
        device_str = str(self.config.DEVICE)
        
        self.web_dashboard.publisher.set_system_info(
            device=device_str,
            torch_compiled=torch_compiled,
            target_episodes=self.config.MAX_EPISODES,
            headless=True  # No pygame, no screenshots
        )

        # ADD: Set number of parallel environments
        self.web_dashboard.publisher.state.num_envs = self.num_envs

        # Set performance mode based on turbo flag
        if self.args.turbo:
            self.web_dashboard.publisher.set_performance_mode('turbo')
        else:
            self.web_dashboard.publisher.set_performance_mode('normal')
    
    def _log_startup_info(self) -> None:
        """Log startup configuration to web dashboard."""
        if not self.web_dashboard:
            return
        
        self.web_dashboard.log("🚀 Headless training started", "success")
        self.web_dashboard.log(f"Device: {self.config.DEVICE}", "info")
        self.web_dashboard.log(f"State size: {self.game.state_size}, Action size: {self.game.action_size}", "info")
        self.web_dashboard.log(f"Network: {self.config.HIDDEN_LAYERS}", "info")
        self.web_dashboard.log(f"Learn every: {self.config.LEARN_EVERY}, Grad steps: {self.config.GRADIENT_STEPS}", "info")
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
            self.web_dashboard.log("🔄 Starting fresh training - resetting agent and clearing memory", "warning")
        
        # Create a new agent (fresh neural network)
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=self.config
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
            self.web_dashboard.socketio.emit('training_reset', {
                'message': 'Training reset - starting fresh'
            })
            
            # Emit cleared logs
            self.web_dashboard.socketio.emit('console_logs', {'logs': []})
            
            # Emit updated state with empty history
            self.web_dashboard.socketio.emit('state_update', publisher.get_snapshot())
            
            self.web_dashboard.log("✅ Fresh training started - agent reset, memory cleared, all charts and logs reset", "success")
        
        print("✅ Fresh training started - agent reset, memory cleared")
    
    def _load_model(self, filepath: str) -> None:
        """Load a model from file and sync history to dashboard."""
        try:
            metadata, training_history = self.agent.load(filepath)
            
            # If load returned None, model is incompatible (architecture mismatch)
            if metadata is None and training_history is None:
                if self.web_dashboard:
                    self.web_dashboard.log(f"⚠️  Model incompatible: {os.path.basename(filepath)} - starting fresh training", "warning")
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
                self.q_values = training_history.q_values.copy() if training_history.q_values else []
                self.losses = training_history.losses.copy() if training_history.losses else []
                self.epsilons = training_history.epsilons.copy() if training_history.epsilons else []
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
                    {'path': filepath, 'epsilon': self.agent.epsilon, 'episode': self.current_episode}
                )
                # Update save status to reflect loaded model
                self.web_dashboard.publisher.record_save(
                    filename=os.path.basename(filepath),
                    reason="loaded",
                    episode=self.current_episode,
                    best_score=self.best_score
                )
        except Exception as e:
            if self.web_dashboard:
                self.web_dashboard.log(f"❌ Failed to load model: {str(e)}", "error")
            import traceback
            traceback.print_exc()
    
    def _sync_web_dashboard_history(self, training_history: TrainingHistory, metadata: Optional[Any]) -> None:
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
            publisher.state.win_rate = sum(1 for w in recent_wins if w) / len(recent_wins) if recent_wins else 0.0
        
        # Emit an update to connected clients
        self.web_dashboard.socketio.emit('state_update', publisher.get_snapshot())
    
    def _save_model_as(self, filename: str) -> bool:
        """Save model with custom filename."""
        if not filename.endswith('.pth'):
            filename += '.pth'
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
            # Enable activation capture temporarily
            self.agent.policy_net.capture_activations = True
            
            # Get layer info
            layer_info = self.agent.policy_net.get_layer_info()
            
            # Get Q-values (this forward pass captures activations)
            q_values = self.agent.get_q_values(state)
            
            # Get activations
            raw_activations = self.agent.policy_net.get_activations()
            
            # Get weights (sampled for performance)
            raw_weights = self.agent.policy_net.get_weights()
            
            # Disable activation capture
            self.agent.policy_net.capture_activations = False
            
            # Format activations for JSON (convert numpy arrays to lists, limit neurons)
            max_neurons = 15  # Match the pygame visualizer limit
            activations: dict[str, list[float]] = {}
            for key, act in raw_activations.items():
                if len(act.shape) > 1:
                    act = act[0]  # Take first batch item
                # Normalize and limit to max_neurons
                act_list = act[:min(max_neurons, len(act))].tolist()
                activations[key] = act_list
            
            # Format weights for JSON (sample connections for performance)
            weights: list[list[list[float]]] = []
            for i, w in enumerate(raw_weights):
                if w is not None:
                    # Sample weights: take first 15 rows and first 15 columns
                    sampled_w = w[:min(15, w.shape[0]), :min(15, w.shape[1])]
                    weights.append(sampled_w.tolist())
            
            # Get action labels from game if available
            action_labels = ["LEFT", "STAY", "RIGHT"]  # Default for Breakout
            if hasattr(self.game, 'get_action_labels'):
                action_labels = self.game.get_action_labels()
            
            # Emit to web dashboard (throttling handled by publisher)
            self.web_dashboard.emit_nn_visualization(
                layer_info=layer_info,
                activations=activations,
                q_values=q_values.tolist(),
                selected_action=selected_action,
                weights=weights,
                step=self.agent.steps,
                action_labels=action_labels
            )
        except Exception:
            # Don't crash training on visualization errors - silently ignore
            pass

    def _apply_config(self, config_data: dict) -> None:
        """Apply configuration changes from web dashboard."""
        changes = []
        
        if 'learning_rate' in config_data:
            old_lr = self.config.LEARNING_RATE
            self.config.LEARNING_RATE = config_data['learning_rate']
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = config_data['learning_rate']
            changes.append(f"LR: {old_lr} → {config_data['learning_rate']}")
        
        if 'epsilon' in config_data:
            try:
                eps = float(config_data['epsilon'])
                if not math.isfinite(eps):
                    raise ValueError("Epsilon must be finite (not NaN or Inf)")
                old_eps = self.agent.epsilon
                # Clamp epsilon to valid range with feedback
                clamped_eps = max(self.config.EPSILON_END, min(self.config.EPSILON_START, eps))
                if clamped_eps != eps:
                    print(f"⚠️  Epsilon {eps:.4f} clamped to valid range [{self.config.EPSILON_END}, {self.config.EPSILON_START}]")
                self.agent.epsilon = clamped_eps
                changes.append(f"Epsilon: {old_eps:.4f} → {self.agent.epsilon:.4f}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid epsilon value: {config_data['epsilon']} - {e}")
        
        if 'epsilon_decay' in config_data:
            try:
                decay = float(config_data['epsilon_decay'])
                if not math.isfinite(decay) or decay <= 0 or decay > 1:
                    raise ValueError("Epsilon decay must be finite and in (0, 1]")
                self.config.EPSILON_DECAY = decay
                changes.append(f"Decay: {decay}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid epsilon_decay value: {config_data['epsilon_decay']} - {e}")

        if 'gamma' in config_data:
            try:
                gamma = float(config_data['gamma'])
                if not math.isfinite(gamma) or gamma < 0 or gamma > 1:
                    raise ValueError("Gamma must be finite and in [0, 1]")
                self.config.GAMMA = gamma
                changes.append(f"Gamma: {gamma}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid gamma value: {config_data['gamma']} - {e}")

        if 'batch_size' in config_data:
            try:
                batch_size = int(config_data['batch_size'])
                if batch_size <= 0:
                    raise ValueError("Batch size must be positive")
                if batch_size > self.config.MEMORY_SIZE:
                    raise ValueError(f"Batch size ({batch_size}) cannot exceed memory size ({self.config.MEMORY_SIZE})")
                self.config.BATCH_SIZE = batch_size
                changes.append(f"Batch: {batch_size}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid batch_size value: {config_data['batch_size']} - {e}")

        if 'learn_every' in config_data:
            try:
                learn_every = int(config_data['learn_every'])
                if learn_every <= 0:
                    raise ValueError("Learn every must be positive")
                self.config.LEARN_EVERY = learn_every
                changes.append(f"LearnEvery: {learn_every}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid learn_every value: {config_data['learn_every']} - {e}")

        if 'gradient_steps' in config_data:
            try:
                grad_steps = int(config_data['gradient_steps'])
                if grad_steps <= 0:
                    raise ValueError("Gradient steps must be positive")
                self.config.GRADIENT_STEPS = grad_steps
                changes.append(f"GradSteps: {grad_steps}")
            except (ValueError, TypeError) as e:
                print(f"⚠️  Invalid gradient_steps value: {config_data['gradient_steps']} - {e}")
        
        if self.web_dashboard and changes:
            self.web_dashboard.log(f"⚙️ Config updated: {', '.join(changes)}", "action", config_data)
    
    def _set_performance_mode(self, mode: str) -> None:
        """Set performance mode preset."""
        if mode == 'normal':
            self.config.LEARN_EVERY = 1
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 1
        elif mode == 'fast':
            self.config.LEARN_EVERY = 4
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 1
        elif mode == 'turbo':
            self.config.LEARN_EVERY = 8
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 2
        elif mode == 'ultra':
            # Maximum throughput: less frequent learning, same batch size as turbo
            # learn_every=32 means learning 4x less often than turbo
            self.config.LEARN_EVERY = 32
            self.config.BATCH_SIZE = 128
            self.config.GRADIENT_STEPS = 2
        else:
            print(f"⚠️  Unknown performance mode: {mode}")
            return

        if self.web_dashboard:
            self.web_dashboard.publisher.set_performance_mode(mode)
            self.web_dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
            self.web_dashboard.publisher.state.batch_size = self.config.BATCH_SIZE
            self.web_dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS
            self.web_dashboard.log(
                f"⚡ Performance mode: {mode.upper()} (learn_every={self.config.LEARN_EVERY}, batch={self.config.BATCH_SIZE}, grad_steps={self.config.GRADIENT_STEPS})",
                "action"
            )
        print(f"⚡ Performance mode: {mode.upper()}")

    def _save_and_quit(self) -> None:
        """Save the model and exit the application gracefully."""
        if self.web_dashboard:
            self.web_dashboard.log("💾 Saving model before shutdown...", "warning")

        # Save the model and verify it completed
        save_success = self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="shutdown")

        if save_success:
            if self.web_dashboard:
                self.web_dashboard.log("✅ Model saved. Shutting down...", "success")
            print("\n👋 Save & Quit requested. Model saved. Exiting...")
        else:
            if self.web_dashboard:
                self.web_dashboard.log("⚠️ Save may have failed. Shutting down...", "warning")
            print("\n⚠️ Save & Quit requested. Save may have failed. Exiting...")

        # Flush output buffers to ensure messages are written
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

        # Give time for the save event to propagate to clients
        import time
        time.sleep(0.5)

        # Exit gracefully - use os._exit() because this is called from SocketIO thread
        self.running = False
        import os
        os._exit(0)  # Terminates process from any thread

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
        while config.MAX_EPISODES == 0 or episode < config.MAX_EPISODES:
            self.current_episode = episode
            
            # Handle pause (only if web dashboard is active)
            while self.paused:
                time.sleep(0.1)
            
            state = self.game.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done:
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
                if self.agent.steps % config.TARGET_UPDATE == 0 and self.agent.steps != self.last_target_update_step:
                    self.target_updates += 1
                    self.last_target_update_step = self.agent.steps
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"🎯 Target network updated (#{self.target_updates})", 
                            "metric",
                            {'step': self.agent.steps, 'update_number': self.target_updates}
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
            self.scores.append(info['score'])
            
            # Track wins (all bricks cleared)
            won = info.get('won', False)
            self.wins.append(won)
            
            # Calculate bricks broken
            initial_bricks = config.BRICK_ROWS * config.BRICK_COLS
            bricks_broken = initial_bricks - info.get('bricks_remaining', initial_bricks)
            
            # Update web dashboard metrics (throttled to every 5 episodes for performance)
            # Always emit on: first 10 episodes, new best score, or every 5th episode
            is_new_best = info['score'] > getattr(self, 'best_score', 0)
            should_emit = (
                self.web_dashboard and
                (episode <= 10 or is_new_best or episode % 5 == 0)
            )
            if should_emit:
                avg_loss = self.agent.get_average_loss(100)

                # Calculate average Q-value for current state (was missing from headless)
                q_values = self.agent.get_q_values(state)
                avg_q_value = float(np.mean(q_values))

                self.web_dashboard.emit_metrics(
                    episode=episode,
                    score=info['score'],
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
                    episode_length=episode_steps
                )
                # Update performance settings in dashboard state
                self.web_dashboard.publisher.state.learn_every = config.LEARN_EVERY
                self.web_dashboard.publisher.state.gradient_steps = config.GRADIENT_STEPS
                self.web_dashboard.publisher.state.batch_size = config.BATCH_SIZE

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
                steps_per_sec = steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                eps_per_hour = (episode - start_episode) / elapsed_total * 3600 if elapsed_total > 0 else 0
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0
                
                progress_msg = (f"Ep {episode:5d} | "
                      f"Score: {info['score']:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"⚡ {steps_per_sec:,.0f} steps/s | "
                      f"📊 {eps_per_hour:,.0f} ep/hr")
                
                print(progress_msg)
                
                # Also log to web dashboard console
                if self.web_dashboard:
                    self.web_dashboard.log(progress_msg, "metric")
                
                last_logged_episode = episode
                last_report_time = current_time
                steps_since_report = 0
            
            # Save checkpoints
            if info['score'] > self.best_score:
                self.best_score = info['score']
                self._save_model(f"{self.config.GAME_NAME}_best.pth", save_reason="best", quiet=True)
                if self.web_dashboard:
                    self.web_dashboard.log(f"🏆 New best score: {self.best_score}", "success")
            
            if episode % config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(
                    f"{self.config.GAME_NAME}_ep{episode}.pth",
                    save_reason="periodic",
                    save_replay_buffer=False  # Periodic saves are lightweight
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
        env_episode_rewards = np.zeros(num_envs, dtype=np.float64)  # Use float64 to prevent precision loss
        env_episode_steps = np.zeros(num_envs, dtype=np.int64)
        episodes_completed = 0
        
        # Initialize all environments
        states = self.vec_env.reset().copy()  # Shape: (num_envs, state_size) - copy to avoid aliasing
        
        # Track last completed episode info for reporting
        last_score = 0
        last_info: dict = {}
        last_logged_episode = start_episode - 1  # Track last logged episode to prevent duplicates
        
        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        while config.MAX_EPISODES == 0 or self.current_episode < config.MAX_EPISODES:
            # Handle pause (only if web dashboard is active)
            while self.paused:
                time.sleep(0.1)
            
            # Batch action selection for all environments
            actions, num_explored, num_exploited = self.agent.select_actions_batch(states, training=True)
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
                    score = infos[i].get('score', 0)
                    self.scores.append(score)
                    
                    won = infos[i].get('won', False)
                    self.wins.append(won)
                    
                    level = infos[i].get('level', 1)
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
                        self._save_model(f"{self.config.GAME_NAME}_best.pth", save_reason="best", quiet=True)
                        if self.web_dashboard:
                            self.web_dashboard.log(f"🏆 New best score: {self.best_score}", "success")
                    
                    # Track target updates (for persistence, independent of dashboard)
                    if self.agent.steps > self.last_target_update_step + config.TARGET_UPDATE:
                        self.target_updates += 1
                        self.last_target_update_step = self.agent.steps
                    
                    # Update web dashboard metrics (throttled to every 5 episodes for performance)
                    # Always emit on: first 10 episodes, new best score, or every 5th episode
                    is_new_best = score > self.best_score
                    should_emit_metrics = (
                        self.web_dashboard and
                        (self.current_episode <= 10 or is_new_best or self.current_episode % 5 == 0)
                    )
                    if should_emit_metrics:
                        initial_bricks = config.BRICK_ROWS * config.BRICK_COLS
                        bricks_broken = initial_bricks - infos[i].get('bricks_remaining', initial_bricks)

                        self.web_dashboard.emit_metrics(
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
                            episode_length=int(env_episode_steps[i])
                        )
                        # Update performance settings in dashboard state
                        self.web_dashboard.publisher.state.learn_every = config.LEARN_EVERY
                        self.web_dashboard.publisher.state.gradient_steps = config.GRADIENT_STEPS
                        self.web_dashboard.publisher.state.batch_size = config.BATCH_SIZE

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
                            save_replay_buffer=False  # Periodic saves are lightweight
                        )
                        self._cleanup_old_periodic_saves(keep_last=5)
                    
                    # Run deterministic evaluation periodically
                    if (self.evaluator is not None and 
                        config.EVAL_EVERY > 0 and 
                        self.current_episode % config.EVAL_EVERY == 0 and 
                        self.current_episode > 0):
                        eval_results = self.evaluator.evaluate(
                            num_episodes=config.EVAL_EPISODES,
                            max_steps=config.EVAL_MAX_STEPS,
                            episode_num=self.current_episode
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
                            print(f"\n🚀 PLATEAU DETECTED! Boosting exploration: "
                                  f"ε {old_epsilon:.3f} → {self.agent.epsilon:.3f} "
                                  f"for {config.EVAL_PLATEAU_BOOST_EPISODES} episodes\n")
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"🚀 Exploration boost activated! ε → {self.agent.epsilon:.2f}",
                                    "warning"
                                )
                        
                        # Log to web dashboard if available
                        if self.web_dashboard:
                            plateau_str = " ⚠️ PLATEAU DETECTED" if self.evaluator.is_plateau() else ""
                            self.web_dashboard.log(
                                f"📊 EVAL: {eval_results.mean_score:.0f} avg, "
                                f"max level {eval_results.max_level}, "
                                f"{eval_results.win_rate*100:.0f}% wins{plateau_str}",
                                "info" if not self.evaluator.is_plateau() else "warning"
                            )
            
            # Decay epsilon once per step if any episodes completed
            # (NOT per environment - that would decay too fast with many parallel envs)
            if np.any(dones):
                # Check if exploration boost period has ended
                if (self._exploration_boost_active and 
                    self.current_episode >= self._exploration_boost_end_episode):
                    self._exploration_boost_active = False
                    # Reset epsilon to minimum and let it decay normally
                    self.agent.epsilon = config.EPSILON_END
                    print(f"\n✓ Exploration boost ended. Resuming normal ε={self.agent.epsilon:.3f}\n")
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"✓ Exploration boost ended, ε → {self.agent.epsilon:.3f}",
                            "info"
                        )
                    # Reset plateau counter so we can detect new plateaus
                    if self.evaluator:
                        self.evaluator.evals_since_improvement = 0
                
                # Only decay epsilon if not in boost mode
                if not self._exploration_boost_active:
                    self.agent.decay_epsilon(self.current_episode)
                self.agent.step_scheduler()  # Step learning rate scheduler
            
            # Update states for next iteration (already auto-reset in VecBreakout)
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
            if self.current_episode > last_logged_episode and (should_log_by_episode or should_log_by_time):
                elapsed_total = current_time - self.training_start_time
                steps_per_sec = steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                eps_per_hour = episodes_completed / elapsed_total * 3600 if elapsed_total > 0 else 0
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0
                avg_loss = self.agent.get_average_loss(100)
                avg_q = np.mean(self.q_values[-100:]) if self.q_values else 0.0
                
                # Get level reached from last completed episode
                level_reached = last_info.get('level', 1) if last_info else 1
                
                progress_msg = (f"Ep {self.current_episode:5d} | "
                      f"Score: {last_score:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Q: {avg_q:.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"⚡ {steps_per_sec:,.0f} steps/s")
                
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
        save_replay_buffer: bool = True
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
            best_score=self.best_score
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
            quiet=quiet
        )
        
        # Notify web dashboard about save
        if result is not None and self.web_dashboard:
            self.web_dashboard.publisher.record_save(
                filename=filename,
                reason=save_reason,
                episode=self.current_episode,
                best_score=self.best_score
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
        import glob
        import re
        
        model_dir = self.config.GAME_MODEL_DIR
        game_name = self.config.GAME_NAME
        
        # Find all periodic saves (e.g., space_invaders_ep100.pth, space_invaders_ep200.pth)
        pattern = os.path.join(model_dir, f"{game_name}_ep*.pth")
        periodic_saves = glob.glob(pattern)
        
        if len(periodic_saves) <= keep_last:
            return  # Nothing to clean up
        
        # Extract episode numbers and sort
        def get_episode_num(path: str) -> int:
            match = re.search(r'_ep(\d+)\.pth$', path)
            return int(match.group(1)) if match else 0
        
        # Sort by episode number (oldest first)
        periodic_saves.sort(key=get_episode_num)
        
        # Delete all but the last `keep_last` saves
        to_delete = periodic_saves[:-keep_last]
        
        for filepath in to_delete:
            try:
                os.remove(filepath)
                if self.web_dashboard:
                    self.web_dashboard.log(
                        f"🗑️ Cleaned up old checkpoint: {os.path.basename(filepath)}",
                        "info"
                    )
            except Exception as e:
                print(f"⚠️ Could not delete {filepath}: {e}")


def parse_args():
    """Parse command line arguments."""
    # Import game registry for --game choices
    from src.game import list_games
    available_games = list_games()

    parser = argparse.ArgumentParser(
        description="DQN Game AI - Train neural networks to play classic arcade games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
EXAMPLES
========

Getting Started:
    python main.py                    Train with visual display (shows game selection)
    python main.py --human            Play a game yourself to test it
    python main.py --game pong        Train a specific game directly

Fast Training (Recommended):
    python main.py --headless --turbo          ~5000 steps/sec on M4 Mac
    python main.py --headless --turbo --web    With web dashboard at localhost:5000
    python main.py --headless --vec-envs 8     Parallel training (~12,000 steps/sec)

Watch Trained Agent:
    python main.py --play --model models/pong_best.pth

Model Management:
    python main.py --list-models               Show all saved models
    python main.py --inspect models/best.pth   Inspect model metadata

AVAILABLE GAMES: {', '.join(available_games)}

TIPS
====
- Use --headless for 10x faster training (no display overhead)
- Add --turbo for optimized batch settings (~4x faster)
- Add --vec-envs 8 for parallel environments (~2-3x faster)
- Use --web to monitor training in browser at http://localhost:5000
- Press Ctrl+C to gracefully stop training (model auto-saves)
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--play', action='store_true',
        help='Play mode: watch trained agent without training'
    )
    mode_group.add_argument(
        '--human', action='store_true',
        help='Human mode: play the game yourself'
    )
    mode_group.add_argument(
        '--headless', action='store_true',
        help='Headless training: no visualization (faster)'
    )
    mode_group.add_argument(
        '--inspect', type=str, metavar='MODEL_PATH',
        help='Inspect a model file and show its metadata'
    )
    mode_group.add_argument(
        '--list-models', action='store_true',
        help='List all saved models with their metadata'
    )
    
    # Game selection
    parser.add_argument(
        '--game', type=str, default=None,
        choices=available_games,
        help=f'Game to train/play. If not specified, shows game selection. Available: {", ".join(available_games)}'
    )
    parser.add_argument(
        '--menu', action='store_true',
        help='Show game selection menu on launch (interactive game picker)'
    )
    
    # Model options
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to model file to load'
    )
    
    # Training parameters
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of training episodes (default: unlimited, trains until stopped)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda', 'mps'],
        default=None, help='Device to use for training'
    )
    
    # Web dashboard
    parser.add_argument(
        '--web', action='store_true',
        help='Enable web dashboard for remote monitoring (http://localhost:5000)'
    )
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port for web dashboard (default: 5000)'
    )
    
    # Performance tuning
    parser.add_argument(
        '--learn-every', type=int, default=None,
        help='Learn every N steps (default: 1, try 4 for ~4x speedup)'
    )
    parser.add_argument(
        '--gradient-steps', type=int, default=None,
        help='Number of gradient updates per learning call (default: 1)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Training batch size (default: 128, try 256 for M4)'
    )
    parser.add_argument(
        '--turbo', action='store_true',
        help='Turbo mode preset: learn-every 8, batch 128, 2 grad steps (~5000 steps/sec on M4)'
    )
    parser.add_argument(
        '--vec-envs', type=int, default=1,
        help='Number of parallel environments for vectorized training (default: 1, try 8 for ~3x speedup)'
    )
    parser.add_argument(
        '--torch-compile', action='store_true',
        help='Enable torch.compile() for ~20-50%% speedup (PyTorch 2.0+)'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU (faster than MPS for small models on M4)'
    )
    
    # Other options
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def inspect_model(filepath: str) -> None:
    """Inspect a model file and display its metadata."""
    from src.ai.agent import Agent
    
    info = Agent.inspect_model(filepath)
    if not info:
        return
    
    print("\n" + "=" * 60)
    print(f"🔍 Model Inspection: {info['filename']}")
    print("=" * 60)
    print(f"   File Size: {info['file_size_mb']:.2f} MB")
    print(f"   Modified:  {info['file_modified']}")
    print(f"\n   Steps:     {info['steps']:,}" if isinstance(info['steps'], int) else f"\n   Steps:     {info['steps']}")
    print(f"   Epsilon:   {info['epsilon']:.4f}" if isinstance(info['epsilon'], float) else f"   Epsilon:   {info['epsilon']}")
    print(f"   State Size: {info['state_size']}")
    print(f"   Action Size: {info['action_size']}")
    
    if info['has_metadata'] and info['metadata']:
        meta = info['metadata']
        print(f"\n   📊 Training Metadata:")
        print(f"   ─────────────────────")
        print(f"   Save Reason:    {meta.get('save_reason', 'unknown')}")
        print(f"   Episode:        {meta.get('episode', 'unknown'):,}" if isinstance(meta.get('episode'), int) else f"   Episode:        {meta.get('episode', 'unknown')}")
        print(f"   Best Score:     {meta.get('best_score', 'unknown')}")
        print(f"   Avg Score(100): {meta.get('avg_score_last_100', 0):.1f}")
        print(f"   Win Rate:       {meta.get('win_rate', 0)*100:.1f}%")
        print(f"   Avg Loss:       {meta.get('avg_loss', 0):.4f}")
        
        training_time = meta.get('total_training_time_seconds', 0)
        if training_time > 0:
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            print(f"   Training Time:  {hours}h {minutes}m")
        
        print(f"\n   ⚙️ Config Snapshot:")
        print(f"   ─────────────────────")
        print(f"   Learning Rate:  {meta.get('learning_rate', 'unknown')}")
        print(f"   Gamma:          {meta.get('gamma', 'unknown')}")
        print(f"   Batch Size:     {meta.get('batch_size', 'unknown')}")
        print(f"   Hidden Layers:  {meta.get('hidden_layers', 'unknown')}")
        print(f"   Dueling DQN:    {meta.get('use_dueling', 'unknown')}")
    else:
        print(f"\n   ⚠️ No detailed metadata (legacy save format)")
    
    print("=" * 60 + "\n")


def list_models(model_dir: str = 'models') -> None:
    """List all model files in the models directory."""
    from src.ai.agent import Agent
    from datetime import datetime
    
    models = Agent.list_models(model_dir)
    
    if not models:
        print(f"\n❌ No model files found in '{model_dir}/'")
        return
    
    print("\n" + "=" * 80)
    print(f"📁 Saved Models in '{model_dir}/' ({len(models)} files)")
    print("=" * 80)
    print(f"{'Filename':<35} {'Episode':>8} {'Steps':>12} {'Best':>6} {'Epsilon':>8} {'Size':>8}")
    print("-" * 80)
    
    for model in models:
        filename = model['filename'][:33] + '..' if len(model['filename']) > 35 else model['filename']
        
        # Get metadata if available
        if model['has_metadata'] and model['metadata']:
            meta = model['metadata']
            episode = meta.get('episode', '?')
            steps = meta.get('total_steps', model.get('steps', '?'))
            best = meta.get('best_score', '?')
            epsilon = meta.get('epsilon', model.get('epsilon', '?'))
        else:
            episode = '?'
            steps = model.get('steps', '?')
            best = '?'
            epsilon = model.get('epsilon', '?')
        
        size_mb = f"{model['file_size_mb']:.1f}MB"
        
        # Format values
        ep_str = f"{episode:,}" if isinstance(episode, int) else str(episode)
        steps_str = f"{steps:,}" if isinstance(steps, int) else str(steps)
        best_str = str(best)
        eps_str = f"{epsilon:.3f}" if isinstance(epsilon, float) else str(epsilon)
        
        print(f"{filename:<35} {ep_str:>8} {steps_str:>12} {best_str:>6} {eps_str:>8} {size_mb:>8}")
    
    print("=" * 80)
    print(f"\nUse --inspect <path> to see detailed info about a specific model.\n")


def restart_with_game(game_name: str, args: argparse.Namespace) -> None:
    """Restart the current process with a different game.
    
    This spawns a new process with the specified game and exits the current one.
    Using subprocess + exit instead of os.execv ensures proper port release.
    """
    import subprocess
    import sys
    import time
    
    # Build new command preserving current args
    new_args = [sys.executable, sys.argv[0]]
    new_args.extend(['--game', game_name])
    
    if args.headless:
        new_args.append('--headless')
    if args.web:
        new_args.extend(['--web', '--port', str(args.port)])
    if hasattr(args, 'turbo') and args.turbo:
        new_args.append('--turbo')
    if hasattr(args, 'vec_envs') and args.vec_envs and args.vec_envs > 1:
        new_args.extend(['--vec-envs', str(args.vec_envs)])
    if args.episodes:
        new_args.extend(['--episodes', str(args.episodes)])
    if hasattr(args, 'cpu') and args.cpu:
        new_args.append('--cpu')
    
    print(f"\n🔄 Restarting with {game_name}...")
    print(f"🚀 Command: {' '.join(new_args)}\n")
    
    # Small delay so web client can receive the message
    time.sleep(0.3)
    
    # Replace this process with the new one (port will be released, Ctrl+C works)
    os.execv(sys.executable, new_args)


def run_web_mode(config: Config, args: argparse.Namespace) -> None:
    """
    Run with web interface for game selection and monitoring.
    Works for both headless and visual modes.

    This function handles all --web scenarios:
    - If game specified: starts immediately with web dashboard
    - If no game: shows web launcher for game selection, then starts training
    """
    import threading

    import socket

    try:
        from src.web.server import WebDashboard
    except ImportError:
        print("❌ Web dashboard requires Flask. Install with:")
        print("   pip install flask flask-socketio eventlet")
        return

    # Find available port (auto-increment if busy)
    port = args.port
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
            break  # Port is available
        except OSError:
            if attempt == 0:
                print(f"⚠️  Port {port} is busy, finding available port...")
            port += 1
    else:
        print(f"❌ Could not find available port after {max_attempts} attempts")
        return

    if port != args.port:
        print(f"✓ Using port {port}")

    # Track game and mode selection
    selected_game = None
    selected_mode = 'ai'  # 'ai' or 'human'
    selection_event = threading.Event()

    def on_game_selected(game_name: str, mode: str) -> None:
        """Called when user selects a game from web UI."""
        nonlocal selected_game, selected_mode
        selected_game = game_name
        selected_mode = mode
        selection_event.set()

    # Start web dashboard in launcher mode
    dashboard = WebDashboard(config, port=port, launcher_mode=True)
    dashboard.on_game_selected_callback = on_game_selected  # Set callback BEFORE start
    dashboard.start()

    # Wait for server to print its startup message
    import time
    time.sleep(0.3)

    # If game already specified, skip selection
    if args.game:
        selected_game = args.game
        # Check if human mode was specified via CLI
        if hasattr(args, 'human') and args.human:
            selected_mode = 'human'
        print(f"\n🎮 Starting {selected_game}...")
    else:
        # Wait for game selection from web UI
        print("\n⏳ Open browser to select a game...")
        print("   Press Ctrl+C to exit\n")

        try:
            # Wait for game selection or keyboard interrupt
            while not selection_event.is_set():
                selection_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            print("\n\n👋 Closed by user")
            dashboard.stop()
            return

        if not selected_game:
            print("No game selected. Exiting.")
            dashboard.stop()
            return

        mode_text = "🎮 Playing" if selected_mode == 'human' else "🤖 Training"
        print(f"\n{mode_text} {selected_game}...")

    # Update config and args based on selection
    config.GAME_NAME = selected_game
    args.game = selected_game

    # Set mode based on web selection
    if selected_mode == 'human':
        args.human = True

    dashboard.launcher_mode = False
    dashboard.socketio.emit('game_ready', {'game': selected_game, 'mode': selected_mode})

    # Game loop - supports returning to menu
    while True:
        return_to_menu = False

        # Start appropriate trainer based on mode
        try:
            if args.headless and selected_mode != 'human':
                # Headless training (only for AI mode)
                trainer = HeadlessTrainer(config, args, existing_dashboard=dashboard)
                trainer.train()
            else:
                # Visual mode (required for human play)
                app = GameApp(config, args, existing_dashboard=dashboard)

                # Run appropriate mode
                if selected_mode == 'human' or args.human:
                    app.run_human_mode()
                elif args.play:
                    app.run_play_mode()
                else:
                    app.run_training()

                # Check if user wants to return to game selector
                return_to_menu = app.return_to_menu

        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            if args.headless:
                if 'trainer' in locals():
                    trainer._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
            else:
                if 'app' in locals():
                    app._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
                    if app.web_dashboard:
                        app.web_dashboard.log("⛔ Training interrupted by user", "warning")
            break  # Exit on keyboard interrupt

        if not return_to_menu:
            break  # Normal exit

        # Return to game selector
        print("\n🏠 Returning to game selector...")

        # Reset selection state
        selected_game = None
        selected_mode = 'ai'
        selection_event.clear()

        # Switch dashboard back to launcher mode
        dashboard.launcher_mode = True

        print("\n⏳ Open browser to select a game...")
        print("   Press Ctrl+C to exit\n")

        try:
            # Wait for new game selection
            while not selection_event.is_set():
                selection_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            print("\n\n👋 Closed by user")
            break

        if not selected_game:
            print("No game selected. Exiting.")
            break

        # Update config for new game
        mode_text = "🎮 Playing" if selected_mode == 'human' else "🤖 Training"
        print(f"\n{mode_text} {selected_game}...")

        config.GAME_NAME = selected_game
        args.game = selected_game
        args.human = (selected_mode == 'human')

        dashboard.launcher_mode = False
        dashboard.socketio.emit('game_ready', {'game': selected_game, 'mode': selected_mode})

    # Clean up resources
    if dashboard:
        dashboard.stop()
    if not args.headless:
        pygame.quit()
    print("\n👋 Done")


def run_web_launcher(config: Config, args: argparse.Namespace) -> None:
    """Run web-based game launcher mode.

    This mode starts a web server without any training, allowing the user
    to select a game from the browser. When a game is selected, training
    starts in the SAME process (no restart needed).
    """
    import threading
    
    try:
        from src.web.server import WebDashboard
    except ImportError:
        print("❌ Web dashboard requires Flask. Install with:")
        print("   pip install flask flask-socketio eventlet")
        return
    
    print("\n" + "=" * 60)
    print("🎮 NEURAL NETWORK AI - GAME LAUNCHER")
    print("=" * 60)
    print(f"\n🌐 Open http://localhost:{args.port} to select a game\n")
    
    # Track selected game
    selected_game = None
    selection_event = threading.Event()
    
    def on_game_selected(game_name: str) -> None:
        """Called when user selects a game from web UI."""
        nonlocal selected_game
        selected_game = game_name
        selection_event.set()
    
    # Create web dashboard in launcher mode
    dashboard = WebDashboard(config, port=args.port, launcher_mode=True)
    dashboard.on_game_selected_callback = on_game_selected
    
    # Start web server in background thread
    server_thread = threading.Thread(
        target=lambda: dashboard.socketio.run(
            dashboard.app,
            host='0.0.0.0',
            port=args.port,
            debug=False,
            use_reloader=False,
            log_output=False,
            allow_unsafe_werkzeug=True
        ),
        daemon=True
    )
    server_thread.start()
    
    print("⏳ Waiting for game selection from web UI...")
    print("   Press Ctrl+C to exit\n")
    
    try:
        # Wait for game selection or keyboard interrupt
        while not selection_event.is_set():
            selection_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        print("\n\n👋 Launcher closed by user")
        dashboard.stop()
        return
    
    if selected_game:
        print(f"\n🎮 Starting {selected_game}...")
        
        # Update config with selected game
        config.GAME_NAME = selected_game
        args.game = selected_game
        
        # Switch dashboard out of launcher mode (same server, same port!)
        dashboard.launcher_mode = False
        
        # Notify browser to switch to dashboard view
        dashboard.socketio.emit('game_ready', {'game': selected_game})
        
        # Now run HeadlessTrainer with the same dashboard (no process restart!)
        print(f"🚀 Training {selected_game} in-place (same server)\n")
        
        # Create trainer and run - reuse the existing dashboard
        trainer = HeadlessTrainer(config, args, existing_dashboard=dashboard)
        
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            trainer._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
        finally:
            # Clean up web dashboard
            if dashboard:
                dashboard.stop()
            print("\n👋 Training complete")


def terminal_game_selector() -> Optional[str]:
    """Terminal-based game selector for headless mode.

    Returns:
        Selected game name, or None if cancelled
    """
    available = list_games()

    print("\n" + "=" * 60)
    print("   SELECT A GAME TO TRAIN")
    print("=" * 60)

    # Difficulty color codes (for terminals that support it)
    difficulty_indicators = {
        'Easy': '(Easy)',
        'Medium': '(Medium)',
        'Hard': '(Hard)'
    }

    for i, game_name in enumerate(available, 1):
        game_info = get_game_info(game_name)
        if game_info:
            difficulty = game_info.get('difficulty', 'Medium')
            difficulty_str = difficulty_indicators.get(difficulty, '')
            actions = game_info.get('actions', [])
            action_str = ', '.join(actions) if actions else 'N/A'

            print(f"\n  [{i}] {game_info['icon']} {game_info['name']} {difficulty_str}")
            print(f"      {game_info['description']}")
            print(f"      Actions: {action_str}")
        else:
            print(f"\n  [{i}] {game_name}")

    print(f"\n  [0] Exit")
    print("\n" + "=" * 60)

    while True:
        try:
            choice = input("\nEnter number (1-{0}): ".format(len(available)))
            if choice.strip() == '0':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx]
            print(f"Please enter a number between 1 and {len(available)}")
        except ValueError:
            print("Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print()
            return None


def print_startup_banner() -> None:
    """Print a welcome banner for the application."""
    print()
    print("=" * 60)
    print("       DQN GAME AI - Deep Q-Learning Trainer")
    print("=" * 60)
    print("   Train neural networks to play classic arcade games!")
    print()
    print("   Quick Start:")
    print("   - python main.py              # Visual training (default)")
    print("   - python main.py --human      # Play a game yourself")
    print("   - python main.py --headless   # Fast training (no display)")
    print("   - python main.py --help       # See all options")
    print("=" * 60)


def main():
    """Main entry point."""
    # Show banner for non-help invocations
    if '--help' not in sys.argv and '-h' not in sys.argv:
        print_startup_banner()

    args = parse_args()

    # Handle --inspect command (no pygame needed)
    if args.inspect:
        inspect_model(args.inspect)
        return
    
    # Handle --list-models command (no pygame needed)
    if args.list_models:
        list_models()
        return
    
    # Load config
    config = Config()

    # Web mode: use web interface for everything
    if hasattr(args, 'web') and args.web and WEB_AVAILABLE:
        run_web_mode(config, args)
        return

    # Show game selection menu if requested OR if no game specified (visual mode only)
    show_menu = (hasattr(args, 'menu') and args.menu) or (args.game is None and not args.headless)
    
    if show_menu:
        pygame.init()
        menu_screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("🧠 Neural Network AI - Game Selection")
        menu_clock = pygame.time.Clock()
        
        menu = GameMenu(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        selected_game = menu.run(menu_screen, menu_clock)
        
        if selected_game is None:
            print("No game selected. Exiting.")
            pygame.quit()
            return
        
        args.game = selected_game
        pygame.quit()  # Quit and reinitialize for proper game window
        print(f"🎮 Selected: {selected_game}")
    
    # Set game from CLI argument
    if hasattr(args, 'game') and args.game:
        config.GAME_NAME = args.game
        game_info = get_game_info(args.game)
        if game_info:
            print(f"🎮 Game: {game_info['icon']} {game_info['name']}")
    
    # Force CPU if specified (faster for small models on M4)
    if hasattr(args, 'cpu') and args.cpu:
        config.FORCE_CPU = True
        print("💻 CPU mode: Using CPU (faster for small models on M4)")
    
    # Set seed if specified
    if args.seed:
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)
        config.SEED = args.seed
    
    # Handle headless mode separately (no pygame)
    if args.headless:
        # If no game specified and web mode, run in launcher mode
        if args.game is None and args.web:
            run_web_launcher(config, args)
            return

        # If no game specified, show terminal game selector
        if args.game is None:
            selected = terminal_game_selector()
            if selected is None:
                print("No game selected. Exiting.")
                return
            args.game = selected
            config.GAME_NAME = selected
            game_info = get_game_info(selected)
            if game_info:
                print(f"🎮 Selected: {game_info['icon']} {game_info['name']}")
            else:
                print(f"🎮 Selected: {selected}")
        
        trainer = HeadlessTrainer(config, args)
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            trainer._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
        finally:
            # Clean up web dashboard if running
            if trainer.web_dashboard:
                trainer.web_dashboard.stop()
        return
    
    # Apply CLI overrides to config for visualized mode
    if args.learn_every:
        config.LEARN_EVERY = args.learn_every
    if args.gradient_steps:
        config.GRADIENT_STEPS = args.gradient_steps
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.torch_compile:
        config.USE_TORCH_COMPILE = True
    
    # Apply turbo preset - optimized for M4 CPU based on benchmarks
    if args.turbo:
        config.LEARN_EVERY = 8
        config.BATCH_SIZE = 128
        config.GRADIENT_STEPS = 2
        config.USE_TORCH_COMPILE = False
        config.FORCE_CPU = True
        print("🚀 Turbo mode: CPU, B=128, LE=8, GS=2 (~5000 steps/sec on M4)")
    
    # Create application (with pygame) and run - supports returning to menu
    app = None
    while True:
        try:
            app = GameApp(config, args)

            # Run appropriate mode
            if args.human:
                app.run_human_mode()
            elif args.play:
                app.run_play_mode()
            else:
                app.run_training()

            # Check if user wants to return to game selector
            if not app.return_to_menu:
                break  # Normal exit

            # Return to game selector
            print("\n🏠 Returning to game selector...")
            pygame.quit()  # Close current window

            selected = terminal_game_selector()
            if not selected:
                print("No game selected. Exiting.")
                break

            # Update config for new game
            config.GAME_NAME = selected
            args.game = selected
            print(f"\n🎮 Starting {selected}...")

            # Reinitialize pygame for new game
            pygame.init()

        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            if app:
                app._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
                if app.web_dashboard:
                    app.web_dashboard.log("⛔ Training interrupted by user", "warning")
            break

    # Clean up resources
    if app and app.web_dashboard:
        app.web_dashboard.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
