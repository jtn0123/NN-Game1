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
    
    # TURBO MODE: Maximum speed training (~4x faster)
    python main.py --headless --turbo --episodes 5000
    
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
    --turbo           Preset: learn-every=4, batch=256, torch.compile
    --learn-every N   Learn every N steps (default: 1, try 4 for ~4x speedup)
    --batch-size N    Training batch size (default: 128, try 256 for M4)
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
from typing import Optional, Callable, Any, Type

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.game.breakout import Breakout
from src.ai.agent import Agent, TrainingHistory
from src.ai.trainer import Trainer
from src.visualizer.nn_visualizer import NeuralNetVisualizer
from src.visualizer.dashboard import Dashboard

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
    
    def __init__(self, config: Config, args: argparse.Namespace):
        """
        Initialize the application.
        
        Args:
            config: Configuration object
            args: Command line arguments
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
        pygame.display.set_caption("🧠 Neural Network AI - Atari Breakout")
        
        # Calculate window size
        # Layout: Game (800x600) | Neural Network Viz (300) | padding
        # The game always renders at its fixed size in the top-left
        self.game_width = config.SCREEN_WIDTH   # 800 - for reference
        self.game_height = config.SCREEN_HEIGHT  # 600 - for reference
        self.viz_width = 320
        self.dashboard_height = 190
        
        # Window size to fit all components
        self.window_width = config.SCREEN_WIDTH + self.viz_width + 25
        self.window_height = config.SCREEN_HEIGHT + self.dashboard_height + 25
        
        # Minimum window dimensions (must fit game + minimal viz/dashboard)
        self.min_window_width = config.SCREEN_WIDTH + 300
        self.min_window_height = config.SCREEN_HEIGHT + 160
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        
        # Create game
        self.game = Breakout(config)
        
        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config
        )
        
        # Load model if specified (basic load - full history restored after dashboard init)
        self._initial_model_path = args.model if args.model and os.path.exists(args.model) else None
        if self._initial_model_path:
            # Basic load to restore agent state, ignore history return for now
            self.agent.load(self._initial_model_path, quiet=True)
        
        # Create visualizations - positioned relative to the fixed game size
        self.nn_visualizer = NeuralNetVisualizer(
            config=config,
            x=config.SCREEN_WIDTH + 15,  # Right of game
            y=10,
            width=self.viz_width,
            height=config.SCREEN_HEIGHT - 20  # Same height as game area
        )
        
        self.dashboard = Dashboard(
            config=config,
            x=10,
            y=config.SCREEN_HEIGHT + 15,  # Below game
            width=self.window_width - 20,
            height=self.dashboard_height
        )
        
        # Training state
        self.episode = 0
        self.total_reward = 0.0
        self.steps = 0
        self.paused = False
        self.running = True
        self.game_speed = 1.0  # Speed multiplier
        
        # Extended training metrics
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.episode_start_time = time.time()
        self.target_updates = 0
        self.last_target_update_step = 0
        
        # Training start time for total training time tracking
        self.training_start_time = time.time()
        
        # Track recent scores for save metadata
        self.recent_scores: list[int] = []
        self.best_score_ever = 0
        
        # Full training history for save/restore (allows dashboard restoration)
        self.training_history_scores: list[int] = []
        self.training_history_rewards: list[float] = []
        self.training_history_steps: list[int] = []
        self.training_history_epsilons: list[float] = []
        self.training_history_bricks: list[int] = []
        self.training_history_wins: list[bool] = []
        
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
        if hasattr(args, 'web') and args.web and WEB_AVAILABLE and WebDashboard is not None:
            self.web_dashboard = WebDashboard(config, port=args.port)
            self.web_dashboard.on_pause_callback = self._toggle_pause
            self.web_dashboard.on_save_callback = lambda: self._save_model("breakout_web_save.pth", save_reason="manual")
            self.web_dashboard.on_save_as_callback = self._save_model_as
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.on_reset_callback = self._reset_episode
            self.web_dashboard.on_load_model_callback = self._load_model
            self.web_dashboard.on_config_change_callback = self._apply_config
            self.web_dashboard.on_performance_mode_callback = self._set_performance_mode
            self.web_dashboard.start()
            
            # Send system info to dashboard
            self._send_system_info()
            
            # Log startup info
            self._log_startup_info()
        
        # Restore training history from initial model load (now that dashboard is ready)
        if self._initial_model_path:
            self._restore_training_history(self._initial_model_path)
    
    def _restore_training_history(self, filepath: str) -> None:
        """Restore training history from a saved model (called after dashboard is ready)."""
        try:
            import torch
            checkpoint = torch.load(filepath, map_location=self.config.DEVICE, weights_only=False)
            
            if 'training_history' in checkpoint:
                history_data = checkpoint['training_history']
                training_history = TrainingHistory.from_dict(history_data)
                
                if len(training_history.scores) > 0:
                    # Restore internal tracking
                    self.training_history_scores = training_history.scores.copy()
                    self.training_history_rewards = training_history.rewards.copy()
                    self.training_history_steps = training_history.steps.copy()
                    self.training_history_epsilons = training_history.epsilons.copy()
                    self.training_history_bricks = training_history.bricks.copy()
                    self.training_history_wins = training_history.wins.copy()
                    self.recent_scores = training_history.scores[-1000:].copy()
                    
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
                self.episode = metadata.get('episode', len(self.training_history_scores))
                self.best_score_ever = metadata.get('best_score', 0)
            elif self.training_history_scores:
                self.episode = len(self.training_history_scores)
                self.best_score_ever = max(self.training_history_scores)
                
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
        self.web_dashboard.log(f"Target episodes: {self.config.MAX_EPISODES}", "info")
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
    
    def _load_model(self, filepath: str) -> None:
        """Load a model from file and restore training history."""
        try:
            metadata, training_history = self.agent.load(filepath)
            
            # Restore training history if available
            if training_history and len(training_history.scores) > 0:
                self.training_history_scores = training_history.scores.copy()
                self.training_history_rewards = training_history.rewards.copy()
                self.training_history_steps = training_history.steps.copy()
                self.training_history_epsilons = training_history.epsilons.copy()
                self.training_history_bricks = training_history.bricks.copy()
                self.training_history_wins = training_history.wins.copy()
                self.recent_scores = training_history.scores[-1000:].copy()
                
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
                else:
                    self.episode = len(training_history.scores)
                    self.best_score_ever = max(training_history.scores) if training_history.scores else 0
                
                history_msg = f" ({len(training_history.scores)} episodes restored)"
            else:
                history_msg = " (no history)"
                # Still restore episode counter from metadata if available
                if metadata:
                    self.episode = metadata.episode
                    self.best_score_ever = metadata.best_score
            
            if self.web_dashboard:
                self.web_dashboard.log(f"📂 Loaded model: {os.path.basename(filepath)}{history_msg}", "success", {
                    'path': filepath,
                    'epsilon': self.agent.epsilon,
                    'steps': self.agent.steps
                })
        except Exception as e:
            if self.web_dashboard:
                self.web_dashboard.log(f"❌ Failed to load model: {str(e)}", "error")
    
    def _apply_config(self, config_data: dict) -> None:
        """Apply configuration changes from web dashboard."""
        changes = []
        
        if 'learning_rate' in config_data:
            old_lr = self.config.LEARNING_RATE
            self.config.LEARNING_RATE = config_data['learning_rate']
            # Update optimizer learning rate
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = config_data['learning_rate']
            changes.append(f"LR: {old_lr} → {config_data['learning_rate']}")
        
        if 'epsilon' in config_data:
            old_eps = self.agent.epsilon
            self.agent.epsilon = config_data['epsilon']
            changes.append(f"Epsilon: {old_eps:.4f} → {config_data['epsilon']:.4f}")
        
        if 'epsilon_decay' in config_data:
            self.config.EPSILON_DECAY = config_data['epsilon_decay']
            changes.append(f"Decay: {config_data['epsilon_decay']}")
        
        if 'gamma' in config_data:
            self.config.GAMMA = config_data['gamma']
            changes.append(f"Gamma: {config_data['gamma']}")
        
        if 'batch_size' in config_data:
            self.config.BATCH_SIZE = config_data['batch_size']
            changes.append(f"Batch: {config_data['batch_size']}")
        
        if 'learn_every' in config_data:
            self.config.LEARN_EVERY = config_data['learn_every']
            changes.append(f"LearnEvery: {config_data['learn_every']}")
        
        if 'gradient_steps' in config_data:
            self.config.GRADIENT_STEPS = config_data['gradient_steps']
            changes.append(f"GradSteps: {config_data['gradient_steps']}")
        
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
            self.config.LEARN_EVERY = 4
            self.config.BATCH_SIZE = 256
            self.config.GRADIENT_STEPS = 1
        
        if self.web_dashboard:
            self.web_dashboard.publisher.set_performance_mode(mode)
            self.web_dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
            self.web_dashboard.publisher.state.batch_size = self.config.BATCH_SIZE
            self.web_dashboard.log(
                f"⚡ Performance mode: {mode.upper()} (learn_every={self.config.LEARN_EVERY}, batch={self.config.BATCH_SIZE})",
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
        game_render_width = self.config.SCREEN_WIDTH   # 800
        game_render_height = self.config.SCREEN_HEIGHT  # 600
        
        # Calculate available space for visualizer (to the right of game)
        # and dashboard (below game)
        available_viz_width = new_width - game_render_width - 30  # 30 for margins
        available_dashboard_height = new_height - game_render_height - 30
        
        # Viz width: use available space but cap it
        self.viz_width = max(280, min(450, available_viz_width))
        
        # Dashboard height: use available space but cap it  
        self.dashboard_height = max(150, min(300, available_dashboard_height))
        
        # Game display area stays fixed
        self.game_width = game_render_width
        self.game_height = game_render_height
        
        # Update neural network visualizer position and size
        # Position it to the right of the game with some margin
        self.nn_visualizer.x = game_render_width + 15
        self.nn_visualizer.y = 10
        self.nn_visualizer.width = self.viz_width
        self.nn_visualizer.height = game_render_height - 20  # Same height as game
        # Clear cached positions so they get recalculated
        self.nn_visualizer._cached_positions = None
        self.nn_visualizer._cached_layer_info = None
        
        # Update dashboard position and size
        # Position it below the game spanning the full width
        self.dashboard.x = 10
        self.dashboard.y = game_render_height + 15
        self.dashboard.width = new_width - 20
        self.dashboard.height = self.dashboard_height
    
    # Speed presets for clean stepping
    SPEED_PRESETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
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
            if preset > self.game_speed:
                self._set_speed(preset, force_log=True)
                return
        # Already at max
        self._set_speed(self.SPEED_PRESETS[-1], force_log=True)
    
    def _speed_down(self) -> None:
        """Decrease speed to previous preset."""
        for preset in reversed(self.SPEED_PRESETS):
            if preset < self.game_speed:
                self._set_speed(preset, force_log=True)
                return
        # Already at min
        self._set_speed(self.SPEED_PRESETS[0], force_log=True)
    
    def run_human_mode(self) -> None:
        """Run in human play mode for testing the game."""
        print("\n🎮 Human Play Mode")
        print("   Use LEFT/RIGHT arrow keys to move paddle")
        print("   Press R to reset, Q to quit\n")
        
        state = self.game.reset()
        
        while self.running:
            self._handle_events()
            
            # Get keyboard input
            keys = pygame.key.get_pressed()
            action = 1  # STAY
            if keys[pygame.K_LEFT]:
                action = 0  # LEFT
            elif keys[pygame.K_RIGHT]:
                action = 2  # RIGHT
            
            # Step game
            state, reward, done, info = self.game.step(action)
            
            if done:
                print(f"   Game Over! Score: {info['score']}")
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
        info: dict = {'score': 0, 'bricks_remaining': 50}  # Default info for paused state
        
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
                        bricks_broken=50-info.get('bricks_remaining', 50)
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
        print(f"   Episodes:       {self.config.MAX_EPISODES}")
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
        info: dict = {}
        
        # Speed slider directly controls training intensity
        # At 1x: 1 training step per render (real-time, ~60 steps/sec)
        # At 100x: 100 training steps per render (~6000 steps/sec)
        # At 1000x: 1000 training steps per render (~60000 steps/sec!)
        
        # Track step time for performance logging
        avg_step_time = 0.001
        step_time_samples: list[float] = []
        
        while self.running and self.episode < self.config.MAX_EPISODES:
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
                    
                    # Track exploration vs exploitation
                    if np.random.random() < self.agent.epsilon:
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
                        
                        # Decay epsilon
                        old_epsilon = self.agent.epsilon
                        self.agent.decay_epsilon()
                        
                        # Calculate average Q-value for current state
                        q_values = self.agent.get_q_values(state)
                        avg_q_value = float(np.mean(q_values))
                        
                        # Update dashboard
                        self.dashboard.update(
                            self.episode,
                            info['score'],
                            self.agent.epsilon,
                            self.agent.get_average_loss(100),
                            bricks_broken=50-info.get('bricks_remaining', 50),
                            won=info.get('won', False),
                            reward=episode_reward
                        )
                        
                        # Update web dashboard if enabled
                        if self.web_dashboard:
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
                            print(f"Episode {self.episode:5d} | "
                                  f"Score: {info['score']:4d} | "
                                  f"Avg: {avg_score:6.1f} | "
                                  f"ε: {self.agent.epsilon:.3f} | "
                                  f"Steps: {episode_steps:5d} | "
                                  f"Time: {episode_duration:.1f}s")
                        
                        # Track scores for metadata
                        self.recent_scores.append(info['score'])
                        if len(self.recent_scores) > 1000:
                            self.recent_scores = self.recent_scores[-1000:]
                        
                        # Track full training history for save/restore
                        self.training_history_scores.append(info['score'])
                        self.training_history_rewards.append(episode_reward)
                        self.training_history_steps.append(episode_steps)
                        self.training_history_epsilons.append(self.agent.epsilon)
                        self.training_history_bricks.append(episode_bricks_broken)
                        self.training_history_wins.append(info.get('won', False))
                        
                        # Save checkpoint
                        if self.episode % self.config.SAVE_EVERY == 0 and self.episode > 0:
                            self._save_model(f"breakout_ep{self.episode}.pth", save_reason="periodic")
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"💾 Checkpoint saved: breakout_ep{self.episode}.pth",
                                    "success"
                                )
                        
                        if info['score'] > self.best_score_ever:
                            self.best_score_ever = info['score']
                            self._save_model("breakout_best.pth", save_reason="best", quiet=True)
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
                    step_time_samples.append(measured_step_time)
                    # Keep rolling average of last 100 samples
                    if len(step_time_samples) > 100:
                        step_time_samples.pop(0)
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
            if not self.args.headless:
                render_action = self.selected_action if self.selected_action is not None else 1
                self._render_frame(state, render_action, info if info else {})
                
                # At speed 1x, throttle to 60 FPS for smooth real-time feel
                # At higher speeds, no throttling - render ASAP after training batch
                if self.game_speed <= 1:
                    self.clock.tick(60)
        
        # Training complete
        self._save_model("breakout_final.pth", save_reason="final")
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
        print(f"   Episodes:       {self.config.MAX_EPISODES}")
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
        start_time = time.time()
        last_report_time = start_time
        steps_since_report = 0
        
        for episode in range(self.config.MAX_EPISODES):
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
            
            self.agent.decay_epsilon()
            scores.append(info['score'])
            
            # Time-based progress reporting
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time
            
            if elapsed_since_report >= self.config.REPORT_INTERVAL_SECONDS or episode % self.config.LOG_EVERY == 0:
                elapsed_total = current_time - start_time
                steps_per_sec = steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                avg_score = np.mean(scores[-100:]) if scores else 0
                
                print(f"Episode {episode:5d} | "
                      f"Score: {info['score']:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"⚡ {steps_per_sec:,.0f} steps/s")
                
                last_report_time = current_time
                steps_since_report = 0
            
            # Save checkpoints
            if episode % self.config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(f"breakout_ep{episode}.pth")
            
            if info['score'] > max(scores[:-1], default=0):
                self._save_model("breakout_best.pth", quiet=True)
        
        self._save_model("breakout_final.pth")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"   Total time:      {total_time/60:.1f} minutes")
        print(f"   Total steps:     {total_steps:,}")
        print(f"   Avg steps/sec:   {total_steps/total_time:,.0f}")
        print(f"   Final avg score: {np.mean(scores[-100:]):.1f}")
        print(f"   Best score:      {max(scores)}")
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
            
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                
                elif event.key == pygame.K_p:
                    self._toggle_pause()
                
                elif event.key == pygame.K_s:
                    self._save_model("breakout_manual_save.pth", save_reason="manual")
                    if self.web_dashboard:
                        self.web_dashboard.log("💾 Manual save: breakout_manual_save.pth", "success")
                
                elif event.key == pygame.K_r:
                    self._reset_episode()
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._speed_up()
                
                elif event.key == pygame.K_MINUS:
                    self._speed_down()
                
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
    
    def _render_frame(self, state: np.ndarray, action: int, info: dict) -> None:
        """Render one frame of the visualization."""
        # Clear screen
        self.screen.fill((10, 10, 15))
        
        # Render game
        self.game.render(self.screen)
        
        # Render neural network visualization
        self.nn_visualizer.render(
            self.screen,
            self.agent,
            state,
            selected_action=action
        )
        
        # Render dashboard
        self.dashboard.render(self.screen)
        
        # Capture screenshot for web dashboard (every 10 frames for responsive updates)
        self.frame_count += 1
        if self.web_dashboard and self.frame_count % 10 == 0:
            self.web_dashboard.capture_screenshot(self.screen)
        
        # Render pause indicator (centered on the game area)
        if self.paused:
            game_center_x = self.config.SCREEN_WIDTH // 2
            game_center_y = self.config.SCREEN_HEIGHT // 2
            
            font = pygame.font.Font(None, 72)
            text = font.render("PAUSED", True, (255, 200, 50))
            text_rect = text.get_rect(center=(game_center_x, game_center_y))
            
            # Background with semi-transparent fill using SRCALPHA surface
            bg_rect = text_rect.inflate(40, 20)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(bg_surface, (0, 0, 0, 200), bg_surface.get_rect(), border_radius=10)
            self.screen.blit(bg_surface, bg_rect.topleft)
            pygame.draw.rect(self.screen, (255, 200, 50), bg_rect, 3, border_radius=10)
            
            self.screen.blit(text, text_rect)
        
        # Render speed indicator if not normal (top right of game area)
        if self.game_speed != 1.0:
            font = pygame.font.Font(None, 24)
            speed_text = font.render(f"Speed: {self.game_speed}x", True, (150, 150, 150))
            self.screen.blit(speed_text, (self.config.SCREEN_WIDTH - 110, 10))
        
        pygame.display.flip()
    
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
        filepath = os.path.join(self.config.MODEL_DIR, filename)
        
        # Calculate metrics for metadata
        avg_score = np.mean(self.recent_scores[-100:]) if self.recent_scores else 0.0
        win_rate = self.dashboard.get_win_rate() if hasattr(self.dashboard, 'get_win_rate') else 0.0
        
        # Build training history for dashboard restoration
        training_history = TrainingHistory(
            scores=self.training_history_scores.copy(),
            rewards=self.training_history_rewards.copy(),
            steps=self.training_history_steps.copy(),
            epsilons=self.training_history_epsilons.copy(),
            bricks=self.training_history_bricks.copy(),
            wins=self.training_history_wins.copy(),
            losses=list(self.agent.losses)[-1000:] if self.agent.losses else []
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


class HeadlessTrainer:
    """
    Lightweight headless trainer that skips pygame entirely.
    
    This provides maximum training throughput by:
        - No pygame initialization
        - No visualization overhead
        - Optimized training loop with configurable learning frequency
        - Progress reporting via terminal
    
    Usage:
        python main.py --headless --turbo --episodes 5000
    """
    
    def __init__(self, config: Config, args: argparse.Namespace):
        """
        Initialize headless trainer.
        
        Args:
            config: Configuration object
            args: Command line arguments
        """
        self.config = config
        self.args = args
        
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
        
        # Create game in headless mode (skips visual effects for max speed)
        self.game = Breakout(config, headless=True)
        
        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config
        )
        
        # Load model if specified (headless mode - just restore agent state)
        if args.model and os.path.exists(args.model):
            self.agent.load(args.model)  # Returns tuple, we ignore history in headless
        
        # Create model directory
        os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    def train(self) -> None:
        """Run headless training loop with optimized throughput."""
        config = self.config
        
        print("\n" + "=" * 70)
        print("🚀 HEADLESS TRAINING - Maximum Performance Mode")
        print("=" * 70)
        print(f"   Episodes:        {config.MAX_EPISODES}")
        print(f"   Device:          {config.DEVICE}")
        print(f"   Batch size:      {config.BATCH_SIZE}")
        print(f"   Learn every:     {config.LEARN_EVERY} steps")
        print(f"   Gradient steps:  {config.GRADIENT_STEPS}")
        print(f"   torch.compile:   {config.USE_TORCH_COMPILE}")
        print("=" * 70 + "\n")
        
        # Training state
        self.scores: list[int] = []
        self.total_steps = 0
        self.training_start_time = time.time()
        last_report_time = self.training_start_time
        steps_since_report = 0
        self.best_score = 0
        self.current_episode = 0
        
        for episode in range(config.MAX_EPISODES):
            self.current_episode = episode
            state = self.game.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.game.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
                self.agent.learn()
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                steps_since_report += 1
            
            # Episode complete
            self.agent.decay_epsilon()
            self.scores.append(info['score'])
            
            # Time-based progress reporting
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time
            
            if elapsed_since_report >= config.REPORT_INTERVAL_SECONDS or episode % config.LOG_EVERY == 0:
                elapsed_total = current_time - self.training_start_time
                steps_per_sec = steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                eps_per_hour = episode / elapsed_total * 3600 if elapsed_total > 0 else 0
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0
                
                print(f"Ep {episode:5d} | "
                      f"Score: {info['score']:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"⚡ {steps_per_sec:,.0f} steps/s | "
                      f"📊 {eps_per_hour:,.0f} ep/hr")
                
                last_report_time = current_time
                steps_since_report = 0
            
            # Save checkpoints
            if info['score'] > self.best_score:
                self.best_score = info['score']
                self._save_model("breakout_best.pth", save_reason="best", quiet=True)
            
            if episode % config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(f"breakout_ep{episode}.pth", save_reason="periodic")
        
        # Final save
        self._save_model("breakout_final.pth", save_reason="final")
        
        # Summary
        total_time = time.time() - self.training_start_time
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Total episodes:   {config.MAX_EPISODES}")
        print(f"   Total steps:      {self.total_steps:,}")
        print(f"   Total time:       {total_time/60:.1f} minutes")
        print(f"   Avg steps/sec:    {self.total_steps/total_time:,.0f}")
        print(f"   Best score:       {self.best_score}")
        print(f"   Final avg score:  {np.mean(self.scores[-100:]):.1f}")
        print("=" * 70)
    
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
        filepath = os.path.join(self.config.MODEL_DIR, filename)
        
        # Calculate metrics for metadata
        avg_score = np.mean(self.scores[-100:]) if self.scores else 0.0
        
        result = self.agent.save(
            filepath=filepath,
            save_reason=save_reason,
            episode=self.current_episode,
            best_score=self.best_score,
            avg_score_last_100=float(avg_score),
            win_rate=0.0,  # Not tracked in headless mode
            training_start_time=self.training_start_time,
            quiet=quiet
        )
        
        return result is not None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural Network Game AI - Train an AI to play Atari Breakout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Train with visualization
    python main.py --headless                # Headless training (no pygame)
    python main.py --headless --turbo        # TURBO: ~4x faster training
    python main.py --headless --learn-every 4 --batch-size 256  # Custom tuning
    python main.py --play --model best.pth   # Watch trained agent play
    python main.py --human                   # Play the game yourself
    python main.py --episodes 5000 --lr 0.001  # Custom parameters
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
    
    # Model options
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to model file to load'
    )
    
    # Training parameters
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of training episodes'
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
        help='Turbo mode preset: --learn-every 4 --batch-size 256 --torch-compile'
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


def main():
    """Main entry point."""
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
        trainer = HeadlessTrainer(config, args)
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            trainer._save_model("breakout_interrupted.pth", save_reason="interrupted")
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
    
    # Create application (with pygame)
    app = GameApp(config, args)
    
    # Run appropriate mode
    try:
        if args.human:
            app.run_human_mode()
        elif args.play:
            app.run_play_mode()
        else:
            app.run_training()
    except KeyboardInterrupt:
        print("\n\n⛔ Training interrupted by user")
        app._save_model("breakout_interrupted.pth", save_reason="interrupted")
        if app.web_dashboard:
            app.web_dashboard.log("⛔ Training interrupted by user", "warning")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
