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
    
    # Play with a trained model
    python main.py --play --model models/breakout_best.pth
    
    # Human play mode (for testing game)
    python main.py --human
    
    # Custom training parameters
    python main.py --episodes 5000 --lr 0.0001

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
from src.ai.agent import Agent
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
        
        # Load model if specified
        if args.model and os.path.exists(args.model):
            self.agent.load(args.model)
        
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
            self.web_dashboard.on_save_callback = lambda: self._save_model("breakout_web_save.pth")
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.on_reset_callback = self._reset_episode
            self.web_dashboard.on_load_model_callback = self._load_model
            self.web_dashboard.on_config_change_callback = self._apply_config
            self.web_dashboard.start()
            
            # Log startup info
            self._log_startup_info()
    
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
        """Load a model from file."""
        try:
            self.agent.load(filepath)
            if self.web_dashboard:
                self.web_dashboard.log(f"📂 Loaded model: {os.path.basename(filepath)}", "success", {
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
        
        if self.web_dashboard and changes:
            self.web_dashboard.log(f"⚙️ Config updated: {', '.join(changes)}", "action", config_data)
    
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
    
    def _set_speed(self, speed: float) -> None:
        """Set game speed (for web dashboard control)."""
        self.game_speed = max(0.25, min(4.0, speed))
        if self.web_dashboard:
            self.web_dashboard.log(f"⏩ Speed set to {self.game_speed}x", "action")
        print(f"⏩ Speed: {self.game_speed}x")
    
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
        print(f"   Episodes: {self.config.MAX_EPISODES}")
        print(f"   Learning Rate: {self.config.LEARNING_RATE}")
        print(f"   Device: {self.config.DEVICE}")
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
        
        while self.running and self.episode < self.config.MAX_EPISODES:
            self._handle_events()
            
            if not self.paused:
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
                
                # Learn
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
                    
                    # Save checkpoint
                    if self.episode % self.config.SAVE_EVERY == 0 and self.episode > 0:
                        self._save_model(f"breakout_ep{self.episode}.pth")
                        if self.web_dashboard:
                            self.web_dashboard.log(
                                f"💾 Checkpoint saved: breakout_ep{self.episode}.pth",
                                "success"
                            )
                    
                    if info['score'] > self.dashboard.best_score:
                        self._save_model("breakout_best.pth", quiet=True)
                        if self.web_dashboard:
                            self.web_dashboard.log(
                                f"🏆 New best score: {info['score']}! Model saved.",
                                "success",
                                {'score': info['score'], 'episode': self.episode}
                            )
                    
                    # Reset for next episode
                    self.episode += 1
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_bricks_broken = 0
                    episode_start_time = time.time()
                    state = self.game.reset()
            
            # Render (at controlled framerate)
            if not self.args.headless:
                render_action = self.selected_action if self.selected_action is not None else 1
                self._render_frame(state, render_action, info if info else {})
                self.clock.tick(int(self.render_fps * self.game_speed))
        
        # Training complete
        self._save_model("breakout_final.pth")
        if self.web_dashboard:
            self.web_dashboard.log("🎉 Training complete!", "success", {
                'total_episodes': self.episode,
                'best_score': self.dashboard.best_score,
                'total_steps': self.steps
            })
        
        print("\n✅ Training complete!")
        print(f"   Total episodes: {self.episode}")
        print(f"   Best score: {self.dashboard.best_score}")
        
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
        """Run training without visualization (faster)."""
        print("\n" + "=" * 60)
        print("🚀 Starting Headless Training (No Visualization)")
        print("=" * 60)
        print(f"   Episodes: {self.config.MAX_EPISODES}")
        print(f"   Device: {self.config.DEVICE}")
        print("=" * 60 + "\n")
        
        state = self.game.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        scores = []
        start_time = time.time()
        
        for episode in range(self.config.MAX_EPISODES):
            done = False
            state = self.game.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            while not done:
                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.game.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.learn()
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            self.agent.decay_epsilon()
            scores.append(info['score'])
            
            # Log progress
            if episode % self.config.LOG_EVERY == 0:
                avg_score = np.mean(scores[-100:]) if scores else 0
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                print(f"Episode {episode:5d} | "
                      f"Score: {info['score']:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"Speed: {eps_per_sec:.1f} ep/s")
            
            # Save checkpoints
            if episode % self.config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(f"breakout_ep{episode}.pth")
            
            if info['score'] > max(scores[:-1], default=0):
                self._save_model("breakout_best.pth", quiet=True)
        
        self._save_model("breakout_final.pth")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Final avg score: {np.mean(scores[-100:]):.1f}")
        print(f"   Best score: {max(scores)}")
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
                    self._save_model("breakout_manual_save.pth")
                    if self.web_dashboard:
                        self.web_dashboard.log("💾 Manual save: breakout_manual_save.pth", "success")
                
                elif event.key == pygame.K_r:
                    self._reset_episode()
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._set_speed(self.game_speed + 0.5)
                
                elif event.key == pygame.K_MINUS:
                    self._set_speed(self.game_speed - 0.25)
                
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
    
    def _save_model(self, filename: str, quiet: bool = False) -> None:
        """Save the current model."""
        filepath = os.path.join(self.config.MODEL_DIR, filename)
        self.agent.save(filepath)
        if not quiet:
            print(f"💾 Model saved: {filename}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Neural Network Game AI - Train an AI to play Atari Breakout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Train with visualization
    python main.py --headless                # Train without visualization (faster)
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
    
    # Other options
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = Config()
    
    # Set seed if specified
    if args.seed:
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)
        config.SEED = args.seed
    
    # Create application
    app = GameApp(config, args)
    
    # Run appropriate mode
    try:
        if args.human:
            app.run_human_mode()
        elif args.play:
            app.run_play_mode()
        elif args.headless:
            app.run_headless_training()
        else:
            app.run_training()
    except KeyboardInterrupt:
        print("\n\n⛔ Training interrupted by user")
        app._save_model("breakout_interrupted.pth")
        if app.web_dashboard:
            app.web_dashboard.log("⛔ Training interrupted by user", "warning")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
