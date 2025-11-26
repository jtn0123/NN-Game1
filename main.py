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
"""

import pygame
import numpy as np
import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.game.breakout import Breakout
from src.ai.agent import Agent
from src.ai.trainer import Trainer
from src.visualizer.nn_visualizer import NeuralNetVisualizer
from src.visualizer.dashboard import Dashboard

# Optional web dashboard
try:
    from src.web import WebDashboard
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


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
        self.game_width = config.SCREEN_WIDTH
        self.game_height = config.SCREEN_HEIGHT
        self.viz_width = 300
        self.dashboard_height = 180
        
        self.window_width = self.game_width + self.viz_width + 20
        self.window_height = self.game_height + self.dashboard_height + 20
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
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
        
        # Create visualizations
        self.nn_visualizer = NeuralNetVisualizer(
            config=config,
            x=self.game_width + 10,
            y=10,
            width=self.viz_width,
            height=self.game_height - 10
        )
        
        self.dashboard = Dashboard(
            config=config,
            x=10,
            y=self.game_height + 10,
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
        
        # Current state
        self.state = self.game.reset()
        self.selected_action = None
        
        # FPS for rendering
        self.render_fps = 60
        self.train_fps = 0  # Unlimited for headless
        
        # Web dashboard (if enabled)
        self.web_dashboard = None
        if hasattr(args, 'web') and args.web and WEB_AVAILABLE:
            self.web_dashboard = WebDashboard(config, port=5000)
            self.web_dashboard.on_pause_callback = self._toggle_pause
            self.web_dashboard.on_save_callback = lambda: self._save_model("breakout_web_save.pth")
            self.web_dashboard.on_speed_callback = self._set_speed
            self.web_dashboard.start()
    
    def _toggle_pause(self):
        """Toggle pause state (for web dashboard control)."""
        self.paused = not self.paused
        if self.web_dashboard:
            self.web_dashboard.publisher.set_paused(self.paused)
        print("⏸️  Paused" if self.paused else "▶️  Resumed")
    
    def _set_speed(self, speed: float):
        """Set game speed (for web dashboard control)."""
        self.game_speed = max(0.25, min(4.0, speed))
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
        episode_reward = 0
        
        while self.running:
            self._handle_events()
            
            if not self.paused:
                # Agent selects action
                action = self.agent.select_action(state, training=False)
                
                # Step game
                state, reward, done, info = self.game.step(action)
                episode_reward += reward
                
                if done:
                    print(f"   Episode complete! Score: {info['score']}, Reward: {episode_reward:.1f}")
                    state = self.game.reset()
                    episode_reward = 0
                    self.episode += 1
                    self.dashboard.update(
                        self.episode, info['score'], 0, 0,
                        bricks_broken=50-info.get('bricks_remaining', 50)
                    )
                
                self.selected_action = action
            
            # Render
            self._render_frame(state, self.selected_action, {'score': info.get('score', 0)})
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
        print(f"   - Q/ESC: Quit")
        print("=" * 60 + "\n")
        
        state = self.game.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_start_time = time.time()
        
        while self.running and self.episode < self.config.MAX_EPISODES:
            self._handle_events()
            
            if not self.paused:
                # Agent selects action
                action = self.agent.select_action(state, training=True)
                self.selected_action = action
                
                # Execute action
                next_state, reward, done, info = self.game.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Learn
                loss = self.agent.learn()
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.steps += 1
                
                if done:
                    # Episode complete
                    episode_duration = time.time() - episode_start_time
                    
                    # Decay epsilon
                    self.agent.decay_epsilon()
                    
                    # Update dashboard
                    self.dashboard.update(
                        self.episode,
                        info['score'],
                        self.agent.epsilon,
                        self.agent.get_average_loss(100),
                        bricks_broken=50-info.get('bricks_remaining', 50),
                        won=info.get('won', False)
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
                            reward=episode_reward
                        )
                    
                    # Log
                    if self.episode % self.config.LOG_EVERY == 0:
                        avg_score = np.mean(self.dashboard.scores[-100:]) if self.dashboard.scores else 0
                        print(f"Episode {self.episode:5d} | "
                              f"Score: {info['score']:4d} | "
                              f"Avg: {avg_score:6.1f} | "
                              f"ε: {self.agent.epsilon:.3f} | "
                              f"Steps: {episode_steps:5d} | "
                              f"Time: {episode_duration:.1f}s")
                    
                    # Save checkpoint
                    if self.episode % self.config.SAVE_EVERY == 0 and self.episode > 0:
                        self._save_model(f"breakout_ep{self.episode}.pth")
                    
                    if info['score'] > self.dashboard.best_score:
                        self._save_model("breakout_best.pth", quiet=True)
                    
                    # Reset for next episode
                    self.episode += 1
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_start_time = time.time()
                    state = self.game.reset()
            
            # Render (at controlled framerate)
            if not self.args.headless:
                self._render_frame(state, self.selected_action, info if 'info' in dir() else {})
                self.clock.tick(int(self.render_fps * self.game_speed))
        
        # Training complete
        self._save_model("breakout_final.pth")
        print("\n✅ Training complete!")
        print(f"   Total episodes: {self.episode}")
        print(f"   Best score: {self.dashboard.best_score}")
        
        pygame.quit()
    
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
            
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    print("⏸️  Paused" if self.paused else "▶️  Resumed")
                
                elif event.key == pygame.K_s:
                    self._save_model("breakout_manual_save.pth")
                
                elif event.key == pygame.K_r:
                    self.state = self.game.reset()
                    print("🔄 Episode reset")
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.game_speed = min(4.0, self.game_speed + 0.5)
                    print(f"⏩ Speed: {self.game_speed}x")
                
                elif event.key == pygame.K_MINUS:
                    self.game_speed = max(0.25, self.game_speed - 0.25)
                    print(f"⏪ Speed: {self.game_speed}x")
    
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
        
        # Capture screenshot for web dashboard (every 30 frames)
        if self.web_dashboard and self.steps % 30 == 0:
            self.web_dashboard.capture_screenshot(self.screen)
        
        # Render pause indicator
        if self.paused:
            font = pygame.font.Font(None, 72)
            text = font.render("PAUSED", True, (255, 200, 50))
            text_rect = text.get_rect(center=(self.game_width // 2, self.game_height // 2))
            
            # Background
            bg_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, (0, 0, 0, 200), bg_rect, border_radius=10)
            pygame.draw.rect(self.screen, (255, 200, 50), bg_rect, 3, border_radius=10)
            
            self.screen.blit(text, text_rect)
        
        # Render speed indicator if not normal
        if self.game_speed != 1.0:
            font = pygame.font.Font(None, 24)
            speed_text = font.render(f"Speed: {self.game_speed}x", True, (150, 150, 150))
            self.screen.blit(speed_text, (self.game_width - 100, 10))
        
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
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

