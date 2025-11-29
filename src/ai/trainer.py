"""
Training Loop
=============

Orchestrates the training process:
    1. Run episodes of the game
    2. Collect experiences
    3. Train the agent
    4. Track metrics
    5. Save checkpoints

This module ties together the game, agent, and visualizer.
"""

import numpy as np
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import os

import sys
sys.path.append('../..')
from config import Config


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode: int
    score: int
    steps: int
    total_reward: float
    epsilon: float
    avg_loss: float
    duration: float
    bricks_broken: int
    won: bool


class TrainingMetrics:
    """
    Tracks and stores training metrics over time.
    
    Metrics tracked:
        - Episode scores
        - Total rewards
        - Steps per episode
        - Loss values
        - Epsilon values
        - Episode durations
    """
    
    def __init__(self, history_length: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            history_length: Maximum history to store
        """
        self.history_length = history_length
        
        self.scores: List[int] = []
        self.rewards: List[float] = []
        self.steps: List[int] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        self.durations: List[float] = []
        self.bricks: List[int] = []
        self.wins: List[bool] = []
    
    def add(self, stats: EpisodeStats) -> None:
        """Add episode statistics."""
        self.scores.append(stats.score)
        self.rewards.append(stats.total_reward)
        self.steps.append(stats.steps)
        self.losses.append(stats.avg_loss)
        self.epsilons.append(stats.epsilon)
        self.durations.append(stats.duration)
        self.bricks.append(stats.bricks_broken)
        self.wins.append(stats.won)
        
        # Trim to history length
        if len(self.scores) > self.history_length:
            for attr in ['scores', 'rewards', 'steps', 'losses', 
                        'epsilons', 'durations', 'bricks', 'wins']:
                setattr(self, attr, getattr(self, attr)[-self.history_length:])
    
    def get_recent_average(self, metric: str, n: int = 100) -> float:
        """Get average of last n values for a metric."""
        values = getattr(self, metric, [])
        if not values:
            return 0.0
        return np.mean(values[-n:])
    
    def get_best_score(self) -> int:
        """Get the highest score achieved."""
        return max(self.scores) if self.scores else 0
    
    def get_win_rate(self, n: int = 100) -> float:
        """Get win rate over last n episodes."""
        if not self.wins:
            return 0.0
        recent = self.wins[-n:]
        return sum(recent) / len(recent)


class Trainer:
    """
    Manages the training loop for the DQN agent.
    
    Responsibilities:
        1. Run training episodes
        2. Coordinate game, agent, and visualizer
        3. Track metrics and save checkpoints
        4. Handle rendering and user interaction
    
    Example:
        >>> game = Breakout()
        >>> agent = Agent(game.state_size, game.action_size)
        >>> trainer = Trainer(game, agent)
        >>> trainer.train(num_episodes=1000)
    """
    
    def __init__(
        self,
        game,
        agent,
        config: Optional[Config] = None,
        visualizer=None
    ):
        """
        Initialize the trainer.
        
        Args:
            game: Game instance (implements BaseGame)
            agent: DQN agent instance
            config: Configuration object
            visualizer: Optional visualizer for neural network
        """
        self.game = game
        self.agent = agent
        self.config = config or Config()
        self.visualizer = visualizer
        
        self.metrics = TrainingMetrics(self.config.PLOT_HISTORY_LENGTH)
        self.current_episode = 0
        self.total_steps = 0
        
        # Create model directory
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
    
    def run_episode(self, render: bool = False, screen=None) -> EpisodeStats:
        """
        Run a single training episode.
        
        Args:
            render: Whether to render the game
            screen: Pygame screen for rendering
            
        Returns:
            Episode statistics
        """
        start_time = time.time()
        
        state = self.game.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < self.config.MAX_STEPS_PER_EPISODE:
            # Select and execute action
            action = self.agent.select_action(state, training=True)
            next_state, reward, done, info = self.game.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = self.agent.learn()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            self.total_steps += 1
            
            # Render if requested
            if render and screen is not None:
                self.game.render(screen)
                if self.visualizer:
                    self.visualizer.render(screen, self.agent, state)
            
            if done:
                break
        
        # Decay epsilon after episode
        self.agent.decay_epsilon()
        
        duration = time.time() - start_time
        
        # Calculate bricks broken
        initial_bricks = self.config.BRICK_ROWS * self.config.BRICK_COLS
        bricks_broken = initial_bricks - info.get('bricks_remaining', initial_bricks)
        
        return EpisodeStats(
            episode=self.current_episode,
            score=info.get('score', 0),
            steps=steps,
            total_reward=total_reward,
            epsilon=self.agent.epsilon,
            avg_loss=self.agent.get_average_loss(100),
            duration=duration,
            bricks_broken=bricks_broken,
            won=info.get('won', False)
        )
    
    def train(
        self,
        num_episodes: Optional[int] = None,
        render_callback=None,
        progress_callback=None
    ) -> TrainingMetrics:
        """
        Run the training loop.
        
        Args:
            num_episodes: Number of episodes (default from config)
            render_callback: Function to call for rendering
            progress_callback: Function to call with progress updates
            
        Returns:
            Training metrics
        """
        num_episodes = num_episodes or self.config.MAX_EPISODES
        
        print("\n" + "=" * 60)
        print("🧠 Starting DQN Training")
        print("=" * 60)
        print(f"   Episodes: {num_episodes}")
        print(f"   Device: {self.config.DEVICE}")
        print(f"   State size: {self.game.state_size}")
        print(f"   Action size: {self.game.action_size}")
        print("=" * 60 + "\n")
        
        best_score = 0
        
        for episode in range(num_episodes):
            self.current_episode = episode
            
            # Run episode
            should_render = (episode % self.config.RENDER_EVERY == 0)
            stats = self.run_episode(render=should_render and render_callback)
            
            # Record metrics
            self.metrics.add(stats)
            
            # Call render callback if provided
            if should_render and render_callback:
                render_callback(stats)
            
            # Log progress
            if episode % self.config.LOG_EVERY == 0:
                avg_score = self.metrics.get_recent_average('scores', 100)
                avg_reward = self.metrics.get_recent_average('rewards', 100)
                win_rate = self.metrics.get_win_rate(100)
                
                print(f"Episode {episode:5d} | "
                      f"Score: {stats.score:4d} | "
                      f"Avg: {avg_score:6.1f} | "
                      f"ε: {stats.epsilon:.3f} | "
                      f"Loss: {stats.avg_loss:.4f} | "
                      f"Win: {win_rate*100:5.1f}%")
            
            # Save checkpoint
            if stats.score > best_score:
                best_score = stats.score
                self.agent.save(
                    os.path.join(self.config.MODEL_DIR, f'{self.config.GAME_NAME}_best.pth')
                )
            
            if episode % self.config.SAVE_EVERY == 0 and episode > 0:
                self.agent.save(
                    os.path.join(self.config.MODEL_DIR, f'{self.config.GAME_NAME}_ep{episode}.pth')
                )
            
            # Progress callback
            if progress_callback:
                progress_callback(episode, num_episodes, stats)
        
        # Final save
        self.agent.save(
            os.path.join(self.config.MODEL_DIR, f'{self.config.GAME_NAME}_final.pth')
        )
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"   Best score: {best_score}")
        print(f"   Final epsilon: {self.agent.epsilon:.4f}")
        print(f"   Total steps: {self.total_steps:,}")
        print(f"   Win rate: {self.metrics.get_win_rate(100)*100:.1f}%")
        print("=" * 60)
        
        return self.metrics
    
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict:
        """
        Evaluate the trained agent without exploration.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render
            
        Returns:
            Evaluation statistics
        """
        scores = []
        wins = 0
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration during evaluation
        
        for ep in range(num_episodes):
            state = self.game.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                state, _, done, info = self.game.step(action)
            
            scores.append(info.get('score', 0))
            if info.get('won', False):
                wins += 1
        
        self.agent.epsilon = original_epsilon
        
        return {
            'mean_score': np.mean(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'win_rate': wins / num_episodes
        }


# Testing
if __name__ == "__main__":
    print("Trainer module - import and use with game and agent")

