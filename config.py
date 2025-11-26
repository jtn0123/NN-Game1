"""
Configuration file for Neural Network Game AI
==============================================

All hyperparameters, game settings, and visualization options are centralized here.
Modify these values to experiment with different training configurations.

Usage:
    from config import Config
    cfg = Config()
    print(cfg.LEARNING_RATE)
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class Config:
    """
    Central configuration for the entire project.
    
    Sections:
    1. Game Settings - Breakout game parameters
    2. Neural Network - Architecture configuration
    3. Training - Learning hyperparameters
    4. Exploration - Epsilon-greedy settings
    5. Visualization - Display options
    6. System - Hardware and paths
    """
    
    # =========================================================================
    # GAME SETTINGS
    # =========================================================================
    
    # Screen dimensions
    SCREEN_WIDTH: int = 800
    SCREEN_HEIGHT: int = 600
    
    # Breakout-specific
    PADDLE_WIDTH: int = 100
    PADDLE_HEIGHT: int = 15
    PADDLE_SPEED: int = 8
    
    BALL_RADIUS: int = 8
    BALL_SPEED: int = 6
    
    BRICK_ROWS: int = 5
    BRICK_COLS: int = 10
    BRICK_WIDTH: int = 70
    BRICK_HEIGHT: int = 25
    BRICK_PADDING: int = 5
    BRICK_OFFSET_TOP: int = 60
    BRICK_OFFSET_LEFT: int = 35
    
    # Game mechanics
    LIVES: int = 3
    FPS: int = 60
    
    # =========================================================================
    # NEURAL NETWORK ARCHITECTURE
    # =========================================================================
    
    # Input size is calculated based on game state:
    # - Ball position (x, y) = 2
    # - Ball velocity (dx, dy) = 2  
    # - Paddle position (x) = 1
    # - Brick states = BRICK_ROWS * BRICK_COLS
    
    @property
    def STATE_SIZE(self) -> int:
        """Calculate input layer size based on game state representation."""
        ball_info = 4        # x, y, dx, dy
        paddle_info = 1      # x position
        brick_info = self.BRICK_ROWS * self.BRICK_COLS  # binary brick states
        return ball_info + paddle_info + brick_info
    
    # Action space
    ACTION_SIZE: int = 3      # LEFT, STAY, RIGHT
    
    # Hidden layer architecture
    # More neurons = more capacity but slower training
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [256, 128])
    
    # Activation function: 'relu', 'leaky_relu', 'tanh'
    ACTIVATION: str = 'relu'
    
    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    
    # Learning rate - How big of steps to take during optimization
    # Too high: unstable training, loss explodes
    # Too low: very slow learning
    LEARNING_RATE: float = 0.0001
    
    # Discount factor (gamma) - How much to value future rewards
    # 0.99 = far-sighted, considers distant future
    # 0.90 = more short-sighted, prefers immediate rewards
    GAMMA: float = 0.99
    
    # Batch size - Number of experiences to sample per training step
    # Larger = more stable gradients but slower
    BATCH_SIZE: int = 64
    
    # Replay buffer capacity
    # Larger = more diverse experiences but more memory
    MEMORY_SIZE: int = 100_000
    
    # Minimum experiences before training starts
    MEMORY_MIN: int = 1000
    
    # Target network update frequency (in steps)
    # How often to sync target network with policy network
    TARGET_UPDATE: int = 1000
    
    # Gradient clipping to prevent exploding gradients
    GRAD_CLIP: float = 1.0
    
    # =========================================================================
    # EXPLORATION SETTINGS (Epsilon-Greedy)
    # =========================================================================
    
    # Starting exploration rate (1.0 = 100% random)
    EPSILON_START: float = 1.0
    
    # Minimum exploration rate (0.01 = 1% random)
    EPSILON_END: float = 0.01
    
    # Decay rate per episode (higher = slower decay)
    # EPSILON_DECAY = 0.995 means epsilon *= 0.995 after each episode
    EPSILON_DECAY: float = 0.995
    
    # =========================================================================
    # REWARD SHAPING
    # =========================================================================
    
    # Rewards for different events
    REWARD_BRICK_HIT: float = 1.0       # Breaking a brick
    REWARD_GAME_OVER: float = -10.0     # Losing a life
    REWARD_WIN: float = 50.0            # Clearing all bricks
    REWARD_PADDLE_HIT: float = 0.1      # Ball hitting paddle (encourages survival)
    REWARD_STEP: float = 0.0            # Small reward each step (can set negative for urgency)
    
    # =========================================================================
    # VISUALIZATION SETTINGS
    # =========================================================================
    
    # Colors (RGB tuples)
    COLOR_BACKGROUND: Tuple[int, int, int] = (15, 15, 35)
    COLOR_PADDLE: Tuple[int, int, int] = (52, 152, 219)
    COLOR_BALL: Tuple[int, int, int] = (241, 196, 15)
    COLOR_BRICK_COLORS: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (231, 76, 60),    # Red
        (230, 126, 34),   # Orange
        (241, 196, 15),   # Yellow
        (46, 204, 113),   # Green
        (52, 152, 219),   # Blue
    ])
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)
    
    # Neural network visualizer
    VIS_NEURON_RADIUS: int = 8
    VIS_LAYER_SPACING: int = 150
    VIS_NEURON_SPACING: int = 20
    VIS_MAX_NEURONS_DISPLAY: int = 20  # Limit for very large layers
    
    # Activation coloring
    VIS_COLOR_INACTIVE: Tuple[int, int, int] = (50, 50, 50)
    VIS_COLOR_ACTIVE: Tuple[int, int, int] = (0, 255, 128)
    VIS_COLOR_WEIGHT_POS: Tuple[int, int, int] = (100, 200, 100)
    VIS_COLOR_WEIGHT_NEG: Tuple[int, int, int] = (200, 100, 100)
    
    # Dashboard
    PLOT_HISTORY_LENGTH: int = 100  # Number of episodes to show in plots
    
    # =========================================================================
    # TRAINING CONTROL
    # =========================================================================
    
    # Total episodes to train
    MAX_EPISODES: int = 2000
    
    # Maximum steps per episode (prevents infinite games)
    MAX_STEPS_PER_EPISODE: int = 10000
    
    # Save model every N episodes
    SAVE_EVERY: int = 100
    
    # Render every N episodes during training (0 = never)
    RENDER_EVERY: int = 1
    
    # Print stats every N episodes
    LOG_EVERY: int = 10
    
    # =========================================================================
    # SYSTEM SETTINGS
    # =========================================================================
    
    # Device selection
    @property
    def DEVICE(self) -> torch.device:
        """Auto-detect CUDA/MPS/CPU."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    # Paths
    MODEL_DIR: str = 'models'
    LOG_DIR: str = 'logs'
    
    # Random seed for reproducibility (None for random)
    SEED: int = None
    
    def __post_init__(self):
        """Validation and derived calculations."""
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"
        assert 0 < self.GAMMA <= 1, "Gamma must be in (0, 1]"
        assert self.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.EPSILON_START >= self.EPSILON_END, "Epsilon start must be >= end"


# Global config instance for easy importing
config = Config()


if __name__ == "__main__":
    # Print configuration summary
    cfg = Config()
    print("=" * 60)
    print("Neural Network Game AI - Configuration Summary")
    print("=" * 60)
    print(f"\n📺 Game: {cfg.SCREEN_WIDTH}x{cfg.SCREEN_HEIGHT}")
    print(f"🧱 Bricks: {cfg.BRICK_ROWS}x{cfg.BRICK_COLS} = {cfg.BRICK_ROWS * cfg.BRICK_COLS}")
    print(f"\n🧠 Neural Network:")
    print(f"   Input size: {cfg.STATE_SIZE}")
    print(f"   Hidden layers: {cfg.HIDDEN_LAYERS}")
    print(f"   Output size: {cfg.ACTION_SIZE}")
    print(f"\n📊 Training:")
    print(f"   Learning rate: {cfg.LEARNING_RATE}")
    print(f"   Batch size: {cfg.BATCH_SIZE}")
    print(f"   Gamma: {cfg.GAMMA}")
    print(f"\n🎲 Exploration:")
    print(f"   Epsilon: {cfg.EPSILON_START} → {cfg.EPSILON_END}")
    print(f"   Decay: {cfg.EPSILON_DECAY}")
    print(f"\n💻 Device: {cfg.DEVICE}")
    print("=" * 60)

