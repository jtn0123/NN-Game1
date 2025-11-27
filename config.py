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
from typing import List, Tuple, Optional
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
        tracking_info = 3    # relative_x, predicted_landing, distance_to_target
        brick_info = self.BRICK_ROWS * self.BRICK_COLS  # binary brick states
        return ball_info + paddle_info + tracking_info + brick_info
    
    # Action space
    ACTION_SIZE: int = 3      # LEFT, STAY, RIGHT
    
    # Hidden layer architecture
    # More neurons = more capacity but slower training
    # [512, 256, 128] provides more capacity for complex patterns
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Activation function: 'relu', 'leaky_relu', 'tanh'
    ACTIVATION: str = 'relu'
    
    # Use Dueling DQN architecture (separates value and advantage streams)
    # This helps the network learn which states are valuable independent of actions
    USE_DUELING: bool = True
    
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
    # 0.97 = balanced, prioritizes near-term rewards for more stable learning
    GAMMA: float = 0.97
    
    # Batch size - Number of experiences to sample per training step
    # Larger = more stable gradients but slower per step (better GPU utilization)
    # M4/MPS benefits from larger batches (256+ for better GPU utilization)
    BATCH_SIZE: int = 256
    
    # Replay buffer capacity
    # Larger = more diverse experiences but more memory
    MEMORY_SIZE: int = 100_000
    
    # Minimum experiences before training starts
    MEMORY_MIN: int = 1000
    
    # Target network update frequency (in steps) - used for hard updates
    # How often to sync target network with policy network
    TARGET_UPDATE: int = 1000
    
    # Soft target update coefficient (TAU)
    # If > 0, uses soft updates instead of hard updates
    # target = TAU * policy + (1 - TAU) * target
    # Typical values: 0.001 to 0.01
    TARGET_TAU: float = 0.005
    
    # Use soft updates instead of hard updates
    USE_SOFT_UPDATE: bool = True
    
    # Gradient clipping to prevent exploding gradients
    GRAD_CLIP: float = 1.0
    
    # =========================================================================
    # PERFORMANCE OPTIMIZATION
    # =========================================================================
    
    # Learn every N steps (1 = every step, 4 = every 4th step for ~4x speedup)
    # Higher values reduce backward passes but may slow learning convergence
    LEARN_EVERY: int = 4  # Skip steps for ~2x speedup on M4
    
    # Number of gradient updates per learning call
    # Useful when LEARN_EVERY > 1 to compensate with more updates
    GRADIENT_STEPS: int = 2  # Compensate for LEARN_EVERY with extra gradient steps
    
    # Use torch.compile() for potential 20-50% speedup (PyTorch 2.0+)
    # May have initial compilation overhead but faster afterwards
    USE_TORCH_COMPILE: bool = True  # Enabled for M4 Mac performance
    
    # Compile mode: 'default', 'reduce-overhead', 'max-autotune'
    # 'reduce-overhead' is best for small models, 'max-autotune' for large
    TORCH_COMPILE_MODE: str = 'reduce-overhead'
    
    # Use mixed precision (float16) for faster computation on GPU/MPS
    # Keeps optimizer state in float32 for numerical stability
    USE_MIXED_PRECISION: bool = True
    
    # =========================================================================
    # EXPLORATION SETTINGS (Epsilon-Greedy)
    # =========================================================================
    
    # Starting exploration rate (1.0 = 100% random)
    EPSILON_START: float = 1.0
    
    # Minimum exploration rate (0.005 = 0.5% random for fewer late-game mistakes)
    EPSILON_END: float = 0.005
    
    # Decay rate per episode (higher = slower decay)
    # EPSILON_DECAY = 0.995 means epsilon *= 0.995 after each episode
    EPSILON_DECAY: float = 0.995
    
    # Exploration decay strategy: 'exponential', 'linear', 'cosine'
    EXPLORATION_STRATEGY: str = 'exponential'
    
    # Warmup episodes before epsilon starts decaying
    EPSILON_WARMUP: int = 0
    
    # =========================================================================
    # PRIORITIZED EXPERIENCE REPLAY
    # =========================================================================
    
    # Enable prioritized replay (samples important experiences more often)
    USE_PRIORITIZED_REPLAY: bool = True
    
    # Priority exponent (0 = uniform sampling, 1 = full prioritization)
    PER_ALPHA: float = 0.6
    
    # Importance sampling start (anneals to 1.0)
    PER_BETA_START: float = 0.4
    
    # Beta annealing rate per sample
    PER_BETA_DECAY: float = 0.001
    
    # =========================================================================
    # REWARD SHAPING
    # =========================================================================
    
    # Rewards for different events
    REWARD_BRICK_HIT: float = 2.0       # Breaking a brick (doubled for more incentive)
    REWARD_GAME_OVER: float = -5.0      # Losing a life (halved for less harsh punishment)
    REWARD_WIN: float = 100.0           # Clearing all bricks (doubled for stronger completion incentive)
    REWARD_PADDLE_HIT: float = 0.2      # Ball hitting paddle (doubled to encourage survival)
    REWARD_STEP: float = 0.0            # Small reward each step (can set negative for urgency)
    
    # Dense reward shaping for ball tracking
    REWARD_TRACKING_GOOD: float = 0.01  # Reward for moving toward predicted ball landing
    REWARD_TRACKING_BAD: float = -0.01  # Penalty for moving away from predicted landing
    
    # Reward clipping to prevent extreme gradients during training
    # Set to 0 to disable clipping
    # Note: Only clips negative rewards to preserve win bonus signal
    REWARD_CLIP: float = 5.0
    
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
    
    # Report interval for headless mode (seconds between progress reports)
    REPORT_INTERVAL_SECONDS: float = 5.0
    
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
    SEED: Optional[int] = None
    
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

