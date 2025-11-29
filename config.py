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
    # GAME SELECTION
    # =========================================================================
    
    # Current game to play/train
    # Options: 'breakout', 'space_invaders'
    GAME_NAME: str = 'breakout'
    
    # =========================================================================
    # SCREEN SETTINGS
    # =========================================================================
    
    # Screen dimensions (shared across all games)
    SCREEN_WIDTH: int = 800
    SCREEN_HEIGHT: int = 600
    
    # =========================================================================
    # BREAKOUT SETTINGS
    # =========================================================================
    
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
    # SPACE INVADERS SETTINGS
    # =========================================================================
    
    # Grid of aliens (classic: 5 rows x 11 columns = 55 aliens)
    SI_ALIEN_ROWS: int = 5
    SI_ALIEN_COLS: int = 11
    SI_ALIEN_WIDTH: int = 36
    SI_ALIEN_HEIGHT: int = 26
    SI_ALIEN_PADDING: int = 12
    SI_ALIEN_OFFSET_TOP: int = 100  # Start aliens higher for more play space
    SI_ALIEN_OFFSET_LEFT: int = 70
    
    # Alien movement - tuned for AI learning with slower initial speed
    SI_ALIEN_SPEED_X: float = 0.8  # Slower initial speed (was 2.0)
    SI_ALIEN_SPEED_Y: int = 10  # Smaller drops (was 20) - gives AI more time
    SI_ALIEN_SPEED_MULTIPLIER: float = 1.03  # Gradual speed increase
    
    # Player ship
    SI_SHIP_WIDTH: int = 50
    SI_SHIP_HEIGHT: int = 30
    SI_SHIP_SPEED: int = 7  # Slightly faster for responsive control
    SI_SHIP_Y_OFFSET: int = 80  # More space from bottom for base visual
    
    # Bullets
    SI_BULLET_WIDTH: int = 4
    SI_BULLET_HEIGHT: int = 15
    SI_BULLET_SPEED: int = 12  # Faster bullets
    SI_MAX_PLAYER_BULLETS: int = 2  # Limited to 2 like original
    SI_ALIEN_SHOOT_CHANCE: float = 0.001  # Reduced from 0.002 - less spam
    SI_ALIEN_BULLET_SPEED: int = 4  # Slower alien bullets for fairness
    
    # UFO bonus
    SI_UFO_CHANCE: float = 0.0008  # Slightly rarer
    SI_UFO_SPEED: int = 3
    SI_UFO_POINTS: int = 100  # 50-300 random in original, we use fixed
    
    # Shields/bunkers (classic Space Invaders defense)
    # Set to False to disable bunkers for simpler gameplay/training
    SI_SHIELDS_ENABLED: bool = False
    SI_SHIELD_COUNT: int = 4
    SI_SHIELD_WIDTH: int = 50
    SI_SHIELD_HEIGHT: int = 35
    
    # Space Invaders rewards - tuned for AI training
    SI_REWARD_ALIEN_HIT: float = 1.0  # Per alien killed
    SI_REWARD_UFO_HIT: float = 5.0  # UFO bonus
    SI_REWARD_PLAYER_DEATH: float = -5.0  # Less harsh (was -10)
    SI_REWARD_LEVEL_CLEAR: float = 30.0  # Per level cleared (was 50)
    SI_REWARD_SHOOT: float = -0.005  # Tiny penalty to discourage spam
    SI_REWARD_STEP: float = 0.001  # Small survival reward
    
    # Space Invaders colors (CRT phosphor aesthetic)
    SI_COLOR_BACKGROUND: Tuple[int, int, int] = (0, 5, 0)  # Near black with green tint
    SI_COLOR_SHIP: Tuple[int, int, int] = (0, 255, 100)  # Bright green
    SI_COLOR_BULLET: Tuple[int, int, int] = (0, 255, 200)  # Cyan
    SI_COLOR_ALIEN_1: Tuple[int, int, int] = (255, 60, 100)  # Pink/magenta (top rows)
    SI_COLOR_ALIEN_2: Tuple[int, int, int] = (100, 255, 100)  # Green (middle)
    SI_COLOR_ALIEN_3: Tuple[int, int, int] = (100, 200, 255)  # Cyan (bottom)
    SI_COLOR_UFO: Tuple[int, int, int] = (255, 50, 50)  # Red
    SI_COLOR_SHIELD: Tuple[int, int, int] = (0, 220, 80)  # Bright green bunkers
    
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
    # Larger = more stable gradients but slower per step
    # M4 CPU optimal: 128 (balances throughput and stability)
    BATCH_SIZE: int = 128
    
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
    # 
    # M4 MacBook Benchmark Results (headless training):
    #   CPU B=128, LE=8, GS=2:  ~5,000 steps/sec, 663 grad/sec (balanced)
    #   CPU B=128, LE=16, GS=4: ~2,900 steps/sec, 719 grad/sec (max learning)
    #   MPS B=256, LE=4:        ~640 steps/sec (GPU overhead dominates)
    # 
    # CONCLUSION: Use CPU for small models - MPS transfer overhead is too high.
    # =========================================================================
    
    # Learn every N steps (1 = every step, higher = faster but less frequent learning)
    # M4 optimal: 8-16 for CPU, 4 for MPS
    LEARN_EVERY: int = 8
    
    # Number of gradient updates per learning call
    # Compensates for LEARN_EVERY > 1 to maintain learning throughput
    # Rule of thumb: GRADIENT_STEPS = LEARN_EVERY / 4 (for similar grad/sec)
    GRADIENT_STEPS: int = 2
    
    # Use torch.compile() for potential speedup (PyTorch 2.0+)
    # Note: Minimal benefit on CPU for small models, can cause overhead
    USE_TORCH_COMPILE: bool = False  # Disabled - minimal benefit for this model size
    
    # Compile mode: 'default', 'reduce-overhead', 'max-autotune'
    TORCH_COMPILE_MODE: str = 'reduce-overhead'
    
    # Use mixed precision (float16) for faster computation on GPU/MPS
    # Only beneficial on GPU - CPU uses float32 regardless
    USE_MIXED_PRECISION: bool = False  # Disabled - using CPU by default
    
    # Force CPU device (faster than MPS for small models on M4)
    # Set via --cpu flag or environment variable
    FORCE_CPU: bool = False
    
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
    # Improves learning efficiency by 30-40% at cost of ~10% speed overhead
    USE_PRIORITIZED_REPLAY: bool = False
    
    # Priority exponent (0 = uniform sampling, 1 = full prioritization)
    PER_ALPHA: float = 0.6
    
    # Importance sampling start (anneals to 1.0 over PER_BETA_FRAMES)
    PER_BETA_START: float = 0.4
    
    # Number of frames over which to anneal beta from start to 1.0
    PER_BETA_FRAMES: int = 100000
    
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
    
    # Total episodes to train (0 = unlimited, train until manually stopped)
    MAX_EPISODES: int = 0
    
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
        """Auto-detect CUDA/MPS/CPU, or force CPU if configured."""
        # Force CPU mode (faster for small models on M4)
        if self.FORCE_CPU:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    # Paths
    MODEL_DIR: str = 'models'
    LOG_DIR: str = 'logs'
    
    @property
    def GAME_MODEL_DIR(self) -> str:
        """Get game-specific model directory (e.g., 'models/breakout/')."""
        import os
        return os.path.join(self.MODEL_DIR, self.GAME_NAME)
    
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

