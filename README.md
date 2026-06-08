# 🧠 Neural Network Game AI

A complete, educational implementation of a Deep Q-Learning (DQN) agent that learns to play classic arcade games **in real-time** with a **live neural network visualizer**.

**Supported Games:** 🎮 Breakout | 👾 Space Invaders | 🏓 Pong | 🐍 Snake | ☄️ Asteroids

![Project Architecture](docs/architecture.png)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture Deep Dive](#-architecture-deep-dive)
3. [Installation](#-installation)
4. [Quick Start](#-quick-start)
5. [How It Works](#-how-it-works)
6. [Configuration Guide](#-configuration-guide)
7. [Advanced DQN Features](#-advanced-dqn-features)
8. [Space Invaders Configuration](#-space-invaders-configuration)
9. [Benchmarking & Performance](#-benchmarking--performance)
10. [Safety Notes](#-safety-notes)
11. [Developer Checks](#-developer-checks)
12. [Extending to Other Games](#-extending-to-other-games)
13. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

### What This Project Does

This project demonstrates **reinforcement learning** by training a neural network to play classic arcade games:

1. **The Games** (`src/game/`) - Complete implementations of Breakout, Space Invaders, Pong, Snake, and Asteroids
2. **The AI Brain** (`src/ai/`) - Advanced DQN with modern enhancements (Dueling, NoisyNets, PER, N-step)
3. **The Visualizer** (`src/visualizer/`) - Real-time neural network activity visualization

### Key Features

- ✅ **Live Training Visualization** - Watch neurons fire as the AI learns
- ✅ **Real-time Gameplay** - See the AI play the game live
- ✅ **Training Metrics Dashboard** - Loss curves, rewards, epsilon decay
- ✅ **Modular Architecture** - Easy to swap games or AI algorithms
- ✅ **Configurable Hyperparameters** - Tune learning rate, network size, etc.
- ✅ **Checkpoint System** - Save/load trained models

---

## 🏗️ Architecture Deep Dive

### Project Structure

```
NN-Game1/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # All hyperparameters & settings
├── main.py                      # Entry point - run this!
│
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   ├── base_game.py         # Abstract base class for games
│   │   ├── breakout.py          # Breakout game logic
│   │   ├── space_invaders.py    # Space Invaders game logic
│   │   ├── pong.py              # Pong game logic
│   │   ├── snake.py             # Snake game logic
│   │   └── asteroids.py         # Asteroids game logic
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── network.py           # Neural network architecture (PyTorch)
│   │   ├── agent.py             # DQN agent (action selection, learning)
│   │   ├── replay_buffer.py     # Experience replay memory
│   │   └── trainer.py           # Training loop orchestration
│   │
│   └── visualizer/
│       ├── __init__.py
│       ├── nn_visualizer.py     # Neural network visualization
│       └── dashboard.py         # Training metrics display
│
├── models/                      # Saved model checkpoints
│   └── .gitkeep
│
└── tests/
    ├── __init__.py
    ├── test_game.py
    ├── test_agent.py
    └── test_network.py
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN TRAINING LOOP                              │
└─────────────────────────────────────────────────────────────────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│               │          │               │          │               │
│   BREAKOUT    │◄────────►│   DQN AGENT   │◄────────►│  VISUALIZER   │
│     GAME      │          │               │          │               │
│               │          │               │          │               │
└───────────────┘          └───────────────┘          └───────────────┘
        │                           │                           │
        │                           │                           │
        ▼                           ▼                           ▼
   Game State               Neural Network              Live Display
   - Ball position          - Input Layer (84)          - Network graph
   - Paddle position        - Hidden (256, 128)         - Activations
   - Brick states           - Output (3 actions)        - Metrics
   - Score/Lives            - Q-values                  - Game view
```

---

## 🔧 Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip package manager

### Setup Steps

```bash
# 1. Clone or navigate to the project
cd /Users/justin/Documents/Github/NN-Game1

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies with the tested constraint pins
pip install -r requirements.txt -c constraints.txt

# 4. Verify installation
python -c "import torch; import pygame; print('Ready!')"
```

---

## 🚀 Quick Start

### Choose a Game in the Web Launcher

```bash
# Start the web launcher on localhost
python main.py --web

# Expose the dashboard only when the network is trusted
python main.py --web --host 0.0.0.0 --port 5000
```

### Watch AI Learn from Scratch

```bash
# Train with live visualization (default)
python main.py

# Or explicitly with visualization
python main.py --episodes 2000
```

### Load Pre-trained Model

```bash
# Play with a trained model (auto-loads most recent save)
python main.py --play

# Or specify a specific model
python main.py --play --model models/breakout_best.pth
```

### TURBO Mode (Maximum Speed Training)

```bash
# Headless training with optimized settings (~4x faster)
python main.py --headless --turbo --episodes 5000

# Custom performance tuning
python main.py --headless --learn-every 8 --batch-size 128
```

### Human Play Mode (Test the Game)

```bash
python main.py --human
```

---

## 📚 How It Works

### 1. The Game: Breakout (`src/game/breakout.py`)

The game maintains a **state** that the AI reads:

```python
# State representation (what the AI "sees")
state = {
    'ball_x': float,        # Ball X position (normalized 0-1)
    'ball_y': float,        # Ball Y position (normalized 0-1)
    'ball_dx': float,       # Ball X velocity (normalized)
    'ball_dy': float,       # Ball Y velocity (normalized)
    'paddle_x': float,      # Paddle X position (normalized 0-1)
    'bricks': [0,1,1,...],  # Binary array: 0=broken, 1=exists
}
```

**Actions the AI can take:**
- `0` = Move paddle LEFT
- `1` = Stay (no movement)
- `2` = Move paddle RIGHT

### 2. The AI Brain: Deep Q-Network (`src/ai/network.py`)

#### What is Q-Learning?

The AI learns a **Q-function** that estimates "how good" each action is:

```
Q(state, action) = Expected future reward if I take this action
```

#### Network Architecture

```
INPUT LAYER                 HIDDEN LAYERS                 OUTPUT LAYER
┌─────────────┐            ┌─────────────┐               ┌─────────────┐
│ State       │            │  256 neurons │               │ Q(s, LEFT)  │
│ (84 values) │───────────►│  ReLU        │──────────────►│ Q(s, STAY)  │
│             │            │  128 neurons │               │ Q(s, RIGHT) │
└─────────────┘            │  ReLU        │               └─────────────┘
                           └─────────────┘
```

**Input (84 values):**
- Ball position (2)
- Ball velocity (2)
- Paddle position (1)
- Brick states (80 bricks = 80 values, or however many bricks you have)

**Output (3 values):**
- Q-value for LEFT, STAY, RIGHT

The AI picks the action with the **highest Q-value**.

### 3. Learning: Experience Replay & Target Network

#### Why Experience Replay?

Instead of learning from just the last experience, we store experiences in a **replay buffer** and sample random batches. This provides:
- ✅ Better sample efficiency
- ✅ Breaks correlation between consecutive samples
- ✅ More stable learning

```python
# Experience tuple
experience = (state, action, reward, next_state, done)

# Training step
batch = replay_buffer.sample(batch_size=64)
loss = compute_td_loss(batch)
optimizer.step()
```

#### Target Network (Stability)

We maintain TWO copies of the network:
1. **Policy Network** - Updated every step
2. **Target Network** - Updated periodically (every 1000 steps)

This prevents the "moving target" problem where Q-value estimates chase themselves.

### 4. Exploration vs Exploitation

**Default: NoisyNets** (learned exploration via noisy network weights)

The AI learns *when* and *how much* to explore through trainable noise parameters. This provides state-dependent exploration that's often more effective than random exploration.

**Alternative: ε-greedy** (set `USE_NOISY_NETWORKS = False` in config.py)

```
┌────────────────────────────────────────────────────────┐
│  Epsilon (ε) starts at 1.0 (100% random exploration)   │
│  ───────────────────────────────────────────────────►  │
│  Epsilon decays to 0.01 (mostly exploitation)          │
└────────────────────────────────────────────────────────┘

if random() < epsilon:
    action = random_action()      # EXPLORE: try random things
else:
    action = best_q_value_action  # EXPLOIT: use learned policy
```

### 5. The Visualizer (`src/visualizer/nn_visualizer.py`)

The visualizer shows:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
│  │   INPUT       │    │   HIDDEN      │    │   OUTPUT      │       │
│  │   LAYER       │    │   LAYER       │    │   LAYER       │       │
│  │               │    │               │    │               │       │
│  │   ○ ○ ○ ○    │───►│   ◉ ○ ◉ ○    │───►│   ◉ LEFT     │       │
│  │   ○ ○ ○ ○    │    │   ○ ◉ ○ ◉    │    │   ○ STAY     │       │
│  │   ○ ○ ○ ○    │    │   ◉ ○ ◉ ○    │    │   ○ RIGHT    │       │
│  │   ...        │    │   ...        │    │               │       │
│  └───────────────┘    └───────────────┘    └───────────────┘       │
│                                                                      │
│  ◉ = Active (high activation)    ○ = Inactive (low activation)    │
│  Line thickness = connection weight strength                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration Guide

All hyperparameters are in `config.py`:

```python
# Learning Parameters
LEARNING_RATE = 0.0003        # Optimized for faster learning
GAMMA = 0.99                  # Discount factor (0.95-0.99)
BATCH_SIZE = 128              # Samples per training step

# Exploration (NoisyNets is default - these are for epsilon-greedy fallback)
USE_NOISY_NETWORKS = True     # Learned exploration (default)
EPSILON_START = 0.0           # Set to 1.0 if using epsilon-greedy
EPSILON_END = 0.0             # Set to 0.02 if using epsilon-greedy
EPSILON_DECAY = 0.995         # Decay rate per episode

# Network Architecture
HIDDEN_LAYERS = [512, 256, 128]  # Neurons per hidden layer
USE_DUELING = True               # Dueling DQN architecture

# Training
TARGET_TAU = 0.005            # Soft update coefficient
MEMORY_SIZE = 100000          # Replay buffer capacity
SAVE_EVERY = 100              # Episodes between checkpoints

# Advanced Features
USE_PRIORITIZED_REPLAY = True # Sample important experiences more
USE_N_STEP_RETURNS = True     # Multi-step bootstrapping
N_STEP_SIZE = 5               # Steps to look ahead
```

### Tuning Tips

| Symptom | Try This |
|---------|----------|
| AI doesn't learn | Increase `LEARNING_RATE`, check reward function |
| Learning is unstable | Decrease `LEARNING_RATE`, increase `BATCH_SIZE` |
| AI gets stuck in local minimum | Increase `EPSILON_DECAY`, increase exploration |
| Training too slow | Decrease `HIDDEN_LAYERS` size, use GPU |

---

## 🚀 Advanced DQN Features

This implementation includes several modern improvements over vanilla DQN:

### Dueling DQN Architecture

Separates value and advantage estimation for better learning:

```
                              ┌─────────────┐
                         ┌───►│ Value V(s)  │───┐
┌────────┐   ┌────────┐  │    └─────────────┘   │    ┌──────────────┐
│ State  │──►│ Shared │──┤                      ├───►│ Q(s,a) = V + │
│        │   │ Layers │  │    ┌─────────────┐   │    │ (A - mean(A))│
└────────┘   └────────┘  └───►│ Advantage   │───┘    └──────────────┘
                              │ A(s,a)      │
                              └─────────────┘
```

Enable with: `USE_DUELING = True` (default)

### NoisyNets for Exploration

Replaces epsilon-greedy with **learned exploration** through noisy network weights:

```python
# config.py
USE_NOISY_NETWORKS = True   # Enable NoisyNets (default)
EPSILON_START = 0.0         # When using NoisyNets, epsilon is disabled
EPSILON_END = 0.0           # Exploration handled by learned noise parameters
```

**Benefits:**
- ✅ State-dependent exploration (explores more in unfamiliar states)
- ✅ No manual epsilon schedule tuning required
- ✅ Often learns faster than epsilon-greedy

**To use epsilon-greedy instead:**
```python
USE_NOISY_NETWORKS = False
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.995
```

### Prioritized Experience Replay (PER)

Samples important experiences more frequently:

```python
USE_PRIORITIZED_REPLAY = True  # Enable PER (default)
PER_ALPHA = 0.6                # Priority exponent (0=uniform, 1=full priority)
PER_BETA_START = 0.4           # Importance sampling correction
```

### N-Step Returns

Faster reward propagation with multi-step bootstrapping:

```python
USE_N_STEP_RETURNS = True  # Enable N-step (default)
N_STEP_SIZE = 5            # Look ahead 5 steps (faster credit assignment)
```

---

## 👾 Space Invaders Configuration

Space Invaders has game-specific settings optimized for AI learning:

### Game Settings

```python
# Alien grid (5x11 = 55 aliens like the original)
SI_ALIEN_ROWS = 5
SI_ALIEN_COLS = 11

# Movement speed (tuned for AI learning)
SI_ALIEN_SPEED_X = 0.8      # Slower initial speed for learning
SI_ALIEN_SPEED_Y = 10       # Smaller drops = more reaction time

# Combat
SI_MAX_PLAYER_BULLETS = 2   # Limited ammo like original
SI_ALIEN_SHOOT_CHANCE = 0.001
```

### Reward Shaping

```python
SI_REWARD_ALIEN_HIT = 1.0       # Per alien killed
SI_REWARD_UFO_HIT = 5.0         # Bonus UFO
SI_REWARD_PLAYER_DEATH = -5.0   # Death penalty
SI_REWARD_LEVEL_CLEAR = 30.0    # Level completion bonus
SI_REWARD_SHOOT = -0.005        # Small penalty to prevent spam
SI_REWARD_STEP = 0.001          # Survival reward
```

### Curriculum Learning (Optional)

Start with easier settings and gradually increase difficulty:

```python
# Enable progressive difficulty (disabled by default)
SI_CURRICULUM_ENABLED = True

# Stages progress from easy to full game
SI_CURRICULUM_STAGES = [
    {'alien_rows': 2, 'alien_shoot_chance': 0.0005, 'episodes': 500},
    {'alien_rows': 3, 'alien_shoot_chance': 0.0008, 'episodes': 500},
    {'alien_rows': 4, 'alien_shoot_chance': 0.001, 'episodes': 500},
    {'alien_rows': 5, 'alien_shoot_chance': 0.001, 'episodes': None},  # Full game
]
```

### Training Space Invaders

```bash
# Train Space Invaders with visualization
python main.py --game space_invaders

# Headless training (faster)
python main.py --game space_invaders --headless

# Resume from checkpoint
python main.py --game space_invaders --model models/space_invaders/space_invaders_best.pth
```

### Optimized Hyperparameters

These settings are tuned for faster Space Invaders learning:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `LEARNING_RATE` | 0.0003 | 3x faster initial learning |
| `TARGET_TAU` | 0.005 | Faster target network updates |
| `N_STEP_SIZE` | 5 | Better credit assignment |
| `LEARN_EVERY` | 4 | More frequent updates |
| `GRADIENT_STEPS` | 2 | Balanced throughput |
| `EPSILON_WARMUP` | 200 | More diverse initial experiences |

---

## 📈 Benchmarking & Performance

### Running Benchmarks

The project includes a comprehensive benchmarking suite to measure training performance:

```bash
# Quick benchmark (3 seconds, small buffer)
python benchmark.py --quick

# Standard benchmark
python benchmark.py

# Realistic benchmark (50k buffer, production-like)
python benchmark.py --realistic

# Test buffer size scaling
python benchmark.py --scaling

# Full benchmark suite
python benchmark.py --full

# List all available configurations
python benchmark.py --list
```

### Benchmark Output

```
============================================================
DQN Training Performance Benchmark
============================================================

System: macOS-26.2-arm64-arm-64bit-Mach-O
Python: 3.13.5 | PyTorch: 2.9.1
CUDA: ✗
MPS: ✓

──────────────────────────────────────────────────
Running: turbo

  📊 turbo
     Device: cpu | Batch: 128 | LE: 8 | GS: 2
     Buffer: 50,000 experiences
     Steps: 10,520 in 5.0s → 2,103 steps/sec
     Gradients: 2,630 → 526 grad/sec
```

### Benchmark Configurations

| Config | Device | Batch | Buffer | Use Case |
|--------|--------|-------|--------|----------|
| `turbo` | CPU | 128 | small | Quick tests, max throughput |
| `cpu_b128` | CPU | 128 | small | Standard CPU training |
| `mps_b256` | MPS | 256 | small | Apple Silicon GPU |
| `realistic_cpu` | CPU | 128 | 50k | Production-like performance |
| `realistic_turbo` | CPU | 128 | 50k | Fast production training |
| `scale_*` | CPU | 128 | 1k-100k | Buffer size impact analysis |

### Performance Optimizations

The codebase includes several performance optimizations:

#### 1. Adaptive Sampling (30x faster for large buffers)
```python
# Small buffers (<3k): Uses np.random.choice (faster)
# Large buffers (>3k): Uses random.sample (30x faster)
```

#### 2. Pre-allocated Tensors
- State tensors reused across steps (no allocation per action)
- Batch tensors pre-allocated for training

#### 3. Efficient Loss Calculation
```python
# O(n) iteration instead of O(buffer_size) list conversion
def get_average_loss(self, n=100):
    it = iter(reversed(self.losses))
    # Only iterates last n items, not entire deque
```

#### 4. Cached Predictions
- `_predict_landing_x()` cached per step to avoid redundant computation

### Performance Tips

| Goal | Recommendation |
|------|----------------|
| Maximum speed | Use `--headless --turbo` flags |
| CPU training | Set `FORCE_CPU = True` (faster than MPS for small models) |
| Large buffer | Adaptive sampling automatically optimizes |
| GPU training | Use larger batch sizes (256-512) to amortize transfer overhead |

### Buffer Size Impact

Buffer size significantly affects performance due to sampling algorithms:

```
Buffer Size    Steps/sec    Sampling Method
       1,000      2,100     np.random.choice (fast for small)
       5,000      2,500     random.sample (fast for large)
      50,000      2,100     random.sample (30x faster than np.random.choice)
     100,000      2,400     random.sample (35x faster than np.random.choice)
```

---

## 🔒 Safety Notes

The web dashboard starts on `127.0.0.1` by default. Use `--host 0.0.0.0` only on a trusted network because the dashboard can pause training, save models, start fresh, load checkpoints, and delete saved model files.

Each dashboard session generates an access token and prints a tokenized dashboard URL. Open that full URL; anonymous requests to `/` are rejected before the page can bootstrap control access. If you need a stable token for automation, set `NN_GAME_DASHBOARD_TOKEN` before launching.

Checkpoint loading uses PyTorch's restricted loader first. Legacy checkpoints that require unrestricted pickle loading should only be loaded from model directories you trust.

---

## 🛠️ Developer Checks

```bash
# Fast test run
make test

# Coverage with the current ratcheting threshold
make coverage

# Strict type check for the full Python app
make typecheck

# Formatting
make format
make format-check

# Fast local gate used by CI
make check

# Fuller local verification, including release config, hygiene, and build when available
make verify
```

The project supports Python 3.10 through 3.12. Install with pinned constraints for reproducible local behavior:

```bash
pip install -r requirements.txt -c constraints.txt
```

---

## 🎮 Extending to Other Games

This architecture is designed to be **game-agnostic**. To add a new game:

### 1. Create a New Game Class

```python
# src/game/your_game.py
from src.game.base_game import BaseGame

class YourGame(BaseGame):
    def __init__(self):
        self.state_size = ...    # How many input values
        self.action_size = ...   # How many possible actions
    
    def reset(self) -> np.ndarray:
        """Reset game and return initial state"""
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action, return (next_state, reward, done)"""
        pass
    
    def render(self, screen):
        """Draw game to pygame screen"""
        pass
```

### 2. Register in Config

```python
# config.py
GAME = "your_game"  # Changed from "breakout"
```

### 3. That's It!

The agent, visualizer, and training loop work with any game that follows the `BaseGame` interface.

---

## 🐛 Troubleshooting

### Common Issues

**"CUDA out of memory"**
```bash
# Force CPU training
python main.py --device cpu
```

**"Pygame window not responding"**
- Make sure you're not running in a headless environment
- Try: `export SDL_VIDEODRIVER=x11`

**"Training seems stuck"**
- Check if epsilon is decaying (should decrease over time)
- Verify rewards are non-zero (add print statements)
- Try different random seeds: `python main.py --seed 42`

**"AI only moves one direction"**
- Reward function may be imbalanced
- Try longer training (at least 1000 episodes)
- Check state normalization

---

## 📊 Understanding the Metrics

During training, you'll see:

```
Episode: 100 | Score: 4 | Avg: 2.5 | ε: 0.45 | Loss: 0.023
        │       │        │         │         │
        │       │        │         │         └── TD Error (should decrease)
        │       │        │         └── Exploration rate (should decrease)
        │       │        └── Running average score (should increase)
        │       └── Score this episode
        └── Episode number
```

### What Good Training Looks Like

1. **Episodes 1-100:** Random movement, low scores, high exploration
2. **Episodes 100-500:** AI starts tracking ball, scores improve
3. **Episodes 500-1000:** Consistent improvement, exploration drops
4. **Episodes 1000+:** Mastery, high scores, minimal exploration

---

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new games
- Improve visualizations
- Try different RL algorithms (A3C, PPO, etc.)
- Optimize performance

---

## 📜 License

MIT License - Use this for learning and teaching!

---

## 🙏 Acknowledgments

- DeepMind's DQN paper (Mnih et al., 2015)
- OpenAI Gym for inspiration
- Pygame community

---

**Happy Learning! 🚀**
