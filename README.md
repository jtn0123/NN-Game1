# 🧠 Neural Network Game AI - Atari Breakout

A complete, educational implementation of a Deep Q-Learning (DQN) agent that learns to play Atari Breakout **in real-time** with a **live neural network visualizer**.

![Project Architecture](docs/architecture.png)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture Deep Dive](#-architecture-deep-dive)
3. [Installation](#-installation)
4. [Quick Start](#-quick-start)
5. [How It Works](#-how-it-works)
6. [Configuration Guide](#-configuration-guide)
7. [Extending to Other Games](#-extending-to-other-games)
8. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

### What This Project Does

This project demonstrates **reinforcement learning** by training a neural network to play Breakout:

1. **The Game** (`src/game/`) - A complete Atari Breakout implementation
2. **The AI Brain** (`src/ai/`) - Deep Q-Network (DQN) that learns to play
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
│   │   ├── breakout.py          # Game logic (state, physics, scoring)
│   │   └── renderer.py          # Pygame rendering
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

- Python 3.9+ (tested with 3.11)
- pip package manager

### Setup Steps

```bash
# 1. Clone or navigate to the project
cd /Users/justin/Documents/Github/NN-Game1

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; import pygame; print('Ready!')"
```

---

## 🚀 Quick Start

### Watch AI Learn from Scratch

```bash
python main.py --mode train --visualize
```

### Load Pre-trained Model

```bash
python main.py --mode play --model models/breakout_best.pth
```

### Train Without Visualization (Faster)

```bash
python main.py --mode train --headless
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

### 4. Exploration vs Exploitation (ε-greedy)

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
LEARNING_RATE = 0.0001        # Higher = faster but unstable
GAMMA = 0.99                   # Discount factor (0.95-0.99)
BATCH_SIZE = 64               # Samples per training step

# Exploration
EPSILON_START = 1.0           # Initial exploration rate
EPSILON_END = 0.01            # Minimum exploration rate
EPSILON_DECAY = 0.995         # Decay rate per episode

# Network Architecture
HIDDEN_LAYERS = [256, 128]    # Neurons per hidden layer

# Training
TARGET_UPDATE = 1000          # Steps between target network updates
MEMORY_SIZE = 100000          # Replay buffer capacity
SAVE_EVERY = 100              # Episodes between checkpoints
```

### Tuning Tips

| Symptom | Try This |
|---------|----------|
| AI doesn't learn | Increase `LEARNING_RATE`, check reward function |
| Learning is unstable | Decrease `LEARNING_RATE`, increase `BATCH_SIZE` |
| AI gets stuck in local minimum | Increase `EPSILON_DECAY`, increase exploration |
| Training too slow | Decrease `HIDDEN_LAYERS` size, use GPU |

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

