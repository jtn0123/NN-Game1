# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Deep Q-Learning (DQN) agent** that learns to play Atari Breakout in real-time with live neural network visualization. The project demonstrates reinforcement learning through an educational, modular implementation.

**Key Components:**
- `src/game/` - Breakout game implementation
- `src/ai/` - DQN agent, neural network, replay buffer, trainer
- `src/visualizer/` - Real-time neural network visualization + metrics dashboard
- `src/web/` - Optional Flask-based web dashboard for training monitoring

## Architecture

### High-Level Flow

```
Game State → DQN Agent → Action Selection → Reward/Next State
                          ↓
                    Replay Buffer
                          ↓
                    Train Network
                          ↓
                      Visualizer Display
```

### Key Architectural Decisions

1. **DQN Architecture:**
   - Input layer: State representation (typically 55-84 values depending on brick configuration)
   - Hidden layers: Configurable (default [256, 128] neurons)
   - Output layer: Q-values for 3 actions (LEFT, STAY, RIGHT)
   - Uses Adam optimizer with gradient clipping

2. **Experience Replay:**
   - ReplayBuffer stores (state, action, reward, next_state, done) tuples
   - Supports both uniform and prioritized sampling (PER)
   - Capacity: 100,000 experiences (configurable)

3. **Target Network:**
   - Two identical networks: policy_net (trained) and target_net (frozen)
   - Target network updated every 1000 steps to stabilize learning
   - Prevents "moving target" problem in Q-value estimation

4. **Exploration Strategy:**
   - Epsilon-greedy with configurable decay (exponential, linear, cosine)
   - Starts at 1.0 (100% random), decays to 0.01 (1% random)
   - Helps agent balance exploration vs exploitation

5. **Visualization:**
   - NeuralNetVisualizer renders network structure with live neuron activations
   - DataFlowPulse objects animate data flow through connections
   - Dashboard shows training metrics (loss, reward, epsilon decay)

### Game State Representation

The game state has 55+ features:
- Ball position (x, y) - normalized 0-1
- Ball velocity (dx, dy) - normalized
- Paddle position (x) - normalized 0-1
- Brick states - binary array (50 bricks = 50 values for 5×10 grid)

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python config.py
```

### Running the Application

```bash
# Train with visualization (default)
python main.py

# Train without visualization (faster)
python main.py --headless

# Play with a trained model
python main.py --play --model models/breakout_best.pth

# Human player mode (test the game)
python main.py --human

# Custom training parameters
python main.py --episodes 5000 --lr 0.0001

# Train on CPU (if CUDA causes issues)
python main.py --device cpu
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py -v

# Run with coverage
pytest --cov=src tests/

# Run single test
pytest tests/test_agent.py::TestAgent::test_select_action -v
```

### Code Quality

```bash
# Type checking
mypy --config-file=mypy.ini src/

# Code formatting
black src/ tests/ *.py

# Format check only (don't modify)
black --check src/ tests/ *.py
```

### Configuration

All hyperparameters are in `config.py`. Key sections:
- **Game Settings** - Screen dimensions, paddle, ball, bricks
- **Neural Network** - Architecture, hidden layers, activation functions
- **Training Hyperparameters** - Learning rate, gamma, batch size, memory
- **Exploration Settings** - Epsilon decay, exploration strategy
- **Reward Shaping** - Event-based rewards (brick hits, game over, win)
- **Visualization** - Display colors, neuron visualization parameters

**Quick Tuning Tips:**
| Symptom | Action |
|---------|--------|
| AI doesn't learn | ↑ LEARNING_RATE, check REWARD values |
| Unstable training | ↓ LEARNING_RATE, ↑ BATCH_SIZE |
| Gets stuck | ↑ EPSILON_DECAY (slower decay), ↓ EXPLORATION_STRATEGY |
| Training too slow | ↓ HIDDEN_LAYERS size, use GPU |

## Key Files & Modules

### Core AI (`src/ai/`)
- **agent.py** - DQN Agent class: select_action(), remember(), learn()
- **network.py** - DQN neural network using PyTorch
- **replay_buffer.py** - Experience replay with optional prioritization (PER)
- **trainer.py** - Training loop orchestration

### Game (`src/game/`)
- **breakout.py** - Main Breakout game: reset(), step(), get_state(), render()
- **base_game.py** - Abstract base class for game interface
- **renderer.py** - Pygame rendering utilities
- **particles.py** - Particle effects

### Visualization (`src/visualizer/`)
- **nn_visualizer.py** - Network visualization with DataFlowPulse animation
- **dashboard.py** - Training metrics plots (loss, reward, epsilon)

### Main Entry Point
- **main.py** - GameApp class managing pygame loop, game, agent, and visualizations
- **config.py** - Global configuration with centralized hyperparameters

## Testing Strategy

Tests are organized by component:
- **test_game.py** - Breakout game: initialization, state shape, actions, physics, rewards
- **test_agent.py** - DQN agent: action selection, learning, epsilon decay, memory management
- **test_network.py** - Neural network: forward pass, output shapes, gradient flow

**Key assertion patterns:**
- State shape validation
- Reward calculation correctness
- Epsilon decay behavior (exponential, linear, cosine)
- Gradient updates occur during learning
- Device handling (CPU/GPU/MPS)

**Note:** All tests must pass. Do not ignore failing tests - investigate and fix the root cause.

## Pygame Integration

The application uses Pygame 2.5+ with event handling:
- **ESC/Q** - Quit
- **P** - Pause/Resume
- **S** - Save model
- **R** - Reset episode
- **+/-** - Speed adjustment
- **F** - Fullscreen toggle

Window features:
- Live game rendering on left side
- Neural network visualization on right
- Training dashboard at bottom
- Resizable window support
- Fullscreen mode support

## Model Management

**Saving & Loading:**
```python
# Saving (happens automatically every N episodes)
torch.save(agent.policy_net.state_dict(), 'models/breakout_best.pth')

# Loading
agent.policy_net.load_state_dict(torch.load('models/breakout_best.pth'))
```

**Model Directory:** `models/` - Git ignores .pth files; add them locally

## GPU/Device Handling

Device selection is automatic in config.py:
1. Check for CUDA availability
2. Fall back to MPS (Apple Silicon) if available
3. Default to CPU

**Force specific device:**
```bash
python main.py --device cpu
python main.py --device cuda
```

## Extending the Project

To add a new game:
1. Inherit from `BaseGame` in `src/game/base_game.py`
2. Implement: `__init__()`, `reset()`, `step()`, `get_state()`, `render()`
3. Set `STATE_SIZE` and `ACTION_SIZE` properties
4. Update config to point to new game class
5. Agent, visualizer, and trainer work automatically with the new game

## Performance Notes

### Turbo Mode (`--turbo`)

The `--turbo` flag applies optimized settings based on M4 Mac benchmarks:

```bash
# Headless turbo mode (fastest, ~5000 steps/sec on M4)
python main.py --headless --turbo

# Visualized turbo mode (uses same settings but with rendering overhead)
python main.py --turbo
```

Turbo mode settings:
- `LEARN_EVERY=8` - Learn every 8th step (vs default 4)
- `GRADIENT_STEPS=2` - Multiple gradient updates per learning call
- `BATCH_SIZE=128` - Optimized for CPU throughput
- `FORCE_CPU=True` - Uses CPU instead of MPS (faster for small models!)
- `USE_TORCH_COMPILE=False` - Disabled (no benefit for small models)

### Headless vs Visualized Training

| Mode | Speed | Use Case |
|------|-------|----------|
| `--headless --turbo` | ~5000 steps/sec | Fast bulk training |
| `--headless` | ~2500 steps/sec | Training without visuals |
| `--turbo` (visualized) | ~600 steps/sec | Turbo with live feedback |
| Default (visualized) | ~300 steps/sec | Educational/debugging |

**Note:** `--headless` skips particle effects, ball trails, and all rendering. The game physics remain identical.

### M4 MacBook Benchmark Results

```
CPU B=128, LE=8, GS=2:  ~5,000 steps/sec, 663 grad/sec (balanced - RECOMMENDED)
CPU B=128, LE=16, GS=4: ~2,900 steps/sec, 719 grad/sec (max learning throughput)
MPS B=256, LE=4:        ~640 steps/sec (GPU overhead dominates)
```

**Key Finding:** For this model size (~50K parameters), CPU is 8x faster than MPS on M4 due to:
- Small batch sizes don't utilize GPU parallelism efficiently
- CPU-GPU memory transfer overhead dominates computation time
- MPS setup/sync costs are significant for short operations

**When to use MPS/CUDA:**
- Much larger networks (millions of parameters)
- Batch sizes of 1024+
- Longer forward/backward pass times that amortize transfer costs

### Manual CPU/GPU Selection

```bash
# Force CPU (recommended for this model)
python main.py --cpu

# Force MPS (Apple Silicon GPU) - only if network is large
python main.py --device mps

# Force CUDA (NVIDIA GPU)
python main.py --device cuda
```

### Other Performance Tips

- **Training without visualization** is ~2-3x faster (`--headless` flag)
- **Prioritized Experience Replay (PER)** improves sample efficiency but adds overhead
- **GPU acceleration** (CUDA/MPS) only beneficial for large models
- **Visualization updates** every frame during training (impacts performance)

## Web Dashboard

Optional Flask-based dashboard available if flask/flask-socketio installed:
- Real-time training metrics over WebSocket
- Model management UI
- Episode replay capabilities
- Configure port in main.py (default: 5000)

## Common Issues

**Pygame window unresponsive:**
- Ensure not in headless environment
- Try: `export SDL_VIDEODRIVER=x11`

**CUDA out of memory:**
- Use `--device cpu` flag
- Reduce `BATCH_SIZE` in config
- Reduce `HIDDEN_LAYERS` size

**Training stuck at random actions:**
- Verify rewards are non-zero
- Check epsilon is decaying (`epsilon` should decrease over episodes)
- Try longer training (1000+ episodes)
- Verify state normalization is correct

**Models not saving:**
- Ensure `models/` directory exists
- Check file permissions
- Verify disk space

## Type Checking & Code Quality

- **Type hints:** All function signatures should include type hints
- **mypy config:** Disables `disallow_untyped_defs` for flexibility but enables return type warnings
- **Black formatting:** 88-character line length
- **No compiler warnings:** Remove unused imports, dead code; prefix intentional unused vars with `_`

## Recent Project Evolution

Recent commits have added:
- Extended training metrics and logging
- Reward parameter tracking
- Fullscreen and window resizing support
- Improved type hints across all modules
- Replay buffer enhancements with type annotations
- Web dashboard integration

Check git log for commit messages explaining architectural changes.
