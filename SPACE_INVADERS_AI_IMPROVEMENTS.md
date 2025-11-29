# Space Invaders Neural Network Improvement Proposals

## Current Status

The model is plateauing around **2000-3000 average score** with decent Q-values and loss metrics. This document proposes backwards-compatible improvements to break through this ceiling while maintaining the existing Python codebase structure.

---

## Table of Contents

1. [Priority 1: Quick Wins (High Impact, Low Effort)](#priority-1-quick-wins)
2. [Priority 2: Learning Dynamics](#priority-2-learning-dynamics)
3. [Priority 3: Exploration Improvements](#priority-3-exploration-improvements)
4. [Priority 4: State Representation](#priority-4-state-representation)
5. [Priority 5: Reward Shaping](#priority-5-reward-shaping)
6. [Priority 6: Advanced Techniques](#priority-6-advanced-techniques)
7. [Implementation Checklist](#implementation-checklist)
8. [Testing Protocol](#testing-protocol)

---

## Priority 1: Quick Wins

### 1.1 Enable Prioritized Experience Replay (PER)

**Current:** `USE_PRIORITIZED_REPLAY: bool = False`

**Problem:** Uniform sampling means the agent treats all experiences equally. Critical learning moments (dying, killing high-value aliens, close calls) are sampled with the same frequency as mundane steps.

**Solution:** Enable PER in `config.py`:

```python
# config.py
USE_PRIORITIZED_REPLAY: bool = True
PER_ALPHA: float = 0.6  # Keep as-is
PER_BETA_START: float = 0.4  # Keep as-is  
PER_BETA_FRAMES: int = 200000  # Increase from 100000 for slower annealing
```

**Expected Impact:** 30-40% faster learning, better sample efficiency.

**Implementation:** Already implemented in `replay_buffer.py` - just enable the flag.

---

### 1.2 Adjust Epsilon Decay Schedule

**Current:** 
- `EPSILON_DECAY: 0.995` (per episode)
- `EPSILON_END: 0.005`

**Problem:** With 0.995 decay, epsilon reaches ~0.01 by episode 460. This might be too fast for Space Invaders which has more complex dynamics than Breakout.

**Solution:** Slower decay with higher minimum:

```python
# config.py
EPSILON_DECAY: float = 0.998  # Slower decay (was 0.995)
EPSILON_END: float = 0.02     # Higher minimum (was 0.005)
EPSILON_WARMUP: int = 100     # Delay decay for 100 episodes (was 0)
```

**Rationale:** 
- Reach 0.02 around episode 1200 instead of 460
- 2% minimum exploration helps escape local optima
- Warmup allows memory buffer to fill with diverse experiences

---

### 1.3 Increase Target Network Update Frequency

**Current:** 
- `USE_SOFT_UPDATE: True`
- `TARGET_TAU: 0.005`

**Problem:** TAU of 0.005 is relatively aggressive. For complex games, slower updates can improve stability.

**Solution:**

```python
# config.py
TARGET_TAU: float = 0.001  # More stable (was 0.005)
```

**Alternative (Hard Updates):**
```python
USE_SOFT_UPDATE: bool = False
TARGET_UPDATE: int = 2000  # Increase from 1000
```

---

## Priority 2: Learning Dynamics

### 2.1 Learning Rate Scheduling

**Problem:** Constant learning rate of 0.0001 may be too high once the model converges to a local optimum, causing oscillation.

**Solution:** Add a learning rate scheduler to `agent.py`:

```python
# In Agent.__init__(), after optimizer creation:
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Option A: Step decay every 500 episodes
self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.5)

# Option B: Cosine annealing (smoother)
self.scheduler = CosineAnnealingLR(
    self.optimizer, 
    T_max=2000,  # Episodes until minimum LR
    eta_min=1e-5  # Minimum LR
)

# Call after each episode in training loop:
def step_scheduler(self):
    self.scheduler.step()
```

**Config additions:**

```python
# config.py
USE_LR_SCHEDULER: bool = True
LR_SCHEDULER_TYPE: str = 'cosine'  # 'step' or 'cosine'
LR_SCHEDULER_STEP: int = 500       # For step scheduler
LR_SCHEDULER_GAMMA: float = 0.5    # Decay factor for step
LR_MIN: float = 1e-5               # Minimum learning rate
```

---

### 2.2 Gradient Accumulation for Larger Effective Batch Size

**Current:** `BATCH_SIZE: 128`, `GRADIENT_STEPS: 2`

**Problem:** Small effective batch sizes can lead to noisy gradients.

**Solution:** Increase gradient steps without slowing down:

```python
# config.py
BATCH_SIZE: int = 128      # Keep for memory efficiency
GRADIENT_STEPS: int = 4    # Double gradient steps (was 2)
LEARN_EVERY: int = 16      # Adjust to maintain throughput (was 8)
```

This effectively gives you 512 samples worth of gradient signal per learning cycle.

---

### 2.3 Adjust Discount Factor (Gamma)

**Current:** `GAMMA: 0.97`

**Problem:** For Space Invaders where survival and level progression matter long-term, 0.97 might be too shortsighted. However, going too high can slow learning.

**Solution:** 

```python
# config.py  
GAMMA: float = 0.99  # More far-sighted (was 0.97)
```

**Alternatively, implement N-step returns (see Priority 6).**

---

## Priority 3: Exploration Improvements

### 3.1 Noisy Networks (NoisyNet)

**Problem:** ε-greedy exploration is random and undirected. The agent might not explore strategically.

**Solution:** Add parameter noise for directed exploration:

```python
# network.py - New NoisyLinear layer

import math
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """Linear layer with learnable parameter noise for exploration."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
```

**Config addition:**

```python
# config.py
USE_NOISY_NETWORKS: bool = True
NOISY_STD_INIT: float = 0.5
```

**When using NoisyNet, set `EPSILON_END = 0.0` as the network handles exploration internally.**

---

### 3.2 Action Repeat Penalty

**Problem:** The agent might repeatedly take the same action (spam shooting or oscillating left-right).

**Solution:** Add a small penalty for action repetition in the reward:

```python
# space_invaders.py - In step() method

# Track last N actions
if not hasattr(self, '_action_history'):
    self._action_history = []

self._action_history.append(action)
if len(self._action_history) > 10:
    self._action_history.pop(0)

# Penalize repetitive actions
if len(self._action_history) >= 5:
    unique_ratio = len(set(self._action_history[-5:])) / 5.0
    if unique_ratio < 0.4:  # Less than 2 unique actions in last 5
        reward -= 0.02  # Small repetition penalty
```

---

## Priority 4: State Representation

### 4.1 Add Missing State Information

**Current state includes:**
- Ship X position
- Player bullet positions (2 bullets × 2 coords)
- Alien alive states (55 values)
- Alien movement direction and offset
- Danger metrics (nearest bullet, lowest alien)

**Missing information that could help:**

```python
# space_invaders.py - Enhanced get_state()

def get_state(self) -> np.ndarray:
    # ... existing code ...
    
    # ADD: Shoot cooldown state (helps agent time shots)
    self._state_array[idx] = self._shoot_cooldown / self._shoot_cooldown_max
    idx += 1
    
    # ADD: Number of active alien bullets (threat level)
    active_alien_bullets = sum(1 for b in self.alien_bullets if b.alive)
    self._state_array[idx] = min(active_alien_bullets / 10.0, 1.0)  # Normalized
    idx += 1
    
    # ADD: Aliens remaining ratio (progress indicator)
    self._state_array[idx] = self._aliens_remaining / self._num_aliens
    idx += 1
    
    # ADD: Lives remaining (risk awareness)
    self._state_array[idx] = self.lives / self.config.LIVES
    idx += 1
    
    # ADD: Level (difficulty awareness)
    self._state_array[idx] = min(self.level / 10.0, 1.0)
    idx += 1
    
    # ADD: UFO presence and position
    if self.ufo is not None and self.ufo.alive:
        self._state_array[idx] = self.ufo.x * self._inv_width
        self._state_array[idx + 1] = 1.0  # UFO present flag
    else:
        self._state_array[idx] = 0.5
        self._state_array[idx + 1] = 0.0
    idx += 2
```

**Update STATE_SIZE:**

```python
# Update _state_size calculation in __init__
self._state_size = (1 + self._max_player_bullets * 2 + self._num_aliens + 5 
                    + 7)  # +7 for new features
```

**IMPORTANT:** After changing state size, you **cannot** load old models. Either:
1. Start fresh training
2. Implement state adapter that pads old states with zeros

---

### 4.2 Frame Stacking (Optional, Higher Complexity)

**Problem:** Single-frame state doesn't capture velocity/acceleration of objects.

**Solution:** Stack last N frames:

```python
# space_invaders.py - Add frame stacking

from collections import deque

class SpaceInvaders(BaseGame):
    def __init__(self, ...):
        # ... existing code ...
        
        # Frame stacking
        self._frame_stack_size = 4
        self._frame_stack = deque(maxlen=self._frame_stack_size)
        
        # Update state size
        self._single_frame_size = self._state_size
        self._state_size = self._single_frame_size * self._frame_stack_size
        self._state_array = np.zeros(self._state_size, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        # ... existing reset code ...
        
        # Initialize frame stack with copies of initial state
        initial_state = self._get_single_frame_state()
        for _ in range(self._frame_stack_size):
            self._frame_stack.append(initial_state.copy())
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        current_frame = self._get_single_frame_state()
        self._frame_stack.append(current_frame)
        
        # Concatenate all frames
        return np.concatenate(list(self._frame_stack))
    
    def _get_single_frame_state(self) -> np.ndarray:
        # Original get_state logic here
        ...
```

**Note:** This significantly increases state size and network complexity. Only implement if other improvements don't help.

---

## Priority 5: Reward Shaping

### 5.1 Progressive Alien Kill Bonus

**Problem:** All aliens give the same AI reward (`SI_REWARD_ALIEN_HIT = 1.0`), but strategically, killing specific aliens is more valuable.

**Solution:** Add progressive and positional bonuses:

```python
# space_invaders.py - In _handle_collisions(), alien kill section

# Base reward
reward += self.config.SI_REWARD_ALIEN_HIT

# Progressive bonus: more points as fewer aliens remain
progress_bonus = 0.5 * (1 - self._aliens_remaining / self._num_aliens)
reward += progress_bonus

# Column-clear bonus: reward for clearing columns (reduces threat)
col = idx % self.config.SI_ALIEN_COLS
aliens_in_col = sum(1 for i, a in enumerate(self.aliens) 
                    if a.alive and i % self.config.SI_ALIEN_COLS == col)
if aliens_in_col == 0:  # Just cleared the column
    reward += 2.0  # Column clear bonus

# Edge alien bonus: killing edge aliens slows horizontal movement
if col == 0 or col == self.config.SI_ALIEN_COLS - 1:
    reward += 0.3  # Edge alien bonus
```

---

### 5.2 Accuracy Bonus

**Problem:** Agent might spam shots without aiming.

**Solution:** Track and reward accuracy:

```python
# space_invaders.py

def __init__(self, ...):
    # ... existing code ...
    self._shots_fired = 0
    self._shots_hit = 0

def _fire_player_bullet(self):
    # ... existing code ...
    self._shots_fired += 1

def _handle_collisions(self):
    # In alien hit section:
    self._shots_hit += 1
    
    # Calculate accuracy bonus
    if self._shots_fired > 10:
        accuracy = self._shots_hit / self._shots_fired
        if accuracy > 0.3:  # Above 30% accuracy
            reward += 0.5 * accuracy  # Up to 0.5 bonus
```

---

### 5.3 Survival Scaling

**Problem:** Fixed step reward doesn't incentivize survival as danger increases.

**Solution:** Scale survival reward with danger level:

```python
# space_invaders.py - In step()

# Dynamic survival reward based on threat level
threat_level = 0.0

# Factor 1: How low are the aliens?
lowest_ratio = lowest_alien_y / self.ground_y
threat_level += lowest_ratio * 0.5

# Factor 2: How many alien bullets are active?
bullet_threat = min(len(self.alien_bullets) / 5.0, 1.0)
threat_level += bullet_threat * 0.3

# Factor 3: How few lives remain?
life_threat = 1 - (self.lives / self.config.LIVES)
threat_level += life_threat * 0.2

# Survival reward scales with threat
base_step_reward = self.config.SI_REWARD_STEP
reward += base_step_reward * (1 + threat_level * 2)  # Up to 3x step reward
```

---

## Priority 6: Advanced Techniques

### 6.1 N-Step Returns

**Problem:** Single-step TD updates can be slow to propagate rewards backwards through time.

**Solution:** Implement N-step returns for faster credit assignment:

```python
# replay_buffer.py - Add NStepReplayBuffer

class NStepReplayBuffer(ReplayBuffer):
    """Replay buffer with N-step returns for faster reward propagation."""
    
    def __init__(self, capacity: int, state_size: int = 0, 
                 n_steps: int = 3, gamma: float = 0.99):
        super().__init__(capacity, state_size)
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Temporary storage for N-step calculation
        self._n_step_buffer = []
    
    def push(self, state, action, reward, next_state, done):
        # Store in temporary buffer
        self._n_step_buffer.append((state, action, reward, next_state, done))
        
        # If episode ended or buffer full, compute N-step returns
        if done or len(self._n_step_buffer) >= self.n_steps:
            self._flush_n_step_buffer(done)
    
    def _flush_n_step_buffer(self, done: bool):
        """Compute and store N-step experiences."""
        if not self._n_step_buffer:
            return
        
        # Compute discounted N-step return
        n = len(self._n_step_buffer)
        
        for i in range(n):
            state, action, _, _, _ = self._n_step_buffer[i]
            
            # Sum discounted rewards
            n_step_reward = 0.0
            for j in range(i, n):
                _, _, r, _, d = self._n_step_buffer[j]
                n_step_reward += (self.gamma ** (j - i)) * r
                if d:
                    break
            
            # Get the N-step next state
            final_idx = min(i + self.n_steps - 1, n - 1)
            _, _, _, n_step_next_state, n_step_done = self._n_step_buffer[final_idx]
            
            # Store the N-step experience
            super().push(state, action, n_step_reward, n_step_next_state, n_step_done)
        
        self._n_step_buffer.clear()
```

**Config:**

```python
# config.py
USE_N_STEP_RETURNS: bool = True
N_STEP_SIZE: int = 3
```

**Update Q-value computation in agent:**

```python
# agent.py - In _compute_q_values

# For N-step returns, gamma becomes gamma^n
effective_gamma = self.config.GAMMA ** self.config.N_STEP_SIZE
target_q = rewards + (1 - dones) * effective_gamma * next_q
```

---

### 6.2 Distributional DQN (C51)

**Problem:** Standard DQN learns expected Q-values, but the distribution of returns matters.

**Solution:** This is a significant change. Add to `network.py`:

```python
# network.py - DistributionalDQN

class DistributionalDQN(nn.Module):
    """
    Categorical DQN (C51) - learns distribution of returns.
    
    Instead of Q(s,a) = E[R], learns P(R|s,a) over discrete atoms.
    """
    
    def __init__(self, state_size, action_size, config, 
                 num_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        super().__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.register_buffer('atoms', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Network outputs logits for each action-atom pair
        hidden = config.HIDDEN_LAYERS
        
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        
        self.value = nn.Linear(hidden[1], num_atoms)
        self.advantage = nn.Linear(hidden[1], action_size * num_atoms)
    
    def forward(self, x):
        features = self.feature(x)
        
        value = self.value(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage(features).view(-1, self.action_size, self.num_atoms)
        
        # Dueling combination
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Softmax over atoms to get probabilities
        return F.softmax(q_atoms, dim=2)
    
    def get_q_values(self, x):
        """Get expected Q-values from distribution."""
        dist = self.forward(x)
        return (dist * self.atoms).sum(dim=2)
```

**Note:** C51 requires significant changes to the training loop. Only implement if simpler methods don't work.

---

### 6.3 Double Dueling with PER (Rainbow-lite)

**Currently implemented:**
- ✅ Double DQN
- ✅ Dueling Architecture
- ✅ PER (disabled by default)

**To enable full Rainbow-lite:**

```python
# config.py - Enable all improvements
USE_DUELING: bool = True           # ✅ Already True
USE_PRIORITIZED_REPLAY: bool = True  # Enable this
USE_N_STEP_RETURNS: bool = True      # Add this (Priority 6.1)
USE_NOISY_NETWORKS: bool = True      # Add this (Priority 3.1)
```

---

## Implementation Checklist

### Phase 1: Quick Wins (Do First)

- [ ] Enable `USE_PRIORITIZED_REPLAY = True` in config.py
- [ ] Adjust epsilon decay: `EPSILON_DECAY = 0.998`, `EPSILON_END = 0.02`
- [ ] Reduce target update rate: `TARGET_TAU = 0.001`
- [ ] Increase gamma: `GAMMA = 0.99`

### Phase 2: Learning Improvements

- [ ] Implement learning rate scheduler in agent.py
- [ ] Add config options for LR scheduling
- [ ] Increase `GRADIENT_STEPS = 4`, adjust `LEARN_EVERY = 16`

### Phase 3: State & Reward

- [ ] Add missing state features (cooldown, bullet count, lives, level, UFO)
- [ ] Update `_state_size` calculation
- [ ] Add progressive alien kill bonuses
- [ ] Add accuracy tracking and bonus
- [ ] Add survival scaling with threat level

### Phase 4: Advanced (If Still Plateauing)

- [ ] Implement NoisyLinear layer
- [ ] Add NoisyNet flag and integration
- [ ] Implement N-step returns buffer
- [ ] Consider frame stacking if temporal info helps

---

## Testing Protocol

### Baseline Measurement

Before making changes, record over 100 episodes:
- Average score
- Max score
- Average episode length
- Win rate (levels completed)
- Average Q-value

### A/B Testing

1. **Single Variable Changes:** Change one parameter at a time
2. **Run Duration:** At least 1000 episodes per configuration
3. **Multiple Seeds:** Test with 3 different random seeds
4. **Metrics to Track:**
   - Score rolling average (100 episodes)
   - Episode length rolling average
   - Loss convergence
   - Q-value trends
   - Epsilon at convergence

### Expected Results

| Change | Expected Improvement |
|--------|---------------------|
| Enable PER | +20-40% learning speed |
| Slower epsilon | +10-20% final performance |
| LR scheduling | +5-15% final performance |
| Increased gamma | +10-25% long-term play |
| State improvements | +15-30% decision quality |
| N-step returns | +10-20% credit assignment |
| NoisyNet | +10-20% exploration efficiency |

---

## Configuration Summary

### Recommended Starting Config Changes

```python
# config.py - Space Invaders Optimized

# Learning
LEARNING_RATE: float = 0.0001       # Keep as-is initially
GAMMA: float = 0.99                 # Increase from 0.97
BATCH_SIZE: int = 128               # Keep as-is
GRADIENT_STEPS: int = 4             # Increase from 2
LEARN_EVERY: int = 16               # Increase from 8

# Target Network
USE_SOFT_UPDATE: bool = True
TARGET_TAU: float = 0.001           # Decrease from 0.005

# Exploration
EPSILON_DECAY: float = 0.998        # Slower decay
EPSILON_END: float = 0.02           # Higher minimum
EPSILON_WARMUP: int = 100           # Add warmup

# Experience Replay
USE_PRIORITIZED_REPLAY: bool = True  # Enable PER
PER_ALPHA: float = 0.6
PER_BETA_START: float = 0.4
PER_BETA_FRAMES: int = 200000       # Increase for slower annealing

# Architecture
USE_DUELING: bool = True            # Keep as-is
HIDDEN_LAYERS: List[int] = [512, 256, 128]  # Keep as-is
```

---

## Notes for Implementation

1. **Backwards Compatibility:** All changes are additive. Old models won't load if state size changes, but the codebase structure remains intact.

2. **Monitoring:** Add logging for new metrics (accuracy, threat level, etc.) to track improvement sources.

3. **Checkpointing:** With longer epsilon decay, save checkpoints more frequently during the exploration phase.

4. **Hardware:** These changes don't significantly increase compute requirements. CPU training should still be effective.

5. **Rollback Plan:** If performance degrades, revert to baseline config and introduce changes one at a time.

---

*Document created for Neural Space Invaders improvement initiative. Start with Phase 1 quick wins before progressing to more complex changes.*

