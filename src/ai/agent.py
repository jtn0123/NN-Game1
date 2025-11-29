"""
DQN Agent
=========

The AI agent that learns to play games using Deep Q-Learning.

Key Components:
    1. Policy Network  - Used for action selection
    2. Target Network  - Used for stable Q-value estimation
    3. Replay Buffer   - Stores experiences for training
    4. Epsilon-Greedy  - Balances exploration vs exploitation

Training Algorithm (DQN):
    1. Observe state s
    2. Choose action a (epsilon-greedy)
    3. Execute action, observe reward r and next state s'
    4. Store (s, a, r, s', done) in replay buffer
    5. Sample mini-batch from replay buffer
    6. Calculate target: y = r + γ * max_a' Q_target(s', a')
    7. Update policy network: minimize (Q(s,a) - y)²
    8. Periodically sync target network with policy network

References:
    Mnih et al., 2015 - "Human-level control through deep reinforcement learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import random
import os
import time
import threading

from .network import DQN, DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import sys
sys.path.append('../..')
from config import Config


@dataclass
class TrainingHistory:
    """
    Training history for dashboard visualization persistence.
    
    Stores arrays of metrics that allow the dashboard to restore
    charts and statistics when resuming training.
    """
    # Episode-level metrics (one per episode)
    scores: List[int]
    rewards: List[float]
    steps: List[int]
    epsilons: List[float]
    bricks: List[int]
    wins: List[bool]
    
    # Running averages (computed, not stored per-episode)
    losses: List[float]  # Recent losses for averaging
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingHistory':
        # Handle backwards compatibility - older saves may not have all fields
        return cls(
            scores=data.get('scores', []),
            rewards=data.get('rewards', []),
            steps=data.get('steps', []),
            epsilons=data.get('epsilons', []),
            bricks=data.get('bricks', []),
            wins=data.get('wins', []),
            losses=data.get('losses', [])
        )
    
    @classmethod
    def empty(cls) -> 'TrainingHistory':
        """Create empty history."""
        return cls(
            scores=[], rewards=[], steps=[], 
            epsilons=[], bricks=[], wins=[], losses=[]
        )


@dataclass
class SaveMetadata:
    """Rich metadata stored with each model checkpoint."""
    # Timing
    timestamp: str
    save_reason: str  # 'best', 'periodic', 'manual', 'final', 'interrupted'
    total_training_time_seconds: float
    
    # Training progress
    episode: int
    total_steps: int
    epsilon: float
    
    # Performance metrics
    best_score: int
    avg_score_last_100: float
    avg_loss: float
    win_rate: float
    memory_buffer_size: int
    
    # Config snapshot
    learning_rate: float
    gamma: float
    batch_size: int
    hidden_layers: List[int]
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    use_dueling: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SaveMetadata':
        return cls(**data)


class Agent:
    """
    DQN Agent for reinforcement learning.
    
    The agent maintains two networks:
        - policy_net: Updated every training step
        - target_net: Updated periodically for stability
    
    Action Selection:
        - With probability epsilon: random action (exploration)
        - With probability (1-epsilon): best Q-value action (exploitation)
    
    Attributes:
        policy_net: Network used for action selection
        target_net: Network used for computing targets
        memory: Experience replay buffer
        epsilon: Current exploration rate
        
    Example:
        >>> agent = Agent(state_size=55, action_size=3)
        >>> action = agent.select_action(state)
        >>> agent.remember(state, action, reward, next_state, done)
        >>> loss = agent.learn()
    """
    
    # Type annotations for instance variables
    # Note: torch.compile() wraps the network but preserves the interface at runtime
    policy_net: Union[DQN, DuelingDQN]
    target_net: Union[DQN, DuelingDQN]
    memory: Union[ReplayBuffer, PrioritizedReplayBuffer]
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[Config] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            config: Configuration object
        """
        self.config = config or Config()
        self.state_size = state_size
        self.action_size = action_size
        self.device = self.config.DEVICE
        
        # Networks - use DuelingDQN if enabled in config
        NetworkClass = DuelingDQN if self.config.USE_DUELING else DQN
        self.policy_net = NetworkClass(state_size, action_size, config).to(self.device)
        self.target_net = NetworkClass(state_size, action_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly
        
        # Apply torch.compile() for potential speedup (PyTorch 2.0+)
        self._compiled = False
        if self.config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            try:
                compile_mode = getattr(self.config, 'TORCH_COMPILE_MODE', 'reduce-overhead')
                # torch.compile() returns a wrapper that preserves the interface but changes the type
                self.policy_net = torch.compile(self.policy_net, mode=compile_mode)  # type: ignore[assignment]
                self.target_net = torch.compile(self.target_net, mode=compile_mode)  # type: ignore[assignment]
                self._compiled = True
                print(f"✓ torch.compile() enabled (mode={compile_mode})")
            except Exception as e:
                print(f"⚠ torch.compile() failed, using eager mode: {e}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),  # type: ignore[attr-defined]
            lr=self.config.LEARNING_RATE
        )

        # Learning rate scheduler
        self.scheduler: Optional[_LRScheduler] = None
        if getattr(self.config, 'USE_LR_SCHEDULER', False):
            scheduler_type = getattr(self.config, 'LR_SCHEDULER_TYPE', 'cosine')
            if scheduler_type == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(  # type: ignore[assignment]
                    self.optimizer,
                    T_max=2000,
                    eta_min=getattr(self.config, 'LR_MIN', 1e-5)
                )
                print(f"✓ Cosine LR scheduler enabled (T_max=2000, eta_min={getattr(self.config, 'LR_MIN', 1e-5)})")
            elif scheduler_type == 'step':
                from torch.optim.lr_scheduler import StepLR
                step_size = getattr(self.config, 'LR_SCHEDULER_STEP', 500)
                gamma = getattr(self.config, 'LR_SCHEDULER_GAMMA', 0.5)
                self.scheduler = StepLR(  # type: ignore[assignment]
                    self.optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
                print(f"✓ Step LR scheduler enabled (step_size={step_size}, gamma={gamma})")

        # Track if using NoisyNets for exploration (disables epsilon-greedy)
        self._use_noisy_nets = getattr(self.config, 'USE_NOISY_NETWORKS', False)
        
        # Replay buffer - prioritize N-step > PER > basic
        use_n_step = getattr(self.config, 'USE_N_STEP_RETURNS', False)
        self._use_per = False  # Default to False, set True only if using PER

        if use_n_step:
            # N-step buffer doesn't support PER currently
            from src.ai.replay_buffer import NStepReplayBuffer
            n_steps = getattr(self.config, 'N_STEP_SIZE', 3)
            self.memory = NStepReplayBuffer(
                capacity=self.config.MEMORY_SIZE,
                state_size=state_size,
                n_steps=n_steps,
                gamma=self.config.GAMMA
            )
            print(f"✓ N-step returns enabled (n={n_steps})")
        elif getattr(self.config, 'USE_PRIORITIZED_REPLAY', False):
            # Prioritized Experience Replay
            self._use_per = True
            self.memory = PrioritizedReplayBuffer(
                capacity=self.config.MEMORY_SIZE,
                state_size=state_size,
                alpha=getattr(self.config, 'PER_ALPHA', 0.6),
                beta_start=getattr(self.config, 'PER_BETA_START', 0.4),
                beta_end=1.0,
                beta_frames=getattr(self.config, 'PER_BETA_FRAMES', 100000)
            )
            print("✓ Prioritized Experience Replay enabled")
        else:
            # Basic uniform replay buffer
            self.memory = ReplayBuffer(capacity=self.config.MEMORY_SIZE, state_size=state_size)
        
        # Exploration
        self.epsilon = self.config.EPSILON_START
        
        # Training step counter (counts total gradient updates)
        self.steps = 0
        
        # Learn step counter (for LEARN_EVERY skipping)
        self._learn_step = 0
        
        # Next target network update threshold (for hard updates)
        self._next_target_update = self.config.TARGET_UPDATE
        
        # Mixed precision setup for MPS/CUDA
        self._use_mixed_precision = getattr(self.config, 'USE_MIXED_PRECISION', False)
        if self._use_mixed_precision:
            # Determine autocast device type
            if self.device.type == 'mps':
                self._autocast_device = 'mps'
            elif self.device.type == 'cuda':
                self._autocast_device = 'cuda'
            else:
                self._use_mixed_precision = False  # CPU doesn't benefit much
                self._autocast_device = 'cpu'
            
            if self._use_mixed_precision:
                print(f"✓ Mixed precision enabled (device={self._autocast_device})")
        
        # Training metrics (bounded to prevent memory growth during long training)
        self.losses: deque[float] = deque(maxlen=10000)
        self._losses_lock = threading.Lock()  # Thread safety for concurrent reads/writes

        # Track whether last action was exploration (for accurate metrics)
        self._last_action_explored: bool = False
        
        # Pre-allocated tensors for action selection (avoids tensor creation per step)
        self._state_tensor = torch.empty((1, state_size), dtype=torch.float32, device=self.device)
        
        # Pre-allocated batch tensors for learning (avoids allocation per learning step)
        batch_size = self.config.BATCH_SIZE
        self._batch_states = torch.empty((batch_size, state_size), dtype=torch.float32, device=self.device)
        self._batch_actions = torch.empty(batch_size, dtype=torch.int64, device=self.device)
        self._batch_rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        self._batch_next_states = torch.empty((batch_size, state_size), dtype=torch.float32, device=self.device)
        self._batch_dones = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        self._cached_batch_size = batch_size
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy with optional NoisyNets.

        Supports hybrid exploration: epsilon-greedy provides fallback random
        exploration while NoisyNets provide learned, state-dependent exploration.
        This prevents policy collapse if NoisyNet sigmas decay too aggressively.

        Args:
            state: Current game state
            training: If True, use exploration; if False, use greedy

        Returns:
            Selected action index

        Note:
            After calling this method, check `agent._last_action_explored` to
            determine if the action was exploration (random) or exploitation (greedy).
        """
        # Epsilon-greedy exploration (works alongside NoisyNets as fallback)
        # Only skip if epsilon is exactly 0 (pure NoisyNets mode)
        if training and self.epsilon > 0 and random.random() < self.epsilon:
            # Exploration: random action
            self._last_action_explored = True
            return random.randrange(self.action_size)

        # Exploitation: best Q-value action (NoisyNets handle exploration internally)
        self._last_action_explored = False

        # Reset noise for NoisyNet exploration (only in training mode)
        if training and hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()  # type: ignore[operator]

        # Set network to eval mode if not training (for NoisyNet and other layers)
        was_training = self.policy_net.training
        if not training:
            self.policy_net.eval()

        # Use inference_mode() for better performance than no_grad()
        with torch.inference_mode():
            # Reuse pre-allocated tensor to avoid allocation overhead
            # copy_() handles CPU→device transfer automatically (no .to(device) needed)
            self._state_tensor.copy_(torch.from_numpy(state.reshape(1, -1)))
            q_values = self.policy_net(self._state_tensor)
            action = q_values.argmax(dim=1).item()

        # Restore training mode
        if was_training:
            self.policy_net.train()

        return action
    
    def select_actions_batch(self, states: np.ndarray, training: bool = True) -> Tuple[np.ndarray, int, int]:
        """
        Select actions for a batch of states using epsilon-greedy or NoisyNets.
        
        This is optimized for vectorized environments, performing a single
        forward pass for all states. When USE_NOISY_NETWORKS is enabled,
        exploration is handled by learned noise parameters instead of epsilon-greedy.
        
        Args:
            states: Batch of game states, shape (batch_size, state_size)
            training: If True, use exploration; if False, use greedy
            
        Returns:
            Tuple of:
            - actions: Array of selected action indices, shape (batch_size,)
            - num_explored: Number of random (exploration) actions taken
            - num_exploited: Number of greedy (exploitation) actions taken
        """
        batch_size = states.shape[0]
        actions = np.empty(batch_size, dtype=np.int64)
        num_explored = 0
        num_exploited = 0
        
        # Reset noise for NoisyNet exploration (only in training mode)
        if training and hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()  # type: ignore[operator]
        
        # Epsilon-greedy exploration (works alongside NoisyNets as fallback)
        # Only apply if epsilon > 0 and in training mode
        if training and self.epsilon > 0:
            # Determine which states get random actions (exploration)
            # Use np.less() for clearer typing than < operator
            explore_mask = np.less(np.random.random(batch_size), self.epsilon)
            num_explored = int(np.sum(explore_mask))
            num_exploited = batch_size - num_explored
            
            if num_explored > 0:
                # Random actions for exploring states
                actions[explore_mask] = np.random.randint(0, self.action_size, size=num_explored)
            
            if num_exploited > 0:
                # Exploitation: best Q-value actions for non-exploring states
                # NoisyNets provide additional exploration within network forward pass
                exploit_mask = ~explore_mask
                exploit_states = states[exploit_mask]
                
                with torch.inference_mode():
                    states_tensor = torch.from_numpy(exploit_states).to(self.device)
                    q_values = self.policy_net(states_tensor)
                    best_actions = q_values.argmax(dim=1).cpu().numpy()
                    actions[exploit_mask] = best_actions
        else:
            # Pure NoisyNets or evaluation mode: all actions from network
            # NoisyNets handle exploration via learned noise parameters
            num_exploited = batch_size
            with torch.inference_mode():
                states_tensor = torch.from_numpy(states).to(self.device)
                q_values = self.policy_net(states_tensor)
                actions = q_values.argmax(dim=1).cpu().numpy()
        
        return actions, num_explored, num_exploited
    
    def remember_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Store a batch of experiences in replay buffer.
        
        Args:
            states: Batch of current states, shape (batch_size, state_size)
            actions: Batch of actions taken, shape (batch_size,)
            rewards: Batch of rewards received, shape (batch_size,)
            next_states: Batch of next states, shape (batch_size, state_size)
            dones: Batch of done flags, shape (batch_size,)
        """
        for i in range(len(states)):
            self.memory.push(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions (useful for visualization).
        
        Args:
            state: Current game state
            
        Returns:
            Array of Q-values for each action
        """
        with torch.inference_mode():
            # Reuse pre-allocated tensor to avoid allocation overhead
            # copy_() handles CPU→device transfer automatically (no .to(device) needed)
            self._state_tensor.copy_(torch.from_numpy(state.reshape(1, -1)))
            q_values = self.policy_net(self._state_tensor)
            return q_values.cpu().numpy()[0]
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self, update_target: bool = True) -> Optional[float]:
        """
        Perform one training step using Double DQN.
        
        Double DQN reduces Q-value overestimation by:
        1. Using policy network to SELECT the best action
        2. Using target network to EVALUATE that action's Q-value
        
        Args:
            update_target: Whether to update target network this step.
                          Set to False for intermediate gradient steps when
                          using GRADIENT_STEPS > 1 to maintain correct update frequency.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Always increment learn step counter to track calls to learn()
        # This ensures consistent LEARN_EVERY behavior across the training lifecycle
        self._learn_step += 1
        
        # Don't learn until we have enough experiences
        if len(self.memory) < self.config.MEMORY_MIN:
            return None
        
        if not self.memory.is_ready(self.config.BATCH_SIZE):
            return None
        
        # Skip learning based on LEARN_EVERY setting for performance
        learn_every = getattr(self.config, 'LEARN_EVERY', 1)
        if self._learn_step % learn_every != 0:
            return None
        
        # Perform multiple gradient steps if configured (compensates for LEARN_EVERY)
        gradient_steps = getattr(self.config, 'GRADIENT_STEPS', 1)
        total_loss = 0.0
        
        for grad_step in range(gradient_steps):
            loss = self._learn_step_internal()
            if loss is not None:
                total_loss += loss
        
        # Increment steps counter by number of gradient steps performed
        # This ensures TARGET_UPDATE frequency is correct regardless of GRADIENT_STEPS setting
        self.steps += gradient_steps
        
        # Target network update
        if update_target:
            if self.config.USE_SOFT_UPDATE:
                # Soft update happens every learning call (regardless of step count)
                self._soft_update_target_network()
            elif self.steps >= self._next_target_update:
                # Hard update based on total gradient steps
                self.update_target_network()
                self._next_target_update = self.steps + self.config.TARGET_UPDATE
        
        return total_loss / gradient_steps if gradient_steps > 0 else None
    
    def _learn_step_internal(self) -> Optional[float]:
        """
        Internal learning step with mixed precision and optimized transfers.

        This performs a single gradient update. The caller (learn()) is responsible
        for incrementing the steps counter and triggering target network updates.

        Supports both uniform and prioritized experience replay.

        Returns:
            Loss value if training occurred, None otherwise
        """
        batch_size = self.config.BATCH_SIZE
        
        # Sample batch - different path for PER vs uniform
        if self._use_per:
            # PER sampling returns indices and importance sampling weights
            assert isinstance(self.memory, PrioritizedReplayBuffer)
            states_np, actions_np, rewards_np, next_states_np, dones_np, indices, weights_np = \
                self.memory.sample_no_copy(batch_size)
            weights = torch.from_numpy(weights_np).to(self.device)
        else:
            # Uniform sampling (no-copy is safe since we consume immediately)
            assert isinstance(self.memory, ReplayBuffer)
            states_np, actions_np, rewards_np, next_states_np, dones_np = \
                self.memory.sample_no_copy(batch_size)
            indices = None
            weights = None
        
        # Resize pre-allocated tensors if batch size changed OR device changed OR not yet allocated
        # This prevents device mismatch errors when loading models trained on different devices
        if (batch_size != self._cached_batch_size or
            not hasattr(self, '_batch_states') or
            self._batch_states.device != self.device):
            self._batch_states = torch.empty((batch_size, self.state_size), dtype=torch.float32, device=self.device)
            self._batch_actions = torch.empty(batch_size, dtype=torch.int64, device=self.device)
            self._batch_rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
            self._batch_next_states = torch.empty((batch_size, self.state_size), dtype=torch.float32, device=self.device)
            self._batch_dones = torch.empty(batch_size, dtype=torch.float32, device=self.device)
            self._cached_batch_size = batch_size
        
        # Copy to pre-allocated tensors (faster than creating new tensors)
        # copy_() handles CPU→device transfer automatically (no .to(device) needed)
        self._batch_states.copy_(torch.from_numpy(states_np))
        self._batch_actions.copy_(torch.from_numpy(actions_np))
        self._batch_rewards.copy_(torch.from_numpy(rewards_np))
        self._batch_next_states.copy_(torch.from_numpy(next_states_np))
        self._batch_dones.copy_(torch.from_numpy(dones_np))
        
        # Use the pre-allocated tensors
        states = self._batch_states
        actions = self._batch_actions
        rewards = self._batch_rewards
        next_states = self._batch_next_states
        dones = self._batch_dones
        
        # Clip negative rewards to prevent extreme gradients (if enabled)
        # Only clip negative side to preserve win bonus signal (REWARD_WIN = 100)
        if self.config.REWARD_CLIP > 0:
            rewards = torch.clamp(rewards, min=-self.config.REWARD_CLIP)

        # Reset noise for NoisyNet exploration BEFORE forward pass
        # This ensures fresh noise for each training step without modifying computation graph
        if hasattr(self.policy_net, 'reset_noise'):
            self.policy_net.reset_noise()  # type: ignore[operator]
            self.target_net.reset_noise()  # type: ignore[operator]

        # Use mixed precision autocast if enabled (significant speedup on MPS/CUDA)
        # Only forward passes use float16; loss computed in float32 for numerical stability
        if self._use_mixed_precision:
            with torch.autocast(device_type=self._autocast_device, dtype=torch.float16):
                current_q, target_q = self._compute_q_values(states, actions, rewards, next_states, dones)
            current_q = current_q.float()
            target_q = target_q.float()
        else:
            current_q, target_q = self._compute_q_values(states, actions, rewards, next_states, dones)
        
        # Compute element-wise TD errors for PER priority updates
        td_errors = (current_q - target_q).detach()
        
        # Compute loss with importance sampling weights if using PER
        if self._use_per and weights is not None:
            # Weighted element-wise loss
            element_loss = nn.SmoothL1Loss(reduction='none')(current_q, target_q)
            loss = (element_loss * weights).mean()
        else:
            loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),  # type: ignore[attr-defined]
                self.config.GRAD_CLIP
            )
        
        self.optimizer.step()
        
        # Update PER priorities with TD errors
        if self._use_per and indices is not None:
            assert isinstance(self.memory, PrioritizedReplayBuffer)
            self.memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        # Store loss for metrics (thread-safe)
        loss_value = loss.item()
        with self._losses_lock:
            self.losses.append(loss_value)
        
        return loss_value
    
    def _compute_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute current and target Q-values using Double DQN.

        Returns:
            Tuple of (current_q, target_q) tensors
        """
        # Compute current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using DOUBLE DQN
        with torch.no_grad():
            # Step 1: Use POLICY network to select best actions for next states
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            # Step 2: Use TARGET network to evaluate those actions
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)

            # Compute TD target (adjust gamma for N-step returns)
            use_n_step = getattr(self.config, 'USE_N_STEP_RETURNS', False)
            if use_n_step:
                n_steps = getattr(self.config, 'N_STEP_SIZE', 3)
                effective_gamma = self.config.GAMMA ** n_steps
            else:
                effective_gamma = self.config.GAMMA

            target_q = rewards + (1 - dones) * effective_gamma * next_q

        return current_q, target_q
    
    def update_target_network(self) -> None:
        """Hard update: Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _soft_update_target_network(self) -> None:
        """
        Soft update: Gradually blend policy network weights into target network.
        
        target = TAU * policy + (1 - TAU) * target
        
        This provides smoother, more stable learning than periodic hard updates.
        """
        tau = self.config.TARGET_TAU
        for target_param, policy_param in zip(
            self.target_net.parameters(),  # type: ignore[attr-defined]
            self.policy_net.parameters()  # type: ignore[attr-defined]
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )
    
    def decay_epsilon(self, episode: Optional[int] = None) -> None:
        """Decay exploration rate.

        Args:
            episode: Current episode number. Epsilon only decays after
                     EPSILON_WARMUP episodes to allow buffer to fill.
                     If None, bypasses warmup check (backward compatible).
        
        Note:
            When USE_NOISY_NETWORKS is enabled with EPSILON_START > 0,
            epsilon-greedy acts as a fallback exploration mechanism.
            This hybrid approach prevents policy collapse if NoisyNet
            sigmas decay too aggressively.
        """
        # Allow epsilon decay even with NoisyNets if EPSILON_START > 0
        # This provides hybrid exploration (NoisyNets + epsilon-greedy fallback)
        
        warmup = getattr(self.config, 'EPSILON_WARMUP', 0)

        # Skip decay during warmup period (only if episode is explicitly provided)
        if episode is not None and episode < warmup:
            return

        self.epsilon = max(
            self.config.EPSILON_END,
            self.epsilon * self.config.EPSILON_DECAY
        )

    def step_scheduler(self) -> None:
        """Step the learning rate scheduler after each episode."""
        if self.scheduler is not None:
            self.scheduler.step()

    def save(
        self,
        filepath: str,
        save_reason: str = "manual",
        episode: int = 0,
        best_score: int = 0,
        avg_score_last_100: float = 0.0,
        win_rate: float = 0.0,
        training_start_time: Optional[float] = None,
        training_history: Optional['TrainingHistory'] = None,
        quiet: bool = False
    ) -> Optional[SaveMetadata]:
        """
        Save agent state to file with rich metadata.
        
        Args:
            filepath: Path to save file
            save_reason: Why this save is happening ('best', 'periodic', 'manual', 'final', 'interrupted')
            episode: Current episode number
            best_score: Best score achieved so far
            avg_score_last_100: Average score over last 100 episodes
            win_rate: Win rate over last 100 episodes
            training_start_time: Unix timestamp when training started (for calculating total time)
            training_history: Training history for dashboard restoration (scores, rewards, etc.)
            quiet: If True, suppress most output
            
        Returns:
            SaveMetadata object if save succeeded, None on failure
        """
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Calculate training time
        total_time = 0.0
        if training_start_time:
            total_time = time.time() - training_start_time
        
        # Build metadata
        metadata = SaveMetadata(
            timestamp=datetime.now().isoformat(),
            save_reason=save_reason,
            total_training_time_seconds=total_time,
            episode=episode,
            total_steps=self.steps,
            epsilon=self.epsilon,
            best_score=best_score,
            avg_score_last_100=avg_score_last_100,
            avg_loss=self.get_average_loss(100),
            win_rate=win_rate,
            memory_buffer_size=len(self.memory),
            learning_rate=self.config.LEARNING_RATE,
            gamma=self.config.GAMMA,
            batch_size=self.config.BATCH_SIZE,
            hidden_layers=list(self.config.HIDDEN_LAYERS),
            epsilon_start=self.config.EPSILON_START,
            epsilon_end=self.config.EPSILON_END,
            epsilon_decay=self.config.EPSILON_DECAY,
            use_dueling=self.config.USE_DUELING
        )
        
        # Get state dicts, stripping _orig_mod. prefix if model is compiled
        # This ensures saved models are portable regardless of torch.compile status
        policy_state = self.policy_net.state_dict()
        target_state = self.target_net.state_dict()
        
        if self._compiled:
            # Strip _orig_mod. prefix for portability
            policy_state = {k.replace('_orig_mod.', ''): v for k, v in policy_state.items()}
            target_state = {k.replace('_orig_mod.', ''): v for k, v in target_state.items()}
        
        checkpoint = {
            'policy_net_state_dict': policy_state,
            'target_net_state_dict': target_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            '_learn_step': self._learn_step,
            '_next_target_update': self._next_target_update,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'metadata': metadata.to_dict(),
        }
        
        # Save training history if provided (for dashboard restoration)
        if training_history is not None:
            checkpoint['training_history'] = training_history.to_dict()
        
        try:
            torch.save(checkpoint, filepath)
            
            # Verify the save by checking file exists and size
            if not os.path.exists(filepath):
                print(f"❌ Save verification FAILED: {filepath} not found after save")
                return None
            
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Less than 1KB is suspicious
                print(f"⚠️ Warning: Saved file seems too small ({file_size} bytes)")
            
            # Format output
            if not quiet:
                size_mb = file_size / (1024 * 1024)
                reason_emoji = {
                    'best': '🏆',
                    'periodic': '📅',
                    'manual': '💾',
                    'final': '✅',
                    'interrupted': '⛔'
                }.get(save_reason, '💾')
                
                print(f"\n{reason_emoji} Model Saved: {os.path.basename(filepath)}")
                print(f"   Episode: {episode:,} | Steps: {self.steps:,} | ε: {self.epsilon:.4f}")
                print(f"   Best Score: {best_score} | Avg(100): {avg_score_last_100:.1f} | Win Rate: {win_rate*100:.1f}%")
                print(f"   Size: {size_mb:.2f} MB | Reason: {save_reason}")
                if total_time > 0:
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    print(f"   Training Time: {hours}h {minutes}m")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Save FAILED: {e}")
            return None
    
    def _adapt_state_dict_for_compile(self, state_dict: Dict[str, Any], target_module) -> Dict[str, Any]:
        """
        Adapt state dict keys between compiled and non-compiled models.
        
        torch.compile() wraps the model and prefixes keys with '_orig_mod.'
        This method handles loading models regardless of compile status.
        
        Args:
            state_dict: The state dict to adapt
            target_module: The module to load into (may be compiled or not)
            
        Returns:
            Adapted state dict with correct key prefixes
        """
        # Check if saved state dict has _orig_mod prefix
        saved_has_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        
        # Check if target module expects _orig_mod prefix (is compiled)
        target_expects_prefix = self._compiled
        
        if saved_has_prefix == target_expects_prefix:
            # No adaptation needed
            return state_dict
        
        adapted = {}
        if saved_has_prefix and not target_expects_prefix:
            # Remove _orig_mod. prefix
            for k, v in state_dict.items():
                new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
                adapted[new_key] = v
        elif not saved_has_prefix and target_expects_prefix:
            # Add _orig_mod. prefix
            for k, v in state_dict.items():
                adapted[f'_orig_mod.{k}'] = v
        
        return adapted
    
    def load(self, filepath: str, quiet: bool = False) -> Tuple[Optional[SaveMetadata], Optional['TrainingHistory']]:
        """
        Load agent state from file with detailed resume summary.
        
        Args:
            filepath: Path to checkpoint file
            quiet: If True, suppress output
            
        Returns:
            Tuple of (SaveMetadata, TrainingHistory) - either may be None for old saves
        """
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return None, None
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None, None
        
        # Check for architecture mismatch
        saved_state_size = checkpoint.get('state_size', self.state_size)
        saved_action_size = checkpoint.get('action_size', self.action_size)
        
        # If architecture doesn't match, cannot load this model
        if saved_state_size != self.state_size or saved_action_size != self.action_size:
            if not quiet:
                if saved_state_size != self.state_size:
                    print(f"⚠️  Model incompatible: State size mismatch (saved: {saved_state_size}, current: {self.state_size})")
                if saved_action_size != self.action_size:
                    print(f"⚠️  Model incompatible: Action size mismatch (saved: {saved_action_size}, current: {self.action_size})")
                print(f"❌ Cannot load model - architecture mismatch. Starting fresh training.")
            return None, None
        
        # Adapt state dicts for torch.compile() compatibility
        policy_state = self._adapt_state_dict_for_compile(
            checkpoint['policy_net_state_dict'], self.policy_net
        )
        target_state = self._adapt_state_dict_for_compile(
            checkpoint['target_net_state_dict'], self.target_net
        )
        
        # Load network weights
        self.policy_net.load_state_dict(policy_state)
        self.target_net.load_state_dict(target_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self._learn_step = checkpoint.get('_learn_step', 0)  # Backwards compatible
        # Calculate next target update based on current steps (backwards compatible)
        self._next_target_update = checkpoint.get(
            '_next_target_update', 
            self.steps + self.config.TARGET_UPDATE
        )
        
        # Load metadata if available
        metadata = None
        if 'metadata' in checkpoint:
            try:
                metadata = SaveMetadata.from_dict(checkpoint['metadata'])
            except Exception:
                pass  # Old format without full metadata
        
        # Load training history if available (for dashboard restoration)
        training_history = None
        if 'training_history' in checkpoint:
            try:
                training_history = TrainingHistory.from_dict(checkpoint['training_history'])
            except Exception:
                pass  # Old format without training history
        
        if not quiet:
            file_size = os.path.getsize(filepath)
            size_mb = file_size / (1024 * 1024)
            
            print(f"\n{'='*60}")
            print(f"📂 Resuming Training")
            print(f"{'='*60}")
            print(f"   Model: {os.path.basename(filepath)} ({size_mb:.2f} MB)")
            
            if metadata:
                # Parse timestamp for human-readable format
                try:
                    save_time = datetime.fromisoformat(metadata.timestamp)
                    time_ago = datetime.now() - save_time
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days}d ago"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600}h ago"
                    else:
                        time_str = f"{time_ago.seconds // 60}m ago"
                    print(f"   Saved: {save_time.strftime('%b %d, %Y %I:%M %p')} ({time_str})")
                except Exception:
                    print(f"   Saved: {metadata.timestamp}")
                
                print(f"\n   Episode: {metadata.episode:,} | Steps: {metadata.total_steps:,} | ε: {metadata.epsilon:.4f}")
                print(f"   Best Score: {metadata.best_score} | Avg(100): {metadata.avg_score_last_100:.1f}")
                print(f"   Win Rate: {metadata.win_rate*100:.1f}% | Avg Loss: {metadata.avg_loss:.4f}")
                
                if metadata.total_training_time_seconds > 0:
                    hours = int(metadata.total_training_time_seconds // 3600)
                    minutes = int((metadata.total_training_time_seconds % 3600) // 60)
                    print(f"   Previous Training Time: {hours}h {minutes}m")
                
                print(f"\n   Config: LR={metadata.learning_rate}, γ={metadata.gamma}, Batch={metadata.batch_size}")
                print(f"   Architecture: {metadata.hidden_layers}")
            else:
                # Old format - show basic info
                print(f"\n   Steps: {self.steps:,} | Epsilon: {self.epsilon:.4f}")
                print(f"   (Legacy save - no detailed metadata)")
            
            # Report training history status
            if training_history and len(training_history.scores) > 0:
                print(f"   Training History: {len(training_history.scores)} episodes restored")
            else:
                print(f"   Training History: Not available (older save format)")
            
            print(f"{'='*60}\n")
        
        return metadata, training_history
    
    @staticmethod
    def inspect_model(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Inspect a model file without loading it into an agent.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with model info, or None on error
        """
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"❌ Failed to read model: {e}")
            return None
        
        file_size = os.path.getsize(filepath)
        file_mtime = os.path.getmtime(filepath)
        
        info = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'file_modified': datetime.fromtimestamp(file_mtime).isoformat(),
            'steps': checkpoint.get('steps', 'unknown'),
            'epsilon': checkpoint.get('epsilon', 'unknown'),
            'state_size': checkpoint.get('state_size', 'unknown'),
            'action_size': checkpoint.get('action_size', 'unknown'),
            'has_metadata': 'metadata' in checkpoint,
            'metadata': checkpoint.get('metadata', None)
        }
        
        return info
    
    @staticmethod
    def list_models(model_dir: str = 'models') -> List[Dict[str, Any]]:
        """
        List all model files in a directory with their metadata.
        
        Args:
            model_dir: Directory to scan for .pth files
            
        Returns:
            List of model info dictionaries, sorted by modified time (newest first)
        """
        models: List[Dict[str, Any]] = []
        
        if not os.path.exists(model_dir):
            return models
        
        for filename in os.listdir(model_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(model_dir, filename)
                info = Agent.inspect_model(filepath)
                if info:
                    models.append(info)
        
        # Sort by file modified time, newest first
        models.sort(key=lambda x: x['file_modified'], reverse=True)
        return models
    
    def get_network_activations(self) -> dict:
        """Get current network activations for visualization."""
        return self.policy_net.get_activations()
    
    def get_average_loss(self, n: int = 100) -> float:
        """Get average of last n losses (thread-safe)."""
        with self._losses_lock:
            if not self.losses:
                return 0.0
            # Iterate from end - O(n) instead of O(len) for converting entire deque to list
            count = min(n, len(self.losses))
            total = 0.0
            it = iter(reversed(self.losses))
            for _ in range(count):
                total += next(it)
        return total / count


# Testing
if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    config = Config()
    agent = Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config
    )
    
    print(f"\n📊 Agent Configuration:")
    print(f"   State size: {agent.state_size}")
    print(f"   Action size: {agent.action_size}")
    print(f"   Device: {agent.device}")
    print(f"   Epsilon: {agent.epsilon}")
    
    # Test action selection
    state = np.random.randn(config.STATE_SIZE).astype(np.float32)
    action = agent.select_action(state)
    print(f"\n🎮 Test action selection:")
    print(f"   State shape: {state.shape}")
    print(f"   Selected action: {action}")
    
    # Test Q-value computation
    q_values = agent.get_q_values(state)
    print(f"   Q-values: {q_values}")
    
    # Test experience storage
    for _ in range(100):
        s = np.random.randn(config.STATE_SIZE).astype(np.float32)
        a = np.random.randint(0, config.ACTION_SIZE)
        r = np.random.randn()
        s2 = np.random.randn(config.STATE_SIZE).astype(np.float32)
        d = False
        agent.remember(s, a, r, s2, d)
    
    print(f"\n📦 Memory buffer size: {len(agent.memory)}")
    
    print("\n✓ Agent tests passed!")