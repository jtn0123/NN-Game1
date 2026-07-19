"""
DQN Agent
=========

The AI agent that learns to play games using Deep Q-Learning.

Key Components:
    1. Policy Network  - Used for action selection
    2. Target Network  - Used for stable Q-value estimation
    3. Replay Buffer   - Stores experiences for training
    4. Epsilon-Greedy  - Balances exploration vs exploitation

References:
    Mnih et al., 2015 - "Human-level control through deep reinforcement learning"
"""

import random
import threading
from collections import deque
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from config import Config

from .agent_experiments import AgentExperimentMixin
from .agent_metadata import SaveMetadata, TrainingHistory
from .agent_persistence import AgentPersistenceMixin
from .extension_contracts import AuxiliaryLossContribution, AuxiliaryLossProvider
from .network import DQN, DuelingDQN, SpatialDQN
from .prioritized_n_step import PrioritizedNStepReplayBuffer
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

__all__ = ["Agent", "SaveMetadata", "TrainingHistory"]


class Agent(AgentExperimentMixin, AgentPersistenceMixin):
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
    policy_net: Union[DQN, DuelingDQN, SpatialDQN]
    target_net: Union[DQN, DuelingDQN, SpatialDQN]
    memory: Union[ReplayBuffer, PrioritizedReplayBuffer, PrioritizedNStepReplayBuffer]

    def __init__(self, state_size: int, action_size: int, config: Optional[Config] = None):
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

        # Networks: a convolutional net for grid-structured (spatial) observations
        # when enabled and the game supplied a layout, else the (dueling) MLP.
        layout = getattr(self.config, "STATE_LAYOUT", None)
        if getattr(self.config, "USE_CNN_STATE", False) and layout is not None:

            def NetworkClass(s: int, a: int, c: Any) -> Any:
                return SpatialDQN(s, a, c, layout)

        else:
            NetworkClass = DuelingDQN if self.config.USE_DUELING else DQN  # type: ignore[assignment]
        self.policy_net = NetworkClass(state_size, action_size, config).to(self.device)
        self.target_net = NetworkClass(state_size, action_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        # Apply torch.compile() for potential speedup (PyTorch 2.0+)
        self._compiled = False
        if self.config.USE_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                compile_mode = getattr(self.config, "TORCH_COMPILE_MODE", "reduce-overhead")
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
            lr=self.config.LEARNING_RATE,
            weight_decay=getattr(self.config, "WEIGHT_DECAY", 0.0),
        )

        # Learning rate scheduler
        self.scheduler: Optional[_LRScheduler] = None
        if getattr(self.config, "USE_LR_SCHEDULER", False):
            scheduler_type = getattr(self.config, "LR_SCHEDULER_TYPE", "cosine")
            if scheduler_type == "cosine":
                from torch.optim.lr_scheduler import CosineAnnealingLR

                self.scheduler = CosineAnnealingLR(  # type: ignore[assignment]
                    self.optimizer,
                    T_max=2000,
                    eta_min=getattr(self.config, "LR_MIN", 1e-5),
                )
                print(
                    f"✓ Cosine LR scheduler enabled (T_max=2000, eta_min={getattr(self.config, 'LR_MIN', 1e-5)})"
                )
            elif scheduler_type == "step":
                from torch.optim.lr_scheduler import StepLR

                step_size = getattr(self.config, "LR_SCHEDULER_STEP", 500)
                gamma = getattr(self.config, "LR_SCHEDULER_GAMMA", 0.5)
                self.scheduler = StepLR(  # type: ignore[assignment]
                    self.optimizer, step_size=step_size, gamma=gamma
                )
                print(f"✓ Step LR scheduler enabled (step_size={step_size}, gamma={gamma})")

        # Track if using NoisyNets for exploration (disables epsilon-greedy)
        self._use_noisy_nets = getattr(self.config, "USE_NOISY_NETWORKS", False)
        self._use_distributional_dqn = bool(getattr(self.config, "USE_DISTRIBUTIONAL_DQN", False))
        if self._use_distributional_dqn:
            if not callable(getattr(self.policy_net, "distributional_logits", None)):
                raise RuntimeError(
                    "USE_DISTRIBUTIONAL_DQN requires networks with distributional_logits()"
                )
            print(
                "✓ Distributional DQN enabled "
                f"(atoms={getattr(self.config, 'C51_NUM_ATOMS', 51)}, "
                f"v=[{getattr(self.config, 'C51_V_MIN', -20.0)}, "
                f"{getattr(self.config, 'C51_V_MAX', 120.0)}])"
            )

        # Replay buffer
        use_n_step = getattr(self.config, "USE_N_STEP_RETURNS", False)
        self._use_per = False  # Default to False, set True only if using PER

        if use_n_step:
            from src.ai.replay_buffer import NStepReplayBuffer

            n_steps = getattr(self.config, "N_STEP_SIZE", 3)
            if getattr(self.config, "USE_PRIORITIZED_REPLAY", False):
                self._use_per = True
                self.memory = PrioritizedNStepReplayBuffer(
                    capacity=self.config.MEMORY_SIZE,
                    state_size=state_size,
                    n_steps=n_steps,
                    gamma=self.config.GAMMA,
                    alpha=getattr(self.config, "PER_ALPHA", 0.6),
                    beta_start=getattr(self.config, "PER_BETA_START", 0.4),
                    beta_end=1.0,
                    beta_frames=getattr(self.config, "PER_BETA_FRAMES", 100000),
                )
                print(f"✓ N-step prioritized replay enabled (n={n_steps})")
            else:
                self.memory = NStepReplayBuffer(
                    capacity=self.config.MEMORY_SIZE,
                    state_size=state_size,
                    n_steps=n_steps,
                    gamma=self.config.GAMMA,
                )
                print(f"✓ N-step returns enabled (n={n_steps})")
        elif getattr(self.config, "USE_PRIORITIZED_REPLAY", False):
            # Prioritized Experience Replay
            self._use_per = True
            self.memory = PrioritizedReplayBuffer(
                capacity=self.config.MEMORY_SIZE,
                state_size=state_size,
                alpha=getattr(self.config, "PER_ALPHA", 0.6),
                beta_start=getattr(self.config, "PER_BETA_START", 0.4),
                beta_end=1.0,
                beta_frames=getattr(self.config, "PER_BETA_FRAMES", 100000),
            )
            print("✓ Prioritized Experience Replay enabled")
        else:
            # Basic uniform replay buffer
            self.memory = ReplayBuffer(capacity=self.config.MEMORY_SIZE, state_size=state_size)

        # Exploration
        self.epsilon = self.config.EPSILON_START

        # Training step counter (counts total gradient updates)
        self.steps = 0
        self._optimizer_steps_since_scheduler = 0

        # Learn step counter (for LEARN_EVERY skipping)
        self._learn_step = 0

        # Next target network update threshold (for hard updates)
        self._next_target_update = self.config.TARGET_UPDATE

        # Mixed precision setup for MPS/CUDA
        self._use_mixed_precision = getattr(self.config, "USE_MIXED_PRECISION", False)
        if self._use_mixed_precision:
            # Determine autocast device type
            if self.device.type == "mps":
                self._autocast_device = "mps"
            elif self.device.type == "cuda":
                self._autocast_device = "cuda"
            else:
                self._use_mixed_precision = False  # CPU doesn't benefit much
                self._autocast_device = "cpu"

            if self._use_mixed_precision:
                print(f"✓ Mixed precision enabled (device={self._autocast_device})")

        # Training metrics (bounded to prevent memory growth during long training)
        self.losses: deque[float] = deque(maxlen=10000)
        self.route_aux_losses: deque[float] = deque(maxlen=10000)
        self.route_aux_accuracies: deque[float] = deque(maxlen=10000)
        self.demo_action_losses: deque[float] = deque(maxlen=10000)
        self.demo_conservative_losses: deque[float] = deque(maxlen=10000)
        self.demo_action_accuracies: deque[float] = deque(maxlen=10000)
        self.close_zone_demo_action_losses: deque[float] = deque(maxlen=10000)
        self.close_zone_demo_action_accuracies: deque[float] = deque(maxlen=10000)
        self.correction_action_losses: deque[float] = deque(maxlen=10000)
        self.correction_action_accuracies: deque[float] = deque(maxlen=10000)
        self.contact_action_losses: deque[float] = deque(maxlen=10000)
        self.contact_action_accuracies: deque[float] = deque(maxlen=10000)
        self.policy_anchor_losses: deque[float] = deque(maxlen=10000)
        self.policy_anchor_accuracies: deque[float] = deque(maxlen=10000)
        self._losses_lock = threading.Lock()  # Thread safety for concurrent reads/writes
        self._demo_action_states: Optional[torch.Tensor] = None
        self._demo_action_actions: Optional[torch.Tensor] = None
        self._close_zone_demo_action_states: Optional[torch.Tensor] = None
        self._close_zone_demo_action_actions: Optional[torch.Tensor] = None
        self._correction_action_states: Optional[torch.Tensor] = None
        self._correction_action_actions: Optional[torch.Tensor] = None
        self._contact_action_states: Optional[torch.Tensor] = None
        self._contact_action_actions: Optional[torch.Tensor] = None
        self._extra_auxiliary_loss_providers: list[AuxiliaryLossProvider] = []

        # Track whether last action was exploration (for accurate metrics)
        self._last_action_explored: bool = False

        # Pre-allocated tensors for action selection (avoids tensor creation per step)
        self._state_tensor = torch.empty((1, state_size), dtype=torch.float32, device=self.device)
        self._action_batch_tensor = torch.empty(
            (1, state_size), dtype=torch.float32, device=self.device
        )
        self._cached_action_batch_size = 1

        # Pre-allocated batch tensors for learning (avoids allocation per learning step)
        batch_size = self.config.BATCH_SIZE
        self._batch_states = torch.empty(
            (batch_size, state_size), dtype=torch.float32, device=self.device
        )
        self._batch_actions = torch.empty(batch_size, dtype=torch.int64, device=self.device)
        self._batch_rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        self._batch_next_states = torch.empty(
            (batch_size, state_size), dtype=torch.float32, device=self.device
        )
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
        if state.size != self.state_size:
            raise ValueError(
                f"State size mismatch: expected {self.state_size} values, got {state.size}"
            )

        # Epsilon-greedy exploration (works alongside NoisyNets as fallback)
        # Only skip if epsilon is exactly 0 (pure NoisyNets mode)
        if training and self.epsilon > 0 and random.random() < self.epsilon:
            # Exploration: random action
            self._last_action_explored = True
            return random.randrange(self.action_size)

        # Exploitation: best Q-value action (NoisyNets handle exploration internally)
        self._last_action_explored = False

        # Reset noise for NoisyNet exploration (only in training mode)
        if training and hasattr(self.policy_net, "reset_noise"):
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

    def select_actions_batch(
        self, states: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, int, int]:
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
        # Ensure states is 2D (handle 1D input gracefully)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.shape[1] != self.state_size:
            raise ValueError(
                f"State batch size mismatch: expected {self.state_size} features, got {states.shape[1]}"
            )

        batch_size = states.shape[0]
        if batch_size == 0:
            return np.empty(0, dtype=np.int64), 0, 0

        actions = np.empty(batch_size, dtype=np.int64)
        num_explored = 0
        num_exploited = 0

        # Reset noise for NoisyNet exploration (only in training mode)
        if training and hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()  # type: ignore[operator]

        # In eval, put the network in eval() mode so NoisyLinear uses the noise-free
        # mean weights (mirrors select_action). Without this, a vectorized eval would
        # still apply learned noise and be non-deterministic.
        was_training = self.policy_net.training
        if not training:
            self.policy_net.eval()
        try:
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
                    actions[explore_mask] = np.random.randint(
                        0, self.action_size, size=num_explored
                    )

                if num_exploited > 0:
                    # Exploitation: best Q-value actions for non-exploring states
                    # NoisyNets provide additional exploration within network forward pass
                    exploit_mask = ~explore_mask
                    actions[exploit_mask] = self._greedy_actions_for_states(states[exploit_mask])
            else:
                # Pure NoisyNets or evaluation mode: all actions from network
                # NoisyNets handle exploration via learned noise parameters
                num_exploited = batch_size
                actions = self._greedy_actions_for_states(states)
        finally:
            if was_training and not training:
                self.policy_net.train()

        return actions, num_explored, num_exploited

    def _greedy_actions_for_states(self, states: np.ndarray) -> np.ndarray:
        """Return greedy actions for a state batch using cached device storage."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
        batch_size = states.shape[0]
        if (
            batch_size != self._cached_action_batch_size
            or self._action_batch_tensor.device != self.device
        ):
            self._action_batch_tensor = torch.empty(
                (batch_size, self.state_size), dtype=torch.float32, device=self.device
            )
            self._cached_action_batch_size = batch_size

        with torch.inference_mode():
            view = self._action_batch_tensor[:batch_size]
            view.copy_(torch.from_numpy(states))
            q_values = self.policy_net(view)
            return q_values.argmax(dim=1).cpu().numpy()

    def remember_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        truncateds: np.ndarray | None = None,
    ) -> None:
        """
        Store a batch of experiences in replay buffer.

        Args:
            states: Batch of current states, shape (batch_size, state_size)
            actions: Batch of actions taken, shape (batch_size,)
            rewards: Batch of rewards received, shape (batch_size,)
            next_states: Batch of next states, shape (batch_size, state_size)
            dones: Batch of done flags, shape (batch_size,)
            truncateds: Optional mask (batch_size,) of transitions that ended on a
                time/no-progress cutoff rather than a real terminal. Where set, the
                stored done is cleared so the TD target bootstraps the final state
                (truncation-aware bootstrapping). When None, behaviour is unchanged.
        """
        # Use push_batch if available (standard ReplayBuffer), otherwise fall back to loop
        if hasattr(self.memory, "push_batch"):
            self.memory.push_batch(
                states, actions, rewards, next_states, dones, truncateds=truncateds
            )
        else:
            trunc = None if truncateds is None else np.asarray(truncateds).astype(bool)
            for i in range(len(states)):
                # Use .item() for numpy scalar conversion (more reliable than int/float)
                action = actions[i].item() if hasattr(actions[i], "item") else int(actions[i])
                reward = rewards[i].item() if hasattr(rewards[i], "item") else float(rewards[i])
                done = dones[i].item() if hasattr(dones[i], "item") else bool(dones[i])
                # Truncation-aware bootstrapping for the non-push_batch fallback path.
                if trunc is not None and trunc[i]:
                    done = False
                self.memory.push(states[i], action, reward, next_states[i], done)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions (useful for visualization).

        Args:
            state: Current game state

        Returns:
            Array of Q-values for each action
        """
        if state.size != self.state_size:
            raise ValueError(
                f"State size mismatch: expected {self.state_size} values, got {state.size}"
            )

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
        done: bool,
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
        learn_every = getattr(self.config, "LEARN_EVERY", 1)
        if self._learn_step % learn_every != 0:
            return None

        # Perform multiple gradient steps if configured (compensates for LEARN_EVERY)
        gradient_steps = getattr(self.config, "GRADIENT_STEPS", 1)
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
        n_step_lengths_np = None
        if self._use_per:
            # PER sampling returns indices and importance sampling weights
            assert isinstance(self.memory, (PrioritizedReplayBuffer, PrioritizedNStepReplayBuffer))
            per_batch = self.memory.sample_no_copy(batch_size)
            if len(per_batch) == 8:
                (
                    states_np,
                    actions_np,
                    rewards_np,
                    next_states_np,
                    dones_np,
                    indices,
                    weights_np,
                    n_step_lengths_np,
                ) = per_batch
            else:
                (
                    states_np,
                    actions_np,
                    rewards_np,
                    next_states_np,
                    dones_np,
                    indices,
                    weights_np,
                ) = per_batch
            weights = torch.from_numpy(weights_np).to(self.device)
        else:
            # Uniform sampling (no-copy is safe since we consume immediately)
            assert isinstance(self.memory, ReplayBuffer)
            replay_batch = self.memory.sample_no_copy(batch_size)
            if len(replay_batch) == 6:
                (
                    states_np,
                    actions_np,
                    rewards_np,
                    next_states_np,
                    dones_np,
                    n_step_lengths_np,
                ) = replay_batch
            else:
                states_np, actions_np, rewards_np, next_states_np, dones_np = replay_batch
            indices = None
            weights = None

        # Resize pre-allocated tensors if batch size changed OR device changed OR not yet allocated
        # This prevents device mismatch errors when loading models trained on different devices
        if (
            batch_size != self._cached_batch_size
            or not hasattr(self, "_batch_states")
            or self._batch_states.device != self.device
        ):
            self._batch_states = torch.empty(
                (batch_size, self.state_size), dtype=torch.float32, device=self.device
            )
            self._batch_actions = torch.empty(batch_size, dtype=torch.int64, device=self.device)
            self._batch_rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
            self._batch_next_states = torch.empty(
                (batch_size, self.state_size), dtype=torch.float32, device=self.device
            )
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
        n_step_lengths = (
            torch.from_numpy(n_step_lengths_np).to(self.device)
            if n_step_lengths_np is not None
            else None
        )

        # Clip negative rewards to prevent extreme gradients (if enabled)
        # Only clip negative side to preserve win bonus signal (REWARD_WIN = 100)
        if self.config.REWARD_CLIP > 0:
            rewards = torch.clamp(rewards, min=-self.config.REWARD_CLIP)

        # Reset noise for NoisyNet exploration BEFORE forward pass
        # This ensures fresh noise for each training step without modifying computation graph
        if hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()  # type: ignore[operator]
            self.target_net.reset_noise()  # type: ignore[operator]

        if self._use_distributional_dqn:
            element_loss, td_errors = self._compute_distributional_loss(
                states, actions, rewards, next_states, dones, n_step_lengths=n_step_lengths
            )
            if self._use_per and weights is not None:
                loss = (element_loss * weights).mean()
            else:
                loss = element_loss.mean()
        else:
            if self._use_mixed_precision:
                # Use mixed precision autocast if enabled (significant speedup on MPS/CUDA).
                # Only forward passes use float16; loss computed in float32 for numerical stability.
                with torch.autocast(device_type=self._autocast_device, dtype=torch.float16):
                    current_q, target_q = self._compute_q_values(
                        states, actions, rewards, next_states, dones, n_step_lengths=n_step_lengths
                    )
                current_q = current_q.float()
                target_q = target_q.float()
            else:
                current_q, target_q = self._compute_q_values(
                    states, actions, rewards, next_states, dones, n_step_lengths=n_step_lengths
                )

            # Compute element-wise TD errors for PER priority updates.
            td_errors = (current_q - target_q).detach()

            # Compute loss with importance sampling weights if using PER.
            if self._use_per and weights is not None:
                element_loss = nn.SmoothL1Loss(reduction="none")(current_q, target_q)
                loss = (element_loss * weights).mean()
            else:
                loss = nn.SmoothL1Loss()(current_q, target_q)

        auxiliary_contributions = self._auxiliary_loss_contributions(states)
        for contribution in auxiliary_contributions:
            if contribution.weight != 0.0:
                loss = loss + contribution.weighted_loss()

        # DQfD-lite: when a demo store is attached, every gradient step also
        # trains on a small never-overwritten demo minibatch (TD + margin loss).
        demo_loss = self._dqfd_loss()
        if demo_loss is not None:
            loss = loss + demo_loss

        # Optimize (outside autocast for numerical stability)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        if self.config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),  # type: ignore[attr-defined]
                self.config.GRAD_CLIP,
            )

        self.optimizer.step()
        self._optimizer_steps_since_scheduler += 1

        # Update PER priorities with TD errors
        if self._use_per and indices is not None:
            assert isinstance(self.memory, (PrioritizedReplayBuffer, PrioritizedNStepReplayBuffer))
            self.memory.update_priorities(indices, td_errors.abs().cpu().numpy())

        # Store loss for metrics (thread-safe)
        loss_value = loss.item()
        with self._losses_lock:
            self.losses.append(loss_value)
            self._append_auxiliary_metrics_locked(auxiliary_contributions)

        return loss_value

    def _append_auxiliary_metrics_locked(
        self, contributions: tuple[AuxiliaryLossContribution, ...]
    ) -> None:
        """Append auxiliary metrics while ``_losses_lock`` is already held."""
        for contribution in contributions:
            for metric in contribution.metrics:
                history = getattr(self, metric.history, None)
                if history is None or not hasattr(history, "append"):
                    raise AttributeError(
                        f"Auxiliary loss '{contribution.name}' emitted unknown "
                        f"metric history '{metric.history}'"
                    )
                history.append(metric.value)

    def _compute_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        n_step_lengths: Optional[torch.Tensor] = None,
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

            gamma_pow = self._target_gamma(rewards, n_step_lengths)
            target_q = rewards + (1 - dones) * gamma_pow * next_q

        return current_q, target_q

    def _target_gamma(
        self,
        rewards: torch.Tensor,
        n_step_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if n_step_lengths is not None:
            lengths = n_step_lengths.to(device=rewards.device, dtype=rewards.dtype)
            base = torch.full_like(lengths, float(self.config.GAMMA))
            return torch.pow(base, lengths)
        return torch.full_like(rewards, self._effective_gamma())

    def _effective_gamma(self) -> float:
        use_n_step = getattr(self.config, "USE_N_STEP_RETURNS", False)
        if use_n_step:
            n_steps = getattr(self.config, "N_STEP_SIZE", 3)
            return float(self.config.GAMMA**n_steps)
        return float(self.config.GAMMA)

    def _distributional_logits(self, net: Any, states: torch.Tensor) -> torch.Tensor:
        logits_fn = getattr(net, "distributional_logits", None)
        if not callable(logits_fn):
            raise RuntimeError("distributional network must define distributional_logits()")
        return logits_fn(states)

    def _distributional_probs(self, net: Any, states: torch.Tensor) -> torch.Tensor:
        probs_fn = getattr(net, "distributional_probs", None)
        if not callable(probs_fn):
            raise RuntimeError("distributional network must define distributional_probs()")
        return probs_fn(states)

    def _distributional_support(self) -> torch.Tensor:
        support = getattr(self.policy_net, "support", None)
        if not isinstance(support, torch.Tensor):
            support = torch.linspace(
                float(getattr(self.config, "C51_V_MIN", -20.0)),
                float(getattr(self.config, "C51_V_MAX", 120.0)),
                int(getattr(self.config, "C51_NUM_ATOMS", 51)),
                device=self.device,
            )
        return support.to(self.device)

    def _compute_distributional_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        n_step_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample C51 cross-entropy loss and scalar TD errors."""
        support = self._distributional_support()
        num_atoms = support.numel()
        batch_size = states.shape[0]

        current_logits = self._distributional_logits(self.policy_net, states)
        chosen_logits = current_logits[torch.arange(batch_size, device=self.device), actions]

        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1)
            next_probs = self._distributional_probs(self.target_net, next_states)
            next_dist = next_probs[torch.arange(batch_size, device=self.device), best_actions]
            target_dist = self._project_distribution(
                next_dist=next_dist,
                rewards=rewards,
                dones=dones,
                support=support,
                gamma=self._target_gamma(rewards, n_step_lengths),
            )

        log_probs = F.log_softmax(chosen_logits, dim=1)
        element_loss = -(target_dist * log_probs).sum(dim=1)

        current_probs = F.softmax(chosen_logits, dim=1)
        current_expected = (current_probs * support.view(1, num_atoms)).sum(dim=1)
        target_expected = (target_dist * support.view(1, num_atoms)).sum(dim=1)
        td_errors = (current_expected - target_expected).detach()
        return element_loss, td_errors

    def _project_distribution(
        self,
        *,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        support: torch.Tensor,
        gamma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Project Bellman-updated atom masses back onto the fixed C51 support."""
        batch_size, num_atoms = next_dist.shape
        v_min = float(support[0].item())
        v_max = float(support[-1].item())
        delta_z = (v_max - v_min) / float(num_atoms - 1)

        if isinstance(gamma, torch.Tensor):
            gamma_values = gamma.to(device=support.device, dtype=rewards.dtype)
            if gamma_values.ndim == 0:
                gamma_values = gamma_values.expand(batch_size)
            gamma_values = gamma_values.view(batch_size, 1)
        else:
            gamma_values = torch.full(
                (batch_size, 1),
                float(gamma),
                dtype=rewards.dtype,
                device=support.device,
            )
        target_atoms = rewards.unsqueeze(1) + (
            (1.0 - dones.unsqueeze(1)) * gamma_values * support.view(1, num_atoms)
        )
        target_atoms = target_atoms.clamp(v_min, v_max)
        b = (target_atoms - v_min) / delta_z
        lower = b.floor().long().clamp(0, num_atoms - 1)
        upper = b.ceil().long().clamp(0, num_atoms - 1)

        lower_weight = (upper.float() - b).clamp(min=0.0)
        upper_weight = (b - lower.float()).clamp(min=0.0)
        same_bin = lower == upper
        lower_weight = torch.where(same_bin, torch.ones_like(lower_weight), lower_weight)
        upper_weight = torch.where(same_bin, torch.zeros_like(upper_weight), upper_weight)

        projected = torch.zeros_like(next_dist)
        offset = (
            torch.arange(batch_size, device=next_dist.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(batch_size, num_atoms)
            * num_atoms
        )
        projected.view(-1).index_add_(
            0, (lower + offset).reshape(-1), (next_dist * lower_weight).reshape(-1)
        )
        projected.view(-1).index_add_(
            0, (upper + offset).reshape(-1), (next_dist * upper_weight).reshape(-1)
        )
        return projected

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
        target_params = list(self.target_net.parameters())  # type: ignore[attr-defined]
        policy_params = list(self.policy_net.parameters())  # type: ignore[attr-defined]

        # Validate parameter counts match (guards against torch.compile edge cases)
        if len(target_params) != len(policy_params):
            raise RuntimeError(
                f"Parameter count mismatch: target_net has {len(target_params)} params, "
                f"policy_net has {len(policy_params)} params"
            )

        for target_param, policy_param in zip(target_params, policy_params):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

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

        # Demo-margin decay rides the same per-episode hook (global episode
        # numbers in both scalar and vectorized training); it must track even
        # during the epsilon warmup window, so update before the warmup gate.
        decay_eps = int(getattr(self.config, "DEMO_MARGIN_DECAY_EPISODES", 0))
        if decay_eps > 0 and episode is not None:
            self._demo_margin_scale = max(0.0, 1.0 - episode / decay_eps)

        warmup = getattr(self.config, "EPSILON_WARMUP", 0)

        # Skip decay during warmup period (only if episode is explicitly provided)
        if episode is not None and episode < warmup:
            return

        self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)

    def step_scheduler(self) -> None:
        """Step the learning rate scheduler after each episode."""
        if self.scheduler is not None and self._optimizer_steps_since_scheduler > 0:
            self.scheduler.step()
            self._optimizer_steps_since_scheduler = 0

    def set_learning_rate(self, lr: float) -> None:
        """Override the optimizer learning rate (used for explicit, horizon-matched
        LR decay that the trainer drives per episode)."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def get_network_activations(self) -> dict:
        """Get current network activations for visualization."""
        return self.policy_net.get_activations()

    def get_average_loss(self, n: int = 100) -> float:
        """Get average of last n losses (thread-safe)."""
        with self._losses_lock:
            if not self.losses:
                return 0.0
            # Copy to list inside lock to avoid iterator invalidation
            count = min(n, len(self.losses))
            recent_losses = list(self.losses)[-count:]
            # Compute average inside lock for full thread safety
            return sum(recent_losses) / count if count > 0 else 0.0
