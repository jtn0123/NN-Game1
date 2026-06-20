"""Serializable dashboard metric and analysis types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Log levels for console messages."""

    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    METRIC = "metric"
    ACTION = "action"


@dataclass
class LogMessage:
    """A single log entry."""

    timestamp: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class TrainingState:
    """Current training state for API."""

    episode: int = 0
    score: int = 0
    best_score: int = 0
    epsilon: float = 1.0
    loss: float = 0.0
    total_steps: int = 0
    win_rate: float = 0.0
    is_paused: bool = False
    is_running: bool = False
    game_speed: float = 1.0
    learning_rate: float = 0.0001
    batch_size: int = 128
    memory_size: int = 0
    memory_capacity: int = 100000
    episodes_per_second: float = 0.0
    exploration_actions: int = 0
    exploitation_actions: int = 0
    target_updates: int = 0
    avg_q_value: float = 0.0
    bricks_broken_total: int = 0
    total_reward: float = 0.0
    # Performance optimization settings
    learn_every: int = 1
    gradient_steps: int = 1
    steps_per_second: float = 0.0
    # System info
    device: str = "cpu"
    torch_compiled: bool = False
    # Training time tracking
    training_start_time: float = 0.0
    target_episodes: int = 0  # 0 = unlimited
    # Performance mode: 'normal', 'fast', 'turbo', 'ultra'
    performance_mode: str = "normal"
    # Number of parallel environments
    num_envs: int = 1
    # Headless mode (no pygame, no screenshots)
    headless: bool = False
    # Per-action Q-values (for Phase 1 enhanced analysis)
    q_value_left: float = 0.0
    q_value_stay: float = 0.0
    q_value_right: float = 0.0
    # Action frequency tracking
    action_count_left: int = 0
    action_count_stay: int = 0
    action_count_right: int = 0
    # Game identity (drives which game-specific panels the dashboard shows)
    game_name: str = ""
    # --- Crystal Caves live telemetry (left at 0/empty for other games) ---
    cc_active: bool = False
    cc_progress: float = 0.0  # overall completion potential Φ, 0..1
    cc_best_progress: float = 0.0  # best Φ seen this run (only goes up)
    cc_crystal_frac: float = 0.0  # fraction of this level's crystals collected, 0..1
    cc_switch_done: float = 0.0  # 1.0 once the switch is thrown (0 if none/unthrown)
    cc_depth_frac: float = 0.0  # how deep into the cave the agent reached, 0..1
    cc_crystals_remaining: int = 0
    cc_initial_crystals: int = 0
    cc_switches_total: int = 0
    cc_switches_used: int = 0
    cc_level_name: str = ""
    cc_difficulty: str = ""
    cc_end_reason: str = ""  # how the last finished episode ended
    cc_end_reason_counts: Dict[str, int] = field(default_factory=dict)
    # --- Held-out evaluation (the trustworthy generalisation measure, distinct
    # from the training win_rate above; populated only when periodic eval runs) ---
    eval_ran: bool = False
    eval_episode: int = 0  # training episode at the last eval
    eval_mean_score: float = 0.0
    eval_std_score: float = 0.0
    eval_median_score: float = 0.0
    eval_win_rate: float = 0.0
    eval_best_mean: float = 0.0  # best eval mean so far (monotonic)
    eval_delta_from_best: float = 0.0  # latest mean minus best mean; 0 means tied/new best
    eval_is_new_best: bool = False
    eval_is_baseline: bool = False  # true when showing a restored saved best before live eval
    eval_num_games: int = 0
    eval_history: List[float] = field(default_factory=list)  # eval-mean trajectory (sparkline)
    # --- Crystal Caves curriculum state ---
    curriculum_active: bool = False
    curriculum_stage_index: int = 0
    curriculum_stage_total: int = 0
    curriculum_stage_id: str = ""
    curriculum_stage_name: str = ""
    curriculum_stage_difficulty: str = ""
    curriculum_stage_families: str = ""
    curriculum_stage_start_episode: int = 0
    curriculum_stage_target_episode: int = 0
    curriculum_stage_status: str = ""
    curriculum_stage_gate: str = ""
    curriculum_next_stage_name: str = ""


@dataclass
class SaveStatus:
    """Track last save information."""

    last_save_time: float = 0.0
    last_save_filename: str = ""
    last_save_reason: str = ""
    last_save_episode: int = 0
    last_save_best_score: int = 0
    saves_this_session: int = 0


@dataclass
class TrainingConfig:
    """Configurable training parameters."""

    learning_rate: float = 0.0001
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 128
    gamma: float = 0.99
    target_update_freq: int = 1000
    # Performance settings
    learn_every: int = 1
    gradient_steps: int = 1


@dataclass
class NNVisualizationData:
    """Neural network visualization data for web streaming."""

    # Layer structure info
    layer_info: List[Dict[str, Any]] = field(default_factory=list)
    # Activations per layer (normalized values)
    activations: Dict[str, List[float]] = field(default_factory=dict)
    # Q-values for current state
    q_values: List[float] = field(default_factory=list)
    # Selected action index
    selected_action: int = 0
    # Sampled weights for connections (list of weight matrices as nested lists)
    weights: List[List[List[float]]] = field(default_factory=list)
    # Current step count
    step: int = 0
    # Action labels
    action_labels: List[str] = field(default_factory=lambda: ["LEFT", "STAY", "RIGHT"])
    # Phase 1.3: Only include weights if explicitly requested (reduces bandwidth)
    include_weights: bool = False
    # Track last weight update to avoid unnecessary transmission
    _last_weights_step: int = 0

    def to_dict(self, include_weights: bool = False) -> Dict[str, Any]:
        """
        Convert to dict, optionally excluding weights to reduce bandwidth.

        Phase 1.3: Only include weights if explicitly requested or if they've changed
        significantly since last transmission.
        """
        data = {
            "layer_info": self.layer_info,
            "activations": self.activations,
            "q_values": self.q_values,
            "selected_action": self.selected_action,
            "step": self.step,
            "action_labels": self.action_labels,
        }
        # Only include weights if requested or every 100 steps
        if include_weights or (self.step - self._last_weights_step > 100):
            data["weights"] = self.weights
            self._last_weights_step = self.step
        else:
            data["weights"] = []  # Empty weights signal "no weight update"
        return data


@dataclass
class NeuronInspectionData:
    """
    Phase 2: Neuron inspection and analysis data.

    Tracks per-neuron activation history and statistics for interactive inspection.
    """

    # Layer and neuron identification
    layer_idx: int = 0
    neuron_idx: int = 0
    layer_name: str = ""

    # Activation history (last 500 steps)
    activation_history: List[float] = field(default_factory=list)

    # Incoming weights (from previous layer)
    incoming_weights: List[float] = field(default_factory=list)
    incoming_weight_stats: Dict[str, float] = field(default_factory=dict)

    # Outgoing weights (to next layer)
    outgoing_weights: List[float] = field(default_factory=list)
    outgoing_weight_stats: Dict[str, float] = field(default_factory=dict)

    # Contribution to Q-values
    q_value_contributions: Dict[str, float] = field(default_factory=dict)

    # Current statistics
    current_activation: float = 0.0
    dead_steps: int = 0  # Steps where activation was near zero

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "neuron_idx": self.neuron_idx,
            "layer_name": self.layer_name,
            "activation_history": self.activation_history[-100:],  # Last 100 for visualization
            "current_activation": self.current_activation,
            "incoming_weights": self.incoming_weights[:50],  # Sample top 50
            "incoming_weight_stats": self.incoming_weight_stats,
            "outgoing_weights": self.outgoing_weights[:50],  # Sample top 50
            "outgoing_weight_stats": self.outgoing_weight_stats,
            "q_value_contributions": self.q_value_contributions,
            "dead_steps": self.dead_steps,
        }


@dataclass
class LayerAnalysisData:
    """
    Phase 2: Per-layer analysis data.

    Tracks statistics for each layer (dead neurons, saturation, etc.)
    """

    layer_idx: int = 0
    layer_name: str = ""
    neuron_count: int = 0

    # Activation statistics
    avg_activation: float = 0.0
    activation_std: float = 0.0
    activation_min: float = 0.0
    activation_max: float = 0.0
    activation_histogram: List[int] = field(default_factory=list)

    # Neuron health
    dead_neuron_count: int = 0  # Activation < 0.01
    saturated_neuron_count: int = 0  # Activation > 0.95

    # Weight statistics
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    weight_histogram: List[int] = field(default_factory=list)

    # Gradient statistics (when available)
    gradient_mean: float = 0.0
    gradient_std: float = 0.0
    gradient_max_magnitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "neuron_count": self.neuron_count,
            "avg_activation": self.avg_activation,
            "activation_std": self.activation_std,
            "activation_min": self.activation_min,
            "activation_max": self.activation_max,
            "activation_histogram": self.activation_histogram,
            "dead_neuron_count": self.dead_neuron_count,
            # Bug 71 fix: Explicit check for neuron_count > 0 instead of masking with max(1, ...)
            "dead_neuron_percent": (
                (self.dead_neuron_count / self.neuron_count * 100) if self.neuron_count > 0 else 0.0
            ),
            "saturated_neuron_count": self.saturated_neuron_count,
            "saturated_percent": (
                (self.saturated_neuron_count / self.neuron_count * 100)
                if self.neuron_count > 0
                else 0.0
            ),
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "weight_histogram": self.weight_histogram,
            "gradient_mean": self.gradient_mean,
            "gradient_std": self.gradient_std,
            "gradient_max_magnitude": self.gradient_max_magnitude,
        }
