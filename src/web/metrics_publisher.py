"""Metrics state, history, and dashboard publisher for the web dashboard."""

from __future__ import annotations

import base64
import io
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
from src.web.metrics_types import (
    LayerAnalysisData,
    LogLevel,
    LogMessage,
    NeuronInspectionData,
    NNVisualizationData,
    SaveStatus,
    TrainingConfig,
    TrainingState,
)

_logger = get_logger(__name__)

__all__ = [
    "LayerAnalysisData",
    "LogLevel",
    "LogMessage",
    "MetricsPublisher",
    "NeuronInspectionData",
    "NNVisualizationData",
    "SaveStatus",
    "TrainingConfig",
    "TrainingState",
]


class MetricsPublisher:
    """
    Collects and publishes training metrics.

    This class acts as a bridge between the training loop
    and the web dashboard, storing metrics and providing
    them to connected clients.
    """

    DEFAULT_SNAPSHOT_HISTORY_LIMIT = 1000

    def __init__(self, history_length: int = 100000):
        # Keep 100000 episodes of history for full chart scrolling
        # Memory usage: ~100000 * 50 bytes = ~5MB (still negligible)
        self.history_length = history_length
        self.state = TrainingState()
        self.save_status = SaveStatus()

        # Metric history - no maxlen means unlimited (we manage via save/load)
        # Using large maxlen to keep full training history visible in charts
        self.scores: Deque[int] = deque(maxlen=history_length)
        self.losses: Deque[float] = deque(maxlen=history_length)
        self.epsilons: Deque[float] = deque(maxlen=history_length)
        self.rewards: Deque[float] = deque(maxlen=history_length)
        self.q_values: Deque[float] = deque(maxlen=history_length)
        self.episode_lengths: Deque[int] = deque(maxlen=history_length)
        self.wins: Deque[bool] = deque(maxlen=history_length)  # Track actual wins per episode

        # Phase 1: Per-action Q-value history (last 1000 steps)
        self.q_values_left: Deque[float] = deque(maxlen=1000)
        self.q_values_stay: Deque[float] = deque(maxlen=1000)
        self.q_values_right: Deque[float] = deque(maxlen=1000)

        # Phase 1: Action frequency tracking
        self.action_frequency: Dict[str, int] = {
            "left": 0,
            "stay": 0,
            "right": 0,
            "exploration": 0,  # Random actions from exploration
            "exploitation": 0,  # Greedy actions from exploitation
        }

        # Console log history
        self.console_logs: Deque[LogMessage] = deque(maxlen=500)

        # Thread safety for callbacks
        self._callback_lock = threading.Lock()

        # Callbacks
        self._on_update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._on_log_callbacks: List[Callable[[LogMessage], None]] = []
        self._on_save_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Screenshot storage
        self._screenshot_data: Optional[str] = None

        # Timing for episodes/second calculation
        self._episode_times: Deque[float] = deque(maxlen=10)
        self._last_episode_time: float = time.time()

        # Steps/second tracking - use time-windowed approach for accurate real-time rate
        # Store (timestamp, step_count) tuples for rolling window calculation
        self._step_samples: Deque[Tuple[float, int]] = deque(maxlen=50)
        self._last_steps_per_sec: float = 0.0
        self._steps_window_seconds: float = 3.0  # Calculate rate over last 3 seconds

        # Neural network visualization data
        self._nn_data = NNVisualizationData()
        self._on_nn_update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._last_nn_update_time: float = 0.0
        self._nn_update_interval: float = 0.1  # 10 FPS throttle (adaptive)
        self._adaptive_update_enabled: bool = True  # Enable adaptive updates

        # Phase 2: Neuron inspection and layer analysis
        self._neuron_inspection_data: Dict[Tuple[int, int], NeuronInspectionData] = (
            {}
        )  # (layer, neuron) -> data
        self._layer_analysis_data: Dict[int, LayerAnalysisData] = {}  # layer_idx -> data
        self._on_neuron_select_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._on_layer_analysis_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def _calculate_adaptive_update_rate(self, steps_per_sec: float) -> None:
        """
        Phase 1.2: Calculate adaptive visualization update rate based on training speed.

        High-speed training (>2000 steps/sec): Send NN data at 10Hz (50ms)
        Medium speed (500-2000 steps/sec): Send at 20-30Hz
        Slow training (<500 steps/sec): Send at 60Hz for smooth visuals

        This reduces bandwidth and overhead during fast training while
        maintaining visual responsiveness during slow training.
        """
        if not self._adaptive_update_enabled:
            return

        if steps_per_sec > 2000:
            # Very high speed - reduce update frequency
            self._nn_update_interval = 0.1  # 10Hz
        elif steps_per_sec > 1000:
            # High speed
            self._nn_update_interval = 0.067  # ~15Hz
        elif steps_per_sec > 500:
            # Medium speed
            self._nn_update_interval = 0.033  # ~30Hz
        else:
            # Slow/visual training - keep responsive
            self._nn_update_interval = 0.016  # ~60Hz

    def update(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: float,
        total_steps: int = 0,
        won: bool = False,
        reward: float = 0.0,
        memory_size: int = 0,
        avg_q_value: float = 0.0,
        exploration_actions: int = 0,
        exploitation_actions: int = 0,
        target_updates: int = 0,
        bricks_broken: int = 0,
        episode_length: int = 0,
        q_value_left: float = 0.0,
        q_value_stay: float = 0.0,
        q_value_right: float = 0.0,
        selected_action: Optional[int] = None,
        game_name: str = "",
        cc_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update metrics with new episode data.

        Args:
            q_value_left, q_value_stay, q_value_right: Q-values for each action (Phase 1)
            selected_action: Index of action taken (0=LEFT, 1=STAY, 2=RIGHT)
            game_name: Active game (drives which game-specific dashboard panels show)
            cc_info: Crystal Caves per-episode info dict (progress, progress_parts,
                crystals, end_reason). Populates the Crystal Caves dashboard panel.
        """
        if game_name:
            self.state.game_name = game_name
        if cc_info is not None:
            self._update_crystal_caves(cc_info, won)
        self.state.episode = episode
        self.state.score = score
        self.state.best_score = max(self.state.best_score, score)
        self.state.epsilon = epsilon
        self.state.loss = loss
        self.state.total_steps = total_steps
        self.state.memory_size = memory_size
        self.state.avg_q_value = avg_q_value
        self.state.exploration_actions = exploration_actions
        self.state.exploitation_actions = exploitation_actions
        self.state.target_updates = target_updates
        self.state.bricks_broken_total += bricks_broken
        self.state.total_reward += reward

        # Phase 1: Record per-action Q-values
        self.state.q_value_left = q_value_left
        self.state.q_value_stay = q_value_stay
        self.state.q_value_right = q_value_right

        # Phase 1: Record per-action Q-value history
        self.q_values_left.append(q_value_left)
        self.q_values_stay.append(q_value_stay)
        self.q_values_right.append(q_value_right)

        # Phase 1: Track action frequency
        if selected_action is not None:
            action_names = ["left", "stay", "right"]
            if 0 <= selected_action < len(action_names):
                self.state.action_count_left += 1 if selected_action == 0 else 0
                self.state.action_count_stay += 1 if selected_action == 1 else 0
                self.state.action_count_right += 1 if selected_action == 2 else 0
                self.action_frequency[action_names[selected_action]] += 1

        # Phase 1: Track exploration vs exploitation
        if exploration_actions > 0 or exploitation_actions > 0:
            if exploration_actions > self.action_frequency.get("exploration", 0):
                self.action_frequency["exploration"] = exploration_actions
            if exploitation_actions > self.action_frequency.get("exploitation", 0):
                self.action_frequency["exploitation"] = exploitation_actions

        self.scores.append(score)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        self.q_values.append(avg_q_value)
        self.episode_lengths.append(episode_length)
        self.wins.append(won)  # Track actual wins

        # Calculate episodes per second
        current_time = time.time()
        self._episode_times.append(current_time - self._last_episode_time)
        self._last_episode_time = current_time
        if self._episode_times:
            avg_time = sum(self._episode_times) / len(self._episode_times)
            self.state.episodes_per_second = 1.0 / avg_time if avg_time > 0 else 0.0

        # Bug 67 fix: Thread-safe access to _step_samples
        # Bug 79 fix: Rate limiting - only sample every 50ms to reduce overhead
        with self._callback_lock:
            # Calculate steps per second using time-windowed approach
            # This gives accurate real-time rate instead of lifetime average
            last_sample_time = self._step_samples[-1][0] if self._step_samples else 0
            if current_time - last_sample_time >= 0.05:  # 50ms minimum between samples
                self._step_samples.append((current_time, total_steps))

            # Remove samples older than window, but keep at least 2 for rate calculation
            cutoff_time = current_time - self._steps_window_seconds
            while len(self._step_samples) > 2 and self._step_samples[0][0] < cutoff_time:
                self._step_samples.popleft()

            # Calculate rate from remaining samples
            if len(self._step_samples) >= 2:
                oldest_time, oldest_steps = self._step_samples[0]
                newest_time, newest_steps = self._step_samples[-1]
                time_delta = newest_time - oldest_time
                step_delta = newest_steps - oldest_steps

                if (
                    time_delta > 0.1 and step_delta > 0
                ):  # Need at least 100ms of data and positive steps
                    self._last_steps_per_sec = step_delta / time_delta
                    self.state.steps_per_second = self._last_steps_per_sec
                else:
                    self.state.steps_per_second = self._last_steps_per_sec
            else:
                self.state.steps_per_second = self._last_steps_per_sec

        # Phase 1.2: Adjust visualization update rate based on training speed
        self._calculate_adaptive_update_rate(self.state.steps_per_second)

        # Calculate win rate from actual wins (game-specific)
        # Use the actual 'won' flag from the game, not hardcoded score thresholds
        if len(self.wins) > 0:
            recent_wins = list(self.wins)[-100:]
            self.state.win_rate = sum(1 for w in recent_wins if w) / len(recent_wins)
        else:
            self.state.win_rate = 0.0

        # Notify callbacks (thread-safe copy to avoid modification during iteration)
        with self._callback_lock:
            callbacks = self._on_update_callbacks.copy()
        for callback in callbacks:
            callback(self.get_snapshot())

    def _update_crystal_caves(self, info: Dict[str, Any], won: bool) -> None:
        """Populate Crystal Caves-specific dashboard state from a game info dict.

        ``info`` is the dict returned by ``CrystalCaves._info()``: it carries the
        completion potential (``progress``), its component breakdown
        (``progress_parts`` = crystal/switch/depth), the crystal counts, and how
        the episode ended (``end_reason``)."""
        self.state.cc_active = True
        progress = float(info.get("progress", 0.0) or 0.0)
        self.state.cc_progress = progress
        self.state.cc_best_progress = max(self.state.cc_best_progress, progress)

        parts = info.get("progress_parts") or {}
        if isinstance(parts, dict):
            self.state.cc_crystal_frac = float(parts.get("crystal_frac", 0.0) or 0.0)
            self.state.cc_switch_done = float(parts.get("switch_done", 0.0) or 0.0)
            self.state.cc_depth_frac = float(parts.get("depth_frac", 0.0) or 0.0)

        self.state.cc_crystals_remaining = int(info.get("crystals_remaining", 0) or 0)
        self.state.cc_initial_crystals = int(info.get("initial_crystals", 0) or 0)
        self.state.cc_level_name = str(info.get("level_name", "") or "")

        # Only count terminal reasons (skip the in-progress "running" sentinel).
        reason = str(info.get("end_reason", "") or "")
        if reason and reason != "running":
            self.state.cc_end_reason = reason
            counts = self.state.cc_end_reason_counts
            counts[reason] = counts.get(reason, 0) + 1

    def log(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None) -> None:
        """Add a log message to the console."""
        log_entry = LogMessage(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level,
            message=message,
            data=data,
        )
        self.console_logs.append(log_entry)

        # Notify log callbacks (thread-safe copy to avoid modification during iteration)
        with self._callback_lock:
            callbacks = self._on_log_callbacks.copy()
        for callback in callbacks:
            callback(log_entry)

    def set_screenshot(self, surface) -> None:
        """Store a screenshot from pygame surface."""
        try:
            import pygame

            # Convert pygame surface to raw string data
            raw_str = pygame.image.tostring(surface, "RGB")
            width, height = surface.get_size()

            # Use PIL to convert to PNG
            try:
                from PIL import Image

                img = Image.frombytes("RGB", (width, height), raw_str)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                self._screenshot_data = base64.b64encode(buffer.read()).decode("utf-8")
            except ImportError:
                # Fallback: try direct pygame save to BytesIO
                try:
                    buffer = io.BytesIO()
                    # Create a temp surface copy for saving
                    temp_surface = surface.copy()
                    # Use PNG format directly without filename extension
                    pygame.image.save(temp_surface, buffer)
                    buffer.seek(0)
                    self._screenshot_data = base64.b64encode(buffer.read()).decode("utf-8")
                except Exception as fallback_error:
                    _logger.warning(f"Screenshot fallback error: {fallback_error}")
                    self._screenshot_data = None
        except Exception as e:
            _logger.warning(f"Screenshot error: {e}")
            self._screenshot_data = None  # Clear corrupted data

    def get_screenshot(self) -> Optional[str]:
        """Get the latest screenshot as base64."""
        return self._screenshot_data

    @staticmethod
    def _history_values(values: Deque[Any], history_limit: Optional[int]) -> List[Any]:
        items = list(values)
        if history_limit is None:
            return items
        if history_limit <= 0:
            return []
        return items[-history_limit:]

    def get_snapshot(self, history_limit: Optional[int] = None) -> Dict[str, Any]:
        """Get current state as dictionary, optionally limiting history arrays."""
        return {
            "state": asdict(self.state),
            "history": {
                "scores": self._history_values(self.scores, history_limit),
                "losses": self._history_values(self.losses, history_limit),
                "epsilons": self._history_values(self.epsilons, history_limit),
                "rewards": self._history_values(self.rewards, history_limit),
                "q_values": self._history_values(self.q_values, history_limit),
                "episode_lengths": self._history_values(self.episode_lengths, history_limit),
            },
        }

    def get_console_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent console logs."""
        logs = list(self.console_logs)[-limit:]
        return [log.to_dict() for log in logs]

    def on_update(self, callback) -> None:
        """Register a callback for metric updates."""
        with self._callback_lock:
            self._on_update_callbacks.append(callback)

    def on_log(self, callback) -> None:
        """Register a callback for log messages."""
        with self._callback_lock:
            self._on_log_callbacks.append(callback)

    def set_paused(self, paused: bool) -> None:
        """Set training paused state."""
        self.state.is_paused = paused

    def set_running(self, running: bool) -> None:
        """Set training running state."""
        self.state.is_running = running

    def set_speed(self, speed: float) -> None:
        """Set game speed."""
        self.state.game_speed = speed

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update training configuration."""
        if "learning_rate" in config:
            self.state.learning_rate = config["learning_rate"]
        if "batch_size" in config:
            self.state.batch_size = config["batch_size"]
        if "learn_every" in config:
            self.state.learn_every = config["learn_every"]
        if "gradient_steps" in config:
            self.state.gradient_steps = config["gradient_steps"]

    def set_performance_mode(self, mode: str) -> None:
        """Set performance mode preset."""
        self.state.performance_mode = mode

    def set_system_info(
        self,
        device: str,
        torch_compiled: bool,
        target_episodes: int,
        headless: bool = False,
    ) -> None:
        """Set system information."""
        self.state.device = device
        self.state.torch_compiled = torch_compiled
        self.state.target_episodes = target_episodes
        self.state.training_start_time = time.time()
        self.state.headless = headless

    def record_save(self, filename: str, reason: str, episode: int, best_score: int) -> None:
        """Record a model save event."""
        self.save_status.last_save_time = time.time()
        self.save_status.last_save_filename = filename
        self.save_status.last_save_reason = reason
        self.save_status.last_save_episode = episode
        self.save_status.last_save_best_score = best_score
        self.save_status.saves_this_session += 1

        # Notify save callbacks (thread-safe copy to avoid modification during iteration)
        save_info = self.get_save_status()
        with self._callback_lock:
            callbacks = self._on_save_callbacks.copy()
        for callback in callbacks:
            callback(save_info)

    def get_save_status(self) -> Dict[str, Any]:
        """Get current save status."""
        time_since_save = 0.0
        if self.save_status.last_save_time > 0:
            time_since_save = time.time() - self.save_status.last_save_time

        return {
            "last_save_time": self.save_status.last_save_time,
            "last_save_filename": self.save_status.last_save_filename,
            "last_save_reason": self.save_status.last_save_reason,
            "last_save_episode": self.save_status.last_save_episode,
            "last_save_best_score": self.save_status.last_save_best_score,
            "saves_this_session": self.save_status.saves_this_session,
            "time_since_save": time_since_save,
            "time_since_save_str": self._format_time_ago(time_since_save),
        }

    def _format_time_ago(self, seconds: float) -> str:
        """Format seconds as human-readable time ago string."""
        if seconds <= 0:
            return "Never"
        if seconds < 60:
            return f"{int(seconds)}s ago"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m ago"

    def on_save(self, callback) -> None:
        """Register a callback for save events."""
        with self._callback_lock:
            self._on_save_callbacks.append(callback)

    def update_nn_visualization(
        self,
        layer_info: List[Dict[str, Any]],
        activations: Dict[str, List[float]],
        q_values: List[float],
        selected_action: int,
        weights: List[List[List[float]]],
        step: int,
        action_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Update neural network visualization data.

        This method is throttled to ~10 FPS to prevent overwhelming the network.

        Args:
            layer_info: List of layer info dicts with 'name', 'neurons', 'type'
            activations: Dict mapping layer keys to lists of activation values
            q_values: Q-values for each action
            selected_action: Currently selected action index
            weights: Sampled weight matrices
            step: Current training step
            action_labels: Labels for each action
        """
        current_time = time.time()

        # Throttle updates to ~10 FPS
        if not self.should_update_nn_visualization(current_time):
            return

        self._last_nn_update_time = current_time

        # Update stored data
        self._nn_data.layer_info = layer_info
        self._nn_data.activations = activations
        self._nn_data.q_values = q_values
        self._nn_data.selected_action = selected_action
        self._nn_data.weights = weights
        self._nn_data.step = step
        if action_labels:
            self._nn_data.action_labels = action_labels

        # Phase 1.3: Notify callbacks with selective weight transmission
        # Only include weights if they were significantly updated (every 100 steps)
        nn_dict = self._nn_data.to_dict(include_weights=False)
        with self._callback_lock:
            callbacks = self._on_nn_update_callbacks.copy()
        for callback in callbacks:
            callback(nn_dict)

    def should_update_nn_visualization(self, current_time: Optional[float] = None) -> bool:
        """Return whether a neural-network visualization update should run."""
        check_time = time.time() if current_time is None else current_time
        return check_time - self._last_nn_update_time >= self._nn_update_interval

    def get_nn_visualization(self, include_weights: bool = False) -> Dict[str, Any]:
        """
        Get current neural network visualization data.

        Phase 1.3: Optionally include weights (only when explicitly requested or
        on periodic updates to reduce bandwidth).
        """
        return self._nn_data.to_dict(include_weights=include_weights)

    def on_nn_update(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for neural network visualization updates."""
        with self._callback_lock:
            self._on_nn_update_callbacks.append(callback)

    # ===== Phase 2: Neuron Inspection & Layer Analysis =====

    def update_neuron_inspection(
        self,
        layer_idx: int,
        neuron_idx: int,
        layer_name: str,
        current_activation: float,
        activation_history: Optional[List[float]] = None,
        incoming_weights: Optional[List[float]] = None,
        outgoing_weights: Optional[List[float]] = None,
        q_contributions: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Phase 2: Update neuron inspection data.

        Args:
            layer_idx: Layer index
            neuron_idx: Neuron index in layer
            layer_name: Human-readable layer name
            current_activation: Current activation value
            activation_history: List of recent activation values
            incoming_weights: Weights from previous layer
            outgoing_weights: Weights to next layer
            q_contributions: Contribution to each Q-value
        """
        key = (layer_idx, neuron_idx)
        if key not in self._neuron_inspection_data:
            self._neuron_inspection_data[key] = NeuronInspectionData(
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
                layer_name=layer_name,
            )

        data = self._neuron_inspection_data[key]
        data.current_activation = current_activation

        if activation_history is not None:
            data.activation_history = list(activation_history)[-500:]  # Keep last 500

        if incoming_weights is not None:
            incoming_weight_array = np.asarray(incoming_weights)
            if incoming_weight_array.size > 0:
                data.incoming_weights = incoming_weight_array.tolist()
                data.incoming_weight_stats = {
                    "mean": float(np.mean(incoming_weight_array)),
                    "std": float(np.std(incoming_weight_array)),
                    "min": float(np.min(incoming_weight_array)),
                    "max": float(np.max(incoming_weight_array)),
                }

        if outgoing_weights is not None:
            outgoing_weight_array = np.asarray(outgoing_weights)
            if outgoing_weight_array.size > 0:
                data.outgoing_weights = outgoing_weight_array.tolist()
                data.outgoing_weight_stats = {
                    "mean": float(np.mean(outgoing_weight_array)),
                    "std": float(np.std(outgoing_weight_array)),
                    "min": float(np.min(outgoing_weight_array)),
                    "max": float(np.max(outgoing_weight_array)),
                }

        if q_contributions:
            data.q_value_contributions = q_contributions

    def get_neuron_details(self, layer_idx: int, neuron_idx: int) -> Dict[str, Any]:
        """
        Phase 2: Get detailed information about a specific neuron.

        Args:
            layer_idx: Layer index
            neuron_idx: Neuron index

        Returns:
            Dictionary with neuron details
        """
        key = (layer_idx, neuron_idx)
        if key not in self._neuron_inspection_data:
            return {"error": "Neuron not found"}
        return self._neuron_inspection_data[key].to_dict()

    def update_layer_analysis(
        self,
        layer_idx: int,
        layer_name: str,
        neuron_count: int,
        activations: np.ndarray,
        weights: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None,
    ) -> None:
        """
        Phase 2: Update layer analysis statistics.

        Args:
            layer_idx: Layer index
            layer_name: Layer name
            neuron_count: Number of neurons in layer
            activations: Current activation values (shape: neuron_count)
            weights: Weight matrix (optional)
            gradients: Gradient values (optional)
        """
        if layer_idx not in self._layer_analysis_data:
            self._layer_analysis_data[layer_idx] = LayerAnalysisData(
                layer_idx=layer_idx,
                layer_name=layer_name,
                neuron_count=neuron_count,
            )

        data = self._layer_analysis_data[layer_idx]

        # Update activation statistics
        activations = np.asarray(activations)
        if activations.size > 0:
            data.avg_activation = float(np.mean(activations))
            data.activation_std = float(np.std(activations))
            data.activation_min = float(np.min(activations))
            data.activation_max = float(np.max(activations))

            # Count dead and saturated neurons
            data.dead_neuron_count = int(np.sum(np.abs(activations) < 0.01))
            data.saturated_neuron_count = int(np.sum(np.abs(activations) > 0.95))

            # Create activation histogram
            hist, _ = np.histogram(activations, bins=20)
            data.activation_histogram = hist.tolist()
        else:
            data.avg_activation = 0.0
            data.activation_std = 0.0
            data.activation_min = 0.0
            data.activation_max = 0.0
            data.dead_neuron_count = 0
            data.saturated_neuron_count = 0
            data.activation_histogram = [0] * 20

        # Update weight statistics if provided
        if weights is not None:
            flat_weights = np.asarray(weights).flatten()
            if flat_weights.size > 0:
                data.weight_mean = float(np.mean(flat_weights))
                data.weight_std = float(np.std(flat_weights))
                data.weight_min = float(np.min(flat_weights))
                data.weight_max = float(np.max(flat_weights))

                # Weight histogram
                hist, _ = np.histogram(flat_weights, bins=20)
                data.weight_histogram = hist.tolist()
            else:
                data.weight_mean = 0.0
                data.weight_std = 0.0
                data.weight_min = 0.0
                data.weight_max = 0.0
                data.weight_histogram = [0] * 20

        # Update gradient statistics if provided
        if gradients is not None:
            flat_grads = np.asarray(gradients).flatten()
            if flat_grads.size > 0:
                data.gradient_mean = float(np.mean(np.abs(flat_grads)))
                data.gradient_std = float(np.std(flat_grads))
                data.gradient_max_magnitude = float(np.max(np.abs(flat_grads)))
            else:
                data.gradient_mean = 0.0
                data.gradient_std = 0.0
                data.gradient_max_magnitude = 0.0

    def get_layer_analysis(self, layer_idx: int) -> Dict[str, Any]:
        """
        Phase 2: Get analysis data for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Dictionary with layer analysis
        """
        if layer_idx not in self._layer_analysis_data:
            return {"error": "Layer not found"}
        return self._layer_analysis_data[layer_idx].to_dict()

    def get_all_layer_analysis(self) -> List[Dict[str, Any]]:
        """
        Phase 2: Get analysis for all layers.

        Returns:
            List of layer analysis dictionaries
        """
        return [
            data.to_dict()
            for data in sorted(self._layer_analysis_data.values(), key=lambda x: x.layer_idx)
        ]

    def on_neuron_select(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Phase 2: Register callback for neuron selection."""
        with self._callback_lock:
            self._on_neuron_select_callbacks.append(callback)

    def on_layer_analysis(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Phase 2: Register callback for layer analysis updates."""
        with self._callback_lock:
            self._on_layer_analysis_callbacks.append(callback)

    def reset_all_state(self) -> None:
        """Reset all training state - used when starting fresh."""
        # Clear all history
        self.scores.clear()
        self.losses.clear()
        self.epsilons.clear()
        self.rewards.clear()
        self.q_values.clear()
        self.episode_lengths.clear()

        # Clear console logs
        self.console_logs.clear()

        # Reset state to initial values
        self.state.episode = 0
        self.state.score = 0
        self.state.best_score = 0
        self.state.epsilon = 1.0
        self.state.loss = 0.0
        self.state.total_steps = 0
        self.state.win_rate = 0.0
        self.state.memory_size = 0
        self.state.avg_q_value = 0.0
        self.state.exploration_actions = 0
        self.state.exploitation_actions = 0
        self.state.target_updates = 0
        self.state.total_reward = 0.0
        self.state.bricks_broken_total = 0
        self.state.episodes_per_second = 0.0
        self.state.steps_per_second = 0.0

        # Reset Crystal Caves telemetry (keep cc_active/difficulty/level — those
        # describe the run, not the per-episode progress)
        self.state.cc_progress = 0.0
        self.state.cc_best_progress = 0.0
        self.state.cc_crystal_frac = 0.0
        self.state.cc_switch_done = 0.0
        self.state.cc_depth_frac = 0.0
        self.state.cc_crystals_remaining = 0
        self.state.cc_end_reason = ""
        self.state.cc_end_reason_counts = {}

        # Reset timing
        self._episode_times.clear()
        self._last_episode_time = time.time()
        self._step_samples.clear()
        self._last_steps_per_sec = 0.0

        # Reset training start time
        self.state.training_start_time = time.time()

        # Reset save status
        self.save_status = SaveStatus()

        # Clear screenshot
        self._screenshot_data = None

        # Reset NN visualization
        self._nn_data = NNVisualizationData()
