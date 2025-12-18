"""
Web Dashboard Server
====================

Flask + SocketIO server for real-time training visualization.

Features:
    - REST API for model info and current stats
    - WebSocket events for live metrics streaming
    - Runs in background thread alongside training
    - Training controls (pause, save, adjust speed, reset, load model)
    - Real-time console logging with filterable log levels

Usage:
    >>> from src.web import WebDashboard
    >>> dashboard = WebDashboard(port=5000)
    >>> dashboard.start()
    >>> # ... during training ...
    >>> dashboard.emit_metrics(episode, score, loss, ...)
    >>> dashboard.log("Training started", level="info")
    >>> dashboard.stop()
"""

import os
import threading
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Deque, Callable, Tuple
from dataclasses import dataclass, asdict, field
from collections import deque
from enum import Enum
import base64
import io
import numpy as np

try:
    # Suppress werkzeug logging BEFORE importing Flask
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').disabled = True
    
    from flask import Flask, render_template, jsonify, request, make_response
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard unavailable.")
    print("Install with: pip install flask flask-socketio eventlet")

import sys
sys.path.append('../..')
from config import Config
from src.utils.logger import get_logger

# Module logger
_logger = get_logger(__name__)


def _make_json_safe(obj: Any) -> Any:
    """
    Convert NumPy types to native Python types for JSON serialization.

    Recursively processes dictionaries and lists to ensure all NumPy types
    are converted to JSON-compatible Python types.

    Args:
        obj: Object to convert (can be NumPy type, dict, list, or primitive)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(item) for item in obj]
    return obj


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
            'timestamp': self.timestamp,
            'level': self.level,
            'message': self.message,
            'data': self.data
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
            'layer_info': self.layer_info,
            'activations': self.activations,
            'q_values': self.q_values,
            'selected_action': self.selected_action,
            'step': self.step,
            'action_labels': self.action_labels
        }
        # Only include weights if requested or every 100 steps
        if include_weights or (self.step - self._last_weights_step > 100):
            data['weights'] = self.weights
            self._last_weights_step = self.step
        else:
            data['weights'] = []  # Empty weights signal "no weight update"
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
            'layer_idx': self.layer_idx,
            'neuron_idx': self.neuron_idx,
            'layer_name': self.layer_name,
            'activation_history': self.activation_history[-100:],  # Last 100 for visualization
            'current_activation': self.current_activation,
            'incoming_weights': self.incoming_weights[:50],  # Sample top 50
            'incoming_weight_stats': self.incoming_weight_stats,
            'outgoing_weights': self.outgoing_weights[:50],  # Sample top 50
            'outgoing_weight_stats': self.outgoing_weight_stats,
            'q_value_contributions': self.q_value_contributions,
            'dead_steps': self.dead_steps,
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
            'layer_idx': self.layer_idx,
            'layer_name': self.layer_name,
            'neuron_count': self.neuron_count,
            'avg_activation': self.avg_activation,
            'activation_std': self.activation_std,
            'activation_min': self.activation_min,
            'activation_max': self.activation_max,
            'activation_histogram': self.activation_histogram,
            'dead_neuron_count': self.dead_neuron_count,
            # Bug 71 fix: Explicit check for neuron_count > 0 instead of masking with max(1, ...)
            'dead_neuron_percent': (self.dead_neuron_count / self.neuron_count * 100) if self.neuron_count > 0 else 0.0,
            'saturated_neuron_count': self.saturated_neuron_count,
            'saturated_percent': (self.saturated_neuron_count / self.neuron_count * 100) if self.neuron_count > 0 else 0.0,
            'weight_mean': self.weight_mean,
            'weight_std': self.weight_std,
            'weight_min': self.weight_min,
            'weight_max': self.weight_max,
            'weight_histogram': self.weight_histogram,
            'gradient_mean': self.gradient_mean,
            'gradient_std': self.gradient_std,
            'gradient_max_magnitude': self.gradient_max_magnitude,
        }


class MetricsPublisher:
    """
    Collects and publishes training metrics.
    
    This class acts as a bridge between the training loop
    and the web dashboard, storing metrics and providing
    them to connected clients.
    """
    
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
            'left': 0,
            'stay': 0,
            'right': 0,
            'exploration': 0,  # Random actions from exploration
            'exploitation': 0,  # Greedy actions from exploitation
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
        self._neuron_inspection_data: Dict[Tuple[int, int], NeuronInspectionData] = {}  # (layer, neuron) -> data
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
        selected_action: Optional[int] = None
    ) -> None:
        """
        Update metrics with new episode data.

        Args:
            q_value_left, q_value_stay, q_value_right: Q-values for each action (Phase 1)
            selected_action: Index of action taken (0=LEFT, 1=STAY, 2=RIGHT)
        """
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
            action_names = ['left', 'stay', 'right']
            if 0 <= selected_action < len(action_names):
                self.state.action_count_left += (1 if selected_action == 0 else 0)
                self.state.action_count_stay += (1 if selected_action == 1 else 0)
                self.state.action_count_right += (1 if selected_action == 2 else 0)
                self.action_frequency[action_names[selected_action]] += 1

        # Phase 1: Track exploration vs exploitation
        if exploration_actions > 0 or exploitation_actions > 0:
            if exploration_actions > self.action_frequency.get('exploration', 0):
                self.action_frequency['exploration'] = exploration_actions
            if exploitation_actions > self.action_frequency.get('exploitation', 0):
                self.action_frequency['exploitation'] = exploitation_actions

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

                if time_delta > 0.1 and step_delta > 0:  # Need at least 100ms of data and positive steps
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
    
    def log(
        self,
        message: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a log message to the console."""
        log_entry = LogMessage(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:12],  # Properly format with milliseconds
            level=level,
            message=message,
            data=data
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
            raw_str = pygame.image.tostring(surface, 'RGB')
            width, height = surface.get_size()
            
            # Use PIL to convert to PNG
            try:
                from PIL import Image
                img = Image.frombytes('RGB', (width, height), raw_str)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                self._screenshot_data = base64.b64encode(buffer.read()).decode('utf-8')
            except ImportError:
                # Fallback: try direct pygame save to BytesIO
                try:
                    buffer = io.BytesIO()
                    # Create a temp surface copy for saving
                    temp_surface = surface.copy()
                    # Use PNG format directly without filename extension
                    pygame.image.save(temp_surface, buffer)
                    buffer.seek(0)
                    self._screenshot_data = base64.b64encode(buffer.read()).decode('utf-8')
                except Exception as fallback_error:
                    _logger.warning(f"Screenshot fallback error: {fallback_error}")
                    self._screenshot_data = None
        except Exception as e:
            _logger.warning(f"Screenshot error: {e}")
            self._screenshot_data = None  # Clear corrupted data
    
    def get_screenshot(self) -> Optional[str]:
        """Get the latest screenshot as base64."""
        return self._screenshot_data
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return {
            'state': asdict(self.state),
            'history': {
                'scores': list(self.scores),
                'losses': list(self.losses),
                'epsilons': list(self.epsilons),
                'rewards': list(self.rewards),
                'q_values': list(self.q_values),
                'episode_lengths': list(self.episode_lengths),
            }
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
        if 'learning_rate' in config:
            self.state.learning_rate = config['learning_rate']
        if 'batch_size' in config:
            self.state.batch_size = config['batch_size']
        if 'learn_every' in config:
            self.state.learn_every = config['learn_every']
        if 'gradient_steps' in config:
            self.state.gradient_steps = config['gradient_steps']
    
    def set_performance_mode(self, mode: str) -> None:
        """Set performance mode preset."""
        self.state.performance_mode = mode
    
    def set_system_info(self, device: str, torch_compiled: bool, target_episodes: int, headless: bool = False) -> None:
        """Set system information."""
        self.state.device = device
        self.state.torch_compiled = torch_compiled
        self.state.target_episodes = target_episodes
        self.state.training_start_time = time.time()
        self.state.headless = headless
    
    def record_save(
        self,
        filename: str,
        reason: str,
        episode: int,
        best_score: int
    ) -> None:
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
            'last_save_time': self.save_status.last_save_time,
            'last_save_filename': self.save_status.last_save_filename,
            'last_save_reason': self.save_status.last_save_reason,
            'last_save_episode': self.save_status.last_save_episode,
            'last_save_best_score': self.save_status.last_save_best_score,
            'saves_this_session': self.save_status.saves_this_session,
            'time_since_save': time_since_save,
            'time_since_save_str': self._format_time_ago(time_since_save)
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
        action_labels: Optional[List[str]] = None
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
        if current_time - self._last_nn_update_time < self._nn_update_interval:
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
            incoming_weights = np.asarray(incoming_weights)
            if incoming_weights.size > 0:
                data.incoming_weights = incoming_weights.tolist()
                data.incoming_weight_stats = {
                    'mean': float(np.mean(incoming_weights)),
                    'std': float(np.std(incoming_weights)),
                    'min': float(np.min(incoming_weights)),
                    'max': float(np.max(incoming_weights)),
                }

        if outgoing_weights is not None:
            outgoing_weights = np.asarray(outgoing_weights)
            if outgoing_weights.size > 0:
                data.outgoing_weights = outgoing_weights.tolist()
                data.outgoing_weight_stats = {
                    'mean': float(np.mean(outgoing_weights)),
                    'std': float(np.std(outgoing_weights)),
                    'min': float(np.min(outgoing_weights)),
                    'max': float(np.max(outgoing_weights)),
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
            return {'error': 'Neuron not found'}
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

        # Update weight statistics if provided
        if weights is not None:
            flat_weights = weights.flatten()
            data.weight_mean = float(np.mean(flat_weights))
            data.weight_std = float(np.std(flat_weights))
            data.weight_min = float(np.min(flat_weights))
            data.weight_max = float(np.max(flat_weights))

            # Weight histogram
            hist, _ = np.histogram(flat_weights, bins=20)
            data.weight_histogram = hist.tolist()

        # Update gradient statistics if provided
        if gradients is not None:
            flat_grads = gradients.flatten()
            data.gradient_mean = float(np.mean(np.abs(flat_grads)))
            data.gradient_std = float(np.std(flat_grads))
            data.gradient_max_magnitude = float(np.max(np.abs(flat_grads)))

    def get_layer_analysis(self, layer_idx: int) -> Dict[str, Any]:
        """
        Phase 2: Get analysis data for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Dictionary with layer analysis
        """
        if layer_idx not in self._layer_analysis_data:
            return {'error': 'Layer not found'}
        return self._layer_analysis_data[layer_idx].to_dict()

    def get_all_layer_analysis(self) -> List[Dict[str, Any]]:
        """
        Phase 2: Get analysis for all layers.

        Returns:
            List of layer analysis dictionaries
        """
        return [data.to_dict() for data in sorted(
            self._layer_analysis_data.values(),
            key=lambda x: x.layer_idx
        )]

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


class WebDashboard:
    """
    Flask web dashboard for training visualization.
    
    Runs a web server in a background thread that serves
    a real-time dashboard with charts, controls, and console logs.
    
    Example:
        >>> dashboard = WebDashboard(port=5000)
        >>> dashboard.start()
        >>> # Training loop...
        >>> dashboard.publisher.update(episode, score, ...)
        >>> dashboard.log("Episode complete", level="success")
        >>> dashboard.stop()
    """
    
    def __init__(self, config: Optional[Config] = None, port: int = 5000, host: str = '0.0.0.0', launcher_mode: bool = False):
        """
        Initialize the web dashboard.
        
        Args:
            config: Configuration object
            port: Port to run the server on
            host: Host address (0.0.0.0 for all interfaces)
            launcher_mode: If True, show game selection instead of training dashboard
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is not installed. Run: pip install flask flask-socketio eventlet")
        
        self.config = config or Config()
        self.port = port
        self.host = host
        self.launcher_mode = launcher_mode
        self.on_game_selected_callback: Optional[Callable[[str, str], None]] = None  # (game, mode)
        self.on_restart_with_game_callback: Optional[Callable[[str], None]] = None
        
        # Metrics publisher
        self.publisher = MetricsPublisher()
        # Set memory capacity from config
        self.publisher.state.memory_capacity = self.config.MEMORY_SIZE
        
        # Flask app setup - use absolute paths relative to this module
        base_dir = os.path.dirname(__file__)
        template_dir = os.path.join(base_dir, 'templates')
        static_dir = os.path.join(base_dir, 'static')
        
        self.app = Flask(__name__,
                        template_folder=template_dir,
                        static_folder=static_dir)
        # Generate secure random secret key (not hardcoded for security)
        self.app.config['SECRET_KEY'] = base64.b64encode(os.urandom(24)).decode('utf-8')
        
        # SocketIO setup
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Register routes
        self._register_routes()
        self._register_socket_events()
        
        # Server thread
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Control callbacks
        self.on_pause_callback: Optional[Callable[[], None]] = None
        self.on_save_callback: Optional[Callable[[], None]] = None
        self.on_save_as_callback: Optional[Callable[[str], None]] = None
        self.on_speed_callback: Optional[Callable[[float], None]] = None
        self.on_reset_callback: Optional[Callable[[], None]] = None
        self.on_start_fresh_callback: Optional[Callable[[], None]] = None
        self.on_load_model_callback: Optional[Callable[[str], None]] = None
        self.on_config_change_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_performance_mode_callback: Optional[Callable[[str], None]] = None
        self.on_save_and_quit_callback: Optional[Callable[[], None]] = None
    
    def _register_routes(self) -> None:
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            if self.launcher_mode:
                response = make_response(render_template('launcher.html'))
            else:
                response = make_response(render_template('dashboard.html'))
            # Prevent browser caching to ensure fresh content
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        @self.app.route('/api/status')
        def api_status():
            snapshot = self.publisher.get_snapshot()
            snapshot['launcher_mode'] = self.launcher_mode
            return jsonify(snapshot)
        
        @self.app.route('/api/config')
        def api_config():
            return jsonify({
                'learning_rate': self.config.LEARNING_RATE,
                'gamma': self.config.GAMMA,
                'epsilon_start': self.config.EPSILON_START,
                'epsilon_end': self.config.EPSILON_END,
                'epsilon_decay': self.config.EPSILON_DECAY,
                'batch_size': self.config.BATCH_SIZE,
                'hidden_layers': self.config.HIDDEN_LAYERS,
                'memory_size': self.config.MEMORY_SIZE,
                'target_update': self.config.TARGET_UPDATE,
                'grad_clip': self.config.GRAD_CLIP,
                # Performance settings
                'learn_every': self.config.LEARN_EVERY,
                'gradient_steps': self.config.GRADIENT_STEPS,
                'device': str(self.config.DEVICE),
                'vec_envs': self.publisher.state.num_envs,
                # Game settings
                'game_name': self.config.GAME_NAME,
            })
        
        @self.app.route('/api/games')
        def api_games():
            """List all available games with their metadata."""
            from src.game import list_games, get_game_info
            
            games = []
            for game_id in list_games():
                info = get_game_info(game_id)
                if info:
                    games.append({
                        'id': game_id,
                        'name': info.get('name', game_id.title()),
                        'description': info.get('description', ''),
                        'actions': info.get('actions', []),
                        'difficulty': info.get('difficulty', 'Unknown'),
                        'icon': info.get('icon', '🎮'),
                        'color': info.get('color', (100, 100, 100)),
                        'is_current': game_id == self.config.GAME_NAME,
                    })
            
            return jsonify({
                'games': games,
                'current_game': self.config.GAME_NAME
            })
        
        @self.app.route('/api/screenshot')
        def api_screenshot():
            # If headless mode, return early with flag (no screenshots available)
            if self.publisher.state.headless:
                return jsonify({'image': None, 'headless': True})
            screenshot = self.publisher.get_screenshot()
            if screenshot:
                return jsonify({'image': screenshot, 'headless': False})
            return jsonify({'image': None, 'headless': False})
        
        @self.app.route('/api/models')
        def api_models():
            """List available model files with metadata.
            
            Searches both game-specific directory and legacy models directory.
            """
            import os
            import torch
            from datetime import datetime
            
            # Search both game-specific and legacy directories
            search_dirs = [
                (self.config.GAME_MODEL_DIR, self.config.GAME_NAME),
                (self.config.MODEL_DIR, 'legacy')
            ]
            
            models = []
            seen_paths = set()  # Avoid duplicates
            
            for model_dir, source in search_dirs:
                if not os.path.exists(model_dir):
                    continue
                    
                for f in os.listdir(model_dir):
                    if f.endswith('.pth'):
                        path = os.path.join(model_dir, f)
                        
                        # Skip if we've already seen this file
                        if path in seen_paths:
                            continue
                        seen_paths.add(path)
                        
                        model_info = {
                            'name': f,
                            'path': path,
                            'source': source,  # Which directory it came from
                            'size': os.path.getsize(path),
                            'modified': os.path.getmtime(path),
                            'modified_str': datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M'),
                            'has_metadata': False,
                            'metadata': None
                        }
                        
                        # Try to extract metadata
                        try:
                            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                            model_info['steps'] = checkpoint.get('steps', None)
                            model_info['epsilon'] = checkpoint.get('epsilon', None)
                            if 'metadata' in checkpoint:
                                model_info['has_metadata'] = True
                                model_info['metadata'] = checkpoint['metadata']
                        except Exception as e:
                            _logger.debug(f"Could not load metadata from {f}: {e}")

                        models.append(model_info)
            
            models.sort(key=lambda x: x['modified'], reverse=True)
            return jsonify({'models': models, 'current_game': self.config.GAME_NAME})

        @self.app.route('/api/models/<path:filepath>', methods=['DELETE'])
        def api_delete_model(filepath):
            """Delete a model file.
            
            Security: Validates that the path is within the model directory
            to prevent path traversal attacks.
            """
            import os
            
            # Security: ensure path is within model directory
            # Check both game-specific and legacy model directories
            game_model_dir = os.path.realpath(self.config.GAME_MODEL_DIR)
            legacy_model_dir = os.path.realpath(self.config.MODEL_DIR)
            # Join filepath with model_dir first to prevent absolute path injection
            full_path = os.path.realpath(os.path.join(legacy_model_dir, filepath))
            
            # Check if path is within either allowed directory
            is_valid = False
            try:
                # Check game-specific directory
                common_game = os.path.commonpath([game_model_dir, full_path])
                if common_game == game_model_dir:
                    is_valid = True
            except ValueError:
                # Expected when paths are on different drives (Windows) or incompatible
                _logger.debug(f"Path {full_path} not in game model dir (different root)")

            try:
                # Check legacy directory
                common_legacy = os.path.commonpath([legacy_model_dir, full_path])
                if common_legacy == legacy_model_dir:
                    is_valid = True
            except ValueError:
                # Expected when paths are on different drives (Windows) or incompatible
                _logger.debug(f"Path {full_path} not in legacy model dir (different root)")
            
            if not is_valid:
                return jsonify({'error': 'Invalid path - model must be in model directory'}), 403
            
            if not os.path.exists(full_path):
                return jsonify({'error': 'Model not found'}), 404
            
            # Security: Reject symlinks in ANY path component to prevent symlink-based attacks
            # Even though realpath resolves them and the containment check catches external targets,
            # it's safer to explicitly reject symlinks in all path components
            path_to_check = os.path.join(legacy_model_dir, filepath)
            current_path = path_to_check
            while current_path and current_path != legacy_model_dir:
                if os.path.islink(current_path):
                    return jsonify({'error': 'Cannot delete files with symbolic links in path'}), 403
                current_path = os.path.dirname(current_path)
            
            # Ensure it's a .pth file
            if not full_path.endswith('.pth'):
                return jsonify({'error': 'Invalid file type'}), 400
            
            try:
                filename = os.path.basename(full_path)
                os.remove(full_path)
                self.publisher.log(f"🗑️ Deleted model: {filename}", level="action")
                return jsonify({
                    'success': True,
                    'message': f'Model {filename} deleted successfully',
                    'filename': filename
                })
            except Exception as e:
                return jsonify({'error': f'Failed to delete model: {str(e)}'}), 500
        
        @self.app.route('/api/save-status')
        def api_save_status():
            """Get last save information."""
            return jsonify(self.publisher.get_save_status())
        
        @self.app.route('/api/game-stats')
        def api_game_stats():
            """Get training statistics for all games (for comparison panel)."""
            import os
            import torch
            from src.game import list_games, get_game_info
            
            stats = {}
            for game_id in list_games():
                game_info = get_game_info(game_id)
                game_model_dir = os.path.join(self.config.MODEL_DIR, game_id)
                
                game_stats = {
                    'name': game_info.get('name', game_id.title()) if game_info else game_id.title(),
                    'icon': game_info.get('icon', '🎮') if game_info else '🎮',
                    'color': game_info.get('color', (100, 100, 100)) if game_info else (100, 100, 100),
                    'best_score': 0,
                    'total_episodes': 0,
                    'total_training_time': 0,
                    'model_count': 0,
                    'best_model': None,
                }
                
                # Scan models for this game
                if os.path.exists(game_model_dir):
                    for f in os.listdir(game_model_dir):
                        if f.endswith('.pth'):
                            game_stats['model_count'] += 1
                            path = os.path.join(game_model_dir, f)
                            try:
                                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                                if 'metadata' in checkpoint:
                                    meta = checkpoint['metadata']
                                    if meta.get('best_score', 0) > game_stats['best_score']:
                                        game_stats['best_score'] = meta['best_score']
                                        game_stats['best_model'] = f
                                    if meta.get('episode', 0) > game_stats['total_episodes']:
                                        game_stats['total_episodes'] = meta['episode']
                                    game_stats['total_training_time'] += meta.get('total_training_time_seconds', 0)
                            except Exception as e:
                                _logger.debug(f"Could not load stats from {f}: {e}")
                
                stats[game_id] = game_stats
            
            return jsonify({
                'stats': stats,
                'current_game': self.config.GAME_NAME
            })

        # ===== Phase 2: Neuron Inspection & Layer Analysis Endpoints =====

        @self.app.route('/api/neuron/<int:layer_idx>/<int:neuron_idx>')
        def api_neuron_details(layer_idx, neuron_idx):
            """Phase 2: Get details for a specific neuron."""
            details = self.publisher.get_neuron_details(layer_idx, neuron_idx)
            return jsonify(_make_json_safe(details))

        @self.app.route('/api/layer/<int:layer_idx>')
        def api_layer_analysis(layer_idx):
            """Phase 2: Get analysis data for a specific layer."""
            analysis = self.publisher.get_layer_analysis(layer_idx)
            return jsonify(_make_json_safe(analysis))

    def _register_socket_events(self) -> None:
        """Register SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            # Send current state on connect (convert NumPy types for JSON serialization)
            emit('state_update', _make_json_safe(self.publisher.get_snapshot()))
            # Send recent logs
            emit('console_logs', {'logs': _make_json_safe(self.publisher.get_console_logs(100))})
            # Send current NN visualization data if available
            nn_data = self.publisher.get_nn_visualization()
            if nn_data.get('layer_info'):
                emit('nn_update', _make_json_safe(nn_data))
        
        @self.socketio.on('control')
        def handle_control(data):
            action = data.get('action')
            
            if action == 'pause':
                if self.on_pause_callback:
                    self.on_pause_callback()
            elif action == 'save':
                if self.on_save_callback:
                    self.on_save_callback()
            elif action == 'save_as':
                filename = data.get('filename', 'custom_save.pth')
                if self.on_save_as_callback:
                    self.on_save_as_callback(filename)
            elif action == 'speed':
                speed = data.get('value', 1.0)
                if self.on_speed_callback:
                    self.on_speed_callback(speed)
                self.publisher.set_speed(speed)
            elif action == 'reset':
                if self.on_reset_callback:
                    self.on_reset_callback()
            elif action == 'start_fresh':
                if self.on_start_fresh_callback:
                    self.on_start_fresh_callback()
                # Notify frontend to clear charts and reset UI
                emit('training_reset', {'message': 'Training reset - starting fresh'})
            elif action == 'load_model':
                model_path = data.get('path')
                if model_path and self.on_load_model_callback:
                    self.on_load_model_callback(model_path)
            elif action == 'config_change':
                config_data = data.get('config', {})
                if self.on_config_change_callback:
                    self.on_config_change_callback(config_data)
                self.publisher.update_config(config_data)
            elif action == 'performance_mode':
                mode = data.get('mode', 'normal')
                if self.on_performance_mode_callback:
                    self.on_performance_mode_callback(mode)
                self.publisher.set_performance_mode(mode)
            elif action == 'save_and_quit':
                if self.on_save_and_quit_callback:
                    self.on_save_and_quit_callback()
            elif action == 'select_game':
                # Launcher mode: user selected a game to start
                game_name = data.get('game')
                mode = data.get('mode', 'ai')  # 'ai' or 'human'
                if game_name and self.on_game_selected_callback:
                    mode_text = 'Playing' if mode == 'human' else 'Training'
                    emit('game_starting', {
                        'game': game_name,
                        'mode': mode,
                        'message': f'{mode_text} {game_name}...'
                    })
                    self.on_game_selected_callback(game_name, mode)
            elif action == 'restart_with_game':
                # Training mode: restart with a different game
                game_name = data.get('game')
                if game_name and self.on_restart_with_game_callback:
                    self.publisher.log(f"🔄 Switching to {game_name}...", level="warning")
                    # Save first
                    if self.on_save_callback:
                        self.on_save_callback()
                    emit('restarting', {
                        'game': game_name,
                        'message': f'Restarting with {game_name}...'
                    })
                    # Trigger restart (will replace process)
                    self.on_restart_with_game_callback(game_name)
            elif action == 'go_to_launcher':
                # Return to launcher mode for game/mode selection
                self.publisher.log("🎮 Returning to launcher...", level="warning")
                # Save current progress
                if self.on_save_callback:
                    self.on_save_callback()
                # Switch back to launcher mode
                self.launcher_mode = True
                # Tell browser to redirect to launcher
                emit('redirect_to_launcher', {
                    'message': 'Returning to game launcher...'
                })
                # Trigger shutdown callback if set
                if self.on_save_and_quit_callback:
                    self.on_save_and_quit_callback()

        @self.socketio.on('clear_logs')
        def handle_clear_logs():
            self.publisher.console_logs.clear()
            emit('console_logs', {'logs': []})
        
        # Auto-emit on metric updates
        # Bug 65 fix: Add null/running checks to prevent AttributeError during shutdown
        def broadcast_update(snapshot):
            if self.socketio and self._running:
                self.socketio.emit('state_update', snapshot)

        def broadcast_log(log_entry: LogMessage):
            if self.socketio and self._running:
                self.socketio.emit('console_log', log_entry.to_dict())

        def broadcast_save(save_info: Dict[str, Any]):
            if self.socketio and self._running:
                self.socketio.emit('save_event', save_info)

        def broadcast_nn_update(nn_data: Dict[str, Any]):
            if self.socketio and self._running:
                self.socketio.emit('nn_update', nn_data)
        
        # Clear and register callbacks atomically (prevents race condition)
        with self.publisher._callback_lock:
            self.publisher._on_update_callbacks.clear()
            self.publisher._on_log_callbacks.clear()
            self.publisher._on_save_callbacks.clear()
            self.publisher._on_nn_update_callbacks.clear()
            # Register new callbacks inside the lock to prevent race condition
            self.publisher._on_update_callbacks.append(broadcast_update)
            self.publisher._on_log_callbacks.append(broadcast_log)
            self.publisher._on_save_callbacks.append(broadcast_save)
            self.publisher._on_nn_update_callbacks.append(broadcast_nn_update)
    
    def start(self) -> None:
        """Start the web server in a background thread."""
        if self._running:
            return
        
        self._running = True
        self.publisher.set_running(True)

        # Suppress Flask/werkzeug logging COMPLETELY - do this BEFORE starting server
        import logging
        import sys
        
        # Disable werkzeug request logging completely
        werkzeug_log = logging.getLogger('werkzeug')
        werkzeug_log.setLevel(logging.ERROR)
        werkzeug_log.disabled = True
        
        # Also suppress Flask's internal logger
        flask_log = logging.getLogger('flask.app')
        flask_log.setLevel(logging.ERROR)
        
        # Suppress socketio and engineio loggers
        logging.getLogger('engineio').setLevel(logging.ERROR)
        logging.getLogger('socketio').setLevel(logging.ERROR)
        logging.getLogger('engineio.server').setLevel(logging.ERROR)
        logging.getLogger('socketio.server').setLevel(logging.ERROR)
        
        # Redirect werkzeug's output stream to devnull
        # This catches cases where werkzeug bypasses the logging system
        class NullWriter:
            def write(self, *args, **kwargs):
                pass
            def flush(self, *args, **kwargs):
                pass
        
        # Disable Flask's click CLI echo (used by werkzeug for request logging)
        try:
            import click
            click.echo = lambda *args, **kwargs: None
        except ImportError:
            _logger.debug("click module not available, skipping echo suppression")

        def run_server():
            _logger.info(f"Web Dashboard running at http://localhost:{self.port}")

            try:
                self.socketio.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    log_output=False,
                    allow_unsafe_werkzeug=True
                )
            # Bug 76 fix: Catch broader exceptions to prevent silent thread crashes
            except (OSError, RuntimeError, ConnectionError, Exception) as e:
                _logger.error(f"Failed to start web dashboard on port {self.port}: {type(e).__name__}: {e}")
                _logger.error(f"Port {self.port} may already be in use. Try a different port with --port")
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
    
    def stop(self) -> None:
        """Stop the web server and release the port."""
        self._running = False
        self.publisher.set_running(False)
        
        # Try to stop the SocketIO server (releases the port)
        try:
            if hasattr(self.socketio, 'stop'):
                self.socketio.stop()
        except Exception as e:
            # Best effort - daemon thread will die with process anyway
            _logger.debug(f"Server stop (best effort): {e}")
    
    def emit_metrics(self, **kwargs) -> None:
        """Convenience method to update and emit metrics."""
        self.publisher.update(**kwargs)
    
    def log(
        self,
        message: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a log message to the console.
        
        Args:
            message: Log message text
            level: One of 'debug', 'info', 'success', 'warning', 'error', 'metric', 'action'
            data: Optional dictionary of additional data
        """
        self.publisher.log(message, level, data)
    
    def capture_screenshot(self, surface) -> None:
        """Capture and store a game screenshot."""
        self.publisher.set_screenshot(surface)
    
    def emit_nn_visualization(
        self,
        layer_info: List[Dict[str, Any]],
        activations: Dict[str, List[float]],
        q_values: List[float],
        selected_action: int,
        weights: List[List[List[float]]],
        step: int,
        action_labels: Optional[List[str]] = None
    ) -> None:
        """
        Emit neural network visualization data to connected clients.
        
        This is a convenience method that updates the publisher's NN data.
        The update is throttled to ~10 FPS internally.
        
        Args:
            layer_info: List of layer info dicts with 'name', 'neurons', 'type'
            activations: Dict mapping layer keys to lists of activation values
            q_values: Q-values for each action
            selected_action: Currently selected action index
            weights: Sampled weight matrices as nested lists
            step: Current training step
            action_labels: Labels for each action (e.g., ["LEFT", "STAY", "RIGHT"])
        """
        self.publisher.update_nn_visualization(
            layer_info=layer_info,
            activations=activations,
            q_values=q_values,
            selected_action=selected_action,
            weights=weights,
            step=step,
            action_labels=action_labels
        )


# Create templates directory structure
def create_web_templates():
    """Helper to create web template files."""
    import os
    
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    return template_dir, static_dir


if __name__ == "__main__":
    print("Web Dashboard Server")
    print("Import and use with training loop")
