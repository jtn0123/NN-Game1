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

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard unavailable.")
    print("Install with: pip install flask flask-socketio eventlet")

import sys
sys.path.append('../..')
from config import Config


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
    # Performance mode: 'normal', 'fast', 'turbo'
    performance_mode: str = "normal"


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_info': self.layer_info,
            'activations': self.activations,
            'q_values': self.q_values,
            'selected_action': self.selected_action,
            'weights': self.weights,
            'step': self.step,
            'action_labels': self.action_labels
        }


class MetricsPublisher:
    """
    Collects and publishes training metrics.
    
    This class acts as a bridge between the training loop
    and the web dashboard, storing metrics and providing
    them to connected clients.
    """
    
    def __init__(self, history_length: int = 500):
        self.history_length = history_length
        self.state = TrainingState()
        self.save_status = SaveStatus()
        
        # Metric history
        self.scores: Deque[int] = deque(maxlen=history_length)
        self.losses: Deque[float] = deque(maxlen=history_length)
        self.epsilons: Deque[float] = deque(maxlen=history_length)
        self.rewards: Deque[float] = deque(maxlen=history_length)
        self.q_values: Deque[float] = deque(maxlen=history_length)
        self.episode_lengths: Deque[int] = deque(maxlen=history_length)
        
        # Console log history
        self.console_logs: Deque[LogMessage] = deque(maxlen=500)
        
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
        self._nn_update_interval: float = 0.1  # 10 FPS throttle
    
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
        episode_length: int = 0
    ) -> None:
        """Update metrics with new episode data."""
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
        
        self.scores.append(score)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        self.q_values.append(avg_q_value)
        self.episode_lengths.append(episode_length)
        
        # Calculate episodes per second
        current_time = time.time()
        self._episode_times.append(current_time - self._last_episode_time)
        self._last_episode_time = current_time
        if self._episode_times:
            avg_time = sum(self._episode_times) / len(self._episode_times)
            self.state.episodes_per_second = 1.0 / avg_time if avg_time > 0 else 0.0
        
        # Calculate steps per second using time-windowed approach
        # This gives accurate real-time rate instead of lifetime average
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
            
            if time_delta > 0.1:  # Need at least 100ms of data
                self._last_steps_per_sec = step_delta / time_delta
                self.state.steps_per_second = self._last_steps_per_sec
            else:
                self.state.steps_per_second = self._last_steps_per_sec
        else:
            self.state.steps_per_second = self._last_steps_per_sec
        
        # Calculate win rate
        if len(self.scores) > 0:
            # Assume score > 300 is a win (configurable)
            recent = list(self.scores)[-100:]
            self.state.win_rate = sum(1 for s in recent if s >= 300) / len(recent)
        
        # Notify callbacks
        for callback in self._on_update_callbacks:
            callback(self.get_snapshot())
    
    def log(
        self,
        message: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a log message to the console."""
        log_entry = LogMessage(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level,
            message=message,
            data=data
        )
        self.console_logs.append(log_entry)
        
        # Notify log callbacks
        for callback in self._on_log_callbacks:
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
                buffer = io.BytesIO()
                # Create a temp surface copy for saving
                temp_surface = surface.copy()
                pygame.image.save(temp_surface, buffer, 'screenshot.png')
                buffer.seek(0)
                self._screenshot_data = base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Screenshot error: {e}")
    
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
        self._on_update_callbacks.append(callback)
    
    def on_log(self, callback) -> None:
        """Register a callback for log messages."""
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
    
    def set_system_info(self, device: str, torch_compiled: bool, target_episodes: int) -> None:
        """Set system information."""
        self.state.device = device
        self.state.torch_compiled = torch_compiled
        self.state.target_episodes = target_episodes
        self.state.training_start_time = time.time()
    
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
        
        # Notify save callbacks
        save_info = self.get_save_status()
        for callback in self._on_save_callbacks:
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
        
        # Notify callbacks
        nn_dict = self._nn_data.to_dict()
        for callback in self._on_nn_update_callbacks:
            callback(nn_dict)
    
    def get_nn_visualization(self) -> Dict[str, Any]:
        """Get current neural network visualization data."""
        return self._nn_data.to_dict()
    
    def on_nn_update(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for neural network visualization updates."""
        self._on_nn_update_callbacks.append(callback)


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
        self.on_game_selected_callback: Optional[Callable[[str], None]] = None
        self.on_restart_with_game_callback: Optional[Callable[[str], None]] = None
        
        # Metrics publisher
        self.publisher = MetricsPublisher()
        
        # Flask app setup - use absolute paths relative to this module
        base_dir = os.path.dirname(__file__)
        template_dir = os.path.join(base_dir, 'templates')
        static_dir = os.path.join(base_dir, 'static')
        
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=static_dir)
        self.app.config['SECRET_KEY'] = 'neural-network-game-ai'
        
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
        self.on_load_model_callback: Optional[Callable[[str], None]] = None
        self.on_config_change_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_performance_mode_callback: Optional[Callable[[str], None]] = None
        self.on_switch_game_callback: Optional[Callable[[str], None]] = None
    
    def _register_routes(self) -> None:
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            if self.launcher_mode:
                return render_template('launcher.html')
            return render_template('dashboard.html')
        
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
            screenshot = self.publisher.get_screenshot()
            if screenshot:
                return jsonify({'image': screenshot})
            return jsonify({'image': None})
        
        @self.app.route('/api/nn-visualization')
        def api_nn_visualization():
            """Get current neural network visualization data."""
            return jsonify(self.publisher.get_nn_visualization())
        
        @self.app.route('/api/logs')
        def api_logs():
            limit = request.args.get('limit', 100, type=int)
            return jsonify({'logs': self.publisher.get_console_logs(limit)})
        
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
                        except Exception:
                            pass
                        
                        models.append(model_info)
            
            models.sort(key=lambda x: x['modified'], reverse=True)
            return jsonify({'models': models, 'current_game': self.config.GAME_NAME})
        
        @self.app.route('/api/model-info/<path:filepath>')
        def api_model_info(filepath):
            """Get detailed info about a specific model."""
            import torch
            from datetime import datetime
            
            # Security: ensure path is within model directory
            # Use realpath to resolve symlinks, preventing symlink attacks
            model_dir = os.path.realpath(self.config.MODEL_DIR)
            full_path = os.path.realpath(filepath)
            # Use os.path.commonpath to properly check containment
            # This prevents path traversal attacks like /models_evil/file.pth
            try:
                common = os.path.commonpath([model_dir, full_path])
                if common != model_dir:
                    return jsonify({'error': 'Invalid path'}), 403
            except ValueError:
                # commonpath raises ValueError if paths are on different drives (Windows)
                return jsonify({'error': 'Invalid path'}), 403
            
            if not os.path.exists(full_path):
                return jsonify({'error': 'Model not found'}), 404
            
            try:
                checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
                file_stat = os.stat(full_path)
                
                info = {
                    'filename': os.path.basename(full_path),
                    'path': full_path,
                    'size': file_stat.st_size,
                    'size_mb': file_stat.st_size / (1024 * 1024),
                    'modified': file_stat.st_mtime,
                    'modified_str': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'steps': checkpoint.get('steps', 'unknown'),
                    'epsilon': checkpoint.get('epsilon', 'unknown'),
                    'state_size': checkpoint.get('state_size', 'unknown'),
                    'action_size': checkpoint.get('action_size', 'unknown'),
                    'has_metadata': 'metadata' in checkpoint,
                    'metadata': checkpoint.get('metadata', None)
                }
                return jsonify(info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
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
                            except Exception:
                                pass
                
                stats[game_id] = game_stats
            
            return jsonify({
                'stats': stats,
                'current_game': self.config.GAME_NAME
            })
    
    def _register_socket_events(self) -> None:
        """Register SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            # Send current state on connect
            emit('state_update', self.publisher.get_snapshot())
            # Send recent logs
            emit('console_logs', {'logs': self.publisher.get_console_logs(100)})
            # Send current NN visualization data if available
            nn_data = self.publisher.get_nn_visualization()
            if nn_data.get('layer_info'):
                emit('nn_update', nn_data)
        
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
            elif action == 'switch_game':
                game_name = data.get('game')
                if game_name and self.on_switch_game_callback:
                    self.on_switch_game_callback(game_name)
                    # Notify all clients about the game switch
                    self.socketio.emit('game_switched', {
                        'game': game_name,
                        'message': f'Switching to {game_name}...'
                    })
            elif action == 'stop_for_game_switch':
                game_name = data.get('game')
                # Log the switch request
                self.publisher.log(
                    f"🔄 Game switch requested: {game_name}",
                    level="warning"
                )
                self.publisher.log(
                    f"💾 Saving current progress before stopping...",
                    level="info"
                )
                # Trigger save then stop
                if self.on_save_callback:
                    self.on_save_callback()
                # Notify client with next command
                self.socketio.emit('stop_for_switch', {
                    'game': game_name,
                    'command': f'python main.py --game {game_name} --headless --turbo --web'
                })
            elif action == 'select_game':
                # Launcher mode: user selected a game to start
                game_name = data.get('game')
                if game_name and self.on_game_selected_callback:
                    emit('game_starting', {
                        'game': game_name,
                        'message': f'Starting {game_name}...'
                    })
                    self.on_game_selected_callback(game_name)
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
        
        @self.socketio.on('clear_logs')
        def handle_clear_logs():
            self.publisher.console_logs.clear()
            emit('console_logs', {'logs': []})
        
        # Auto-emit on metric updates
        def broadcast_update(snapshot):
            self.socketio.emit('state_update', snapshot)
        
        def broadcast_log(log_entry: LogMessage):
            self.socketio.emit('console_log', log_entry.to_dict())
        
        def broadcast_save(save_info: Dict[str, Any]):
            self.socketio.emit('save_event', save_info)
        
        def broadcast_nn_update(nn_data: Dict[str, Any]):
            self.socketio.emit('nn_update', nn_data)
        
        self.publisher.on_update(broadcast_update)
        self.publisher.on_log(broadcast_log)
        self.publisher.on_save(broadcast_save)
        self.publisher.on_nn_update(broadcast_nn_update)
    
    def start(self) -> None:
        """Start the web server in a background thread."""
        if self._running:
            return
        
        self._running = True
        self.publisher.set_running(True)
        
        def run_server():
            # Suppress Flask logging
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            print(f"\n🌐 Web Dashboard running at http://localhost:{self.port}")
            print("   Open in browser to view training progress\n")
            
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                log_output=False
            )
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
    
    def stop(self) -> None:
        """Stop the web server."""
        self._running = False
        self.publisher.set_running(False)
    
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
