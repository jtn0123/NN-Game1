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
from typing import Optional, Dict, Any, List, Deque, Callable
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
    batch_size: int = 64
    memory_size: int = 0
    memory_capacity: int = 100000
    episodes_per_second: float = 0.0
    exploration_actions: int = 0
    exploitation_actions: int = 0
    target_updates: int = 0
    avg_q_value: float = 0.0
    bricks_broken_total: int = 0
    total_reward: float = 0.0


@dataclass 
class TrainingConfig:
    """Configurable training parameters."""
    learning_rate: float = 0.0001
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 64
    gamma: float = 0.99
    target_update_freq: int = 1000


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
        
        # Screenshot storage
        self._screenshot_data: Optional[str] = None
        
        # Timing for episodes/second calculation
        self._episode_times: Deque[float] = deque(maxlen=10)
        self._last_episode_time: float = time.time()
    
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
    
    def __init__(self, config: Optional[Config] = None, port: int = 5000, host: str = '0.0.0.0'):
        """
        Initialize the web dashboard.
        
        Args:
            config: Configuration object
            port: Port to run the server on
            host: Host address (0.0.0.0 for all interfaces)
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask is not installed. Run: pip install flask flask-socketio eventlet")
        
        self.config = config or Config()
        self.port = port
        self.host = host
        
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
        self.on_speed_callback: Optional[Callable[[float], None]] = None
        self.on_reset_callback: Optional[Callable[[], None]] = None
        self.on_load_model_callback: Optional[Callable[[str], None]] = None
        self.on_config_change_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def _register_routes(self) -> None:
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.publisher.get_snapshot())
        
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
            })
        
        @self.app.route('/api/screenshot')
        def api_screenshot():
            screenshot = self.publisher.get_screenshot()
            if screenshot:
                return jsonify({'image': screenshot})
            return jsonify({'image': None})
        
        @self.app.route('/api/logs')
        def api_logs():
            limit = request.args.get('limit', 100, type=int)
            return jsonify({'logs': self.publisher.get_console_logs(limit)})
        
        @self.app.route('/api/models')
        def api_models():
            """List available model files."""
            import os
            model_dir = self.config.MODEL_DIR
            models = []
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.endswith('.pth'):
                        path = os.path.join(model_dir, f)
                        models.append({
                            'name': f,
                            'path': path,
                            'size': os.path.getsize(path),
                            'modified': os.path.getmtime(path)
                        })
            models.sort(key=lambda x: x['modified'], reverse=True)
            return jsonify({'models': models})
    
    def _register_socket_events(self) -> None:
        """Register SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            # Send current state on connect
            emit('state_update', self.publisher.get_snapshot())
            # Send recent logs
            emit('console_logs', {'logs': self.publisher.get_console_logs(100)})
        
        @self.socketio.on('control')
        def handle_control(data):
            action = data.get('action')
            
            if action == 'pause':
                if self.on_pause_callback:
                    self.on_pause_callback()
            elif action == 'save':
                if self.on_save_callback:
                    self.on_save_callback()
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
        
        @self.socketio.on('clear_logs')
        def handle_clear_logs():
            self.publisher.console_logs.clear()
            emit('console_logs', {'logs': []})
        
        # Auto-emit on metric updates
        def broadcast_update(snapshot):
            self.socketio.emit('state_update', snapshot)
        
        def broadcast_log(log_entry: LogMessage):
            self.socketio.emit('console_log', log_entry.to_dict())
        
        self.publisher.on_update(broadcast_update)
        self.publisher.on_log(broadcast_log)
    
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
