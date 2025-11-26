"""
Web Dashboard Server
====================

Flask + SocketIO server for real-time training visualization.

Features:
    - REST API for model info and current stats
    - WebSocket events for live metrics streaming
    - Runs in background thread alongside training
    - Training controls (pause, save, adjust speed)

Usage:
    >>> from src.web import WebDashboard
    >>> dashboard = WebDashboard(port=5000)
    >>> dashboard.start()
    >>> # ... during training ...
    >>> dashboard.emit_metrics(episode, score, loss, ...)
    >>> dashboard.stop()
"""

import threading
import time
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from collections import deque
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
        self.scores = deque(maxlen=history_length)
        self.losses = deque(maxlen=history_length)
        self.epsilons = deque(maxlen=history_length)
        self.rewards = deque(maxlen=history_length)
        
        # Callbacks
        self._on_update_callbacks = []
        
        # Screenshot storage
        self._screenshot_data: Optional[str] = None
    
    def update(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: float,
        total_steps: int = 0,
        won: bool = False,
        reward: float = 0.0
    ) -> None:
        """Update metrics with new episode data."""
        self.state.episode = episode
        self.state.score = score
        self.state.best_score = max(self.state.best_score, score)
        self.state.epsilon = epsilon
        self.state.loss = loss
        self.state.total_steps = total_steps
        
        self.scores.append(score)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.rewards.append(reward)
        
        # Calculate win rate
        if len(self.scores) > 0:
            # Assume score > 300 is a win (configurable)
            recent = list(self.scores)[-100:]
            self.state.win_rate = sum(1 for s in recent if s >= 300) / len(recent)
        
        # Notify callbacks
        for callback in self._on_update_callbacks:
            callback(self.get_snapshot())
    
    def set_screenshot(self, surface) -> None:
        """Store a screenshot from pygame surface."""
        try:
            import pygame
            # Convert pygame surface to PNG bytes
            buffer = io.BytesIO()
            pygame.image.save(surface, buffer, 'PNG')
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
            }
        }
    
    def on_update(self, callback) -> None:
        """Register a callback for metric updates."""
        self._on_update_callbacks.append(callback)
    
    def set_paused(self, paused: bool) -> None:
        """Set training paused state."""
        self.state.is_paused = paused
    
    def set_running(self, running: bool) -> None:
        """Set training running state."""
        self.state.is_running = running


class WebDashboard:
    """
    Flask web dashboard for training visualization.
    
    Runs a web server in a background thread that serves
    a real-time dashboard with charts and controls.
    
    Example:
        >>> dashboard = WebDashboard(port=5000)
        >>> dashboard.start()
        >>> # Training loop...
        >>> dashboard.publisher.update(episode, score, ...)
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
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
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
        self.on_pause_callback = None
        self.on_save_callback = None
        self.on_speed_callback = None
    
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
                'batch_size': self.config.BATCH_SIZE,
                'hidden_layers': self.config.HIDDEN_LAYERS,
            })
        
        @self.app.route('/api/screenshot')
        def api_screenshot():
            screenshot = self.publisher.get_screenshot()
            if screenshot:
                return jsonify({'image': screenshot})
            return jsonify({'image': None})
    
    def _register_socket_events(self) -> None:
        """Register SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            # Send current state on connect
            emit('state_update', self.publisher.get_snapshot())
        
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
        
        # Auto-emit on metric updates
        def broadcast_update(snapshot):
            self.socketio.emit('state_update', snapshot)
        
        self.publisher.on_update(broadcast_update)
    
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
        # Note: Flask-SocketIO doesn't have a clean shutdown, 
        # but daemon thread will exit when main process ends
    
    def emit_metrics(self, **kwargs) -> None:
        """Convenience method to update and emit metrics."""
        self.publisher.update(**kwargs)
    
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
