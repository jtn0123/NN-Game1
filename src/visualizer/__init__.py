"""
Visualizer Module
=================

Real-time visualization of neural network training.

Classes:
    NeuralNetVisualizer - Draws the neural network structure and activations
    Dashboard           - Training metrics and statistics display
    TrainingHUD         - On-screen training statistics overlay
    PauseMenu           - Interactive pause menu
"""

from .nn_visualizer import NeuralNetVisualizer
from .dashboard import Dashboard
from .hud import TrainingHUD
from .pause_menu import PauseMenu

__all__ = ['NeuralNetVisualizer', 'Dashboard', 'TrainingHUD', 'PauseMenu']

