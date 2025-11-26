"""
Visualizer Module
=================

Real-time visualization of neural network training.

Classes:
    NeuralNetVisualizer - Draws the neural network structure and activations
    Dashboard           - Training metrics and statistics display
"""

from .nn_visualizer import NeuralNetVisualizer
from .dashboard import Dashboard

__all__ = ['NeuralNetVisualizer', 'Dashboard']

