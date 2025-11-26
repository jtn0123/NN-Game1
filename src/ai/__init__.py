"""
AI Module
=========

Deep Reinforcement Learning components for game playing.

Classes:
    DQN          - Deep Q-Network neural network architecture
    Agent        - DQN agent with epsilon-greedy exploration
    ReplayBuffer - Experience replay memory
    Trainer      - Training loop orchestration
"""

from .network import DQN
from .agent import Agent
from .replay_buffer import ReplayBuffer
from .trainer import Trainer

__all__ = ['DQN', 'Agent', 'ReplayBuffer', 'Trainer']

