"""
Base Game Interface
===================

Abstract base class that defines the interface all games must implement.
This allows the AI agent to work with any game that follows this interface.

To add a new game:
1. Create a new file in src/game/
2. Inherit from BaseGame
3. Implement all abstract methods
4. Register in __init__.py
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseGame(ABC):
    """
    Abstract base class for games.
    
    Any game that the AI can learn to play must implement this interface.
    This ensures consistency and allows easy swapping of games.
    
    Properties:
        state_size: int - Dimension of the state vector
        action_size: int - Number of possible actions
        
    Methods:
        reset() -> np.ndarray
            Reset game to initial state, return state vector
            
        step(action: int) -> Tuple[np.ndarray, float, bool, dict]
            Execute action, return (next_state, reward, done, info)
            
        render(screen) -> None
            Draw game to pygame screen
            
        get_state() -> np.ndarray
            Get current state vector
    """
    
    @property
    @abstractmethod
    def state_size(self) -> int:
        """Return the dimension of the state vector."""
        pass
    
    @property
    @abstractmethod
    def action_size(self) -> int:
        """Return the number of possible actions."""
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            np.ndarray: Initial state vector
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one game step with the given action.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            Tuple containing:
                - next_state (np.ndarray): State after action
                - reward (float): Reward received
                - done (bool): True if game is over
                - info (dict): Additional information (score, lives, etc.)
        """
        pass
    
    @abstractmethod
    def render(self, screen) -> None:
        """
        Render the current game state to a pygame screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Get the current state as a normalized vector.
        
        Returns:
            np.ndarray: Current state vector (values typically in [0, 1])
        """
        pass
    
    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility. Override if game has randomness."""
        pass

