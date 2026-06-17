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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Protocol, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from config import Config


def validate_action(action: int, action_size: int, game_name: str = "game") -> int:
    """Return a valid integer action or raise a clear error."""
    try:
        action_int = int(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{game_name} action must be an integer") from exc

    if action_int < 0 or action_int >= action_size:
        raise ValueError(f"{game_name} action {action_int} outside valid range 0-{action_size - 1}")
    return action_int


def validate_action_batch(
    actions: np.ndarray,
    expected_size: int,
    action_size: int,
    game_name: str = "vectorized game",
) -> np.ndarray:
    """Return a 1D action array with exactly one valid action per environment."""
    action_array = np.asarray(actions)
    if action_array.ndim != 1:
        raise ValueError(f"{game_name} actions must be a 1D array")
    if len(action_array) != expected_size:
        raise ValueError(f"{game_name} expected {expected_size} actions, got {len(action_array)}")
    for action in action_array:
        validate_action(int(action), action_size, game_name)
    return action_array


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


class BaseVecGame(Protocol):
    """Protocol for vectorized game environments used by headless training."""

    envs: Sequence[BaseGame]

    def reset(self) -> np.ndarray:
        """Reset every environment and return batched states."""
        ...

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step every environment with one action per environment."""
        ...

    def step_no_copy(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step every environment and return reusable internal buffers when supported."""
        ...

    def close(self) -> None:
        """Clean up every environment."""
        ...

    def seed(self, seeds: List[int]) -> None:
        """Seed every environment."""
        ...


class GameConstructor(Protocol):
    """Constructor contract for registered single-game environments."""

    def __call__(self, config: Config | None = None, headless: bool = False) -> BaseGame:
        """Create a single game environment."""
        ...


class VecGameConstructor(Protocol):
    """Constructor contract for registered vectorized game environments."""

    def __call__(self, num_envs: int, config: Config, headless: bool = True) -> BaseVecGame:
        """Create a vectorized game environment."""
        ...


class HumanActionProvider(Protocol):
    """Optional game capability for keyboard-to-action conversion."""

    def get_human_action(self, keys: Dict[int, bool]) -> int:
        """Return the game action represented by the current key state."""
        ...


class HumanStepProvider(Protocol):
    """Optional game capability for simultaneous-key human stepping."""

    def step_human(self, keys: Dict[int, bool]) -> Tuple[np.ndarray, float, bool, dict]:
        """Step the game directly from keyboard state."""
        ...


class ControlDisplayProvider(Protocol):
    """Optional game capability for rendering built-in control hints."""

    show_controls: bool


def step_vector_env_no_copy(
    envs: Sequence[BaseGame],
    states: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Step vectorized envs and return reusable internal buffers.

    Completed envs are reset before returning so the state buffer is ready for
    the next batched action selection. The done mask still tells replay logic to
    ignore the reset next-state value for terminal transitions.
    """
    if not envs:
        return states, rewards, dones, []
    actions = validate_action_batch(
        actions,
        expected_size=len(envs),
        action_size=envs[0].action_size,
        game_name=envs[0].__class__.__name__,
    )
    infos = []

    for i, (env, action) in enumerate(zip(envs, actions)):
        next_state, reward, done, info = env.step(int(action))

        states[i] = next_state
        rewards[i] = reward
        dones[i] = done
        infos.append(info)

        if done:
            states[i] = env.reset()

    return states, rewards, dones, infos
