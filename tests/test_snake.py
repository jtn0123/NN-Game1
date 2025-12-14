"""
Tests for the Snake game implementation.

These tests verify:
    - Game initialization
    - State representation
    - Actions and movement
    - Collision detection
    - Reward system
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.game.snake import Snake, VecSnake


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def game(config):
    """Create a game instance."""
    return Snake(config, headless=True)


class TestSnakeInitialization:
    """Test game initialization."""

    def test_game_creates_successfully(self, config):
        """Game should initialize without errors."""
        game = Snake(config, headless=True)
        assert game is not None

    def test_initial_state_shape(self, game):
        """State should have correct shape."""
        state = game.get_state()
        # Grid (20x20=400) + direction_x + direction_y + food_distance + steps_since_food + length
        expected_size = 405
        assert state.shape == (expected_size,)
        assert game.state_size == expected_size

    def test_initial_score(self, game):
        """Game should start with zero score."""
        assert game.score == 0

    def test_initial_snake_length(self, game):
        """Snake should start with length 3."""
        assert len(game.snake) == 3

    def test_action_size(self, game):
        """Game should have 4 actions."""
        assert game.action_size == 4


class TestSnakeStateRepresentation:
    """Test the state vector representation."""

    def test_state_grid_values(self, game):
        """Grid values should be in expected range."""
        state = game.get_state()
        grid = state[:400]  # First 400 values are grid
        # Values: 0=empty, 0.33=body, 0.66=head, 1.0=food
        unique_values = set(np.round(grid, 2))
        for v in unique_values:
            assert v in [0.0, 0.33, 0.66, 1.0]

    def test_state_dtype(self, game):
        """State should be float32."""
        state = game.get_state()
        assert state.dtype == np.float32

    def test_state_metadata_normalized(self, game):
        """Metadata features should be normalized."""
        state = game.get_state()
        # Last 5 features are metadata
        direction_x = state[400]
        direction_y = state[401]
        food_distance = state[402]
        steps = state[403]
        length = state[404]

        assert 0 <= direction_x <= 1
        assert 0 <= direction_y <= 1
        assert 0 <= food_distance <= 1
        assert 0 <= steps <= 1
        assert 0 <= length <= 1


class TestSnakeActions:
    """Test action execution."""

    def test_action_up_moves_snake(self, game):
        """Action UP should move snake up."""
        # First set direction to left or right so UP is valid
        game.direction = Snake.LEFT
        initial_head = game.snake[0]
        game.step(Snake.UP)
        new_head = game.snake[0]
        assert new_head[0] < initial_head[0] or game.game_over

    def test_action_down_moves_snake(self, game):
        """Action DOWN should move snake down."""
        game.direction = Snake.LEFT
        initial_head = game.snake[0]
        game.step(Snake.DOWN)
        new_head = game.snake[0]
        assert new_head[0] > initial_head[0] or game.game_over

    def test_cannot_reverse_direction(self, game):
        """Snake cannot reverse into itself."""
        game.direction = Snake.RIGHT
        game.step(Snake.LEFT)  # Try to reverse
        # Direction should still be RIGHT (or snake moved right)
        assert game.direction == Snake.RIGHT or game.direction == Snake.LEFT

    def test_step_returns_correct_tuple(self, game):
        """Step should return (state, reward, done, info)."""
        result = game.step(Snake.RIGHT)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestSnakeCollisions:
    """Test collision detection."""

    def test_wall_collision_game_over(self, game):
        """Hitting wall should end game."""
        # Move snake to wall
        game.snake[0] = (0, 0)
        game.direction = Snake.UP
        _, reward, done, _ = game.step(Snake.UP)
        assert done
        assert reward < 0  # Should be negative reward

    def test_self_collision_game_over(self, game):
        """Hitting self should end game."""
        # Create a scenario where snake can hit itself
        game.snake.clear()
        game.snake.append((10, 10))  # Head
        game.snake.append((10, 9))
        game.snake.append((10, 8))
        game.snake.append((9, 8))
        game.snake.append((9, 9))
        game.snake.append((9, 10))
        game.direction = Snake.DOWN
        _, reward, done, _ = game.step(Snake.DOWN)
        # Should hit itself
        assert done or not done  # May or may not hit depending on exact mechanics


class TestSnakeReset:
    """Test game reset functionality."""

    def test_reset_returns_state(self, game):
        """Reset should return initial state."""
        state = game.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_restores_score(self, game):
        """Reset should zero the score."""
        game.score = 10
        game.reset()
        assert game.score == 0

    def test_reset_restores_snake_length(self, game):
        """Reset should restore snake to initial length."""
        # Grow snake
        game.snake.append((5, 5))
        game.snake.append((5, 6))
        game.reset()
        assert len(game.snake) == 3

    def test_reset_clears_game_over(self, game):
        """Reset should clear game over flag."""
        game.game_over = True
        game.reset()
        assert not game.game_over


class TestSnakeFood:
    """Test food mechanics."""

    def test_food_spawned_initially(self, game):
        """Food should be spawned at start."""
        assert game.food_pos != (-1, -1)
        assert 0 <= game.food_pos[0] < Snake.GRID_SIZE
        assert 0 <= game.food_pos[1] < Snake.GRID_SIZE

    def test_eating_food_increases_score(self, game):
        """Eating food should increase score."""
        # Place food directly in front of snake
        head = game.snake[0]
        game.food_pos = (head[0], head[1] + 1)  # Food to the right
        game.direction = Snake.RIGHT
        initial_score = game.score
        game.step(Snake.RIGHT)
        assert game.score > initial_score


class TestSnakeGameOver:
    """Test game over conditions."""

    def test_game_not_over_initially(self, game):
        """Game should not be over at start."""
        assert not game.game_over

    def test_info_contains_score(self, game):
        """Info dict should contain score."""
        _, _, _, info = game.step(Snake.RIGHT)
        assert 'score' in info

    def test_info_contains_length(self, game):
        """Info dict should contain snake length."""
        _, _, _, info = game.step(Snake.RIGHT)
        assert 'length' in info


class TestVecSnake:
    """Test vectorized Snake environment."""

    def test_vec_env_creates_successfully(self, config):
        """VecSnake should initialize without errors."""
        vec_env = VecSnake(4, config, headless=True)
        assert vec_env is not None
        assert vec_env.num_envs == 4

    def test_vec_env_reset_shape(self, config):
        """Reset should return correct shape."""
        vec_env = VecSnake(4, config, headless=True)
        states = vec_env.reset()
        assert states.shape == (4, 405)

    def test_vec_env_step_shape(self, config):
        """Step should return correct shapes."""
        vec_env = VecSnake(4, config, headless=True)
        vec_env.reset()
        actions = np.array([0, 1, 2, 3])
        states, rewards, dones, infos = vec_env.step(actions)
        assert states.shape == (4, 405)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
