"""
Tests for the Pong game implementation.

These tests verify:
    - Game initialization
    - State representation
    - Actions and physics
    - Collision detection
    - Reward system
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.game.pong import Pong, VecPong


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def game(config):
    """Create a game instance."""
    return Pong(config, headless=True)


class TestPongInitialization:
    """Test game initialization."""

    def test_game_creates_successfully(self, config):
        """Game should initialize without errors."""
        game = Pong(config, headless=True)
        assert game is not None

    def test_initial_state_shape(self, game):
        """State should have correct shape."""
        state = game.get_state()
        expected_size = 10  # Pong has 10 state features
        assert state.shape == (expected_size,)
        assert game.state_size == expected_size

    def test_initial_scores(self, game):
        """Game should start with zero scores."""
        assert game.player_score == 0
        assert game.ai_score == 0

    def test_action_size(self, game):
        """Game should have 3 actions."""
        assert game.action_size == 3


class TestPongStateRepresentation:
    """Test the state vector representation."""

    def test_state_is_normalized(self, game):
        """State values should be roughly in [0, 1] range."""
        state = game.get_state()
        # Check paddle and ball positions
        assert 0 <= state[0] <= 1  # paddle_y
        assert 0 <= state[1] <= 1  # opponent_y
        assert 0 <= state[2] <= 1  # ball_x
        assert 0 <= state[3] <= 1  # ball_y

    def test_state_dtype(self, game):
        """State should be float32."""
        state = game.get_state()
        assert state.dtype == np.float32

    def test_state_ball_approaching_indicator(self, game):
        """Ball approaching indicator should be 0 or 1."""
        state = game.get_state()
        # ball_approaching is last feature
        assert state[9] in [0.0, 1.0]


class TestPongActions:
    """Test action execution."""

    def test_action_up_moves_paddle(self, game):
        """Action 0 (UP) should move paddle up."""
        initial_y = game.player_paddle.y
        game.step(0)  # UP
        assert game.player_paddle.y < initial_y or game.player_paddle.y == 0

    def test_action_down_moves_paddle(self, game):
        """Action 2 (DOWN) should move paddle down."""
        initial_y = game.player_paddle.y
        game.step(2)  # DOWN
        assert game.player_paddle.y > initial_y or game.player_paddle.y >= game.height - game.PADDLE_HEIGHT

    def test_action_stay_no_movement(self, game):
        """Action 1 (STAY) should not move paddle."""
        initial_y = game.player_paddle.y
        game.step(1)  # STAY
        assert game.player_paddle.y == initial_y

    def test_step_returns_correct_tuple(self, game):
        """Step should return (state, reward, done, info)."""
        result = game.step(1)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestPongCollisions:
    """Test collision detection."""

    def test_ball_bounces_off_top_wall(self, game):
        """Ball should bounce off top wall."""
        game.ball.y = 5
        game.ball.dy = -5
        game.step(1)
        assert game.ball.dy > 0  # Should be moving down now

    def test_ball_bounces_off_bottom_wall(self, game):
        """Ball should bounce off bottom wall."""
        game.ball.y = game.height - 5
        game.ball.dy = 5
        game.step(1)
        assert game.ball.dy < 0  # Should be moving up now

    def test_paddle_bounds_top(self, game):
        """Paddle should not go past top edge."""
        for _ in range(1000):
            game.step(0)  # UP
        assert game.player_paddle.y >= 0

    def test_paddle_bounds_bottom(self, game):
        """Paddle should not go past bottom edge."""
        for _ in range(1000):
            game.step(2)  # DOWN
        assert game.player_paddle.y <= game.height - game.PADDLE_HEIGHT


class TestPongReset:
    """Test game reset functionality."""

    def test_reset_returns_state(self, game):
        """Reset should return initial state."""
        state = game.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_restores_scores(self, game):
        """Reset should zero the scores."""
        game.player_score = 5
        game.ai_score = 3
        game.reset()
        assert game.player_score == 0
        assert game.ai_score == 0

    def test_reset_clears_game_over(self, game):
        """Reset should clear game over flag."""
        game.game_over = True
        game.reset()
        assert not game.game_over


class TestPongGameOver:
    """Test game over conditions."""

    def test_game_not_over_initially(self, game):
        """Game should not be over at start."""
        assert not game.game_over

    def test_info_contains_score(self, game):
        """Info dict should contain score."""
        _, _, _, info = game.step(1)
        assert 'score' in info

    def test_info_contains_ai_score(self, game):
        """Info dict should contain AI score."""
        _, _, _, info = game.step(1)
        assert 'ai_score' in info


class TestVecPong:
    """Test vectorized Pong environment."""

    def test_vec_env_creates_successfully(self, config):
        """VecPong should initialize without errors."""
        vec_env = VecPong(4, config, headless=True)
        assert vec_env is not None
        assert vec_env.num_envs == 4

    def test_vec_env_reset_shape(self, config):
        """Reset should return correct shape."""
        vec_env = VecPong(4, config, headless=True)
        states = vec_env.reset()
        assert states.shape == (4, 10)

    def test_vec_env_step_shape(self, config):
        """Step should return correct shapes."""
        vec_env = VecPong(4, config, headless=True)
        vec_env.reset()
        actions = np.array([1, 0, 2, 1])
        states, rewards, dones, infos = vec_env.step(actions)
        assert states.shape == (4, 10)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
