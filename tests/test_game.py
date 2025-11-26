"""
Tests for the Breakout game implementation.

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
from src.game.breakout import Breakout


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def game(config):
    """Create a game instance."""
    return Breakout(config)


class TestBreakoutInitialization:
    """Test game initialization."""
    
    def test_game_creates_successfully(self, config):
        """Game should initialize without errors."""
        game = Breakout(config)
        assert game is not None
    
    def test_initial_state_shape(self, game, config):
        """State should have correct shape."""
        state = game.get_state()
        expected_size = 5 + config.BRICK_ROWS * config.BRICK_COLS
        assert state.shape == (expected_size,)
    
    def test_initial_lives(self, game, config):
        """Game should start with configured lives."""
        assert game.lives == config.LIVES
    
    def test_initial_score(self, game):
        """Score should start at zero."""
        assert game.score == 0
    
    def test_bricks_created(self, game, config):
        """All bricks should be created and alive."""
        expected_bricks = config.BRICK_ROWS * config.BRICK_COLS
        assert len(game.bricks) == expected_bricks
        assert all(brick.alive for brick in game.bricks)


class TestBreakoutStateRepresentation:
    """Test the state vector representation."""
    
    def test_state_is_normalized(self, game):
        """State values should be roughly in [0, 1] range."""
        state = game.get_state()
        # Ball and paddle positions (first 5 values)
        assert 0 <= state[0] <= 1  # ball_x
        assert 0 <= state[1] <= 1  # ball_y
        # Velocities might be slightly outside [0,1] due to normalization
        assert -0.5 <= state[2] <= 1.5  # ball_dx
        assert -0.5 <= state[3] <= 1.5  # ball_dy
        assert 0 <= state[4] <= 1  # paddle_x
    
    def test_brick_states_binary(self, game, config):
        """Brick states should be binary (0 or 1)."""
        state = game.get_state()
        brick_states = state[5:]  # After ball and paddle info
        assert all(b in [0.0, 1.0] for b in brick_states)
    
    def test_state_dtype(self, game):
        """State should be float32."""
        state = game.get_state()
        assert state.dtype == np.float32


class TestBreakoutActions:
    """Test action execution."""
    
    def test_action_left_moves_paddle(self, game):
        """Action 0 (LEFT) should move paddle left."""
        initial_x = game.paddle.x
        game.step(0)  # LEFT
        assert game.paddle.x < initial_x or game.paddle.x == 0
    
    def test_action_right_moves_paddle(self, game):
        """Action 2 (RIGHT) should move paddle right."""
        initial_x = game.paddle.x
        game.step(2)  # RIGHT
        assert game.paddle.x > initial_x or game.paddle.x >= game.width - game.paddle.width
    
    def test_action_stay_no_movement(self, game):
        """Action 1 (STAY) should not move paddle."""
        initial_x = game.paddle.x
        game.step(1)  # STAY
        assert game.paddle.x == initial_x
    
    def test_step_returns_correct_tuple(self, game):
        """Step should return (state, reward, done, info)."""
        result = game.step(1)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestBreakoutCollisions:
    """Test collision detection."""
    
    def test_ball_bounces_off_walls(self, game, config):
        """Ball should bounce off left and right walls."""
        # Force ball to left wall
        game.ball.x = 5
        game.ball.dx = -5
        game.step(1)
        assert game.ball.dx > 0  # Should be moving right now
    
    def test_paddle_bounds_left(self, game):
        """Paddle should not go past left edge."""
        for _ in range(1000):  # Move left many times
            game.step(0)
        assert game.paddle.x >= 0
    
    def test_paddle_bounds_right(self, game, config):
        """Paddle should not go past right edge."""
        for _ in range(1000):  # Move right many times
            game.step(2)
        assert game.paddle.x <= config.SCREEN_WIDTH - game.paddle.width


class TestBreakoutReset:
    """Test game reset functionality."""
    
    def test_reset_returns_state(self, game):
        """Reset should return initial state."""
        state = game.reset()
        assert isinstance(state, np.ndarray)
    
    def test_reset_restores_lives(self, game, config):
        """Reset should restore lives."""
        game.lives = 1
        game.reset()
        assert game.lives == config.LIVES
    
    def test_reset_restores_score(self, game):
        """Reset should zero the score."""
        game.score = 100
        game.reset()
        assert game.score == 0
    
    def test_reset_restores_bricks(self, game, config):
        """Reset should restore all bricks."""
        # Break some bricks
        for brick in game.bricks[:5]:
            brick.alive = False
        
        game.reset()
        assert all(brick.alive for brick in game.bricks)


class TestBreakoutGameOver:
    """Test game over conditions."""
    
    def test_game_not_over_initially(self, game):
        """Game should not be over at start."""
        assert not game.game_over
    
    def test_info_contains_score(self, game):
        """Info dict should contain score."""
        _, _, _, info = game.step(1)
        assert 'score' in info
    
    def test_info_contains_lives(self, game):
        """Info dict should contain lives."""
        _, _, _, info = game.step(1)
        assert 'lives' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

