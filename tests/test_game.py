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
        # 8 features: ball(x,y,dx,dy), paddle(x), tracking(relative_x, predicted_landing, distance_to_target)
        # plus brick states
        expected_size = config.STATE_SIZE
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
        # New tracking features (indices 5, 6, 7)
        assert 0 <= state[5] <= 1  # relative_x
        assert 0 <= state[6] <= 1  # predicted_landing
        assert 0 <= state[7] <= 1  # distance_to_target
    
    def test_brick_states_binary(self, game, config):
        """Brick states should be binary (0 or 1)."""
        state = game.get_state()
        brick_states = state[8:]  # After ball, paddle, and tracking info (8 features)
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


class TestBallBrickCollisionPhysics:
    """Test ball-brick collision bounce direction physics."""

    def test_ball_bounces_horizontal_on_side_hit(self, game, config):
        """Ball should reverse dx (horizontal) when hitting brick from side."""
        # Position ball to hit brick from the side
        # Find a brick and position ball to its left, moving right
        brick = game.bricks[0]

        # Position ball just to the left of brick, moving right
        game.ball.x = brick.rect.left - game.ball.radius - 1
        game.ball.y = brick.rect.centery  # Center vertically
        game.ball.dx = 5  # Moving right
        game.ball.dy = 0.5  # Slight downward to ensure collision

        initial_dx = game.ball.dx
        initial_dy = game.ball.dy

        # Step the game
        game.step(1)

        # If brick was hit from side, dx should flip
        if not brick.alive:
            # Ball was moving right (positive dx), should now be moving left (negative dx)
            assert game.ball.dx != initial_dx, "Ball dx should change on side collision"

    def test_ball_bounces_vertical_on_top_bottom_hit(self, game, config):
        """Ball should reverse dy (vertical) when hitting brick from top/bottom."""
        # Position ball above a brick, moving down
        brick = game.bricks[0]

        # Position ball just above the brick, moving down
        game.ball.x = brick.rect.centerx  # Center horizontally
        game.ball.y = brick.rect.top - game.ball.radius - 1
        game.ball.dx = 0.5  # Slight horizontal movement
        game.ball.dy = 5  # Moving down

        initial_dy = game.ball.dy

        # Step the game
        game.step(1)

        # If brick was hit from top, dy should flip
        if not brick.alive:
            assert game.ball.dy != initial_dy, "Ball dy should change on top/bottom collision"

    def test_brick_destroyed_on_collision(self, game, config):
        """Brick should be destroyed when ball collides with it."""
        brick = game.bricks[0]
        assert brick.alive is True

        # Position ball inside brick bounds
        game.ball.x = brick.rect.centerx
        game.ball.y = brick.rect.centery
        game.ball.dx = 1
        game.ball.dy = 1

        game.step(1)

        # Brick should be destroyed
        assert brick.alive is False
        assert game.score > 0


class TestPaddleBounceAngle:
    """Test paddle bounce angle mechanics."""

    def test_center_hit_bounces_straight_up(self, game, config):
        """Ball hitting paddle center should bounce roughly straight up."""
        # Position ball directly above paddle center
        paddle_center_x = game.paddle.x + game.paddle.width / 2
        game.ball.x = paddle_center_x
        game.ball.y = game.paddle.y - game.ball.radius - 5
        game.ball.dx = 0  # No horizontal velocity
        game.ball.dy = 5  # Moving down

        # Force ball to hit paddle
        game.ball.y = game.paddle.y - game.ball.radius - 1

        game.step(1)

        # After center hit, dx should be small (close to straight up)
        # Angle should be -90° ± some tolerance
        assert game.ball.dy < 0, "Ball should be moving up after paddle hit"
        # Center hit should result in small dx
        assert abs(game.ball.dx) < abs(game.ball.dy) * 0.5, "Center hit should be mostly vertical"

    def test_edge_hit_bounces_at_angle(self, game, config):
        """Ball hitting paddle edge should bounce at sharper angle."""
        # Position ball at right edge of paddle
        paddle_right = game.paddle.x + game.paddle.width
        game.ball.x = paddle_right - 5  # Near right edge
        game.ball.y = game.paddle.y - game.ball.radius - 1
        game.ball.dx = 0
        game.ball.dy = 5  # Moving down

        game.step(1)

        # After edge hit, ball should have significant horizontal component
        assert game.ball.dy < 0, "Ball should be moving up"
        # Right edge hit should result in positive dx (moving right)
        assert game.ball.dx > 0, "Right edge hit should bounce ball to the right"

    def test_left_edge_hit_bounces_left(self, game, config):
        """Ball hitting left paddle edge should bounce to the left."""
        # Position ball at left edge of paddle
        game.ball.x = game.paddle.x + 5  # Near left edge
        game.ball.y = game.paddle.y - game.ball.radius - 1
        game.ball.dx = 0
        game.ball.dy = 5  # Moving down

        game.step(1)

        # After left edge hit, ball should move left (negative dx)
        assert game.ball.dy < 0, "Ball should be moving up"
        assert game.ball.dx < 0, "Left edge hit should bounce ball to the left"

    def test_ball_y_adjusted_after_paddle_hit(self, game, config):
        """Ball should be repositioned above paddle after collision to prevent sticking."""
        # Position ball at paddle level
        game.ball.x = game.paddle.x + game.paddle.width / 2
        game.ball.y = game.paddle.y - game.ball.radius  # Just touching
        game.ball.dx = 0
        game.ball.dy = 5  # Moving down

        game.step(1)

        # Ball should be positioned above paddle
        assert game.ball.y <= game.paddle.y - game.ball.radius


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

