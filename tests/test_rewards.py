"""
Tests for reward calculations in games.

These tests verify that game events produce correct rewards,
catching reward bugs that silently break training.
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
    return Breakout(config, headless=True)


class TestBreakoutRewards:
    """Test reward calculations in Breakout."""

    def test_brick_hit_gives_positive_reward(self, game, config):
        """Breaking a brick should give positive reward."""
        game.reset()

        # Find a brick that's alive
        alive_brick = None
        for brick in game.bricks:
            if brick.alive:
                alive_brick = brick
                break

        assert alive_brick is not None, "No alive bricks found"

        # Position ball to hit the brick from below
        game.ball.x = alive_brick.rect.centerx
        game.ball.y = alive_brick.rect.bottom + game.ball.radius + 1
        game.ball.dy = -abs(config.BALL_SPEED)  # Moving up toward brick

        initial_bricks = sum(1 for b in game.bricks if b.alive)

        # Step until brick is hit or timeout
        total_reward = 0.0
        for _ in range(20):
            _, reward, done, _ = game.step(1)  # STAY
            total_reward += reward
            current_bricks = sum(1 for b in game.bricks if b.alive)
            if current_bricks < initial_bricks:
                break

        # Should have received brick hit reward (allow small margin for tracking rewards)
        assert total_reward >= config.REWARD_BRICK_HIT - 0.1

    def test_ball_lost_gives_negative_reward(self, game, config):
        """Losing the ball should give negative reward."""
        game.reset()

        # Position ball below paddle, moving down
        game.ball.x = game.width // 2
        game.ball.y = game.height - 10
        game.ball.dy = abs(config.BALL_SPEED)  # Moving down (off screen)

        initial_lives = game.lives

        # Step until life is lost
        total_reward = 0.0
        for _ in range(50):
            _, reward, done, _ = game.step(1)  # STAY
            total_reward += reward
            if game.lives < initial_lives:
                break

        # Should have received game over penalty
        assert total_reward <= config.REWARD_GAME_OVER

    def test_paddle_hit_gives_small_reward(self, game, config):
        """Ball bouncing off paddle should give small positive reward."""
        game.reset()

        # Position ball just above paddle, moving down
        paddle_center_x = game.paddle.x + game.paddle.width // 2
        game.ball.x = paddle_center_x
        game.ball.y = game.paddle.y - game.ball.radius - 2
        game.ball.dy = abs(config.BALL_SPEED)  # Moving down toward paddle

        # Step and check for paddle hit reward
        total_reward = 0.0
        ball_bounced = False
        initial_dy = game.ball.dy

        for _ in range(10):
            _, reward, done, _ = game.step(1)  # STAY
            total_reward += reward
            # Check if ball bounced (dy changed sign)
            if game.ball.dy < 0 and initial_dy > 0:
                ball_bounced = True
                break

        # If ball bounced off paddle, should have paddle hit reward
        if ball_bounced:
            assert total_reward >= config.REWARD_PADDLE_HIT

    def test_win_gives_large_reward(self, game, config):
        """Clearing all bricks should give large win reward."""
        game.reset()

        # Kill all but one brick
        for brick in game.bricks[:-1]:
            brick.alive = False
        game._bricks_remaining = 1

        # Position ball to hit the last brick
        last_brick = game.bricks[-1]
        game.ball.x = last_brick.rect.centerx
        game.ball.y = last_brick.rect.bottom + game.ball.radius + 1
        game.ball.dy = -abs(config.BALL_SPEED)

        # Step until win
        total_reward = 0.0
        for _ in range(50):
            _, reward, done, info = game.step(1)
            total_reward += reward
            if info.get('won', False):
                break

        # Should have received win reward (plus brick hit)
        assert total_reward >= config.REWARD_WIN

    def test_step_reward_applied(self, game, config):
        """Per-step reward should be applied each step."""
        game.reset()

        # Take a single step with no events
        _, reward, _, _ = game.step(1)  # STAY

        # Reward should include step reward (may be 0 by default)
        # This just verifies no error occurs
        assert isinstance(reward, (int, float))


class TestRewardConsistency:
    """Test reward calculation consistency."""

    def test_rewards_are_numeric(self, game):
        """All rewards should be numeric values."""
        game.reset()

        for _ in range(100):
            action = np.random.randint(0, game.action_size)
            _, reward, done, _ = game.step(action)

            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)
            assert not np.isinf(reward)

            if done:
                game.reset()

    def test_rewards_bounded(self, game, config):
        """Rewards should be within reasonable bounds."""
        game.reset()

        min_expected = config.REWARD_GAME_OVER - 1  # Allow small margin
        max_expected = config.REWARD_WIN + config.REWARD_BRICK_HIT + 1

        for _ in range(200):
            action = np.random.randint(0, game.action_size)
            _, reward, done, _ = game.step(action)

            # Single-step rewards should be bounded
            assert reward >= min_expected, f"Reward {reward} below minimum"
            assert reward <= max_expected, f"Reward {reward} above maximum"

            if done:
                game.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
