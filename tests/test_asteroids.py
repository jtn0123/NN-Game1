"""
Tests for the Asteroids game implementation.

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
from src.game.asteroids import Asteroids, VecAsteroids, Asteroid, Ship


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def game(config):
    """Create a game instance."""
    return Asteroids(config, headless=True)


class TestAsteroidsInitialization:
    """Test game initialization."""

    def test_game_creates_successfully(self, config):
        """Game should initialize without errors."""
        game = Asteroids(config, headless=True)
        assert game is not None

    def test_initial_state_shape(self, game):
        """State should have correct shape."""
        state = game.get_state()
        # ship(5) + asteroids(8*5=40) + bullets(1) + lives(1) = 47
        expected_size = 47
        assert state.shape == (expected_size,)
        assert game.state_size == expected_size

    def test_initial_score(self, game):
        """Game should start with zero score."""
        assert game.score == 0

    def test_initial_lives(self, game):
        """Game should start with max lives."""
        assert game.lives == Asteroids.MAX_LIVES

    def test_initial_asteroids(self, game):
        """Game should start with asteroids."""
        assert len(game.asteroids) == Asteroids.INITIAL_ASTEROID_COUNT

    def test_action_size(self, game):
        """Game should have 5 actions."""
        assert game.action_size == 5


class TestAsteroidsStateRepresentation:
    """Test the state vector representation."""

    def test_state_is_normalized(self, game):
        """State values should be in expected ranges."""
        state = game.get_state()
        # Ship position should be in [0, 1]
        assert 0 <= state[0] <= 1  # ship_x
        assert 0 <= state[1] <= 1  # ship_y
        assert 0 <= state[2] <= 1  # ship_angle

    def test_state_dtype(self, game):
        """State should be float32."""
        state = game.get_state()
        assert state.dtype == np.float32

    def test_state_bullets_normalized(self, game):
        """Bullet count should be normalized."""
        state = game.get_state()
        bullets_active = state[45]  # bullets_active feature
        assert 0 <= bullets_active <= 1

    def test_state_lives_normalized(self, game):
        """Lives should be normalized."""
        state = game.get_state()
        lives = state[46]  # lives feature
        assert 0 <= lives <= 1


class TestAsteroidsActions:
    """Test action execution."""

    def test_action_rotate_left(self, game):
        """Action 0 (ROTATE_LEFT) should rotate ship left."""
        initial_angle = game.ship.angle
        game.step(0)  # ROTATE_LEFT
        assert game.ship.angle < initial_angle

    def test_action_rotate_right(self, game):
        """Action 1 (ROTATE_RIGHT) should rotate ship right."""
        initial_angle = game.ship.angle
        game.step(1)  # ROTATE_RIGHT
        assert game.ship.angle > initial_angle

    def test_action_thrust(self, game):
        """Action 2 (THRUST) should increase velocity."""
        initial_speed = game.ship.velocity.length()
        game.step(2)  # THRUST
        new_speed = game.ship.velocity.length()
        assert new_speed > initial_speed or new_speed == game.ship.max_speed

    def test_action_shoot_creates_bullet(self, game):
        """Action 3 (SHOOT) should create a bullet."""
        initial_bullets = len(game.bullets)
        game.step(3)  # SHOOT
        assert len(game.bullets) > initial_bullets

    def test_action_nothing_no_change(self, game):
        """Action 4 (NOTHING) should not rotate or thrust."""
        initial_angle = game.ship.angle
        game.step(4)  # NOTHING
        # Angle should be same (no rotation)
        assert game.ship.angle == initial_angle

    def test_step_returns_correct_tuple(self, game):
        """Step should return (state, reward, done, info)."""
        result = game.step(4)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestAsteroidsPhysics:
    """Test physics simulation."""

    def test_ship_screen_wrapping(self, game):
        """Ship should wrap around screen edges."""
        game.ship.x = game.width + 10
        game.ship.update(game.width, game.height)
        assert 0 <= game.ship.x < game.width

    def test_ship_drag(self, game):
        """Ship velocity should decrease due to drag."""
        game.ship.velocity.x = 5
        game.ship.velocity.y = 0
        initial_vx = game.ship.velocity.x
        game.ship.update(game.width, game.height)
        assert game.ship.velocity.x < initial_vx

    def test_asteroid_movement(self, game):
        """Asteroids should move."""
        if game.asteroids:
            asteroid = game.asteroids[0]
            initial_x = asteroid.x
            asteroid.dx = 5  # Set velocity
            asteroid.update(game.width, game.height)
            assert asteroid.x != initial_x or asteroid.dx == 0


class TestAsteroidsCollisions:
    """Test collision detection."""

    def test_bullet_destroys_asteroid(self, game):
        """Bullet hitting asteroid should destroy it."""
        # Place bullet on asteroid
        if game.asteroids:
            asteroid = game.asteroids[0]
            game.bullets.clear()
            from src.game.asteroids import Bullet
            bullet = Bullet(asteroid.x, asteroid.y, 0)
            game.bullets.append(bullet)

            initial_asteroids = len(game.asteroids)
            game._check_collisions()
            # Asteroid should be destroyed or split
            assert len(game.asteroids) != initial_asteroids or len(game.bullets) == 0

    def test_asteroid_splits(self):
        """Large asteroid should split into smaller ones."""
        asteroid = Asteroid(100, 100, Asteroid.LARGE)
        children = asteroid.split()
        assert len(children) == 2
        assert all(a.size == Asteroid.MEDIUM for a in children)

    def test_small_asteroid_no_split(self):
        """Small asteroid should not split."""
        asteroid = Asteroid(100, 100, Asteroid.SMALL)
        children = asteroid.split()
        assert len(children) == 0


class TestAsteroidsReset:
    """Test game reset functionality."""

    def test_reset_returns_state(self, game):
        """Reset should return initial state."""
        state = game.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_restores_score(self, game):
        """Reset should zero the score."""
        game.score = 1000
        game.reset()
        assert game.score == 0

    def test_reset_restores_lives(self, game):
        """Reset should restore lives."""
        game.lives = 1
        game.reset()
        assert game.lives == Asteroids.MAX_LIVES

    def test_reset_clears_bullets(self, game):
        """Reset should clear all bullets."""
        game.bullets.append(object())  # Add dummy bullet
        game.reset()
        assert len(game.bullets) == 0

    def test_reset_spawns_asteroids(self, game):
        """Reset should spawn initial asteroids."""
        game.asteroids.clear()
        game.reset()
        assert len(game.asteroids) == Asteroids.INITIAL_ASTEROID_COUNT


class TestAsteroidsGameOver:
    """Test game over conditions."""

    def test_game_not_over_initially(self, game):
        """Game should not be over at start."""
        assert not game.game_over

    def test_info_contains_score(self, game):
        """Info dict should contain score."""
        _, _, _, info = game.step(4)
        assert 'score' in info

    def test_info_contains_lives(self, game):
        """Info dict should contain lives."""
        _, _, _, info = game.step(4)
        assert 'lives' in info

    def test_info_contains_level(self, game):
        """Info dict should contain level."""
        _, _, _, info = game.step(4)
        assert 'level' in info


class TestShip:
    """Test Ship class."""

    def test_ship_creates_successfully(self):
        """Ship should initialize without errors."""
        ship = Ship(100, 100)
        assert ship is not None
        assert ship.alive

    def test_ship_rotation(self):
        """Ship rotation should work."""
        ship = Ship(100, 100)
        initial_angle = ship.angle
        ship.rotate_left()
        assert ship.angle < initial_angle

    def test_ship_thrust(self):
        """Ship thrust should increase velocity."""
        ship = Ship(100, 100)
        initial_speed = ship.velocity.length()
        ship.thrust()
        assert ship.velocity.length() > initial_speed


class TestVecAsteroids:
    """Test vectorized Asteroids environment."""

    def test_vec_env_creates_successfully(self, config):
        """VecAsteroids should initialize without errors."""
        vec_env = VecAsteroids(4, config, headless=True)
        assert vec_env is not None
        assert vec_env.num_envs == 4

    def test_vec_env_reset_shape(self, config):
        """Reset should return correct shape."""
        vec_env = VecAsteroids(4, config, headless=True)
        states = vec_env.reset()
        assert states.shape == (4, 47)

    def test_vec_env_step_shape(self, config):
        """Step should return correct shapes."""
        vec_env = VecAsteroids(4, config, headless=True)
        vec_env.reset()
        actions = np.array([0, 1, 2, 3])
        states, rewards, dones, infos = vec_env.step(actions)
        assert states.shape == (4, 47)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
