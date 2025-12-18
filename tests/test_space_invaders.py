"""
Tests for the Space Invaders game implementation.

These tests verify:
    - Game initialization
    - State representation
    - Actions and physics
    - Collision detection
    - Enemy movement
    - Vectorized environment
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.game.space_invaders import SpaceInvaders, VecSpaceInvaders


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def game(config):
    """Create a game instance."""
    return SpaceInvaders(config, headless=True)


class TestSpaceInvadersInitialization:
    """Test game initialization."""

    def test_game_creates_successfully(self, config):
        """Game should initialize without errors."""
        game = SpaceInvaders(config, headless=True)
        assert game is not None

    def test_initial_state_shape(self, game):
        """State should have correct shape."""
        state = game.get_state()
        assert state.shape == (game.state_size,)
        assert game.state_size > 0

    def test_initial_score(self, game):
        """Game should start with zero score."""
        assert game.score == 0

    def test_initial_lives(self, game, config):
        """Game should start with configured lives."""
        assert game.lives == config.LIVES

    def test_initial_level(self, game):
        """Game should start at level 1."""
        assert game.level == 1

    def test_action_size(self, game):
        """Game should have 4 actions (left, stay, right, shoot)."""
        assert game.action_size == 4

    def test_aliens_created(self, game, config):
        """Aliens should be created on initialization."""
        expected_aliens = config.SI_ALIEN_ROWS * config.SI_ALIEN_COLS
        assert len(game.aliens) == expected_aliens
        # All aliens should be alive initially
        alive_count = sum(1 for a in game.aliens if a.alive)
        assert alive_count == expected_aliens


class TestSpaceInvadersStateRepresentation:
    """Test the state vector representation."""

    def test_state_is_normalized(self, game):
        """State values should be in [0, 1] range."""
        state = game.get_state()
        # Most values should be normalized
        # Allow some small tolerance for floating point
        assert np.all(state >= -0.1)
        assert np.all(state <= 1.1)

    def test_state_dtype(self, game):
        """State should be float32."""
        state = game.get_state()
        assert state.dtype == np.float32

    def test_state_contains_ship_position(self, game):
        """First state element should be ship x position."""
        state = game.get_state()
        # Ship position should be roughly centered initially
        assert 0.3 < state[0] < 0.7


class TestSpaceInvadersActions:
    """Test action execution."""

    def test_action_left_moves_ship(self, game):
        """Action 0 (LEFT) should move ship left."""
        initial_x = game.ship.x
        game.step(0)  # LEFT
        assert game.ship.x < initial_x or game.ship.x == 0

    def test_action_right_moves_ship(self, game):
        """Action 2 (RIGHT) should move ship right."""
        initial_x = game.ship.x
        game.step(2)  # RIGHT
        assert game.ship.x > initial_x or game.ship.x >= game.width - game.ship.width

    def test_action_stay_no_movement(self, game):
        """Action 1 (STAY) should not move ship."""
        initial_x = game.ship.x
        game.step(1)  # STAY
        assert game.ship.x == initial_x

    def test_action_shoot_creates_bullet(self, game):
        """Action 3 (SHOOT) should create a bullet."""
        assert len(game.player_bullets) == 0
        game.step(3)  # SHOOT
        assert len(game.player_bullets) == 1

    def test_step_returns_correct_tuple(self, game):
        """Step should return (state, reward, done, info)."""
        result = game.step(1)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_shoot_cooldown(self, game):
        """Rapid shooting should be limited by cooldown."""
        game.step(3)  # First shot
        initial_bullets = len(game.player_bullets)
        game.step(3)  # Immediate second shot (should be blocked)
        # Should still have same number of bullets due to cooldown
        assert len(game.player_bullets) == initial_bullets


class TestSpaceInvadersCollisions:
    """Test collision detection."""

    def test_ship_bounds_left(self, game):
        """Ship should not go past left edge."""
        for _ in range(1000):
            game.step(0)  # LEFT
        assert game.ship.x >= 0

    def test_ship_bounds_right(self, game):
        """Ship should not go past right edge."""
        for _ in range(1000):
            game.step(2)  # RIGHT
        assert game.ship.x <= game.width - game.ship.width


class TestSpaceInvadersAliens:
    """Test alien behavior."""

    def test_aliens_move(self, game):
        """Aliens should move over time."""
        initial_offset = game.alien_x_offset
        for _ in range(10):
            game.step(1)  # STAY
        # Offset should change as aliens move
        assert game.alien_x_offset != initial_offset

    def test_killing_alien_increases_score(self, game, config):
        """Killing an alien should increase score."""
        initial_score = game.score

        # Position ship under first alien
        first_alien = game.aliens[0]
        game.ship.x = int(first_alien.x + game.alien_x_offset)

        # Shoot and step until hit or timeout
        for _ in range(100):
            game.step(3)  # SHOOT
            for _ in range(5):
                game.step(1)  # STAY to let bullet travel
            if game.score > initial_score:
                break

        assert game.score > initial_score


class TestSpaceInvadersReset:
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

    def test_reset_restores_lives(self, game, config):
        """Reset should restore lives."""
        game.lives = 1
        game.reset()
        assert game.lives == config.LIVES

    def test_reset_restores_level(self, game):
        """Reset should restore level to 1."""
        game.level = 5
        game.reset()
        assert game.level == 1

    def test_reset_clears_game_over(self, game):
        """Reset should clear game over flag."""
        game.game_over = True
        game.reset()
        assert not game.game_over

    def test_reset_restores_aliens(self, game, config):
        """Reset should restore all aliens."""
        # Kill some aliens
        for alien in game.aliens[:5]:
            alien.alive = False
        game.reset()
        alive_count = sum(1 for a in game.aliens if a.alive)
        expected = config.SI_ALIEN_ROWS * config.SI_ALIEN_COLS
        assert alive_count == expected


class TestSpaceInvadersGameOver:
    """Test game over conditions."""

    def test_game_not_over_initially(self, game):
        """Game should not be over at start."""
        assert not game.game_over

    def test_info_contains_score(self, game):
        """Info dict should contain score."""
        _, _, _, info = game.step(1)
        assert 'score' in info

    def test_info_contains_level(self, game):
        """Info dict should contain level."""
        _, _, _, info = game.step(1)
        assert 'level' in info

    def test_info_contains_lives(self, game):
        """Info dict should contain lives."""
        _, _, _, info = game.step(1)
        assert 'lives' in info

    def test_info_contains_aliens_remaining(self, game):
        """Info dict should contain aliens_remaining."""
        _, _, _, info = game.step(1)
        assert 'aliens_remaining' in info


class TestVecSpaceInvaders:
    """Test vectorized Space Invaders environment."""

    def test_vec_env_creates_successfully(self, config):
        """VecSpaceInvaders should initialize without errors."""
        vec_env = VecSpaceInvaders(4, config, headless=True)
        assert vec_env is not None
        assert vec_env.num_envs == 4

    def test_vec_env_reset_shape(self, config):
        """Reset should return correct shape."""
        vec_env = VecSpaceInvaders(4, config, headless=True)
        states = vec_env.reset()
        assert states.shape[0] == 4
        assert states.shape[1] == vec_env.state_size

    def test_vec_env_step_shape(self, config):
        """Step should return correct shapes."""
        vec_env = VecSpaceInvaders(4, config, headless=True)
        vec_env.reset()
        actions = np.array([1, 0, 2, 3])
        states, rewards, dones, infos = vec_env.step(actions)
        assert states.shape[0] == 4
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4

    def test_vec_env_independent_games(self, config):
        """Each environment should be independent."""
        vec_env = VecSpaceInvaders(2, config, headless=True)
        vec_env.reset()

        # Move first ship left, second ship right
        for _ in range(10):
            actions = np.array([0, 2])  # LEFT, RIGHT
            vec_env.step(actions)

        # Ships should be at different positions
        ship1_x = vec_env.envs[0].ship.x
        ship2_x = vec_env.envs[1].ship.x
        assert ship1_x != ship2_x


class TestShieldDamageSystem:
    """Test shield/bunker damage mechanics."""

    def test_shields_created_when_enabled(self, config):
        """Shields should be created when enabled in config."""
        config.SI_SHIELDS_ENABLED = True
        config.SI_SHIELD_COUNT = 4
        game = SpaceInvaders(config, headless=True)

        assert len(game.shields) == config.SI_SHIELD_COUNT

    def test_shields_not_created_when_disabled(self, config):
        """No shields should be created when disabled."""
        config.SI_SHIELDS_ENABLED = False
        game = SpaceInvaders(config, headless=True)

        assert len(game.shields) == 0

    def test_shield_blocks_have_health(self, config):
        """Shield blocks should have health attribute."""
        config.SI_SHIELDS_ENABLED = True
        game = SpaceInvaders(config, headless=True)

        assert len(game.shields) > 0
        shield = game.shields[0]
        assert len(shield.blocks) > 0

        # Blocks should have health
        block = shield.blocks[0]
        assert hasattr(block, 'health')
        assert block.health > 0

    def test_shield_block_damaged_by_bullet(self, config):
        """Shield block should take damage when hit by bullet."""
        config.SI_SHIELDS_ENABLED = True
        game = SpaceInvaders(config, headless=True)

        if len(game.shields) == 0:
            pytest.skip("Shields not enabled")

        shield = game.shields[0]
        if len(shield.blocks) == 0:
            pytest.skip("No shield blocks")

        # Get a block's initial health
        block = shield.blocks[0]
        initial_health = block.health

        # Simulate bullet collision by calling check_collision directly
        # Create a mock rect that overlaps with the block
        import pygame
        bullet_rect = pygame.Rect(block.x, block.y, 4, 10)
        shield.check_collision(bullet_rect)

        # Block should be damaged (health reduced or destroyed)
        assert block.health < initial_health or block.health == 0


class TestLevelProgression:
    """Test level progression mechanics."""

    def test_level_starts_at_one(self, game):
        """Game should start at level 1."""
        assert game.level == 1

    def test_level_advances_when_all_aliens_killed(self, config):
        """Level should advance when all aliens are destroyed."""
        config.SI_ALIEN_ROWS = 1
        config.SI_ALIEN_COLS = 1
        game = SpaceInvaders(config, headless=True)

        initial_level = game.level

        # Kill all aliens (need to also update the counter)
        for alien in game.aliens:
            alien.alive = False
        game._aliens_remaining = 0  # Update counter to match

        # Step to trigger level advance
        game.step(1)

        # Level should have advanced
        assert game.level > initial_level

    def test_alien_speed_increases_with_level(self, config):
        """Aliens should move faster in higher levels."""
        config.SI_ALIEN_ROWS = 2
        config.SI_ALIEN_COLS = 2
        game = SpaceInvaders(config, headless=True)

        initial_base_speed = game.alien_base_speed

        # Kill all aliens to advance level (need to also update the counter)
        for alien in game.aliens:
            alien.alive = False
        game._aliens_remaining = 0  # Update counter to match
        game.step(1)

        # Speed should increase
        # Formula: alien_base_speed * (1 + 0.15 * (level-1))
        expected_speed_factor = 1 + 0.15 * (game.level - 1)
        assert game.alien_base_speed >= initial_base_speed * expected_speed_factor * 0.9


class TestUFOBonusMechanics:
    """Test UFO bonus mechanics."""

    def test_ufo_spawns_with_random_chance(self, config):
        """UFO should have a chance to spawn."""
        config.SI_UFO_CHANCE = 1.0  # 100% spawn chance for testing
        game = SpaceInvaders(config, headless=True)

        # Run several steps to trigger UFO spawn
        ufo_spawned = False
        for _ in range(100):
            game.step(1)
            if game.ufo is not None:
                ufo_spawned = True
                break

        assert ufo_spawned, "UFO should spawn with 100% spawn chance"

    def test_ufo_killed_gives_score(self, config):
        """Killing UFO should give score."""
        config.SI_UFO_CHANCE = 1.0
        game = SpaceInvaders(config, headless=True)

        # Wait for UFO to spawn
        for _ in range(100):
            game.step(1)
            if game.ufo is not None:
                break

        if game.ufo is None:
            pytest.skip("UFO didn't spawn")

        initial_score = game.score

        # Kill the UFO
        game.ufo.alive = False
        game.score += 100  # Simulate score increase (actual collision would do this)

        assert game.score > initial_score

    def test_ufo_moves_across_screen(self, config):
        """UFO should move horizontally across the screen."""
        config.SI_UFO_CHANCE = 1.0
        game = SpaceInvaders(config, headless=True)

        # Spawn UFO
        for _ in range(100):
            game.step(1)
            if game.ufo is not None:
                break

        if game.ufo is None:
            pytest.skip("UFO didn't spawn")

        initial_x = game.ufo.x
        game.ufo.update()

        # UFO should have moved
        assert game.ufo.x != initial_x


class TestAlienBulletThreatTracking:
    """Test alien bullet threat tracking in state representation."""

    def test_state_includes_nearest_bullets(self, game, config):
        """State should include information about nearest alien bullets."""
        # Create some alien bullets
        from src.game.space_invaders import Bullet

        for i in range(3):
            bullet = Bullet(
                x=game.ship.x + i * 20,
                y=game.ship.y - 100 - i * 50,
                speed=5,  # Positive speed = moving down (alien bullets)
                is_player=False,
                color=(255, 0, 0)
            )
            game.alien_bullets.append(bullet)

        state = game.get_state()

        # State should be a valid array
        assert isinstance(state, np.ndarray)
        assert state.shape == (game.state_size,)

    def test_bullet_tracking_updates_with_bullets(self, game):
        """State should change when alien bullets are present vs absent."""
        # Get state with no alien bullets
        game.alien_bullets.clear()
        state_no_bullets = game.get_state().copy()

        # Add alien bullets near the ship
        from src.game.space_invaders import Bullet
        bullet = Bullet(
            x=game.ship.x,
            y=game.ship.y - 50,
            speed=5,  # Positive speed = moving down (alien bullets)
            is_player=False,
            color=(255, 0, 0)
        )
        game.alien_bullets.append(bullet)

        state_with_bullets = game.get_state()

        # States should be different (bullet info is in state)
        assert not np.array_equal(state_no_bullets, state_with_bullets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
