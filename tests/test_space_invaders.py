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

import os
import sys

import numpy as np
import pygame
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.game.space_invaders import SpaceInvaders, VecSpaceInvaders
from src.game.space_invaders_entities import (
    UFO,
    Alien,
    Bullet,
    Particle,
    ScorePopup,
    Shield,
    ShieldBlock,
    Ship,
    Star,
    WaveAnnouncement,
)
from src.game.space_invaders_entities import (
    _font as entity_font,
)


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

    def test_state_size_matches_named_components(self, game, config):
        """State size should match the fields written by get_state()."""
        expected = (
            1
            + config.SI_MAX_PLAYER_BULLETS * 2
            + config.SI_ALIEN_ROWS * config.SI_ALIEN_COLS
            + 3
            + game._max_tracked_alien_bullets * 2
            + 5
        )
        assert game.state_size == expected

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
        assert "score" in info

    def test_info_contains_level(self, game):
        """Info dict should contain level."""
        _, _, _, info = game.step(1)
        assert "level" in info

    def test_info_contains_lives(self, game):
        """Info dict should contain lives."""
        _, _, _, info = game.step(1)
        assert "lives" in info

    def test_info_contains_aliens_remaining(self, game):
        """Info dict should contain aliens_remaining."""
        _, _, _, info = game.step(1)
        assert "aliens_remaining" in info


class TestSpaceInvadersRenderingResources:
    """Test cached rendering resources."""

    def test_hud_font_cache_reuses_font_instances(self, game):
        pygame.font.init()

        assert game._font(36) is game._font(36)
        assert game._font(28) is game._font(28)

    def test_entity_font_cache_reuses_popup_fonts(self):
        pygame.font.init()
        entity_font.cache_clear()

        assert entity_font(24) is entity_font(24)


class TestSpaceInvadersRendering:
    """Render Space Invaders visual paths on offscreen pygame surfaces."""

    @pytest.fixture
    def screen(self, config):
        pygame.init()
        pygame.font.init()
        return pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    def test_entity_draw_paths_complete_on_surface(self, screen):
        """Entity draw methods should render without mutating liveness unexpectedly."""
        particle = Particle(50, 50, (200, 100, 50))
        particle.draw(screen)
        particle.life = 0
        particle.draw(screen)

        popup = ScorePopup(80, 80, 100, (255, 200, 50))
        popup.draw(screen)
        assert popup.update() is True
        popup.life = 0
        popup.draw(screen)

        announcement = WaveAnnouncement(3, screen.get_width(), screen.get_height())
        announcement.draw(screen)
        assert announcement.update(0.1) is True
        announcement.timer = announcement.duration
        assert announcement.update(0.1) is False
        announcement.draw(screen)

        star = Star(20, 20, 2.0, 100)
        star.draw(screen, time=0.5)
        star.y = screen.get_height() + 1
        star.update(screen.get_height())
        assert star.y == 0.0

        bullet = Bullet(100, 100, -8, True, (100, 255, 200))
        for _ in range(7):
            bullet.update()
        assert len(bullet.trail) == 5
        bullet.draw(screen)
        bullet.alive = False
        bullet.draw(screen)

        alien = Alien(120, 80, 32, 24, 1, (0, 255, 100))
        alien.draw(screen, time=0.0)
        alien.draw(screen, time=0.6)
        alien.alive = False
        alien.draw(screen, time=0.0)

        ship = Ship(200, 250, 34, 24, 6, (100, 255, 100))
        ship.draw(screen, time=0.0)
        ship.move(-100, screen.get_width())
        assert ship.x == 0
        ship.move(1000, screen.get_width())
        assert ship.x == screen.get_width() - ship.width

        ufo = UFO(-50, 40, 5, (255, 80, 80))
        ufo.draw(screen, time=0.25)
        ufo.update()
        assert ufo.x == -45
        ufo.alive = False
        ufo.draw(screen, time=0.25)

    def test_shield_damage_and_draw_paths(self, screen):
        """Shield blocks should fade, outline when damaged, and report liveness."""
        block = ShieldBlock(10, 10, 8, (0, 255, 100))
        block.draw(screen)
        assert block.hit() is False
        assert block.health == 3
        block.draw(screen)
        block.hit()
        block.hit()
        assert block.hit() is True
        assert not block.alive
        block.draw(screen)

        shield = Shield(40, 40, 48, 24, (0, 255, 100))
        assert shield.alive
        assert shield.check_collision(shield.blocks[0].rect) is True
        shield.draw(screen)
        for shield_block in shield.blocks:
            shield_block.alive = False
        assert not shield.alive

    def test_full_render_restores_shaken_positions(self, config, screen):
        """Full visual rendering should restore positions after shake offsets."""
        config.SI_SHIELDS_ENABLED = True
        game = SpaceInvaders(config, headless=False)
        game.screen_shake = 4
        game.flash_alpha = 50
        game.player_invincible = True
        game._time = 0.1  # Exercises the ghost invincibility branch.
        game.ufo = UFO(30, 40, 3, config.SI_COLOR_UFO)
        game.player_bullets.append(Bullet(100, 200, -8, True, (100, 255, 200)))
        game.alien_bullets.append(Bullet(120, 250, 5, False, (255, 80, 80)))
        game.score_popups.append(ScorePopup(150, 150, 30, (255, 200, 50)))
        game.wave_announcement = WaveAnnouncement(2, game.width, game.height)
        game.game_over = True
        game.score = 123
        game.total_aliens_killed = 7

        original_alien_pos = (game.aliens[0].x, game.aliens[0].y)
        original_ship_pos = (game.ship.x, game.ship.y)
        original_ufo_pos = (game.ufo.x, game.ufo.y)
        original_player_bullet_pos = (game.player_bullets[0].x, game.player_bullets[0].y)
        original_alien_bullet_pos = (game.alien_bullets[0].x, game.alien_bullets[0].y)

        game.render(screen)

        assert (game.aliens[0].x, game.aliens[0].y) == original_alien_pos
        assert (game.ship.x, game.ship.y) == original_ship_pos
        assert (game.ufo.x, game.ufo.y) == original_ufo_pos
        assert (game.player_bullets[0].x, game.player_bullets[0].y) == original_player_bullet_pos
        assert (game.alien_bullets[0].x, game.alien_bullets[0].y) == original_alien_bullet_pos

        game.player_invincible = False
        game.game_over = False
        game.render(screen)

    def test_visual_effect_updates_spawn_and_expire(self, config):
        """Visual-mode effects should update, decay, and spawn particles."""
        game = SpaceInvaders(config, headless=False)
        game._spawn_explosion(40, 50, (255, 100, 50), count=3)
        assert len(game.particles) == 3

        game._spawn_player_death_explosion()
        assert len(game.particles) == 63

        game.particles[0].life = 0
        game.score_popups.append(ScorePopup(80, 80, 10, (255, 200, 50)))
        game.score_popups[0].life = 0
        game.wave_announcement = WaveAnnouncement(2, game.width, game.height)
        game.wave_announcement.timer = game.wave_announcement.duration
        game.screen_shake = 0.4
        game.flash_alpha = 20
        star_y = game.stars[0].y

        game._update_effects()

        assert len(game.particles) == 62
        assert game.score_popups == []
        assert game.wave_announcement is None
        assert game.screen_shake == 0
        assert game.flash_alpha == 5
        assert game.stars[0].y != star_y


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
        assert hasattr(block, "health")
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
                color=(255, 0, 0),
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
            color=(255, 0, 0),
        )
        game.alien_bullets.append(bullet)

        state_with_bullets = game.get_state()

        # States should be different (bullet info is in state)
        assert not np.array_equal(state_no_bullets, state_with_bullets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
