"""
Space Invaders Game Implementation
===================================

A classic Space Invaders clone designed for AI training.

Key Features:
- Clean state representation for neural network input
- Configurable rewards for reinforcement learning
- CRT phosphor aesthetic with scanlines and glow effects
- Pixel art aliens with animation frames
- Particle explosions and screen effects
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional
import math
import random
import heapq

from .base_game import BaseGame, validate_action
from src.game.space_invaders_entities import (
    Alien,
    Bullet,
    Particle,
    ScorePopup,
    Shield,
    Ship,
    Star,
    UFO,
    WaveAnnouncement,
)
from src.game.space_invaders_rendering import SpaceInvadersRenderingMixin
from config import Config
from src.game.space_invaders_rules import (
    alien_points,
    alien_pressure_ratio,
    alien_pulse_speed,
    alien_shoot_chance,
    alien_speed_after_kill,
    invasion_reached,
    level_speed,
    level_y_offset,
)


class SpaceInvaders(SpaceInvadersRenderingMixin, BaseGame):
    """Space Invaders game implementation with enhanced visuals."""

    def __init__(self, config: Optional[Config] = None, headless: bool = False):
        """Initialize the Space Invaders game."""
        self.config = config or Config()
        self.headless = headless

        # Screen dimensions
        self.width = self.config.SCREEN_WIDTH
        self.height = self.config.SCREEN_HEIGHT

        # Game objects
        self.ship: Optional[Ship] = None
        self.aliens: List[Alien] = []
        self.player_bullets: List[Bullet] = []
        self.alien_bullets: List[Bullet] = []
        self.ufo: Optional[UFO] = None
        self.shields: List[Shield] = []  # Protective bunkers

        # Visual effects
        self.particles: List[Particle] = []
        self.stars: List[Star] = []
        self.score_popups: List[ScorePopup] = []
        self.wave_announcement: Optional[WaveAnnouncement] = None
        self.screen_shake = 0.0
        self.flash_alpha = 0

        # Alien movement - start slower for better AI learning
        self.alien_direction = 1
        self.alien_base_speed = self.config.SI_ALIEN_SPEED_X
        self.alien_speed = self.alien_base_speed
        self.alien_x_offset = 0.0
        self.alien_drop_distance = self.config.SI_ALIEN_SPEED_Y
        self.alien_pulse_phase = 0.0  # For visual pulse effect

        # Game state
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        self.level = 1
        self.total_aliens_killed = 0

        # Player invincibility after death
        self.player_invincible = False
        self.invincibility_timer = 0.0
        self.invincibility_duration = 2.0  # 2 seconds of invincibility

        # Ground base position (what the player defends)
        self.ground_y = self.height - 25

        # State representation - enhanced with danger info
        self._num_aliens = self.config.SI_ALIEN_ROWS * self.config.SI_ALIEN_COLS
        self._aliens_remaining = self._num_aliens

        self._max_player_bullets = self.config.SI_MAX_PLAYER_BULLETS
        self._max_tracked_alien_bullets = 5  # Track top 5 nearest alien bullets
        movement_feature_count = 3  # alien offset, direction, lowest alien y
        status_feature_count = 5  # cooldown, active bullets, aliens ratio, lives, level
        self._state_size = (
            1
            + self._max_player_bullets * 2
            + self._num_aliens
            + movement_feature_count
            + self._max_tracked_alien_bullets * 2
            + status_feature_count
        )
        self._state_array = np.zeros(self._state_size, dtype=np.float32)
        self._alien_states = np.ones(self._num_aliens, dtype=np.float32)

        self._inv_width = 1.0 / self.width
        self._inv_height = 1.0 / self.height

        self._time = 0.0
        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 10

        # Accuracy tracking
        self._shots_fired = 0
        self._shots_hit = 0

        # Create visual effects
        self._scanline_surface: Optional[pygame.Surface] = None
        self._crt_vignette: Optional[pygame.Surface] = None

        if not headless:
            self._create_stars()
            self._scanline_surface = self._create_scanlines()
            self._crt_vignette = self._create_vignette()

        self.reset()

    def _create_stars(self) -> None:
        """Create background starfield."""
        self.stars = []
        for _ in range(80):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            speed = random.uniform(0.1, 0.5)
            brightness = random.randint(40, 120)
            self.stars.append(Star(x, y, speed, brightness))

    def _create_scanlines(self) -> pygame.Surface:
        """Create scanline overlay for CRT effect."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(0, self.height, 2):
            pygame.draw.line(surface, (0, 0, 0, 50), (0, y), (self.width, y))
        return surface

    def _create_vignette(self) -> pygame.Surface:
        """Create vignette effect for CRT corners."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Create radial gradient from center
        center_x, center_y = self.width // 2, self.height // 2
        max_dist = math.sqrt(center_x**2 + center_y**2)

        for y in range(0, self.height, 4):
            for x in range(0, self.width, 4):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                alpha = int((dist / max_dist) ** 2 * 100)
                pygame.draw.rect(surface, (0, 0, 0, alpha), (x, y, 4, 4))

        return surface

    def _spawn_explosion(
        self, x: int, y: int, color: Tuple[int, int, int], count: int = 15
    ) -> None:
        """Spawn explosion particles."""
        if self.headless:
            return
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _spawn_player_death_explosion(self) -> None:
        """Spawn a dramatic player death explosion."""
        if self.headless or self.ship is None:
            return

        cx = self.ship.rect.centerx
        cy = self.ship.rect.centery

        # Multi-color explosion for dramatic effect
        colors = [
            (255, 100, 50),  # Orange
            (255, 200, 50),  # Yellow
            (255, 50, 50),  # Red
            (100, 255, 100),  # Green (ship color)
        ]

        for color in colors:
            for _ in range(15):
                self.particles.append(Particle(cx, cy, color))

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def action_size(self) -> int:
        return 4

    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        self.level = 1
        self.total_aliens_killed = 0

        # Reset player invincibility
        self.player_invincible = False
        self.invincibility_timer = 0.0

        ship_x = (self.width - self.config.SI_SHIP_WIDTH) // 2
        ship_y = self.height - self.config.SI_SHIP_Y_OFFSET - self.config.SI_SHIP_HEIGHT
        self.ship = Ship(
            ship_x,
            ship_y,
            self.config.SI_SHIP_WIDTH,
            self.config.SI_SHIP_HEIGHT,
            self.config.SI_SHIP_SPEED,
            self.config.SI_COLOR_SHIP,
        )

        self._create_aliens()
        self._create_shields()

        self.player_bullets = []
        self.alien_bullets = []
        self.ufo = None
        self.particles = []
        self.score_popups = []
        self.wave_announcement = None

        self.alien_direction = 1
        self.alien_base_speed = self.config.SI_ALIEN_SPEED_X
        self.alien_pulse_phase = 0.0
        self.alien_speed = self.alien_base_speed
        self.alien_x_offset = 0.0
        self.alien_drop_distance = self.config.SI_ALIEN_SPEED_Y

        self._aliens_remaining = self._num_aliens
        self._alien_states.fill(1.0)
        self._shoot_cooldown = 0
        self._shots_fired = 0
        self._shots_hit = 0
        self.screen_shake = 0
        self.flash_alpha = 0

        return self.get_state()

    def _create_shields(self) -> None:
        """Create the protective bunkers/shields."""
        self.shields = []

        if not self.config.SI_SHIELDS_ENABLED:
            return

        shield_count = self.config.SI_SHIELD_COUNT
        shield_width = self.config.SI_SHIELD_WIDTH
        shield_height = self.config.SI_SHIELD_HEIGHT

        # Position shields evenly across the screen, above the ship
        total_shield_width = shield_count * shield_width
        spacing = (self.width - total_shield_width) / (shield_count + 1)

        # Shield Y position - between aliens and ship
        assert self.ship is not None
        shield_y = self.ship.y - shield_height - 40

        for i in range(shield_count):
            x = int(spacing + i * (shield_width + spacing))
            shield = Shield(x, shield_y, shield_width, shield_height, self.config.SI_COLOR_SHIELD)
            self.shields.append(shield)

    def _next_level(self) -> None:
        """Progress to the next level with increased difficulty."""
        self.level += 1

        # Reset aliens for new level
        self._create_aliens()

        # Clear bullets but keep shields (they persist across levels, just like original)
        self.player_bullets = []
        self.alien_bullets = []
        self.ufo = None

        # Reset alien movement
        self.alien_direction = 1
        self.alien_x_offset = 0.0
        self.alien_pulse_phase = 0.0

        self._aliens_remaining = self._num_aliens
        self._alien_states.fill(1.0)

        # Increase difficulty each level:
        # - Aliens start lower (closer to player)
        # - Base speed increases
        # - Aliens shoot more frequently (handled in config multiplier)
        self.alien_base_speed = level_speed(self.config.SI_ALIEN_SPEED_X, self.level)
        self.alien_speed = self.alien_base_speed

        # Move aliens closer to player on higher levels (original game behavior)
        level_offset = level_y_offset(self.level)
        for alien in self.aliens:
            alien.y += level_offset

        # Show wave announcement
        if not self.headless:
            self.wave_announcement = WaveAnnouncement(self.level, self.width, self.height)

        # Visual feedback for new level
        self.flash_alpha = 100
        self.screen_shake = 5

    def _create_aliens(self) -> None:
        """Create the alien grid with authentic arcade layout.

        Original Space Invaders layout (5 rows, 11 columns):
        - Row 0 (top): Squids - 30 points - type 0 (pink/magenta)
        - Rows 1-2: Crabs - 20 points - type 1 (green)
        - Rows 3-4 (bottom): Octopuses - 10 points - type 2 (cyan)
        """
        self.aliens = []

        colors = [
            self.config.SI_COLOR_ALIEN_1,  # Pink/magenta for squids (top)
            self.config.SI_COLOR_ALIEN_2,  # Green for crabs (middle)
            self.config.SI_COLOR_ALIEN_3,  # Cyan for octopuses (bottom)
        ]

        for row in range(self.config.SI_ALIEN_ROWS):
            # Authentic alien types: row 0 = squids, rows 1-2 = crabs, rows 3-4 = octopuses
            if row == 0:
                alien_type = 0  # Squid - 30 points
            elif row <= 2:
                alien_type = 1  # Crab - 20 points
            else:
                alien_type = 2  # Octopus - 10 points

            color = colors[alien_type % len(colors)]

            for col in range(self.config.SI_ALIEN_COLS):
                x = self.config.SI_ALIEN_OFFSET_LEFT + col * (
                    self.config.SI_ALIEN_WIDTH + self.config.SI_ALIEN_PADDING
                )
                y = self.config.SI_ALIEN_OFFSET_TOP + row * (
                    self.config.SI_ALIEN_HEIGHT + self.config.SI_ALIEN_PADDING
                )

                alien = Alien(
                    x,
                    y,
                    self.config.SI_ALIEN_WIDTH,
                    self.config.SI_ALIEN_HEIGHT,
                    alien_type,
                    color,
                )
                self.aliens.append(alien)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step."""
        if self.game_over or self.won:
            return self.get_state(), 0.0, True, self._get_info()

        action = validate_action(action, self.action_size, "SpaceInvaders")
        assert self.ship is not None

        self._time += 1.0 / 60.0
        reward = 0.0

        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1

        # Handle action with anti-passive rewards
        if action == 0:
            self.ship.move(-1, self.width)
        elif action == 1:
            # Penalty for doing nothing - discourages passive play
            reward += self.config.SI_REWARD_STAY
        elif action == 2:
            self.ship.move(1, self.width)
        elif action == 3:
            if self._shoot_cooldown == 0 and len(self.player_bullets) < self._max_player_bullets:
                self._fire_player_bullet()
                self._shoot_cooldown = self._shoot_cooldown_max
                # Reward for shooting - encourages aggression
                reward += self.config.SI_REWARD_SHOOT

        # Update game objects
        self._update_bullets()
        reward += self._update_aliens()
        self._update_ufo()
        reward += self._handle_collisions()

        # Simple survival reward (constant per step survived)
        reward += self.config.SI_REWARD_STEP

        # Update visual effects
        self._update_effects()

        # Check level clear condition - progress to next level!
        if self._aliens_remaining == 0:
            reward += self.config.SI_REWARD_LEVEL_CLEAR

            # Check win condition: complete SI_WIN_LEVELS levels to win.
            # This block runs after clearing the current level, before incrementing it.
            # (0 means endless mode, no wins possible)
            if self.config.SI_WIN_LEVELS > 0 and self.level >= self.config.SI_WIN_LEVELS:
                # We've completed enough levels - win!
                self.won = True
                self.game_over = True
                reward += self.config.SI_REWARD_LEVEL_CLEAR * 2  # Bonus for winning
            else:
                # Advance to next level if we haven't won yet
                self._next_level()

        # Check if aliens reached the ground base (invasion!)
        for alien in self.aliens:
            if alien.alive and invasion_reached(alien.y, alien.height, self.ground_y):
                self.game_over = True
                reward += self.config.SI_REWARD_PLAYER_DEATH * 2  # Extra penalty for invasion
                break

        return self.get_state(), reward, self.game_over, self._get_info()

    def _update_effects(self) -> None:
        """Update visual effects."""
        # Update invincibility timer (even in headless mode)
        if self.player_invincible:
            self.invincibility_timer -= 1.0 / 60.0
            if self.invincibility_timer <= 0:
                self.player_invincible = False
                self.invincibility_timer = 0.0

        if self.headless:
            return

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Update score popups
        self.score_popups = [p for p in self.score_popups if p.update()]

        # Update wave announcement
        if self.wave_announcement is not None:
            if not self.wave_announcement.update():
                self.wave_announcement = None

        # Update stars
        for star in self.stars:
            star.update(self.height)

        # Decay screen shake
        if self.screen_shake > 0:
            self.screen_shake *= 0.9
            if self.screen_shake < 0.5:
                self.screen_shake = 0

        # Decay flash
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 15)

    def _fire_player_bullet(self) -> None:
        assert self.ship is not None
        bullet = Bullet(
            self.ship.x + self.ship.width // 2,
            self.ship.y - 5,
            -self.config.SI_BULLET_SPEED,
            True,
            (100, 255, 200),  # Cyan-green for player bullets
        )
        self.player_bullets.append(bullet)
        self._shots_fired += 1  # Track for accuracy bonus

    def _update_bullets(self) -> None:
        for bullet in self.player_bullets:
            bullet.update()
            if bullet.y < -10:
                bullet.alive = False
        self.player_bullets = [b for b in self.player_bullets if b.alive]

        for bullet in self.alien_bullets:
            bullet.update()
            if bullet.y > self.height + 10:
                bullet.alive = False
        self.alien_bullets = [b for b in self.alien_bullets if b.alive]

    def _update_aliens(self) -> float:
        reward = 0.0

        self.alien_x_offset += self.alien_direction * self.alien_speed

        # Update alien pulse phase (speeds up as fewer aliens remain)
        self.alien_pulse_phase += alien_pulse_speed(self._aliens_remaining, self._num_aliens) * (
            1.0 / 60.0
        )

        for alien in self.aliens:
            if not alien.alive:
                continue
            actual_x = alien.x + self.alien_x_offset
            if actual_x <= 0 or actual_x + alien.width >= self.width:
                self.alien_direction *= -1
                for a in self.aliens:
                    if a.alive:
                        a.y += self.config.SI_ALIEN_SPEED_Y
                break

        # Only bottom aliens in each column can shoot (authentic behavior)
        bottom_aliens = self._get_bottom_aliens()
        for alien in bottom_aliens:
            # Increase shoot chance slightly as aliens are destroyed
            shoot_chance = alien_shoot_chance(
                self.config.SI_ALIEN_SHOOT_CHANCE,
                self._aliens_remaining,
                self._num_aliens,
            )
            if random.random() < shoot_chance:
                actual_x = alien.x + self.alien_x_offset + alien.width // 2
                bullet = Bullet(
                    actual_x,
                    alien.y + alien.height,
                    self.config.SI_ALIEN_BULLET_SPEED,
                    False,
                    (255, 80, 80),  # Red for alien bullets
                )
                self.alien_bullets.append(bullet)

        # Spawn UFO
        if self.ufo is None and random.random() < self.config.SI_UFO_CHANCE:
            direction = random.choice([-1, 1])
            x = -50 if direction > 0 else self.width + 50
            self.ufo = UFO(x, 50, direction * self.config.SI_UFO_SPEED, self.config.SI_COLOR_UFO)

        return reward

    def _get_bottom_aliens(self) -> List[Alien]:
        """Get the bottom-most alive alien in each column (only these can shoot)."""
        # Aliens are stored row by row, so we need to find the lowest in each column
        cols = self.config.SI_ALIEN_COLS
        bottom_per_col: List[Optional[Alien]] = [None] * cols

        for idx, alien in enumerate(self.aliens):
            if not alien.alive:
                continue
            col = idx % cols
            # Later rows have higher Y values, so we want the highest Y (lowest on screen)
            if bottom_per_col[col] is None or alien.y > bottom_per_col[col].y:  # type: ignore
                bottom_per_col[col] = alien

        return [a for a in bottom_per_col if a is not None]

    def _update_ufo(self) -> None:
        if self.ufo is not None:
            self.ufo.update()
            if self.ufo.x < -100 or self.ufo.x > self.width + 100:
                self.ufo = None

    def _handle_collisions(self) -> float:
        assert self.ship is not None
        reward = 0.0

        # Player bullets vs shields
        for bullet in self.player_bullets:
            if not bullet.alive:
                continue
            for shield in self.shields:
                if shield.check_collision(bullet.rect):
                    bullet.alive = False
                    # Small particle effect when hitting shield
                    self._spawn_explosion(
                        int(bullet.x),
                        int(bullet.y),
                        self.config.SI_COLOR_SHIELD,
                        count=5,
                    )
                    break

        # Player bullets vs aliens
        for bullet in self.player_bullets:
            if not bullet.alive:
                continue
            for idx, alien in enumerate(self.aliens):
                if not alien.alive:
                    continue
                alien_rect = pygame.Rect(
                    alien.x + self.alien_x_offset, alien.y, alien.width, alien.height
                )
                if bullet.rect.colliderect(alien_rect):
                    bullet.alive = False
                    alien.alive = False
                    self._aliens_remaining -= 1
                    self.total_aliens_killed += 1
                    self._alien_states[idx] = 0.0

                    # Authentic arcade scoring:
                    # Type 0 (squid, top row): 30 points
                    # Type 1 (crab, middle rows): 20 points
                    # Type 2 (octopus, bottom rows): 10 points
                    points = alien_points(alien.alien_type)
                    self.score += points
                    reward += self.config.SI_REWARD_ALIEN_HIT

                    # Track accuracy
                    self._shots_hit += 1

                    # Visual effects
                    self._spawn_explosion(
                        alien_rect.centerx, alien_rect.centery, alien.color, count=20
                    )

                    # Score popup
                    if not self.headless:
                        self.score_popups.append(
                            ScorePopup(
                                alien_rect.centerx,
                                alien_rect.centery,
                                points,
                                alien.color,
                            )
                        )

                    self.screen_shake = 3
                    self.flash_alpha = 30

                    # Speed up as aliens are destroyed (classic behavior)
                    self.alien_speed = alien_speed_after_kill(
                        self.alien_base_speed,
                        self._num_aliens,
                        self._aliens_remaining,
                    )
                    break

        # Player bullets vs UFO
        if self.ufo is not None and self.ufo.alive:
            for bullet in self.player_bullets:
                if not bullet.alive:
                    continue
                if bullet.rect.colliderect(self.ufo.rect):
                    bullet.alive = False
                    self._spawn_explosion(
                        self.ufo.rect.centerx,
                        self.ufo.rect.centery,
                        self.ufo.color,
                        count=30,
                    )
                    self.ufo.alive = False

                    # Random UFO points like original arcade (50, 100, 150, or 300)
                    ufo_points = random.choice([50, 100, 100, 150, 150, 300])
                    self.score += ufo_points
                    reward += self.config.SI_REWARD_UFO_HIT

                    # Score popup for UFO
                    if not self.headless:
                        self.score_popups.append(
                            ScorePopup(
                                self.ufo.rect.centerx,
                                self.ufo.rect.centery,
                                ufo_points,
                                (255, 255, 100),  # Yellow for UFO
                            )
                        )

                    self.screen_shake = 8
                    self.flash_alpha = 60
                    self.ufo = None
                    break

        # Alien bullets vs shields
        for bullet in self.alien_bullets:
            if not bullet.alive:
                continue
            for shield in self.shields:
                if shield.check_collision(bullet.rect):
                    bullet.alive = False
                    self._spawn_explosion(int(bullet.x), int(bullet.y), (255, 100, 50), count=5)
                    break

        # Alien bullets vs player (skip if invincible)
        if not self.player_invincible:
            for bullet in self.alien_bullets:
                if not bullet.alive:
                    continue
                if bullet.rect.colliderect(self.ship.rect):
                    bullet.alive = False

                    # Big death explosion
                    self._spawn_player_death_explosion()

                    self.lives -= 1
                    reward += self.config.SI_REWARD_PLAYER_DEATH
                    self.screen_shake = 15
                    self.flash_alpha = 120

                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        # Grant invincibility after death
                        self.player_invincible = True
                        self.invincibility_timer = self.invincibility_duration
                    break

        # Aliens colliding with shields (destroy shield blocks)
        for alien in self.aliens:
            if not alien.alive:
                continue
            alien_rect = pygame.Rect(
                alien.x + self.alien_x_offset, alien.y, alien.width, alien.height
            )
            for shield in self.shields:
                for block in shield.blocks:
                    if block.alive and alien_rect.colliderect(block.rect):
                        block.alive = False  # Aliens destroy shields on contact

        return reward

    def get_state(self) -> np.ndarray:
        assert self.ship is not None

        idx = 0
        self._state_array[idx] = self.ship.x * self._inv_width
        idx += 1

        for i in range(self._max_player_bullets):
            if i < len(self.player_bullets):
                b = self.player_bullets[i]
                self._state_array[idx] = b.x * self._inv_width
                self._state_array[idx + 1] = b.y * self._inv_height
            else:
                self._state_array[idx] = 0.5
                self._state_array[idx + 1] = 0.0
            idx += 2

        self._state_array[idx : idx + self._num_aliens] = self._alien_states
        idx += self._num_aliens

        max_offset = self.width * 0.3
        self._state_array[idx] = (self.alien_x_offset / max_offset + 1) * 0.5
        idx += 1

        self._state_array[idx] = (self.alien_direction + 1) * 0.5
        idx += 1

        # Lowest alien Y position (how close invasion is)
        lowest_alien_y = 0
        for alien in self.aliens:
            if alien.alive:
                lowest_alien_y = max(lowest_alien_y, alien.y + alien.height)
        self._state_array[idx] = lowest_alien_y * self._inv_height
        idx += 1

        # IMPROVED: Track multiple alien bullets (top N nearest to ship)
        # Use heapq.nsmallest for O(n log k) instead of O(n log n) sorting
        ship_y = self.ship.y
        ship_center_x = self.ship.x + self.ship.width // 2

        alive_bullets = heapq.nsmallest(
            self._max_tracked_alien_bullets,
            [(ship_y - b.y, b) for b in self.alien_bullets if b.alive and b.y < ship_y],
            key=lambda x: x[0],
        )

        for i in range(self._max_tracked_alien_bullets):
            if i < len(alive_bullets):
                bullet = alive_bullets[i][1]  # bullet is now at index 1
                # Normalized X position relative to ship (clamped to [0, 1] range)
                self._state_array[idx] = np.clip(
                    (bullet.x - ship_center_x) * self._inv_width + 0.5, 0.0, 1.0
                )
                # Normalized Y position (0 = at ship level, 1 = top of screen)
                self._state_array[idx + 1] = bullet.y * self._inv_height
            else:
                # No bullet in this slot - use neutral values
                self._state_array[idx] = 0.5  # Centered
                self._state_array[idx + 1] = 0.0  # Off screen (no threat)
            idx += 2

        # Shoot cooldown (helps agent time shots)
        self._state_array[idx] = self._shoot_cooldown / self._shoot_cooldown_max
        idx += 1

        # Active alien bullets count (threat level)
        active_alien_bullets = sum(1 for b in self.alien_bullets if b.alive)
        self._state_array[idx] = min(active_alien_bullets / 10.0, 1.0)
        idx += 1

        # Aliens remaining ratio (progress indicator)
        self._state_array[idx] = alien_pressure_ratio(self._aliens_remaining, self._num_aliens)
        idx += 1

        # Lives remaining (risk awareness)
        # Bug 78 fix: Guard against division by zero if config.LIVES = 0
        self._state_array[idx] = self.lives / self.config.LIVES if self.config.LIVES > 0 else 0.0
        idx += 1

        # Level (difficulty awareness)
        self._state_array[idx] = min(self.level / 10.0, 1.0)
        idx += 1

        assert (
            idx == self._state_size
        ), f"Space Invaders state writer filled {idx} values, expected {self._state_size}"

        return self._state_array.copy()

    def _get_info(self) -> dict:
        return {
            "score": self.score,
            "lives": self.lives,
            "aliens_remaining": self._aliens_remaining,
            "won": self.won,
            "level": self.level,
            "total_aliens_killed": self.total_aliens_killed,
            "bricks": self.total_aliens_killed,  # Compatibility with training metrics
        }

    def close(self) -> None:
        pass

    def seed(self, seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)


from src.game.space_invaders_vec import VecSpaceInvaders

if __name__ == "__main__":
    pygame.init()
    config = Config()
    config.GAME_NAME = "space_invaders"
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Invaders - Test")
    clock = pygame.time.Clock()

    game = SpaceInvaders(config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = 1
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_SPACE]:
            action = 3

        state, reward, done, info = game.step(action)

        if done:
            game.reset()

        game.render(screen)
        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()
