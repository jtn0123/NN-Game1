"""
Space Invaders Game Implementation
===================================

A classic Space Invaders clone designed for AI training.

Key Features:
- Clean state representation for neural network input
- Configurable rewards for reinforcement learning
- CRT phosphor aesthetic with scanlines and glow effects
- Full alien marching, shooting, and UFO bonus mechanics

Game Rules:
- Player controls a ship at the bottom
- Aliens march left/right and descend
- Goal: Destroy all aliens before they reach the bottom
- Player has 3 lives
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional
import math
import random

from .base_game import BaseGame
import sys
sys.path.append('..')
from config import Config


class Bullet:
    """A bullet (player or alien)."""
    
    def __init__(self, x: float, y: float, speed: float, is_player: bool, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.speed = speed  # Positive = down, Negative = up
        self.is_player = is_player
        self.color = color
        self.alive = True
        self.width = 4
        self.height = 15
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.width // 2),
            int(self.y - self.height // 2),
            self.width,
            self.height
        )
    
    def update(self) -> None:
        """Move the bullet."""
        self.y += self.speed
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the bullet with glow effect."""
        if not self.alive:
            return
        
        # Glow
        glow_rect = self.rect.inflate(4, 4)
        glow_color = tuple(min(255, c + 50) for c in self.color)
        pygame.draw.rect(screen, glow_color, glow_rect, border_radius=2)
        
        # Main bullet
        pygame.draw.rect(screen, self.color, self.rect, border_radius=2)


class Alien:
    """An alien invader."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 alien_type: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.alien_type = alien_type  # 0, 1, or 2 for different sprites
        self.color = color
        self.alive = True
        self.animation_frame = 0
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        """Draw the alien with simple pixel art style."""
        if not self.alive:
            return
        
        rect = self.rect
        
        # Animate - subtle pulsing
        pulse = 1.0 + 0.05 * math.sin(time * 4 + self.x * 0.1)
        
        # Draw alien body (simple geometric representation)
        body_color = self.color
        
        # Main body
        body_rect = pygame.Rect(
            rect.x + rect.width // 4,
            rect.y + rect.height // 4,
            rect.width // 2,
            rect.height // 2
        )
        pygame.draw.rect(screen, body_color, body_rect, border_radius=4)
        
        # "Eyes" or features based on type
        if self.alien_type == 0:
            # Top row aliens - antennae
            pygame.draw.line(screen, body_color, 
                           (rect.centerx - 8, rect.y + 5),
                           (rect.centerx - 8, rect.y - 3), 2)
            pygame.draw.line(screen, body_color,
                           (rect.centerx + 8, rect.y + 5),
                           (rect.centerx + 8, rect.y - 3), 2)
        elif self.alien_type == 1:
            # Middle row - side wings
            pygame.draw.rect(screen, body_color,
                           (rect.x + 2, rect.centery - 3, 8, 6), border_radius=2)
            pygame.draw.rect(screen, body_color,
                           (rect.right - 10, rect.centery - 3, 8, 6), border_radius=2)
        else:
            # Bottom row - legs
            pygame.draw.line(screen, body_color,
                           (rect.x + 8, rect.bottom - 5),
                           (rect.x + 3, rect.bottom + 2), 2)
            pygame.draw.line(screen, body_color,
                           (rect.right - 8, rect.bottom - 5),
                           (rect.right - 3, rect.bottom + 2), 2)
        
        # Glow effect
        glow_alpha = int(30 * pulse)
        glow_surface = pygame.Surface((rect.width + 10, rect.height + 10), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*body_color, glow_alpha), 
                        glow_surface.get_rect(), border_radius=6)
        screen.blit(glow_surface, (rect.x - 5, rect.y - 5))


class Ship:
    """The player's ship."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 speed: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def move(self, direction: int, screen_width: int) -> None:
        """Move ship horizontally."""
        self.x += direction * self.speed
        self.x = max(0, min(self.x, screen_width - self.width))
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        """Draw the ship with engine glow."""
        rect = self.rect
        
        # Engine glow (pulsing)
        glow_intensity = int(150 + 50 * math.sin(time * 8))
        glow_color = (0, glow_intensity, glow_intensity // 2)
        glow_rect = pygame.Rect(rect.centerx - 8, rect.bottom - 5, 16, 10)
        pygame.draw.ellipse(screen, glow_color, glow_rect)
        
        # Ship body (triangle-ish shape)
        points = [
            (rect.centerx, rect.top),  # Nose
            (rect.left + 5, rect.bottom - 5),  # Left
            (rect.centerx, rect.bottom - 10),  # Center notch
            (rect.right - 5, rect.bottom - 5),  # Right
        ]
        pygame.draw.polygon(screen, self.color, points)
        
        # Cockpit
        cockpit_rect = pygame.Rect(rect.centerx - 5, rect.top + 8, 10, 8)
        pygame.draw.ellipse(screen, (200, 255, 200), cockpit_rect)
        
        # Ship outline glow
        glow_surface = pygame.Surface((rect.width + 10, rect.height + 10), pygame.SRCALPHA)
        glow_points = [(p[0] - rect.x + 5, p[1] - rect.y + 5) for p in points]
        pygame.draw.polygon(glow_surface, (*self.color, 50), glow_points)
        screen.blit(glow_surface, (rect.x - 5, rect.y - 5))


class UFO:
    """Bonus UFO that flies across the top."""
    
    def __init__(self, x: int, y: int, speed: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.speed = speed  # Can be negative (left) or positive (right)
        self.color = color
        self.width = 50
        self.height = 20
        self.alive = True
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), self.y, self.width, self.height)
    
    def update(self) -> None:
        """Move the UFO."""
        self.x += self.speed
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        """Draw the UFO with blinking lights."""
        if not self.alive:
            return
        
        rect = self.rect
        
        # UFO body (ellipse)
        pygame.draw.ellipse(screen, self.color, rect)
        
        # Dome
        dome_rect = pygame.Rect(rect.centerx - 12, rect.top - 8, 24, 16)
        pygame.draw.ellipse(screen, (255, 100, 100), dome_rect)
        
        # Blinking lights
        for i, offset in enumerate([-15, 0, 15]):
            blink = (time * 5 + i) % 1 > 0.5
            light_color = (255, 255, 0) if blink else (100, 100, 0)
            pygame.draw.circle(screen, light_color, 
                             (rect.centerx + offset, rect.centery), 3)


class SpaceInvaders(BaseGame):
    """
    Space Invaders game implementation.
    
    State representation (normalized to [0, 1]):
        - ship_x: Ship X position
        - ship_bullets: Positions of player bullets (flattened)
        - alien_grid: Flattened array of alien alive states
        - alien_x_offset: Current alien formation X offset
        - alien_direction: Current movement direction (-1 or 1, normalized)
        - nearest_alien_bullet_y: Y position of closest enemy bullet
    
    Actions:
        0 = Move LEFT
        1 = STAY (no movement)
        2 = Move RIGHT
        3 = SHOOT
    """
    
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
        
        # Alien movement
        self.alien_direction = 1  # 1 = right, -1 = left
        self.alien_speed = self.config.SI_ALIEN_SPEED_X
        self.alien_x_offset = 0.0
        
        # Game state
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        self.level = 1
        
        # For state representation
        self._num_aliens = self.config.SI_ALIEN_ROWS * self.config.SI_ALIEN_COLS
        
        # Track aliens remaining
        self._aliens_remaining = self._num_aliens
        
        # Pre-allocated arrays for get_state()
        # State: ship_x(1) + ship_bullets(3*2) + alien_grid(55) + alien_offset(1) + alien_dir(1) + nearest_bullet(1) = 64
        self._max_player_bullets = self.config.SI_MAX_PLAYER_BULLETS
        self._state_size = 1 + self._max_player_bullets * 2 + self._num_aliens + 3
        self._state_array = np.zeros(self._state_size, dtype=np.float32)
        self._alien_states = np.ones(self._num_aliens, dtype=np.float32)
        
        # Pre-computed normalization constants
        self._inv_width = 1.0 / self.width
        self._inv_height = 1.0 / self.height
        
        # Animation time
        self._time = 0.0
        
        # Shoot cooldown
        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 10  # Frames between shots
        
        # Background scanlines (for CRT effect)
        if not headless:
            self._scanline_surface = self._create_scanlines()
        else:
            self._scanline_surface = None
        
        # Initialize the game
        self.reset()
    
    def _create_scanlines(self) -> pygame.Surface:
        """Create a scanline overlay for CRT effect."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(0, self.height, 3):
            pygame.draw.line(surface, (0, 0, 0, 30), (0, y), (self.width, y))
        return surface
    
    @property
    def state_size(self) -> int:
        """State vector dimension."""
        return self._state_size
    
    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return 4  # LEFT, STAY, RIGHT, SHOOT
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        
        # Create ship
        ship_x = (self.width - self.config.SI_SHIP_WIDTH) // 2
        ship_y = self.height - self.config.SI_SHIP_Y_OFFSET - self.config.SI_SHIP_HEIGHT
        self.ship = Ship(
            ship_x, ship_y,
            self.config.SI_SHIP_WIDTH,
            self.config.SI_SHIP_HEIGHT,
            self.config.SI_SHIP_SPEED,
            self.config.SI_COLOR_SHIP
        )
        
        # Create aliens
        self._create_aliens()
        
        # Clear bullets and UFO
        self.player_bullets = []
        self.alien_bullets = []
        self.ufo = None
        
        # Reset alien movement
        self.alien_direction = 1
        self.alien_speed = self.config.SI_ALIEN_SPEED_X
        self.alien_x_offset = 0.0
        
        # Reset tracking
        self._aliens_remaining = self._num_aliens
        self._alien_states.fill(1.0)
        self._shoot_cooldown = 0
        
        return self.get_state()
    
    def _create_aliens(self) -> None:
        """Create the alien grid."""
        self.aliens = []
        
        colors = [
            self.config.SI_COLOR_ALIEN_1,
            self.config.SI_COLOR_ALIEN_2,
            self.config.SI_COLOR_ALIEN_3,
        ]
        
        for row in range(self.config.SI_ALIEN_ROWS):
            alien_type = min(row, 2)  # Type 0, 1, or 2
            color = colors[alien_type % len(colors)]
            
            for col in range(self.config.SI_ALIEN_COLS):
                x = self.config.SI_ALIEN_OFFSET_LEFT + col * (self.config.SI_ALIEN_WIDTH + self.config.SI_ALIEN_PADDING)
                y = self.config.SI_ALIEN_OFFSET_TOP + row * (self.config.SI_ALIEN_HEIGHT + self.config.SI_ALIEN_PADDING)
                
                alien = Alien(x, y, self.config.SI_ALIEN_WIDTH, self.config.SI_ALIEN_HEIGHT, alien_type, color)
                self.aliens.append(alien)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step."""
        if self.game_over or self.won:
            return self.get_state(), 0.0, True, self._get_info()
        
        assert self.ship is not None
        
        self._time += 1.0 / 60.0
        reward = self.config.SI_REWARD_STEP
        
        # Update shoot cooldown
        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1
        
        # Handle action
        if action == 0:  # LEFT
            self.ship.move(-1, self.width)
        elif action == 2:  # RIGHT
            self.ship.move(1, self.width)
        elif action == 3:  # SHOOT
            if self._shoot_cooldown == 0 and len(self.player_bullets) < self._max_player_bullets:
                self._fire_player_bullet()
                reward += self.config.SI_REWARD_SHOOT
                self._shoot_cooldown = self._shoot_cooldown_max
        # action == 1 is STAY, do nothing
        
        # Update bullets
        self._update_bullets()
        
        # Update aliens
        reward += self._update_aliens()
        
        # Update UFO
        self._update_ufo()
        
        # Check collisions
        reward += self._handle_collisions()
        
        # Check win condition
        if self._aliens_remaining == 0:
            self.won = True
            self.game_over = True
            reward += self.config.SI_REWARD_LEVEL_CLEAR
        
        # Check if aliens reached the bottom
        for alien in self.aliens:
            if alien.alive and alien.y + alien.height >= self.ship.y:
                self.game_over = True
                reward += self.config.SI_REWARD_PLAYER_DEATH
                break
        
        return self.get_state(), reward, self.game_over, self._get_info()
    
    def _fire_player_bullet(self) -> None:
        """Fire a bullet from the player's ship."""
        assert self.ship is not None
        bullet = Bullet(
            self.ship.x + self.ship.width // 2,
            self.ship.y - 5,
            -self.config.SI_BULLET_SPEED,
            True,
            self.config.SI_COLOR_BULLET
        )
        self.player_bullets.append(bullet)
    
    def _update_bullets(self) -> None:
        """Update all bullets and remove dead ones."""
        # Update player bullets
        for bullet in self.player_bullets:
            bullet.update()
            if bullet.y < -10:
                bullet.alive = False
        self.player_bullets = [b for b in self.player_bullets if b.alive]
        
        # Update alien bullets
        for bullet in self.alien_bullets:
            bullet.update()
            if bullet.y > self.height + 10:
                bullet.alive = False
        self.alien_bullets = [b for b in self.alien_bullets if b.alive]
    
    def _update_aliens(self) -> float:
        """Update alien movement and shooting. Returns reward."""
        reward = 0.0
        
        # Move aliens
        self.alien_x_offset += self.alien_direction * self.alien_speed
        
        # Check if aliens hit edge
        for alien in self.aliens:
            if not alien.alive:
                continue
            actual_x = alien.x + self.alien_x_offset
            if actual_x <= 0 or actual_x + alien.width >= self.width:
                # Reverse direction and move down
                self.alien_direction *= -1
                for a in self.aliens:
                    if a.alive:
                        a.y += self.config.SI_ALIEN_SPEED_Y
                break
        
        # Alien shooting
        alive_aliens = [a for a in self.aliens if a.alive]
        for alien in alive_aliens:
            if random.random() < self.config.SI_ALIEN_SHOOT_CHANCE:
                actual_x = alien.x + self.alien_x_offset + alien.width // 2
                bullet = Bullet(
                    actual_x,
                    alien.y + alien.height,
                    self.config.SI_ALIEN_BULLET_SPEED,
                    False,
                    (255, 100, 100)
                )
                self.alien_bullets.append(bullet)
        
        # Spawn UFO
        if self.ufo is None and random.random() < self.config.SI_UFO_CHANCE:
            direction = random.choice([-1, 1])
            x = -50 if direction > 0 else self.width + 50
            self.ufo = UFO(x, 40, direction * self.config.SI_UFO_SPEED, self.config.SI_COLOR_UFO)
        
        return reward
    
    def _update_ufo(self) -> None:
        """Update UFO movement."""
        if self.ufo is not None:
            self.ufo.update()
            if self.ufo.x < -100 or self.ufo.x > self.width + 100:
                self.ufo = None
    
    def _handle_collisions(self) -> float:
        """Handle all collisions. Returns reward."""
        assert self.ship is not None
        reward = 0.0
        
        # Player bullets vs aliens
        for bullet in self.player_bullets:
            if not bullet.alive:
                continue
            for idx, alien in enumerate(self.aliens):
                if not alien.alive:
                    continue
                # Account for alien x offset
                alien_rect = pygame.Rect(
                    alien.x + self.alien_x_offset,
                    alien.y,
                    alien.width,
                    alien.height
                )
                if bullet.rect.colliderect(alien_rect):
                    bullet.alive = False
                    alien.alive = False
                    self._aliens_remaining -= 1
                    self._alien_states[idx] = 0.0
                    self.score += 10 + alien.alien_type * 10
                    reward += self.config.SI_REWARD_ALIEN_HIT
                    
                    # Speed up as fewer aliens remain
                    self.alien_speed = self.config.SI_ALIEN_SPEED_X * (1 + (self._num_aliens - self._aliens_remaining) * 0.02)
                    break
        
        # Player bullets vs UFO
        if self.ufo is not None and self.ufo.alive:
            for bullet in self.player_bullets:
                if not bullet.alive:
                    continue
                if bullet.rect.colliderect(self.ufo.rect):
                    bullet.alive = False
                    self.ufo.alive = False
                    self.score += self.config.SI_UFO_POINTS
                    reward += self.config.SI_REWARD_UFO_HIT
                    self.ufo = None
                    break
        
        # Alien bullets vs player
        for bullet in self.alien_bullets:
            if not bullet.alive:
                continue
            if bullet.rect.colliderect(self.ship.rect):
                bullet.alive = False
                self.lives -= 1
                reward += self.config.SI_REWARD_PLAYER_DEATH
                
                if self.lives <= 0:
                    self.game_over = True
                break
        
        return reward
    
    def get_state(self) -> np.ndarray:
        """Get the current game state as a normalized vector."""
        assert self.ship is not None
        
        idx = 0
        
        # Ship x position
        self._state_array[idx] = self.ship.x * self._inv_width
        idx += 1
        
        # Player bullet positions (x, y for each)
        for i in range(self._max_player_bullets):
            if i < len(self.player_bullets):
                b = self.player_bullets[i]
                self._state_array[idx] = b.x * self._inv_width
                self._state_array[idx + 1] = b.y * self._inv_height
            else:
                self._state_array[idx] = 0.5
                self._state_array[idx + 1] = 0.0
            idx += 2
        
        # Alien states
        self._state_array[idx:idx + self._num_aliens] = self._alien_states
        idx += self._num_aliens
        
        # Alien formation x offset (normalized)
        max_offset = self.width * 0.3
        self._state_array[idx] = (self.alien_x_offset / max_offset + 1) * 0.5
        idx += 1
        
        # Alien direction
        self._state_array[idx] = (self.alien_direction + 1) * 0.5  # 0 or 1
        idx += 1
        
        # Nearest alien bullet y position
        if self.alien_bullets:
            nearest_y = max(b.y for b in self.alien_bullets if b.alive) if any(b.alive for b in self.alien_bullets) else 0
            self._state_array[idx] = nearest_y * self._inv_height
        else:
            self._state_array[idx] = 0.0
        
        return self._state_array.copy()
    
    def _get_info(self) -> dict:
        """Get additional game information."""
        return {
            'score': self.score,
            'lives': self.lives,
            'aliens_remaining': self._aliens_remaining,
            'won': self.won,
            'level': self.level
        }
    
    def render(self, screen: pygame.Surface) -> None:
        """Render the game to a pygame screen."""
        if self.headless:
            return
        
        assert self.ship is not None
        
        # CRT-style dark green background
        screen.fill(self.config.SI_COLOR_BACKGROUND)
        
        # Draw aliens
        for alien in self.aliens:
            if alien.alive:
                # Offset for movement
                original_x = alien.x
                alien.x += int(self.alien_x_offset)
                alien.draw(screen, self._time)
                alien.x = original_x
        
        # Draw UFO
        if self.ufo is not None and self.ufo.alive:
            self.ufo.draw(screen, self._time)
        
        # Draw bullets
        for bullet in self.player_bullets:
            bullet.draw(screen)
        for bullet in self.alien_bullets:
            bullet.draw(screen)
        
        # Draw ship
        self.ship.draw(screen, self._time)
        
        # Draw scanlines (CRT effect)
        if self._scanline_surface:
            screen.blit(self._scanline_surface, (0, 0))
        
        # Draw HUD
        self._draw_hud(screen)
    
    def _draw_hud(self, screen: pygame.Surface) -> None:
        """Draw heads-up display."""
        font = pygame.font.Font(None, 36)
        
        # Score (top left) - green CRT style
        score_text = font.render(f"SCORE: {self.score}", True, (0, 255, 100))
        screen.blit(score_text, (10, 10))
        
        # Lives (top right)
        lives_text = font.render(f"LIVES: {self.lives}", True, (0, 255, 100))
        screen.blit(lives_text, (self.width - 120, 10))
        
        # Game over or win message
        if self.game_over:
            big_font = pygame.font.Font(None, 72)
            if self.won:
                msg = "VICTORY!"
                color = (0, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            text = big_font.render(msg, True, color)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            
            # Draw background box
            box_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(screen, (0, 0, 0, 180), box_rect)
            pygame.draw.rect(screen, color, box_rect, 3)
            
            screen.blit(text, text_rect)
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)


# Test the game
if __name__ == "__main__":
    pygame.init()
    config = Config()
    config.GAME_NAME = 'space_invaders'
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Space Invaders - Test")
    clock = pygame.time.Clock()
    
    game = SpaceInvaders(config)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        action = 1  # STAY
        if keys[pygame.K_LEFT]:
            action = 0  # LEFT
        elif keys[pygame.K_RIGHT]:
            action = 2  # RIGHT
        elif keys[pygame.K_SPACE]:
            action = 3  # SHOOT
        
        # Step game
        state, reward, done, info = game.step(action)
        
        if done:
            game.reset()
        
        # Render
        game.render(screen)
        pygame.display.flip()
        clock.tick(config.FPS)
    
    pygame.quit()

