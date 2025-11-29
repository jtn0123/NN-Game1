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

from .base_game import BaseGame
import sys
sys.path.append('..')
from config import Config


# Classic Space Invader pixel patterns (8x8)
ALIEN_PATTERNS = {
    0: [  # Squid (top row)
        [
            "   XX   ",
            "  XXXX  ",
            " XXXXXX ",
            "XX XX XX",
            "XXXXXXXX",
            "  X  X  ",
            " X    X ",
            "X      X",
        ],
        [
            "   XX   ",
            "  XXXX  ",
            " XXXXXX ",
            "XX XX XX",
            "XXXXXXXX",
            " X XX X ",
            "X      X",
            " X    X ",
        ]
    ],
    1: [  # Crab (middle rows)
        [
            "  X   X ",
            "   X X  ",
            "  XXXXX ",
            " XX XXX ",
            "XXXXXXXX",
            "X XXXXX X",
            "X X   X X",
            "   XX XX ",
        ],
        [
            "  X   X ",
            "X  X X  X",
            "X XXXXX X",
            "XXX XXX X",
            "XXXXXXXX",
            " XXXXXXX",
            "  X   X ",
            " X     X ",
        ]
    ],
    2: [  # Octopus (bottom rows)
        [
            "   XX   ",
            "  XXXX  ",
            " XXXXXX ",
            "XX XX XX",
            "XXXXXXXX",
            "  X  X  ",
            " X XX X ",
            "XX    XX",
        ],
        [
            "   XX   ",
            "  XXXX  ",
            " XXXXXX ",
            "XX XX XX",
            "XXXXXXXX",
            "  X  X  ",
            " X XX X ",
            "  X  X  ",
        ]
    ]
}


class Particle:
    """A visual particle for explosions."""
    
    def __init__(self, x: float, y: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.05)
        self.size = random.randint(2, 5)
    
    def update(self) -> bool:
        """Update particle. Returns True if still alive."""
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity
        self.life -= self.decay
        return self.life > 0
    
    def draw(self, screen: pygame.Surface) -> None:
        if self.life <= 0:
            return
        alpha = int(255 * self.life)
        color = tuple(min(255, int(c * self.life)) for c in self.color)
        size = int(self.size * self.life)
        if size > 0:
            pygame.draw.rect(screen, color, (int(self.x), int(self.y), size, size))


class Star:
    """A background star."""
    
    def __init__(self, x: int, y: int, speed: float, brightness: int):
        self.x = x
        self.y = y
        self.speed = speed
        self.brightness = brightness
        self.twinkle_phase = random.uniform(0, math.pi * 2)
    
    def update(self, height: int) -> None:
        self.y += self.speed
        if self.y > height:
            self.y = 0
            self.x = random.randint(0, 800)
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        twinkle = 0.7 + 0.3 * math.sin(time * 3 + self.twinkle_phase)
        brightness = int(self.brightness * twinkle)
        # Slight green tint for CRT feel
        color = (brightness - 20, brightness, brightness - 10)
        color = tuple(max(0, min(255, c)) for c in color)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 1)


class Bullet:
    """A bullet (player or alien)."""
    
    def __init__(self, x: float, y: float, speed: float, is_player: bool, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.speed = speed
        self.is_player = is_player
        self.color = color
        self.alive = True
        self.width = 3
        self.height = 12 if is_player else 10
        self.trail = []  # Trail positions for effect
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.width // 2),
            int(self.y - self.height // 2),
            self.width,
            self.height
        )
    
    def update(self) -> None:
        # Store trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > 5:
            self.trail.pop(0)
        
        self.y += self.speed
    
    def draw(self, screen: pygame.Surface) -> None:
        if not self.alive:
            return
        
        # Draw trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) * 0.5
            trail_color = tuple(int(c * alpha) for c in self.color)
            trail_rect = pygame.Rect(int(tx - 1), int(ty - self.height // 4), 2, self.height // 2)
            pygame.draw.rect(screen, trail_color, trail_rect)
        
        # Glow effect
        glow_surf = pygame.Surface((self.width + 8, self.height + 8), pygame.SRCALPHA)
        glow_color = (*self.color, 60)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=2)
        screen.blit(glow_surf, (self.rect.x - 4, self.rect.y - 4))
        
        # Main bullet
        pygame.draw.rect(screen, self.color, self.rect, border_radius=1)
        
        # Bright center
        center_rect = self.rect.inflate(-2, -4)
        bright_color = tuple(min(255, c + 100) for c in self.color)
        pygame.draw.rect(screen, bright_color, center_rect)


class Alien:
    """An alien invader with pixel art sprite."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 alien_type: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.alien_type = alien_type
        self.color = color
        self.alive = True
        self.animation_frame = 0
        self._pattern = ALIEN_PATTERNS.get(alien_type, ALIEN_PATTERNS[0])
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        if not self.alive:
            return
        
        rect = self.rect
        
        # Animation frame based on time
        frame = int(time * 2) % 2
        pattern = self._pattern[frame]
        
        # Calculate pixel size
        pixel_w = self.width // 8
        pixel_h = self.height // 8
        
        # Glow effect underneath
        glow_surf = pygame.Surface((self.width + 10, self.height + 10), pygame.SRCALPHA)
        glow_color = (*self.color, 40)
        pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
        screen.blit(glow_surf, (rect.x - 5, rect.y - 5))
        
        # Draw pixel art pattern
        for row_idx, row in enumerate(pattern):
            for col_idx, char in enumerate(row):
                if char == 'X':
                    px = rect.x + col_idx * pixel_w
                    py = rect.y + row_idx * pixel_h
                    pygame.draw.rect(screen, self.color, (px, py, pixel_w, pixel_h))
        
        # Subtle highlight on top pixels
        highlight = tuple(min(255, c + 60) for c in self.color)
        for col_idx, char in enumerate(pattern[0]):
            if char == 'X':
                px = rect.x + col_idx * pixel_w
                py = rect.y
                pygame.draw.rect(screen, highlight, (px, py, pixel_w, 1))


class Ship:
    """The player's ship with detailed design."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 speed: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
        self.thrust_phase = 0.0
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def move(self, direction: int, screen_width: int) -> None:
        self.x += direction * self.speed
        self.x = max(0, min(self.x, screen_width - self.width))
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        rect = self.rect
        self.thrust_phase += 0.3
        
        # Engine glow (animated)
        thrust_intensity = 0.7 + 0.3 * math.sin(self.thrust_phase * 5)
        
        # Main thruster glow
        for i in range(3):
            glow_height = int((15 - i * 4) * thrust_intensity)
            glow_width = 8 - i * 2
            glow_color = (
                int(50 + i * 30),
                int((200 - i * 50) * thrust_intensity),
                int((255 - i * 40) * thrust_intensity)
            )
            glow_rect = pygame.Rect(
                rect.centerx - glow_width // 2,
                rect.bottom - 3 + i * 2,
                glow_width,
                glow_height
            )
            pygame.draw.ellipse(screen, glow_color, glow_rect)
        
        # Ship body - layered design
        # Base hull
        hull_points = [
            (rect.centerx, rect.top),  # Nose tip
            (rect.left + 3, rect.bottom - 8),  # Left wing inner
            (rect.left - 5, rect.bottom - 3),  # Left wing tip
            (rect.left + 8, rect.bottom),  # Left base
            (rect.right - 8, rect.bottom),  # Right base
            (rect.right + 5, rect.bottom - 3),  # Right wing tip
            (rect.right - 3, rect.bottom - 8),  # Right wing inner
        ]
        pygame.draw.polygon(screen, self.color, hull_points)
        
        # Hull detail lines
        detail_color = tuple(max(0, c - 40) for c in self.color)
        pygame.draw.line(screen, detail_color, 
                        (rect.centerx, rect.top + 5), 
                        (rect.centerx, rect.bottom - 5), 2)
        pygame.draw.line(screen, detail_color,
                        (rect.centerx - 10, rect.centery),
                        (rect.centerx + 10, rect.centery), 1)
        
        # Cockpit (glowing canopy)
        cockpit_points = [
            (rect.centerx, rect.top + 4),
            (rect.centerx - 6, rect.top + 16),
            (rect.centerx + 6, rect.top + 16),
        ]
        # Cockpit glow
        glow_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surf, (100, 255, 200, 80), 
                          [(10, 0), (4, 12), (16, 12)])
        screen.blit(glow_surf, (rect.centerx - 10, rect.top + 2))
        
        # Cockpit glass
        pygame.draw.polygon(screen, (150, 255, 220), cockpit_points)
        # Glass highlight
        pygame.draw.line(screen, (200, 255, 240),
                        (rect.centerx - 2, rect.top + 6),
                        (rect.centerx - 4, rect.top + 12), 1)
        
        # Wing highlights
        highlight = tuple(min(255, c + 40) for c in self.color)
        pygame.draw.line(screen, highlight,
                        (rect.left + 5, rect.bottom - 6),
                        (rect.left - 3, rect.bottom - 3), 2)
        pygame.draw.line(screen, highlight,
                        (rect.right - 5, rect.bottom - 6),
                        (rect.right + 3, rect.bottom - 3), 2)
        
        # Ship glow
        glow_surf = pygame.Surface((self.width + 20, self.height + 20), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surf, (*self.color, 30),
                          [(p[0] - rect.x + 10, p[1] - rect.y + 10) for p in hull_points])
        screen.blit(glow_surf, (rect.x - 10, rect.y - 10))


class UFO:
    """Bonus UFO that flies across the top."""
    
    def __init__(self, x: int, y: int, speed: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.speed = speed
        self.color = color
        self.width = 50
        self.height = 20
        self.alive = True
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), self.y, self.width, self.height)
    
    def update(self) -> None:
        self.x += self.speed
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        if not self.alive:
            return
        
        rect = self.rect
        
        # UFO glow
        glow_surf = pygame.Surface((self.width + 20, self.height + 20), pygame.SRCALPHA)
        pulse = 0.7 + 0.3 * math.sin(time * 8)
        glow_color = (*self.color, int(60 * pulse))
        pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
        screen.blit(glow_surf, (rect.x - 10, rect.y - 10))
        
        # UFO body
        pygame.draw.ellipse(screen, self.color, rect)
        
        # Dome (brighter)
        dome_color = tuple(min(255, c + 50) for c in self.color)
        dome_rect = pygame.Rect(rect.centerx - 12, rect.top - 8, 24, 14)
        pygame.draw.ellipse(screen, dome_color, dome_rect)
        
        # Dome highlight
        highlight_rect = pygame.Rect(rect.centerx - 6, rect.top - 6, 8, 5)
        pygame.draw.ellipse(screen, (255, 200, 200), highlight_rect)
        
        # Blinking lights
        light_positions = [-18, -8, 0, 8, 18]
        for i, offset in enumerate(light_positions):
            phase = (time * 8 + i * 0.5) % 1
            if phase > 0.5:
                light_color = (255, 255, 100)
            else:
                light_color = (100, 100, 40)
            pygame.draw.circle(screen, light_color, 
                             (rect.centerx + offset, rect.centery + 2), 2)


class SpaceInvaders(BaseGame):
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
        
        # Visual effects
        self.particles: List[Particle] = []
        self.stars: List[Star] = []
        self.screen_shake = 0.0
        self.flash_alpha = 0
        
        # Alien movement
        self.alien_direction = 1
        self.alien_speed = self.config.SI_ALIEN_SPEED_X
        self.alien_x_offset = 0.0
        
        # Game state
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        self.level = 1
        
        # State representation
        self._num_aliens = self.config.SI_ALIEN_ROWS * self.config.SI_ALIEN_COLS
        self._aliens_remaining = self._num_aliens
        
        self._max_player_bullets = self.config.SI_MAX_PLAYER_BULLETS
        self._state_size = 1 + self._max_player_bullets * 2 + self._num_aliens + 3
        self._state_array = np.zeros(self._state_size, dtype=np.float32)
        self._alien_states = np.ones(self._num_aliens, dtype=np.float32)
        
        self._inv_width = 1.0 / self.width
        self._inv_height = 1.0 / self.height
        
        self._time = 0.0
        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 10
        
        # Create visual effects
        if not headless:
            self._create_stars()
            self._scanline_surface = self._create_scanlines()
            self._crt_vignette = self._create_vignette()
        else:
            self._scanline_surface = None
            self._crt_vignette = None
        
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
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                alpha = int((dist / max_dist) ** 2 * 100)
                pygame.draw.rect(surface, (0, 0, 0, alpha), (x, y, 4, 4))
        
        return surface
    
    def _spawn_explosion(self, x: int, y: int, color: Tuple[int, int, int], count: int = 15) -> None:
        """Spawn explosion particles."""
        if self.headless:
            return
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
    
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
        
        ship_x = (self.width - self.config.SI_SHIP_WIDTH) // 2
        ship_y = self.height - self.config.SI_SHIP_Y_OFFSET - self.config.SI_SHIP_HEIGHT
        self.ship = Ship(
            ship_x, ship_y,
            self.config.SI_SHIP_WIDTH,
            self.config.SI_SHIP_HEIGHT,
            self.config.SI_SHIP_SPEED,
            self.config.SI_COLOR_SHIP
        )
        
        self._create_aliens()
        
        self.player_bullets = []
        self.alien_bullets = []
        self.ufo = None
        self.particles = []
        
        self.alien_direction = 1
        self.alien_speed = self.config.SI_ALIEN_SPEED_X
        self.alien_x_offset = 0.0
        
        self._aliens_remaining = self._num_aliens
        self._alien_states.fill(1.0)
        self._shoot_cooldown = 0
        self.screen_shake = 0
        self.flash_alpha = 0
        
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
            alien_type = min(row, 2)
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
        
        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1
        
        # Handle action
        if action == 0:
            self.ship.move(-1, self.width)
        elif action == 2:
            self.ship.move(1, self.width)
        elif action == 3:
            if self._shoot_cooldown == 0 and len(self.player_bullets) < self._max_player_bullets:
                self._fire_player_bullet()
                reward += self.config.SI_REWARD_SHOOT
                self._shoot_cooldown = self._shoot_cooldown_max
        
        # Update game objects
        self._update_bullets()
        reward += self._update_aliens()
        self._update_ufo()
        reward += self._handle_collisions()
        
        # Update visual effects
        self._update_effects()
        
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
    
    def _update_effects(self) -> None:
        """Update visual effects."""
        if self.headless:
            return
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
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
            (100, 255, 200)  # Cyan-green for player bullets
        )
        self.player_bullets.append(bullet)
    
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
                    (255, 80, 80)  # Red for alien bullets
                )
                self.alien_bullets.append(bullet)
        
        # Spawn UFO
        if self.ufo is None and random.random() < self.config.SI_UFO_CHANCE:
            direction = random.choice([-1, 1])
            x = -50 if direction > 0 else self.width + 50
            self.ufo = UFO(x, 50, direction * self.config.SI_UFO_SPEED, self.config.SI_COLOR_UFO)
        
        return reward
    
    def _update_ufo(self) -> None:
        if self.ufo is not None:
            self.ufo.update()
            if self.ufo.x < -100 or self.ufo.x > self.width + 100:
                self.ufo = None
    
    def _handle_collisions(self) -> float:
        assert self.ship is not None
        reward = 0.0
        
        # Player bullets vs aliens
        for bullet in self.player_bullets:
            if not bullet.alive:
                continue
            for idx, alien in enumerate(self.aliens):
                if not alien.alive:
                    continue
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
                    
                    # Visual effects
                    self._spawn_explosion(
                        alien_rect.centerx, alien_rect.centery, 
                        alien.color, count=20
                    )
                    self.screen_shake = 3
                    self.flash_alpha = 30
                    
                    self.alien_speed = self.config.SI_ALIEN_SPEED_X * (1 + (self._num_aliens - self._aliens_remaining) * 0.02)
                    break
        
        # Player bullets vs UFO
        if self.ufo is not None and self.ufo.alive:
            for bullet in self.player_bullets:
                if not bullet.alive:
                    continue
                if bullet.rect.colliderect(self.ufo.rect):
                    bullet.alive = False
                    self._spawn_explosion(
                        self.ufo.rect.centerx, self.ufo.rect.centery,
                        self.ufo.color, count=30
                    )
                    self.ufo.alive = False
                    self.score += self.config.SI_UFO_POINTS
                    reward += self.config.SI_REWARD_UFO_HIT
                    self.screen_shake = 8
                    self.flash_alpha = 60
                    self.ufo = None
                    break
        
        # Alien bullets vs player
        for bullet in self.alien_bullets:
            if not bullet.alive:
                continue
            if bullet.rect.colliderect(self.ship.rect):
                bullet.alive = False
                self._spawn_explosion(
                    self.ship.rect.centerx, self.ship.rect.centery,
                    (255, 100, 100), count=25
                )
                self.lives -= 1
                reward += self.config.SI_REWARD_PLAYER_DEATH
                self.screen_shake = 10
                self.flash_alpha = 80
                
                if self.lives <= 0:
                    self.game_over = True
                break
        
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
        
        self._state_array[idx:idx + self._num_aliens] = self._alien_states
        idx += self._num_aliens
        
        max_offset = self.width * 0.3
        self._state_array[idx] = (self.alien_x_offset / max_offset + 1) * 0.5
        idx += 1
        
        self._state_array[idx] = (self.alien_direction + 1) * 0.5
        idx += 1
        
        if self.alien_bullets:
            nearest_y = max(b.y for b in self.alien_bullets if b.alive) if any(b.alive for b in self.alien_bullets) else 0
            self._state_array[idx] = nearest_y * self._inv_height
        else:
            self._state_array[idx] = 0.0
        
        return self._state_array.copy()
    
    def _get_info(self) -> dict:
        return {
            'score': self.score,
            'lives': self.lives,
            'aliens_remaining': self._aliens_remaining,
            'won': self.won,
            'level': self.level
        }
    
    def render(self, screen: pygame.Surface) -> None:
        if self.headless:
            return
        
        assert self.ship is not None
        
        # Apply screen shake offset
        shake_x = int(random.uniform(-self.screen_shake, self.screen_shake)) if self.screen_shake > 0 else 0
        shake_y = int(random.uniform(-self.screen_shake, self.screen_shake)) if self.screen_shake > 0 else 0
        
        # Background - deep space black with subtle blue tint
        screen.fill((2, 4, 12))
        
        # Draw stars
        for star in self.stars:
            star.draw(screen, self._time)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
        
        # Draw aliens with shake offset
        for alien in self.aliens:
            if alien.alive:
                original_x = alien.x
                alien.x += int(self.alien_x_offset) + shake_x
                original_y = alien.y
                alien.y += shake_y
                alien.draw(screen, self._time)
                alien.x = original_x
                alien.y = original_y
        
        # Draw UFO
        if self.ufo is not None and self.ufo.alive:
            self.ufo.draw(screen, self._time)
        
        # Draw bullets with shake
        for bullet in self.player_bullets:
            bullet.x += shake_x
            bullet.y += shake_y
            bullet.draw(screen)
            bullet.x -= shake_x
            bullet.y -= shake_y
        for bullet in self.alien_bullets:
            bullet.x += shake_x
            bullet.y += shake_y
            bullet.draw(screen)
            bullet.x -= shake_x
            bullet.y -= shake_y
        
        # Draw ship with shake
        original_ship_x = self.ship.x
        original_ship_y = self.ship.y
        self.ship.x += shake_x
        self.ship.y += shake_y
        self.ship.draw(screen, self._time)
        self.ship.x = original_ship_x
        self.ship.y = original_ship_y
        
        # Apply CRT scanlines
        if self._scanline_surface:
            screen.blit(self._scanline_surface, (0, 0))
        
        # Apply vignette
        if self._crt_vignette:
            screen.blit(self._crt_vignette, (0, 0))
        
        # Flash effect
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, self.flash_alpha))
            screen.blit(flash_surface, (0, 0))
        
        # Draw HUD (not affected by shake)
        self._draw_hud(screen)
    
    def _draw_hud(self, screen: pygame.Surface) -> None:
        """Draw heads-up display with retro style."""
        # Use a pixelated font effect
        font = pygame.font.Font(None, 36)
        
        # Score with glow
        score_text = f"SCORE: {self.score}"
        # Glow
        glow_surf = font.render(score_text, True, (0, 150, 50))
        screen.blit(glow_surf, (12, 12))
        # Main text
        text_surf = font.render(score_text, True, (0, 255, 100))
        screen.blit(text_surf, (10, 10))
        
        # Lives with glow
        lives_text = f"LIVES: {self.lives}"
        glow_surf = font.render(lives_text, True, (0, 150, 50))
        screen.blit(glow_surf, (self.width - 122, 12))
        text_surf = font.render(lives_text, True, (0, 255, 100))
        screen.blit(text_surf, (self.width - 120, 10))
        
        # Level indicator
        level_text = f"LEVEL {self.level}"
        level_surf = font.render(level_text, True, (100, 200, 100))
        level_rect = level_surf.get_rect(centerx=self.width // 2, top=10)
        screen.blit(level_surf, level_rect)
        
        # Game over / Victory message
        if self.game_over:
            big_font = pygame.font.Font(None, 72)
            if self.won:
                msg = "VICTORY!"
                color = (0, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            # Shadow
            shadow = big_font.render(msg, True, (0, 0, 0))
            shadow_rect = shadow.get_rect(center=(self.width // 2 + 3, self.height // 2 + 3))
            screen.blit(shadow, shadow_rect)
            
            # Main text with glow
            glow = big_font.render(msg, True, tuple(c // 2 for c in color))
            glow_rect = glow.get_rect(center=(self.width // 2 + 1, self.height // 2 + 1))
            screen.blit(glow, glow_rect)
            
            text = big_font.render(msg, True, color)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            screen.blit(text, text_rect)
            
            # Final score
            score_font = pygame.font.Font(None, 36)
            score_msg = f"Final Score: {self.score}"
            score_surf = score_font.render(score_msg, True, (200, 200, 200))
            score_rect = score_surf.get_rect(center=(self.width // 2, self.height // 2 + 50))
            screen.blit(score_surf, score_rect)
    
    def close(self) -> None:
        pass
    
    def seed(self, seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)


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
