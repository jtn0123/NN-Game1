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


class ScorePopup:
    """Floating score text that rises and fades."""
    
    def __init__(self, x: int, y: int, score: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = float(y)
        self.score = score
        self.color = color
        self.life = 1.0
        self.speed = 1.5  # Rise speed
    
    def update(self) -> bool:
        """Update popup. Returns True if still alive."""
        self.y -= self.speed
        self.life -= 0.025
        return self.life > 0
    
    def draw(self, screen: pygame.Surface) -> None:
        if self.life <= 0:
            return
        
        font = pygame.font.Font(None, 24)
        alpha = int(255 * self.life)
        
        # Create text with current alpha
        text = f"+{self.score}"
        color = tuple(int(c * self.life) for c in self.color)
        
        # Glow effect
        glow_color = tuple(c // 2 for c in color)
        glow_surf = font.render(text, True, glow_color)
        screen.blit(glow_surf, (self.x - glow_surf.get_width() // 2 + 1, int(self.y) + 1))
        
        # Main text
        text_surf = font.render(text, True, color)
        screen.blit(text_surf, (self.x - text_surf.get_width() // 2, int(self.y)))


class WaveAnnouncement:
    """Large wave number announcement that fades in and out."""
    
    def __init__(self, wave: int, screen_width: int, screen_height: int):
        self.wave = wave
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.timer = 0.0
        self.duration = 2.0  # Total display time in seconds
        self.alive = True
    
    def update(self, dt: float = 1/60) -> bool:
        """Update announcement. Returns True if still alive."""
        self.timer += dt
        if self.timer >= self.duration:
            self.alive = False
        return self.alive
    
    def draw(self, screen: pygame.Surface) -> None:
        if not self.alive:
            return
        
        # Calculate alpha: fade in for 0.3s, hold, fade out for 0.3s
        if self.timer < 0.3:
            alpha = self.timer / 0.3
        elif self.timer > self.duration - 0.3:
            alpha = (self.duration - self.timer) / 0.3
        else:
            alpha = 1.0
        
        # Scale effect: start big, settle to normal
        if self.timer < 0.2:
            scale = 1.5 - 0.5 * (self.timer / 0.2)
        else:
            scale = 1.0
        
        font_size = int(72 * scale)
        font = pygame.font.Font(None, font_size)
        
        text = f"WAVE {self.wave}"
        color = (255, 200, 50)  # Gold
        
        # Glow effect
        glow_color = tuple(int(c * 0.5 * alpha) for c in color)
        glow_surf = font.render(text, True, glow_color)
        glow_rect = glow_surf.get_rect(center=(self.screen_width // 2 + 2, self.screen_height // 2 + 2))
        screen.blit(glow_surf, glow_rect)
        
        # Main text
        text_color = tuple(int(c * alpha) for c in color)
        text_surf = font.render(text, True, text_color)
        text_rect = text_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        screen.blit(text_surf, text_rect)


class Star:
    """A background star."""
    
    def __init__(self, x: int, y: int, speed: float, brightness: int):
        self.x: float = float(x)
        self.y: float = float(y)
        self.speed = speed
        self.brightness = brightness
        self.twinkle_phase = random.uniform(0, math.pi * 2)
    
    def update(self, height: int) -> None:
        self.y += self.speed
        if self.y > height:
            self.y = 0.0
            self.x = float(random.randint(0, 800))
    
    def draw(self, screen: pygame.Surface, time: float) -> None:
        twinkle = 0.7 + 0.3 * math.sin(time * 3 + self.twinkle_phase)
        brightness = int(self.brightness * twinkle)
        # Slight green tint for CRT feel
        r = max(0, min(255, brightness - 20))
        g = max(0, min(255, brightness))
        b = max(0, min(255, brightness - 10))
        pygame.draw.circle(screen, (r, g, b), (int(self.x), int(self.y)), 1)


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
        self.trail: List[Tuple[float, float]] = []  # Trail positions for effect
    
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


class ShieldBlock:
    """A single destructible block of a shield/bunker."""
    
    def __init__(self, x: int, y: int, size: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.health = 4  # Can take 4 hits before destroyed
        self.alive = True
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.size, self.size)
    
    def hit(self) -> bool:
        """Take damage. Returns True if destroyed."""
        self.health -= 1
        if self.health <= 0:
            self.alive = False
            return True
        return False
    
    def draw(self, screen: pygame.Surface) -> None:
        if not self.alive:
            return
        
        # Color fades as health decreases
        fade = self.health / 4.0
        color = tuple(int(c * fade) for c in self.color)
        
        # Draw with slight erosion effect based on health
        rect = self.rect
        if self.health < 4:
            # Add erosion visual
            eroded_rect = rect.inflate(-1, -1)
            pygame.draw.rect(screen, color, eroded_rect)
        else:
            pygame.draw.rect(screen, color, rect)


class Shield:
    """A protective bunker made of destructible blocks - classic Space Invaders defense."""
    
    def __init__(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.blocks: List[ShieldBlock] = []
        self._create_blocks()
    
    def _create_blocks(self) -> None:
        """Create the classic bunker shape with small destructible blocks."""
        block_size = 4  # Small blocks for granular destruction
        
        # Classic bunker shape pattern (wider at bottom, arch in middle bottom)
        rows = self.height // block_size
        cols = self.width // block_size
        
        for row in range(rows):
            for col in range(cols):
                bx = self.x + col * block_size
                by = self.y + row * block_size
                
                # Create the classic bunker shape
                # Full width for most of the bunker
                # Arch cut-out at the bottom center
                row_ratio = row / rows
                col_ratio = col / cols
                center_dist = abs(col_ratio - 0.5)
                
                # Skip blocks in the arch area (bottom center)
                if row_ratio > 0.6 and center_dist < 0.25:
                    continue
                
                # Skip corners for rounded top
                if row_ratio < 0.2:
                    if center_dist > 0.4:
                        continue
                
                self.blocks.append(ShieldBlock(bx, by, block_size, self.color))
    
    def check_collision(self, bullet_rect: pygame.Rect) -> bool:
        """Check if a bullet hits any block. Returns True if hit."""
        for block in self.blocks:
            if block.alive and bullet_rect.colliderect(block.rect):
                block.hit()
                return True
        return False
    
    def draw(self, screen: pygame.Surface) -> None:
        for block in self.blocks:
            block.draw(screen)
    
    @property
    def alive(self) -> bool:
        """Shield is alive if any blocks remain."""
        return any(block.alive for block in self.blocks)


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
        # Enhanced state: ship_x, player_bullets, aliens, movement, danger metrics
        # + alien bullets (x,y for each) + shoot_cooldown, active_bullets, aliens_ratio, lives, level, ufo
        self._state_size = (1 + self._max_player_bullets * 2 + self._num_aliens + 5 
                          + self._max_tracked_alien_bullets * 2 + 5)  # +5 for cooldown, active, ratio, lives, level
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
    
    def _spawn_player_death_explosion(self) -> None:
        """Spawn a dramatic player death explosion."""
        if self.headless or self.ship is None:
            return
        
        cx = self.ship.rect.centerx
        cy = self.ship.rect.centery
        
        # Multi-color explosion for dramatic effect
        colors = [
            (255, 100, 50),   # Orange
            (255, 200, 50),   # Yellow
            (255, 50, 50),    # Red
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
            ship_x, ship_y,
            self.config.SI_SHIP_WIDTH,
            self.config.SI_SHIP_HEIGHT,
            self.config.SI_SHIP_SPEED,
            self.config.SI_COLOR_SHIP
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
        self.alien_base_speed = self.config.SI_ALIEN_SPEED_X * (1 + 0.15 * (self.level - 1))
        self.alien_speed = self.alien_base_speed
        
        # Move aliens closer to player on higher levels (original game behavior)
        level_offset = min((self.level - 1) * 20, 100)  # Max 100 pixels closer
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
            
            # Check win condition: complete SI_WIN_LEVELS levels to win
            # Level starts at 1, so after completing level N, level becomes N+1
            # We win when we've completed SI_WIN_LEVELS levels (i.e., level > SI_WIN_LEVELS)
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
            if alien.alive and alien.y + alien.height >= self.ground_y:
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
            (100, 255, 200)  # Cyan-green for player bullets
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
        pulse_speed = 0.5 + (1 - self._aliens_remaining / self._num_aliens) * 3
        self.alien_pulse_phase += pulse_speed * (1.0 / 60.0)
        
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
            shoot_chance = self.config.SI_ALIEN_SHOOT_CHANCE * (1 + 0.5 * (1 - self._aliens_remaining / self._num_aliens))
            if random.random() < shoot_chance:
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
                        int(bullet.x), int(bullet.y),
                        self.config.SI_COLOR_SHIELD, count=5
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
                    alien.x + self.alien_x_offset,
                    alien.y,
                    alien.width,
                    alien.height
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
                    points = 30 - alien.alien_type * 10
                    self.score += points
                    reward += self.config.SI_REWARD_ALIEN_HIT
                    
                    # Track accuracy
                    self._shots_hit += 1

                    # Visual effects
                    self._spawn_explosion(
                        alien_rect.centerx, alien_rect.centery, 
                        alien.color, count=20
                    )
                    
                    # Score popup
                    if not self.headless:
                        self.score_popups.append(ScorePopup(
                            alien_rect.centerx, alien_rect.centery,
                            points, alien.color
                        ))
                    
                    self.screen_shake = 3
                    self.flash_alpha = 30
                    
                    # Speed up as aliens are destroyed (classic behavior)
                    self.alien_speed = self.alien_base_speed * (1 + (self._num_aliens - self._aliens_remaining) * 0.015)
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
                    
                    # Random UFO points like original arcade (50, 100, 150, or 300)
                    ufo_points = random.choice([50, 100, 100, 150, 150, 300])
                    self.score += ufo_points
                    reward += self.config.SI_REWARD_UFO_HIT
                    
                    # Score popup for UFO
                    if not self.headless:
                        self.score_popups.append(ScorePopup(
                            self.ufo.rect.centerx, self.ufo.rect.centery,
                            ufo_points, (255, 255, 100)  # Yellow for UFO
                        ))
                    
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
                    self._spawn_explosion(
                        int(bullet.x), int(bullet.y),
                        (255, 100, 50), count=5
                    )
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
                alien.x + self.alien_x_offset,
                alien.y,
                alien.width,
                alien.height
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
        
        self._state_array[idx:idx + self._num_aliens] = self._alien_states
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
        # Sort alien bullets by distance to ship Y (closest first = most dangerous)
        ship_y = self.ship.y
        ship_center_x = self.ship.x + self.ship.width // 2
        
        alive_bullets = [(b, ship_y - b.y) for b in self.alien_bullets if b.alive and b.y < ship_y]
        alive_bullets.sort(key=lambda x: x[1])  # Sort by distance (closest first)
        
        for i in range(self._max_tracked_alien_bullets):
            if i < len(alive_bullets):
                bullet = alive_bullets[i][0]
                # Normalized X position relative to ship (-0.5 to 1.5 range, 0.5 = centered on ship)
                self._state_array[idx] = (bullet.x - ship_center_x) * self._inv_width + 0.5
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
        self._state_array[idx] = self._aliens_remaining / self._num_aliens
        idx += 1

        # Lives remaining (risk awareness)
        self._state_array[idx] = self.lives / self.config.LIVES
        idx += 1

        # Level (difficulty awareness)
        self._state_array[idx] = min(self.level / 10.0, 1.0)
        idx += 1

        return self._state_array.copy()
    
    def _get_info(self) -> dict:
        return {
            'score': self.score,
            'lives': self.lives,
            'aliens_remaining': self._aliens_remaining,
            'won': self.won,
            'level': self.level,
            'total_aliens_killed': self.total_aliens_killed,
            'bricks': self.total_aliens_killed,  # Compatibility with training metrics
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
        
        # Draw the ground base that the player is defending
        self._draw_ground_base(screen)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
        
        # Draw shields/bunkers
        for shield in self.shields:
            shield.draw(screen)
        
        # Draw aliens with shake offset and pulse effect
        # Pulse intensity increases as fewer aliens remain (like the audio in original)
        pulse_intensity = 0.15 * (1 - self._aliens_remaining / self._num_aliens)
        pulse_offset = math.sin(self.alien_pulse_phase * math.pi * 2) * pulse_intensity * 3
        
        for alien in self.aliens:
            if alien.alive:
                original_x = alien.x
                original_y = alien.y
                alien.x += int(self.alien_x_offset) + shake_x
                alien.y += shake_y + int(pulse_offset)
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
        
        # Draw ship with shake (flash when invincible)
        original_ship_x = self.ship.x
        original_ship_y = self.ship.y
        self.ship.x += shake_x
        self.ship.y += shake_y
        
        # Invincibility flashing effect
        if self.player_invincible:
            # Flash on/off rapidly
            if int(self._time * 15) % 2 == 0:
                self.ship.draw(screen, self._time)
        else:
            self.ship.draw(screen, self._time)
        
        self.ship.x = original_ship_x
        self.ship.y = original_ship_y
        
        # Apply CRT scanlines
        if self._scanline_surface:
            screen.blit(self._scanline_surface, (0, 0))
        
        # Apply vignette
        if self._crt_vignette:
            screen.blit(self._crt_vignette, (0, 0))
        
        # Draw score popups
        for popup in self.score_popups:
            popup.draw(screen)
        
        # Flash effect
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, self.flash_alpha))
            screen.blit(flash_surface, (0, 0))
        
        # Draw HUD (not affected by shake)
        self._draw_hud(screen)
        
        # Draw wave announcement (on top of everything)
        if self.wave_announcement is not None:
            self.wave_announcement.draw(screen)
    
    def _draw_ground_base(self, screen: pygame.Surface) -> None:
        """Draw the ground base that the player is defending."""
        # Main ground line
        ground_color = (0, 180, 80)
        pygame.draw.rect(screen, ground_color, (0, self.ground_y, self.width, 3))
        
        # Glow effect on ground line
        glow_surface = pygame.Surface((self.width, 10), pygame.SRCALPHA)
        for i in range(5):
            alpha = 40 - i * 8
            pygame.draw.rect(glow_surface, (*ground_color, alpha), (0, i, self.width, 1))
        screen.blit(glow_surface, (0, self.ground_y - 5))
        
        # City/base silhouette at the bottom
        base_color = (0, 60, 30)
        highlight_color = (0, 100, 50)
        
        # Draw stylized buildings/structures
        buildings = [
            # (x_offset_ratio, width, height)
            (0.05, 30, 15),
            (0.12, 20, 10),
            (0.18, 40, 20),
            (0.28, 25, 12),
            (0.35, 35, 18),
            (0.45, 15, 8),
            (0.52, 45, 22),
            (0.62, 20, 14),
            (0.70, 30, 16),
            (0.78, 25, 10),
            (0.85, 35, 18),
            (0.92, 20, 12),
        ]
        
        for x_ratio, bwidth, bheight in buildings:
            bx = int(self.width * x_ratio)
            by = self.ground_y + 3
            
            # Building body
            pygame.draw.rect(screen, base_color, (bx, by, bwidth, bheight))
            
            # Building top highlight
            pygame.draw.rect(screen, highlight_color, (bx, by, bwidth, 2))
            
            # Window lights (flickering)
            if bheight > 10:
                for wy in range(by + 4, by + bheight - 2, 4):
                    for wx in range(bx + 3, bx + bwidth - 3, 6):
                        if random.random() > 0.3:  # 70% of windows lit
                            flicker = 0.7 + 0.3 * math.sin(self._time * 5 + wx * 0.1)
                            window_color = (
                                int(255 * flicker),
                                int(200 * flicker),
                                int(50 * flicker)
                            )
                            pygame.draw.rect(screen, window_color, (wx, wy, 2, 2))
    
    def _draw_hud(self, screen: pygame.Surface) -> None:
        """Draw heads-up display with retro style."""
        # Use a pixelated font effect
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 28)
        
        # Score with glow (left side)
        score_text = f"SCORE: {self.score}"
        # Glow
        glow_surf = font.render(score_text, True, (0, 150, 50))
        screen.blit(glow_surf, (12, 12))
        # Main text
        text_surf = font.render(score_text, True, (0, 255, 100))
        screen.blit(text_surf, (10, 10))
        
        # Lives with ship icons (right side)
        lives_x = self.width - 140
        lives_text = "LIVES:"
        glow_surf = font.render(lives_text, True, (0, 150, 50))
        screen.blit(glow_surf, (lives_x + 2, 12))
        text_surf = font.render(lives_text, True, (0, 255, 100))
        screen.blit(text_surf, (lives_x, 10))
        
        # Draw ship icons for lives
        for i in range(self.lives):
            ship_x = lives_x + 80 + i * 20
            ship_y = 18
            # Mini ship shape
            points = [
                (ship_x, ship_y - 6),
                (ship_x - 8, ship_y + 6),
                (ship_x + 8, ship_y + 6),
            ]
            pygame.draw.polygon(screen, (0, 255, 100), points)
        
        # Level indicator (center, prominent)
        level_text = f"WAVE {self.level}"
        # Glow effect
        glow_surf = font.render(level_text, True, (150, 100, 0))
        glow_rect = glow_surf.get_rect(centerx=self.width // 2 + 1, top=11)
        screen.blit(glow_surf, glow_rect)
        # Main text in gold/yellow
        level_surf = font.render(level_text, True, (255, 200, 50))
        level_rect = level_surf.get_rect(centerx=self.width // 2, top=10)
        screen.blit(level_surf, level_rect)
        
        # Aliens killed counter (below score)
        kills_text = f"KILLS: {self.total_aliens_killed}"
        kills_surf = small_font.render(kills_text, True, (150, 150, 150))
        screen.blit(kills_surf, (10, 38))
        
        # Game over message
        if self.game_over:
            big_font = pygame.font.Font(None, 72)
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
            
            # Final stats
            score_font = pygame.font.Font(None, 36)
            stats_y = self.height // 2 + 50
            
            final_score = f"Final Score: {self.score}"
            score_surf = score_font.render(final_score, True, (200, 200, 200))
            score_rect = score_surf.get_rect(center=(self.width // 2, stats_y))
            screen.blit(score_surf, score_rect)
            
            wave_reached = f"Waves Completed: {self.level - 1}"
            wave_surf = small_font.render(wave_reached, True, (150, 150, 150))
            wave_rect = wave_surf.get_rect(center=(self.width // 2, stats_y + 30))
            screen.blit(wave_surf, wave_rect)
            
            kills_final = f"Total Kills: {self.total_aliens_killed}"
            kills_surf = small_font.render(kills_final, True, (150, 150, 150))
            kills_rect = kills_surf.get_rect(center=(self.width // 2, stats_y + 55))
            screen.blit(kills_surf, kills_rect)
    
    def close(self) -> None:
        pass
    
    def seed(self, seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)


class VecSpaceInvaders:
    """
    Vectorized Space Invaders environment for parallel game execution.

    Runs N independent game instances simultaneously, allowing batched
    action selection and experience collection. This amortizes Python/PyTorch
    overhead across multiple environments.

    Example:
        >>> vec_env = VecSpaceInvaders(num_envs=8, config=config)
        >>> states = vec_env.reset()  # Shape: (8, state_size)
        >>> actions = agent.select_actions_batch(states)  # Shape: (8,)
        >>> next_states, rewards, dones, infos = vec_env.step(actions)
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            config: Game configuration
            headless: Whether to run in headless mode (no rendering)
        """
        self.num_envs = num_envs
        self.config = config
        self.headless = headless

        # Create N independent game instances
        self.envs = [SpaceInvaders(config, headless=headless) for _ in range(num_envs)]

        # Pre-allocate arrays for batched returns (avoid allocation each step)
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size

        self._states = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(num_envs, dtype=np.float32)
        self._dones = np.empty(num_envs, dtype=np.bool_)

    def reset(self) -> np.ndarray:
        """
        Reset all environments.

        Returns:
            Batched initial states of shape (num_envs, state_size)
        """
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def reset_single(self, env_idx: int) -> np.ndarray:
        """
        Reset a single environment.

        Args:
            env_idx: Index of environment to reset

        Returns:
            Initial state for that environment
        """
        return self.envs[env_idx].reset()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step all environments with batched actions.

        Args:
            actions: Array of actions, shape (num_envs,)

        Returns:
            Tuple of (next_states, rewards, dones, infos)
            - next_states: shape (num_envs, state_size)
            - rewards: shape (num_envs,)
            - dones: shape (num_envs,)
            - infos: list of info dicts
        """
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)

            # Auto-reset environments that are done
            if done:
                self._states[i] = env.reset()

        return self._states.copy(), self._rewards.copy(), self._dones.copy(), infos

    def step_no_copy(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Step without copying arrays (faster but caller must use immediately).

        Args:
            actions: Array of actions, shape (num_envs,)

        Returns:
            Tuple of (next_states, rewards, dones, infos) - arrays are views
        """
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)

            # Auto-reset environments that are done
            if done:
                self._states[i] = env.reset()

        return self._states, self._rewards, self._dones, infos

    def get_states(self) -> np.ndarray:
        """Get current states of all environments."""
        for i, env in enumerate(self.envs):
            self._states[i] = env.get_state()
        return self._states.copy()

    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()

    def seed(self, seeds: List[int]) -> None:
        """Set random seeds for each environment."""
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)


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
