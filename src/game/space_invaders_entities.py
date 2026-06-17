"""Entity and effect types for Space Invaders."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import pygame

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
        ],
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
        ],
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
        ],
    ],
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

    def update(self, dt: float = 1 / 60) -> bool:
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
        glow_rect = glow_surf.get_rect(
            center=(self.screen_width // 2 + 2, self.screen_height // 2 + 2)
        )
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

    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        is_player: bool,
        color: Tuple[int, int, int],
    ):
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
            self.height,
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

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        alien_type: int,
        color: Tuple[int, int, int],
    ):
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
                if char == "X":
                    px = rect.x + col_idx * pixel_w
                    py = rect.y + row_idx * pixel_h
                    pygame.draw.rect(screen, self.color, (px, py, pixel_w, pixel_h))

        # Subtle highlight on top pixels
        highlight = tuple(min(255, c + 60) for c in self.color)
        for col_idx, char in enumerate(pattern[0]):
            if char == "X":
                px = rect.x + col_idx * pixel_w
                py = rect.y
                pygame.draw.rect(screen, highlight, (px, py, pixel_w, 1))


class Ship:
    """The player's ship with detailed design."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        speed: int,
        color: Tuple[int, int, int],
    ):
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
                int((255 - i * 40) * thrust_intensity),
            )
            glow_rect = pygame.Rect(
                rect.centerx - glow_width // 2,
                rect.bottom - 3 + i * 2,
                glow_width,
                glow_height,
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
        pygame.draw.line(
            screen,
            detail_color,
            (rect.centerx, rect.top + 5),
            (rect.centerx, rect.bottom - 5),
            2,
        )
        pygame.draw.line(
            screen,
            detail_color,
            (rect.centerx - 10, rect.centery),
            (rect.centerx + 10, rect.centery),
            1,
        )

        # Cockpit (glowing canopy)
        cockpit_points = [
            (rect.centerx, rect.top + 4),
            (rect.centerx - 6, rect.top + 16),
            (rect.centerx + 6, rect.top + 16),
        ]
        # Cockpit glow
        glow_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surf, (100, 255, 200, 80), [(10, 0), (4, 12), (16, 12)])
        screen.blit(glow_surf, (rect.centerx - 10, rect.top + 2))

        # Cockpit glass
        pygame.draw.polygon(screen, (150, 255, 220), cockpit_points)
        # Glass highlight
        pygame.draw.line(
            screen,
            (200, 255, 240),
            (rect.centerx - 2, rect.top + 6),
            (rect.centerx - 4, rect.top + 12),
            1,
        )

        # Wing highlights
        highlight = tuple(min(255, c + 40) for c in self.color)
        pygame.draw.line(
            screen,
            highlight,
            (rect.left + 5, rect.bottom - 6),
            (rect.left - 3, rect.bottom - 3),
            2,
        )
        pygame.draw.line(
            screen,
            highlight,
            (rect.right - 5, rect.bottom - 6),
            (rect.right + 3, rect.bottom - 3),
            2,
        )

        # Ship glow
        glow_surf = pygame.Surface((self.width + 20, self.height + 20), pygame.SRCALPHA)
        pygame.draw.polygon(
            glow_surf,
            (*self.color, 30),
            [(p[0] - rect.x + 10, p[1] - rect.y + 10) for p in hull_points],
        )
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
            pygame.draw.circle(screen, light_color, (rect.centerx + offset, rect.centery + 2), 2)


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
        # Bug 109: Ensure minimum brightness so damaged shields are still visible
        min_brightness = 0.3  # At least 30% brightness
        effective_fade = max(min_brightness, fade)
        color = tuple(int(c * effective_fade) for c in self.color)

        # Draw with slight erosion effect based on health
        rect = self.rect
        if self.health < 4:
            # Add erosion visual
            eroded_rect = rect.inflate(-1, -1)
            pygame.draw.rect(screen, color, eroded_rect)
            # Bug 109: Add outline to make damaged shields more visible
            outline_color = tuple(min(255, c + 40) for c in color)
            pygame.draw.rect(screen, outline_color, eroded_rect, 1)
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
