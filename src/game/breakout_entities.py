"""Breakout entity primitives."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pygame


class Brick:
    """Represents a single brick in the game."""

    def __init__(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.alive = True

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the brick if it's still alive."""
        if self.alive:
            pygame.draw.rect(screen, self.color, self.rect)
            highlight = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 3)
            darker = tuple(max(0, channel - 40) for channel in self.color)
            lighter = tuple(min(255, channel + 40) for channel in self.color)
            pygame.draw.rect(screen, lighter, highlight)
            shadow = pygame.Rect(self.rect.x, self.rect.bottom - 3, self.rect.width, 3)
            pygame.draw.rect(screen, darker, shadow)


class Ball:
    """The bouncing ball."""

    def __init__(self, x: float, y: float, radius: int, speed: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        angle = np.random.uniform(-np.pi / 3, np.pi / 3) - np.pi / 2
        self.dx = speed * np.cos(angle)
        self.dy = speed * np.sin(angle)

    @property
    def rect(self) -> pygame.Rect:
        """Get bounding rectangle for collision detection."""
        return pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2,
        )

    def move(self) -> None:
        """Update ball position."""
        self.x += self.dx
        self.y += self.dy


class Paddle:
    """Player-controlled paddle at the bottom."""

    def __init__(self, x: int, y: int, width: int, height: int, speed: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed

    @property
    def rect(self) -> pygame.Rect:
        """Get paddle rectangle."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, direction: int, screen_width: int) -> None:
        """Move paddle horizontally and keep it on screen."""
        self.x += direction * self.speed
        self.x = max(0, min(self.x, screen_width - self.width))
