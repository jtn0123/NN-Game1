"""Asteroids entity types shared by the game facade and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Vector2:
    """Simple 2D vector."""

    x: float
    y: float

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return float(np.sqrt(self.x * self.x + self.y * self.y))

    def normalize(self) -> "Vector2":
        length = self.length()
        if length > 0:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)


class Ship:
    """The player's spaceship."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.angle = -np.pi / 2
        self.velocity = Vector2(0, 0)
        self.alive = True
        self.invincible_timer = 0
        self.size = 15
        self.rotation_speed = 0.15
        self.thrust_power = 0.15
        self.max_speed = 8.0
        self.drag = 0.99

    def rotate_left(self) -> None:
        self.angle -= self.rotation_speed

    def rotate_right(self) -> None:
        self.angle += self.rotation_speed

    def thrust(self) -> None:
        """Apply thrust in the direction the ship is facing."""
        thrust_vec = Vector2(
            np.cos(self.angle) * self.thrust_power,
            np.sin(self.angle) * self.thrust_power,
        )
        self.velocity = self.velocity + thrust_vec

        speed = self.velocity.length()
        if speed > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed

    def update(self, width: int, height: int) -> None:
        """Update ship position with screen wrapping."""
        self.velocity = self.velocity * self.drag
        self.x += self.velocity.x
        self.y += self.velocity.y
        self.x = self.x % width
        self.y = self.y % height

        if self.invincible_timer > 0:
            self.invincible_timer -= 1

    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get the vertices of the ship triangle."""
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        points = [
            (self.size, 0),
            (-self.size * 0.7, -self.size * 0.6),
            (-self.size * 0.4, 0),
            (-self.size * 0.7, self.size * 0.6),
        ]

        vertices = []
        for px, py in points:
            rx = px * cos_a - py * sin_a + self.x
            ry = px * sin_a + py * cos_a + self.y
            vertices.append((rx, ry))
        return vertices


class Bullet:
    """A bullet fired by the ship."""

    def __init__(self, x: float, y: float, angle: float):
        self.x = x
        self.y = y
        self.speed = 12.0
        self.dx = np.cos(angle) * self.speed
        self.dy = np.sin(angle) * self.speed
        self.lifetime = 50
        self.radius = 3

    def update(self, width: int, height: int) -> bool:
        """Update bullet position. Returns False if bullet should be removed."""
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= 1
        self.x = self.x % width
        self.y = self.y % height
        return self.lifetime > 0


class Asteroid:
    """An asteroid that can be destroyed."""

    LARGE = 3
    MEDIUM = 2
    SMALL = 1
    POINTS = {LARGE: 20, MEDIUM: 50, SMALL: 100}
    RADIUS = {LARGE: 40, MEDIUM: 25, SMALL: 12}

    def __init__(self, x: float, y: float, size: int, dx: float = 0, dy: float = 0):
        self.x = x
        self.y = y
        self.size = size
        self.radius = self.RADIUS[size]

        if dx == 0 and dy == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1, 3) * (4 - size)
            self.dx = np.cos(angle) * speed
            self.dy = np.sin(angle) * speed
        else:
            self.dx = dx
            self.dy = dy

        self.angle = np.random.uniform(0, 2 * np.pi)
        self.rotation_speed = np.random.uniform(-0.05, 0.05)
        self.vertices = self._generate_shape()

    def _generate_shape(self) -> List[Tuple[float, float]]:
        """Generate an irregular polygon shape."""
        num_vertices = np.random.randint(8, 12)
        vertices = []
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * np.pi
            radius = self.radius * np.random.uniform(0.7, 1.0)
            vertices.append((np.cos(angle) * radius, np.sin(angle) * radius))
        return vertices

    def update(self, width: int, height: int) -> None:
        """Update asteroid position with screen wrapping."""
        self.x += self.dx
        self.y += self.dy
        self.angle += self.rotation_speed
        self.x = self.x % width
        self.y = self.y % height

    def get_world_vertices(self) -> List[Tuple[float, float]]:
        """Get vertices in world coordinates."""
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        world_verts = []
        for vx, vy in self.vertices:
            wx = vx * cos_a - vy * sin_a + self.x
            wy = vx * sin_a + vy * cos_a + self.y
            world_verts.append((wx, wy))
        return world_verts

    def split(self) -> List["Asteroid"]:
        """Split into smaller asteroids. Returns empty list if already small."""
        if self.size <= self.SMALL:
            return []

        new_size = self.size - 1
        asteroids = []
        for _ in range(2):
            angle = np.arctan2(self.dy, self.dx) + np.random.uniform(-0.8, 0.8)
            speed = np.sqrt(self.dx**2 + self.dy**2) * np.random.uniform(1.2, 1.8)
            dx = np.cos(angle) * speed
            dy = np.sin(angle) * speed
            asteroids.append(Asteroid(self.x, self.y, new_size, dx, dy))
        return asteroids
