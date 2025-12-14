"""
Asteroids Game Implementation
=============================

Classic Asteroids game designed for AI training with vector-style graphics.

Key Features:
- Physics-based movement with momentum and drag
- Screen wrapping
- Asteroids split into smaller pieces
- Vector wireframe visual style with glow effects

Game Rules:
- Control a ship with rotation, thrust, and shooting
- Destroy all asteroids to clear the level
- Asteroids split: large -> medium -> small -> destroyed
- Don't collide with asteroids
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .base_game import BaseGame
import sys
sys.path.append('..')
from config import Config


@dataclass
class Vector2:
    """Simple 2D vector."""
    x: float
    y: float

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return np.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> 'Vector2':
        length = self.length()
        if length > 0:
            return Vector2(self.x / length, self.y / length)
        return Vector2(0, 0)


class Ship:
    """The player's spaceship."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.angle = -np.pi / 2  # Pointing up
        self.velocity = Vector2(0, 0)
        self.alive = True
        self.invincible_timer = 0  # Frames of invincibility after respawn

        # Ship dimensions
        self.size = 15

        # Movement constants
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
            np.sin(self.angle) * self.thrust_power
        )
        self.velocity = self.velocity + thrust_vec

        # Clamp speed
        speed = self.velocity.length()
        if speed > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed

    def update(self, width: int, height: int) -> None:
        """Update ship position with screen wrapping."""
        # Apply drag
        self.velocity = self.velocity * self.drag

        # Update position
        self.x += self.velocity.x
        self.y += self.velocity.y

        # Screen wrapping
        self.x = self.x % width
        self.y = self.y % height

        # Update invincibility
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get the vertices of the ship triangle."""
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        # Ship is a triangle pointing in direction of angle
        points = [
            (self.size, 0),       # Nose
            (-self.size * 0.7, -self.size * 0.6),  # Left rear
            (-self.size * 0.4, 0),   # Center rear (notch)
            (-self.size * 0.7, self.size * 0.6),   # Right rear
        ]

        # Rotate and translate
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
        self.lifetime = 50  # Frames until bullet disappears
        self.radius = 3

    def update(self, width: int, height: int) -> bool:
        """Update bullet position. Returns False if bullet should be removed."""
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= 1

        # Screen wrapping
        self.x = self.x % width
        self.y = self.y % height

        return self.lifetime > 0


class Asteroid:
    """An asteroid that can be destroyed."""

    # Size categories
    LARGE = 3
    MEDIUM = 2
    SMALL = 1

    # Points by size
    POINTS = {LARGE: 20, MEDIUM: 50, SMALL: 100}

    # Radius by size
    RADIUS = {LARGE: 40, MEDIUM: 25, SMALL: 12}

    def __init__(self, x: float, y: float, size: int, dx: float = 0, dy: float = 0):
        self.x = x
        self.y = y
        self.size = size
        self.radius = self.RADIUS[size]

        # Random velocity if not specified
        if dx == 0 and dy == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1, 3) * (4 - size)  # Smaller = faster
            self.dx = np.cos(angle) * speed
            self.dy = np.sin(angle) * speed
        else:
            self.dx = dx
            self.dy = dy

        # Random rotation
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.rotation_speed = np.random.uniform(-0.05, 0.05)

        # Generate irregular polygon shape
        self.vertices = self._generate_shape()

    def _generate_shape(self) -> List[Tuple[float, float]]:
        """Generate an irregular polygon shape."""
        num_vertices = np.random.randint(8, 12)
        vertices = []

        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * np.pi
            # Random radius variation for irregular shape
            r = self.radius * np.random.uniform(0.7, 1.0)
            vertices.append((np.cos(angle) * r, np.sin(angle) * r))

        return vertices

    def update(self, width: int, height: int) -> None:
        """Update asteroid position with screen wrapping."""
        self.x += self.dx
        self.y += self.dy
        self.angle += self.rotation_speed

        # Screen wrapping
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

    def split(self) -> List['Asteroid']:
        """Split into smaller asteroids. Returns empty list if already small."""
        if self.size <= self.SMALL:
            return []

        new_size = self.size - 1
        asteroids = []

        for _ in range(2):
            # Random velocity based on current velocity
            angle = np.arctan2(self.dy, self.dx) + np.random.uniform(-0.8, 0.8)
            speed = np.sqrt(self.dx**2 + self.dy**2) * np.random.uniform(1.2, 1.8)
            dx = np.cos(angle) * speed
            dy = np.sin(angle) * speed

            asteroids.append(Asteroid(self.x, self.y, new_size, dx, dy))

        return asteroids


class Asteroids(BaseGame):
    """
    Asteroids game implementation.

    State representation (~47 features):
        - ship_x, ship_y: Ship position (normalized)
        - ship_angle: Ship rotation (normalized to [0, 1])
        - ship_velocity_x, ship_velocity_y: Ship velocity (normalized)
        - nearest_asteroids[8]: (x, y, size, dx, dy) for 8 nearest asteroids (40 values)
        - bullets_active: Number of active bullets (normalized)
        - lives: Lives remaining (normalized)

    Actions:
        0 = ROTATE_LEFT
        1 = ROTATE_RIGHT
        2 = THRUST
        3 = SHOOT
        4 = NOTHING

    Human Controls:
        Arrow keys: Rotate (left/right), Thrust (up)
        Space: Shoot
        Multiple keys can be pressed simultaneously
    """

    # Game constants
    MAX_BULLETS = 5
    SHOOT_COOLDOWN = 10  # Frames between shots
    MAX_ASTEROIDS_TRACKED = 8
    INITIAL_ASTEROID_COUNT = 4
    MAX_LIVES = 3
    RESPAWN_INVINCIBILITY = 120  # Frames of invincibility after death
    SAFE_SPAWN_RADIUS = 150  # Minimum distance from ship center for asteroids

    def __init__(self, config: Optional[Config] = None, headless: bool = False):
        """Initialize the Asteroids game."""
        self.config = config or Config()
        self.headless = headless

        # Screen dimensions
        self.width = self.config.SCREEN_WIDTH
        self.height = self.config.SCREEN_HEIGHT

        # Game objects
        self.ship: Optional[Ship] = None
        self.bullets: List[Bullet] = []
        self.asteroids: List[Asteroid] = []

        # Game state
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.level = 1
        self.shoot_cooldown = 0
        self.show_controls = True  # Show controls helper for human players

        # Pre-allocated state array
        # ship(5) + asteroids(8*5=40) + bullets(1) + lives(1) = 47
        self._state_array = np.zeros(47, dtype=np.float32)

        # Normalization constants
        self._inv_width = 1.0 / self.width
        self._inv_height = 1.0 / self.height
        self._inv_max_speed = 1.0 / 10.0  # Max asteroid/ship speed
        self._inv_2pi = 1.0 / (2 * np.pi)

        # Visual effects
        self._explosion_particles: List[dict] = []
        self._thrust_particles: List[dict] = []
        self._starfield_surface: Optional[pygame.Surface] = None
        self._starfield_data: List[Tuple[int, int, int, float]] = []  # (x, y, brightness, phase)

        if not headless:
            pygame.font.init()
            self._font = pygame.font.Font(None, 48)
            self._small_font = pygame.font.Font(None, 24)
            self._generate_starfield()
        else:
            self._font = None
            self._small_font = None

        self.reset()

    def _generate_starfield(self) -> None:
        """Generate background stars with pre-rendered base and twinkle data."""
        self._starfield_data.clear()

        # Create base starfield surface (static stars)
        self._starfield_surface = pygame.Surface((self.width, self.height))
        self._starfield_surface.fill((0, 0, 0))

        for _ in range(100):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            brightness = np.random.randint(50, 150)
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase for twinkle
            self._starfield_data.append((x, y, brightness, phase))
            # Draw base star
            self._starfield_surface.set_at((x, y), (brightness, brightness, brightness))

    @property
    def state_size(self) -> int:
        """State vector dimension."""
        return 47

    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return 5  # ROTATE_LEFT, ROTATE_RIGHT, THRUST, SHOOT, NOTHING

    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.level = 1
        self.shoot_cooldown = 0

        # Create ship at center
        self.ship = Ship(self.width / 2, self.height / 2)

        # Clear bullets
        self.bullets.clear()

        # Spawn asteroids
        self._spawn_asteroids(self.INITIAL_ASTEROID_COUNT)

        # Clear particles
        self._explosion_particles.clear()
        self._thrust_particles.clear()

        return self.get_state()

    def _spawn_asteroids(self, count: int) -> None:
        """Spawn asteroids away from the ship (safe spawn zone)."""
        self.asteroids.clear()
        center_x, center_y = self.width / 2, self.height / 2

        for _ in range(count):
            # Spawn at edges of screen, never too close to center
            edge = np.random.randint(4)
            if edge == 0:  # Top
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, 50)
            elif edge == 1:  # Bottom
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(self.height - 50, self.height)
            elif edge == 2:  # Left
                x = np.random.uniform(0, 50)
                y = np.random.uniform(0, self.height)
            else:  # Right
                x = np.random.uniform(self.width - 50, self.width)
                y = np.random.uniform(0, self.height)

            self.asteroids.append(Asteroid(x, y, Asteroid.LARGE))

    def _ensure_safe_spawn(self) -> None:
        """Push asteroids away from the ship spawn point (center)."""
        center_x, center_y = self.width / 2, self.height / 2

        for asteroid in self.asteroids:
            dist = np.sqrt((asteroid.x - center_x)**2 + (asteroid.y - center_y)**2)
            if dist < self.SAFE_SPAWN_RADIUS:
                # Push asteroid away from center
                if dist > 0:
                    push_x = (asteroid.x - center_x) / dist
                    push_y = (asteroid.y - center_y) / dist
                else:
                    # Random direction if exactly at center
                    angle = np.random.uniform(0, 2 * np.pi)
                    push_x, push_y = np.cos(angle), np.sin(angle)

                # Move asteroid to safe distance
                asteroid.x = center_x + push_x * (self.SAFE_SPAWN_RADIUS + asteroid.radius + 10)
                asteroid.y = center_y + push_y * (self.SAFE_SPAWN_RADIUS + asteroid.radius + 10)

                # Wrap to screen bounds
                asteroid.x = asteroid.x % self.width
                asteroid.y = asteroid.y % self.height

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step."""
        if self.game_over:
            return self.get_state(), 0.0, True, self._get_info()

        assert self.ship is not None

        reward = 0.01  # Small survival reward

        # Handle input
        if action == 0:  # ROTATE_LEFT
            self.ship.rotate_left()
        elif action == 1:  # ROTATE_RIGHT
            self.ship.rotate_right()
        elif action == 2:  # THRUST
            self.ship.thrust()
            if not self.headless:
                self._emit_thrust_particles()
        elif action == 3:  # SHOOT
            if self.shoot_cooldown <= 0 and len(self.bullets) < self.MAX_BULLETS:
                self._fire_bullet()
                self.shoot_cooldown = self.SHOOT_COOLDOWN

        # Update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # Update ship
        self.ship.update(self.width, self.height)

        # Update bullets
        self.bullets = [b for b in self.bullets if b.update(self.width, self.height)]

        # Update asteroids
        for asteroid in self.asteroids:
            asteroid.update(self.width, self.height)

        # Check collisions
        reward += self._check_collisions()

        # Update particles
        self._update_particles()

        # Check if level cleared
        if len(self.asteroids) == 0:
            self.level += 1
            self._spawn_asteroids(self.INITIAL_ASTEROID_COUNT + self.level - 1)
            reward += 50  # Level clear bonus

        return self.get_state(), reward, self.game_over, self._get_info()

    def _fire_bullet(self) -> None:
        """Fire a bullet from the ship."""
        assert self.ship is not None

        # Bullet starts at nose of ship
        nose_x = self.ship.x + np.cos(self.ship.angle) * self.ship.size
        nose_y = self.ship.y + np.sin(self.ship.angle) * self.ship.size

        self.bullets.append(Bullet(nose_x, nose_y, self.ship.angle))

    def _emit_thrust_particles(self) -> None:
        """Emit thrust particles behind the ship."""
        assert self.ship is not None

        # Particles emit from rear of ship
        rear_x = self.ship.x - np.cos(self.ship.angle) * self.ship.size * 0.7
        rear_y = self.ship.y - np.sin(self.ship.angle) * self.ship.size * 0.7

        for _ in range(3):
            angle = self.ship.angle + np.pi + np.random.uniform(-0.3, 0.3)
            speed = np.random.uniform(2, 5)
            self._thrust_particles.append({
                'x': rear_x,
                'y': rear_y,
                'dx': np.cos(angle) * speed,
                'dy': np.sin(angle) * speed,
                'life': np.random.randint(10, 20),
                'color': (255, np.random.randint(100, 200), 50)
            })

    def _check_collisions(self) -> float:
        """Check for all collisions. Returns reward."""
        assert self.ship is not None
        reward = 0.0

        # Bullet-asteroid collisions
        bullets_to_remove = set()
        asteroids_to_remove = []
        new_asteroids = []

        for b_idx, bullet in enumerate(self.bullets):
            for a_idx, asteroid in enumerate(self.asteroids):
                # Simple circle collision
                dist = np.sqrt((bullet.x - asteroid.x)**2 + (bullet.y - asteroid.y)**2)
                if dist < asteroid.radius + bullet.radius:
                    bullets_to_remove.add(b_idx)
                    asteroids_to_remove.append(a_idx)
                    reward += Asteroid.POINTS[asteroid.size]
                    self.score += Asteroid.POINTS[asteroid.size]

                    # Emit explosion particles
                    if not self.headless:
                        self._emit_explosion(asteroid.x, asteroid.y, asteroid.size)

                    # Split asteroid
                    new_asteroids.extend(asteroid.split())
                    break

        # Remove hit bullets and asteroids
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]
        self.asteroids.extend(new_asteroids)

        # Ship-asteroid collision
        if self.ship.alive and self.ship.invincible_timer <= 0:
            for asteroid in self.asteroids:
                dist = np.sqrt((self.ship.x - asteroid.x)**2 + (self.ship.y - asteroid.y)**2)
                if dist < asteroid.radius + self.ship.size:
                    self.lives -= 1
                    reward -= 100

                    if not self.headless:
                        self._emit_explosion(self.ship.x, self.ship.y, 3)

                    if self.lives <= 0:
                        self.game_over = True
                        self.ship.alive = False
                    else:
                        # Respawn ship at center with safe zone
                        self.ship = Ship(self.width / 2, self.height / 2)
                        self.ship.invincible_timer = self.RESPAWN_INVINCIBILITY
                        self._ensure_safe_spawn()  # Push asteroids away from spawn point
                    break

        return reward

    def _emit_explosion(self, x: float, y: float, size: int) -> None:
        """Emit explosion particles."""
        particle_count = size * 8

        for _ in range(particle_count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1, 4) * size
            self._explosion_particles.append({
                'x': x,
                'y': y,
                'dx': np.cos(angle) * speed,
                'dy': np.sin(angle) * speed,
                'life': np.random.randint(20, 40),
                'color': (255, 255, 255)
            })

    def _update_particles(self) -> None:
        """Update all particles."""
        # Update explosion particles
        for p in self._explosion_particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['life'] -= 1
            p['dx'] *= 0.95
            p['dy'] *= 0.95

        self._explosion_particles = [p for p in self._explosion_particles if p['life'] > 0]

        # Update thrust particles
        for p in self._thrust_particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['life'] -= 1

        self._thrust_particles = [p for p in self._thrust_particles if p['life'] > 0]

    def get_state(self) -> np.ndarray:
        """Get the current game state as a normalized vector."""
        assert self.ship is not None

        idx = 0

        # Ship state
        self._state_array[idx] = self.ship.x * self._inv_width
        self._state_array[idx + 1] = self.ship.y * self._inv_height
        self._state_array[idx + 2] = ((self.ship.angle % (2 * np.pi)) * self._inv_2pi)
        self._state_array[idx + 3] = (self.ship.velocity.x * self._inv_max_speed + 1) * 0.5
        self._state_array[idx + 4] = (self.ship.velocity.y * self._inv_max_speed + 1) * 0.5
        idx += 5

        # Sort asteroids by distance to ship
        if self.asteroids:
            sorted_asteroids = sorted(
                self.asteroids,
                key=lambda a: (a.x - self.ship.x)**2 + (a.y - self.ship.y)**2
            )
        else:
            sorted_asteroids = []

        # Nearest asteroids (8 asteroids * 5 features = 40)
        for i in range(self.MAX_ASTEROIDS_TRACKED):
            if i < len(sorted_asteroids):
                a = sorted_asteroids[i]
                self._state_array[idx] = a.x * self._inv_width
                self._state_array[idx + 1] = a.y * self._inv_height
                self._state_array[idx + 2] = a.size / 3.0  # Normalized size
                self._state_array[idx + 3] = (a.dx * self._inv_max_speed + 1) * 0.5
                self._state_array[idx + 4] = (a.dy * self._inv_max_speed + 1) * 0.5
            else:
                # No asteroid - fill with zeros
                self._state_array[idx:idx + 5] = 0.0
            idx += 5

        # Bullets active (normalized)
        self._state_array[idx] = len(self.bullets) / self.MAX_BULLETS
        idx += 1

        # Lives remaining (normalized)
        self._state_array[idx] = self.lives / self.MAX_LIVES

        return self._state_array.copy()

    def _get_info(self) -> dict:
        """Get additional game information."""
        return {
            'score': self.score,
            'lives': self.lives,
            'level': self.level,
            'asteroids_remaining': len(self.asteroids),
            'won': False  # Asteroids doesn't really have a win condition
        }

    def get_human_action(self, keys: dict) -> int:
        """
        Convert keyboard input to game action for human play.

        Note: Asteroids supports multiple simultaneous actions (rotate+thrust+shoot),
        but this method returns the single highest-priority action for simple control.
        For full human control, use step_human() instead.

        Args:
            keys: Dictionary of pressed keys from pygame.key.get_pressed()

        Returns:
            Action integer (0-4)
        """
        # Priority: Shoot > Thrust > Rotate
        if keys.get(pygame.K_SPACE):
            return 3  # SHOOT
        elif keys.get(pygame.K_UP) or keys.get(pygame.K_w):
            return 2  # THRUST
        elif keys.get(pygame.K_LEFT) or keys.get(pygame.K_a):
            return 0  # ROTATE_LEFT
        elif keys.get(pygame.K_RIGHT) or keys.get(pygame.K_d):
            return 1  # ROTATE_RIGHT
        return 4  # NOTHING

    def step_human(self, keys: dict) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one game step with human keyboard input.

        This allows simultaneous actions (rotate + thrust + shoot).

        Args:
            keys: Dictionary of pressed keys from pygame.key.get_pressed()

        Returns:
            (state, reward, done, info) tuple
        """
        if self.game_over:
            return self.get_state(), 0.0, True, self._get_info()

        assert self.ship is not None

        reward = 0.01  # Small survival reward

        # Handle rotation
        if keys.get(pygame.K_LEFT) or keys.get(pygame.K_a):
            self.ship.rotate_left()
        if keys.get(pygame.K_RIGHT) or keys.get(pygame.K_d):
            self.ship.rotate_right()

        # Handle thrust
        if keys.get(pygame.K_UP) or keys.get(pygame.K_w):
            self.ship.thrust()
            if not self.headless:
                self._emit_thrust_particles()

        # Handle shooting
        if keys.get(pygame.K_SPACE):
            if self.shoot_cooldown <= 0 and len(self.bullets) < self.MAX_BULLETS:
                self._fire_bullet()
                self.shoot_cooldown = self.SHOOT_COOLDOWN

        # Update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # Update ship
        self.ship.update(self.width, self.height)

        # Update bullets
        self.bullets = [b for b in self.bullets if b.update(self.width, self.height)]

        # Update asteroids
        for asteroid in self.asteroids:
            asteroid.update(self.width, self.height)

        # Check collisions
        reward += self._check_collisions()

        # Update particles
        self._update_particles()

        # Check if level cleared
        if len(self.asteroids) == 0:
            self.level += 1
            self._spawn_asteroids(self.INITIAL_ASTEROID_COUNT + self.level - 1)
            reward += 50  # Level clear bonus

        return self.get_state(), reward, self.game_over, self._get_info()

    def render(self, screen: pygame.Surface) -> None:
        """Render the game with vector-style graphics."""
        if self.headless:
            return

        assert self.ship is not None

        # Draw pre-rendered starfield background
        if self._starfield_surface:
            screen.blit(self._starfield_surface, (0, 0))

            # Add subtle twinkle effect to a few bright stars (optimized - only update 10 stars)
            current_time = pygame.time.get_ticks() * 0.003
            for i in range(0, min(20, len(self._starfield_data)), 2):  # Every other star, max 10
                x, y, brightness, phase = self._starfield_data[i]
                twinkle = int(brightness + np.sin(current_time + phase) * 40)
                twinkle = max(60, min(220, twinkle))
                pygame.draw.circle(screen, (twinkle, twinkle, twinkle), (x, y), 1)
        else:
            screen.fill((0, 0, 0))

        # Draw explosion particles (vector lines radiating out)
        for p in self._explosion_particles:
            alpha = min(255, p['life'] * 8)
            color = (alpha, alpha, alpha)
            end_x = p['x'] + p['dx'] * 3
            end_y = p['y'] + p['dy'] * 3
            pygame.draw.line(screen, color, (int(p['x']), int(p['y'])),
                           (int(end_x), int(end_y)), 1)

        # Draw thrust particles
        for p in self._thrust_particles:
            color = p['color']
            pygame.draw.circle(screen, color, (int(p['x']), int(p['y'])), 2)

        # Draw asteroids (wireframe polygons with glow)
        for asteroid in self.asteroids:
            verts = asteroid.get_world_vertices()

            # Glow effect (draw multiple times with decreasing alpha)
            for glow in range(3):
                offset = glow * 2
                glow_verts = [(v[0], v[1]) for v in verts]
                alpha = 100 - glow * 30
                pygame.draw.polygon(screen, (alpha, alpha, alpha), glow_verts, 1)

            # Main wireframe
            pygame.draw.polygon(screen, (200, 200, 200), verts, 2)

        # Draw bullets (bright dots with short trails)
        for bullet in self.bullets:
            # Trail
            trail_x = bullet.x - bullet.dx * 2
            trail_y = bullet.y - bullet.dy * 2
            pygame.draw.line(screen, (100, 100, 255),
                           (int(trail_x), int(trail_y)),
                           (int(bullet.x), int(bullet.y)), 2)
            # Bullet
            pygame.draw.circle(screen, (255, 255, 255),
                             (int(bullet.x), int(bullet.y)), bullet.radius)

        # Draw ship (vector triangle with glow)
        if self.ship.alive:
            verts = self.ship.get_vertices()

            # Flashing when invincible
            if self.ship.invincible_timer > 0 and (self.ship.invincible_timer // 5) % 2 == 0:
                ship_color = (100, 100, 100)
            else:
                ship_color = (255, 255, 255)

            # Glow
            for glow in range(2):
                alpha = 150 - glow * 50
                pygame.draw.polygon(screen, (0, alpha, 0), verts, 1)

            # Main ship
            pygame.draw.polygon(screen, ship_color, verts, 2)

        # Draw HUD
        if self._font:
            # Score
            score_text = self._font.render(str(self.score), True, (255, 255, 255))
            screen.blit(score_text, (20, 20))

            # Lives (small ships)
            for i in range(self.lives):
                life_x = 20 + i * 25
                life_y = 70
                life_verts = [
                    (life_x + 10, life_y),
                    (life_x, life_y + 15),
                    (life_x + 5, life_y + 12),
                    (life_x + 10, life_y + 15),
                ]
                pygame.draw.polygon(screen, (255, 255, 255), life_verts, 1)

            # Level
            if self._small_font:
                level_text = self._small_font.render(f"Level {self.level}", True, (150, 150, 150))
                screen.blit(level_text, (self.width - 80, 20))

                # Ammo indicator (bullets remaining)
                ammo_label = self._small_font.render("AMMO", True, (100, 100, 100))
                screen.blit(ammo_label, (self.width - 80, 50))

                bullets_remaining = self.MAX_BULLETS - len(self.bullets)
                for i in range(self.MAX_BULLETS):
                    bullet_x = self.width - 80 + i * 12
                    bullet_y = 72
                    if i < bullets_remaining:
                        pygame.draw.circle(screen, (100, 200, 255), (bullet_x + 5, bullet_y), 4)
                    else:
                        pygame.draw.circle(screen, (50, 50, 50), (bullet_x + 5, bullet_y), 4)
                        pygame.draw.circle(screen, (80, 80, 80), (bullet_x + 5, bullet_y), 4, 1)

                # Cooldown bar
                if self.shoot_cooldown > 0:
                    cooldown_pct = self.shoot_cooldown / self.SHOOT_COOLDOWN
                    bar_width = 60
                    bar_height = 4
                    bar_x = self.width - 80
                    bar_y = 90
                    # Background
                    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                    # Cooldown fill
                    fill_width = int(bar_width * cooldown_pct)
                    pygame.draw.rect(screen, (255, 100, 100), (bar_x, bar_y, fill_width, bar_height))

        # Human controls helper (bottom of screen)
        if self.show_controls and self._small_font:
            controls_text = self._small_font.render(
                "ARROWS: Move/Rotate  SPACE: Shoot  (Multiple keys OK)",
                True, (80, 80, 80)
            )
            text_rect = controls_text.get_rect(centerx=self.width // 2, bottom=self.height - 10)
            screen.blit(controls_text, text_rect)

        # Game over
        if self.game_over and self._font:
            text = self._font.render("GAME OVER", True, (255, 50, 50))
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            screen.blit(text, text_rect)

            # Final score
            if self._small_font:
                final_text = self._small_font.render(f"Final Score: {self.score}", True, (200, 200, 200))
                final_rect = final_text.get_rect(center=(self.width // 2, self.height // 2 + 40))
                screen.blit(final_text, final_rect)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)


class VecAsteroids:
    """
    Vectorized Asteroids environment for parallel game execution.

    Runs N independent game instances simultaneously for faster training.
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """Initialize vectorized environment."""
        self.num_envs = num_envs
        self.config = config
        self.headless = headless

        # Create N independent game instances
        self.envs = [Asteroids(config, headless=headless) for _ in range(num_envs)]

        # Pre-allocate arrays
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size

        self._states = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(num_envs, dtype=np.float32)
        self._dones = np.empty(num_envs, dtype=np.bool_)

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Step all environments with batched actions."""
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)

            if done:
                env.reset()

        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        # Update state array for done episodes
        for i, done in enumerate(self._dones):
            if done:
                self._states[i] = self.envs[i].get_state()

        return states_to_return, rewards_to_return, dones_to_return, infos

    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]) -> None:
        """Set random seeds for each environment."""
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)


# Test the game
if __name__ == "__main__":
    pygame.init()
    config = Config()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroids - Human Play Test")
    clock = pygame.time.Clock()

    game = Asteroids(config)
    game.show_controls = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()  # Manual reset with R key
                elif event.key == pygame.K_h:
                    game.show_controls = not game.show_controls  # Toggle controls

        # Get keyboard input as dict for step_human
        pressed = pygame.key.get_pressed()
        keys = {key: pressed[key] for key in range(len(pressed))}

        # Step game with human controls (allows simultaneous actions)
        state, reward, done, info = game.step_human(keys)

        if done:
            pygame.time.wait(2000)
            game.reset()

        # Render
        game.render(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
