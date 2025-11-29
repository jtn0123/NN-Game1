"""
Particle System
===============

A flexible particle system for visual effects like:
    - Brick destruction sparks and debris
    - Ball trail effects
    - Impact effects
    - Ambient particles

The system is designed to be lightweight and integrate
seamlessly with the pygame rendering loop.
"""

import pygame
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
import math


@dataclass
class Particle:
    """A single particle with physics properties."""
    x: float
    y: float
    vx: float
    vy: float
    life: float  # Remaining life (0-1)
    max_life: float
    size: float
    color: Tuple[int, int, int]
    gravity: float = 0.0
    decay: float = 0.02
    
    def update(self, dt: float = 1.0) -> bool:
        """
        Update particle physics.
        
        Args:
            dt: Delta time multiplier
            
        Returns:
            True if particle is still alive
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += self.gravity * dt
        self.life -= self.decay * dt
        
        # Shrink as it dies
        self.size = max(0.5, self.size * (0.95 + 0.05 * self.life))
        
        return self.life > 0
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the particle with alpha based on life."""
        if self.life <= 0:
            return
        
        # Calculate alpha based on remaining life
        alpha = int(255 * min(1.0, self.life * 2))
        
        # Create color with alpha effect (darken as it dies)
        r = int(self.color[0] * self.life)
        g = int(self.color[1] * self.life)
        b = int(self.color[2] * self.life)
        
        # Draw with glow effect
        if self.size > 2:
            glow_size = int(self.size * 1.5)
            glow_color = (r // 2, g // 2, b // 2)
            pygame.draw.circle(
                screen, glow_color,
                (int(self.x), int(self.y)),
                glow_size
            )
        
        pygame.draw.circle(
            screen, (r, g, b),
            (int(self.x), int(self.y)),
            max(1, int(self.size))
        )


class ParticleSystem:
    """
    Manages collections of particles for visual effects.
    
    Features:
        - Efficient batch updates
        - Multiple effect types (sparks, debris, trails)
        - Automatic cleanup of dead particles
        - Performance limiting (max particles)
    
    Example:
        >>> particles = ParticleSystem()
        >>> particles.emit_brick_break(x=100, y=200, color=(255, 0, 0))
        >>> particles.update()
        >>> particles.draw(screen)
    """
    
    def __init__(self, max_particles: int = 500):
        """
        Initialize the particle system.
        
        Args:
            max_particles: Maximum particles allowed (performance limit)
        """
        self.particles: List[Particle] = []
        self.max_particles = max_particles
        
        # Screen shake state
        self.shake_intensity = 0.0
        self.shake_decay = 0.9
    
    def update(self, dt: float = 1.0) -> None:
        """Update all particles and remove dead ones."""
        # In-place filtering: avoid list recreation each frame
        write_idx = 0
        for read_idx in range(len(self.particles)):
            if self.particles[read_idx].update(dt):
                if write_idx != read_idx:
                    self.particles[write_idx] = self.particles[read_idx]
                write_idx += 1
        # Truncate list to new size (reuses existing list object)
        del self.particles[write_idx:]

        # Update screen shake
        self.shake_intensity *= self.shake_decay
        if self.shake_intensity < 0.5:
            self.shake_intensity = 0
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw all particles to the screen."""
        for particle in self.particles:
            particle.draw(screen)
    
    def get_shake_offset(self) -> Tuple[int, int]:
        """
        Get current screen shake offset.
        
        Returns:
            (x_offset, y_offset) tuple
        """
        if self.shake_intensity <= 0:
            return (0, 0)
        
        angle = random.random() * 2 * math.pi
        offset = self.shake_intensity
        return (
            int(math.cos(angle) * offset),
            int(math.sin(angle) * offset)
        )
    
    def trigger_shake(self, intensity: float = 5.0) -> None:
        """Trigger screen shake effect."""
        self.shake_intensity = max(self.shake_intensity, intensity)
    
    def _add_particle(self, particle: Particle) -> None:
        """Add a particle with overflow protection."""
        if len(self.particles) >= self.max_particles:
            # Remove oldest particles
            self.particles = self.particles[10:]
        self.particles.append(particle)
    
    def emit_brick_break(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        count: int = 12
    ) -> None:
        """
        Emit particles for brick destruction effect.
        
        Creates a burst of sparks and debris pieces.
        
        Args:
            x: Center X position
            y: Center Y position
            color: Base color (will be varied slightly)
            count: Number of particles to emit
        """
        for _ in range(count):
            # Random velocity in all directions (biased upward)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2  # Bias upward
            
            # Vary color slightly
            r = min(255, max(0, color[0] + random.randint(-30, 30)))
            g = min(255, max(0, color[1] + random.randint(-30, 30)))
            b = min(255, max(0, color[2] + random.randint(-30, 30)))
            
            # Random offset from center
            offset_x = random.uniform(-15, 15)
            offset_y = random.uniform(-8, 8)
            
            particle = Particle(
                x=x + offset_x,
                y=y + offset_y,
                vx=vx,
                vy=vy,
                life=1.0,
                max_life=1.0,
                size=random.uniform(2, 5),
                color=(r, g, b),
                gravity=0.3,
                decay=random.uniform(0.02, 0.04)
            )
            self._add_particle(particle)
        
        # Trigger small screen shake
        self.trigger_shake(3.0)
    
    def emit_ball_trail(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        velocity: Tuple[float, float] = (0, 0)
    ) -> None:
        """
        Emit trail particles behind the ball.
        
        Args:
            x: Ball X position
            y: Ball Y position
            color: Ball color
            velocity: Ball velocity (for trailing direction)
        """
        # Emit fewer particles for performance
        if random.random() > 0.5:
            return
        
        # Trail in opposite direction of velocity
        vx = -velocity[0] * 0.1 + random.uniform(-0.5, 0.5)
        vy = -velocity[1] * 0.1 + random.uniform(-0.5, 0.5)
        
        # Lighter color for trail
        r = min(255, color[0] + 50)
        g = min(255, color[1] + 50)
        b = min(255, color[2] + 50)
        
        particle = Particle(
            x=x + random.uniform(-2, 2),
            y=y + random.uniform(-2, 2),
            vx=vx,
            vy=vy,
            life=0.6,
            max_life=0.6,
            size=random.uniform(2, 4),
            color=(r, g, b),
            gravity=0,
            decay=0.08
        )
        self._add_particle(particle)
    
    def emit_paddle_hit(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        count: int = 5
    ) -> None:
        """
        Emit sparkle effect when ball hits paddle.
        
        Args:
            x: Impact X position
            y: Impact Y position
            color: Paddle color
            count: Number of particles
        """
        for _ in range(count):
            angle = random.uniform(-math.pi, 0)  # Upward arc
            speed = random.uniform(1, 4)
            
            particle = Particle(
                x=x + random.uniform(-5, 5),
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 1,
                life=0.5,
                max_life=0.5,
                size=random.uniform(1, 3),
                color=color,
                gravity=0.1,
                decay=0.05
            )
            self._add_particle(particle)
    
    def emit_sparkle(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int] = (255, 255, 200),
        count: int = 3
    ) -> None:
        """
        Emit ambient sparkle effect.
        
        Args:
            x: Center X
            y: Center Y
            color: Sparkle color
            count: Number of particles
        """
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            
            particle = Particle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=0.4,
                max_life=0.4,
                size=random.uniform(1, 2),
                color=color,
                gravity=0,
                decay=0.06
            )
            self._add_particle(particle)
    
    def clear(self) -> None:
        """Remove all particles."""
        self.particles.clear()
        self.shake_intensity = 0


class TrailRenderer:
    """
    Renders smooth motion trails for objects like the ball.
    
    Stores recent positions and draws a fading trail.
    """
    
    def __init__(self, length: int = 10):
        """
        Initialize trail renderer.
        
        Args:
            length: Number of positions to store
        """
        self.positions: List[Tuple[float, float]] = []
        self.max_length = length
    
    def update(self, x: float, y: float) -> None:
        """Add new position to trail."""
        self.positions.append((x, y))
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
    
    def draw(
        self,
        screen: pygame.Surface,
        color: Tuple[int, int, int],
        radius: int
    ) -> None:
        """
        Draw the trail with fading effect.
        
        Args:
            screen: Pygame surface
            color: Base color
            radius: Base radius of the trail
        """
        if len(self.positions) < 2:
            return
        
        for i, (x, y) in enumerate(self.positions):
            # Fade based on position in trail
            progress = i / len(self.positions)
            alpha = progress * 0.5  # Max 50% opacity
            
            r = int(color[0] * alpha)
            g = int(color[1] * alpha)
            b = int(color[2] * alpha)
            
            size = int(radius * progress)
            if size > 0:
                pygame.draw.circle(
                    screen, (r, g, b),
                    (int(x), int(y)), size
                )
    
    def clear(self) -> None:
        """Clear the trail."""
        self.positions.clear()


# Convenience function for creating gradient surfaces
def create_gradient_surface(
    width: int,
    height: int,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int]
) -> pygame.Surface:
    """
    Create a vertical gradient surface.
    
    Args:
        width: Surface width
        height: Surface height
        top_color: Color at top
        bottom_color: Color at bottom
        
    Returns:
        Pygame surface with gradient
    """
    surface = pygame.Surface((width, height))
    
    for y in range(height):
        progress = y / height
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * progress)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * progress)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * progress)
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))
    
    return surface


def create_radial_gradient(
    radius: int,
    center_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int]
) -> pygame.Surface:
    """
    Create a radial gradient surface (for glows).
    
    Args:
        radius: Radius of the gradient
        center_color: Color at center
        edge_color: Color at edge
        
    Returns:
        Pygame surface with radial gradient
    """
    size = radius * 2
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    for r in range(radius, 0, -1):
        progress = 1 - (r / radius)
        color_r = int(center_color[0] + (edge_color[0] - center_color[0]) * (1 - progress))
        color_g = int(center_color[1] + (edge_color[1] - center_color[1]) * (1 - progress))
        color_b = int(center_color[2] + (edge_color[2] - center_color[2]) * (1 - progress))
        alpha = int(255 * progress * progress)  # Quadratic falloff
        
        pygame.draw.circle(
            surface, (color_r, color_g, color_b, alpha),
            (radius, radius), r
        )
    
    return surface

