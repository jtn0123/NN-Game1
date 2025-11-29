"""
Breakout Game Implementation
============================

A complete Atari Breakout clone designed for AI training.

Key Features:
- Clean state representation for neural network input
- Configurable rewards for reinforcement learning
- Efficient physics (no external physics library needed)
- Beautiful rendering with Pygame

Game Rules:
- Player controls a paddle at the bottom
- Ball bounces around, breaking bricks
- Goal: Break all bricks without letting ball fall
- 3 lives per game
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional
import math

from .base_game import BaseGame
from .particles import ParticleSystem, TrailRenderer, create_gradient_surface
import sys
sys.path.append('..')
from config import Config


class Brick:
    """Represents a single brick in the game."""
    
    def __init__(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.alive = True
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the brick if it's still alive."""
        if self.alive:
            # Main brick
            pygame.draw.rect(screen, self.color, self.rect)
            # Highlight (3D effect)
            highlight = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 3)
            darker = tuple(max(0, c - 40) for c in self.color)
            lighter = tuple(min(255, c + 40) for c in self.color)
            pygame.draw.rect(screen, lighter, highlight)
            # Shadow
            shadow = pygame.Rect(self.rect.x, self.rect.bottom - 3, self.rect.width, 3)
            pygame.draw.rect(screen, darker, shadow)


class Ball:
    """The bouncing ball."""
    
    def __init__(self, x: float, y: float, radius: int, speed: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        # Initial direction: upward with random horizontal component
        angle = np.random.uniform(-np.pi/3, np.pi/3) - np.pi/2
        self.dx = speed * np.cos(angle)
        self.dy = speed * np.sin(angle)
    
    @property
    def rect(self) -> pygame.Rect:
        """Get bounding rectangle for collision detection."""
        return pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )
    
    def move(self) -> None:
        """Update ball position."""
        self.x += self.dx
        self.y += self.dy
    
    def draw(self, screen: pygame.Surface, color: Tuple[int, int, int]) -> None:
        """Draw the ball with a glow effect."""
        # Glow
        glow_radius = self.radius + 4
        glow_color = tuple(min(255, c + 50) for c in color)
        pygame.draw.circle(screen, glow_color, (int(self.x), int(self.y)), glow_radius)
        # Main ball
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        # Highlight
        highlight_pos = (int(self.x - self.radius//3), int(self.y - self.radius//3))
        pygame.draw.circle(screen, (255, 255, 255), highlight_pos, self.radius//3)


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
        """
        Move paddle horizontally.
        
        Args:
            direction: -1 for left, 0 for stay, 1 for right
            screen_width: Width of the game screen
        """
        self.x += direction * self.speed
        # Keep paddle on screen
        self.x = max(0, min(self.x, screen_width - self.width))
    
    def draw(self, screen: pygame.Surface, color: Tuple[int, int, int]) -> None:
        """Draw the paddle with rounded corners and gradient effect."""
        rect = self.rect
        # Main paddle (rounded)
        pygame.draw.rect(screen, color, rect, border_radius=5)
        # Top highlight
        highlight = pygame.Rect(rect.x + 5, rect.y + 2, rect.width - 10, 3)
        lighter = tuple(min(255, c + 60) for c in color)
        pygame.draw.rect(screen, lighter, highlight, border_radius=2)


class Breakout(BaseGame):
    """
    Atari Breakout game implementation.
    
    State representation (normalized to [0, 1]):
        - ball_x: Ball X position (0 = left, 1 = right)
        - ball_y: Ball Y position (0 = top, 1 = bottom)
        - ball_dx: Ball X velocity (normalized)
        - ball_dy: Ball Y velocity (normalized)
        - paddle_x: Paddle X position (0 = left, 1 = right)
        - bricks: Flattened array of brick states (1 = exists, 0 = broken)
    
    Actions:
        0 = Move LEFT
        1 = STAY (no movement)
        2 = Move RIGHT
    """
    
    def __init__(self, config: Optional[Config] = None, headless: bool = False):
        """
        Initialize the Breakout game.
        
        Args:
            config: Configuration object (uses default if None)
            headless: If True, skip visual effects for faster training
        """
        self.config = config or Config()
        self.headless = headless
        
        # Screen dimensions
        self.width = self.config.SCREEN_WIDTH
        self.height = self.config.SCREEN_HEIGHT
        
        # Game objects (initialized in reset())
        self.paddle: Optional[Paddle] = None
        self.ball: Optional[Ball] = None
        self.bricks: List[Brick] = []
        
        # Game state
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        
        # For state representation
        self._num_bricks = self.config.BRICK_ROWS * self.config.BRICK_COLS
        
        # Track brick count incrementally (avoids counting every step)
        self._bricks_remaining = self._num_bricks
        
        # Pre-allocated arrays for get_state() (avoids allocation per step)
        self._state_array = np.zeros(8 + self._num_bricks, dtype=np.float32)
        self._brick_states = np.ones(self._num_bricks, dtype=np.float32)
        
        # Pre-computed normalization constants for get_state() (avoids division per call)
        self._inv_width = 1.0 / self.width
        self._inv_height = 1.0 / self.height
        self._max_speed = max(config.BALL_SPEED * 1.5, 1.0)  # Guard against BALL_SPEED=0
        self._inv_max_speed = 1.0 / self._max_speed
        self._inv_paddle_range = 1.0 / (self.width - config.PADDLE_WIDTH)
        
        # For reward shaping (tracking ball)
        self._prev_distance_to_target: float = 0.0
        
        # Cached predicted landing x (avoid redundant computation)
        self._cached_predicted_x: float = 0.0
        
        # Visual effects (skip in headless mode for performance)
        if not headless:
            self.particles = ParticleSystem(max_particles=500)
            self.ball_trail = TrailRenderer(length=12)
            self.background_surface: Optional[pygame.Surface] = None  # Cached gradient background
            self._create_background()
            # Cache fonts to avoid recreating them every frame
            pygame.font.init()  # Ensure fonts are initialized
            self._hud_font = pygame.font.Font(None, 36)
            self._big_font = pygame.font.Font(None, 72)
        else:
            self.particles = None  # type: ignore
            self.ball_trail = None  # type: ignore
            self._hud_font = None  # type: ignore
            self._big_font = None  # type: ignore
            self.background_surface = None
        
        # Initialize the game
        self.reset()
    
    @property
    def state_size(self) -> int:
        """State vector dimension."""
        # ball(4) + paddle(1) + tracking(3) + bricks(rows * cols)
        # tracking = relative_x, predicted_landing, distance_to_target
        return 8 + self._num_bricks
    
    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return 3  # LEFT, STAY, RIGHT
    
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            Initial state vector
        """
        # Reset game state
        self.score = 0
        self.lives = self.config.LIVES
        self.game_over = False
        self.won = False
        
        # Create paddle at center-bottom
        paddle_x = (self.width - self.config.PADDLE_WIDTH) // 2
        paddle_y = self.height - 50
        self.paddle = Paddle(
            paddle_x, paddle_y,
            self.config.PADDLE_WIDTH,
            self.config.PADDLE_HEIGHT,
            self.config.PADDLE_SPEED
        )
        
        # Create ball above paddle
        self._reset_ball()
        
        # Create bricks
        self._create_bricks()
        
        # Reset brick count and pre-allocated brick states
        self._bricks_remaining = self._num_bricks
        self._brick_states.fill(1.0)
        
        # Reset visual effects (only if not headless)
        if not self.headless:
            self.particles.clear()
            self.ball_trail.clear()
        
        # Initialize cached predicted landing for get_state()
        self._cached_predicted_x = self._predict_landing_x()
        
        return self.get_state()
    
    def _create_background(self) -> None:
        """Create cached gradient background surface with grid."""
        self.background_surface = create_gradient_surface(
            self.width, self.height,
            (10, 10, 30),   # Dark blue at top
            (25, 15, 45)    # Dark purple at bottom
        )
        # Bake grid pattern into background (avoids redrawing each frame)
        grid_color = (30, 30, 50)
        for x in range(0, self.width, 40):
            pygame.draw.line(self.background_surface, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 40):
            pygame.draw.line(self.background_surface, grid_color, (0, y), (self.width, y), 1)
    
    def _reset_ball(self) -> None:
        """Reset ball position and velocity."""
        ball_x = self.width // 2
        ball_y = self.height - 100
        self.ball = Ball(
            ball_x, ball_y,
            self.config.BALL_RADIUS,
            self.config.BALL_SPEED
        )
    
    def _create_bricks(self) -> None:
        """Create the grid of bricks."""
        self.bricks = []
        colors = self.config.COLOR_BRICK_COLORS
        
        for row in range(self.config.BRICK_ROWS):
            color = colors[row % len(colors)]
            for col in range(self.config.BRICK_COLS):
                x = self.config.BRICK_OFFSET_LEFT + col * (self.config.BRICK_WIDTH + self.config.BRICK_PADDING)
                y = self.config.BRICK_OFFSET_TOP + row * (self.config.BRICK_HEIGHT + self.config.BRICK_PADDING)
                brick = Brick(x, y, self.config.BRICK_WIDTH, self.config.BRICK_HEIGHT, color)
                self.bricks.append(brick)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one game step.
        
        Args:
            action: 0 = LEFT, 1 = STAY, 2 = RIGHT
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.game_over or self.won:
            return self.get_state(), 0.0, True, self._get_info()
        
        assert self.paddle is not None, "Paddle must be initialized"
        assert self.ball is not None, "Ball must be initialized"
        
        reward = self.config.REWARD_STEP
        
        # Calculate distance to predicted landing BEFORE moving
        paddle_center = self.paddle.x + self.paddle.width / 2
        predicted_x = self._predict_landing_x()
        prev_distance = abs(predicted_x - paddle_center)
        
        # Move paddle based on action
        direction = action - 1  # Convert 0,1,2 to -1,0,1
        self.paddle.move(direction, self.width)
        
        # Move ball
        self.ball.move()
        
        # Update visual effects (skip in headless mode)
        if not self.headless:
            self.ball_trail.update(self.ball.x, self.ball.y)
            self.particles.emit_ball_trail(
                self.ball.x, self.ball.y,
                self.config.COLOR_BALL,
                (self.ball.dx, self.ball.dy)
            )
        
        # Handle collisions
        reward += self._handle_collisions()
        
        # Cache predicted landing position (computed once, used for reward shaping AND get_state)
        # This saves one _predict_landing_x() call per step (~3x → 2x calls)
        self._cached_predicted_x = self._predict_landing_x()
        
        # Dense reward shaping: reward for moving toward predicted landing
        # Only apply when ball is moving toward paddle (dy > 0)
        if self.ball.dy > 0 and not self.game_over:
            new_paddle_center = self.paddle.x + self.paddle.width / 2
            curr_distance = abs(self._cached_predicted_x - new_paddle_center)
            
            # Reward for reducing distance to target
            if curr_distance < prev_distance:
                reward += self.config.REWARD_TRACKING_GOOD
            elif curr_distance > prev_distance:
                reward += self.config.REWARD_TRACKING_BAD
        
        # Update particle system (skip in headless mode)
        if not self.headless:
            self.particles.update()
        
        # Check win condition (use tracked count for speed)
        if self._bricks_remaining == 0:
            self.won = True
            self.game_over = True
            reward += self.config.REWARD_WIN
        
        return self.get_state(), reward, self.game_over, self._get_info()
    
    def _handle_collisions(self) -> float:
        """
        Handle all collision detection and response.
        
        Returns:
            Reward from collisions
        """
        assert self.paddle is not None, "Paddle must be initialized"
        assert self.ball is not None, "Ball must be initialized"
        
        reward = 0.0
        
        # Wall collisions
        # Left wall
        if self.ball.x - self.ball.radius <= 0:
            self.ball.x = self.ball.radius
            self.ball.dx = abs(self.ball.dx)
        
        # Right wall
        if self.ball.x + self.ball.radius >= self.width:
            self.ball.x = self.width - self.ball.radius
            self.ball.dx = -abs(self.ball.dx)
        
        # Top wall
        if self.ball.y - self.ball.radius <= 0:
            self.ball.y = self.ball.radius
            self.ball.dy = abs(self.ball.dy)
        
        # Bottom (lose life)
        if self.ball.y + self.ball.radius >= self.height:
            self.lives -= 1
            reward += self.config.REWARD_GAME_OVER
            
            if self.lives <= 0:
                self.game_over = True
            else:
                self._reset_ball()
        
        # Paddle collision
        if self.ball.rect.colliderect(self.paddle.rect) and self.ball.dy > 0:
            reward += self.config.REWARD_PADDLE_HIT
            
            # Emit paddle hit particles (skip in headless mode)
            if not self.headless:
                self.particles.emit_paddle_hit(
                    self.ball.x,
                    self.paddle.y,
                    self.config.COLOR_PADDLE,
                    count=8
                )
            
            # Calculate bounce angle based on where ball hits paddle
            # Hitting edges = sharper angle, center = straight up
            paddle_center = self.paddle.x + self.paddle.width / 2
            ball_paddle_offset = (self.ball.x - paddle_center) / (self.paddle.width / 2)
            
            # New angle: -150° to -30° (upward arc)
            angle = -np.pi/2 + ball_paddle_offset * np.pi/3
            
            speed = np.sqrt(self.ball.dx**2 + self.ball.dy**2)
            self.ball.dx = speed * np.cos(angle)
            self.ball.dy = speed * np.sin(angle)
            
            # Make sure ball is above paddle (prevent sticking)
            self.ball.y = self.paddle.y - self.ball.radius - 1
        
        # Brick collisions
        for brick_idx, brick in enumerate(self.bricks):
            if brick.alive and self.ball.rect.colliderect(brick.rect):
                brick.alive = False
                self.score += 10
                reward += self.config.REWARD_BRICK_HIT
                self._bricks_remaining -= 1  # Track incrementally
                self._brick_states[brick_idx] = 0.0  # Update pre-allocated array
                
                # Emit brick destruction particles (skip in headless mode)
                if not self.headless:
                    brick_center_x = brick.rect.centerx
                    brick_center_y = brick.rect.centery
                    self.particles.emit_brick_break(
                        brick_center_x,
                        brick_center_y,
                        brick.color,
                        count=15
                    )
                
                # Determine collision side for proper bounce
                # Calculate overlap on each side
                ball_rect = self.ball.rect
                brick_rect = brick.rect
                
                # Check which side was hit
                overlap_left = ball_rect.right - brick_rect.left
                overlap_right = brick_rect.right - ball_rect.left
                overlap_top = ball_rect.bottom - brick_rect.top
                overlap_bottom = brick_rect.bottom - ball_rect.top
                
                min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)
                
                if min_overlap == overlap_left or min_overlap == overlap_right:
                    self.ball.dx = -self.ball.dx
                else:
                    self.ball.dy = -self.ball.dy
                
                break  # Only one brick collision per frame
        
        return reward
    
    def _predict_landing_x(self) -> float:
        """
        Predict where the ball will land at paddle level.
        Uses simple physics with wall bounce simulation.
        
        Returns:
            Predicted x position (in pixels) where ball will reach paddle level
        """
        assert self.ball is not None, "Ball must be initialized"
        assert self.paddle is not None, "Paddle must be initialized"
        
        # If ball is moving up, return current x (no immediate landing)
        if self.ball.dy <= 0:
            return self.ball.x
        
        # Calculate time to reach paddle level
        target_y = self.paddle.y - self.ball.radius
        if self.ball.y >= target_y:
            return self.ball.x  # Already at or past paddle level
        
        time_to_paddle = (target_y - self.ball.y) / self.ball.dy
        
        # Predict x position accounting for wall bounces
        predicted_x = self.ball.x + self.ball.dx * time_to_paddle

        # Simulate wall bounces (handle multiple bounces)
        # Add max iteration guard to prevent infinite loops with extreme velocities
        max_bounces = 10
        iterations = 0
        while (predicted_x < 0 or predicted_x > self.width) and iterations < max_bounces:
            if predicted_x < 0:
                predicted_x = -predicted_x
            elif predicted_x > self.width:
                predicted_x = 2 * self.width - predicted_x
            iterations += 1

        # Clamp to screen bounds if we hit max iterations
        predicted_x = max(0, min(self.width, predicted_x))

        return predicted_x
    
    def get_state(self) -> np.ndarray:
        """
        Get the current game state as a normalized vector.
        
        Returns:
            State vector with values in [0, 1] range (mostly)
            
        State components:
            - ball_x, ball_y: Ball position (normalized)
            - ball_dx, ball_dy: Ball velocity (normalized)
            - paddle_x: Paddle position (normalized)
            - relative_x: Ball x relative to paddle center (normalized, centered at 0.5)
            - predicted_landing: Where ball will land at paddle level (normalized)
            - distance_to_target: Distance from paddle to predicted landing (normalized)
            - brick_states: Binary array of brick alive/broken states
        """
        assert self.paddle is not None, "Paddle must be initialized"
        assert self.ball is not None, "Ball must be initialized"
        
        # Normalize positions to [0, 1] using pre-computed inverse values (multiply is faster than divide)
        ball_x = self.ball.x * self._inv_width
        ball_y = self.ball.y * self._inv_height
        
        # Normalize velocities (approximate range: -speed to +speed)
        ball_dx = (self.ball.dx * self._inv_max_speed + 1) * 0.5  # Map [-1, 1] to [0, 1]
        ball_dy = (self.ball.dy * self._inv_max_speed + 1) * 0.5
        
        # Paddle position using pre-computed inverse
        paddle_x = self.paddle.x * self._inv_paddle_range
        
        # Paddle center for relative calculations
        paddle_center = self.paddle.x + self.paddle.width * 0.5
        
        # Relative position (ball x relative to paddle center)
        # Normalized so 0.5 means ball is directly above paddle
        relative_x = (self.ball.x - paddle_center) * self._inv_width + 0.5
        relative_x = max(0.0, min(1.0, relative_x))  # Faster than np.clip for scalars
        
        # Use cached predicted landing position (computed in step() to avoid redundant calls)
        predicted_x = self._cached_predicted_x
        predicted_landing = predicted_x * self._inv_width
        
        # Distance from paddle center to predicted landing
        # Normalized: 0.5 means paddle is at landing spot, 0 or 1 means far away
        distance_to_target = (predicted_x - paddle_center) * self._inv_width + 0.5
        distance_to_target = max(0.0, min(1.0, distance_to_target))  # Faster than np.clip
        
        # Use pre-allocated state array (avoids allocation per step)
        self._state_array[0] = ball_x
        self._state_array[1] = ball_y
        self._state_array[2] = ball_dx
        self._state_array[3] = ball_dy
        self._state_array[4] = paddle_x
        self._state_array[5] = relative_x
        self._state_array[6] = predicted_landing
        self._state_array[7] = distance_to_target
        # Brick states are updated incrementally in _handle_collisions()
        self._state_array[8:] = self._brick_states
        
        # IMPORTANT: Return a copy! Returning the reference would corrupt replay buffer
        # since all stored states would point to the same array that gets overwritten.
        return self._state_array.copy()
    
    def _get_info(self) -> dict:
        """Get additional game information."""
        return {
            'score': self.score,
            'lives': self.lives,
            'bricks_remaining': self._bricks_remaining,  # Use tracked count (faster)
            'won': self.won
        }
    
    def render(self, screen: pygame.Surface) -> None:
        """
        Render the game to a pygame screen.
        
        Args:
            screen: Pygame surface to draw on
        
        Note:
            In headless mode, this method does nothing since visual effects are disabled.
        """
        # Skip rendering in headless mode (particles/ball_trail are None)
        if self.headless:
            return
        
        assert self.paddle is not None, "Paddle must be initialized"
        assert self.ball is not None, "Ball must be initialized"
        
        # Get screen shake offset
        shake_x, shake_y = self.particles.get_shake_offset()
        
        # Create offset rect for shake effect
        if shake_x != 0 or shake_y != 0:
            # Fill edges that might be exposed during shake
            screen.fill((5, 5, 15))
        
        # Draw gradient background (with grid baked in)
        if self.background_surface:
            screen.blit(self.background_surface, (shake_x, shake_y))
        else:
            screen.fill(self.config.COLOR_BACKGROUND)
        
        # Draw bricks with offset
        for brick in self.bricks:
            if brick.alive:
                # Apply shake offset
                original_x = brick.rect.x
                original_y = brick.rect.y
                brick.rect.x += shake_x
                brick.rect.y += shake_y
                brick.draw(screen)
                brick.rect.x = original_x
                brick.rect.y = original_y
        
        # Draw ball trail (behind ball)
        self.ball_trail.draw(screen, self.config.COLOR_BALL, self.ball.radius)
        
        # Draw particles (behind ball but in front of bricks)
        self.particles.draw(screen)
        
        # Draw paddle with offset
        paddle_color = self.config.COLOR_PADDLE
        paddle_rect = pygame.Rect(
            self.paddle.x + shake_x,
            self.paddle.y + shake_y,
            self.paddle.width,
            self.paddle.height
        )
        pygame.draw.rect(screen, paddle_color, paddle_rect, border_radius=5)
        # Paddle highlight
        highlight = pygame.Rect(paddle_rect.x + 5, paddle_rect.y + 2, paddle_rect.width - 10, 3)
        lighter = tuple(min(255, c + 60) for c in paddle_color)
        pygame.draw.rect(screen, lighter, highlight, border_radius=2)
        
        # Draw ball with offset and enhanced glow
        ball_x = int(self.ball.x) + shake_x
        ball_y = int(self.ball.y) + shake_y
        ball_color = self.config.COLOR_BALL
        
        # Outer glow
        glow_radius = self.ball.radius + 6
        glow_color = tuple(max(0, c // 3) for c in ball_color)
        pygame.draw.circle(screen, glow_color, (ball_x, ball_y), glow_radius)
        
        # Inner glow
        glow_radius = self.ball.radius + 3
        glow_color = tuple(min(255, c + 30) for c in ball_color)
        pygame.draw.circle(screen, glow_color, (ball_x, ball_y), glow_radius)
        
        # Main ball
        pygame.draw.circle(screen, ball_color, (ball_x, ball_y), self.ball.radius)
        
        # Ball highlight
        highlight_pos = (ball_x - self.ball.radius // 3, ball_y - self.ball.radius // 3)
        pygame.draw.circle(screen, (255, 255, 255), highlight_pos, self.ball.radius // 3)
        
        # Draw HUD (score and lives)
        self._draw_hud(screen)
    
    def _draw_hud(self, screen: pygame.Surface) -> None:
        """Draw heads-up display (score, lives)."""
        # Score (top left)
        score_text = self._hud_font.render(f"Score: {self.score}", True, self.config.COLOR_TEXT)
        screen.blit(score_text, (10, 10))

        # Lives (top right)
        lives_text = self._hud_font.render(f"Lives: {self.lives}", True, self.config.COLOR_TEXT)
        screen.blit(lives_text, (self.width - 100, 10))

        # Game over or win message
        if self.game_over:
            if self.won:
                msg = "YOU WIN!"
                color = (46, 204, 113)
            else:
                msg = "GAME OVER"
                color = (231, 76, 60)

            text = self._big_font.render(msg, True, color)
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


class VecBreakout:
    """
    Vectorized Breakout environment for parallel game execution.
    
    Runs N independent game instances simultaneously, allowing batched
    action selection and experience collection. This amortizes Python/PyTorch
    overhead across multiple environments.
    
    Example:
        >>> vec_env = VecBreakout(num_envs=8, config=config)
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
        self.envs = [Breakout(config, headless=headless) for _ in range(num_envs)]
        
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
    pygame.display.set_caption("Breakout - Test")
    clock = pygame.time.Clock()
    
    game = Breakout(config)
    
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
        
        # Step game
        state, reward, done, info = game.step(action)
        
        if done:
            game.reset()
        
        # Render
        game.render(screen)
        pygame.display.flip()
        clock.tick(config.FPS)
    
    pygame.quit()

