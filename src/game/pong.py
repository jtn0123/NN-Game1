"""
Pong Game Implementation
========================

Classic Pong game designed for AI training with a retro aesthetic.

Key Features:
- Simple state representation for neural network input
- AI opponent with configurable difficulty
- Retro visual style (black/white, CRT effects)
- Full human playability support

Game Rules:
- Player controls right paddle (UP/DOWN or W/S)
- AI controls left paddle
- First to WIN_SCORE points wins
- Ball bounces off paddles and top/bottom walls
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional

from .base_game import BaseGame
import sys
sys.path.append('..')
from config import Config


class PongBall:
    """The bouncing ball for Pong."""

    def __init__(self, x: float, y: float, radius: int, speed: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.base_speed = speed
        self.speed = speed
        self.dx = speed
        self.dy = 0.0

    def reset(self, x: float, y: float, direction: int = 1) -> None:
        """Reset ball to center with random angle."""
        self.x = x
        self.y = y
        self.speed = self.base_speed
        # Random angle between -45 and 45 degrees
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        self.dx = self.speed * direction * np.cos(angle)
        self.dy = self.speed * np.sin(angle)

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


class PongPaddle:
    """A paddle for Pong (player or AI)."""

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

    @property
    def center_y(self) -> float:
        """Get vertical center of paddle."""
        return self.y + self.height / 2

    def move(self, direction: int, screen_height: int) -> None:
        """
        Move paddle vertically.

        Args:
            direction: -1 for up, 0 for stay, 1 for down
            screen_height: Height of the game screen
        """
        self.y += direction * self.speed
        # Keep paddle on screen
        self.y = max(0, min(self.y, screen_height - self.height))


class Pong(BaseGame):
    """
    Classic Pong game implementation.

    State representation (normalized to [0, 1]):
        - paddle_y: Player paddle Y position
        - opponent_y: AI paddle Y position
        - ball_x: Ball X position
        - ball_y: Ball Y position
        - ball_dx: Ball X velocity (normalized)
        - ball_dy: Ball Y velocity (normalized)
        - relative_y: Ball Y relative to player paddle center
        - predicted_y: Where ball will arrive at player's side
        - distance_to_predicted: Signed distance from paddle to predicted landing
        - ball_approaching: 1 if ball moving toward player, 0 otherwise

    Actions:
        0 = Move UP
        1 = STAY (no movement)
        2 = Move DOWN
    """

    # Game constants
    PADDLE_WIDTH = 15
    PADDLE_HEIGHT = 80
    PADDLE_SPEED = 6  # Slightly slower for better control
    PADDLE_MARGIN = 30  # Distance from edge

    BALL_RADIUS = 8
    BALL_SPEED = 6  # Slightly slower base speed
    BALL_MAX_SPEED = 12  # Maximum speed after acceleration

    WIN_SCORE = 11

    # AI difficulty settings
    AI_SKILL_LEVELS = {
        'easy': {'error': 60, 'reaction': 0.3, 'speed_mult': 0.7},
        'medium': {'error': 30, 'reaction': 0.15, 'speed_mult': 0.9},
        'hard': {'error': 10, 'reaction': 0.05, 'speed_mult': 1.0},
    }

    def __init__(self, config: Optional[Config] = None, headless: bool = False,
                 ai_difficulty: str = 'medium'):
        """
        Initialize the Pong game.

        Args:
            config: Configuration object (uses default if None)
            headless: If True, skip visual effects for faster training
            ai_difficulty: 'easy', 'medium', or 'hard'
        """
        self.config = config or Config()
        self.headless = headless

        # Screen dimensions
        self.width = self.config.SCREEN_WIDTH
        self.height = self.config.SCREEN_HEIGHT

        # AI difficulty
        ai_settings = self.AI_SKILL_LEVELS.get(ai_difficulty, self.AI_SKILL_LEVELS['medium'])
        self.ai_error = ai_settings['error']
        self.ai_reaction = ai_settings['reaction']
        self.ai_speed_mult = ai_settings['speed_mult']
        self.ai_target_y = self.height / 2  # Smoothed target

        # Game objects
        self.player_paddle: Optional[PongPaddle] = None
        self.ai_paddle: Optional[PongPaddle] = None
        self.ball: Optional[PongBall] = None

        # Game state
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.rally_count = 0  # Track hits in current rally
        self.last_hit_by = None  # 'player' or 'ai'

        # Pre-allocated state array
        self._state_array = np.zeros(10, dtype=np.float32)

        # Pre-computed normalization constants
        self._inv_width = 1.0 / self.width
        self._inv_height = 1.0 / self.height
        self._inv_max_speed = 1.0 / (self.BALL_MAX_SPEED * 1.5)

        # Cached predicted landing
        self._cached_predicted_y = self.height / 2

        # Ball trail for retro effect (only in visual mode)
        self._ball_trail: List[Tuple[float, float]] = []
        # Bug 81 fix: Ensure trail length is always at least 1 to prevent issues with pop(0) on empty list
        self._trail_length = max(1, getattr(config, 'BALL_TRAIL_LENGTH', 10))

        # Visual effects timing
        self._score_flash_timer = 0
        self._hit_flash_timer = 0

        # Visual effects
        if not headless:
            pygame.font.init()
            self._font = pygame.font.Font(None, 72)
            self._small_font = pygame.font.Font(None, 28)
            self._label_font = pygame.font.Font(None, 24)
        else:
            self._font = None
            self._small_font = None
            self._label_font = None

        self.reset()

    @property
    def state_size(self) -> int:
        """State vector dimension."""
        return 10

    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return 3  # UP, STAY, DOWN

    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        # Reset scores
        self.player_score = 0
        self.ai_score = 0
        self.game_over = False
        self.rally_count = 0
        self.last_hit_by = None

        # Create player paddle (right side)
        player_x = self.width - self.PADDLE_MARGIN - self.PADDLE_WIDTH
        player_y = (self.height - self.PADDLE_HEIGHT) // 2
        self.player_paddle = PongPaddle(
            player_x, player_y,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT,
            self.PADDLE_SPEED
        )

        # Create AI paddle (left side)
        ai_x = self.PADDLE_MARGIN
        ai_y = (self.height - self.PADDLE_HEIGHT) // 2
        self.ai_paddle = PongPaddle(
            ai_x, ai_y,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT,
            int(self.PADDLE_SPEED * self.ai_speed_mult)
        )

        # Create ball at center
        self.ball = PongBall(
            self.width / 2, self.height / 2,
            self.BALL_RADIUS, self.BALL_SPEED
        )
        # Alternate starting direction
        direction = 1 if np.random.random() > 0.5 else -1
        self.ball.reset(self.width / 2, self.height / 2, direction=direction)

        # Clear ball trail
        self._ball_trail.clear()

        # Reset AI target
        self.ai_target_y = self.height / 2

        # Initialize predicted landing
        self._cached_predicted_y = self._predict_landing_y()

        return self.get_state()

    def _reset_ball(self, direction: int = 1) -> None:
        """Reset ball to center after a point."""
        assert self.ball is not None
        self.ball.reset(self.width / 2, self.height / 2, direction)
        self._ball_trail.clear()
        self.rally_count = 0
        self.last_hit_by = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step."""
        if self.game_over:
            return self.get_state(), 0.0, True, self._get_info()

        assert self.player_paddle is not None
        assert self.ai_paddle is not None
        assert self.ball is not None

        reward = 0.0

        # Calculate distance to predicted landing BEFORE moving
        predicted_y = self._predict_landing_y()
        prev_distance = abs(predicted_y - self.player_paddle.center_y)

        # Move player paddle based on action
        direction = action - 1  # Convert 0,1,2 to -1,0,1
        self.player_paddle.move(direction, self.height)

        # Move AI paddle (smarter AI)
        self._update_ai_paddle()

        # Update ball trail
        if not self.headless:
            self._ball_trail.append((self.ball.x, self.ball.y))
            if len(self._ball_trail) > self._trail_length:
                self._ball_trail.pop(0)

            # Decay flash timers
            if self._score_flash_timer > 0:
                self._score_flash_timer -= 1
            if self._hit_flash_timer > 0:
                self._hit_flash_timer -= 1

        # Move ball
        self.ball.move()

        # Handle collisions
        reward += self._handle_collisions()

        # Cache predicted landing
        self._cached_predicted_y = self._predict_landing_y()

        # Tracking reward (only when ball approaching player)
        if self.ball.dx > 0 and not self.game_over:
            curr_distance = abs(self._cached_predicted_y - self.player_paddle.center_y)
            if curr_distance < prev_distance:
                reward += 0.01  # Small reward for tracking
            elif curr_distance > prev_distance:
                reward -= 0.01  # Small penalty for moving away

        # Check win condition
        if self.player_score >= self.WIN_SCORE or self.ai_score >= self.WIN_SCORE:
            self.game_over = True
            # Bonus for winning
            if self.player_score >= self.WIN_SCORE:
                reward += 5.0

        return self.get_state(), reward, self.game_over, self._get_info()

    def _update_ai_paddle(self) -> None:
        """Update AI paddle position with improved AI."""
        assert self.ai_paddle is not None
        assert self.ball is not None

        # Calculate target position
        if self.ball.dx < 0:
            # Ball coming toward AI - predict landing
            target_y = self._predict_landing_y_for_ai()
            # Add error based on ball speed and distance
            distance_factor = self.ball.x / self.width
            error = self.ai_error * distance_factor
            target_y += np.random.uniform(-error, error)
        else:
            # Ball going away - return to center with some anticipation
            target_y = self.height / 2

        # Smooth target movement (reaction time)
        self.ai_target_y += (target_y - self.ai_target_y) * (1 - self.ai_reaction)

        paddle_center = self.ai_paddle.center_y

        # Move toward smoothed target
        threshold = 3
        if paddle_center < self.ai_target_y - threshold:
            self.ai_paddle.move(1, self.height)
        elif paddle_center > self.ai_target_y + threshold:
            self.ai_paddle.move(-1, self.height)

    def _predict_landing_y_for_ai(self) -> float:
        """Predict where ball will arrive at AI's side."""
        assert self.ball is not None
        assert self.ai_paddle is not None

        if self.ball.dx >= 0:
            return self.ball.y

        # Simple prediction
        time_to_target = (self.ai_paddle.x + self.ai_paddle.width - self.ball.x) / self.ball.dx
        if time_to_target >= 0:
            return self.ball.y

        predicted_y = self.ball.y + self.ball.dy * abs(time_to_target)

        # Simulate bounces (with limit to prevent infinite loops)
        bounce_count = 0
        while (predicted_y < 0 or predicted_y > self.height) and bounce_count < 10:
            if predicted_y < 0:
                predicted_y = -predicted_y
            elif predicted_y > self.height:
                predicted_y = 2 * self.height - predicted_y
            bounce_count += 1

        return max(0.0, min(float(self.height), predicted_y))

    def _handle_collisions(self) -> float:
        """Handle all collision detection and response."""
        assert self.ball is not None
        assert self.player_paddle is not None
        assert self.ai_paddle is not None

        reward = 0.0

        # Top/bottom wall collisions
        if self.ball.y - self.ball.radius <= 0:
            self.ball.y = self.ball.radius
            self.ball.dy = abs(self.ball.dy)

        if self.ball.y + self.ball.radius >= self.height:
            self.ball.y = self.height - self.ball.radius
            self.ball.dy = -abs(self.ball.dy)

        # Player paddle collision (right side)
        if self.ball.rect.colliderect(self.player_paddle.rect) and self.ball.dx > 0:
            reward += 0.2  # Reward for hitting ball
            self.rally_count += 1
            self.last_hit_by = 'player'
            self._hit_flash_timer = 10
            self._paddle_bounce(self.player_paddle, direction=-1)

        # AI paddle collision (left side)
        if self.ball.rect.colliderect(self.ai_paddle.rect) and self.ball.dx < 0:
            self.rally_count += 1
            self.last_hit_by = 'ai'
            self._paddle_bounce(self.ai_paddle, direction=1)

        # Scoring (ball past paddles)
        # Player scores (ball past AI)
        if self.ball.x - self.ball.radius <= 0:
            self.player_score += 1
            reward += 1.0  # Big reward for scoring
            self._score_flash_timer = 30
            # Ball goes toward player next
            self._reset_ball(direction=1)

        # AI scores (ball past player)
        if self.ball.x + self.ball.radius >= self.width:
            self.ai_score += 1
            reward -= 1.0  # Penalty for being scored on
            self._score_flash_timer = 30
            # Ball goes toward AI next
            self._reset_ball(direction=-1)

        return reward

    def _paddle_bounce(self, paddle: PongPaddle, direction: int) -> None:
        """Handle ball bouncing off a paddle."""
        assert self.ball is not None

        # Calculate bounce angle based on where ball hits paddle
        paddle_center = paddle.center_y
        offset = (self.ball.y - paddle_center) / (paddle.height / 2)
        offset = max(-1, min(1, offset))  # Clamp to [-1, 1]

        # Bounce angle: -60 to 60 degrees from horizontal
        angle = offset * np.pi / 3

        # Increase speed slightly with each hit (up to max)
        self.ball.speed = min(self.ball.speed * 1.08, self.BALL_MAX_SPEED)

        self.ball.dx = self.ball.speed * direction * np.cos(angle)
        self.ball.dy = self.ball.speed * np.sin(angle)

        # Push ball out of paddle
        if direction == 1:  # Bouncing right (from AI paddle)
            self.ball.x = paddle.x + paddle.width + self.ball.radius + 1
        else:  # Bouncing left (from player paddle)
            self.ball.x = paddle.x - self.ball.radius - 1

    def _predict_landing_y(self) -> float:
        """Predict where ball will arrive at player's side."""
        assert self.ball is not None
        assert self.player_paddle is not None

        # If ball moving away from player, return ball's current y
        if self.ball.dx <= 0:
            return self.ball.y

        # Simulate trajectory
        target_x = self.player_paddle.x
        time_to_target = (target_x - self.ball.x) / self.ball.dx

        if time_to_target <= 0:
            return self.ball.y

        # Predict y position
        predicted_y = self.ball.y + self.ball.dy * time_to_target

        # Simulate wall bounces using reflection
        bounce_count = 0
        while (predicted_y < 0 or predicted_y > self.height) and bounce_count < 10:
            if predicted_y < 0:
                predicted_y = -predicted_y
            elif predicted_y > self.height:
                predicted_y = 2 * self.height - predicted_y
            bounce_count += 1

        return max(0.0, min(float(self.height), predicted_y))

    def get_state(self) -> np.ndarray:
        """Get the current game state as a normalized vector."""
        assert self.player_paddle is not None
        assert self.ai_paddle is not None
        assert self.ball is not None

        # Normalize positions
        paddle_range = self.height - self.PADDLE_HEIGHT
        # Bug 70 fix: Use raw normalized position instead of misleading 0.5 when paddle >= screen height
        if paddle_range > 0:
            paddle_y = self.player_paddle.y / paddle_range
            opponent_y = self.ai_paddle.y / paddle_range
        else:
            # Invalid config - normalize using screen height as fallback
            paddle_y = self.player_paddle.y / max(1, self.height)
            opponent_y = self.ai_paddle.y / max(1, self.height)
        ball_x = self.ball.x * self._inv_width
        ball_y = self.ball.y * self._inv_height

        # Normalize velocities to [0, 1]
        ball_dx = (self.ball.dx * self._inv_max_speed + 1) * 0.5
        ball_dy = (self.ball.dy * self._inv_max_speed + 1) * 0.5

        # Relative position (ball y relative to paddle center)
        paddle_center_y = self.player_paddle.center_y
        relative_y = (self.ball.y - paddle_center_y) * self._inv_height + 0.5
        relative_y = max(0.0, min(1.0, relative_y))

        # Predicted landing
        predicted_y = self._cached_predicted_y * self._inv_height

        # Signed distance to predicted landing (preserves direction info)
        distance_to_pred = (self._cached_predicted_y - paddle_center_y) * self._inv_height
        # Map from [-0.5, 0.5] to [0, 1] while preserving sign info
        distance_to_pred = distance_to_pred + 0.5
        distance_to_pred = max(0.0, min(1.0, distance_to_pred))

        # Ball approaching player (1 if moving right, 0 if moving left)
        ball_approaching = 1.0 if self.ball.dx > 0 else 0.0

        self._state_array[0] = paddle_y
        self._state_array[1] = opponent_y
        self._state_array[2] = ball_x
        self._state_array[3] = ball_y
        self._state_array[4] = ball_dx
        self._state_array[5] = ball_dy
        self._state_array[6] = relative_y
        self._state_array[7] = predicted_y
        self._state_array[8] = distance_to_pred
        self._state_array[9] = ball_approaching

        return self._state_array.copy()

    def _get_info(self) -> dict:
        """Get additional game information."""
        return {
            'score': self.player_score,
            'ai_score': self.ai_score,
            'lives': max(0, self.WIN_SCORE - self.ai_score),  # Pseudo-lives
            'won': self.player_score >= self.WIN_SCORE,
            'rally': self.rally_count,
        }

    def render(self, screen: pygame.Surface) -> None:
        """Render the game with retro Pong aesthetic."""
        if self.headless:
            return

        assert self.player_paddle is not None
        assert self.ai_paddle is not None
        assert self.ball is not None

        # Black background
        screen.fill((0, 0, 0))

        # Draw center line (dashed) - brighter for visibility
        center_x = self.width // 2
        dash_height = 15
        dash_gap = 10
        line_color = (150, 150, 150)
        for y in range(0, self.height, dash_height + dash_gap):
            pygame.draw.rect(screen, line_color, (center_x - 2, y, 4, dash_height))

        # Draw ball trail (fading rectangles for retro look)
        trail_len = len(self._ball_trail)
        if trail_len > 0:
            for i, (tx, ty) in enumerate(self._ball_trail):
                alpha = int(100 * (i + 1) / trail_len)
                trail_size = max(2, self.ball.radius * (i + 1) // trail_len)
                trail_color = (alpha, alpha, alpha)
                pygame.draw.rect(screen, trail_color,
                               (int(tx) - trail_size, int(ty) - trail_size,
                                trail_size * 2, trail_size * 2))

        # Draw paddles
        # AI paddle (left) - slightly blue tint
        ai_color = (200, 200, 255)
        pygame.draw.rect(screen, ai_color, self.ai_paddle.rect)

        # Player paddle (right) - slightly green tint when hit
        if self._hit_flash_timer > 0:
            player_color = (200, 255, 200)
        else:
            player_color = (255, 255, 255)
        pygame.draw.rect(screen, player_color, self.player_paddle.rect)

        # Draw ball (white square for retro look)
        ball_color = (255, 255, 255)
        if self._hit_flash_timer > 0:
            ball_color = (255, 255, 150)  # Yellow flash on hit

        ball_rect = pygame.Rect(
            int(self.ball.x) - self.ball.radius,
            int(self.ball.y) - self.ball.radius,
            self.ball.radius * 2,
            self.ball.radius * 2
        )
        pygame.draw.rect(screen, ball_color, ball_rect)

        # Draw scores with labels
        if self._font and self._label_font:
            # Score flash effect
            score_color = (255, 255, 255)
            if self._score_flash_timer > 0:
                flash_intensity = int(255 * self._score_flash_timer / 30)
                score_color = (255, flash_intensity, flash_intensity)

            # AI label and score (left)
            ai_label = self._label_font.render("CPU", True, (150, 150, 200))
            screen.blit(ai_label, (self.width // 4 - ai_label.get_width() // 2, 15))

            ai_text = self._font.render(str(self.ai_score), True, score_color)
            screen.blit(ai_text, (self.width // 4 - ai_text.get_width() // 2, 35))

            # Player label and score (right)
            player_label = self._label_font.render("YOU", True, (150, 200, 150))
            screen.blit(player_label, (3 * self.width // 4 - player_label.get_width() // 2, 15))

            player_text = self._font.render(str(self.player_score), True, score_color)
            screen.blit(player_text, (3 * self.width // 4 - player_text.get_width() // 2, 35))

            # Rally counter
            if self.rally_count > 0 and self._small_font:
                rally_text = self._small_font.render(f"Rally: {self.rally_count}", True, (100, 100, 100))
                screen.blit(rally_text, (center_x - rally_text.get_width() // 2, self.height - 30))

        # Draw CRT scanline effect (lighter for less visual noise)
        for y in range(0, self.height, 3):
            pygame.draw.line(screen, (0, 0, 0, 30), (0, y), (self.width, y), 1)

        # Match point indicator
        if self._small_font:
            if self.player_score == self.WIN_SCORE - 1:
                mp_text = self._small_font.render("MATCH POINT!", True, (100, 255, 100))
                screen.blit(mp_text, (self.width - mp_text.get_width() - 10, self.height - 30))
            elif self.ai_score == self.WIN_SCORE - 1:
                mp_text = self._small_font.render("MATCH POINT!", True, (255, 100, 100))
                screen.blit(mp_text, (10, self.height - 30))

        # Game over message
        if self.game_over and self._font:
            if self.player_score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            text = self._font.render(msg, True, color)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            screen.blit(text, text_rect)

            # Final score
            if self._small_font:
                final = self._small_font.render(
                    f"Final: {self.player_score} - {self.ai_score}",
                    True, (200, 200, 200)
                )
                screen.blit(final, (self.width // 2 - final.get_width() // 2, self.height // 2 + 40))

    def close(self) -> None:
        """Clean up resources."""
        pass

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)

    # Human play helper
    def get_human_action(self, keys: dict) -> int:
        """Convert keyboard input to action for human play."""
        if keys.get(pygame.K_UP) or keys.get(pygame.K_w):
            return 0  # UP
        elif keys.get(pygame.K_DOWN) or keys.get(pygame.K_s):
            return 2  # DOWN
        return 1  # STAY


class VecPong:
    """
    Vectorized Pong environment for parallel game execution.

    Runs N independent game instances simultaneously for faster training.
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """Initialize vectorized environment."""
        self.num_envs = num_envs
        self.config = config
        self.headless = headless

        # Create N independent game instances
        self.envs = [Pong(config, headless=headless) for _ in range(num_envs)]

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
    pygame.display.set_caption("Pong - Human Play (UP/DOWN or W/S)")
    clock = pygame.time.Clock()

    game = Pong(config, ai_difficulty='medium')

    print("\n🏓 PONG - Human Play Mode")
    print("   Controls: UP/DOWN arrows or W/S keys")
    print("   Press R to restart, Q to quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    print("   Game reset!")

        # Get keyboard input
        keys = pygame.key.get_pressed()
        action = 1  # STAY
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 0  # UP
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 2  # DOWN

        # Step game
        state, reward, done, info = game.step(action)

        if done:
            if info['won']:
                print(f"   You WIN! Final: {info['score']} - {info['ai_score']}")
            else:
                print(f"   Game Over! Final: {info['score']} - {info['ai_score']}")
            pygame.time.wait(1500)
            game.reset()

        # Render
        game.render(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
