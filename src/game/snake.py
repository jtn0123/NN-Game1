"""
Snake Game Implementation
=========================

Classic Snake game designed for AI training with modern visual effects.

Key Features:
- Grid-based state representation (~405 features)
- Modern visual style with particles and glow effects
- Growing state complexity as snake lengthens
- Full human playability support

Game Rules:
- Control the snake's direction (Arrow keys or WASD)
- Eat food to grow longer
- Don't hit walls or yourself
- Maximize score by eating as much food as possible
"""

import numpy as np
import pygame
from typing import Tuple, List, Optional, Deque
from collections import deque

from .base_game import BaseGame
from .particles import ParticleSystem
import sys
sys.path.append('..')
from config import Config


class Snake(BaseGame):
    """
    Snake game implementation with grid-based state.

    State representation (~405 features):
        - grid[400]: 20x20 grid with values:
            - 0.0 = empty
            - 0.33 = snake body
            - 0.66 = snake head
            - 1.0 = food
        - direction_x: Normalized direction x (-1, 0, 1) -> (0, 0.5, 1)
        - direction_y: Normalized direction y (-1, 0, 1) -> (0, 0.5, 1)
        - food_distance: normalized manhattan distance to food
        - steps_since_food: hunger pressure (normalized)
        - snake_length: normalized length

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
    """

    # Game constants
    GRID_SIZE = 20
    CELL_SIZE = 25  # Pixels per cell

    # Direction constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # Direction vectors (dy, dx)
    DIRECTION_VECTORS = {
        0: (-1, 0),   # UP
        1: (1, 0),    # DOWN
        2: (0, -1),   # LEFT
        3: (0, 1),    # RIGHT
    }

    # Opposite directions (can't reverse)
    OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

    # Timeout scales with grid size - gives more time to explore
    MAX_STEPS_WITHOUT_FOOD = 400  # Increased from 200

    def __init__(self, config: Optional[Config] = None, headless: bool = False):
        """
        Initialize the Snake game.

        Args:
            config: Configuration object (uses default if None)
            headless: If True, skip visual effects for faster training
        """
        self.config = config or Config()
        self.headless = headless

        # Screen dimensions (use config, center the grid)
        self.screen_width = self.config.SCREEN_WIDTH
        self.screen_height = self.config.SCREEN_HEIGHT

        # Calculate grid area dimensions
        self.grid_pixel_width = self.GRID_SIZE * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_SIZE * self.CELL_SIZE

        # Center the grid in the screen
        self.grid_offset_x = (self.screen_width - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_pixel_height) // 2

        # For BaseGame compatibility
        self.width = self.screen_width
        self.height = self.screen_height

        # Game state
        self.snake: Deque[Tuple[int, int]] = deque()  # List of (row, col) positions
        self.direction = self.RIGHT
        self.next_direction = self.RIGHT  # Buffer for input
        self.food_pos: Tuple[int, int] = (0, 0)
        self.score = 0
        self.game_over = False
        self.steps_since_food = 0

        # Pre-allocated arrays - use 405 features (grid + 5 metadata)
        self._grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self._state_array = np.zeros(self.GRID_SIZE * self.GRID_SIZE + 5, dtype=np.float32)

        # Normalization constants
        self._inv_grid_size = 1.0 / self.GRID_SIZE
        self._max_length = self.GRID_SIZE * self.GRID_SIZE
        self._inv_max_length = 1.0 / self._max_length
        self._max_distance = self.GRID_SIZE * 2  # Max manhattan distance
        self._inv_max_distance = 1.0 / self._max_distance
        self._inv_max_steps = 1.0 / self.MAX_STEPS_WITHOUT_FOOD

        # Visual effects timing
        self._food_pulse_phase = 0.0
        self._death_timer = 0

        # Visual effects (only in visual mode)
        if not headless:
            self.particles = ParticleSystem(max_particles=300)
            pygame.font.init()
            self._font = pygame.font.Font(None, 48)
            self._small_font = pygame.font.Font(None, 24)
            # Pre-render gradient background
            self._background = self._create_background()
        else:
            self.particles = None  # type: ignore
            self._font = None
            self._small_font = None
            self._background = None

        self.reset()

    @property
    def state_size(self) -> int:
        """State vector dimension."""
        # Grid (20x20=400) + direction_x + direction_y + food_distance + steps_since_food + length
        return self.GRID_SIZE * self.GRID_SIZE + 5

    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return 4  # UP, DOWN, LEFT, RIGHT

    def _create_background(self) -> pygame.Surface:
        """Create gradient background with subtle grid."""
        surface = pygame.Surface((self.screen_width, self.screen_height))

        # Dark gradient for full screen
        for y in range(self.screen_height):
            t = y / self.screen_height
            r = int(15 + t * 10)
            g = int(20 + t * 15)
            b = int(30 + t * 20)
            pygame.draw.line(surface, (r, g, b), (0, y), (self.screen_width, y))

        # Draw border around play area
        border_rect = pygame.Rect(
            self.grid_offset_x - 2,
            self.grid_offset_y - 2,
            self.grid_pixel_width + 4,
            self.grid_pixel_height + 4
        )
        pygame.draw.rect(surface, (60, 70, 80), border_rect, 2)

        # Subtle grid lines within play area
        grid_color = (35, 45, 55)
        for i in range(self.GRID_SIZE + 1):
            x = self.grid_offset_x + i * self.CELL_SIZE
            y = self.grid_offset_y + i * self.CELL_SIZE
            pygame.draw.line(surface, grid_color,
                           (x, self.grid_offset_y),
                           (x, self.grid_offset_y + self.grid_pixel_height), 1)
            pygame.draw.line(surface, grid_color,
                           (self.grid_offset_x, y),
                           (self.grid_offset_x + self.grid_pixel_width, y), 1)

        return surface

    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.score = 0
        self.game_over = False
        self.steps_since_food = 0
        self._death_timer = 0

        # Start snake in center, length 3, facing right
        center = self.GRID_SIZE // 2
        self.snake.clear()
        self.snake.append((center, center))
        self.snake.append((center, center - 1))
        self.snake.append((center, center - 2))

        self.direction = self.RIGHT
        self.next_direction = self.RIGHT

        # Spawn food
        self._spawn_food()

        # Clear particles
        if not self.headless:
            self.particles.clear()

        return self.get_state()

    def _spawn_food(self) -> None:
        """Spawn food at a random empty location."""
        snake_set = set(self.snake)

        # Use rejection sampling for efficiency when snake is small
        if len(self.snake) < self._max_length * 0.5:
            max_attempts = 100
            for _ in range(max_attempts):
                row = np.random.randint(0, self.GRID_SIZE)
                col = np.random.randint(0, self.GRID_SIZE)
                if (row, col) not in snake_set:
                    self.food_pos = (row, col)
                    return

        # Fallback: build list of empty cells (when snake is long)
        empty_cells = []
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if (row, col) not in snake_set:
                    empty_cells.append((row, col))

        if empty_cells:
            self.food_pos = empty_cells[np.random.randint(len(empty_cells))]
        else:
            # Snake fills entire grid - win!
            self.food_pos = (-1, -1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one game step."""
        if self.game_over:
            return self.get_state(), 0.0, True, self._get_info()

        reward = -0.01  # Small step penalty to encourage efficiency

        # Buffer direction change (prevents 180-degree turns in same frame)
        if action != self.OPPOSITE.get(self.direction, -1):
            self.next_direction = action

        # Apply buffered direction
        self.direction = self.next_direction

        # Calculate new head position
        dy, dx = self.DIRECTION_VECTORS[self.direction]
        head_row, head_col = self.snake[0]
        new_head = (head_row + dy, head_col + dx)

        # Calculate distance to food before move
        old_dist = abs(head_row - self.food_pos[0]) + abs(head_col - self.food_pos[1])

        # Check for collisions
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= self.GRID_SIZE):
            self.game_over = True
            reward = -10.0
            self._death_timer = 60
            return self.get_state(), reward, True, self._get_info()

        # Self collision (check against body, excluding tail which will move)
        # Convert to set for O(1) lookup, exclude last element (tail)
        # Bug 75 clarification: This check runs BEFORE appendleft(), and excludes the tail
        # because the tail will pop() if food isn't eaten, so new_head can safely occupy that space
        # Guard: Only check collision if snake has > 1 segment
        body_set = set(list(self.snake)[:-1]) if len(self.snake) > 1 else set()
        if new_head in body_set:
            self.game_over = True
            reward = -10.0
            self._death_timer = 60

            # Death particles
            if not self.headless:
                px = self.grid_offset_x + new_head[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.grid_offset_y + new_head[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                self.particles.emit_brick_break(px, py, (255, 50, 50), count=30)

            return self.get_state(), reward, True, self._get_info()

        # Move snake
        self.snake.appendleft(new_head)

        # Check for food
        if new_head == self.food_pos:
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0

            # Food eaten particles
            if not self.headless:
                px = self.grid_offset_x + new_head[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.grid_offset_y + new_head[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                self.particles.emit_brick_break(px, py, (50, 255, 100), count=20)

            self._spawn_food()

            # Check for win (filled entire grid)
            if self.food_pos == (-1, -1):
                self.game_over = True
                reward = 100.0  # Big win bonus
        else:
            # Remove tail if didn't eat
            self.snake.pop()
            self.steps_since_food += 1

            # Distance-based reward shaping
            new_dist = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
            if new_dist < old_dist:
                reward += 0.1  # Moving closer to food

        # Starvation check (scaled by snake length - longer snake gets more time)
        timeout = self.MAX_STEPS_WITHOUT_FOOD + len(self.snake) * 5
        if self.steps_since_food >= timeout:
            self.game_over = True
            reward = -5.0

        # Update particles
        if not self.headless:
            self.particles.update()
            self._food_pulse_phase += 0.1

        return self.get_state(), reward, self.game_over, self._get_info()

    def get_state(self) -> np.ndarray:
        """Get the current game state as a normalized vector."""
        # Build grid representation
        self._grid.fill(0.0)

        # Mark snake body
        for i, (row, col) in enumerate(self.snake):
            if i == 0:
                self._grid[row, col] = 0.66  # Head
            else:
                self._grid[row, col] = 0.33  # Body

        # Mark food
        if self.food_pos != (-1, -1):
            self._grid[self.food_pos[0], self.food_pos[1]] = 1.0

        # Flatten grid into state
        self._state_array[:self.GRID_SIZE * self.GRID_SIZE] = self._grid.flatten()

        # Additional features
        base_idx = self.GRID_SIZE * self.GRID_SIZE

        # Direction as x and y components (better encoding than single value)
        dy, dx = self.DIRECTION_VECTORS[self.direction]
        self._state_array[base_idx] = (dx + 1) * 0.5  # Map -1,0,1 to 0,0.5,1
        self._state_array[base_idx + 1] = (dy + 1) * 0.5

        # Food distance (normalized manhattan distance)
        if self.food_pos != (-1, -1) and len(self.snake) > 0:
            head = self.snake[0]
            distance = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])
            self._state_array[base_idx + 2] = distance * self._inv_max_distance
        else:
            self._state_array[base_idx + 2] = 0.0

        # Steps since food (hunger pressure)
        self._state_array[base_idx + 3] = min(1.0, self.steps_since_food * self._inv_max_steps)

        # Snake length (normalized)
        self._state_array[base_idx + 4] = len(self.snake) * self._inv_max_length

        return self._state_array.copy()

    def _get_info(self) -> dict:
        """Get additional game information."""
        return {
            'score': self.score,
            'length': len(self.snake),
            'lives': 1 if not self.game_over else 0,
            'won': len(self.snake) >= self._max_length
        }

    def render(self, screen: pygame.Surface) -> None:
        """Render the game with modern visual effects."""
        if self.headless:
            return

        # Draw background
        if self._background:
            screen.blit(self._background, (0, 0))
        else:
            screen.fill((20, 25, 35))

        # Draw food with glow/pulse
        if self.food_pos != (-1, -1):
            fx = self.grid_offset_x + self.food_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            fy = self.grid_offset_y + self.food_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2

            # Pulsing glow (pre-computed phase)
            pulse = abs(np.sin(self._food_pulse_phase)) * 0.5 + 0.5
            glow_radius = int(self.CELL_SIZE * 0.6 + pulse * 8)

            # Draw glow layers (no alpha, just RGB)
            for i, (r, g, b) in enumerate([(30, 80, 30), (50, 120, 50), (80, 180, 80)]):
                radius = glow_radius - i * 4
                if radius > 0:
                    pygame.draw.circle(screen, (r, g, b), (fx, fy), radius)

            # Food core
            food_rect = pygame.Rect(
                self.grid_offset_x + self.food_pos[1] * self.CELL_SIZE + 4,
                self.grid_offset_y + self.food_pos[0] * self.CELL_SIZE + 4,
                self.CELL_SIZE - 8,
                self.CELL_SIZE - 8
            )
            pygame.draw.rect(screen, (100, 255, 100), food_rect, border_radius=6)

        # Draw snake with gradient
        if len(self.snake) > 0:
            snake_len = len(self.snake)
            for i, (row, col) in enumerate(self.snake):
                # Calculate color gradient (head brightest)
                t = 1.0 - (i / max(1, snake_len - 1)) * 0.7
                r = int(50 + t * 100)
                g = int(180 + t * 75)
                b = int(50 + t * 80)
                color = (r, g, b)

                x = self.grid_offset_x + col * self.CELL_SIZE
                y = self.grid_offset_y + row * self.CELL_SIZE

                # Body segment
                segment_rect = pygame.Rect(x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                pygame.draw.rect(screen, color, segment_rect, border_radius=4)

                # Head has extra glow and eyes
                if i == 0:
                    # Glow around head
                    glow_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(screen, (80, 200, 80), glow_rect, 2, border_radius=5)

                    # Eyes based on direction
                    dy, dx = self.DIRECTION_VECTORS[self.direction]
                    eye_offset = 5
                    eye_size = 4

                    if dx == 1:  # Right
                        eye1 = (x + self.CELL_SIZE - eye_offset, y + eye_offset)
                        eye2 = (x + self.CELL_SIZE - eye_offset, y + self.CELL_SIZE - eye_offset - eye_size)
                    elif dx == -1:  # Left
                        eye1 = (x + eye_offset - eye_size, y + eye_offset)
                        eye2 = (x + eye_offset - eye_size, y + self.CELL_SIZE - eye_offset - eye_size)
                    elif dy == -1:  # Up
                        eye1 = (x + eye_offset, y + eye_offset - eye_size)
                        eye2 = (x + self.CELL_SIZE - eye_offset - eye_size, y + eye_offset - eye_size)
                    else:  # Down
                        eye1 = (x + eye_offset, y + self.CELL_SIZE - eye_offset)
                        eye2 = (x + self.CELL_SIZE - eye_offset - eye_size, y + self.CELL_SIZE - eye_offset)

                    pygame.draw.rect(screen, (255, 255, 255),
                                   (eye1[0], eye1[1], eye_size, eye_size), border_radius=1)
                    pygame.draw.rect(screen, (255, 255, 255),
                                   (eye2[0], eye2[1], eye_size, eye_size), border_radius=1)

        # Draw particles
        self.particles.draw(screen)

        # Draw HUD
        if self._font:
            # Score
            score_text = self._font.render(f"Score: {self.score}", True, (200, 255, 200))
            screen.blit(score_text, (10, 10))

            # Length
            if self._small_font:
                length_text = self._small_font.render(f"Length: {len(self.snake)}", True, (150, 200, 150))
                screen.blit(length_text, (10, 55))

                # Hunger indicator
                timeout = self.MAX_STEPS_WITHOUT_FOOD + len(self.snake) * 5
                hunger_pct = self.steps_since_food / timeout
                if hunger_pct > 0.7:
                    hunger_color = (255, 100, 100)
                elif hunger_pct > 0.4:
                    hunger_color = (255, 200, 100)
                else:
                    hunger_color = (100, 200, 100)

                # Hunger bar
                bar_width = 100
                bar_height = 8
                bar_x = self.screen_width - bar_width - 10
                bar_y = 15
                pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                fill_width = int(bar_width * (1 - hunger_pct))
                pygame.draw.rect(screen, hunger_color, (bar_x, bar_y, fill_width, bar_height))

                hunger_label = self._small_font.render("Energy", True, (150, 150, 150))
                screen.blit(hunger_label, (bar_x, bar_y + 12))

        # Game over message
        if self.game_over and self._font:
            if len(self.snake) >= self._max_length:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            text = self._font.render(msg, True, color)
            text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))

            # Background box (solid, no alpha)
            box_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(screen, (20, 20, 30), box_rect)
            pygame.draw.rect(screen, color, box_rect, 3)

            screen.blit(text, text_rect)

            # Final stats
            if self._small_font:
                final_text = self._small_font.render(
                    f"Final Length: {len(self.snake)} | Score: {self.score}",
                    True, (180, 180, 180)
                )
                screen.blit(final_text, (self.screen_width // 2 - final_text.get_width() // 2,
                                         self.screen_height // 2 + 35))

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
            return self.UP
        elif keys.get(pygame.K_DOWN) or keys.get(pygame.K_s):
            return self.DOWN
        elif keys.get(pygame.K_LEFT) or keys.get(pygame.K_a):
            return self.LEFT
        elif keys.get(pygame.K_RIGHT) or keys.get(pygame.K_d):
            return self.RIGHT
        return self.direction  # Keep current direction


class VecSnake:
    """
    Vectorized Snake environment for parallel game execution.

    Runs N independent game instances simultaneously for faster training.
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """Initialize vectorized environment."""
        self.num_envs = num_envs
        self.config = config
        self.headless = headless

        # Create N independent game instances
        self.envs = [Snake(config, headless=headless) for _ in range(num_envs)]

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

    game = Snake(config)
    screen = pygame.display.set_mode((game.screen_width, game.screen_height))
    pygame.display.set_caption("Snake - Human Play (Arrow Keys or WASD)")
    clock = pygame.time.Clock()

    print("\n🐍 SNAKE - Human Play Mode")
    print("   Controls: Arrow keys or WASD")
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
        action = game.direction  # Default to current direction
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = Snake.UP
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = Snake.DOWN
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = Snake.LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = Snake.RIGHT

        # Step game
        state, reward, done, info = game.step(action)

        if done:
            if info['won']:
                print(f"   You WIN! Length: {info['length']}, Score: {info['score']}")
            else:
                print(f"   Game Over! Length: {info['length']}, Score: {info['score']}")
            pygame.time.wait(1000)
            game.reset()

        # Render
        game.render(screen)
        pygame.display.flip()
        clock.tick(10)  # Slower for snake

    pygame.quit()
