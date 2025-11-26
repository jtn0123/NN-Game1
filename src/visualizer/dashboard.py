"""
Training Dashboard
==================

Displays training metrics and statistics in real-time.

Features:
    - Episode score history (line graph)
    - Running average score
    - Epsilon decay visualization
    - Loss over time
    - Win rate display
    - Current episode stats

This provides immediate feedback on training progress and helps
diagnose issues (e.g., learning rate too high, epsilon not decaying).
"""

import pygame
import numpy as np
from typing import Optional, List, Tuple
import math

import sys
sys.path.append('../..')
from config import Config


class Dashboard:
    """
    Training metrics dashboard.
    
    Displays:
        - Score over episodes (line chart)
        - Running average score
        - Current epsilon
        - Average loss
        - Win rate
        - Bricks broken stats
    
    Example:
        >>> dashboard = Dashboard(config, x=0, y=400, width=500, height=200)
        >>> dashboard.update(metrics)
        >>> dashboard.render(screen)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        x: int = 0,
        y: int = 0,
        width: int = 500,
        height: int = 200
    ):
        """
        Initialize the dashboard.
        
        Args:
            config: Configuration object
            x: X position
            y: Y position
            width: Dashboard width
            height: Dashboard height
        """
        self.config = config or Config()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Colors
        self.bg_color = (20, 20, 35)
        self.grid_color = (40, 40, 60)
        self.text_color = (200, 200, 200)
        self.score_color = (46, 204, 113)      # Green
        self.avg_color = (241, 196, 15)         # Yellow
        self.epsilon_color = (52, 152, 219)     # Blue
        self.loss_color = (231, 76, 60)         # Red
        
        # Fonts
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Data storage
        self.scores: List[int] = []
        self.avg_scores: List[float] = []
        self.epsilons: List[float] = []
        self.losses: List[float] = []
        self.max_history = self.config.PLOT_HISTORY_LENGTH
        
        # Current stats
        self.current_episode = 0
        self.current_score = 0
        self.current_epsilon = 1.0
        self.current_loss = 0.0
        self.best_score = 0
        self.win_rate = 0.0
        self.total_bricks = 0
    
    def update(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: float,
        bricks_broken: int = 0,
        won: bool = False
    ) -> None:
        """
        Update dashboard with new episode data.
        
        Args:
            episode: Current episode number
            score: Episode score
            epsilon: Current exploration rate
            loss: Average loss this episode
            bricks_broken: Number of bricks broken
            won: Whether the game was won
        """
        self.current_episode = episode
        self.current_score = score
        self.current_epsilon = epsilon
        self.current_loss = loss
        self.best_score = max(self.best_score, score)
        self.total_bricks += bricks_broken
        
        # Update history
        self.scores.append(score)
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        
        # Calculate running average
        window = min(100, len(self.scores))
        avg = np.mean(self.scores[-window:])
        self.avg_scores.append(avg)
        
        # Calculate win rate
        if len(self.scores) > 0:
            recent_wins = sum(1 for s in self.scores[-100:] if s >= 300)  # Approximate win threshold
            self.win_rate = recent_wins / min(100, len(self.scores))
        
        # Trim history
        if len(self.scores) > self.max_history:
            self.scores = self.scores[-self.max_history:]
            self.avg_scores = self.avg_scores[-self.max_history:]
            self.epsilons = self.epsilons[-self.max_history:]
            self.losses = self.losses[-self.max_history:]
    
    def render(self, screen: pygame.Surface) -> None:
        """
        Render the dashboard.
        
        Args:
            screen: Pygame surface to draw on
        """
        # Draw background
        self._draw_background(screen)
        
        # Draw title
        self._draw_title(screen)
        
        # Draw score chart
        chart_rect = pygame.Rect(
            self.x + 10, 
            self.y + 40, 
            self.width - 180, 
            self.height - 60
        )
        self._draw_chart(screen, chart_rect)
        
        # Draw stats panel
        stats_rect = pygame.Rect(
            self.x + self.width - 160,
            self.y + 40,
            150,
            self.height - 60
        )
        self._draw_stats(screen, stats_rect)
    
    def _draw_background(self, screen: pygame.Surface) -> None:
        """Draw dashboard background."""
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, rect)
        pygame.draw.rect(screen, (50, 50, 70), rect, 2)
    
    def _draw_title(self, screen: pygame.Surface) -> None:
        """Draw dashboard title."""
        title = self.font_large.render("Training Progress", True, (100, 200, 255))
        screen.blit(title, (self.x + 10, self.y + 8))
    
    def _draw_chart(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw the score history chart."""
        # Background
        pygame.draw.rect(screen, (15, 15, 25), rect)
        pygame.draw.rect(screen, (50, 50, 70), rect, 1)
        
        # Grid lines
        for i in range(5):
            y = rect.top + i * rect.height // 4
            pygame.draw.line(
                screen, self.grid_color,
                (rect.left, y), (rect.right, y), 1
            )
        
        if len(self.scores) < 2:
            # Not enough data
            text = self.font_medium.render("Collecting data...", True, (100, 100, 100))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
            return
        
        # Normalize data
        max_score = max(max(self.scores), 1)
        
        # Draw score line
        self._draw_line_graph(
            screen, rect, self.scores, 
            max_score, self.score_color, alpha=180
        )
        
        # Draw average line
        self._draw_line_graph(
            screen, rect, self.avg_scores,
            max_score, self.avg_color, thickness=2
        )
        
        # Legend
        legend_y = rect.bottom - 15
        
        # Score legend
        pygame.draw.line(
            screen, self.score_color,
            (rect.left + 10, legend_y),
            (rect.left + 30, legend_y), 2
        )
        text = self.font_small.render("Score", True, self.score_color)
        screen.blit(text, (rect.left + 35, legend_y - 6))
        
        # Average legend
        pygame.draw.line(
            screen, self.avg_color,
            (rect.left + 90, legend_y),
            (rect.left + 110, legend_y), 2
        )
        text = self.font_small.render("Avg (100)", True, self.avg_color)
        screen.blit(text, (rect.left + 115, legend_y - 6))
    
    def _draw_line_graph(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        data: List[float],
        max_val: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
        alpha: int = 255
    ) -> None:
        """Draw a line graph of the data."""
        if len(data) < 2:
            return
        
        points = []
        for i, val in enumerate(data):
            x = rect.left + (i / max(len(data) - 1, 1)) * rect.width
            y = rect.bottom - (val / max(max_val, 1)) * (rect.height - 20)
            y = max(rect.top, min(rect.bottom, y))
            points.append((int(x), int(y)))
        
        if len(points) >= 2:
            # Draw with alpha (create temp surface)
            if alpha < 255:
                for i in range(len(points) - 1):
                    pygame.draw.line(
                        screen, color,
                        points[i], points[i + 1], thickness
                    )
            else:
                pygame.draw.lines(screen, color, False, points, thickness)
    
    def _draw_stats(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw statistics panel."""
        # Background
        pygame.draw.rect(screen, (25, 25, 40), rect, border_radius=5)
        pygame.draw.rect(screen, (60, 60, 80), rect, 1, border_radius=5)
        
        stats = [
            ("Episode", f"{self.current_episode}", self.text_color),
            ("Score", f"{self.current_score}", self.score_color),
            ("Best", f"{self.best_score}", self.avg_color),
            ("Epsilon", f"{self.current_epsilon:.3f}", self.epsilon_color),
            ("Loss", f"{self.current_loss:.4f}", self.loss_color),
            ("Win Rate", f"{self.win_rate*100:.1f}%", (150, 255, 150)),
        ]
        
        y_offset = 8
        for label, value, color in stats:
            # Label
            label_text = self.font_small.render(label, True, (150, 150, 150))
            screen.blit(label_text, (rect.left + 10, rect.top + y_offset))
            
            # Value
            value_text = self.font_medium.render(value, True, color)
            screen.blit(value_text, (rect.left + 80, rect.top + y_offset - 2))
            
            y_offset += 22
    
    def render_mini(
        self,
        screen: pygame.Surface,
        x: int,
        y: int
    ) -> None:
        """
        Render a minimal stats display.
        
        Args:
            screen: Pygame surface
            x: X position
            y: Y position
        """
        # Compact stats display
        text = (
            f"Ep: {self.current_episode} | "
            f"Score: {self.current_score} | "
            f"Best: {self.best_score} | "
            f"ε: {self.current_epsilon:.2f}"
        )
        
        rendered = self.font_medium.render(text, True, self.text_color)
        
        # Background
        bg_rect = rendered.get_rect(topleft=(x, y)).inflate(10, 6)
        pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect, border_radius=3)
        
        screen.blit(rendered, (x, y))


# Testing
if __name__ == "__main__":
    print("Dashboard - import and use with pygame")
    print("See main.py for usage example")

