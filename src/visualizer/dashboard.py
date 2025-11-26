"""
Training Dashboard - Enhanced
=============================

Real-time visualization of training metrics with scrolling charts.

Features:
    - Live scrolling score chart with smoothed trend line
    - Multiple metrics on one chart (score, average, epsilon)
    - Mini metric cards with delta indicators
    - Color-coded performance zones
    - Loss chart with moving average
    - Win rate gauge

This provides immediate feedback on training progress and helps
diagnose issues (e.g., learning rate too high, epsilon not decaying).
"""

import pygame
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
import math
import time

import sys
sys.path.append('../..')
from config import Config


class MetricCard:
    """A small card displaying a single metric with trend."""
    
    def __init__(self, label: str, color: Tuple[int, int, int]):
        self.label = label
        self.color = color
        self.value = 0.0
        self.prev_value = 0.0
        self.history: deque = deque(maxlen=20)
        
    def update(self, value: float) -> None:
        self.prev_value = self.value
        self.value = value
        self.history.append(value)
    
    @property
    def delta(self) -> float:
        return self.value - self.prev_value
    
    @property
    def trend(self) -> str:
        if len(self.history) < 2:
            return "→"
        recent_avg = np.mean(list(self.history)[-5:])
        older_avg = np.mean(list(self.history)[:5]) if len(self.history) >= 5 else recent_avg
        if recent_avg > older_avg * 1.05:
            return "↑"
        elif recent_avg < older_avg * 0.95:
            return "↓"
        return "→"


class Dashboard:
    """
    Enhanced training metrics dashboard with live scrolling charts.
    
    Features:
        - Scrolling line charts that update in real-time
        - Multiple metrics overlaid (score, average, epsilon scaled)
        - Mini metric cards with trend indicators
        - Performance zone coloring
        - Smooth animations and transitions
    
    Example:
        >>> dashboard = Dashboard(config, x=0, y=400, width=800, height=200)
        >>> dashboard.update(episode, score, epsilon, loss, ...)
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
        
        # Enhanced color scheme
        self.bg_color = (12, 14, 24)
        self.panel_color = (18, 20, 32)
        self.grid_color = (35, 40, 55)
        self.text_color = (200, 205, 220)
        
        # Metric colors
        self.score_color = (46, 204, 113)       # Emerald green
        self.avg_color = (241, 196, 15)          # Gold
        self.epsilon_color = (52, 152, 219)      # Sky blue
        self.loss_color = (231, 76, 60)          # Red
        self.reward_color = (155, 89, 182)       # Purple
        
        # Performance zone colors
        self.zone_good = (46, 204, 113, 30)
        self.zone_ok = (241, 196, 15, 30)
        self.zone_bad = (231, 76, 60, 30)
        
        # Fonts
        pygame.font.init()
        self.font_tiny = pygame.font.Font(None, 16)
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 26)
        self.font_large = pygame.font.Font(None, 34)
        self.font_title = pygame.font.Font(None, 30)
        
        # Data storage with longer history
        self.max_history = 200
        self.scores: deque = deque(maxlen=self.max_history)
        self.avg_scores: deque = deque(maxlen=self.max_history)
        self.epsilons: deque = deque(maxlen=self.max_history)
        self.losses: deque = deque(maxlen=self.max_history)
        self.rewards: deque = deque(maxlen=self.max_history)
        
        # Smoothed values for display
        self.smoothed_score = 0.0
        self.smoothed_loss = 0.0
        
        # Metric cards
        self.cards = {
            'episode': MetricCard("Episode", self.text_color),
            'score': MetricCard("Score", self.score_color),
            'best': MetricCard("Best", self.avg_color),
            'epsilon': MetricCard("Epsilon", self.epsilon_color),
            'loss': MetricCard("Loss", self.loss_color),
            'winrate': MetricCard("Win%", (150, 255, 150)),
        }
        
        # Current stats
        self.current_episode = 0
        self.current_score = 0
        self.current_epsilon = 1.0
        self.current_loss = 0.0
        self.best_score = 0
        self.win_rate = 0.0
        self.total_bricks = 0
        self.wins = 0
        self.total_games = 0
        
        # Animation
        self.pulse_phase = 0.0
        self.scroll_offset = 0
    
    def update(
        self,
        episode: int,
        score: int,
        epsilon: float,
        loss: float,
        bricks_broken: int = 0,
        won: bool = False,
        reward: float = 0.0
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
            reward: Total episode reward
        """
        self.current_episode = episode
        self.current_score = score
        self.current_epsilon = epsilon
        self.current_loss = loss
        self.best_score = max(self.best_score, score)
        self.total_bricks += bricks_broken
        
        if won:
            self.wins += 1
        self.total_games += 1
        
        # Update history
        self.scores.append(score)
        self.epsilons.append(epsilon)
        self.losses.append(loss if loss > 0 else 0.001)
        self.rewards.append(reward)
        
        # Calculate running average with exponential smoothing
        if len(self.scores) > 0:
            window = min(100, len(self.scores))
            avg = np.mean(list(self.scores)[-window:])
            self.avg_scores.append(avg)
            
            # Smooth current values
            self.smoothed_score = self.smoothed_score * 0.8 + score * 0.2
            self.smoothed_loss = self.smoothed_loss * 0.9 + loss * 0.1
        
        # Calculate win rate
        if self.total_games > 0:
            self.win_rate = self.wins / self.total_games
        
        # Update metric cards
        self.cards['episode'].update(episode)
        self.cards['score'].update(score)
        self.cards['best'].update(self.best_score)
        self.cards['epsilon'].update(epsilon)
        self.cards['loss'].update(loss)
        self.cards['winrate'].update(self.win_rate * 100)
    
    def render(self, screen: pygame.Surface) -> None:
        """Render the enhanced dashboard."""
        self.pulse_phase = (self.pulse_phase + 0.05) % (2 * math.pi)
        
        # Draw background with gradient
        self._draw_background(screen)
        
        # Draw title bar
        self._draw_title_bar(screen)
        
        # Calculate layout
        chart_width = self.width - 200
        chart_height = self.height - 50
        chart_x = self.x + 10
        chart_y = self.y + 40
        
        # Draw main chart area
        chart_rect = pygame.Rect(chart_x, chart_y, chart_width, chart_height)
        self._draw_chart(screen, chart_rect)
        
        # Draw metric cards on the right
        cards_x = self.x + chart_width + 20
        self._draw_metric_cards(screen, cards_x, chart_y)
        
        # Draw mini epsilon gauge
        self._draw_epsilon_gauge(screen, cards_x + 80, self.y + self.height - 35)
    
    def _draw_background(self, screen: pygame.Surface) -> None:
        """Draw dashboard background with subtle gradient."""
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        # Gradient background
        for i in range(self.height):
            progress = i / self.height
            r = int(self.bg_color[0] + (self.panel_color[0] - self.bg_color[0]) * progress * 0.5)
            g = int(self.bg_color[1] + (self.panel_color[1] - self.bg_color[1]) * progress * 0.5)
            b = int(self.bg_color[2] + (self.panel_color[2] - self.bg_color[2]) * progress * 0.5)
            pygame.draw.line(screen, (r, g, b), 
                           (self.x, self.y + i), (self.x + self.width, self.y + i))
        
        # Border
        pygame.draw.rect(screen, (45, 50, 70), rect, 2, border_radius=5)
    
    def _draw_title_bar(self, screen: pygame.Surface) -> None:
        """Draw the title bar with training status."""
        # Title
        title = self.font_title.render("Training Progress", True, (100, 180, 255))
        screen.blit(title, (self.x + 12, self.y + 8))
        
        # Status indicator (pulsing)
        pulse = 0.6 + 0.4 * math.sin(self.pulse_phase)
        status_color = (int(100 * pulse), int(220 * pulse), int(130 * pulse))
        status_text = self.font_small.render("● TRAINING", True, status_color)
        screen.blit(status_text, (self.x + 200, self.y + 12))
        
        # FPS/Speed indicator
        speed_text = self.font_tiny.render(f"Episodes: {self.current_episode}", True, (120, 120, 140))
        screen.blit(speed_text, (self.x + self.width - 100, self.y + 12))
    
    def _draw_chart(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw the main scrolling chart."""
        # Background
        pygame.draw.rect(screen, (8, 10, 18), rect, border_radius=5)
        pygame.draw.rect(screen, (40, 45, 60), rect, 1, border_radius=5)
        
        # Grid lines
        num_h_lines = 4
        for i in range(1, num_h_lines + 1):
            y = rect.top + (i * rect.height // (num_h_lines + 1))
            pygame.draw.line(screen, self.grid_color, (rect.left + 5, y), (rect.right - 5, y), 1)
        
        num_v_lines = 6
        for i in range(1, num_v_lines + 1):
            x = rect.left + (i * rect.width // (num_v_lines + 1))
            pygame.draw.line(screen, (30, 35, 45), (x, rect.top + 5), (x, rect.bottom - 20), 1)
        
        if len(self.scores) < 2:
            text = self.font_medium.render("Collecting data...", True, (80, 80, 100))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
            return
        
        # Get max for scaling
        max_score = max(max(self.scores), 50)
        
        # Draw performance zones (subtle background)
        self._draw_performance_zones(screen, rect, max_score)
        
        # Draw filled area under score line
        self._draw_filled_area(screen, rect, list(self.scores), max_score, self.score_color)
        
        # Draw score line
        self._draw_line_graph(screen, rect, list(self.scores), max_score, self.score_color, 2)
        
        # Draw smoothed average line (thicker, different style)
        if len(self.avg_scores) >= 2:
            self._draw_line_graph(screen, rect, list(self.avg_scores), max_score, self.avg_color, 3)
        
        # Draw epsilon (scaled to fit)
        if len(self.epsilons) >= 2:
            scaled_epsilon = [e * max_score for e in self.epsilons]
            self._draw_line_graph(screen, rect, scaled_epsilon, max_score, self.epsilon_color, 1, dashed=True)
        
        # Legend
        self._draw_chart_legend(screen, rect)
        
        # Y-axis labels
        self._draw_y_axis(screen, rect, max_score)
    
    def _draw_performance_zones(self, screen: pygame.Surface, rect: pygame.Rect, max_score: float) -> None:
        """Draw subtle performance zone backgrounds."""
        # Good zone (top 25%)
        good_height = rect.height // 4
        good_rect = pygame.Rect(rect.left + 2, rect.top + 2, rect.width - 4, good_height)
        s = pygame.Surface((good_rect.width, good_rect.height), pygame.SRCALPHA)
        s.fill((46, 204, 113, 15))
        screen.blit(s, good_rect.topleft)
    
    def _draw_filled_area(self, screen: pygame.Surface, rect: pygame.Rect, 
                          data: List[float], max_val: float, color: Tuple[int, int, int]) -> None:
        """Draw a filled area under the line graph."""
        if len(data) < 2:
            return
        
        points = [(rect.left, rect.bottom - 15)]
        for i, val in enumerate(data):
            x = rect.left + (i / max(len(data) - 1, 1)) * rect.width
            y = rect.bottom - 15 - (val / max(max_val, 1)) * (rect.height - 25)
            y = max(rect.top + 5, min(rect.bottom - 15, y))
            points.append((int(x), int(y)))
        points.append((rect.right, rect.bottom - 15))
        
        if len(points) >= 3:
            # Create surface with alpha
            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            adjusted_points = [(p[0] - rect.left, p[1] - rect.top) for p in points]
            pygame.draw.polygon(s, (*color, 30), adjusted_points)
            screen.blit(s, rect.topleft)
    
    def _draw_line_graph(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        data: List[float],
        max_val: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
        dashed: bool = False
    ) -> None:
        """Draw a line graph with optional dashing."""
        if len(data) < 2:
            return
        
        points = []
        for i, val in enumerate(data):
            x = rect.left + (i / max(len(data) - 1, 1)) * rect.width
            y = rect.bottom - 15 - (val / max(max_val, 1)) * (rect.height - 25)
            y = max(rect.top + 5, min(rect.bottom - 15, y))
            points.append((int(x), int(y)))
        
        if dashed:
            # Draw dashed line
            for i in range(0, len(points) - 1, 2):
                if i + 1 < len(points):
                    pygame.draw.line(screen, color, points[i], points[i + 1], thickness)
        else:
            if len(points) >= 2:
                pygame.draw.lines(screen, color, False, points, thickness)
    
    def _draw_chart_legend(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw chart legend."""
        legend_y = rect.bottom - 12
        legend_x = rect.left + 10
        
        items = [
            ("Score", self.score_color),
            ("Avg", self.avg_color),
            ("ε", self.epsilon_color),
        ]
        
        for label, color in items:
            pygame.draw.line(screen, color, (legend_x, legend_y), (legend_x + 15, legend_y), 2)
            text = self.font_tiny.render(label, True, color)
            screen.blit(text, (legend_x + 20, legend_y - 5))
            legend_x += 60
    
    def _draw_y_axis(self, screen: pygame.Surface, rect: pygame.Rect, max_val: float) -> None:
        """Draw Y-axis labels."""
        for i in range(5):
            val = int(max_val * (1 - i / 4))
            y = rect.top + 5 + (i * (rect.height - 25) // 4)
            text = self.font_tiny.render(str(val), True, (80, 85, 100))
            screen.blit(text, (rect.left + 3, y - 5))
    
    def _draw_metric_cards(self, screen: pygame.Surface, x: int, y: int) -> None:
        """Draw metric cards on the right side."""
        card_height = 22
        card_spacing = 24
        
        metrics = [
            ('episode', f"{int(self.cards['episode'].value)}", ""),
            ('score', f"{int(self.current_score)}", self.cards['score'].trend),
            ('best', f"{int(self.best_score)}", "★"),
            ('epsilon', f"{self.current_epsilon:.3f}", self.cards['epsilon'].trend),
            ('loss', f"{self.current_loss:.4f}", self.cards['loss'].trend),
            ('winrate', f"{self.win_rate*100:.1f}%", self.cards['winrate'].trend),
        ]
        
        for i, (key, value, indicator) in enumerate(metrics):
            card = self.cards[key]
            cy = y + i * card_spacing
            
            # Card background
            card_rect = pygame.Rect(x, cy, 165, card_height)
            pygame.draw.rect(screen, (22, 25, 38), card_rect, border_radius=4)
            pygame.draw.rect(screen, (45, 50, 65), card_rect, 1, border_radius=4)
            
            # Color indicator bar
            indicator_rect = pygame.Rect(x, cy, 3, card_height)
            pygame.draw.rect(screen, card.color, indicator_rect, border_radius=2)
            
            # Label
            label_text = self.font_tiny.render(card.label, True, (130, 135, 150))
            screen.blit(label_text, (x + 8, cy + 4))
            
            # Value
            value_text = self.font_small.render(value, True, card.color)
            screen.blit(value_text, (x + 70, cy + 3))
            
            # Trend indicator
            if indicator:
                trend_color = (100, 220, 130) if indicator == "↑" else (220, 100, 100) if indicator == "↓" else (150, 150, 150)
                if indicator == "★":
                    trend_color = self.avg_color
                trend_text = self.font_small.render(indicator, True, trend_color)
                screen.blit(trend_text, (x + 145, cy + 3))
    
    def _draw_epsilon_gauge(self, screen: pygame.Surface, x: int, y: int) -> None:
        """Draw a mini epsilon gauge."""
        # Gauge background
        gauge_width = 80
        gauge_height = 8
        
        pygame.draw.rect(screen, (30, 35, 50), (x, y, gauge_width, gauge_height), border_radius=4)
        
        # Fill based on epsilon
        fill_width = int(gauge_width * self.current_epsilon)
        if fill_width > 0:
            fill_color = self.epsilon_color
            pygame.draw.rect(screen, fill_color, (x, y, fill_width, gauge_height), border_radius=4)
        
        # Border
        pygame.draw.rect(screen, (60, 70, 90), (x, y, gauge_width, gauge_height), 1, border_radius=4)
        
        # Label
        label = self.font_tiny.render("Exploration", True, (100, 105, 120))
        screen.blit(label, (x, y - 12))
    
    def render_mini(self, screen: pygame.Surface, x: int, y: int) -> None:
        """Render a minimal stats display."""
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
    print("Dashboard - Enhanced version")
    print("See main.py for usage example")
