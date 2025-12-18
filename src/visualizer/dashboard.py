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

        # Cache gradient background surface to avoid redrawing every frame
        self._cached_gradient: Optional[pygame.Surface] = None
        self._create_gradient_surface()
    
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
            # Guard against NaN/Inf in loss smoothing
            if math.isfinite(loss):
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
        
        # Calculate layout - cards panel width based on available space
        cards_panel_width = 175
        chart_width = self.width - cards_panel_width - 30  # 30 for margins
        chart_height = self.height - 50
        chart_x = self.x + 10
        chart_y = self.y + 40
        
        # Draw main chart area
        chart_rect = pygame.Rect(chart_x, chart_y, chart_width, chart_height)
        self._draw_chart(screen, chart_rect)
        
        # Draw metric cards on the right (with epsilon gauge integrated)
        cards_x = self.x + chart_width + 20
        self._draw_metric_cards(screen, cards_x, chart_y, cards_panel_width)

    def _create_gradient_surface(self) -> None:
        """Create and cache the gradient background surface."""
        # Guard against zero dimensions
        if self.width <= 0 or self.height <= 0:
            self._cached_gradient = pygame.Surface((1, 1))
            return
        self._cached_gradient = pygame.Surface((self.width, self.height))
        for i in range(self.height):
            progress = i / self.height
            r = int(self.bg_color[0] + (self.panel_color[0] - self.bg_color[0]) * progress * 0.5)
            g = int(self.bg_color[1] + (self.panel_color[1] - self.bg_color[1]) * progress * 0.5)
            b = int(self.bg_color[2] + (self.panel_color[2] - self.bg_color[2]) * progress * 0.5)
            pygame.draw.line(self._cached_gradient, (r, g, b),
                           (0, i), (self.width, i))

    def _draw_background(self, screen: pygame.Surface) -> None:
        """Draw dashboard background with subtle gradient."""
        rect = pygame.Rect(self.x, self.y, self.width, self.height)

        # Blit cached gradient surface
        screen.blit(self._cached_gradient, (self.x, self.y))

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
        
        # Episode counter on the right
        episode_text = self.font_tiny.render(f"Episodes: {self.current_episode}", True, (120, 120, 140))
        episode_rect = episode_text.get_rect(right=self.x + self.width - 15, top=self.y + 12)
        screen.blit(episode_text, episode_rect)
    
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
        # Bug 74 note: Early return handles single data point case - need at least 2 points for line graph
        if len(data) < 2:
            return

        # Filter out NaN and Inf values - replace with 0
        clean_data = [v if math.isfinite(v) else 0.0 for v in data]

        points = [(rect.left, rect.bottom - 15)]
        for i, val in enumerate(clean_data):
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

        # Filter out NaN and Inf values - replace with 0
        clean_data = [v if math.isfinite(v) else 0.0 for v in data]

        points = []
        for i, val in enumerate(clean_data):
            x = rect.left + (i / max(len(data) - 1, 1)) * rect.width
            y = rect.bottom - 15 - (val / max(max_val, 1)) * (rect.height - 25)
            y = max(rect.top + 5, min(rect.bottom - 15, y))
            points.append((int(x), int(y)))
        
        if dashed:
            # Draw improved dashed line with better visibility
            dash_length = 6
            gap_length = 4
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                
                # Calculate line length and direction
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = max(1, (dx * dx + dy * dy) ** 0.5)
                
                # Normalize direction
                ux = dx / length
                uy = dy / length
                
                # Draw dashes along the line
                pos = 0
                drawing = True
                while pos < length:
                    if drawing:
                        dash_end = min(pos + dash_length, length)
                        x1 = int(start[0] + ux * pos)
                        y1 = int(start[1] + uy * pos)
                        x2 = int(start[0] + ux * dash_end)
                        y2 = int(start[1] + uy * dash_end)
                        pygame.draw.line(screen, color, (x1, y1), (x2, y2), thickness + 1)
                        pos = dash_end + gap_length
                    else:
                        pos += gap_length
                    drawing = not drawing
        else:
            if len(points) >= 2:
                pygame.draw.lines(screen, color, False, points, thickness)
    
    def _draw_chart_legend(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw chart legend with improved visibility."""
        legend_y = rect.bottom - 14
        legend_x = rect.left + 10
        
        # Background for legend
        legend_items = [
            ("Score", self.score_color, False),
            ("Avg", self.avg_color, False),
            ("Epsilon", self.epsilon_color, True),  # True = dashed line
        ]
        
        # Calculate total legend width for background (increased opacity for better contrast)
        total_width = len(legend_items) * 75 + 10
        legend_bg = pygame.Rect(legend_x - 5, legend_y - 8, total_width, 18)
        pygame.draw.rect(screen, (15, 18, 28, 230), legend_bg, border_radius=4)
        pygame.draw.rect(screen, (60, 65, 80), legend_bg, 1, border_radius=4)
        
        for label, color, is_dashed in legend_items:
            line_y = legend_y
            
            if is_dashed:
                # Draw dashed line for epsilon
                for dx in range(0, 18, 5):
                    pygame.draw.line(screen, color, 
                                   (legend_x + dx, line_y), 
                                   (legend_x + dx + 3, line_y), 2)
            else:
                # Solid line
                pygame.draw.line(screen, color, 
                               (legend_x, line_y), 
                               (legend_x + 18, line_y), 3)
            
            # Label text with better contrast
            text = self.font_tiny.render(label, True, color)
            screen.blit(text, (legend_x + 22, legend_y - 6))
            legend_x += 75
    
    def _draw_y_axis(self, screen: pygame.Surface, rect: pygame.Rect, max_val: float) -> None:
        """Draw Y-axis labels."""
        for i in range(5):
            val = int(max_val * (1 - i / 4))
            y = rect.top + 5 + (i * (rect.height - 25) // 4)
            text = self.font_tiny.render(str(val), True, (80, 85, 100))
            screen.blit(text, (rect.left + 3, y - 5))
    
    def _draw_metric_cards(self, screen: pygame.Surface, x: int, y: int, panel_width: int) -> None:
        """Draw metric cards on the right side with integrated epsilon gauge."""
        card_height = 22
        card_spacing = 24
        card_width = panel_width - 10
        
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
            card_rect = pygame.Rect(x, cy, card_width, card_height)
            pygame.draw.rect(screen, (22, 25, 38), card_rect, border_radius=4)
            pygame.draw.rect(screen, (45, 50, 65), card_rect, 1, border_radius=4)
            
            # Color indicator bar
            indicator_rect = pygame.Rect(x, cy, 3, card_height)
            pygame.draw.rect(screen, card.color, indicator_rect, border_radius=2)
            
            # Label
            label_text = self.font_tiny.render(card.label, True, (130, 135, 150))
            screen.blit(label_text, (x + 8, cy + 4))
            
            # Value - position based on card width
            value_x = x + min(70, card_width * 0.4)
            value_text = self.font_small.render(value, True, card.color)
            screen.blit(value_text, (value_x, cy + 3))
            
            # Trend indicator
            if indicator:
                trend_color = (100, 220, 130) if indicator == "↑" else (220, 100, 100) if indicator == "↓" else (150, 150, 150)
                if indicator == "★":
                    trend_color = self.avg_color
                trend_text = self.font_small.render(indicator, True, trend_color)
                trend_rect = trend_text.get_rect(right=x + card_width - 5, top=cy + 3)
                screen.blit(trend_text, trend_rect)
        
        # Draw epsilon gauge below the cards
        gauge_y = y + len(metrics) * card_spacing + 5
        self._draw_epsilon_gauge(screen, x, gauge_y, card_width)
    
    def _draw_epsilon_gauge(self, screen: pygame.Surface, x: int, y: int, gauge_width: int) -> None:
        """Draw a mini epsilon gauge with better visibility."""
        # Gauge dimensions - fit within the cards panel
        gauge_height = 12
        
        # Label above the gauge
        label = self.font_small.render("Exploration", True, self.epsilon_color)
        label_rect = label.get_rect(centerx=x + gauge_width // 2, bottom=y)
        screen.blit(label, label_rect)
        
        # Gauge below label
        gauge_y = y + 4
        
        # Background with subtle gradient effect
        bg_rect = pygame.Rect(x, gauge_y, gauge_width, gauge_height)
        pygame.draw.rect(screen, (25, 30, 45), bg_rect, border_radius=6)
        
        # Fill based on epsilon (exploration rate) - clamp to [0, 1]
        clamped_epsilon = max(0.0, min(1.0, self.current_epsilon))
        fill_width = max(2, int(gauge_width * clamped_epsilon))
        if fill_width > 0:
            # Gradient fill effect - brighter on left
            fill_rect = pygame.Rect(x, gauge_y, fill_width, gauge_height)
            pygame.draw.rect(screen, self.epsilon_color, fill_rect, border_radius=6)
            
            # Add highlight strip at top
            highlight_rect = pygame.Rect(x + 2, gauge_y + 2, max(0, fill_width - 4), 3)
            highlight_color = (min(255, self.epsilon_color[0] + 40), 
                             min(255, self.epsilon_color[1] + 40), 
                             min(255, self.epsilon_color[2] + 40))
            if highlight_rect.width > 0:
                pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=2)
        
        # Border with glow effect when epsilon is high
        border_color = (70, 85, 110)
        if self.current_epsilon > 0.5:
            pulse = 0.7 + 0.3 * math.sin(self.pulse_phase * 2)
            border_color = (int(70 + 30 * pulse), int(100 + 50 * pulse), int(140 + 50 * pulse))
        pygame.draw.rect(screen, border_color, bg_rect, 1, border_radius=6)
        
        # Percentage text inside/beside gauge
        pct_text = f"{self.current_epsilon * 100:.0f}%"
        pct_render = self.font_tiny.render(pct_text, True, (200, 210, 230))
        pct_rect = pct_render.get_rect(right=x + gauge_width - 4, centery=gauge_y + gauge_height // 2)
        screen.blit(pct_render, pct_rect)
    
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
