"""
Training HUD (Heads-Up Display)
================================

On-screen overlay showing training statistics and progress during AI training.
"""

import pygame
import numpy as np
from typing import List, Optional
from config import Config


class TrainingHUD:
    """
    On-screen training statistics overlay.

    Displays:
    - Episode counter with progress
    - Score and best score
    - Epsilon (exploration rate) bar
    - Current action indicator
    - Training progress bar
    - Speed multiplier
    """

    def __init__(self, config: Config):
        """
        Initialize the HUD.

        Args:
            config: Configuration object
        """
        self.config = config

        # Fonts
        self._font_small = pygame.font.Font(None, 20)
        self._font_medium = pygame.font.Font(None, 24)
        self._font_large = pygame.font.Font(None, 28)

        # Colors
        self.text_color = (220, 220, 220)
        self.text_dim = (150, 150, 150)
        self.accent_color = (52, 152, 219)  # Blue
        self.good_color = (46, 204, 113)  # Green
        self.warn_color = (241, 196, 15)  # Yellow

        # State
        self.enabled = getattr(config, 'HUD_ENABLED', True)
        self.opacity = getattr(config, 'HUD_OPACITY', 0.8)

        # Bug 98: Use config opacity instead of hardcoded 180
        bg_alpha = int(self.opacity * 255) if self.opacity <= 1.0 else int(self.opacity)
        self.bg_color = (0, 0, 0, bg_alpha)

        # Bug 90: Smooth score color transition state
        self._current_score_color = list(self.text_color)
        self._color_lerp_speed = 0.1

    def render(
        self,
        surface: pygame.Surface,
        episode: int,
        score: int,
        best_score: int,
        epsilon: float,
        loss: float,
        speed: float,
        max_episodes: int,
        selected_action: Optional[int],
        action_labels: List[str]
    ) -> None:
        """
        Render all HUD elements onto the surface.

        Args:
            surface: Pygame surface to render onto
            episode: Current episode number
            score: Current episode score
            best_score: Best score achieved
            epsilon: Current epsilon value (0-1)
            loss: Recent training loss
            speed: Speed multiplier
            max_episodes: Maximum episodes (0 = unlimited)
            selected_action: Currently selected action index (or None)
            action_labels: List of action labels
        """
        if not self.enabled:
            return

        # Render each HUD element
        self._render_episode_counter(surface, episode, max_episodes)
        self._render_score_display(surface, score, best_score)
        self._render_epsilon_bar(surface, epsilon)
        self._render_speed_indicator(surface, speed)

        if selected_action is not None:
            self._render_action_indicator(surface, selected_action, action_labels)

        if max_episodes > 0:
            self._render_progress_bar(surface, episode, max_episodes)

    def _render_episode_counter(self, surface: pygame.Surface, episode: int, max_episodes: int) -> None:
        """Render episode counter in top-left."""
        if max_episodes > 0:
            text = f"Episode: {episode:,} / {max_episodes:,}"
        else:
            text = f"Episode: {episode:,}"

        text_surface = self._font_medium.render(text, True, self.text_color)

        # Background
        bg_rect = text_surface.get_rect(topleft=(10, 10))
        bg_rect.inflate_ip(16, 8)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.bg_color, bg_surface.get_rect(), border_radius=5)
        surface.blit(bg_surface, bg_rect.topleft)

        # Text
        surface.blit(text_surface, (18, 14))

    def _render_score_display(self, surface: pygame.Surface, score: int, best_score: int) -> None:
        """Render score display below episode counter."""
        text = f"Score: {score:,}  |  Best: {best_score:,}"

        # Bug 90: Smooth color transition instead of instant flip
        target_color = self.good_color if score >= best_score * 0.8 else self.text_color
        for i in range(3):
            self._current_score_color[i] += (target_color[i] - self._current_score_color[i]) * self._color_lerp_speed
        color = tuple(int(c) for c in self._current_score_color)

        text_surface = self._font_small.render(text, True, color)

        # Background
        bg_rect = text_surface.get_rect(topleft=(10, 42))
        bg_rect.inflate_ip(16, 6)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.bg_color, bg_surface.get_rect(), border_radius=5)
        surface.blit(bg_surface, bg_rect.topleft)

        # Text
        surface.blit(text_surface, (18, 45))

    def _render_epsilon_bar(self, surface: pygame.Surface, epsilon: float) -> None:
        """Render epsilon gauge below score."""
        label = "Exploration:"
        label_surface = self._font_small.render(label, True, self.text_dim)

        # Bar dimensions
        bar_x = 18
        bar_y = 74
        bar_width = 150
        bar_height = 12

        # Background
        bg_rect = pygame.Rect(10, 68, 170, 24)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.bg_color, bg_surface.get_rect(), border_radius=5)
        surface.blit(bg_surface, bg_rect.topleft)

        # Label
        surface.blit(label_surface, (bar_x, bar_y - 14))

        # Bar background
        pygame.draw.rect(surface, (40, 40, 40), (bar_x, bar_y, bar_width, bar_height), border_radius=3)

        # Bar fill
        fill_width = int(bar_width * epsilon)
        if fill_width > 0:
            fill_color = self.warn_color if epsilon > 0.5 else self.accent_color
            pygame.draw.rect(surface, fill_color, (bar_x, bar_y, fill_width, bar_height), border_radius=3)

        # Percentage text
        pct_text = f"{epsilon*100:.0f}%"
        pct_surface = self._font_small.render(pct_text, True, self.text_color)
        surface.blit(pct_surface, (bar_x + bar_width + 8, bar_y - 2))

    def _render_speed_indicator(self, surface: pygame.Surface, speed: float) -> None:
        """Render speed multiplier in top-right."""
        if speed == 1.0:
            text = "Speed: 1x"
        elif speed < 10:
            text = f"Speed: {speed:.1f}x"
        else:
            text = f"Speed: {speed:.0f}x"

        color = self.warn_color if speed > 10 else self.text_color
        text_surface = self._font_medium.render(text, True, color)

        # Position in top-right
        screen_width = surface.get_width()
        bg_rect = text_surface.get_rect(topright=(screen_width - 10, 10))
        bg_rect.inflate_ip(16, 8)

        # Background
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.bg_color, bg_surface.get_rect(), border_radius=5)
        surface.blit(bg_surface, bg_rect.topleft)

        # Text
        surface.blit(text_surface, (bg_rect.left + 8, 14))

    def _render_action_indicator(
        self,
        surface: pygame.Surface,
        selected_action: int,
        action_labels: List[str]
    ) -> None:
        """Render current action indicator at bottom-center."""
        # Bug 68 fix: Check for negative action indices to prevent wrong action via Python's negative indexing
        if selected_action < 0 or selected_action >= len(action_labels):
            return

        action_name = action_labels[selected_action]
        text = f"Action: {action_name}"

        text_surface = self._font_medium.render(text, True, self.accent_color)

        # Position at bottom-center
        screen_width = surface.get_width()
        screen_height = surface.get_height()

        bg_rect = text_surface.get_rect(centerx=screen_width // 2, bottom=screen_height - 35)
        bg_rect.inflate_ip(20, 10)

        # Background with accent border
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.bg_color, bg_surface.get_rect(), border_radius=6)
        surface.blit(bg_surface, bg_rect.topleft)
        pygame.draw.rect(surface, self.accent_color, bg_rect, 2, border_radius=6)

        # Text
        text_rect = text_surface.get_rect(center=bg_rect.center)
        surface.blit(text_surface, text_rect)

    def _render_progress_bar(self, surface: pygame.Surface, episode: int, max_episodes: int) -> None:
        """Render training progress bar at bottom of screen."""
        # Bar dimensions
        screen_width = surface.get_width()
        screen_height = surface.get_height()

        # Bug 72 fix: Ensure bar_width is never negative for very small windows
        bar_width = max(10, screen_width - 40)
        bar_height = 8
        bar_x = 20
        bar_y = screen_height - 12

        # Calculate progress
        progress = min(episode / max_episodes, 1.0) if max_episodes > 0 else 0.0

        # Bar background
        pygame.draw.rect(surface, (40, 40, 40), (bar_x, bar_y, bar_width, bar_height), border_radius=4)

        # Bar fill
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            # Color gradient based on progress
            if progress < 0.3:
                fill_color = self.accent_color
            elif progress < 0.7:
                fill_color = self.warn_color
            else:
                fill_color = self.good_color

            pygame.draw.rect(surface, fill_color, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

        # Progress percentage (small text above bar)
        if progress > 0:
            pct_text = f"{progress*100:.1f}%"
            pct_surface = self._font_small.render(pct_text, True, self.text_dim)
            pct_rect = pct_surface.get_rect(centerx=screen_width // 2, bottom=bar_y - 2)
            surface.blit(pct_surface, pct_rect)
