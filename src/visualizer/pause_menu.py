"""
Pause Menu
==========

Interactive pause overlay with button options.
"""

import pygame
from typing import Optional, List, Tuple, Dict


class PauseButton:
    """A button in the pause menu."""

    def __init__(self, label: str, action: str):
        """
        Create a pause button.

        Args:
            label: Button text
            action: Action identifier
        """
        self.label = label
        self.action = action
        self.rect = pygame.Rect(0, 0, 300, 50)
        self.hovered = False

    def update_position(self, x: int, y: int) -> None:
        """Update button position."""
        self.rect.topleft = (x, y)

    def contains_point(self, pos: Tuple[int, int]) -> bool:
        """Check if point is inside button."""
        return self.rect.collidepoint(pos)

    def render(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Render the button."""
        # Colors
        if self.hovered:
            bg_color = (70, 130, 180)  # Steel blue
            text_color = (255, 255, 255)
            border_color = (100, 160, 210)
        else:
            bg_color = (40, 40, 50)
            text_color = (200, 200, 200)
            border_color = (80, 80, 90)

        # Background
        bg_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, bg_color, bg_surface.get_rect(), border_radius=8)
        surface.blit(bg_surface, self.rect.topleft)

        # Border
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=8)

        # Text (centered)
        text_surface = font.render(self.label, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class PauseMenu:
    """
    Interactive pause overlay with buttons.

    Displays pause context (episode, score, etc.) and actionable buttons.
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the pause menu.

        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts
        self._title_font = pygame.font.Font(None, 72)
        self._context_font = pygame.font.Font(None, 24)
        self._button_font = pygame.font.Font(None, 28)

        # Buttons
        self.buttons: List[PauseButton] = [
            PauseButton("Resume", "resume"),
            PauseButton("Save Model", "save"),
            PauseButton("Return to Menu", "menu"),
            PauseButton("Quit", "quit"),
        ]

        self.selected_index = 0
        self._update_button_positions()

    def _update_button_positions(self) -> None:
        """Update button positions to be centered."""
        center_x = self.screen_width // 2
        start_y = self.screen_height // 2 + 20

        for i, button in enumerate(self.buttons):
            button.update_position(
                center_x - button.rect.width // 2,
                start_y + i * (button.rect.height + 15)
            )

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle keyboard/mouse input.

        Args:
            event: Pygame event

        Returns:
            Action string if button activated, None otherwise
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.buttons)
                return None

            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.buttons)
                return None

            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                return self.buttons[self.selected_index].action

            elif event.key == pygame.K_p:
                # P also resumes
                return "resume"

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            for i, button in enumerate(self.buttons):
                if button.contains_point(mouse_pos):
                    self.selected_index = i

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in self.buttons:
                if button.contains_point(mouse_pos):
                    return button.action

        return None

    def render(self, surface: pygame.Surface, context: Dict) -> None:
        """
        Render the pause menu.

        Args:
            surface: Surface to render onto
            context: Dictionary with context info (episode, score, epsilon, etc.)
        """
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        # Title
        title_text = self._title_font.render("PAUSED", True, (255, 200, 50))
        title_rect = title_text.get_rect(center=(center_x, center_y - 200))
        surface.blit(title_text, title_rect)

        # Context information
        context_lines = [
            f"Episode: {context.get('episode', 0)}",
            f"Score: {context.get('score', 0)} / Best: {context.get('best_score', 0)}",
            f"Epsilon: {context.get('epsilon', 0.0):.3f}",
        ]

        # Add training time if available
        if 'training_time' in context:
            training_time = context['training_time']
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            context_lines.append(f"Training Time: {hours}h {minutes}m")

        # Add memory buffer if available
        if 'memory_size' in context and 'memory_capacity' in context:
            mem_pct = (context['memory_size'] / context['memory_capacity']) * 100
            context_lines.append(f"Memory: {context['memory_size']:,} / {context['memory_capacity']:,} ({mem_pct:.0f}%)")

        # Render context
        y_offset = center_y - 130
        for line in context_lines:
            context_surface = self._context_font.render(line, True, (200, 200, 200))
            context_rect = context_surface.get_rect(center=(center_x, y_offset))
            surface.blit(context_surface, context_rect)
            y_offset += 30

        # Update hover states based on selected index
        for i, button in enumerate(self.buttons):
            button.hovered = (i == self.selected_index)

        # Render buttons
        for button in self.buttons:
            button.render(surface, self._button_font)

        # Hint text
        hint = "↑↓ Navigate  •  Enter/Space Select  •  P Resume"
        hint_surface = self._context_font.render(hint, True, (150, 150, 150))
        hint_rect = hint_surface.get_rect(center=(center_x, self.screen_height - 30))
        surface.blit(hint_surface, hint_rect)
