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
            PauseButton("Resume (P)", "resume"),
            PauseButton("Save Model (S)", "save"),
            PauseButton("Game Selector (H)", "menu"),
            PauseButton("Quit (Q)", "quit"),
        ]

        self.selected_index = 0
        self._update_button_positions()

        # Quit confirmation state
        self._confirm_quit = False

        # Bug 94: Fade-in animation state
        self._fade_alpha = 0  # Current overlay opacity (0-180)
        self._target_alpha = 180  # Target opacity
        self._fade_speed = 15  # Alpha increase per frame

    def _update_button_positions(self) -> None:
        """Update button positions to be centered."""
        center_x = self.screen_width // 2
        start_y = self.screen_height // 2 + 20
        # Bug 114: Use scalable spacing based on screen height (minimum 10px)
        button_spacing = max(10, int(self.screen_height * 0.02))

        for i, button in enumerate(self.buttons):
            button.update_position(
                center_x - button.rect.width // 2,
                start_y + i * (button.rect.height + button_spacing)
            )

    def handle_resize(self, new_width: int, new_height: int) -> None:
        """
        Handle window resize by updating button positions.

        Args:
            new_width: New screen width
            new_height: New screen height
        """
        self.screen_width = new_width
        self.screen_height = new_height
        # Bug 84: Must reposition buttons after window resize
        self._update_button_positions()

    def reset_state(self) -> None:
        """
        Reset menu state. Call when showing/hiding the pause menu
        to ensure quit confirmation dialog doesn't persist.
        """
        self._confirm_quit = False
        self.selected_index = 0
        self._fade_alpha = 0  # Bug 94: Reset fade for new animation
        self._update_button_positions()

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle keyboard/mouse input.

        Args:
            event: Pygame event

        Returns:
            Action string if button activated, None otherwise
        """
        # Handle quit confirmation mode
        if self._confirm_quit:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    self._confirm_quit = False
                    return "quit"
                elif event.key in (pygame.K_n, pygame.K_ESCAPE):
                    self._confirm_quit = False
                    return None
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.buttons)
                return None

            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.buttons)
                return None

            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                action = self.buttons[self.selected_index].action
                if action == "quit":
                    self._confirm_quit = True
                    return None
                return action

            elif event.key == pygame.K_p:
                # P resumes
                return "resume"

            elif event.key == pygame.K_s:
                # S saves model
                return "save"

            elif event.key == pygame.K_h:
                # H goes to game selector (Home)
                return "menu"

            elif event.key == pygame.K_q:
                # Q triggers quit confirmation
                self._confirm_quit = True
                return None

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            for i, button in enumerate(self.buttons):
                if button.contains_point(mouse_pos):
                    self.selected_index = i

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in self.buttons:
                if button.contains_point(mouse_pos):
                    action = button.action
                    if action == "quit":
                        self._confirm_quit = True
                        return None
                    return action

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

        # Bug 94: Animate fade-in instead of instant full opacity
        if self._fade_alpha < self._target_alpha:
            self._fade_alpha = min(self._fade_alpha + self._fade_speed, self._target_alpha)

        # Semi-transparent overlay with animated fade
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, self._fade_alpha))
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

        # Add training time if available (with improved formatting)
        if 'training_time' in context:
            training_time = context['training_time']
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            if hours > 0:
                context_lines.append(f"Training Time: {hours}h {minutes}m")
            elif minutes > 0:
                context_lines.append(f"Training Time: {minutes}m {seconds}s")
            else:
                context_lines.append(f"Training Time: {seconds}s")

        # Add memory buffer if available (with human-readable format)
        if 'memory_size' in context and 'memory_capacity' in context:
            mem_size = context['memory_size']
            mem_cap = context['memory_capacity']
            mem_pct = (mem_size / mem_cap) * 100 if mem_cap > 0 else 0

            # Format numbers in human-readable form (K for thousands)
            def format_num(n: int) -> str:
                if n >= 1000000:
                    return f"{n / 1000000:.1f}M"
                elif n >= 1000:
                    return f"{n / 1000:.1f}K"
                return str(n)

            context_lines.append(f"Memory: {format_num(mem_size)} / {format_num(mem_cap)} ({mem_pct:.0f}%)")

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
        hint = "↑↓ Navigate  •  Enter Select  •  H Home  •  Q Quit"
        hint_surface = self._context_font.render(hint, True, (150, 150, 150))
        hint_rect = hint_surface.get_rect(center=(center_x, self.screen_height - 30))
        surface.blit(hint_surface, hint_rect)

        # Quit confirmation dialog
        if self._confirm_quit:
            # Draw confirmation overlay
            confirm_overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            confirm_overlay.fill((0, 0, 0, 150))
            surface.blit(confirm_overlay, (0, 0))

            # Confirmation box
            box_width = 350
            box_height = 120
            box_x = (self.screen_width - box_width) // 2
            box_y = (self.screen_height - box_height) // 2

            # Box background
            box_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            pygame.draw.rect(box_surface, (40, 40, 50, 240), box_surface.get_rect(), border_radius=10)
            surface.blit(box_surface, (box_x, box_y))
            pygame.draw.rect(surface, (255, 100, 100), (box_x, box_y, box_width, box_height), 2, border_radius=10)

            # Confirmation text
            confirm_text = self._button_font.render("Quit training?", True, (255, 255, 255))
            confirm_rect = confirm_text.get_rect(center=(center_x, box_y + 35))
            surface.blit(confirm_text, confirm_rect)

            # Sub text
            sub_text = self._context_font.render("Unsaved progress will be lost.", True, (200, 150, 150))
            sub_rect = sub_text.get_rect(center=(center_x, box_y + 60))
            surface.blit(sub_text, sub_rect)

            # Bug 101: Y/N options - use consistent warning colors (amber, not green)
            options_text = self._context_font.render("Press Y to confirm, N to cancel", True, (255, 200, 150))
            options_rect = options_text.get_rect(center=(center_x, box_y + 90))
            surface.blit(options_text, options_rect)
