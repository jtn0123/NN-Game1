"""Rendering, input, and notification helpers for the interactive runtime."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pygame


class InteractiveRenderingMixin:
    def _update_layout(self: Any, new_width: int, new_height: int) -> None:
        """Update component positions based on new window size."""
        # Enforce minimum window size
        new_width = max(new_width, self.min_window_width)
        new_height = max(new_height, self.min_window_height)

        self.window_width = new_width
        self.window_height = new_height

        # The game has a FIXED render size (config.SCREEN_WIDTH x config.SCREEN_HEIGHT)
        # We position other elements around it
        # Game display area stays fixed (training stats and NN viz are on web dashboard)
        self.game_width = self.config.SCREEN_WIDTH  # 800
        self.game_height = self.config.SCREEN_HEIGHT  # 600

        # Update scaling for the new window size
        self._update_scale()

        # Update pause menu positions
        if hasattr(self, "pause_menu") and self.pause_menu:
            self.pause_menu.handle_resize(new_width, new_height)

    def _update_scale(self: Any) -> None:
        """Calculate scaling factor to fit game in window while maintaining aspect ratio."""
        # Guard against zero dimensions during window minimize/restore
        if self.window_width <= 0 or self.window_height <= 0:
            return

        game_aspect = self.config.SCREEN_WIDTH / self.config.SCREEN_HEIGHT
        window_aspect = self.window_width / self.window_height

        if window_aspect > game_aspect:
            # Window is wider than game - scale by height
            self.scale_factor = self.window_height / self.config.SCREEN_HEIGHT
            scaled_width = int(self.config.SCREEN_WIDTH * self.scale_factor)
            self.game_offset_x = (self.window_width - scaled_width) // 2
            self.game_offset_y = 0
        else:
            # Window is taller than game - scale by width
            self.scale_factor = self.window_width / self.config.SCREEN_WIDTH
            scaled_height = int(self.config.SCREEN_HEIGHT * self.scale_factor)
            self.game_offset_x = 0
            self.game_offset_y = (self.window_height - scaled_height) // 2

        # Ensure minimum scale factor to prevent division by zero or rendering issues
        self.scale_factor = max(0.1, self.scale_factor)

    def _show_notification(
        self: Any, text: str, color: tuple = (100, 200, 255), duration: float = 2.0
    ) -> None:
        """
        Show a notification on screen.

        Args:
            text: Notification text
            color: Text color (RGB tuple)
            duration: How long to show the notification in seconds
        """
        import time

        self._notifications.append(
            {
                "text": text,
                "color": color,
                "start_time": time.time(),
                "duration": duration,
            }
        )

    def _update_notifications(self: Any) -> None:
        """Remove expired notifications."""
        import time

        current_time = time.time()
        self._notifications = [
            n for n in self._notifications if current_time - n["start_time"] < n["duration"]
        ]

    def _render_notifications(self: Any, surface: pygame.Surface) -> None:
        """Render all active notifications."""
        import time

        if not self._notifications:
            return

        current_time = time.time()
        y_offset = 10

        for notification in self._notifications:
            elapsed = current_time - notification["start_time"]
            # Fade out in the last 0.5 seconds
            alpha = 255
            if elapsed > notification["duration"] - 0.5:
                alpha = int(255 * (notification["duration"] - elapsed) / 0.5)
            alpha = max(0, min(255, alpha))

            text_surface = self._notification_font.render(
                notification["text"], True, notification["color"]
            )

            # Create background with alpha
            bg_width = text_surface.get_width() + 20
            bg_height = text_surface.get_height() + 10
            bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
            pygame.draw.rect(
                bg_surface,
                (0, 0, 0, int(alpha * 0.7)),
                bg_surface.get_rect(),
                border_radius=5,
            )

            # Position at top-center
            x = (surface.get_width() - bg_width) // 2
            y = y_offset

            # Apply alpha to text
            text_surface.set_alpha(alpha)

            surface.blit(bg_surface, (x, y))
            surface.blit(text_surface, (x + 10, y + 5))

            y_offset += bg_height + 5

    def _set_speed(self: Any, speed: float, force_log: bool = False) -> None:
        """Set game speed (for web dashboard control)."""
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            print(f"⚠️  Invalid speed value: {speed}")
            return
        if not math.isfinite(speed):
            print(f"⚠️  Invalid speed value: {speed}")
            return
        new_speed = max(1.0, min(1000.0, speed))
        self.game_speed = new_speed

        if self.web_dashboard:
            self.web_dashboard.publisher.set_speed(self.game_speed)

        # Only log if speed changed significantly (avoid spam when dragging slider)
        # Log when: forced, or speed is a preset value, or changed by >10%
        speed_changed_significantly = (
            abs(new_speed - self._last_logged_speed) / max(1, self._last_logged_speed) > 0.1
        )
        is_preset = int(new_speed) in self.SPEED_PRESETS

        if force_log or (speed_changed_significantly and is_preset):
            if self.web_dashboard:
                self.web_dashboard.log(f"⏩ Speed set to {int(self.game_speed)}x", "action")
            print(f"⏩ Speed: {int(self.game_speed)}x")
            self._last_logged_speed = new_speed

    def _speed_up(self: Any) -> None:
        """Increase speed to next preset."""
        for preset in self.SPEED_PRESETS:
            if preset > self.game_speed + 0.01:  # Epsilon comparison for float precision
                self._set_speed(preset, force_log=True)
                return
        # Already at max
        self._set_speed(self.SPEED_PRESETS[-1], force_log=True)

    def _speed_down(self: Any) -> None:
        """Decrease speed to previous preset."""
        for preset in reversed(self.SPEED_PRESETS):
            if preset < self.game_speed - 0.01:  # Epsilon comparison for float precision
                self._set_speed(preset, force_log=True)
                return
        # Already at min
        self._set_speed(self.SPEED_PRESETS[0], force_log=True)

    def _handle_events(self: Any) -> None:
        """Handle pygame events and keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                new_width = max(event.w, self.min_window_width)
                new_height = max(event.h, self.min_window_height)
                self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                self._update_layout(new_width, new_height)

            # Handle pause menu interactions when paused
            if self.paused:
                action = self.pause_menu.handle_event(event)
                if action == "resume":
                    self._toggle_pause()
                elif action == "save":
                    self._save_model(
                        f"{self.config.GAME_NAME}_manual_save.pth", save_reason="manual"
                    )
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"💾 Manual save: {self.config.GAME_NAME}_manual_save.pth",
                            "success",
                        )
                elif action == "menu":
                    # Save current progress before returning to menu
                    self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="menu_exit")
                    if self.web_dashboard:
                        self.web_dashboard.log("🏠 Returning to game selector...", "warning")
                        self.web_dashboard.launcher_mode = True
                        self.web_dashboard.socketio.emit(
                            "redirect_to_launcher",
                            {"message": "Returning to game selector..."},
                        )
                    self.return_to_menu = True
                    self.running = False
                elif action == "quit":
                    self.running = False
                continue  # Skip other event handling when paused

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False

                elif event.key == pygame.K_p:
                    self._toggle_pause()

                elif event.key == pygame.K_s:
                    success = self._save_model(
                        f"{self.config.GAME_NAME}_manual_save.pth", save_reason="manual"
                    )
                    if success:
                        self._show_notification("Model Saved", (100, 255, 100), 1.5)
                    else:
                        self._show_notification("Save Failed", (255, 100, 100), 2.0)
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"💾 Manual save: {self.config.GAME_NAME}_manual_save.pth",
                            "success",
                        )

                elif event.key == pygame.K_r:
                    self._reset_episode()
                    self._show_notification("Episode Reset", (255, 200, 100), 1.0)

                elif event.key == pygame.K_o and hasattr(self.game, "show_agent_overlay"):
                    self.game.show_agent_overlay = not self.game.show_agent_overlay
                    on = self.game.show_agent_overlay
                    self._show_notification(
                        "Agent View: ON" if on else "Agent View: OFF",
                        (120, 220, 255),
                        1.0,
                    )

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._speed_up()
                    self._show_notification(f"Speed: {self.game_speed:.0f}x", (100, 200, 255), 1.0)

                elif event.key == pygame.K_MINUS:
                    self._speed_down()
                    self._show_notification(f"Speed: {self.game_speed:.0f}x", (100, 200, 255), 1.0)

                elif event.key == pygame.K_f:
                    # Toggle fullscreen
                    if self.screen.get_flags() & pygame.FULLSCREEN:
                        self.screen = pygame.display.set_mode(
                            (self.window_width, self.window_height), pygame.RESIZABLE
                        )
                        # Bug 95: Show notification for fullscreen toggle
                        self._show_notification("Windowed", (100, 200, 255), 1.0)
                    else:
                        self.screen = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN | pygame.RESIZABLE
                        )
                        # Update layout to new screen size
                        display_info = pygame.display.Info()
                        self._update_layout(display_info.current_w, display_info.current_h)
                        # Bug 95: Show notification for fullscreen toggle
                        self._show_notification("Fullscreen", (100, 200, 255), 1.0)

                elif event.key == pygame.K_h:
                    # Toggle help legend
                    self.show_help_legend = not self.show_help_legend
                    # Bug 96: Show notification for help legend toggle
                    msg = "Help: ON" if self.show_help_legend else "Help: OFF"
                    self._show_notification(msg, (150, 150, 200), 1.0)

    def _render_frame(self: Any, state: np.ndarray, action: int, info: dict) -> None:
        """Render one frame of the visualization with proper scaling."""
        # Clear the game surface (fixed size)
        self.game_surface.fill((10, 10, 15))

        # Render game to the fixed-size game surface
        self.game.render(self.game_surface)

        # Render HUD (training stats overlay) if not paused
        if not self.paused:
            # Get action labels from game
            action_labels = []
            if hasattr(self.game, "get_action_labels"):
                action_labels = self.game.get_action_labels()
            else:
                # Fallback generic labels
                action_labels = [f"Action {i}" for i in range(self.game.action_size)]

            self.hud.render(
                surface=self.game_surface,
                episode=self.episode,
                score=info.get("score", 0),
                best_score=self.best_score_ever,
                epsilon=self.agent.epsilon if hasattr(self, "agent") else 0.0,
                loss=0.0,  # Could track recent loss
                speed=self.game_speed,
                max_episodes=self.config.MAX_EPISODES,
                selected_action=action,
                action_labels=action_labels,
            )

        # Render pause menu with context (centered on the game surface)
        if self.paused:
            # Build context for pause menu
            pause_context = {
                "episode": self.episode,
                "score": info.get("score", 0),
                "best_score": self.best_score_ever,
                "epsilon": self.agent.epsilon if hasattr(self, "agent") else 0.0,
                "training_time": time.time() - self.training_start_time,
                "memory_size": len(self.agent.memory) if hasattr(self, "agent") else 0,
                "memory_capacity": (
                    self.config.MEMORY_SIZE if hasattr(self.config, "MEMORY_SIZE") else 0
                ),
            }
            self.pause_menu.render(self.game_surface, pause_context)

        # Render help legend (bottom left of game area, toggle with H)
        if self.show_help_legend:
            controls = [
                ("P", "Pause/Resume"),
                ("S", "Save Model"),
                ("R", "Reset Episode"),
                ("+/-", "Speed Up/Down"),
                ("F", "Fullscreen"),
                ("H", "Hide Help"),
                ("ESC", "Quit"),
            ]
            padding = 15
            line_height = 24
            legend_width = 180
            legend_height = len(controls) * line_height + padding * 2 + 30

            # Position at bottom-left
            legend_x = 10
            legend_y = self.config.SCREEN_HEIGHT - legend_height - 10

            # Draw background
            legend_bg = pygame.Surface((legend_width, legend_height), pygame.SRCALPHA)
            pygame.draw.rect(legend_bg, (0, 0, 0, 200), legend_bg.get_rect(), border_radius=8)
            self.game_surface.blit(legend_bg, (legend_x, legend_y))
            pygame.draw.rect(
                self.game_surface,
                (100, 100, 100),
                (legend_x, legend_y, legend_width, legend_height),
                1,
                border_radius=8,
            )

            # Draw title
            title = self._help_title_font.render("Controls", True, (255, 255, 255))
            self.game_surface.blit(title, (legend_x + padding, legend_y + padding))

            # Draw controls
            y = legend_y + padding + 30
            for key, desc in controls:
                key_surface = self._help_font.render(key, True, (100, 200, 255))
                desc_surface = self._help_font.render(f" - {desc}", True, (180, 180, 180))
                self.game_surface.blit(key_surface, (legend_x + padding, y))
                self.game_surface.blit(
                    desc_surface, (legend_x + padding + key_surface.get_width(), y)
                )
                y += line_height
        else:
            # Show hint to display help
            hint_text = self._speed_font.render("Press H for controls", True, (80, 80, 80))
            self.game_surface.blit(hint_text, (10, self.config.SCREEN_HEIGHT - 25))

        # Capture screenshot for web dashboard (before scaling, every 10 frames)
        self.frame_count = (self.frame_count + 1) % 10000  # Keep bounded to avoid overflow
        if self.web_dashboard and self.frame_count % 10 == 0:
            self.web_dashboard.capture_screenshot(self.game_surface)

        # Emit NN visualization data to web dashboard (throttled by server)
        if self.web_dashboard:
            self._emit_nn_visualization(state, action)

        # Clear main screen and scale game surface to fit window
        self.screen.fill((0, 0, 0))

        # Scale the game surface (optimize: skip scaling at 1:1 ratio)
        if abs(self.scale_factor - 1.0) < 0.001:  # Effectively 1.0
            # No scaling needed - blit directly
            self.screen.blit(self.game_surface, (self.game_offset_x, self.game_offset_y))
        else:
            # Calculate scaled size
            scaled_width = int(self.config.SCREEN_WIDTH * self.scale_factor)
            scaled_height = int(self.config.SCREEN_HEIGHT * self.scale_factor)
            # Use smoothscale for better quality
            scaled_surface = pygame.transform.smoothscale(
                self.game_surface, (scaled_width, scaled_height)
            )
            # Blit centered on screen
            self.screen.blit(scaled_surface, (self.game_offset_x, self.game_offset_y))

        # Update and render notifications (on top of scaled game)
        self._update_notifications()
        self._render_notifications(self.screen)

        pygame.display.flip()
