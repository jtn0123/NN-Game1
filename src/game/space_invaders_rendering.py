"""Rendering helpers for Space Invaders."""

from __future__ import annotations

import math
import random
from typing import Any

import pygame


class SpaceInvadersRenderingMixin:
    def render(self: Any, screen: pygame.Surface) -> None:
        if self.headless:
            return

        assert self.ship is not None

        # Apply screen shake offset
        shake_x = (
            int(random.uniform(-self.screen_shake, self.screen_shake))
            if self.screen_shake > 0
            else 0
        )
        shake_y = (
            int(random.uniform(-self.screen_shake, self.screen_shake))
            if self.screen_shake > 0
            else 0
        )

        # Background - deep space black with subtle blue tint
        screen.fill((2, 4, 12))

        # Draw stars
        for star in self.stars:
            star.draw(screen, self._time)

        # Draw the ground base that the player is defending
        self._draw_ground_base(screen)

        # Draw particles
        for particle in self.particles:
            particle.draw(screen)

        # Draw shields/bunkers
        for shield in self.shields:
            shield.draw(screen)

        # Draw aliens with shake offset and pulse effect
        # Pulse intensity increases as fewer aliens remain (like the audio in original)
        # Bug 63 fix: Guard against division by zero if _num_aliens = 0
        aliens_ratio = self._aliens_remaining / self._num_aliens if self._num_aliens > 0 else 1.0
        pulse_intensity = 0.15 * (1 - aliens_ratio)
        pulse_offset = math.sin(self.alien_pulse_phase * math.pi * 2) * pulse_intensity * 3

        for alien in self.aliens:
            if alien.alive:
                original_x = alien.x
                original_y = alien.y
                alien.x += int(self.alien_x_offset) + shake_x
                alien.y += shake_y + int(pulse_offset)
                # Bug 87: Use try/finally to guarantee position restoration
                try:
                    alien.draw(screen, self._time)
                finally:
                    alien.x = original_x
                    alien.y = original_y

        # Draw UFO
        if self.ufo is not None and self.ufo.alive:
            # Bug 91: Apply shake offset to UFO for visual coherence
            original_ufo_x = self.ufo.x
            original_ufo_y = self.ufo.y
            self.ufo.x += shake_x
            self.ufo.y += shake_y
            try:
                self.ufo.draw(screen, self._time)
            finally:
                self.ufo.x = original_ufo_x
                self.ufo.y = original_ufo_y

        # Draw bullets with shake
        for bullet in self.player_bullets:
            bullet.x += shake_x
            bullet.y += shake_y
            bullet.draw(screen)
            bullet.x -= shake_x
            bullet.y -= shake_y
        for bullet in self.alien_bullets:
            bullet.x += shake_x
            bullet.y += shake_y
            bullet.draw(screen)
            bullet.x -= shake_x
            bullet.y -= shake_y

        # Draw ship with shake (flash when invincible)
        original_ship_x = self.ship.x
        original_ship_y = self.ship.y
        self.ship.x += shake_x
        self.ship.y += shake_y

        # Invincibility flashing effect
        if self.player_invincible:
            # Bug 88: Always draw ship but add visual indicator when invincible
            # Use alpha blending for "ghost" effect on off frames
            if int(self._time * 15) % 2 == 0:
                self.ship.draw(screen, self._time)
            else:
                # Draw semi-transparent version instead of hiding completely
                ghost_surface = pygame.Surface(
                    (self.ship.width + 20, self.ship.height + 20), pygame.SRCALPHA
                )
                # Draw glow effect to indicate shield is active
                glow_color = (100, 200, 255, 80)
                pygame.draw.ellipse(ghost_surface, glow_color, ghost_surface.get_rect())
                screen.blit(ghost_surface, (self.ship.x - 10, self.ship.y - 10))
                # Draw ship with reduced opacity effect (tinted)
                self.ship.draw(screen, self._time)
        else:
            self.ship.draw(screen, self._time)

        self.ship.x = original_ship_x
        self.ship.y = original_ship_y

        # Apply CRT scanlines
        if self._scanline_surface:
            screen.blit(self._scanline_surface, (0, 0))

        # Apply vignette
        if self._crt_vignette:
            screen.blit(self._crt_vignette, (0, 0))

        # Draw score popups
        for popup in self.score_popups:
            popup.draw(screen)

        # Flash effect
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, self.flash_alpha))
            screen.blit(flash_surface, (0, 0))

        # Draw HUD (not affected by shake)
        self._draw_hud(screen)

        # Draw wave announcement (on top of everything)
        if self.wave_announcement is not None:
            self.wave_announcement.draw(screen)

    def _draw_ground_base(self: Any, screen: pygame.Surface) -> None:
        """Draw the ground base that the player is defending."""
        # Main ground line
        ground_color = (0, 180, 80)
        pygame.draw.rect(screen, ground_color, (0, self.ground_y, self.width, 3))

        # Glow effect on ground line
        glow_surface = pygame.Surface((self.width, 10), pygame.SRCALPHA)
        for i in range(5):
            alpha = 40 - i * 8
            pygame.draw.rect(glow_surface, (*ground_color, alpha), (0, i, self.width, 1))
        screen.blit(glow_surface, (0, self.ground_y - 5))

        # City/base silhouette at the bottom
        base_color = (0, 60, 30)
        highlight_color = (0, 100, 50)

        # Draw stylized buildings/structures
        buildings = [
            # (x_offset_ratio, width, height)
            (0.05, 30, 15),
            (0.12, 20, 10),
            (0.18, 40, 20),
            (0.28, 25, 12),
            (0.35, 35, 18),
            (0.45, 15, 8),
            (0.52, 45, 22),
            (0.62, 20, 14),
            (0.70, 30, 16),
            (0.78, 25, 10),
            (0.85, 35, 18),
            (0.92, 20, 12),
        ]

        for x_ratio, bwidth, bheight in buildings:
            bx = int(self.width * x_ratio)
            by = self.ground_y + 3

            # Building body
            pygame.draw.rect(screen, base_color, (bx, by, bwidth, bheight))

            # Building top highlight
            pygame.draw.rect(screen, highlight_color, (bx, by, bwidth, 2))

            # Window lights (flickering)
            if bheight > 10:
                for wy in range(by + 4, by + bheight - 2, 4):
                    for wx in range(bx + 3, bx + bwidth - 3, 6):
                        if random.random() > 0.3:  # 70% of windows lit
                            flicker = 0.7 + 0.3 * math.sin(self._time * 5 + wx * 0.1)
                            window_color = (
                                int(255 * flicker),
                                int(200 * flicker),
                                int(50 * flicker),
                            )
                            pygame.draw.rect(screen, window_color, (wx, wy, 2, 2))

    def _draw_hud(self: Any, screen: pygame.Surface) -> None:
        """Draw heads-up display with retro style."""
        # Use a pixelated font effect
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 28)

        # Score with glow (left side)
        score_text = f"SCORE: {self.score}"
        # Glow
        glow_surf = font.render(score_text, True, (0, 150, 50))
        screen.blit(glow_surf, (12, 12))
        # Main text
        text_surf = font.render(score_text, True, (0, 255, 100))
        screen.blit(text_surf, (10, 10))

        # Lives with ship icons (right side)
        lives_x = self.width - 140
        lives_text = "LIVES:"
        glow_surf = font.render(lives_text, True, (0, 150, 50))
        screen.blit(glow_surf, (lives_x + 2, 12))
        text_surf = font.render(lives_text, True, (0, 255, 100))
        screen.blit(text_surf, (lives_x, 10))

        # Draw ship icons for lives
        for i in range(self.lives):
            ship_x = lives_x + 80 + i * 20
            ship_y = 18
            # Mini ship shape
            points = [
                (ship_x, ship_y - 6),
                (ship_x - 8, ship_y + 6),
                (ship_x + 8, ship_y + 6),
            ]
            pygame.draw.polygon(screen, (0, 255, 100), points)

        # Level indicator (center, prominent)
        level_text = f"WAVE {self.level}"
        # Glow effect
        glow_surf = font.render(level_text, True, (150, 100, 0))
        glow_rect = glow_surf.get_rect(centerx=self.width // 2 + 1, top=11)
        screen.blit(glow_surf, glow_rect)
        # Main text in gold/yellow
        level_surf = font.render(level_text, True, (255, 200, 50))
        level_rect = level_surf.get_rect(centerx=self.width // 2, top=10)
        screen.blit(level_surf, level_rect)

        # Aliens killed counter (below score)
        kills_text = f"KILLS: {self.total_aliens_killed}"
        kills_surf = small_font.render(kills_text, True, (150, 150, 150))
        screen.blit(kills_surf, (10, 38))

        # Game over message
        if self.game_over:
            big_font = pygame.font.Font(None, 72)
            msg = "GAME OVER"
            color = (255, 50, 50)

            # Shadow
            shadow = big_font.render(msg, True, (0, 0, 0))
            shadow_rect = shadow.get_rect(center=(self.width // 2 + 3, self.height // 2 + 3))
            screen.blit(shadow, shadow_rect)

            # Main text with glow
            glow = big_font.render(msg, True, tuple(c // 2 for c in color))
            glow_rect = glow.get_rect(center=(self.width // 2 + 1, self.height // 2 + 1))
            screen.blit(glow, glow_rect)

            text = big_font.render(msg, True, color)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            screen.blit(text, text_rect)

            # Final stats
            score_font = pygame.font.Font(None, 36)
            stats_y = self.height // 2 + 50

            final_score = f"Final Score: {self.score}"
            score_surf = score_font.render(final_score, True, (200, 200, 200))
            score_rect = score_surf.get_rect(center=(self.width // 2, stats_y))
            screen.blit(score_surf, score_rect)

            wave_reached = f"Waves Completed: {self.level - 1}"
            wave_surf = small_font.render(wave_reached, True, (150, 150, 150))
            wave_rect = wave_surf.get_rect(center=(self.width // 2, stats_y + 30))
            screen.blit(wave_surf, wave_rect)

            kills_final = f"Total Kills: {self.total_aliens_killed}"
            kills_surf = small_font.render(kills_final, True, (150, 150, 150))
            kills_rect = kills_surf.get_rect(center=(self.width // 2, stats_y + 55))
            screen.blit(kills_surf, kills_rect)
