"""Core rendering mixin for Crystal Caves (terrain, hazards, HUD)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pygame

from .crystal_caves_art import EGA, CrystalCavesArt


class CrystalCavesRenderingMixin:
    def render(self: Any, screen) -> None:
        """Render the cave, HUD, and DOS-era pixel art overlays."""
        camera_x, camera_y = self._camera()
        screen.fill((0, 0, 0))
        self._draw_background(screen, camera_x, camera_y)
        self._draw_tiles(screen, camera_x, camera_y)
        self._draw_ladders(screen, camera_x, camera_y)
        self._draw_elevators(screen, camera_x, camera_y)
        self._draw_switch_wires(screen, camera_x, camera_y)
        self._draw_level_dressing(screen, camera_x, camera_y)
        self._draw_pickups(screen, camera_x, camera_y)
        self._draw_enemies(screen, camera_x, camera_y)
        self._draw_bullets(screen, camera_x, camera_y)
        self._draw_player(screen, camera_x, camera_y)
        self._draw_visual_events(screen, camera_x, camera_y)
        if getattr(self, "show_agent_overlay", False):
            self._draw_agent_overlay(screen, camera_x, camera_y)
        self._draw_gravity_overlay(screen)
        self._draw_hud(screen)

        if self.game_over and self._font:
            title = "CAVE CLEARED" if self.won else "MYLO DOWN"
            color = (88, 255, 88) if self.won else (255, 80, 80)
            text = self._font.render(title, True, color)
            rect = text.get_rect(center=(self.width // 2, self.height // 2 - 18))
            screen.blit(text, rect)
            if self._small_font:
                detail = self._small_font.render(
                    f"Score {self.score} | Crystals left {len(self.crystals)}",
                    True,
                    (230, 230, 230),
                )
                detail_rect = detail.get_rect(center=(self.width // 2, self.height // 2 + 18))
                screen.blit(detail, detail_rect)

    def render_title_screen(self: Any, screen) -> None:
        """Render a Crystal Caves-style title and instruction screen."""
        art = self._art or CrystalCavesArt()
        width, height = screen.get_size()
        palette = self._episode_palette()
        screen.fill((0, 0, 0))

        for y in range(0, height, self.TILE_SIZE):
            for x in range(0, width, self.TILE_SIZE):
                edge = x < 64 or x > width - 96 or y < 48 or y > height - 92
                if not edge and (x // 32 + y // 32) % 5:
                    continue
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(screen, palette["rock_dark"], rect)
                pygame.draw.rect(screen, palette["rock"], rect.inflate(-3, -3))
                pygame.draw.line(
                    screen,
                    palette["rock_mid"],
                    (rect.x + 5, rect.y + 20),
                    (rect.x + 24, rect.y + 11),
                    2,
                )
                if (x + y) % 96 == 0:
                    pygame.draw.rect(screen, palette["rock_light"], (rect.x + 10, rect.y + 7, 4, 4))

        title = "CRYSTAL CAVES"
        title_surface = art.text(title, EGA["C"], scale=5)
        title_x = (width - title_surface.get_width()) // 2
        art.draw_text(screen, title, title_x, 78, EGA["C"], scale=5)
        art.draw_text(
            screen,
            "TROUBLE WITH TWINKLES",
            (width - art.text("TROUBLE WITH TWINKLES", EGA["Y"], scale=2).get_width()) // 2,
            136,
            EGA["Y"],
            scale=2,
        )

        panel = pygame.Rect(104, 184, width - 208, 230)
        pygame.draw.rect(screen, EGA["K"], panel.inflate(8, 8))
        pygame.draw.rect(screen, (6, 6, 24), panel)
        pygame.draw.rect(screen, EGA["w"], panel, 2)
        pygame.draw.rect(screen, EGA["G"], (panel.x, panel.y, panel.w, 5))
        pygame.draw.rect(screen, EGA["G"], (panel.x, panel.bottom - 5, panel.w, 5))

        art.draw_sprite(screen, "mylo_shoot", panel.x + 28, panel.y + 36, scale=3)
        art.draw_sprite(screen, "bat_enemy", panel.x + 32, panel.y + 142, scale=2)
        art.draw_sprite(screen, "crystal_blue", panel.x + 136, panel.y + 38, scale=3)
        art.draw_sprite(screen, "door_locked", panel.x + 146, panel.y + 130, scale=2)

        instructions = (
            ("COLLECT EVERY CRYSTAL", EGA["C"]),
            ("FIND SWITCHES AND ELEVATORS", EGA["G"]),
            ("Z FIRES  E USES  SPACE JUMPS", EGA["Y"]),
            ("ARROWS OR A D MOVE MYLO", EGA["W"]),
        )
        art.draw_text(screen, "MISSION", panel.x + 256, panel.y + 22, EGA["Y"], scale=2)
        for index, (line, color) in enumerate(instructions):
            art.draw_text(
                screen,
                line,
                panel.x + 256,
                panel.y + 58 + index * 25,
                color,
                scale=1,
            )

        score_panel = pygame.Rect(panel.right - 150, panel.y + 26, 118, 166)
        pygame.draw.rect(screen, EGA["K"], score_panel.inflate(4, 4))
        pygame.draw.rect(screen, (18, 28, 54), score_panel)
        pygame.draw.rect(screen, EGA["w"], score_panel, 1)
        art.draw_text(
            screen,
            "HIGH SCORES",
            score_panel.x + 8,
            score_panel.y + 10,
            EGA["Y"],
            scale=1,
        )
        for index, (name, score) in enumerate(
            (("MYLO", "12500"), ("NOVA", "09000"), ("BOT", "05000"))
        ):
            y = score_panel.y + 38 + index * 31
            art.draw_text(screen, name, score_panel.x + 10, y, EGA["G"], scale=1)
            art.draw_text(screen, score, score_panel.x + 64, y, EGA["C"], scale=1)

        menu_panel = pygame.Rect(panel.x + 248, panel.bottom - 48, 190, 27)
        pygame.draw.rect(screen, EGA["K"], menu_panel.inflate(3, 3))
        pygame.draw.rect(screen, (16, 18, 42), menu_panel)
        pygame.draw.rect(screen, EGA["G"], menu_panel, 1)
        art.draw_text(
            screen,
            "1 START  2 EPISODE  3 HELP",
            menu_panel.x + 7,
            menu_panel.y + 8,
            EGA["G"],
            scale=1,
        )

        episodes = ("EP1 TWINKLES", "EP2 SLUGS", "EP3 SUPERNOVA")
        for index, episode in enumerate(episodes):
            box = pygame.Rect(132 + index * 178, 436, 158, 54)
            selected = index == self.level_index % len(self.CAVES)
            pygame.draw.rect(screen, EGA["K"], box.inflate(4, 4))
            pygame.draw.rect(screen, (16, 18, 42), box)
            pygame.draw.rect(screen, EGA["Y"] if selected else EGA["w"], box, 2)
            art.draw_text(
                screen,
                episode,
                box.x + 11,
                box.y + 18,
                EGA["Y"] if selected else EGA["w"],
                scale=1,
            )

        art.draw_text(
            screen,
            "PRESS ANY KEY",
            (width - art.text("PRESS ANY KEY", EGA["G"], scale=2).get_width()) // 2,
            height - 62,
            EGA["G"],
            scale=2,
        )

    def _camera(self: Any) -> Tuple[int, int]:
        visible_height = self.height - self.HUD_HEIGHT
        target_x = self.player_x + self.PLAYER_WIDTH / 2 - self.width / 2
        target_y = self.player_y + self.PLAYER_HEIGHT / 2 - visible_height / 2
        camera_x = int(np.clip(target_x, 0, max(0, self.level_width - self.width)))
        camera_y = int(np.clip(target_y, 0, max(0, self.level_height - visible_height)))
        return camera_x, camera_y

    def _world_to_screen(
        self: Any, x: float, y: float, camera_x: int, camera_y: int
    ) -> Tuple[int, int]:
        return int(x - camera_x), int(y - camera_y)

    def _draw_space(self: Any, screen, camera_x: int, camera_y: int, surface_y: int) -> None:
        """Outer-space backdrop above the planet surface: a black starfield with
        Earth + the Moon and a pink horizon glow fading into the ground — the
        "you start above everything" entrance (CaveSpec.sky_rows)."""
        prev = screen.get_clip()
        screen.set_clip(pygame.Rect(0, 0, self.width, surface_y))
        pygame.draw.rect(screen, (5, 4, 16), (0, 0, self.width, surface_y))

        star_x = camera_x // 4
        star_y = camera_y // 4
        for i in range(70):
            x = (i * 137 - star_x) % self.width
            y = (i * 79 - star_y) % max(1, surface_y)
            shade = 245 if i % 5 == 0 else (180 if i % 2 else 120)
            size = 2 if i % 9 == 0 else 1
            pygame.draw.rect(screen, (shade, shade, shade), (x, y, size, size))

        # Earth + Moon, anchored near the top-left of the world (slow parallax).
        ex = 158 - camera_x // 3
        ey = 64 - camera_y // 3
        pygame.draw.circle(screen, (36, 88, 176), (ex, ey), 22)
        pygame.draw.circle(screen, (58, 158, 92), (ex - 7, ey - 3), 8)
        pygame.draw.circle(screen, (58, 158, 92), (ex + 8, ey + 7), 6)
        pygame.draw.circle(screen, (120, 184, 255), (ex - 9, ey - 9), 4)
        mx, my = ex + 58, ey + 8
        pygame.draw.circle(screen, (198, 198, 208), (mx, my), 12)
        pygame.draw.circle(screen, (150, 150, 166), (mx + 3, my - 2), 3)
        pygame.draw.circle(screen, (150, 150, 166), (mx - 4, my + 4), 2)

        # Pink atmospheric glow fading down into the surface line.
        band = 28
        glow = pygame.Surface((self.width, band), pygame.SRCALPHA)
        for i in range(band):
            alpha = int(170 * (i / band))
            pygame.draw.line(glow, (212, 72, 140, alpha), (0, i), (self.width, i))
        screen.blit(glow, (0, surface_y - band))
        screen.set_clip(prev)

    def _draw_background(self: Any, screen, camera_x: int, camera_y: int) -> None:
        play_bottom = self.height - self.HUD_HEIGHT
        palette = self._episode_palette()
        sky_rows = getattr(self, "sky_rows", 0)
        surface_y = sky_rows * self.TILE_SIZE - camera_y if sky_rows else 0
        cave_top = max(0, min(surface_y, play_bottom)) if sky_rows else 0

        if sky_rows and surface_y > 0:
            self._draw_space(screen, camera_x, camera_y, min(surface_y, play_bottom))

        # Cave back-wall fill, clipped to below the surface when a sky is present.
        prev = screen.get_clip()
        if cave_top > 0:
            screen.set_clip(pygame.Rect(0, cave_top, self.width, play_bottom - cave_top))
        # Dense theme-colored cave back-wall fill replaces the old black void so
        # terrain reads as carved out of a cave room (backlog V001/V019).
        self._draw_wall_fill(screen, camera_x, camera_y, play_bottom, palette)
        self._draw_cave_depth(screen, camera_x, camera_y, play_bottom, palette)

        # Later episodes keep sparse background machinery. Episode 1 now relies
        # on authored pipes and props so the first screen does not become a grid.
        if self.level_index % len(self.CAVES) != 0:
            for x in range(76 - (camera_x // 5) % 260, self.width + 96, 260):
                pygame.draw.rect(screen, palette["pipe_shadow"], (x + 4, 0, 7, play_bottom))
                pygame.draw.rect(screen, palette["pipe_dark"], (x, 0, 7, play_bottom))
                pygame.draw.line(
                    screen,
                    palette["pipe_light"],
                    (x + 2, 0),
                    (x + 2, play_bottom),
                    1,
                )
                for y in range(38 - (camera_y // 6) % 128, play_bottom + 36, 128):
                    pygame.draw.circle(screen, palette["pipe_light"], (x + 3, y), 4)

            for y in range(42 - (camera_y // 5) % 172, play_bottom, 172):
                for x in range(24 - (camera_x // 6) % 248, self.width + 80, 248):
                    segment = pygame.Rect(x, y, 86, 5)
                    pygame.draw.rect(screen, palette["pipe_shadow"], segment.move(4, 4))
                    pygame.draw.rect(screen, palette["pipe_dark"], segment)
                    pygame.draw.line(
                        screen,
                        palette["pipe_light"],
                        (segment.left, segment.y + 1),
                        (segment.right, segment.y + 1),
                        1,
                    )
                    pygame.draw.circle(screen, palette["pipe_light"], (segment.left + 8, y + 2), 3)
                    pygame.draw.circle(screen, palette["pipe_light"], (segment.right - 8, y + 2), 3)

        if cave_top > 0:
            screen.set_clip(prev)

    def _draw_wall_fill(
        self: Any,
        screen,
        camera_x: int,
        camera_y: int,
        play_bottom: int,
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        """Paint a dense, dim, theme-colored masonry back-wall behind gameplay.

        This is the single biggest visual-identity change: the play area is no
        longer a black void with floating platforms, but a carved cave room.
        Tones sit well below the foreground palette so solid tiles, bright lips
        and bolts still pop in front of the fill (backlog V001/V004/V019/V082).
        """

        def dim(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
            return (
                int(color[0] * factor),
                int(color[1] * factor),
                int(color[2] * factor),
            )

        # A FLAT, recessed back-wall: a dim fill behind everything, textured only
        # with a faint diagonal diamond lattice. It is deliberately flat (no 3D
        # bevels) so the open space reads clearly as the wall *behind* the chunky
        # foreground terrain — nothing here mimics a solid boundary tile.
        base = palette["wall_fill"]
        lattice = dim(palette["wall_accent"], 0.55)

        pygame.draw.rect(screen, base, (0, 0, self.width, play_bottom))

        # Parallax: the wall drifts at half camera speed for a sense of depth.
        px = camera_x // 2
        py = camera_y // 2
        step = 18
        # Slope +1 and slope -1 diagonals tile the screen into faint diamonds.
        offset = (px + py) % step
        for k in range(-play_bottom - step, self.width + step, step):
            x0 = k - offset
            pygame.draw.line(
                screen, lattice, (x0, 0), (x0 + play_bottom, play_bottom), 1
            )
        offset2 = (px - py) % step
        for k in range(0, self.width + play_bottom + step, step):
            x0 = k - offset2
            pygame.draw.line(
                screen, lattice, (x0, 0), (x0 - play_bottom, play_bottom), 1
            )

    def _draw_cave_depth(
        self: Any,
        screen,
        camera_x: int,
        camera_y: int,
        play_bottom: int,
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        """Paint distant cave cuts and machinery shadows behind gameplay tiles."""
        depth = pygame.Surface((self.width, play_bottom), pygame.SRCALPHA)
        style = self.level_index % len(self.CAVES)
        # Boosted alphas (CCV-01) so carved recesses read against the dim wall
        # fill instead of being black-on-black texture.
        alpha = 86 if style == 0 else 68
        dark = (*palette["edge_dark"], alpha)
        mid = (*palette["wall_accent"], 60)
        accent = (*palette["pipe_light"], 74 if style else 58)
        rim = (*palette["rock_light"], 70)

        for i in range(6):
            base_x = (i * 211 - camera_x // 7) % (self.width + 180) - 90
            base_y = 40 + ((i * 83 - camera_y // 8) % max(1, play_bottom - 120))
            points = [
                (base_x, base_y + 72),
                (base_x + 34, base_y + 18),
                (base_x + 91, base_y),
                (base_x + 138, base_y + 37),
                (base_x + 154, base_y + 96),
                (base_x + 54, base_y + 116),
            ]
            pygame.draw.polygon(depth, dark, points)
            pygame.draw.lines(depth, mid, False, points, 2)
            # A brighter rim on every third recess gives a clear carved silhouette.
            if i % 3 == 0:
                pygame.draw.lines(depth, rim, True, points, 2)
            if i % 2 == 0:
                pygame.draw.rect(depth, accent, (base_x + 42, base_y + 45, 54, 5))
                pygame.draw.rect(depth, accent, (base_x + 48, base_y + 62, 5, 36))

        screen.blit(depth, (0, 0))

    def _episode_palette(self: Any) -> Dict[str, Tuple[int, int, int]]:
        palettes = (
            {
                "rock_dark": (0, 0, 72),
                "rock": (18, 34, 150),
                "rock_mid": (38, 72, 204),
                "rock_light": (106, 150, 255),
                "platform_dark": (0, 50, 64),
                "platform": (28, 84, 118),
                "platform_light": (92, 232, 162),
                "pipe_dark": (0, 76, 34),
                "pipe_light": (34, 238, 80),
                "pipe_shadow": (0, 28, 10),
                "spark": (94, 120, 255),
                "wall_fill": (8, 28, 80),
                "wall_accent": (28, 72, 150),
                "ledge_lip": (106, 150, 255),
                "edge_dark": (0, 0, 40),
                "grass": (96, 220, 132),
            },
            {
                "rock_dark": (70, 24, 0),
                "rock": (150, 54, 18),
                "rock_mid": (204, 88, 28),
                "rock_light": (255, 158, 52),
                "platform_dark": (0, 70, 20),
                "platform": (22, 164, 48),
                "platform_light": (180, 255, 72),
                "pipe_dark": (86, 54, 0),
                "pipe_light": (255, 190, 52),
                "pipe_shadow": (40, 20, 0),
                "spark": (255, 190, 72),
                "wall_fill": (64, 32, 16),
                "wall_accent": (140, 70, 28),
                "ledge_lip": (255, 96, 72),
                "edge_dark": (40, 16, 0),
                "grass": (150, 230, 64),
            },
            {
                "rock_dark": (40, 40, 48),
                "rock": (92, 96, 108),
                "rock_mid": (138, 144, 162),
                "rock_light": (218, 226, 255),
                "platform_dark": (36, 50, 74),
                "platform": (92, 108, 170),
                "platform_light": (210, 224, 255),
                "pipe_dark": (54, 38, 80),
                "pipe_light": (194, 128, 255),
                "pipe_shadow": (22, 18, 36),
                "spark": (188, 220, 255),
                "wall_fill": (40, 48, 64),
                "wall_accent": (90, 110, 140),
                "ledge_lip": (120, 240, 255),
                "edge_dark": (20, 24, 34),
            },
        )
        return palettes[self.level_index % len(palettes)]

    def _draw_ledge_growth(
        self: Any,
        screen,
        rect: pygame.Rect,
        col: int,
        row: int,
        palette: Dict[str, Tuple[int, int, int]],
        under_open: bool,
    ) -> None:
        """Bright moss fringe + tufts on a walkable ledge, with hanging vines
        under exposed edges (CCV-17). Only natural-cave themes define ``grass``;
        industrial/tech episodes skip it for a clean metal look.
        """
        grass = palette.get("grass")
        if grass is None:
            return
        grass_dark = (grass[0] * 6 // 10, grass[1] * 6 // 10, grass[2] * 6 // 10)
        seed = col * 31 + row * 17 + self.level_index * 5

        # Mossy fringe sitting on the very top lip.
        pygame.draw.rect(screen, grass_dark, (rect.x, rect.y, rect.w, 4))
        pygame.draw.rect(screen, grass, (rect.x, rect.y, rect.w, 2))

        # Blades poking up above the ledge, with a gentle idle sway.
        for i, x in enumerate(range(rect.left + 2, rect.right - 1, 7)):
            blade = 4 + ((seed + i * 5) % 4)
            sway = -1 if (self.steps // 16 + i) % 2 else 0
            pygame.draw.rect(screen, grass_dark, (x + sway, rect.y - blade, 2, blade))
            pygame.draw.rect(screen, grass, (x + sway, rect.y - blade, 1, max(1, blade - 1)))

        # Vines drooping from an exposed underside.
        if under_open and seed % 3 == 0:
            for i, x in enumerate(range(rect.left + 5, rect.right - 4, 12)):
                vine = 6 + ((seed + i * 7) % 8)
                pygame.draw.rect(screen, grass_dark, (x, rect.bottom - 1, 2, vine))
                pygame.draw.rect(screen, grass, (x, rect.bottom - 1, 1, vine - 2))
                pygame.draw.rect(screen, grass, (x - 1, rect.bottom - 1 + vine, 3, 2))

    def _draw_solid_tile(self: Any, screen, rect: pygame.Rect, col: int, row: int) -> None:
        palette = self._episode_palette()
        is_surface = not self._solid_at(col, row - 1)
        pygame.draw.rect(screen, (0, 0, 0), rect)

        if is_surface:
            left_edge = not self._solid_at(col - 1, row)
            right_edge = not self._solid_at(col + 1, row)
            under_open = not self._solid_at(col, row + 1)
            seed = col * 19 + row * 23 + self.level_index * 7
            pygame.draw.rect(screen, palette["rock_dark"], rect)
            pygame.draw.rect(screen, palette["platform_dark"], rect.inflate(-1, -1))
            pygame.draw.rect(screen, palette["platform"], (rect.x + 1, rect.y, rect.w - 2, 10))
            pygame.draw.rect(screen, EGA["K"], (rect.x, rect.y + 10, rect.w, 3))
            pygame.draw.rect(
                screen,
                palette["platform_light"],
                (rect.x + 1, rect.y + 1, rect.w - 2, 3),
            )
            pygame.draw.line(
                screen,
                EGA["W"] if self.level_index % len(self.CAVES) == 2 else EGA["C"],
                (rect.left + 3, rect.y + 5),
                (rect.right - 4, rect.y + 5),
                2,
            )
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (rect.left, rect.y + 12),
                (rect.right - 1, rect.y + 12),
                2,
            )
            pygame.draw.rect(screen, palette["rock_mid"], (rect.x + 2, rect.y + 13, rect.w - 4, 8))
            pygame.draw.rect(
                screen,
                palette["rock_dark"],
                (rect.x + 2, rect.y + 22, rect.w - 4, 8),
            )
            if left_edge:
                pygame.draw.rect(screen, palette["platform_light"], (rect.x, rect.y, 5, 15))
            if right_edge:
                pygame.draw.rect(screen, (0, 0, 0), (rect.right - 5, rect.y, 5, 15))
            for x in range(rect.left + 5, rect.right - 2, 12):
                bolt_color = palette["platform_light"] if (seed + x) % 2 else palette["rock_light"]
                pygame.draw.rect(screen, bolt_color, (x, rect.y + 15, 3, 3))
                if under_open:
                    pygame.draw.rect(screen, (0, 0, 0), (x + 1, rect.y + 23, 2, 7))
            if seed % 5 == 0:
                pygame.draw.rect(screen, palette["pipe_dark"], (rect.x + 7, rect.y + 22, 18, 4))
                pygame.draw.rect(screen, palette["pipe_light"], (rect.x + 9, rect.y + 22, 5, 2))
            # Edge-aware outline (CCV-03): only the exposed perimeter of a mass is
            # outlined, so adjacent/stacked tiles fuse into one carved ledge
            # instead of a grid of bricks. The bright lip stays the top edge.
            if left_edge:
                pygame.draw.rect(screen, EGA["K"], (rect.x, rect.y, 3, rect.h))
            if right_edge:
                pygame.draw.rect(screen, EGA["K"], (rect.right - 3, rect.y, 3, rect.h))
            if under_open:
                pygame.draw.rect(screen, EGA["K"], (rect.x, rect.bottom - 2, rect.w, 2))
            self._draw_ledge_growth(screen, rect, col, row, palette, under_open)
            return

        open_left = not self._solid_at(col - 1, row)
        open_right = not self._solid_at(col + 1, row)
        open_top = not self._solid_at(col, row - 1)
        open_bottom = not self._solid_at(col, row + 1)
        seed = col * 17 + row * 31 + self.level_index * 11
        variant = seed % 11

        pygame.draw.rect(screen, palette["rock_dark"], rect)
        inner = rect.inflate(-2, -2)
        pygame.draw.rect(screen, palette["rock"], inner)

        if open_left:
            pygame.draw.rect(screen, palette["rock_light"], (rect.x, rect.y, 4, rect.h))
            pygame.draw.rect(screen, palette["rock_mid"], (rect.x + 4, rect.y, 3, rect.h))
        if open_right:
            pygame.draw.rect(screen, (0, 0, 0), (rect.right - 4, rect.y, 4, rect.h))
            pygame.draw.rect(screen, palette["rock_mid"], (rect.right - 7, rect.y, 3, rect.h))
        if open_top:
            pygame.draw.rect(screen, palette["rock_light"], (rect.x, rect.y, rect.w, 4))
            pygame.draw.rect(screen, palette["rock_mid"], (rect.x, rect.y + 4, rect.w, 3))
        if open_bottom:
            pygame.draw.rect(screen, (0, 0, 0), (rect.x, rect.bottom - 4, rect.w, 4))
            pygame.draw.rect(screen, palette["rock_mid"], (rect.x, rect.bottom - 7, rect.w, 3))

        # Interior rock texture only — clean carved stone, never an embedded
        # machine plate or pipe (those read as "tiles inside the boundary").
        veins = (
            ((3 + seed % 8, 6), (16 + seed % 7, 4)),
            ((8, 17 + seed % 6), (25, 21 + seed % 4)),
            ((4 + seed % 5, 26), (14 + seed % 8, 23)),
        )
        for start, end in veins:
            pygame.draw.line(
                screen,
                palette["rock_mid"],
                (rect.x + start[0], rect.y + start[1]),
                (rect.x + end[0], rect.y + end[1]),
                2,
            )
        for x_off, y_off in ((5, 5), (21, 11), (12, 25)):
            if (seed + x_off + y_off) % 3:
                pygame.draw.rect(
                    screen,
                    palette["rock_light"],
                    (rect.x + x_off, rect.y + y_off, 3, 3),
                )

        if self.level_index % len(self.CAVES) == 1 and variant in (3, 7):
            pygame.draw.rect(screen, (255, 188, 80), (rect.x + 8, rect.y + 9, 5, 5))
            pygame.draw.rect(screen, (150, 54, 18), (rect.x + 16, rect.y + 18, 7, 4))
        elif self.level_index % len(self.CAVES) == 2 and variant in (2, 8):
            pygame.draw.rect(screen, (188, 220, 255), (rect.x + 7, rect.y + 7, 4, 4))
            pygame.draw.line(
                screen,
                (194, 128, 255),
                (rect.x + 11, rect.y + 21),
                (rect.x + 24, rect.y + 13),
                1,
            )

    def _draw_spike_tile(self: Any, screen, rect: pygame.Rect, col: int, row: int) -> None:
        phase = (self.steps + col * 5 + row * 3) % 36
        # Full-tile hazard volume (CCV-06): dark base, warning crust, six teeth
        # spanning most of the tile, framed by a black outline like solid tiles.
        pygame.draw.rect(screen, (44, 8, 8), rect)
        pygame.draw.rect(screen, (96, 0, 0), (rect.x, rect.y + 16, rect.w, rect.h - 16))
        pygame.draw.line(screen, EGA["Y"], (rect.left, rect.y + 15), (rect.right, rect.y + 15), 2)
        teeth = 6
        step = rect.w / teeth
        for i in range(teeth):
            x = rect.x + int(i * step)
            jut = 1 if (phase < 6 and i == phase % teeth) else 0
            tip_x = x + int(step / 2)
            tip_y = rect.y + 3 + jut
            base_y = rect.bottom - 2
            pygame.draw.polygon(
                screen,
                EGA["K"],
                [(x - 1, base_y), (tip_x, tip_y - 2), (x + int(step) + 1, base_y)],
            )
            pygame.draw.polygon(
                screen,
                EGA["W"],
                [(x + 1, base_y), (tip_x, tip_y), (x + int(step) - 1, base_y)],
            )
            pygame.draw.line(screen, EGA["w"], (tip_x, tip_y + 3), (tip_x, base_y - 3), 1)
        pygame.draw.rect(screen, EGA["K"], rect, 2)

    def _draw_acid_tile(self: Any, screen, rect: pygame.Rect, col: int, row: int) -> None:
        wave = (self.steps // 4 + col * 3) % 8
        # Full-tile molten pool (CCV-07): maroon body, bright animated crust,
        # rising bubbles, framed by a black outline.
        pygame.draw.rect(screen, (28, 0, 0), rect)
        pygame.draw.rect(screen, (130, 0, 0), (rect.x, rect.y + 8, rect.w, rect.h - 8))
        pygame.draw.rect(screen, EGA["t"], (rect.x, rect.y + 12, rect.w, rect.h - 12))
        for i in range(-4, rect.w + 4, 7):
            x = rect.x + i
            y = rect.y + 8 + ((i + wave) % 4)
            pygame.draw.rect(screen, EGA["O"], (x, y, 6, 4))
            pygame.draw.rect(screen, EGA["L"], (x + 1, y, 3, 2))
        for i in range(3):
            bubble_x = rect.x + 4 + ((self.steps + col * 11 + i * 13) % (rect.w - 8))
            bubble_y = rect.y + 16 + ((self.steps // 3 + row + i * 5) % (rect.h - 18))
            pygame.draw.circle(screen, EGA["Y"], (bubble_x, bubble_y), 2)
        pygame.draw.rect(screen, EGA["K"], rect, 2)

    def _draw_ladders(self: Any, screen, camera_x: int, camera_y: int) -> None:
        """Draw a wooden ladder down every 1-wide vertical shaft (an EMPTY tile
        flanked by SOLID rock that continues vertically). These are the carved
        connectors between galleries; dressing them as ladders reads like the
        DOS-era caves without adding a climb mechanic (the player still jumps).
        """
        ts = self.TILE_SIZE
        first_col = max(1, camera_x // ts)
        last_col = min(self.level_cols - 1, (camera_x + self.width) // ts + 2)
        first_row = max(0, camera_y // ts)
        last_row = min(self.level_rows, (camera_y + self.height - self.HUD_HEIGHT) // ts + 2)
        rail, rung = EGA["N"], EGA["S"]
        for row in range(first_row, last_row):
            grid_row = self.grid[row]
            for col in range(first_col, last_col):
                if grid_row[col] != self.EMPTY:
                    continue
                if grid_row[col - 1] != self.SOLID or grid_row[col + 1] != self.SOLID:
                    continue
                above = row > 0 and self.grid[row - 1][col] == self.EMPTY
                below = row + 1 < self.level_rows and self.grid[row + 1][col] == self.EMPTY
                if not (above or below):
                    continue  # a single pocket, not a climbable shaft
                x, y = self._world_to_screen(col * ts, row * ts, camera_x, camera_y)
                lx, rx = x + 7, x + ts - 8
                pygame.draw.line(screen, rail, (lx, y), (lx, y + ts), 2)
                pygame.draw.line(screen, rail, (rx, y), (rx, y + ts), 2)
                for ry in (y + 6, y + 16, y + 26):
                    pygame.draw.line(screen, rung, (lx, ry), (rx, ry), 2)

    def _draw_elevators(self: Any, screen, camera_x: int, camera_y: int) -> None:
        """Draw each elevator's vertical guide track and its moving metal platform
        at the platform's current position (CCV: switch-free lift transport)."""
        ts = self.TILE_SIZE
        for e in self.elevators:
            tx = e.col * ts
            track_x = tx + ts // 2
            top_y = e.top * ts
            bot_y = (e.bottom + 1) * ts
            sx, sy_top = self._world_to_screen(track_x, top_y, camera_x, camera_y)
            _, sy_bot = self._world_to_screen(track_x, bot_y, camera_x, camera_y)
            # twin guide rails framing the shaft
            for rail_dx in (-ts // 2 + 3, ts // 2 - 3):
                pygame.draw.line(
                    screen, EGA["m"], (sx + rail_dx, sy_top), (sx + rail_dx, sy_bot), 2
                )
            # the platform: a chunky metal bar the player rides
            px, py = self._world_to_screen(tx, int(e.pos * ts), camera_x, camera_y)
            plate = pygame.Rect(px + 1, py + ts - 9, ts - 2, 9)
            pygame.draw.rect(screen, EGA["M"], plate)
            pygame.draw.rect(screen, EGA["w"], (plate.x, plate.y, plate.width, 3))
            pygame.draw.rect(screen, EGA["m"], plate, 1)

    def _draw_agent_overlay(self: Any, screen, camera_x: int, camera_y: int) -> None:
        """Educational overlay (toggle with 'O' in play/human mode): outlines the
        exact tile window the DQN agent perceives (WINDOW_COLS x WINDOW_ROWS,
        centred on the player) and draws a compass to its current target. Makes
        the agent's narrow field of view and its goal legible — when the goal
        line runs off the perception box, you can SEE it's chasing something it
        cannot currently observe."""
        ts = self.TILE_SIZE
        pcol, prow = self._player_tile()
        wx = (pcol - self.WINDOW_COLS // 2) * ts - camera_x
        wy = (prow - self.WINDOW_ROWS // 2) * ts - camera_y
        ww, wh = self.WINDOW_COLS * ts, self.WINDOW_ROWS * ts
        tint = pygame.Surface((ww, wh), pygame.SRCALPHA)
        tint.fill((90, 200, 255, 26))
        screen.blit(tint, (wx, wy))
        pygame.draw.rect(screen, (120, 220, 255), (wx, wy, ww, wh), 2)
        if self._tiny_font:
            lbl = self._tiny_font.render(
                f"AGENT VIEW {self.WINDOW_COLS}x{self.WINDOW_ROWS}", True, (150, 230, 255)
            )
            screen.blit(lbl, (wx + 3, max(2, wy - 11)))

        target, _ = self._current_target()
        if target is not None:
            kind, tcol, trow = target
            px, py = self._player_center()
            sx, sy = self._world_to_screen(px, py, camera_x, camera_y)
            tx, ty = self._tile_center((tcol, trow))
            ex, ey = self._world_to_screen(tx, ty, camera_x, camera_y)
            color = {"crystal": EGA["G"], "switch": EGA["A"], "exit": EGA["C"]}.get(
                kind, EGA["W"]
            )
            pygame.draw.line(screen, color, (sx, sy), (ex, ey), 2)
            pygame.draw.circle(screen, color, (ex, ey), 6, 2)
            if self._tiny_font:
                screen.blit(self._tiny_font.render(f"goal:{kind}", True, color), (ex + 7, ey - 6))

        if self._tiny_font:
            prog = getattr(self, "_progress", 0.0)
            read = self._tiny_font.render(
                f"progress {prog:.2f}  crystals left {len(self.crystals)}",
                True,
                (220, 220, 235),
            )
            screen.blit(read, (6, self.height - self.HUD_HEIGHT - 14))

    def _draw_switch_wires(self: Any, screen, camera_x: int, camera_y: int) -> None:
        """Draw a taut cable from each switch to the nearest door it controls
        (CCV-18). The core glows green once thrown, amber while still armed, so
        the switch->target relationship is readable like the DOS references.
        """
        if not self.switches or not self.doors:
            return
        half = self.TILE_SIZE // 2
        for switch in self.switches:
            sc, sr = switch
            color = self.switch_color.get(switch, "red")
            same_color = [d for d in self.doors if self.door_color.get(d, "red") == color]
            if not same_color:
                continue
            door = min(same_color, key=lambda d: (d[0] - sc) ** 2 + (d[1] - sr) ** 2)
            sx, sy = self._world_to_screen(
                sc * self.TILE_SIZE + half,
                sr * self.TILE_SIZE + half,
                camera_x,
                camera_y,
            )
            dx, dy = self._world_to_screen(
                door[0] * self.TILE_SIZE + half,
                door[1] * self.TILE_SIZE + half,
                camera_x,
                camera_y,
            )
            core = EGA["G"] if switch in self.used_switches else EGA["A"]
            # Route the cable orthogonally (horizontal then vertical) so it reads
            # as conduit pinned to the rock, not a diagonal debug line strung
            # across open space. The elbow turns at the door's column.
            elbow = (dx, sy)
            points = [(sx, sy), elbow, (dx, dy)]
            pygame.draw.lines(screen, EGA["K"], False, points, 3)
            pygame.draw.lines(screen, core, False, points, 1)
            # Small anchor bolts at each end + the elbow seat the cable.
            for px_, py_ in ((sx, sy), elbow, (dx, dy)):
                pygame.draw.circle(screen, EGA["m"], (px_, py_), 2)

    def _draw_tiles(self: Any, screen, camera_x: int, camera_y: int) -> None:
        first_col = max(0, camera_x // self.TILE_SIZE)
        last_col = min(self.level_cols, (camera_x + self.width) // self.TILE_SIZE + 2)
        first_row = max(0, camera_y // self.TILE_SIZE)
        last_row = min(
            self.level_rows,
            (camera_y + self.height - self.HUD_HEIGHT) // self.TILE_SIZE + 2,
        )

        for row in range(first_row, last_row):
            for col in range(first_col, last_col):
                x, y = self._world_to_screen(
                    col * self.TILE_SIZE,
                    row * self.TILE_SIZE,
                    camera_x,
                    camera_y,
                )
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                tile = (col, row)

                if self.grid[row][col] == self.SOLID:
                    self._draw_solid_tile(screen, rect, col, row)
                elif tile in self.hazards:
                    if self.hazard_kinds.get(tile) == self.ACID:
                        self._draw_acid_tile(screen, rect, col, row)
                    else:
                        self._draw_spike_tile(screen, rect, col, row)
                elif tile in self.doors and not self._door_open(tile):
                    self._draw_locked_door(screen, rect, self.door_color.get(tile, "red"))
                elif tile == self.exit_pos:
                    self._draw_exit_airlock(screen, rect)
                elif tile in self.switches:
                    self._draw_lever_switch(
                        screen,
                        rect,
                        tile in self.used_switches,
                        self.switch_color.get(tile, "red"),
                    )

    def _draw_gravity_overlay(self: Any, screen) -> None:
        """Full-screen treatment while the gravity field is inverted (CCV-19):
        a violet edge vignette, debris floating upward, and a period-style
        REVERSE GRAVITY banner — so the altered field reads at a glance.
        """
        if self.gravity_timer <= 0:
            return
        play_bottom = self.height - self.HUD_HEIGHT
        fade = min(1.0, self.gravity_timer / 60.0)
        overlay = pygame.Surface((self.width, play_bottom), pygame.SRCALPHA)
        tint = (150, 60, 230)

        band = 56
        for i in range(band):
            alpha = int(64 * fade * (1 - i / band))
            pygame.draw.line(overlay, (*tint, alpha), (0, i), (self.width, i))
            pygame.draw.line(
                overlay,
                (*tint, alpha),
                (0, play_bottom - 1 - i),
                (self.width, play_bottom - 1 - i),
            )

        # Debris floats upward to sell the inverted field.
        for k in range(26):
            sx = (k * 137 + (self.steps * 2)) % self.width
            sy = play_bottom - ((self.steps * 3 + k * 53) % play_bottom)
            pygame.draw.rect(overlay, (*EGA["C"], 150), (sx, sy, 2, 4))
        screen.blit(overlay, (0, 0))

        if self._art:
            label = "REVERSE GRAVITY"
            text_w = self._art.text(label, EGA["Y"], scale=2).get_width()
            bx = (self.width - text_w) // 2
            sign = pygame.Rect(bx - 8, 10, text_w + 16, 24)
            pygame.draw.rect(screen, EGA["A"], sign)
            pygame.draw.rect(screen, EGA["K"], sign, 2)
            self._art.draw_text(screen, label, bx, 16, EGA["Y"], scale=2)

    def _draw_heart(self: Any, screen, x: int, y: int, alive: bool) -> None:
        """Draw a small pixel heart pip (bright-red alive, dark husk when lost)."""
        body = (255, 85, 85) if alive else (62, 30, 30)
        pygame.draw.circle(screen, body, (x + 4, y + 4), 4)
        pygame.draw.circle(screen, body, (x + 12, y + 4), 4)
        pygame.draw.polygon(screen, body, [(x, y + 5), (x + 16, y + 5), (x + 8, y + 15)])
        if alive:
            pygame.draw.rect(screen, (255, 190, 190), (x + 2, y + 2, 2, 2))

    def _draw_hud(self: Any, screen) -> None:
        hud_y = self.height - self.HUD_HEIGHT
        # Period-authentic footer (CCV-04/05): one thin black bar with a bright
        # top rule and label-less data clusters — score, crystals, ammo, hearts.
        # No CAVE/EXIT/MYLO labels, no compartment dividers, no controls line.
        pygame.draw.rect(screen, EGA["K"], (0, hud_y, self.width, self.HUD_HEIGHT))
        pygame.draw.line(screen, EGA["G"], (0, hud_y), (self.width, hud_y), 1)
        pygame.draw.line(screen, EGA["g"], (0, hud_y + 2), (self.width, hud_y + 2), 1)

        collected = self.initial_crystals - len(self.crystals)
        row = hud_y + 10

        if self._art:
            # Score — gold "$" + chunky green numerals.
            self._art.draw_text(screen, "$", 12, row, EGA["Y"], scale=2)
            self._art.draw_text(screen, f"{self.score:06d}", 34, row, EGA["G"], scale=2)

            # Crystals collected toward the exit goal.
            self._art.draw_sprite(screen, "crystal_blue", 198, row - 1, scale=1)
            self._art.draw_text(
                screen,
                f"{collected:02d}/{self.initial_crystals:02d}",
                226,
                row,
                EGA["Y"],
                scale=2,
            )

            # Ammo — raygun icon + green count.
            self._art.draw_sprite(screen, "raygun", 380, row + 1, scale=1)
            self._art.draw_text(screen, f"{self.ammo:02d}", 418, row, EGA["G"], scale=2)

            # Exit lock state — a single iconographic cue, no text label.
            self._art.draw_sprite(
                screen,
                "door_open" if self.exit_unlocked else "door_locked",
                494,
                row - 5,
                scale=1,
            )

            # Hearts — right-aligned; lost hearts drop to a dark husk.
            for i in range(self.MAX_HEALTH):
                hx = self.width - 26 - (self.MAX_HEALTH - 1 - i) * 26
                self._draw_heart(screen, hx, row, i < self.health)
            return

        if not self._small_font:
            return

        # Font fallback (no pixel-art atlas) — same label-less layout.
        score_text = self._small_font.render(f"$ {self.score:06d}", True, (64, 236, 80))
        screen.blit(score_text, (12, hud_y + 10))

        gem_color = self.GEM_COLORS[collected % len(self.GEM_COLORS)]
        gem_x = 210
        pygame.draw.polygon(
            screen,
            gem_color,
            [
                (gem_x + 9, hud_y + 6),
                (gem_x + 18, hud_y + 16),
                (gem_x + 9, hud_y + 28),
                (gem_x, hud_y + 16),
            ],
        )
        gem_text = self._small_font.render(
            f"{collected}/{self.initial_crystals}", True, (255, 224, 64)
        )
        screen.blit(gem_text, (gem_x + 26, hud_y + 10))

        ammo_x = 360
        pygame.draw.rect(screen, (220, 220, 230), (ammo_x, hud_y + 14, 18, 5))
        pygame.draw.polygon(
            screen,
            (255, 230, 78),
            [
                (ammo_x + 18, hud_y + 12),
                (ammo_x + 25, hud_y + 16),
                (ammo_x + 18, hud_y + 20),
            ],
        )
        ammo_text = self._small_font.render(str(self.ammo), True, (64, 236, 80))
        screen.blit(ammo_text, (ammo_x + 34, hud_y + 10))

        for i in range(self.MAX_HEALTH):
            hx = self.width - 26 - (self.MAX_HEALTH - 1 - i) * 26
            self._draw_heart(screen, hx, hud_y + 10, i < self.health)
