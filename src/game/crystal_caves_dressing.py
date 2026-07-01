"""Prop/dressing and actor rendering mixin for Crystal Caves."""

from __future__ import annotations

from typing import Any, List, Set, Tuple

import pygame

from .crystal_caves_art import EGA
from .crystal_caves_entities import DressingPiece


class CrystalCavesDressingMixin:
    def _draw_locked_door(self: Any, screen, rect: pygame.Rect, color: str = "red") -> None:
        body = (40, 70, 200) if color == "blue" else (160, 0, 0)
        edge = (96, 150, 255) if color == "blue" else (255, 78, 78)
        if self._art:
            self._art.draw_sprite(screen, "door_locked", rect.x, rect.y, scale=2)
            # a colour key bar so red vs blue locks read at a glance
            pygame.draw.rect(screen, edge, (rect.x + 10, rect.y + 2, 12, 4))
            return
        pygame.draw.rect(screen, (8, 8, 16), rect)
        pygame.draw.rect(screen, body, rect.inflate(-8, -2))
        pygame.draw.rect(screen, edge, rect.inflate(-8, -2), 2)
        pygame.draw.rect(screen, (210, 210, 220), (rect.x + 11, rect.y + 6, 10, 20))
        pygame.draw.rect(screen, (0, 0, 0), (rect.x + 14, rect.y + 9, 4, 14))

    def _draw_exit_airlock(self: Any, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(
                screen,
                "door_open" if self.exit_unlocked else "door_locked",
                rect.x,
                rect.y,
                scale=2,
            )
            if self.exit_unlocked:
                pulse = 1 + (self.steps // 8) % 3
                pygame.draw.rect(screen, (98, 255, 98), rect.inflate(pulse, pulse), 2)
            return
        glow = (70, 255, 82) if self.exit_unlocked else (120, 120, 130)
        door = (0, 112, 44) if self.exit_unlocked else (68, 68, 82)
        pygame.draw.rect(screen, (0, 0, 0), rect)
        pygame.draw.rect(screen, (210, 210, 220), rect.inflate(-2, 0), 2)
        pygame.draw.rect(screen, (118, 118, 132), (rect.x + 3, rect.y + 2, 5, rect.h - 4))
        pygame.draw.rect(screen, door, rect.inflate(-10, -6))
        pygame.draw.rect(screen, glow, rect.inflate(-16, -14))
        pygame.draw.rect(screen, (0, 0, 0), rect.inflate(-21, -21))
        pygame.draw.rect(screen, (255, 255, 255), (rect.x + 7, rect.y + 5, 3, 4))
        if self._tiny_font:
            label = self._tiny_font.render("EXIT", True, glow)
            screen.blit(label, label.get_rect(center=(rect.centerx, rect.y + 6)))

    def _draw_lever_switch(
        self: Any, screen, rect: pygame.Rect, used: bool, color: str = "red"
    ) -> None:
        key = (96, 150, 255) if color == "blue" else (255, 80, 48)
        if self._art:
            self._art.draw_sprite(
                screen,
                "switch_on" if used else "switch_off",
                rect.x,
                rect.y,
                scale=2,
            )
            pygame.draw.rect(screen, key, (rect.x + 10, rect.y + 1, 12, 3))
            return
        pygame.draw.rect(screen, (0, 0, 0), rect)
        pygame.draw.rect(screen, (166, 166, 176), (rect.x + 7, rect.y + 20, 18, 8))
        base_color = (70, 255, 80) if used else key
        pivot = (rect.x + 16, rect.y + 20)
        lever_end = (rect.x + 23, rect.y + 8) if used else (rect.x + 9, rect.y + 7)
        pygame.draw.line(screen, (230, 230, 230), pivot, lever_end, 3)
        pygame.draw.circle(screen, base_color, lever_end, 5)
        pygame.draw.rect(screen, (0, 0, 0), (rect.x + 9, rect.y + 24, 14, 3))

    def _draw_surface_props(self: Any, screen, camera_x: int, camera_y: int) -> None:
        """Transmitter pylons and a MINE -> sign along the planet surface of a
        sky-entrance level (CaveSpec.sky_rows)."""
        sky_rows = getattr(self, "sky_rows", 0)
        if not sky_rows:
            return
        surface_y = sky_rows * self.TILE_SIZE - camera_y
        if surface_y < 10 or surface_y > self.height - self.HUD_HEIGHT:
            return

        spacing = 112
        start = 36 - camera_x % spacing
        for x in range(start, self.width + spacing, spacing):
            pygame.draw.rect(screen, EGA["K"], (x - 4, surface_y - 22, 10, 24))
            pygame.draw.rect(screen, (120, 124, 140), (x - 2, surface_y - 22, 5, 24))
            for ring_y in (surface_y - 20, surface_y - 15, surface_y - 10):
                pygame.draw.rect(screen, (180, 180, 196), (x - 5, ring_y, 11, 2))
            pygame.draw.rect(screen, EGA["C"], (x - 1, surface_y - 26, 3, 3))

        sign_x = 150 - camera_x
        if -80 < sign_x < self.width:
            sign = pygame.Rect(sign_x, surface_y - 32, 66, 18)
            pygame.draw.rect(screen, EGA["K"], (sign.x + 6, sign.bottom, 4, 14))
            pygame.draw.rect(screen, EGA["K"], (sign.right - 12, sign.bottom, 4, 14))
            pygame.draw.rect(screen, EGA["K"], sign.inflate(4, 4))
            pygame.draw.rect(screen, (150, 14, 22), sign)
            pygame.draw.rect(screen, EGA["A"], sign, 2)
            if self._art:
                self._art.draw_text(screen, "MINE", sign.x + 6, sign.y + 5, EGA["Y"], scale=1)
            arrow_x = sign.right - 13
            pygame.draw.polygon(
                screen,
                EGA["Y"],
                [(arrow_x, sign.y + 5), (arrow_x + 8, sign.y + 9), (arrow_x, sign.y + 13)],
            )

    def _draw_level_dressing(self: Any, screen, camera_x: int, camera_y: int) -> None:
        self._draw_surface_props(screen, camera_x, camera_y)
        first_col = max(0, camera_x // self.TILE_SIZE)
        last_col = min(self.level_cols, (camera_x + self.width) // self.TILE_SIZE + 2)
        first_row = max(0, camera_y // self.TILE_SIZE)
        last_row = min(
            self.level_rows,
            (camera_y + self.height - self.HUD_HEIGHT) // self.TILE_SIZE + 2,
        )

        # Vertical rails hanging from platforms.
        for col in range(first_col, last_col):
            for row in range(first_row, last_row):
                if not self._should_draw_support_rail(col, row):
                    continue
                x, y = self._world_to_screen(
                    col * self.TILE_SIZE,
                    row * self.TILE_SIZE,
                    camera_x,
                    camera_y,
                )
                self._draw_support_rail(screen, pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE))

        self._draw_authored_dressing(
            screen, camera_x, camera_y, first_col, last_col, first_row, last_row
        )

        for tile in self._visible_tiles(self.hazards, first_col, last_col, first_row, last_row):
            if not self._should_draw_hazard_sign(tile):
                continue
            if self.hazard_kinds.get(tile) == self.ACID:
                if not self._same_hazard_at(tile[0] - 1, tile[1], self.ACID):
                    self._draw_sign_for_tile(
                        screen, camera_x, camera_y, tile, "ACID", (255, 230, 64)
                    )
            elif not self._same_hazard_at(tile[0] - 1, tile[1], self.SPIKE):
                self._draw_sign_for_tile(screen, camera_x, camera_y, tile, "DANGER", (255, 72, 72))

        for tile, power in self.powerups.items():
            if not (first_col <= tile[0] < last_col and first_row <= tile[1] < last_row):
                continue
            label = {
                self.POWER_SHOT: "POWER",
                self.GRAVITY_POWER: "LOW G",
                self.FREEZE_POWER: "STOP",
            }[power]
            self._draw_sign_for_tile(screen, camera_x, camera_y, tile, label, (255, 216, 64))

        for tank in self._visible_tiles(self.air_tanks, first_col, last_col, first_row, last_row):
            self._draw_sign_for_tile(screen, camera_x, camera_y, tank, "AIR", (88, 240, 255))

        if self.level_index % len(self.CAVES) != 0:
            # Mine props on empty floor-adjacent cells. Deterministic so
            # screenshots and tests remain stable.
            for row in range(first_row, last_row):
                for col in range(first_col, last_col):
                    if not self._empty_dressing_cell(col, row):
                        continue
                    selector = (col * 13 + row * 17 + self.level_index * 5) % 37
                    x, y = self._world_to_screen(
                        col * self.TILE_SIZE,
                        row * self.TILE_SIZE,
                        camera_x,
                        camera_y,
                    )
                    rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                    if selector == 0:
                        self._draw_crate(screen, rect)
                    elif selector == 7:
                        self._draw_pickaxe(screen, rect)
                    elif selector == 13:
                        self._draw_lamp(screen, rect)

    def _should_draw_hazard_sign(self: Any, tile: Tuple[int, int]) -> bool:
        """Sparse deterministic hazard signage; hazards themselves remain legible."""

        return (tile[0] * 11 + tile[1] * 7 + self.level_index) % 3 == 0

    @staticmethod
    def _visible_tiles(
        tiles: Set[Tuple[int, int]],
        first_col: int,
        last_col: int,
        first_row: int,
        last_row: int,
    ) -> List[Tuple[int, int]]:
        return [
            tile
            for tile in tiles
            if first_col <= tile[0] < last_col and first_row <= tile[1] < last_row
        ]

    def _draw_authored_dressing(
        self: Any,
        screen,
        camera_x: int,
        camera_y: int,
        first_col: int,
        last_col: int,
        first_row: int,
        last_row: int,
    ) -> None:
        pieces = self.CAVE_DRESSING.get(self.level_index % len(self.CAVES), ())
        for piece in pieces:
            length = self._dressing_length(piece)
            if (
                piece.col + length < first_col
                or piece.col >= last_col
                or piece.row < first_row - 1
                or piece.row >= last_row + 1
            ):
                continue
            x, y = self._world_to_screen(
                piece.col * self.TILE_SIZE,
                piece.row * self.TILE_SIZE,
                camera_x,
                camera_y,
            )
            rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
            self._draw_dressing_piece(screen, rect, piece)

    @staticmethod
    def _dressing_length(piece: DressingPiece) -> int:
        if piece.kind not in {"cable_h", "clear_blocks"}:
            return 1
        try:
            return max(1, int(piece.label))
        except ValueError:
            return 1

    def _draw_dressing_piece(self: Any, screen, rect: pygame.Rect, piece: DressingPiece) -> None:
        if piece.kind == "cable_h":
            self._draw_cable_run(screen, rect, self._dressing_length(piece))
            return
        if piece.kind == "elevator_frame":
            self._draw_elevator_frame(screen, rect, piece.label)
            return
        if piece.kind == "crystal_light":
            self._draw_crystal_light(screen, rect)
            return
        if piece.kind == "clear_blocks":
            self._draw_clear_block_run(screen, rect, self._dressing_length(piece))
            return
        if piece.kind == "room_label":
            self._draw_room_label(screen, rect, piece.label)
            return

        sprite_name = {
            "beacon": "beacon",
            "mine_sign": "mine_sign",
            "generator": "generator",
            "terminal": "terminal",
            "pipe_stack": "pipe_stack",
            "warning_post": "warning_post",
            "mushroom": "mushroom",
            "hammer_marker": "hammer_marker",
            "zapper": "zapper",
            "vacuum": "vacuum",
            "eye_turret": "eye_turret",
            "slug_enemy": "slug_enemy",
            "bat_perch": "bat_enemy",
        }.get(piece.kind)
        if sprite_name and self._art:
            self._art.draw_sprite(screen, sprite_name, rect.x, rect.y, scale=2)
            if piece.kind == "mine_sign":
                label = piece.label or "MINE"
                self._art.draw_text(
                    screen,
                    label[:4],
                    rect.x + 5,
                    rect.y + 7,
                    EGA["K"],
                    scale=1,
                    shadow=False,
                )
            return

        pygame.draw.rect(screen, (0, 0, 0), rect.inflate(-4, -8))
        pygame.draw.rect(screen, (116, 116, 132), rect.inflate(-8, -12))
        pygame.draw.rect(screen, (255, 224, 64), (rect.x + 11, rect.y + 10, 5, 5))

    def _draw_clear_block_run(self: Any, screen, rect: pygame.Rect, length: int) -> None:
        phase = (self.steps // 12) % 2
        for index in range(length):
            x = rect.x + index * self.TILE_SIZE
            y = rect.y
            block = pygame.Rect(x + 2, y + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
            pygame.draw.rect(screen, EGA["K"], block.inflate(4, 4))
            glass = pygame.Surface((block.w, block.h), pygame.SRCALPHA)
            pygame.draw.rect(glass, (88, 232, 255, 44), (0, 0, block.w, block.h))
            pygame.draw.rect(glass, (255, 255, 255, 110), (0, 0, block.w, block.h), 2)
            pygame.draw.line(glass, (255, 255, 255, 125), (4, 5), (block.w - 7, 5), 2)
            pygame.draw.line(
                glass,
                (88, 232, 255, 120),
                (block.w - 7, 8),
                (7, block.h - 8),
                2,
            )
            screen.blit(glass, block.topleft)
            if phase:
                pygame.draw.rect(screen, EGA["W"], (x + 8, y + 7, 9, 2))
                pygame.draw.rect(screen, EGA["C"], (x + 20, y + 21, 5, 2))

    def _draw_room_label(self: Any, screen, rect: pygame.Rect, label: str) -> None:
        if not self._art or not label:
            return
        # Fit the plate to the measured text so it never overflows the border.
        pad = 7
        text_width = self._art.text(label, EGA["Y"], scale=1).get_width()
        width = max(34, text_width + pad * 2)
        sign = pygame.Rect(rect.x, rect.y + 7, width, 18)
        pygame.draw.rect(screen, EGA["K"], sign.inflate(4, 4))
        pygame.draw.rect(screen, (28, 28, 54), sign)
        pygame.draw.rect(screen, EGA["Y"], sign, 1)
        self._art.draw_text(screen, label, sign.x + pad, sign.y + 5, EGA["Y"], scale=1)

    def _draw_cable_run(self: Any, screen, rect: pygame.Rect, length: int) -> None:
        palette = self._episode_palette()
        run = pygame.Rect(rect.x, rect.y + 13, self.TILE_SIZE * length, 9)
        pygame.draw.rect(screen, (0, 0, 0), run.inflate(4, 4))
        pygame.draw.rect(screen, palette["pipe_shadow"], run.move(3, 4))
        pygame.draw.rect(screen, palette["pipe_dark"], run)
        pygame.draw.line(
            screen,
            palette["pipe_light"],
            (run.left + 2, run.top + 2),
            (run.right - 3, run.top + 2),
            2,
        )
        for x in range(run.left + 11, run.right, 28):
            pygame.draw.rect(screen, (0, 0, 0), (x - 2, run.top - 3, 7, run.h + 6))
            pygame.draw.rect(screen, palette["pipe_light"], (x, run.top - 1, 3, run.h + 2))

    def _draw_elevator_frame(self: Any, screen, rect: pygame.Rect, label: str = "EXIT") -> None:
        palette = self._episode_palette()
        frame = pygame.Rect(rect.x - 13, rect.y - 18, 58, 58)
        pygame.draw.rect(screen, (0, 0, 0), frame.inflate(6, 6))
        pygame.draw.rect(screen, palette["pipe_shadow"], frame)
        pygame.draw.rect(screen, (176, 176, 190), (frame.x, frame.y, 8, frame.h))
        pygame.draw.rect(screen, (86, 86, 104), (frame.right - 8, frame.y, 8, frame.h))
        pygame.draw.rect(screen, (176, 176, 190), (frame.x, frame.y, frame.w, 8))
        pygame.draw.rect(screen, (86, 86, 104), (frame.x, frame.bottom - 8, frame.w, 8))
        pygame.draw.rect(screen, palette["pipe_light"], (frame.x + 4, frame.y + 4, 8, 8))
        pygame.draw.rect(
            screen,
            EGA["G"] if self.exit_unlocked else EGA["A"],
            (frame.right - 12, frame.y + 4, 8, 8),
        )
        if self._art:
            self._art.draw_text(
                screen,
                label or "EXIT",
                frame.x + 15,
                frame.y + 5,
                EGA["G"] if self.exit_unlocked else EGA["Y"],
                scale=1,
            )

    def _draw_crystal_light(self: Any, screen, rect: pygame.Rect) -> None:
        glow = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.circle(glow, (88, 232, 255, 45), (32, 32), 30)
        pygame.draw.circle(glow, (255, 255, 255, 36), (32, 32), 13)
        screen.blit(glow, (rect.x - 16, rect.y - 16))
        if self._art:
            self._art.draw_sprite(screen, "lamp", rect.x, rect.y, scale=2)
        else:
            self._draw_lamp(screen, rect)

    def _should_draw_support_rail(self: Any, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.level_cols or row >= self.level_rows:
            return False
        if self.level_index % len(self.CAVES) == 0:
            return False
        if self._solid_at(col, row):
            return False
        if not self._solid_at(col, row - 1):
            return False
        if self._solid_at(col, row + 1):
            return False
        return (col * 5 + row * 2 + self.level_index) % 19 == 0

    def _empty_dressing_cell(self: Any, col: int, row: int) -> bool:
        tile = (col, row)
        if col <= 0 or row <= 0 or col >= self.level_cols - 1 or row >= self.level_rows - 1:
            return False
        if self._solid_at(col, row) or not self._solid_at(col, row + 1):
            return False
        occupied = (
            self.crystals
            | self.switches
            | self.hazards
            | self.ammo_pickups
            | self.treasures
            | self.air_tanks
            | set(self.powerups.keys())
            | {self.exit_pos}
        )
        return tile not in occupied

    def _same_hazard_at(self: Any, col: int, row: int, kind: str) -> bool:
        return (col, row) in self.hazards and self.hazard_kinds.get((col, row)) == kind

    def _draw_support_rail(self: Any, screen, rect: pygame.Rect) -> None:
        palette = self._episode_palette()
        x = rect.centerx - 2
        pygame.draw.rect(screen, (0, 0, 0), (x - 2, rect.y, 8, rect.h))
        pygame.draw.rect(screen, palette["pipe_dark"], (x, rect.y, 4, rect.h))
        for y in range(rect.y + 2, rect.bottom, 8):
            pygame.draw.line(screen, palette["pipe_light"], (x - 2, y), (x + 6, y + 5), 1)

    def _draw_sign_for_tile(
        self: Any,
        screen,
        camera_x: int,
        camera_y: int,
        tile: Tuple[int, int],
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        col, row = tile
        x, y = self._world_to_screen(
            col * self.TILE_SIZE,
            (row - 1) * self.TILE_SIZE,
            camera_x,
            camera_y,
        )
        if y < -self.TILE_SIZE or y > self.height - self.HUD_HEIGHT:
            return
        # Size the plate to its contents so text never spills past the border:
        # icon width (DANGER triangle) + measured text width + symmetric padding.
        text_width = 0
        if self._art:
            text_width = self._art.text(label, color, scale=1).get_width()
        elif self._tiny_font:
            text_width = self._tiny_font.size(label)[0]
        icon_w = 18 if label == "DANGER" else 0
        pad = 7
        sign_width = max(34, icon_w + text_width + pad * 2)
        sign = pygame.Rect(x + 1, y + 8, sign_width, 18)
        pygame.draw.rect(screen, (0, 0, 0), sign.inflate(4, 4))
        fill = (86, 0, 0) if label == "DANGER" else (72, 46, 0)
        pygame.draw.rect(screen, fill, sign)
        pygame.draw.rect(screen, color, sign, 2)
        if label == "DANGER":
            pygame.draw.polygon(
                screen,
                EGA["Y"],
                [
                    (sign.x + 5, sign.y + 14),
                    (sign.x + 11, sign.y + 4),
                    (sign.x + 17, sign.y + 14),
                ],
            )
            pygame.draw.rect(screen, EGA["K"], (sign.x + 10, sign.y + 8, 2, 4))
            pygame.draw.rect(screen, EGA["K"], (sign.x + 10, sign.y + 13, 2, 2))
            text_x = sign.x + icon_w + pad
        else:
            text_x = sign.x + pad
        if self._art:
            self._art.draw_text(
                screen,
                label,
                text_x,
                sign.y + 6,
                color,
                scale=1,
                shadow=False,
            )
        elif self._tiny_font:
            text = self._tiny_font.render(label, True, color)
            screen.blit(text, text.get_rect(center=sign.center))

    def _draw_crate(self: Any, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "crate", rect.x, rect.y, scale=2)
            return
        crate = pygame.Rect(rect.x + 6, rect.y + 10, 20, 18)
        pygame.draw.rect(screen, (0, 0, 0), crate.inflate(2, 2))
        pygame.draw.rect(screen, (156, 82, 26), crate)
        pygame.draw.rect(screen, (236, 150, 52), crate, 2)
        pygame.draw.line(screen, (92, 44, 14), crate.topleft, crate.bottomright, 2)
        pygame.draw.line(screen, (92, 44, 14), crate.topright, crate.bottomleft, 2)

    def _draw_pickaxe(self: Any, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "pickaxe", rect.x, rect.y, scale=2)
            return
        pygame.draw.line(
            screen,
            (176, 118, 62),
            (rect.x + 9, rect.y + 25),
            (rect.x + 21, rect.y + 10),
            3,
        )
        pygame.draw.line(
            screen,
            (220, 220, 230),
            (rect.x + 10, rect.y + 10),
            (rect.x + 27, rect.y + 7),
            3,
        )
        pygame.draw.line(
            screen,
            (220, 220, 230),
            (rect.x + 12, rect.y + 10),
            (rect.x + 5, rect.y + 16),
            3,
        )
        pygame.draw.line(
            screen, (0, 0, 0), (rect.x + 9, rect.y + 25), (rect.x + 21, rect.y + 10), 1
        )

    def _draw_lamp(self: Any, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "lamp", rect.x, rect.y, scale=2)
            return
        pygame.draw.rect(screen, (0, 0, 0), (rect.x + 8, rect.y + 10, 16, 18))
        pygame.draw.rect(screen, (90, 90, 105), (rect.x + 11, rect.y + 13, 10, 13))
        pygame.draw.rect(screen, (255, 230, 64), (rect.x + 13, rect.y + 15, 6, 6))
        pygame.draw.line(
            screen,
            (200, 200, 210),
            (rect.x + 16, rect.y + 8),
            (rect.x + 16, rect.y + 13),
            2,
        )

    def _draw_pickups(self: Any, screen, camera_x: int, camera_y: int) -> None:
        for col, row in self.crystals:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                crystal_key = (
                    "crystal_blue",
                    "crystal_green",
                    "crystal_yellow",
                    "crystal_red",
                )[(col + row) % len(self.GEM_COLORS)]
                glow = pygame.Surface((40, 40), pygame.SRCALPHA)
                pulse = 16 + ((self.steps + col * 3 + row) // 8) % 4
                pygame.draw.circle(glow, (88, 232, 255, 30 + pulse), (20, 20), pulse)
                screen.blit(glow, (x - 4, y - 4))
                bob = -1 if ((self.steps + col * 7) // 16) % 2 else 0
                self._art.draw_sprite(screen, crystal_key, x, y + bob, scale=2)
                if (self.steps + col + row) % 48 < 6:
                    pygame.draw.rect(screen, EGA["W"], (x + 10, y + 7 + bob, 5, 2))
                continue
            color = self.GEM_COLORS[(col + row) % len(self.GEM_COLORS)]
            shine = tuple(min(255, c + 70) for c in color)
            points = [
                (x + 16, y + 4),
                (x + 27, y + 16),
                (x + 16, y + 29),
                (x + 5, y + 16),
            ]
            pygame.draw.polygon(
                screen,
                (0, 0, 0),
                [(x + 16, y + 1), (x + 30, y + 16), (x + 16, y + 31), (x + 2, y + 16)],
            )
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(
                screen, shine, [(x + 16, y + 4), (x + 27, y + 16), (x + 16, y + 16)]
            )
            pygame.draw.polygon(
                screen,
                (255, 255, 255),
                [(x + 11, y + 13), (x + 16, y + 7), (x + 14, y + 15)],
            )
            pygame.draw.polygon(screen, (0, 0, 0), points, 1)

        for col, row in self.ammo_pickups:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_sprite(screen, "ammo", x, y, scale=2)
                if (self.steps // 10) % 2 == 0:
                    pygame.draw.rect(screen, EGA["Y"], (x + 16, y + 13, 9, 3))
                continue
            pygame.draw.rect(screen, (0, 0, 0), (x + 6, y + 8, 20, 14))
            pygame.draw.rect(screen, (190, 190, 205), (x + 7, y + 9, 18, 12))
            pygame.draw.rect(screen, (255, 255, 255), (x + 10, y + 11, 5, 3))
            pygame.draw.rect(screen, (255, 220, 72), (x + 13, y + 14, 9, 4))
            if self._tiny_font:
                label = self._tiny_font.render("R", True, (0, 0, 0))
                screen.blit(label, label.get_rect(center=(x + 16, y + 15)))

        for col, row in self.air_tanks:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_sprite(screen, "air_tank", x, y, scale=2)
                if (self.steps // 14) % 2 == 0:
                    pygame.draw.rect(screen, EGA["C"], (x + 20, y + 6, 3, 18))
                continue
            pygame.draw.rect(screen, (0, 0, 0), (x + 7, y + 2, 18, 28))
            pygame.draw.rect(screen, (50, 210, 240), (x + 9, y + 6, 14, 22))
            pygame.draw.rect(screen, (220, 255, 255), (x + 9, y + 6, 14, 22), 2)
            pygame.draw.rect(screen, (250, 250, 255), (x + 12, y + 9, 3, 14))
            pygame.draw.rect(screen, (160, 160, 170), (x + 12, y + 2, 8, 5))
            if self._tiny_font:
                label = self._tiny_font.render("AIR", True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=(x + 16, y + 17)))

        for col, row in self.treasures:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_text(screen, "$", x + 9, y + 8, EGA["Y"], scale=2)
                continue
            pygame.draw.circle(screen, (255, 215, 95), (x + 16, y + 16), 8)

        for (col, row), power in self.powerups.items():
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_sprite(screen, "power", x, y, scale=2)
                if (self.steps // 8) % 2 == 0:
                    pygame.draw.rect(screen, EGA["W"], (x + 5, y + 5, 22, 2))
                self._art.draw_text(
                    screen,
                    power.upper(),
                    x + 11,
                    y + 10,
                    EGA["K"],
                    scale=1,
                    shadow=False,
                )
                continue
            color = {
                self.POWER_SHOT: (255, 95, 95),
                self.GRAVITY_POWER: (150, 130, 255),
                self.FREEZE_POWER: (100, 220, 255),
            }[power]
            pygame.draw.rect(screen, (0, 0, 0), (x + 5, y + 5, 22, 22))
            pygame.draw.rect(screen, color, (x + 7, y + 7, 18, 18))
            pygame.draw.rect(screen, (255, 255, 255), (x + 10, y + 9, 5, 4))
            if self._tiny_font:
                label = self._tiny_font.render(power.upper(), True, (20, 20, 25))
                screen.blit(label, label.get_rect(center=(x + 16, y + 16)))

    def _draw_enemies(self: Any, screen, camera_x: int, camera_y: int) -> None:
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            x, y = self._world_to_screen(enemy.x, enemy.y, camera_x, camera_y)
            if self._art:
                if enemy.kind == "flyer":
                    bob = 2 if int((self.steps + enemy.x) / 10) % 2 else 0
                    sprite = (
                        "bat_enemy"
                        if (self.level_index + int(enemy.x // self.TILE_SIZE)) % 2
                        else "eye_flyer"
                    )
                    self._art.draw_sprite(screen, sprite, x - 4, y + bob, scale=2)
                    if (self.steps // 9) % 2 == 0:
                        pygame.draw.rect(screen, EGA["A"], (x + 11, y + 18 + bob, 5, 2))
                else:
                    sprite = (
                        "slug_enemy"
                        if (self.level_index + int(enemy.x // self.TILE_SIZE)) % 2
                        else "walking_rock"
                    )
                    step_bob = 1 if int((self.steps + enemy.x) / 12) % 2 else 0
                    self._art.draw_sprite(
                        screen,
                        sprite,
                        x - 4,
                        y + 2 + step_bob,
                        scale=2,
                        flip_x=enemy.vx < 0,
                    )
                continue
            if enemy.kind == "flyer":
                wing_phase = int((self.steps + int(enemy.x)) / 8) % 2
                wing_y = y + (0 if wing_phase else 4)
                pygame.draw.rect(screen, (0, 0, 0), (x - 2, y, 29, 22))
                pygame.draw.polygon(
                    screen,
                    (190, 42, 210),
                    [(x + 3, y + 12), (x + 9, wing_y + 3), (x + 15, y + 12)],
                )
                pygame.draw.polygon(
                    screen,
                    (250, 88, 255),
                    [(x + 11, y + 12), (x + 19, wing_y + 2), (x + 24, y + 12)],
                )
                pygame.draw.rect(screen, (120, 250, 88), (x + 9, y + 12, 12, 7))
                pygame.draw.rect(screen, (255, 255, 255), (x + 16, y + 14, 3, 3))
            else:
                foot_offset = 2 if int((self.steps + enemy.x) / 12) % 2 else 0
                pygame.draw.rect(screen, (0, 0, 0), (x - 2, y + 2, enemy.width + 4, 24))
                pygame.draw.rect(screen, (52, 202, 60), (x + 1, y + 9, 20, 12))
                pygame.draw.rect(screen, (112, 255, 88), (x + 4, y + 6, 14, 5))
                pygame.draw.rect(screen, (0, 108, 20), (x + 3, y + 20, 8, 4))
                pygame.draw.rect(screen, (0, 108, 20), (x + 13, y + 20 + foot_offset, 8, 4))
                pygame.draw.rect(screen, (255, 255, 255), (x + 16, y + 10, 3, 3))
                pygame.draw.rect(screen, (255, 64, 64), (x + 2, y + 14, 4, 3))

    def _draw_bullets(self: Any, screen, camera_x: int, camera_y: int) -> None:
        for bullet in self.bullets:
            x, y = self._world_to_screen(bullet.x, bullet.y, camera_x, camera_y)
            color = (255, 255, 120) if bullet.powered else (255, 210, 60)
            pygame.draw.rect(screen, (0, 0, 0), (x - 1, y - 1, 14, 6))
            pygame.draw.rect(screen, color, (x, y, 11, 4))
            pygame.draw.rect(screen, (255, 90, 40), (x - 3 if bullet.vx > 0 else x + 10, y, 3, 4))

    def _draw_player(self: Any, screen, camera_x: int, camera_y: int) -> None:
        x, y = self._world_to_screen(self.player_x, self.player_y, camera_x, camera_y)
        flash = self.invuln_timer > 0 and (self.invuln_timer // 5) % 2 == 0
        if flash:
            return

        if self._art:
            if self.invuln_timer > 0:
                sprite = "mylo_hurt"
            elif self.shoot_cooldown > self.SHOOT_COOLDOWN - 7:
                sprite = "mylo_shoot"
            elif not self.grounded:
                sprite = "mylo_jump"
            elif abs(self.vx) > 0.2:
                sprite = "mylo_walk_1" if (self.steps // 8) % 2 == 0 else "mylo_walk_2"
            else:
                sprite = "mylo_idle"

            pygame.draw.ellipse(
                screen,
                EGA["K"],
                (x - 4, y + self.PLAYER_HEIGHT - 2, 34, 7),
            )
            pygame.draw.rect(screen, EGA["K"], (x - 4, y - 4, 33, 38), 1)
            self._art.draw_sprite(
                screen,
                sprite,
                x - 1,
                y - 2,
                scale=2,
                flip_x=self.facing < 0,
            )
            pygame.draw.rect(screen, EGA["Y"], (x + 7, y - 2, 10, 2))
            if sprite != "mylo_shoot":
                gun_x = x + 16 if self.facing > 0 else x - 10
                self._art.draw_sprite(
                    screen,
                    "raygun",
                    gun_x,
                    y + 12,
                    scale=1,
                    flip_x=self.facing < 0,
                )
            if self.gravity_dir < 0:
                pygame.draw.rect(screen, EGA["C"], (x + 1, y - 5, 22, 3))
            if self.super_timer > 0:
                pygame.draw.rect(screen, EGA["Y"], (x - 1, y + 10, 4, 18))
            if self.freeze_timer > 0:
                pygame.draw.rect(screen, EGA["C"], (x + 22, y + 2, 3, 24))
            return

        pygame.draw.rect(screen, (0, 0, 0), (x - 2, y - 1, 28, 33))
        walk_frame = int(self.steps / 8) % 2 if abs(self.vx) > 0.2 else 0
        recoil = 2 if self.shoot_cooldown > self.SHOOT_COOLDOWN - 5 else 0

        # Helmet and face.
        pygame.draw.rect(screen, (255, 222, 52), (x + 5, y, 14, 7))
        pygame.draw.rect(screen, (255, 172, 40), (x + 4, y + 5, 17, 4))
        pygame.draw.rect(screen, (255, 190, 132), (x + 6, y + 9, 13, 8))
        eye_x = x + 15 if self.facing > 0 else x + 8
        pygame.draw.rect(screen, (0, 0, 0), (eye_x, y + 11, 2, 2))

        # Pink shirt, blue overalls, red boots: intentionally loud EGA.
        pygame.draw.rect(screen, (255, 76, 172), (x + 4, y + 17, 17, 6))
        pygame.draw.rect(screen, (68, 84, 255), (x + 7, y + 20, 11, 8))
        pygame.draw.rect(screen, (255, 64, 52), (x + 4, y + 28 + walk_frame, 8, 4))
        pygame.draw.rect(screen, (255, 64, 52), (x + 15, y + 29 - walk_frame, 8, 4))

        if self.gravity_dir < 0:
            pygame.draw.rect(screen, (100, 190, 255), (x + 2, y - 4, 20, 3))
        if self.super_timer > 0:
            pygame.draw.rect(screen, (255, 255, 90), (x + 2, y + 15, 3, 9))
        if self.freeze_timer > 0:
            pygame.draw.rect(screen, (80, 220, 255), (x + 20, y + 3, 3, 20))

        gun_x = x + 19 - recoil if self.facing > 0 else x - 7 + recoil
        pygame.draw.rect(screen, (225, 225, 232), (gun_x, y + 17, 11, 4))
        pygame.draw.rect(
            screen,
            (255, 236, 80),
            (gun_x + (9 if self.facing > 0 else -2), y + 18, 3, 2),
        )

    def _draw_visual_events(self: Any, screen, camera_x: int, camera_y: int) -> None:
        if not self.visual_events:
            return

        for event in self.visual_events:
            x, y = self._world_to_screen(event.x, event.y, camera_x, camera_y)
            age = event.max_ttl - event.ttl
            rise = age // 2
            if event.kind == "sparkle":
                if self._art:
                    self._art.draw_sprite(screen, "sparkle", x - 16, y - 20 - rise, scale=2)
                else:
                    pygame.draw.circle(screen, event.color, (x, y - rise), 10, 2)
                if event.text and self._art:
                    self._art.draw_text(
                        screen,
                        event.text,
                        x - 20,
                        y - 31 - rise,
                        event.color,
                        scale=1,
                    )
            elif event.kind == "poof":
                if self._art:
                    self._art.draw_sprite(screen, "poof", x - 16, y - 18 - rise, scale=2)
                else:
                    pygame.draw.circle(screen, (210, 210, 220), (x, y - rise), 12)
                if event.text and self._art:
                    self._art.draw_text(
                        screen,
                        event.text,
                        x - 14,
                        y - 34 - rise,
                        event.color,
                        scale=1,
                    )
            elif event.kind == "spark":
                radius = max(2, 10 - age // 2)
                pygame.draw.line(screen, event.color, (x - radius, y), (x + radius, y), 2)
                pygame.draw.line(screen, event.color, (x, y - radius), (x, y + radius), 2)
                pygame.draw.rect(screen, EGA["W"], (x - 1, y - 1, 3, 3))
                if event.text and self._art:
                    self._art.draw_text(
                        screen,
                        event.text,
                        x - 12,
                        y - 28 - rise,
                        event.color,
                        scale=1,
                    )
            elif event.text and self._art:
                self._art.draw_text(
                    screen,
                    event.text,
                    x - 16,
                    y - 24 - rise,
                    event.color,
                    scale=1,
                )
