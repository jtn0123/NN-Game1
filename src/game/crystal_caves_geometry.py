"""Geometry, collision, and tile-encoding helpers for Crystal Caves."""

from __future__ import annotations

from typing import Any, Optional, Set, Tuple

import numpy as np
import pygame

from .crystal_caves_entities import Enemy


class CrystalCavesGeometryMixin:
    def _player_rect(
        self: Any, x: Optional[float] = None, y: Optional[float] = None
    ) -> pygame.Rect:
        return pygame.Rect(
            int(self.player_x if x is None else x),
            int(self.player_y if y is None else y),
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )

    def _player_tile(self: Any) -> Tuple[int, int]:
        return (
            int((self.player_x + self.PLAYER_WIDTH / 2) // self.TILE_SIZE),
            int((self.player_y + self.PLAYER_HEIGHT / 2) // self.TILE_SIZE),
        )

    def _player_center(self: Any) -> Tuple[float, float]:
        return (
            self.player_x + self.PLAYER_WIDTH / 2,
            self.player_y + self.PLAYER_HEIGHT / 2,
        )

    def _tile_center(self: Any, tile: Tuple[int, int]) -> Tuple[float, float]:
        col, row = tile
        return (
            col * self.TILE_SIZE + self.TILE_SIZE / 2,
            row * self.TILE_SIZE + self.TILE_SIZE / 2,
        )

    def _tile_rect(self: Any, tile: Tuple[int, int]) -> pygame.Rect:
        col, row = tile
        return pygame.Rect(
            col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE
        )

    def _tiles_for_rect(self: Any, rect: pygame.Rect) -> Set[Tuple[int, int]]:
        left = rect.left // self.TILE_SIZE
        right = (rect.right - 1) // self.TILE_SIZE
        top = rect.top // self.TILE_SIZE
        bottom = (rect.bottom - 1) // self.TILE_SIZE
        return {(col, row) for row in range(top, bottom + 1) for col in range(left, right + 1)}

    def _rect_collides_solid(self: Any, rect: pygame.Rect) -> bool:
        if any(self._solid_at(col, row) for col, row in self._tiles_for_rect(rect)):
            return True
        return any(rect.colliderect(er) for er in self._elevator_solid)

    @property
    def doors_open(self: Any) -> bool:
        """True once every lever has been thrown."""

        return not (self.switches - self.used_switches)

    def _door_open(self: Any, tile: Tuple[int, int]) -> bool:
        return self.door_color.get(tile, "red") in self.open_colors

    def _solid_at(self: Any, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.level_cols or row >= self.level_rows:
            return True
        if self.grid[row][col] == self.SOLID:
            return True
        return (col, row) in self.doors and not self._door_open((col, row))

    def _is_on_surface(self: Any) -> bool:
        rect = self._player_rect()
        rect.y += self.gravity_dir
        return self._rect_collides_solid(rect)

    def _refresh_elevator_rects(self: Any) -> None:
        ts = self.TILE_SIZE
        self._elevator_solid = [
            pygame.Rect(e.col * ts, int(e.pos * ts), ts, ts) for e in self.elevators
        ]

    def _update_elevators(self: Any) -> None:
        """Advance each lift platform and carry the player up when needed."""

        if not self.elevators:
            return
        for e in self.elevators:
            e.pos += self.ELEVATOR_SPEED * e.direction
            if e.pos >= e.bottom:
                e.pos = float(e.bottom)
                e.direction = -1
            elif e.pos <= e.top:
                e.pos = float(e.top)
                e.direction = 1
        self._refresh_elevator_rects()
        prect = self._player_rect()
        for er in self._elevator_solid:
            if prect.colliderect(er) and prect.centery < er.centery:
                self.player_y = er.top - self.PLAYER_HEIGHT
                self.vy = 0.0
                prect = self._player_rect()

    def _code_grid(self: Any) -> np.ndarray:
        """Vectorized whole-level tile-code grid."""

        tc = self.TILE_CODES
        grid = np.full((self.level_rows, self.level_cols), tc[self.EMPTY], dtype=np.float32)

        for c, r in self.air_tanks:
            grid[r, c] = tc[self.AIR_TANK]
        for (c, r), power in self.powerups.items():
            grid[r, c] = tc[power]
        for c, r in self.treasures:
            grid[r, c] = tc[self.TREASURE]
        for c, r in self.ammo_pickups:
            grid[r, c] = tc[self.AMMO]
        for c, r in self.switches:
            grid[r, c] = tc[self.SWITCH]
        ec, er = self.exit_pos
        grid[er, ec] = tc[self.EXIT] if self.exit_unlocked else 0.38
        for c, r in self.crystals:
            grid[r, c] = tc[self.CRYSTAL]
        for c, r in self.hazards:
            grid[r, c] = tc[self.hazard_kinds.get((c, r), self.SPIKE)]
        grid[self._elevator_mask] = tc[self.ELEVATOR]
        grid[self._wall_mask] = tc[self.SOLID]
        for c, r in self.doors:
            if not self._door_open((c, r)):
                grid[r, c] = tc[self.DOOR]
        for enemy in self.enemies:
            if enemy.alive:
                c, r = self._tile_for_enemy(enemy)
                if 0 <= r < self.level_rows and 0 <= c < self.level_cols:
                    grid[r, c] = tc[self.FLYER if enemy.kind == "flyer" else self.CRAWLER]
        return grid

    def _fill_window(self: Any, player_col: int, player_row: int) -> np.ndarray:
        """Return the perception window centered on the player."""

        half_c = self.WINDOW_COLS // 2
        half_r = self.WINDOW_ROWS // 2
        code_grid = self._code_grid()
        window = np.full(
            (self.WINDOW_ROWS, self.WINDOW_COLS),
            self.TILE_CODES[self.SOLID],
            dtype=np.float32,
        )
        r0, r1 = player_row - half_r, player_row + half_r + 1
        c0, c1 = player_col - half_c, player_col + half_c + 1
        sr0, sr1 = max(0, r0), min(self.level_rows, r1)
        sc0, sc1 = max(0, c0), min(self.level_cols, c1)
        if sr0 < sr1 and sc0 < sc1:
            window[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = code_grid[sr0:sr1, sc0:sc1]
        return window

    def _tile_code(self: Any, col: int, row: int) -> float:
        if col < 0 or row < 0 or col >= self.level_cols or row >= self.level_rows:
            return self.TILE_CODES[self.SOLID]

        tile = (col, row)
        for enemy in self.enemies:
            if enemy.alive and self._tile_for_enemy(enemy) == tile:
                return self.TILE_CODES[self.FLYER if enemy.kind == "flyer" else self.CRAWLER]

        if self._solid_at(col, row):
            return self.TILE_CODES[self.DOOR] if tile in self.doors else self.TILE_CODES[self.SOLID]
        if self.grid[row][col] == self.ELEVATOR:
            return self.TILE_CODES[self.ELEVATOR]
        if tile in self.hazards:
            return self.TILE_CODES[self.hazard_kinds.get(tile, self.SPIKE)]
        if tile in self.crystals:
            return self.TILE_CODES[self.CRYSTAL]
        if tile == self.exit_pos:
            return self.TILE_CODES[self.EXIT] if self.exit_unlocked else 0.38
        if tile in self.switches:
            return self.TILE_CODES[self.SWITCH]
        if tile in self.ammo_pickups:
            return self.TILE_CODES[self.AMMO]
        if tile in self.treasures:
            return self.TILE_CODES[self.TREASURE]
        if tile in self.powerups:
            return self.TILE_CODES[self.powerups[tile]]
        if tile in self.air_tanks:
            return self.TILE_CODES[self.AIR_TANK]
        return self.TILE_CODES[self.EMPTY]

    def _tile_for_enemy(self: Any, enemy: Enemy) -> Tuple[int, int]:
        return (
            int((enemy.x + enemy.width / 2) // self.TILE_SIZE),
            int((enemy.y + enemy.height / 2) // self.TILE_SIZE),
        )

    @staticmethod
    def _normalize_signed(value: float, max_abs: float) -> float:
        return float(np.clip((value / max_abs + 1.0) * 0.5, 0.0, 1.0))
