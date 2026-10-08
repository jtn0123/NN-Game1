"""Active floor thorns and rideable lifts for the classic human-play profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pygame

from src.game.crystal_caves_geometry import CrystalCavesGeometryMixin

from .classic_flyers import ClassicFlyers


@dataclass
class GreenThorn:
    col: int
    row: int
    frame: int = 0
    clock: int = 0
    triggered: bool = False

    @property
    def rect(self) -> pygame.Rect:
        height = self.frame * 8
        return pygame.Rect(self.col * 32 + 12, (self.row + 1) * 32 - height, 8, height)


@dataclass
class Stalactite:
    col: int
    row: int
    y: int = field(init=False)
    clock: int = 0
    falling: bool = False
    alive: bool = True

    def __post_init__(self) -> None:
        self.y = self.row * 32

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.col * 32 + 10, self.y + 2, 12, 30)


class ClassicMechanisms(ClassicFlyers):
    def _is_on_ladder(self: Any) -> bool:
        # An elevator shaft is not an invisible ladder. Jump leaves the platform.
        return any(tile in self.ladders for tile in self._tiles_for_rect(self._player_rect()))

    def _update_elevators(self: Any) -> None:
        for index, lift in enumerate(self.elevators):
            old_pos, old_direction = lift.pos, lift.direction
            old_rect = self._elevator_solid[index]
            player = self._player_rect()
            overlaps = player.right > old_rect.left and player.left < old_rect.right
            riding = (
                overlaps
                and self.vy * self.gravity_dir >= 0
                and (
                    player.bottom == old_rect.top
                    if self.gravity_dir > 0
                    else player.top == old_rect.bottom
                )
            )
            lift.pos += (68 / 60 / 32) * lift.direction
            if lift.pos >= lift.bottom:
                lift.pos, lift.direction = float(lift.bottom), -1
            elif lift.pos <= lift.top:
                lift.pos, lift.direction = float(lift.top), 1
            self._refresh_elevator_rects()
            rect = self._elevator_solid[index]
            # Also catch a rising pad that meets a player waiting at the shaft base.
            caught = player.colliderect(rect) and not player.colliderect(old_rect)
            if not riding and not caught:
                continue
            above = self.gravity_dir > 0 if riding else player.centery < old_rect.centery
            target = rect.top - self.PLAYER_HEIGHT if above else rect.bottom
            other_lifts = self._elevator_solid[:index] + self._elevator_solid[index + 1 :]
            self._elevator_solid = other_lifts
            start = self.player_y
            self._move_axis(0.0, target - start)
            if abs(self.player_y - target) > 0.01:
                # A low ceiling stops/reverses the pad; it cannot shove Mylo into stone.
                self.player_y = start
                lift.pos, lift.direction = old_pos, -old_direction
            else:
                self.vy = 0.0
            self._refresh_elevator_rects()

    def _update_thorns(self: Any) -> None:
        player = self._player_rect()
        for thorn in self.thorns:
            top = thorn.row
            while top > 0 and not self._solid_at(thorn.col, top - 1):
                top -= 1
            detection = pygame.Rect(thorn.col * 32, top * 32, 32, (thorn.row - top + 1) * 32)
            if not player.colliderect(detection):
                thorn.frame = thorn.clock = 0
                thorn.triggered = False
                continue
            if not thorn.triggered:
                self.audio.play("thorn")
                thorn.triggered = True
            # Discrete original-style poses at 17 Hz, independent of native rendering.
            thorn.clock += 17
            if thorn.clock >= 60:
                thorn.frame = min(4, thorn.frame + 1)
                thorn.clock -= 60

    def _check_player_danger(self: Any) -> float:
        if self.game_over:
            return 0.0
        self._update_thorns()
        falling_hit = self._update_stalactites()
        egg_hit = self._update_bat_eggs()
        player = self._player_rect()
        hazard = any(player.colliderect(self._tile_rect(tile)) for tile in self.hazards)
        hazard = hazard or any(player.colliderect(thorn.rect) for thorn in self.thorns)
        hazard = hazard or falling_hit
        hazard = hazard or egg_hit
        enemy = any(e.alive and player.colliderect(e.rect) for e in self.enemies)
        if not hazard and not enemy:
            return 0.0
        return float(
            self._damage_player("both" if hazard and enemy else "enemy" if enemy else "hazard")
        )

    def _update_stalactites(self: Any) -> bool:
        """Release on approach; original-style 16px steps at 17 Hz, swept safely."""
        player = self._player_rect()
        hit = False
        for trap in self.stalactites:
            if not trap.alive:
                continue
            hit = hit or player.colliderect(trap.rect)
            if not trap.falling:
                bottom = trap.row + 1
                while bottom < self.level_rows and not self._solid_at(trap.col, bottom):
                    bottom += 1
                detection = pygame.Rect(trap.col * 32, trap.y, 32, bottom * 32 - trap.y)
                if not player.colliderect(detection):
                    continue
                trap.falling = True
                self.audio.play("stalactite")
            trap.clock += 17
            if trap.clock < 60:
                continue
            trap.clock -= 60
            # Sweep the narrow visible tip so fast motion cannot tunnel through
            # Mylo or damage him on the far side of a platform or closed gate.
            for _ in range(16):
                trap.y += 1
                if self._rect_collides_solid(trap.rect) or trap.y >= self.level_height:
                    trap.alive = False
                    break
                hit = hit or player.colliderect(trap.rect)
        return hit

    def _dynamic_hazard_tiles(self: Any) -> set[tuple[int, int]]:
        tiles = {(thorn.col, thorn.row) for thorn in getattr(self, "thorns", [])}
        for trap in getattr(self, "stalactites", []):
            if trap.alive:
                tiles.update(self._tiles_for_rect(trap.rect))
        for egg in getattr(self, "bat_eggs", []):
            tiles.update(self._tiles_for_rect(egg.rect))
        return tiles

    def _tile_code(self: Any, col: int, row: int) -> float:
        if (col, row) in self._dynamic_hazard_tiles():
            return float(self.TILE_CODES[self.SPIKE])
        return float(CrystalCavesGeometryMixin._tile_code(self, col, row))

    def _code_grid(self: Any) -> Any:
        grid = CrystalCavesGeometryMixin._code_grid(self)
        for col, row in self._dynamic_hazard_tiles():
            if 0 <= row < self.level_rows and 0 <= col < self.level_cols:
                grid[row, col] = self.TILE_CODES[self.SPIKE]
        return grid
