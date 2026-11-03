"""Clued head-bump caches for human caves, retaining existing crystal totals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from src.game.crystal_caves_entities import CaveSpec
from src.game.crystal_caves_rendering import EGA


@dataclass(frozen=True)
class SecretCache:
    block: tuple[int, int]
    crystal: tuple[int, int]


SITES = {
    "Ore Shaft": (SecretCache((10, 7), (10, 9)),),
    "Twin Vaults": (SecretCache((8, 14), (8, 17)),),
}


class ClassicSecrets:
    secrets_enabled = False

    def _setup_secrets(self: Any, level: CaveSpec) -> None:
        self.secret_caches = (
            SITES.get(level.name, ()) if getattr(self, "secrets_enabled", False) else ()
        )
        self.hidden_crystals = {cache.crystal for cache in self.secret_caches}
        for cache in self.secret_caches:
            col, row = cache.block
            if level.layout[row][col] != "#" or cache.crystal not in self.crystals:
                raise ValueError(f"invalid secret cache in {level.name}: {cache}")

    def _pickup_tiles(self: Any) -> set[tuple[int, int]]:
        # Keep hidden gems in the authoritative crystal set so the last visible
        # gem cannot unlock the exit or award the all-crystals bonus early.
        return cast(Any, super())._pickup_tiles() - self.hidden_crystals

    def _move_axis(self: Any, dx: float, dy: float) -> None:
        old_y, old_vy = self.player_y, self.vy
        cast(Any, super())._move_axis(dx, dy)
        if not (
            dy < 0
            and old_vy < 0
            and self.gravity_dir == 1
            and self.vy == 0
            and self.player_y - old_y > dy + 0.001
        ):
            return
        touched = self._tiles_for_rect(self._player_rect(self.player_x, self.player_y - 1))
        for cache in self.secret_caches:
            if cache.block in touched and cache.crystal in self.hidden_crystals:
                self.hidden_crystals.remove(cache.crystal)
                self._add_tile_event(cache.block, "sparkle", "SECRET", EGA["C"], ttl=45)
                self._mark_progress()
                self.audio.play("pickup")
