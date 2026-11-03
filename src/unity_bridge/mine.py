"""A playable main mine that connects the sixteen authored caves."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.game.base_game import validate_action
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_entities import CaveSpec

from .human_controls import HumanControlsMixin

ENTRANCES = tuple((col, row) for row in (6, 10, 14, 18) for col in (8, 15, 23, 31))


def mine_spec() -> CaveSpec:
    grid = [list("#" + "." * 38 + "#") for _ in range(24)]
    grid[0] = list("#" * 40)
    for row in (7, 11, 15, 19, 23):
        grid[row] = list("#" * 40)
    # Continuous chain shafts connect every tier, using the existing ladder rules.
    for col in (4, 35):
        for row in range(3, 23):
            grid[row][col] = "H"
    grid[6][5] = "P"
    return CaveSpec("MAIN MINE", tuple("".join(row) for row in grid), (7, 12, 23), (247, 192, 83))


MINE_SPEC = mine_spec()


class MainMine(HumanControlsMixin, CrystalCaves):
    """Shared movement and collision, without cave objectives or fabricated victories."""

    LEGACY_CLIMB_HOLD = True

    def near_entrance(self) -> int:
        player = self._player_rect()
        for level, (col, row) in enumerate(ENTRANCES):
            if (
                abs(player.centerx - (col * 32 + 16)) <= 22
                and abs(player.bottom - (row + 1) * 32) <= 8
            ):
                return level
        return -1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = validate_action(action, self.action_size, "MainMine")
        self.steps += 1
        self._update_visual_events()
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        move, jump, shoot, interact = self._decode_action(action)
        self._apply_player_input(move, jump)
        if self._is_on_ladder() and not jump and not interact:
            self.vy = 0.0  # Hold a chain until Up/Down is pressed.
        if shoot:
            if self.ammo <= 0 and self.shoot_cooldown == 0:
                self.audio.play("empty")
                self.shoot_cooldown = self.SHOOT_COOLDOWN
            else:
                self._try_shoot()
        self._move_player()
        self._update_bullets()
        self.portal_level = self.near_entrance() if interact else -1
        return self.get_state(), 0.0, False, {"realm": "mine"}
