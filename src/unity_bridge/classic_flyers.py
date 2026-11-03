"""Human-profile flying threats; reference-informed rather than exact DOS timing."""

from dataclasses import dataclass
from typing import Any

import pygame


@dataclass
class BatEgg:
    identity: int
    x: float
    y: float
    vy: float = 0.5
    ttl: int = 240

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), 12, 16)


class ClassicFlyers:
    def _advance_flyer(self: Any, enemy: Any, index: int) -> None:
        if enemy.appearance == "bat_enemy":
            interval = 60 + (index * 137 + self.level_index * 73) % 540
            if self.steps > 0 and self.steps % interval == 0:
                enemy.vx *= -1
        enemy.x += enemy.vx
        if self._rect_collides_solid(enemy.rect):
            enemy.x -= enemy.vx
            enemy.vx *= -1
        if enemy.appearance == "eye_flyer":
            # A two-axis room patrol, not a sine-wave animation or a chase.
            if not hasattr(enemy, "roam_vy"):
                enemy.roam_vy = 0.75 * (1 if (index + self.level_index) % 2 else -1)
            interval = 150 + (index * 47 + self.level_index * 29) % 180
            if self.steps > 0 and self.steps % interval == 0:
                enemy.roam_vy *= -1
            enemy.y += enemy.roam_vy
            if self._rect_collides_solid(enemy.rect):
                enemy.y -= enemy.roam_vy
                enemy.roam_vy *= -1
        elif enemy.appearance == "bat_enemy":
            interval = 180 + (index * 31 + self.level_index * 17) % 120
            if self.steps > 0 and self.steps % interval == 0 and len(self.bat_eggs) < 8:
                egg = BatEgg(self._next_egg, enemy.x + 6, enemy.y + enemy.height)
                if not self._rect_collides_solid(egg.rect):
                    self.bat_eggs.append(egg)
                    self._next_egg += 1

    def _update_bat_eggs(self: Any) -> bool:
        player = self._player_rect()
        hit = False
        remaining = []
        for egg in self.bat_eggs:
            struck = next((b for b in self.bullets if b.rect.colliderect(egg.rect)), None)
            if struck is not None:
                self.bullets.remove(struck)
                self._add_visual_event("spark", egg.x + 6, egg.y + 8, 10)
                continue
            if self.freeze_timer > 0:
                hit = hit or player.colliderect(egg.rect)
                remaining.append(egg)
                continue
            egg.ttl -= 1
            egg.vy = min(4.0, egg.vy + 0.12)
            # Sweep each pixel: the egg cannot tunnel through floors or Mylo.
            landed = False
            distance = egg.vy
            while distance > 0:
                step = min(1.0, distance)
                egg.y += step
                distance -= step
                if self._rect_collides_solid(egg.rect):
                    landed = True
                    self._add_visual_event("poof", egg.x + 6, egg.y + 8, 12)
                    break
                if player.colliderect(egg.rect):
                    hit = landed = True
                    break
            if not landed and egg.ttl > 0 and egg.y < self.level_height:
                remaining.append(egg)
        self.bat_eggs = remaining
        return hit
