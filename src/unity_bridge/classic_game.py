"""Reference-informed creature rules for human play; training remains unchanged."""

from __future__ import annotations

from dataclasses import dataclass

from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_entities import CaveSpec, Enemy, VisualEvent

from .classic_mechanisms import ClassicMechanisms, GreenThorn, Stalactite


@dataclass
class BoneBurst(VisualEvent):
    """Classic creature breakup direction; never used by training observations."""

    facing: int = 1


@dataclass
class SlimePulse(VisualEvent):
    """Render-only red defeat stages retain the fatal shot's direction."""

    facing: int = 1


@dataclass
class ClassicEnemy(Enemy):
    appearance: str = "slug_enemy"
    health: int = 1
    asleep: bool = False
    wake_timer: int = 0
    charging: bool = False
    hit_until: int = 0

    @property
    def height(self) -> int:
        return 64 if self.appearance == "dinosaur_enemy" else 24


def tall_patrol_fits(game: CrystalCaves, enemy: Enemy) -> bool:
    """The complete connected floor must have room for the green creature's head."""
    layout = game.level.layout
    col = int((enemy.x + enemy.width / 2) // 32)
    foot = int((enemy.y + enemy.height) // 32)
    top = int((enemy.y + enemy.height - 64) // 32)
    if top < 0 or foot >= len(layout):
        return False

    def corridor(c: int) -> bool:
        return (
            0 <= c < len(layout[foot])
            and layout[foot][c] == "#"
            and layout[foot - 1][c] not in "#Dd"
        )

    left = right = col
    while corridor(left - 1):
        left -= 1
    while corridor(right + 1):
        right += 1
    return corridor(col) and all(
        layout[r][c] not in "#Dd" for r in range(top, foot) for c in range(left, right + 1)
    )


class ClassicCaves(ClassicMechanisms, CrystalCaves):
    """Use the original engine, with species-specific combat and patrol rules."""

    def _load_level(self, level: CaveSpec) -> None:
        super()._load_level(level)
        self.thorns = [
            GreenThorn(col, row)
            for row, line in enumerate(level.layout)
            for col, marker in enumerate(line)
            if marker == "t"
        ]
        self.stalactites = [
            Stalactite(col, row)
            for row, line in enumerate(level.layout)
            for col, marker in enumerate(line)
            if marker == "v"
        ]
        # Keep the lower approach trap first for deterministic replay/capture IDs.
        self.stalactites.sort(key=lambda trap: (-trap.row, trap.col))
        # The reference scene's floor thorn sits directly beneath a hover lift.
        if self.level_index == 2 and self.elevators:
            lift = self.elevators[0]
            self.thorns.append(GreenThorn(lift.col, lift.bottom))
        ground = flying = 0
        creatures: list[Enemy] = []
        for enemy in self.enemies:
            if enemy.kind == "flyer":
                appearance = ("bat_enemy", "eye_flyer")[(flying + self.level_index) % 2]
                flying += 1
            else:
                appearance = ("slug_enemy", "walking_rock", "dinosaur_enemy")[
                    (ground + self.level_index) % 3
                ]
                ground += 1
                if appearance == "dinosaur_enemy" and not tall_patrol_fits(self, enemy):
                    appearance = "walking_rock"
            creature = ClassicEnemy(
                enemy.x,
                enemy.y,
                68 / 60,
                enemy.kind,
                appearance=appearance,
                health=5 if appearance == "dinosaur_enemy" else 1,
                asleep=appearance == "walking_rock",
            )
            creature.y -= creature.height - enemy.height
            creatures.append(creature)
        self.enemies = creatures

    def _sees_player(self, enemy: ClassicEnemy, forward_only: bool = False) -> bool:
        player = self._player_rect()
        if player.bottom <= enemy.rect.top or player.top >= enemy.rect.bottom:
            return False
        dx = player.centerx - enemy.rect.centerx
        if forward_only and dx * enemy.vx < 0:
            return False
        # Walls and closed gates block awareness at the player's mid-body height.
        row = player.centery // self.TILE_SIZE
        start, end = sorted((player.centerx // 32, enemy.rect.centerx // 32))
        return all(not self._solid_at(col, row) for col in range(start + 1, end))

    def _try_shoot(self) -> float:
        if self.ammo <= 0 and self.shoot_cooldown == 0:
            self.shoot_cooldown = self.SHOOT_COOLDOWN
            self.audio.play("empty")
        return super()._try_shoot()

    def _update_enemies(self) -> float:
        reward = 0.0
        for bullet in list(self.bullets):
            for enemy in self.enemies:
                if (
                    not isinstance(enemy, ClassicEnemy)
                    or not enemy.alive
                    or not bullet.rect.colliderect(enemy.rect)
                ):
                    continue
                self.bullets.remove(bullet)
                if enemy.appearance == "walking_rock" and not bullet.powered:
                    self._add_visual_event("spark", bullet.x, bullet.y, 10)
                    break
                enemy.health = 0 if bullet.powered else enemy.health - 1
                if enemy.health > 0:
                    # Presentation follows the existing clock without changing
                    # health, movement, observations or freeze behavior.
                    enemy.hit_until = self.steps + 30
                    enemy.charging = True
                    enemy.vx = abs(enemy.vx) * (1 if self.player_x > enemy.x else -1)
                    self.audio.play("enemy_hurt")
                    self._add_visual_event("spark", bullet.x, bullet.y, 10)
                else:
                    enemy.alive = False
                    points = 250 if bullet.powered else 200
                    self.score += points
                    reward += 4.0
                    if enemy.appearance == "dinosaur_enemy":
                        if not self.headless:
                            self.visual_events.append(
                                BoneBurst(
                                    kind="bones",
                                    x=enemy.x + enemy.width / 2,
                                    y=enemy.y + enemy.height / 2,
                                    ttl=72,
                                    max_ttl=72,
                                    text=f"+{points}",
                                    facing=1 if bullet.vx >= 0 else -1,
                                )
                            )
                    elif enemy.appearance == "eye_flyer":
                        if not self.headless:
                            self.visual_events.append(
                                SlimePulse(
                                    kind="slime_pulse",
                                    x=enemy.x + enemy.width / 2,
                                    y=enemy.y + enemy.height / 2,
                                    ttl=36,
                                    max_ttl=36,
                                    text=f"+{points}",
                                    facing=1 if bullet.vx >= 0 else -1,
                                )
                            )
                    else:
                        self._add_visual_event(
                            "poof",
                            enemy.x + enemy.width / 2,
                            enemy.y + enemy.height / 2,
                            36,
                            f"+{points}",
                        )
                    self._mark_progress()
                break

        if self.freeze_timer > 0:
            return reward
        for index, enemy in enumerate(self.enemies):
            if not isinstance(enemy, ClassicEnemy) or not enemy.alive:
                continue
            if enemy.asleep:
                if self._sees_player(enemy):
                    enemy.asleep = False
                    enemy.wake_timer = 14
                    enemy.vx = abs(enemy.vx) * (1 if self.player_x > enemy.x else -1)
                continue
            if enemy.wake_timer:
                enemy.wake_timer -= 1
                continue
            if enemy.appearance == "dinosaur_enemy":
                enemy.charging = enemy.charging or self._sees_player(enemy, True)
                enemy.vx = (136 if enemy.charging else 68) / 60 * (1 if enemy.vx > 0 else -1)
            if enemy.kind == "flyer":
                # Irregular horizontal bat reversals, without a fabricated sine-wave swoop.
                if enemy.appearance == "bat_enemy":
                    interval = 60 + (index * 137 + self.level_index * 73) % 540
                    if self.steps > 0 and self.steps % interval == 0:
                        enemy.vx *= -1
                enemy.x += enemy.vx
                if self._rect_collides_solid(enemy.rect):
                    enemy.x -= enemy.vx
                    enemy.vx *= -1
            else:
                enemy.x += enemy.vx
                ahead = enemy.x + (enemy.width + 2 if enemy.vx > 0 else -2)
                foot = enemy.y + enemy.height + 2
                if self._rect_collides_solid(enemy.rect) or not self._solid_at(
                    int(ahead // 32), int(foot // 32)
                ):
                    enemy.x -= enemy.vx
                    enemy.vx *= -1
        return reward
