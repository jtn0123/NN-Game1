"""Simulation/step logic mixin for Crystal Caves."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pygame

from .crystal_caves_art import EGA
from .crystal_caves_entities import Bullet, VisualEvent


class CrystalCavesLogicMixin:
    def _decode_action(self: Any, action: int) -> Tuple[int, bool, bool, bool]:
        move_dir = 0
        wants_jump = False
        wants_shoot = False
        wants_interact = False

        if action in (self.LEFT, self.LEFT_JUMP, self.LEFT_SHOOT):
            move_dir = -1
        elif action in (self.RIGHT, self.RIGHT_JUMP, self.RIGHT_SHOOT):
            move_dir = 1

        if action in (self.JUMP, self.LEFT_JUMP, self.RIGHT_JUMP):
            wants_jump = True
        if action in (self.SHOOT, self.LEFT_SHOOT, self.RIGHT_SHOOT):
            wants_shoot = True
        if action == self.INTERACT:
            wants_interact = True

        return move_dir, wants_jump, wants_shoot, wants_interact

    def _apply_player_input(self: Any, move_dir: int, wants_jump: bool) -> None:
        self.grounded = self._is_on_surface()
        if self.grounded:
            self.coyote_timer = 6
        else:
            self.coyote_timer = max(0, self.coyote_timer - 1)

        speed = self.MOVE_SPEED if self.grounded else self.AIR_SPEED
        if move_dir:
            self.vx = move_dir * speed
            self.facing = move_dir
        else:
            self.vx *= self.FRICTION
            if abs(self.vx) < 0.05:
                self.vx = 0.0

        if wants_jump and self.coyote_timer > 0:
            self.vy = -self.JUMP_SPEED * self.gravity_dir
            self.grounded = False
            self.coyote_timer = 0
            self.audio.play("jump")

        self.vy += self.GRAVITY * self.gravity_dir
        self.vy = float(np.clip(self.vy, -self.MAX_FALL_SPEED, self.MAX_FALL_SPEED))

    def _move_player(self: Any) -> None:
        was_airborne = not self.grounded
        falling_speed = abs(self.vy)
        self._move_axis(self.vx, 0.0)
        self._move_axis(0.0, self.vy)
        self.grounded = self._is_on_surface()
        if was_airborne and self.grounded and falling_speed > 3.0:
            self.audio.play("land")

    def _move_axis(self: Any, dx: float, dy: float) -> None:
        remaining = dx if dx != 0 else dy
        if remaining == 0:
            return

        sign = 1.0 if remaining > 0 else -1.0
        axis = "x" if dx != 0 else "y"

        while abs(remaining) > 0.001:
            step = sign * min(1.0, abs(remaining))
            next_x = self.player_x + step if axis == "x" else self.player_x
            next_y = self.player_y + step if axis == "y" else self.player_y
            rect = self._player_rect(next_x, next_y)

            if self._rect_collides_solid(rect):
                if axis == "x":
                    self.vx = 0.0
                else:
                    self.vy = 0.0
                return

            self.player_x = next_x
            self.player_y = next_y
            remaining -= step

    def _invalid_shoot_penalty(self: Any) -> float:
        if not getattr(self.config, "CRYSTAL_CAVES_INVALID_SHOOT_PENALTY", False):
            return 0.0
        self._invalid_shoot_count += 1
        self._invalid_shoot_total += self.INVALID_SHOOT_PENALTY
        return self.INVALID_SHOOT_PENALTY

    def _shot_has_plausible_target(self: Any) -> bool:
        direction = 1 if self.facing >= 0 else -1
        start_x = self.player_x + (self.PLAYER_WIDTH if direction > 0 else -8)
        start_y = self.player_y + self.PLAYER_HEIGHT * 0.45
        max_range = self.TILE_SIZE * 10
        left = start_x if direction > 0 else start_x - max_range
        corridor = pygame.Rect(int(left), int(start_y - 8), int(max_range), 16)

        for enemy in self.enemies:
            if enemy.alive and corridor.colliderect(enemy.rect):
                return True

        for tank in self.air_tanks:
            if corridor.colliderect(self._tile_rect(tank)):
                return True

        return False

    def _try_shoot(self: Any) -> float:
        if self.shoot_cooldown > 0 or self.ammo <= 0:
            return -0.03 + self._invalid_shoot_penalty()

        has_target = self._shot_has_plausible_target()
        self.ammo -= 1
        self.shoot_cooldown = self.SHOOT_COOLDOWN
        self.audio.play("shoot")
        y = self.player_y + self.PLAYER_HEIGHT * 0.45
        x = self.player_x + (self.PLAYER_WIDTH if self.facing > 0 else -8)
        self.bullets.append(
            Bullet(
                x=x,
                y=y,
                vx=self.BULLET_SPEED * self.facing,
                ttl=80,
                powered=self.super_timer > 0,
            )
        )
        self._add_visual_event(
            "spark",
            x + (12 if self.facing > 0 else -2),
            y,
            10,
            color=EGA["Y"],
        )

        if not self.grounded:
            self.vx -= 0.35 * self.facing

        reward = -0.01
        if not has_target:
            reward += self._invalid_shoot_penalty()
        return reward

    def _try_interact(self: Any) -> float:
        reward = 0.0
        player_col, player_row = self._player_tile()
        penalize_invalid = getattr(self.config, "CRYSTAL_CAVES_INVALID_INTERACT_PENALTY", False)
        for switch in self.switches:
            col, row = switch
            if abs(col - player_col) <= 1 and abs(row - player_row) <= 1:
                if switch not in self.used_switches:
                    self.used_switches.add(switch)
                    color = self.switch_color.get(switch, "red")
                    self.open_colors.add(color)  # opens only this lever's colour
                    self.score += 250
                    reward += self.SWITCH_THROW_BONUS
                    self._add_tile_event(switch, "score", "+250", EGA["G"], ttl=34, y_offset=-10)
                    for door in self.doors:
                        if self.door_color.get(door, "red") == color:
                            self._add_tile_event(door, "sparkle", "OPEN", EGA["G"], ttl=42)
                    self._mark_progress()
                    self.audio.play("switch")
                elif penalize_invalid:
                    reward += self.INVALID_INTERACT_PENALTY
                    self._invalid_interact_count += 1
                    self._invalid_interact_total += self.INVALID_INTERACT_PENALTY
                else:
                    reward += 0.05
                break
        else:
            if penalize_invalid:
                reward += self.INVALID_INTERACT_PENALTY
                self._invalid_interact_count += 1
                self._invalid_interact_total += self.INVALID_INTERACT_PENALTY
        return reward

    def _collect_pickups(self: Any) -> float:
        reward = 0.0
        touched_tiles = self._tiles_for_rect(self._player_rect())
        collected_crystal = False

        for tile in list(self.crystals.intersection(touched_tiles)):
            self.crystals.remove(tile)
            self.score += 100
            reward += 5.0
            collected_crystal = True
            self._add_tile_event(tile, "sparkle", "+100", EGA["Y"], ttl=34)
            self._mark_progress()
            self.audio.play("gem")

        if collected_crystal and getattr(self.config, "CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL", False):
            self.game_over = True
            self.won = True
            self._end_reason = "first_crystal_goal"
            reward += self.FIRST_CRYSTAL_GOAL_BONUS
            return reward

        if not self.crystals and not self.exit_unlocked:
            self.exit_unlocked = True
            self.score += 500
            reward += self.ALL_CRYSTALS_COLLECTED_BONUS
            self._add_tile_event(self.exit_pos, "sparkle", "EXIT OPEN", EGA["G"], ttl=58)
            self._mark_progress()
            self.audio.play("win")

        for tile in list(self.ammo_pickups.intersection(touched_tiles)):
            self.ammo_pickups.remove(tile)
            self.ammo += 5
            self.score += 75
            reward += 1.0
            self._add_tile_event(tile, "score", "AMMO", EGA["Y"], ttl=30)
            self._mark_progress()
            self.audio.play("pickup")

        for tile in list(self.treasures.intersection(touched_tiles)):
            self.treasures.remove(tile)
            self.score += 300
            reward += 1.5
            self._add_tile_event(tile, "sparkle", "+300", EGA["Y"], ttl=38)
            self._mark_progress()

        for tile, power in list(self.powerups.items()):
            if tile not in touched_tiles:
                continue
            del self.powerups[tile]
            self.score += 125
            reward += 1.5
            self._add_tile_event(tile, "sparkle", power.upper(), EGA["C"], ttl=42)
            self._mark_progress()
            self.audio.play("pickup")
            if power == self.POWER_SHOT:
                self.super_timer = self.MAX_POWER_TIMER
            elif power == self.GRAVITY_POWER:
                self.gravity_dir *= -1
                self.gravity_timer = 360
                self.vy = 0.0
                self.audio.play("gravity")
            elif power == self.FREEZE_POWER:
                self.freeze_timer = 300

        return reward

    def _update_bullets(self: Any) -> float:
        reward = 0.0
        updated: List[Bullet] = []
        for bullet in self.bullets:
            bullet.x += bullet.vx
            bullet.ttl -= 1
            if bullet.ttl <= 0:
                continue
            hit_tank = self._air_tank_for_rect(bullet.rect)
            if hit_tank is not None:
                self.air_tanks.remove(hit_tank)
                self.score = max(0, self.score - 100)
                reward -= 2.0
                self._add_tile_event(hit_tank, "poof", "-AIR", EGA["A"], ttl=44)
                self._damage_from_air_tank(hit_tank)
                continue
            if self._rect_collides_solid(bullet.rect):
                self._add_visual_event("spark", bullet.x, bullet.y, 12, color=EGA["Y"])
                continue
            updated.append(bullet)
        self.bullets = updated
        return reward

    def _update_enemies(self: Any) -> float:
        reward = 0.0

        for bullet in list(self.bullets):
            bullet_rect = bullet.rect
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                if bullet_rect.colliderect(enemy.rect):
                    enemy.alive = False
                    self.score += 200 if not bullet.powered else 250
                    reward += 4.0
                    self._add_visual_event(
                        "poof",
                        enemy.x + enemy.width / 2,
                        enemy.y + enemy.height / 2,
                        36,
                        "+200",
                        EGA["Y"],
                    )
                    self._mark_progress()
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break

        if self.freeze_timer > 0:
            return reward

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            if enemy.kind == "flyer":
                enemy.x += enemy.vx
                if self._rect_collides_solid(enemy.rect):
                    enemy.x -= enemy.vx
                    enemy.vx *= -1
            else:
                enemy.x += enemy.vx
                ahead_x = enemy.x + (enemy.width + 2 if enemy.vx > 0 else -2)
                foot_y = enemy.y + enemy.height + 2
                ahead_col = int(ahead_x // self.TILE_SIZE)
                foot_row = int(foot_y // self.TILE_SIZE)
                if self._rect_collides_solid(enemy.rect) or not self._solid_at(ahead_col, foot_row):
                    enemy.x -= enemy.vx
                    enemy.vx *= -1

        return reward

    def _check_player_danger(self: Any) -> float:
        reward = 0.0
        player_rect = self._player_rect()
        danger = False
        source = "hazard"

        for tile in self.hazards:
            if player_rect.colliderect(self._tile_rect(tile)):
                danger = True
                break

        if not danger:
            for enemy in self.enemies:
                if enemy.alive and player_rect.colliderect(enemy.rect):
                    danger = True
                    source = "enemy"
                    break

        if danger:
            reward += self._damage_player(source)

        return reward

    def _damage_player(self: Any, source: str = "hazard") -> float:
        if self.invuln_timer > 0:
            return 0.0

        self._last_damage_source = source
        self.health -= 1
        self.invuln_timer = self.INVULN_FRAMES
        self.shake_timer = self.SHAKE_FRAMES  # juice: kick the camera on a hit
        self.vy = -5.5 * self.gravity_dir
        self.vx = -self.facing * 2.0
        self._add_visual_event(
            "spark",
            self.player_x + self.PLAYER_WIDTH / 2,
            self.player_y + self.PLAYER_HEIGHT / 2,
            28,
            "OUCH",
            EGA["A"],
        )

        if self.health <= 0:
            self.health = 0
            self.game_over = True
            self.won = False
            self._end_reason = "killed"
            self.audio.play("lose")
            return -12.0

        self.audio.play("damage")
        return -3.0

    def _check_exit(self: Any) -> float:
        if not self.exit_unlocked:
            return 0.0

        exit_rect = self._tile_rect(self.exit_pos).inflate(-6, -2)
        if self._player_rect().colliderect(exit_rect):
            self.game_over = True
            self.won = True
            self._end_reason = "won"
            self.score += 1000 + self.health * 250 + self.ammo * 10
            self.level_index = (self.level_index + 1) % len(self.CAVES)
            self.audio.play("door")
            return 25.0

        return 0.0

    def _current_target(self: Any) -> Tuple[Optional[Tuple[str, int, int]], float]:
        player_x, player_y = self._player_center()
        candidates: List[Tuple[str, int, int]] = []

        unused_switches = self.switches - self.used_switches
        # Phase order for the switch-gates-a-crystal model: throw every lever
        # FIRST (each opens a door gating a crystal), then collect all crystals,
        # then head for the exit. Targeting the switch while crystals remain is
        # essential — otherwise the compass points at the still-gated crystal the
        # player can't yet reach, and the switch only gets thrown by accident.
        if unused_switches and self.crystals:
            candidates = [("switch", col, row) for col, row in unused_switches]
        elif self.crystals:
            candidates = [("crystal", col, row) for col, row in self.crystals]
        elif unused_switches:
            candidates = [("switch", col, row) for col, row in unused_switches]
        else:
            candidates = [("exit", self.exit_pos[0], self.exit_pos[1])]

        best_target: Optional[Tuple[str, int, int]] = None
        best_distance = float("inf")
        for target in candidates:
            _, col, row = target
            target_x, target_y = self._tile_center((col, row))
            distance = float(np.hypot(target_x - player_x, target_y - player_y))
            if distance < best_distance:
                best_target = target
                best_distance = distance

        return best_target, best_distance

    def _target_features(self: Any) -> Tuple[float, float, float, float]:
        target, distance = self._current_target()
        if target is None:
            return 0.5, 0.5, 1.0, 0.0

        kind, col, row = target
        player_x, player_y = self._player_center()
        target_x, target_y = self._tile_center((col, row))
        diagonal = max(1.0, float(np.hypot(self.level_width, self.level_height)))
        kind_code = {
            "crystal": 0.25,
            "switch": 0.5,
            "exit": 0.75,
        }.get(kind, 0.0)

        return (
            self._normalize_signed(target_x - player_x, max(1.0, self.level_width)),
            self._normalize_signed(target_y - player_y, max(1.0, self.level_height)),
            float(np.clip(distance / diagonal, 0.0, 1.0)),
            kind_code,
        )

    def _target_progress_reward(
        self: Any,
        previous_target: Optional[Tuple[str, int, int]],
        previous_distance: float,
    ) -> float:
        if self.game_over or previous_target is None:
            return 0.0

        current_target, current_distance = self._current_target()
        if current_target != previous_target or not np.isfinite(previous_distance):
            return 0.0

        tile_progress = (previous_distance - current_distance) / self.TILE_SIZE
        # Reset the stall timer on real approach regardless of which approach-signal
        # source is active, so the agent is never timed out mid-approach to the exit.
        if tile_progress > 0.03:
            self._mark_progress()

        # When the telescoping geodesic potential is on it supplies the (unfarmable)
        # approach gradient; skip the additive per-step approach reward to avoid
        # double-counting and the back-and-forth farming the additive term enables.
        if self._geodesic_active():
            return 0.0

        reward = self._target_best_approach_reward(current_target, current_distance)
        reward += float(
            np.clip(
                tile_progress * self.APPROACH_REWARD_SCALE,
                self.APPROACH_REWARD_CLIP_NEG,
                self.APPROACH_REWARD_CLIP_POS,
            )
        )
        return reward

    def _target_best_approach_reward(
        self: Any,
        target: Tuple[str, int, int],
        current_distance: float,
    ) -> float:
        """Reward only new closest approaches to the active objective.

        The per-frame distance reward above can be noisy. This monotonic bonus
        gives clearer credit for genuinely getting closer to the current crystal,
        switch, or exit, while avoiding back-and-forth farming.
        """
        if not np.isfinite(current_distance):
            return 0.0

        best = self._target_best_distances.get(target)
        if best is None:
            self._target_best_distances[target] = current_distance
            return 0.0

        improvement = best - current_distance
        if improvement <= self.TILE_SIZE * 0.10:
            return 0.0

        self._target_best_distances[target] = current_distance
        kind = target[0]
        multiplier = {"crystal": 1.5, "switch": 1.1, "exit": 1.2}.get(kind, 1.0)
        reward = (improvement / self.TILE_SIZE) * self.TARGET_BEST_APPROACH_SCALE * multiplier
        self._mark_progress()
        return float(np.clip(reward, 0.0, self.TARGET_BEST_APPROACH_CAP))

    def _anti_loop_penalty(
        self: Any,
        previous_target: Optional[Tuple[str, int, int]],
        previous_distance: float,
    ) -> float:
        """Penalize repeated no-progress tile loops during opt-in experiments."""
        if not getattr(self.config, "CRYSTAL_CAVES_ANTI_LOOP_REWARD", False):
            return 0.0
        if self.game_over or previous_target is None or not np.isfinite(previous_distance):
            return 0.0

        current_target, current_distance = self._current_target()
        if current_target != previous_target or not np.isfinite(current_distance):
            self._anti_loop_same_tile_steps = 0
            self._anti_loop_no_approach_steps = 0
            self._anti_loop_recent_tiles.clear()
            return 0.0

        tile = self._player_tile()
        if tile == self._anti_loop_tile:
            self._anti_loop_same_tile_steps += 1
        else:
            self._anti_loop_tile = tile
            self._anti_loop_same_tile_steps = 1
        self._anti_loop_recent_tiles.append(tile)

        tile_progress = (previous_distance - current_distance) / self.TILE_SIZE
        if tile_progress > 0.015:
            self._anti_loop_no_approach_steps = 0
            return 0.0
        self._anti_loop_no_approach_steps += 1

        penalty = 0.0
        if self._anti_loop_same_tile_steps >= 18:
            penalty -= 0.02
        if self._anti_loop_same_tile_steps >= 60:
            penalty -= 0.02
        if (
            len(self._anti_loop_recent_tiles) >= 45
            and len(set(self._anti_loop_recent_tiles)) <= 3
            and self._anti_loop_no_approach_steps >= 30
        ):
            penalty -= 0.025
        if self._anti_loop_no_approach_steps >= 120:
            penalty -= 0.015

        if penalty:
            self._anti_loop_total += penalty
        return penalty

    def _mark_progress(self: Any) -> None:
        self.steps_since_progress = 0

    def _add_visual_event(
        self: Any,
        kind: str,
        x: float,
        y: float,
        ttl: int = 24,
        text: str = "",
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if self.headless:
            return
        self.visual_events.append(
            VisualEvent(
                kind=kind,
                x=x,
                y=y,
                ttl=ttl,
                max_ttl=ttl,
                text=text,
                color=color,
            )
        )

    def _add_tile_event(
        self: Any,
        tile: Tuple[int, int],
        kind: str,
        text: str = "",
        color: Tuple[int, int, int] = (255, 255, 255),
        ttl: int = 24,
        y_offset: float = 0.0,
    ) -> None:
        x, y = self._tile_center(tile)
        self._add_visual_event(kind, x, y + y_offset, ttl, text, color)

    def _update_visual_events(self: Any) -> None:
        if not self.visual_events:
            return
        updated = []
        for event in self.visual_events:
            event.ttl -= 1
            if event.ttl > 0:
                updated.append(event)
        self.visual_events = updated

    def _air_tank_for_rect(self: Any, rect: pygame.Rect) -> Optional[Tuple[int, int]]:
        for tank in self.air_tanks:
            if rect.colliderect(self._tile_rect(tank).inflate(-8, -6)):
                return tank
        return None

    def _damage_from_air_tank(self: Any, tank: Tuple[int, int]) -> None:
        tank_x, tank_y = self._tile_center(tank)
        player_x, player_y = self._player_center()
        if np.hypot(tank_x - player_x, tank_y - player_y) <= self.TILE_SIZE * 2.2:
            self._damage_player("air")
