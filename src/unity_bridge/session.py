"""Authoritative cave sessions. Unity never implements game physics."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import numpy as np

from config import Config
from src.app.demo_recorder import HumanDemoRecorder
from src.game.crystal_caves import CrystalCaves

from .classic_game import ClassicCaves, tall_patrol_fits
from .mine import ENTRANCES, MINE_SPEC, MainMine
from .playfeel import apply_classic_controls
from .visual_doors import door_render_height
from .visual_equipment import air_render_height

MAX_BATCH = 8


def integer(value: Any, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < maximum:
        raise ValueError(f"{label} must be an integer from 0 to {maximum - 1}")
    return value


class EventAudio:
    """Capture the engine's actual sound events without opening an audio device."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def play(self, name: str) -> None:
        self.events.append(name)

    def start_music(self) -> None:
        pass

    def stop_music(self) -> None:
        pass


class CaveSession:
    def __init__(
        self,
        level: int = 0,
        seed: int = 0,
        config: Optional[Config] = None,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        policy_name: str = "",
        record_dir: Optional[str] = None,
        classic_controls: bool = True,
    ) -> None:
        self.config = config or Config(GAME_NAME="crystal_caves", CRYSTAL_CAVES_IMPORTED=True)
        game_type = ClassicCaves if classic_controls else CrystalCaves
        self.game = game_type(self.config, headless=True)
        self.cave_game = self.game
        self.mine_game: Optional[MainMine] = None
        self.realm = "cave"
        self.cleared: set[int] = set()
        self.classic_controls = classic_controls
        if classic_controls:
            apply_classic_controls(self.game)
        # Display initialization stays headless; enable only render-only events
        # afterwards. No display, mixer or replacement physics is used.
        self.game.headless = False
        self.game.use_eval_levels(len(self.game.CAVES))
        self.audio = EventAudio()
        self.game.audio = self.audio  # type: ignore[assignment]
        self.seed = seed
        self.policy = policy
        self.policy_name = policy_name
        self.mode = "human"
        self.episode = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_action = 0
        self.q_values: list[float] = []
        self._muzzle_effects: set[int] = set()
        self.recorder = HumanDemoRecorder(record_dir) if record_dir else None
        self.recording_eligible = True
        self.reset(level)

    def reset(self, level: int) -> None:
        level = integer(level, len(self.cave_game.CAVES), "level")
        self.game = self.cave_game
        self.realm = "cave"
        self.game._eval_cursor = level
        self.game.seed(self.seed)
        self.game.reset()
        self.enemy_sprites = self._enemy_sprites()
        self.audio.events.clear()
        self._muzzle_effects.clear()
        self.episode += 1
        self.total_reward = self.last_reward = 0.0
        self.last_action = 0
        self.q_values = []
        self.recording_eligible = self.mode == "human"

    def _enemy_sprites(self) -> list[str]:
        """Assign art once per spawn, independently of movement and observations."""
        sprites = []
        ground = flying = 0
        for enemy in self.game.enemies:
            if hasattr(enemy, "appearance"):
                sprites.append(enemy.appearance)
                continue
            if enemy.kind == "flyer":
                sprite = ("bat_enemy", "eye_flyer")[(flying + self.game.level_index) % 2]
                flying += 1
            else:
                sprite = ("slug_enemy", "walking_rock", "dinosaur_enemy")[
                    (ground + self.game.level_index) % 3
                ]
                ground += 1
                if sprite == "dinosaur_enemy" and not self._tall_patrol_fits(enemy):
                    sprite = "walking_rock"
            sprites.append(sprite)
        return sprites

    def _tall_patrol_fits(self, enemy: Any) -> bool:
        return tall_patrol_fits(self.game, enemy)

    def open_mine(self, cleared: Any) -> None:
        if not isinstance(cleared, list):
            raise ValueError("cleared must be a list of cave numbers")
        completed = {integer(level, len(self.cave_game.CAVES), "cleared cave") for level in cleared}
        if self.mine_game is None:
            mine = MainMine(self.config, headless=True)
            mine.CAVES = (MINE_SPEC,)
            apply_classic_controls(mine)
            mine.use_eval_levels(1)
            mine.headless = False
            mine.audio = self.audio  # type: ignore[assignment]
            self.mine_game = mine
        self.cleared = completed
        self.game = self.mine_game
        position = (
            (self.game.player_x, self.game.player_y, self.game.facing)
            if hasattr(self.game, "portal_level")
            else None
        )
        self.game._eval_cursor = 0
        self.game.reset()
        if position is not None:
            self.game.player_x, self.game.player_y, self.game.facing = position
        self.game.portal_level = -1
        self.realm = "mine"
        self.mode = "human"
        self.recording_eligible = False
        self.enemy_sprites = []
        self.audio.events.clear()
        self._muzzle_effects.clear()
        self.episode += 1
        self.total_reward = self.last_reward = 0.0
        self.last_action = 0
        self.q_values = []

    def handle(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        operation = request.get("op", "snapshot")
        if operation == "snapshot":
            self.audio.events.clear()
        elif operation == "mine":
            self.open_mine(request.get("cleared", []))
        elif operation == "reset":
            self.reset(request.get("level", self.game.level_index))
            if self.classic_controls:
                self.audio.events.append("enter")
        elif operation == "mode":
            mode = request.get("mode")
            if mode not in ("human", "ai"):
                raise ValueError("mode must be human or ai")
            if mode == "ai" and self.realm == "mine":
                raise ValueError("AI observation is available inside caves")
            if mode == "ai" and self.policy is None:
                raise ValueError("AI requires an explicitly loaded compatible checkpoint")
            self.mode = mode
            if mode == "ai":
                self.recording_eligible = False
            self.audio.events.clear()
        elif operation == "step":
            actions = request.get("actions")
            if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_BATCH:
                raise ValueError(f"actions must contain 1 to {MAX_BATCH} actions")
            # Validate the entire batch before any mutation, including in AI mode.
            actions = [integer(action, self.game.action_size, "action") for action in actions]
            self.audio.events.clear()
            for action in actions:
                if self.game.game_over or (
                    self.realm == "mine" and cast(MainMine, self.game).portal_level >= 0
                ):
                    break
                if self.mode == "ai" and self.policy is not None:
                    q = np.asarray(self.policy(self.game.get_state()), dtype=np.float32)
                    if q.shape != (self.game.action_size,) or not np.isfinite(q).all():
                        raise ValueError("checkpoint produced invalid action values")
                    self.q_values = q.tolist()
                    action = int(q.argmax())
                before_ammo = len(self.game.ammo_pickups)
                before_treasure = len(self.game.treasures)
                before_enemies = sum(enemy.alive for enemy in self.game.enemies)
                before_power = dict(self.game.powerups)
                before_health = self.game.health
                powered = self.game.super_timer > 0
                event_start = len(self.audio.events)
                # Hold the previous objects through this frame so expired event
                # IDs cannot be reused by a new impact or muzzle effect.
                previous_effects = tuple(self.game.visual_events)
                _, reward, done, info = self.game.step(action)
                if "shoot" in self.audio.events[event_start:]:
                    # _try_shoot emits its ten-step spark before bullet/target
                    # impacts. Hide only that newborn cue from presentation;
                    # the authoritative event and all impacts stay intact.
                    muzzle = next(
                        (
                            event
                            for event in self.game.visual_events
                            if event.kind == "spark"
                            and event.max_ttl == 10
                            and not any(event is previous for previous in previous_effects)
                        ),
                        None,
                    )
                    if muzzle is not None:
                        self._muzzle_effects.add(id(muzzle))
                self._muzzle_effects.intersection_update(
                    id(event) for event in self.game.visual_events
                )
                if self.classic_controls:
                    self._classic_sound_events(
                        event_start,
                        before_ammo,
                        before_treasure,
                        before_enemies,
                        before_power,
                        before_health,
                        powered,
                    )
                self.last_action = action
                self.last_reward = float(reward)
                self.total_reward += float(reward)
                if self.recorder and self.recording_eligible:
                    self.recorder.after_step(self.game, action, done, info)
        else:
            raise ValueError(f"unknown operation: {operation}")
        return self.snapshot()

    def _classic_sound_events(
        self,
        start: int,
        ammo: int,
        treasure: int,
        enemies: int,
        powerups: dict[tuple[int, int], str],
        health: int,
        powered: bool,
    ) -> None:
        """Voice actual engine events, without changing its state or rewards."""
        pickups = ["ammo"] * (ammo - len(self.game.ammo_pickups))
        pickups += [
            "freeze" if kind == self.game.FREEZE_POWER else "pickup"
            for tile, kind in powerups.items()
            if tile not in self.game.powerups
        ]
        events = []
        for name in self.audio.events[start:]:
            if name == "pickup" and pickups:
                name = pickups.pop(0)
            if name == "shoot" and powered:
                name = "power_shoot"
            if name == "win" and not self.game.won:
                continue  # Last crystal keeps its pickup cue; exit owns the fanfare.
            if name == "door" and self.game.won:
                name = "win"
            events.append(name)
        if len(self.game.treasures) < treasure:
            events.append("treasure")
        if sum(enemy.alive for enemy in self.game.enemies) < enemies and self.game.health == health:
            events.append("enemy_defeat")
        self.audio.events[start:] = events

    def terrain_layout(self) -> list[str]:
        """Project static hazards into the render grid without mutating physics."""
        layout = [row.copy() for row in self.game.grid]
        for (col, row), kind in self.game.hazard_kinds.items():
            layout[row][col] = kind
        return ["".join(row) for row in layout]

    def snapshot(self) -> dict[str, Any]:
        game = self.game
        entities: list[dict[str, Any]] = []

        def entity(
            identity: str,
            sprite: str,
            x: float,
            y: float,
            width: float = 32,
            height: float = 32,
            glow: str = "",
            flip: bool = False,
        ) -> None:
            entities.append(
                {
                    "id": identity,
                    "sprite": sprite,
                    "x": float(x),
                    "y": float(y),
                    "w": width,
                    "h": height,
                    "glow": glow,
                    "flip": flip,
                }
            )

        for kind, positions, sprite, glow in (
            ("crystal", game.crystals, "crystal_blue", "cyan"),
            ("ammo", game.ammo_pickups, "ammo", "amber"),
            ("treasure", game.treasures, "treasure", "amber"),
        ):
            for col, row in sorted(positions):
                entity(f"{kind}_{col}_{row}", sprite, col * 32, row * 32, glow=glow)
        for col, row in sorted(game.air_tanks):
            height = air_render_height(game.level.layout, col, row)
            entity(
                f"air_{col}_{row}",
                "air_tank_tall" if height == 64 else "air_tank",
                col * 32,
                (row + 1) * 32 - height,
                height=height,
                glow="cyan",
            )
        for (col, row), kind in sorted(game.powerups.items()):
            sprite = {"p": "power_shot", "g": "gravity", "z": "freeze"}[kind]
            entity(f"power_{col}_{row}", sprite, col * 32, row * 32, glow="violet")
        for col, row in sorted(game.doors):
            if not game._door_open((col, row)):
                color = game.door_color.get((col, row), "red")
                height = door_render_height(game.level.layout, col, row)
                entity(
                    f"door_{col}_{row}",
                    f"door_{color}" + ("_tall" if height == 64 else ""),
                    col * 32,
                    (row + 1) * 32 - height,
                    height=height,
                )
        for col, row in sorted(game.switches):
            used = (col, row) in game.used_switches
            color = game.switch_color.get((col, row), "red")
            entity(
                f"switch_{col}_{row}",
                f"switch_{color}_{'on' if used else 'off'}",
                col * 32,
                row * 32,
            )
        col, row = game.exit_pos
        height = door_render_height(game.level.layout, col, row)
        entity(
            "exit",
            ("exit_open" if game.exit_unlocked else "exit_locked")
            + ("_tall" if height == 64 else ""),
            col * 32,
            (row + 1) * 32 - height,
            height=height,
            glow="cyan" if game.exit_unlocked else "violet",
        )
        for index, enemy in enumerate(game.enemies):
            if enemy.alive:
                sprite = self.enemy_sprites[index]
                entity(
                    f"enemy_{index}",
                    sprite,
                    enemy.x,
                    enemy.y,
                    enemy.width,
                    enemy.height,
                    flip=enemy.vx < 0,
                )
                entities[-1]["asleep"] = getattr(enemy, "asleep", False)
                entities[-1]["hit"] = getattr(enemy, "hit_until", 0) > game.steps
        for index, elevator in enumerate(game.elevators):
            entity(f"lift_{index}", "elevator", elevator.col * 32, elevator.pos * 32)
            entities[-1]["frame"] = game.steps // 4 % 4
        for index, thorn in enumerate(getattr(game, "thorns", [])):
            entity(f"thorn_{index}", f"green_thorn_{thorn.frame}", thorn.col * 32, thorn.row * 32)
        for index, trap in enumerate(getattr(game, "stalactites", [])):
            if trap.alive:
                entity(f"stalactite_{index}", "stalactite", trap.col * 32, trap.y)
        for index, bullet in enumerate(game.bullets):
            entity(
                f"bullet_{index}",
                "bullet",
                bullet.x,
                bullet.y,
                10,
                4,
                glow="amber",
                flip=bullet.vx < 0,
            )
        climbing = game._is_on_ladder() and abs(game.vy) > 0.1
        if game.shoot_cooldown > game.SHOOT_COOLDOWN - 7:
            sprite = "mylo_shoot_air" if not game.grounded and not climbing else "mylo_shoot"
        elif not game.grounded:
            sprite = "mylo_jump"
        elif abs(game.vx) > 0.2:
            sprite = "mylo_walk_1" if (game.steps // 8) % 2 == 0 else "mylo_walk_2"
        else:
            sprite = "mylo_idle"
        player = {
            "x": game.player_x,
            "y": game.player_y,
            "vx": game.vx,
            "vy": game.vy,
            "sprite": sprite,
            "facing": game.facing,
            "grounded": game.grounded,
            "gravity_dir": game.gravity_dir,
            "invulnerable": game.invuln_timer > 0,
            "invulnerability_left": game.invuln_timer,
            "invulnerability_frames": game.INVULN_FRAMES,
            "climbing": climbing,
        }
        entity("player", sprite, game.player_x - 1, game.player_y - 2, 24, 32, flip=game.facing < 0)
        if self.realm == "mine":
            entities = [
                item
                for item in entities
                if item["id"] == "player" or item["id"].startswith("bullet_")
            ]
            for index, (col, row) in enumerate(ENTRANCES):
                entity(
                    f"entrance_{index}",
                    "mine_door_cleared" if index in self.cleared else "mine_door",
                    col * 32,
                    row * 32,
                )
            for index, (col, row) in enumerate(
                ((11, 6), (27, 6), (11, 10), (27, 10), (11, 14), (27, 14), (11, 18), (27, 18))
            ):
                entity(f"torch_{index}", "mine_torch_0", col * 32, row * 32)
        return {
            "protocol": 1,
            "realm": self.realm,
            "cleared_caves": len(self.cleared),
            "near_entrance": cast(MainMine, game).near_entrance() if self.realm == "mine" else -1,
            "portal_level": cast(MainMine, game).portal_level if self.realm == "mine" else -1,
            "movement_profile": "classic" if self.classic_controls else "training",
            "training_limits": not self.classic_controls,
            "episode": self.episode,
            "level": game.level_index if self.realm == "cave" else -1,
            "level_name": game.level.name,
            "levels": [cave.name for cave in self.cave_game.CAVES],
            "layout": self.terrain_layout(),
            "cols": game.level_cols,
            "rows": game.level_rows,
            "player": player,
            "entities": entities,
            "health": game.health,
            "ammo": game.ammo,
            "score": game.score,
            "crystals": len(game.crystals),
            "initial_crystals": game.initial_crystals,
            "exit_unlocked": game.exit_unlocked,
            "steps": game.steps,
            "freeze_timer": game.freeze_timer,
            "max_steps": game.MAX_STEPS,
            "stall_steps": game.steps_since_progress,
            "stall_limit": game.MAX_STEPS_WITHOUT_PROGRESS,
            "done": game.game_over,
            "won": game.won,
            "end_reason": game._end_reason,
            "mode": self.mode,
            "ai_available": self.policy is not None and self.realm == "cave",
            "policy_name": self.policy_name,
            "action": self.last_action,
            "action_labels": game.ACTION_LABELS,
            "q_values": self.q_values,
            "state_size": game.state_size,
            "last_reward": self.last_reward,
            "total_reward": self.total_reward,
            "sounds": list(self.audio.events),
            "recording": self.recorder is not None and self.recording_eligible,
            "human_only": self.recording_eligible,
            "demos_saved": len(self.recorder.saved) if self.recorder else 0,
            "effects": [
                {
                    "id": f"{event.kind}_{event.x}_{event.y}_{game.steps - event.max_ttl + event.ttl}",
                    "kind": event.kind,
                    "x": event.x,
                    "y": event.y,
                    "ttl": event.ttl,
                    "max_ttl": event.max_ttl,
                    "text": event.text,
                    "facing": getattr(event, "facing", 1),
                }
                for event in game.visual_events
                if id(event) not in self._muzzle_effects
            ],
        }
