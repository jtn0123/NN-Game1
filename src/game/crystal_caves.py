"""
Crystal Caves-style game implementation (core game class).

Clean-room DOS-era puzzle platformer: collect every crystal, open doors with
switches, avoid hazards, shoot enemies with limited ammo, and escape. Designed
for human play and DQN training. Rendering, dressing, and step-simulation logic
live in sibling mixin modules to keep each file focused and under budget.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame

from config import Config

from .base_game import BaseGame, validate_action
from .crystal_caves_art import CrystalCavesArt
from .crystal_caves_audio import CrystalCavesAudio
from .crystal_caves_dressing import CrystalCavesDressingMixin
from .crystal_caves_entities import CAVE_DRESSING as _CAVE_DRESSING
from .crystal_caves_entities import CAVES as _CAVES
from .crystal_caves_entities import (
    Bullet,
    CaveSpec,
    DressingPiece,
    Enemy,
    VisualEvent,
)
from .crystal_caves_logic import CrystalCavesLogicMixin
from .crystal_caves_rendering import CrystalCavesRenderingMixin


class CrystalCaves(
    CrystalCavesRenderingMixin,
    CrystalCavesDressingMixin,
    CrystalCavesLogicMixin,
    BaseGame,
):
    """
    Puzzle platform game tailored for DQN training.

    Actions:
        0 = IDLE
        1 = LEFT
        2 = RIGHT
        3 = JUMP
        4 = LEFT_JUMP
        5 = RIGHT_JUMP
        6 = SHOOT
        7 = LEFT_SHOOT
        8 = RIGHT_SHOOT
        9 = INTERACT

    State representation:
        - 11x9 local tile window around the player (99 features)
        - 20 normalized metadata features
    """

    IDLE = 0

    LEFT = 1

    RIGHT = 2

    JUMP = 3

    LEFT_JUMP = 4

    RIGHT_JUMP = 5

    SHOOT = 6

    LEFT_SHOOT = 7

    RIGHT_SHOOT = 8

    INTERACT = 9

    ACTION_LABELS = [
        "IDLE",
        "LEFT",
        "RIGHT",
        "JUMP",
        "LEFT_JUMP",
        "RIGHT_JUMP",
        "SHOOT",
        "LEFT_SHOOT",
        "RIGHT_SHOOT",
        "INTERACT",
    ]

    TILE_SIZE = 32

    HUD_HEIGHT = 38

    PLAYER_WIDTH = 22

    PLAYER_HEIGHT = 30

    WINDOW_COLS = 11

    WINDOW_ROWS = 9

    METADATA_SIZE = 20

    MAX_HEALTH = 3

    MAX_AMMO_FOR_STATE = 20

    MAX_STEPS = 3000

    MAX_STEPS_WITHOUT_PROGRESS = 720

    MAX_POWER_TIMER = 420

    # Completion-progress potential (0..1) for reward shaping + info["progress"].
    # Monotonic components so the potential-based reward is always >= 0 (it never
    # penalises exploration). Weights sum to 1.0.
    PROGRESS_W_CRYSTAL = 0.50  # fraction of crystals collected
    PROGRESS_W_SWITCH = 0.15  # every required switch thrown
    PROGRESS_W_DEPTH = 0.15  # deepest row reached (how far into the cave)
    PROGRESS_W_WIN = 0.20  # reached the exit
    PROGRESS_REWARD_SCALE = 6.0  # total shaping reward earned across a full clear

    # TS-01: discrete one-time bonus the step the gating switch is thrown
    # (closed->open). The switch is the causal crux of a clear yet was rewarded
    # less than a single crystal (+3 vs +5); reward it sharply on top of the
    # continuous progress shaping.
    SWITCH_THROW_BONUS = 8.0

    MOVE_SPEED = 4.2

    AIR_SPEED = 3.3

    JUMP_SPEED = 10.5

    GRAVITY = 0.52

    MAX_FALL_SPEED = 10.0

    FRICTION = 0.82

    BULLET_SPEED = 9.0

    SHOOT_COOLDOWN = 14

    INVULN_FRAMES = 70

    EMPTY = "."

    SOLID = "#"

    CRYSTAL = "*"

    EXIT = "E"

    DOOR = "D"

    SWITCH = "s"

    AMMO = "A"

    TREASURE = "$"

    POWER_SHOT = "p"

    GRAVITY_POWER = "g"

    FREEZE_POWER = "z"

    AIR_TANK = "O"

    CRAWLER = "M"

    FLYER = "F"

    SPIKE = "^"

    ACID = "~"

    PLAYER = "P"

    CAVES: Tuple[CaveSpec, ...] = _CAVES

    CAVE_DRESSING: Dict[int, Tuple[DressingPiece, ...]] = _CAVE_DRESSING

    TILE_CODES: Dict[str, float] = {
        EMPTY: 0.0,
        SOLID: 0.14,
        CRYSTAL: 0.25,
        TREASURE: 0.32,
        EXIT: 0.42,
        DOOR: 0.58,
        SWITCH: 0.68,
        AMMO: 0.76,
        POWER_SHOT: 0.84,
        GRAVITY_POWER: 0.88,
        FREEZE_POWER: 0.92,
        AIR_TANK: 0.72,
        SPIKE: 0.96,
        ACID: 0.98,
        CRAWLER: 0.62,
        FLYER: 0.66,
        PLAYER: 1.0,
    }

    GEM_COLORS: Tuple[Tuple[int, int, int], ...] = (
        (82, 170, 255),
        (50, 230, 95),
        (255, 225, 80),
        (255, 70, 70),
    )

    def __init__(self, config: Optional[Config] = None, headless: bool = False):
        self.config = config or Config()
        self.headless = headless
        self.width = self.config.SCREEN_WIDTH
        self.height = self.config.SCREEN_HEIGHT

        self.level_index = 0
        # Procedural mode: replace the three authored caves with freshly generated
        # ones (one per theme, in palette order) so level_index keeps theming them
        # correctly. Authored dressing is cleared since generated caves have none.
        if getattr(self.config, "CRYSTAL_CAVES_PROCEDURAL", False):
            from .crystal_caves_gen import FAMILY_NAMES, THEME_NAMES, generate_cave

            base = getattr(self.config, "CRYSTAL_CAVES_SEED", 0)
            wanted = [
                f.strip()
                for f in getattr(self.config, "CRYSTAL_CAVES_FAMILIES", "").split(",")
                if f.strip() in FAMILY_NAMES
            ]
            families = wanted or list(FAMILY_NAMES)
            difficulty = getattr(self.config, "CRYSTAL_CAVES_DIFFICULTY", "normal")
            # Always four caves (themes cycle so the renderer palette stays
            # consistent); the families are drawn from the requested set, which a
            # curriculum widens stage by stage.
            self.CAVES = tuple(
                generate_cave(
                    base * 10 + i,
                    THEME_NAMES[i % len(THEME_NAMES)],
                    families[i % len(families)],
                    difficulty=difficulty,
                )
                for i in range(len(FAMILY_NAMES))
            )
            self.CAVE_DRESSING = {i: () for i in range(len(FAMILY_NAMES))}
        self.level: CaveSpec = self.CAVES[0]
        self.grid: List[List[str]] = []
        self.level_cols = 0
        self.level_rows = 0
        self.level_width = 0
        self.level_height = 0

        self.player_x = 0.0
        self.player_y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.facing = 1
        self.grounded = False
        self.coyote_timer = 0

        self.health = self.MAX_HEALTH
        self.ammo = 5
        self.score = 0
        self.game_over = False
        self.won = False
        self.steps = 0
        self.steps_since_progress = 0
        self.invuln_timer = 0
        self.shoot_cooldown = 0
        self.super_timer = 0
        self.freeze_timer = 0
        self.gravity_timer = 0
        self.gravity_dir = 1
        self.doors_open = False
        self.exit_unlocked = False
        self.show_controls = False
        self._end_reason = "running"
        self._max_depth_row = 0
        self._progress = 0.0

        self.crystals: Set[Tuple[int, int]] = set()
        self.initial_crystals = 0
        self.exit_pos = (0, 0)
        self.doors: Set[Tuple[int, int]] = set()
        self.switches: Set[Tuple[int, int]] = set()
        self.used_switches: Set[Tuple[int, int]] = set()
        self.hazards: Set[Tuple[int, int]] = set()
        self.hazard_kinds: Dict[Tuple[int, int], str] = {}
        self.ammo_pickups: Set[Tuple[int, int]] = set()
        self.treasures: Set[Tuple[int, int]] = set()
        self.powerups: Dict[Tuple[int, int], str] = {}
        self.air_tanks: Set[Tuple[int, int]] = set()
        self.enemies: List[Enemy] = []
        self.bullets: List[Bullet] = []
        self.visual_events: List[VisualEvent] = []

        self._state_array: np.ndarray = np.zeros(self.state_size, dtype=np.float32)
        self._font: Optional[pygame.font.Font]
        self._small_font: Optional[pygame.font.Font]
        self._tiny_font: Optional[pygame.font.Font]
        self._art: Optional[CrystalCavesArt]

        if not self.headless:
            pygame.font.init()
            self._font = pygame.font.Font(None, 34)
            self._small_font = pygame.font.Font(None, 22)
            self._tiny_font = pygame.font.Font(None, 18)
            self._art = CrystalCavesArt()
        else:
            self._font = None
            self._small_font = None
            self._tiny_font = None
            self._art = None

        # Procedural SFX layer; self-disables in headless/training/CI contexts.
        self.audio = CrystalCavesAudio(enabled=not self.headless)
        self.audio.start_music()

        self.reset()

    @property
    def state_size(self) -> int:
        """State vector dimension."""
        return self.WINDOW_COLS * self.WINDOW_ROWS + self.METADATA_SIZE

    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return len(self.ACTION_LABELS)

    def reset(self) -> np.ndarray:
        """Reset to the current cave and return the initial state."""
        self.level = self.CAVES[self.level_index % len(self.CAVES)]
        self._load_level(self.level)

        self.vx = 0.0
        self.vy = 0.0
        self.facing = 1
        self.grounded = False
        self.coyote_timer = 0
        self.health = self.MAX_HEALTH
        self.ammo = 5
        self.score = 0
        self.game_over = False
        self.won = False
        self.steps = 0
        self.steps_since_progress = 0
        self.invuln_timer = 0
        self.shoot_cooldown = 0
        self.super_timer = 0
        self.freeze_timer = 0
        self.gravity_timer = 0
        self.gravity_dir = 1
        self.doors_open = False
        self.exit_unlocked = False
        self.bullets.clear()
        self.visual_events.clear()

        # Completion-progress tracker (deepest row reached + last potential).
        self._max_depth_row = self._player_tile()[1]
        self._progress = self._progress_potential()[0]
        self._end_reason = "running"

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance the cave simulation by one frame."""
        action = validate_action(action, self.action_size, "CrystalCaves")

        if self.game_over:
            return self.get_state(), 0.0, True, self._info()
        reward = -0.01
        self.steps += 1
        self.steps_since_progress += 1
        self.invuln_timer = max(0, self.invuln_timer - 1)
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.super_timer = max(0, self.super_timer - 1)
        self.freeze_timer = max(0, self.freeze_timer - 1)
        self._update_visual_events()

        previous_target, previous_distance = self._current_target()

        if self.gravity_timer > 0:
            self.gravity_timer -= 1
            if self.gravity_timer == 0:
                self.gravity_dir = 1

        move_dir, wants_jump, wants_shoot, wants_interact = self._decode_action(action)
        self._apply_player_input(move_dir, wants_jump)

        if wants_shoot:
            reward += self._try_shoot()

        self._move_player()

        if wants_interact:
            reward += self._try_interact()

        reward += self._collect_pickups()
        reward += self._update_bullets()
        reward += self._update_enemies()
        reward += self._check_player_danger()
        reward += self._check_exit()
        reward += self._target_progress_reward(previous_target, previous_distance)

        # Completion-progress shaping (PT-01/PT-02): reward any increase in the
        # monotonic progress potential — deeper into the cave, more crystals, the
        # switch thrown, or the exit reached. Potential-based, so it stays >= 0
        # and gives the agent dense credit on the long path to a full clear.
        self._max_depth_row = max(self._max_depth_row, self._player_tile()[1])
        new_progress = self._progress_potential()[0]
        if new_progress > self._progress:
            reward += self.PROGRESS_REWARD_SCALE * (new_progress - self._progress)
            self._progress = new_progress
            self._mark_progress()

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            self._end_reason = "timeout"
            reward -= 8.0
        elif self.steps_since_progress >= self.MAX_STEPS_WITHOUT_PROGRESS and not self.game_over:
            self.game_over = True
            self._end_reason = "stalled"
            reward -= 6.0

        return self.get_state(), float(reward), self.game_over, self._info()

    def _progress_potential(self) -> Tuple[float, Dict[str, float]]:
        """Monotonic 0..1 completion potential and its components — how close the
        player is to clearing the cave (collect all crystals, throw the switch,
        reach the exit). Drives progress reward shaping and info['progress']."""
        total = max(1, self.initial_crystals)
        crystal_frac = (self.initial_crystals - len(self.crystals)) / total
        switch_done = 0.0 if (self.switches - self.used_switches) else 1.0
        span = max(1, self.level_rows - self.sky_rows - 1)
        depth_frac = float(
            np.clip((self._max_depth_row - self.sky_rows) / span, 0.0, 1.0)
        )
        won = 1.0 if self.won else 0.0
        phi = (
            self.PROGRESS_W_CRYSTAL * crystal_frac
            + self.PROGRESS_W_SWITCH * switch_done
            + self.PROGRESS_W_DEPTH * depth_frac
            + self.PROGRESS_W_WIN * won
        )
        return phi, {
            "crystal_frac": round(crystal_frac, 3),
            "switch_done": switch_done,
            "depth_frac": round(depth_frac, 3),
            "won": won,
        }

    def get_state(self) -> np.ndarray:
        """Return a normalized local state vector for DQN training."""
        idx = 0
        player_col = int((self.player_x + self.PLAYER_WIDTH / 2) // self.TILE_SIZE)
        player_row = int((self.player_y + self.PLAYER_HEIGHT / 2) // self.TILE_SIZE)
        half_cols = self.WINDOW_COLS // 2
        half_rows = self.WINDOW_ROWS // 2

        for row in range(player_row - half_rows, player_row + half_rows + 1):
            for col in range(player_col - half_cols, player_col + half_cols + 1):
                self._state_array[idx] = self._tile_code(col, row)
                idx += 1

        center_idx = (self.WINDOW_ROWS * self.WINDOW_COLS) // 2
        self._state_array[center_idx] = self.TILE_CODES[self.PLAYER]

        max_x = max(1.0, self.level_width - self.PLAYER_WIDTH)
        max_y = max(1.0, self.level_height - self.PLAYER_HEIGHT)
        max_level = max(1, len(self.CAVES) - 1)
        target_dx, target_dy, target_distance, target_kind = self._target_features()
        metadata = [
            self.player_x / max_x,
            self.player_y / max_y,
            self._normalize_signed(self.vx, self.MOVE_SPEED),
            self._normalize_signed(self.vy, self.MAX_FALL_SPEED),
            1.0 if self.facing > 0 else 0.0,
            1.0 if self.grounded else 0.0,
            len(self.crystals) / max(1, self.initial_crystals),
            self.health / self.MAX_HEALTH,
            min(1.0, self.ammo / self.MAX_AMMO_FOR_STATE),
            1.0 if self.exit_unlocked else 0.0,
            self.level_index / max_level,
            self.super_timer / self.MAX_POWER_TIMER,
            self.freeze_timer / self.MAX_POWER_TIMER,
            1.0 if self.gravity_dir > 0 else 0.0,
            min(1.0, self.steps / self.MAX_STEPS),
            target_dx,
            target_dy,
            target_distance,
            target_kind,
            min(1.0, self.steps_since_progress / self.MAX_STEPS_WITHOUT_PROGRESS),
        ]
        self._state_array[idx:] = np.array(metadata, dtype=np.float32)
        return self._state_array.copy()

    def close(self) -> None:
        """Clean up resources."""
        pass

    def seed(self, seed: int) -> None:
        """Set random seed for deterministic tests or training."""
        np.random.seed(seed)

    def get_action_labels(self) -> List[str]:
        """Return action labels for neural network visualization."""
        return self.ACTION_LABELS.copy()

    def get_human_action(self, keys: dict) -> int:
        """Convert keyboard input to a single discrete action."""
        left = bool(keys.get(pygame.K_LEFT) or keys.get(pygame.K_a))
        right = bool(keys.get(pygame.K_RIGHT) or keys.get(pygame.K_d))
        jump = bool(keys.get(pygame.K_SPACE) or keys.get(pygame.K_UP) or keys.get(pygame.K_w))
        shoot = bool(keys.get(pygame.K_z) or keys.get(pygame.K_LCTRL) or keys.get(pygame.K_RCTRL))
        interact = bool(keys.get(pygame.K_e) or keys.get(pygame.K_RETURN))

        if interact:
            return self.INTERACT
        if shoot and left:
            return self.LEFT_SHOOT
        if shoot and right:
            return self.RIGHT_SHOOT
        if shoot:
            return self.SHOOT
        if jump and left:
            return self.LEFT_JUMP
        if jump and right:
            return self.RIGHT_JUMP
        if jump:
            return self.JUMP
        if left:
            return self.LEFT
        if right:
            return self.RIGHT
        return self.IDLE

    def advance_cave(self) -> None:
        """Move to the next cave for human-play sessions or curriculum tests."""
        self.level_index = (self.level_index + 1) % len(self.CAVES)
        self.reset()

    def _load_level(self, level: CaveSpec) -> None:
        width = max(len(row) for row in level.layout)
        rows = [row.ljust(width, self.EMPTY) for row in level.layout]
        self.grid = [[self.EMPTY for _ in range(width)] for _ in rows]
        self.level_cols = width
        self.level_rows = len(rows)
        self.level_width = self.level_cols * self.TILE_SIZE
        self.level_height = self.level_rows * self.TILE_SIZE
        self.sky_rows = getattr(level, "sky_rows", 0)

        self.crystals.clear()
        self.doors.clear()
        self.switches.clear()
        self.used_switches.clear()
        self.hazards.clear()
        self.hazard_kinds.clear()
        self.ammo_pickups.clear()
        self.treasures.clear()
        self.powerups.clear()
        self.air_tanks.clear()
        self.enemies.clear()

        for row, line in enumerate(rows):
            for col, char in enumerate(line):
                if char == self.SOLID:
                    self.grid[row][col] = self.SOLID
                elif char == self.PLAYER:
                    self.player_x = col * self.TILE_SIZE + 5
                    self.player_y = row * self.TILE_SIZE + 1
                elif char == self.CRYSTAL:
                    self.crystals.add((col, row))
                elif char == self.EXIT:
                    self.exit_pos = (col, row)
                elif char == self.DOOR:
                    self.doors.add((col, row))
                elif char == self.SWITCH:
                    self.switches.add((col, row))
                elif char in (self.SPIKE, self.ACID):
                    self.hazards.add((col, row))
                    self.hazard_kinds[(col, row)] = char
                elif char == self.AMMO:
                    self.ammo_pickups.add((col, row))
                elif char == self.TREASURE:
                    self.treasures.add((col, row))
                elif char in (self.POWER_SHOT, self.GRAVITY_POWER, self.FREEZE_POWER):
                    self.powerups[(col, row)] = char
                elif char == self.AIR_TANK:
                    self.air_tanks.add((col, row))
                elif char == self.CRAWLER:
                    self.enemies.append(
                        Enemy(col * self.TILE_SIZE + 4, row * self.TILE_SIZE + 8, 1.1)
                    )
                elif char == self.FLYER:
                    self.enemies.append(
                        Enemy(
                            col * self.TILE_SIZE + 4,
                            row * self.TILE_SIZE + 4,
                            1.6,
                            "flyer",
                        )
                    )

        self.initial_crystals = len(self.crystals)

    def _info(self) -> dict:
        return {
            "score": self.score,
            "health": self.health,
            "ammo": self.ammo,
            "crystals_remaining": len(self.crystals),
            "initial_crystals": self.initial_crystals,
            "exit_unlocked": self.exit_unlocked,
            "doors_open": self.doors_open,
            "level": self.level_index,
            "level_name": self.level.name,
            "won": self.won,
            "steps": self.steps,
            "steps_since_progress": self.steps_since_progress,
            "progress": round(self._progress, 3),
            "progress_parts": self._progress_potential()[1],
            "end_reason": self._end_reason,
        }

    def _player_rect(self, x: Optional[float] = None, y: Optional[float] = None) -> pygame.Rect:
        return pygame.Rect(
            int(self.player_x if x is None else x),
            int(self.player_y if y is None else y),
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )

    def _player_tile(self) -> Tuple[int, int]:
        return (
            int((self.player_x + self.PLAYER_WIDTH / 2) // self.TILE_SIZE),
            int((self.player_y + self.PLAYER_HEIGHT / 2) // self.TILE_SIZE),
        )

    def _player_center(self) -> Tuple[float, float]:
        return (
            self.player_x + self.PLAYER_WIDTH / 2,
            self.player_y + self.PLAYER_HEIGHT / 2,
        )

    def _tile_center(self, tile: Tuple[int, int]) -> Tuple[float, float]:
        col, row = tile
        return (
            col * self.TILE_SIZE + self.TILE_SIZE / 2,
            row * self.TILE_SIZE + self.TILE_SIZE / 2,
        )

    def _tile_rect(self, tile: Tuple[int, int]) -> pygame.Rect:
        col, row = tile
        return pygame.Rect(
            col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE
        )

    def _tiles_for_rect(self, rect: pygame.Rect) -> Set[Tuple[int, int]]:
        left = rect.left // self.TILE_SIZE
        right = (rect.right - 1) // self.TILE_SIZE
        top = rect.top // self.TILE_SIZE
        bottom = (rect.bottom - 1) // self.TILE_SIZE
        return {(col, row) for row in range(top, bottom + 1) for col in range(left, right + 1)}

    def _rect_collides_solid(self, rect: pygame.Rect) -> bool:
        return any(self._solid_at(col, row) for col, row in self._tiles_for_rect(rect))

    def _solid_at(self, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.level_cols or row >= self.level_rows:
            return True
        if self.grid[row][col] == self.SOLID:
            return True
        return (col, row) in self.doors and not self.doors_open

    def _is_on_surface(self) -> bool:
        rect = self._player_rect()
        rect.y += self.gravity_dir
        return self._rect_collides_solid(rect)

    def _tile_code(self, col: int, row: int) -> float:
        if col < 0 or row < 0 or col >= self.level_cols or row >= self.level_rows:
            return self.TILE_CODES[self.SOLID]

        tile = (col, row)
        for enemy in self.enemies:
            if enemy.alive and self._tile_for_enemy(enemy) == tile:
                return self.TILE_CODES[self.FLYER if enemy.kind == "flyer" else self.CRAWLER]

        if self._solid_at(col, row):
            return self.TILE_CODES[self.DOOR] if tile in self.doors else self.TILE_CODES[self.SOLID]
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

    def _tile_for_enemy(self, enemy: Enemy) -> Tuple[int, int]:
        return (
            int((enemy.x + enemy.width / 2) // self.TILE_SIZE),
            int((enemy.y + enemy.height / 2) // self.TILE_SIZE),
        )

    @staticmethod
    def _normalize_signed(value: float, max_abs: float) -> float:
        return float(np.clip((value / max_abs + 1.0) * 0.5, 0.0, 1.0))
