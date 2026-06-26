"""
Crystal Caves-style game implementation (core game class).

Clean-room DOS-era puzzle platformer: collect every crystal, open doors with
switches, avoid hazards, shoot enemies with limited ammo, and escape. Designed
for human play and DQN training. Rendering, dressing, and step-simulation logic
live in sibling mixin modules to keep each file focused and under budget.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

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
    Elevator,
    Enemy,
    VisualEvent,
)
from .crystal_caves_geometry import CrystalCavesGeometryMixin
from .crystal_caves_logic import CrystalCavesLogicMixin
from .crystal_caves_rendering import CrystalCavesRenderingMixin


class CrystalCaves(
    CrystalCavesRenderingMixin,
    CrystalCavesDressingMixin,
    CrystalCavesLogicMixin,
    CrystalCavesGeometryMixin,
    BaseGame,
):
    """Puzzle platform game tailored for DQN training."""

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

    # Local perception window — sized to roughly what a 1991 Crystal Caves player
    # saw on screen (~20x10 tiles), not the old tiny 11x9 sliver. (AI-1)
    WINDOW_COLS = 19
    WINDOW_ROWS = 11

    # Coarse global objective map: the level downsampled to GLOBAL_MAP_COLS x
    # GLOBAL_MAP_ROWS cells, each marking the highest-priority remaining objective
    # in that region (crystal/switch/exit). Gives the agent a memory of where the
    # objectives are beyond its window — the fix for local-window blindness. (AI-1)
    GLOBAL_MAP_COLS = 11
    GLOBAL_MAP_ROWS = 6
    BASE_METADATA_SIZE = 20
    METADATA_SIZE = BASE_METADATA_SIZE
    HISTORY_FEATURES_PER_STEP = 7
    MAX_HEALTH = 3
    MAX_AMMO_FOR_STATE = 20
    MAX_STEPS = 3000
    MAX_STEPS_WITHOUT_PROGRESS = 720
    MAX_POWER_TIMER = 420

    # Completion-progress potential (0..1); shaping uses PBRS with terminal Phi=0.
    PROGRESS_W_CRYSTAL = 0.56  # fraction of crystals collected
    PROGRESS_W_SWITCH = 0.15  # every required switch thrown
    PROGRESS_W_DEPTH = 0.09  # deepest row reached (how far into the cave)
    PROGRESS_W_WIN = 0.20  # reached the exit
    # Potential scale for completion-progress shaping.
    PROGRESS_REWARD_SCALE = 10.0

    # Capped one-time bonuses for reaching objective/new coarse map regions.
    OBJECTIVE_REGION_BONUS = 0.4
    OBJECTIVE_REGION_CAP = 4.0
    NOVELTY_REGION_BONUS = 0.08
    NOVELTY_REGION_CAP = 3.0

    # Discrete bonuses for causal objective milestones.
    SWITCH_THROW_BONUS = 8.0
    ALL_CRYSTALS_COLLECTED_BONUS = 16.0
    FIRST_CRYSTAL_GOAL_BONUS = 20.0
    INVALID_INTERACT_PENALTY = -0.05
    INVALID_SHOOT_PENALTY = -0.04

    # Nonfarmable closest-approach bonus for the current objective.
    TARGET_BEST_APPROACH_SCALE = 0.18
    TARGET_BEST_APPROACH_CAP = 0.18

    # Per-frame delta-distance reward, tuned to clear the -0.01 living penalty.
    APPROACH_REWARD_SCALE = 0.15
    APPROACH_REWARD_CLIP_POS = 0.12
    APPROACH_REWARD_CLIP_NEG = -0.03

    MOVE_SPEED = 4.2
    AIR_SPEED = 3.3
    JUMP_SPEED = 10.5
    GRAVITY = 0.52
    MAX_FALL_SPEED = 10.0
    FRICTION = 0.82
    BULLET_SPEED = 9.0
    SHOOT_COOLDOWN = 14
    INVULN_FRAMES = 70

    SHAKE_FRAMES = 16  # screen-shake duration on taking damage (render-only juice)

    EMPTY = "."
    SOLID = "#"
    CRYSTAL = "*"
    EXIT = "E"
    DOOR = "D"
    SWITCH = "s"
    DOOR2 = "d"  # second colour-keyed door
    SWITCH2 = "S"  # second colour-keyed lever
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
    ELEVATOR = "="
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
        DOOR2: 0.55,
        SWITCH: 0.68,
        SWITCH2: 0.70,
        AMMO: 0.76,
        POWER_SHOT: 0.84,
        GRAVITY_POWER: 0.88,
        FREEZE_POWER: 0.92,
        AIR_TANK: 0.72,
        SPIKE: 0.96,
        ACID: 0.98,
        ELEVATOR: 0.52,
        CRAWLER: 0.62,
        FLYER: 0.66,
        PLAYER: 1.0,
    }

    # Elevator platform speed (tiles per physics step).
    ELEVATOR_SPEED = 0.06

    # Colour-keyed lever/door pairs: a lever opens only the door of its colour.
    DOOR_COLOR_OF: Dict[str, str] = {DOOR: "red", DOOR2: "blue"}
    SWITCH_COLOR_OF: Dict[str, str] = {SWITCH: "red", SWITCH2: "blue"}

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

        # AI-1 state toggle: legacy uses the old 11x9 window with no global map
        # (119-feature state); rich uses the class defaults (19x11 + 11x6 map).
        if not getattr(self.config, "CRYSTAL_CAVES_RICH_STATE", True):
            self.WINDOW_COLS = 11
            self.WINDOW_ROWS = 9
            self.GLOBAL_MAP_COLS = 0
            self.GLOBAL_MAP_ROWS = 0

        self._history_state_enabled = bool(
            getattr(self.config, "CRYSTAL_CAVES_HISTORY_STATE", False)
        )
        self._history_steps = (
            max(0, int(getattr(self.config, "CRYSTAL_CAVES_HISTORY_STEPS", 4) or 0))
            if self._history_state_enabled
            else 0
        )
        self._history_metadata_size = self._history_steps * self.HISTORY_FEATURES_PER_STEP
        self.METADATA_SIZE = self.BASE_METADATA_SIZE + self._history_metadata_size

        # Publish the spatial layout so a convolutional network (USE_CNN_STATE) can
        # reshape the flat state into the perception window + global map + metadata.
        self.config.STATE_LAYOUT = {
            "window": (self.WINDOW_ROWS, self.WINDOW_COLS),
            "gmap": (self.GLOBAL_MAP_ROWS, self.GLOBAL_MAP_COLS),
            "meta": self.METADATA_SIZE,
        }

        self.level_index = 0
        # Per-episode level selection. Training samples a random cave from a pool so
        # the agent generalises instead of memorising one level; evaluation switches
        # to a fixed held-out set (see use_eval_levels). These hold the procedural
        # generation params so both the training pool and the held-out set can be
        # built from the same family/difficulty with disjoint seed ranges.
        self._proc_params: Optional[dict] = None
        self._randomize_levels = False
        self._eval_mode = False
        self._eval_caves: Tuple[CaveSpec, ...] = ()
        self._eval_cursor = 0
        # Drill mode: replace the cave set with the hand-authored single-skill drills
        # (for skill diagnostics and motor-skill pre-training). Takes precedence.
        if getattr(self.config, "CRYSTAL_CAVES_DRILLS", False):
            from .crystal_caves_drills import DRILL_CAVES

            self.CAVES = DRILL_CAVES
            self.CAVE_DRESSING = {i: () for i in range(len(DRILL_CAVES))}
            self._randomize_levels = len(DRILL_CAVES) > 1
        # Bridge mode: small compositional teaching levels between the single
        # skill drills and procedural tutorial caves. Drill mode takes
        # precedence when both flags are accidentally enabled.
        elif getattr(self.config, "CRYSTAL_CAVES_BRIDGES", False):
            from .crystal_caves_drills import BRIDGE_CAVES

            self.CAVES = BRIDGE_CAVES
            self.CAVE_DRESSING = {i: () for i in range(len(BRIDGE_CAVES))}
            self._randomize_levels = len(BRIDGE_CAVES) > 1
        # Contact mode: tiny final-objective rooms used only for interleaved
        # training lanes. This stays separate from the game-faithful final caves.
        elif getattr(self.config, "CRYSTAL_CAVES_CONTACT_LEVELS", False):
            from .crystal_caves_drills import CONTACT_CAVES, contact_pool_caves

            contact_pool_size = int(getattr(self.config, "CRYSTAL_CAVES_CONTACT_POOL_SIZE", 0) or 0)
            if contact_pool_size > 0:
                contact_seed = int(
                    getattr(
                        self.config,
                        "CRYSTAL_CAVES_CONTACT_POOL_SEED",
                        getattr(self.config, "CRYSTAL_CAVES_SEED", 0),
                    )
                    or 0
                )
                contact_caves = contact_pool_caves(contact_pool_size, seed=contact_seed)
            else:
                contact_caves = CONTACT_CAVES
            self.CAVES = contact_caves
            self.CAVE_DRESSING = {i: () for i in range(len(contact_caves))}
            self._randomize_levels = len(contact_caves) > 1
        # Procedural mode: replace the authored caves with freshly generated ones.
        # Authored dressing is cleared since generated caves have none.
        elif getattr(self.config, "CRYSTAL_CAVES_PROCEDURAL", False):
            from .crystal_caves_gen import FAMILY_NAMES, THEME_NAMES, generate_cave

            base = getattr(self.config, "CRYSTAL_CAVES_SEED", 0)
            requested = [
                f.strip()
                for f in getattr(self.config, "CRYSTAL_CAVES_FAMILIES", "").split(",")
                if f.strip()
            ]
            invalid = [f for f in requested if f not in FAMILY_NAMES]
            if invalid:  # surface config typos instead of silently using all families
                warnings.warn(
                    f"Unknown Crystal Caves families ignored: {invalid}. "
                    f"Valid families: {', '.join(FAMILY_NAMES)}",
                    stacklevel=2,
                )
            wanted = [f for f in requested if f in FAMILY_NAMES]
            difficulty = getattr(self.config, "CRYSTAL_CAVES_DIFFICULTY", "normal")
            self._proc_params = {
                "base": base,
                "families": wanted or list(FAMILY_NAMES),
                "themes": THEME_NAMES,
                "difficulty": difficulty,
                "generate": generate_cave,
            }
            pool_size = int(getattr(self.config, "CRYSTAL_CAVES_POOL_SIZE", 0))
            count = max(pool_size, len(FAMILY_NAMES))
            self.CAVES = self._build_cave_set(count, seed_offset=0)
            self.CAVE_DRESSING = {i: () for i in range(count)}
            # Sample a random cave per episode once the pool has variety to offer.
            self._randomize_levels = pool_size > 0 and count > 1
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
        self.shake_timer = 0
        self.super_timer = 0
        self.freeze_timer = 0
        self.gravity_timer = 0
        self.gravity_dir = 1
        self.open_colors: Set[str] = set()
        self.exit_unlocked = False
        self.show_controls = False
        self.show_agent_overlay = False  # educational: draw the agent's view + goal
        self._end_reason = "running"
        self._max_depth_row = 0
        self._progress = 0.0
        self._progress_initial = 0.0
        self._closeness_initial = 0.0
        self._progress_phi = 0.0
        self._geodesic_field: Optional[Dict[Tuple[int, int], int]] = None
        self._geodesic_field_key: Optional[Tuple] = None
        self._visited_obj_cells: Set[Tuple[int, int]] = set()  # AI-2
        self._obj_region_total = 0.0  # AI-2
        self._visited_novelty_cells: Set[Tuple[int, int]] = set()
        self._novelty_bonus_total = 0.0
        self._anti_loop_tile: Optional[Tuple[int, int]] = None
        self._anti_loop_same_tile_steps = 0
        self._anti_loop_no_approach_steps = 0
        self._anti_loop_recent_tiles: Deque[Tuple[int, int]] = deque(maxlen=80)
        self._anti_loop_total = 0.0

        self.crystals: Set[Tuple[int, int]] = set()
        self.initial_crystals = 0
        self.exit_pos = (0, 0)
        self.doors: Set[Tuple[int, int]] = set()
        self.switches: Set[Tuple[int, int]] = set()
        self.used_switches: Set[Tuple[int, int]] = set()
        self.door_color: Dict[Tuple[int, int], str] = {}
        self.switch_color: Dict[Tuple[int, int], str] = {}
        self.hazards: Set[Tuple[int, int]] = set()
        self.hazard_kinds: Dict[Tuple[int, int], str] = {}
        self.ammo_pickups: Set[Tuple[int, int]] = set()
        self.treasures: Set[Tuple[int, int]] = set()
        self.powerups: Dict[Tuple[int, int], str] = {}
        self.air_tanks: Set[Tuple[int, int]] = set()
        self.enemies: List[Enemy] = []
        self.elevators: List[Elevator] = []
        self._elevator_solid: List[pygame.Rect] = []  # platform collision rects
        self.bullets: List[Bullet] = []
        self.visual_events: List[VisualEvent] = []
        self._action_history: Deque[Tuple[int, float]] = deque(maxlen=max(1, self._history_steps))

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
        """State vector dimension: local window + global objective map + metadata."""
        return (
            self.WINDOW_COLS * self.WINDOW_ROWS
            + self.GLOBAL_MAP_COLS * self.GLOBAL_MAP_ROWS
            + self.METADATA_SIZE
        )

    @property
    def action_size(self) -> int:
        """Number of possible actions."""
        return len(self.ACTION_LABELS)

    def _build_cave_set(self, count: int, seed_offset: int) -> Tuple[CaveSpec, ...]:
        """Generate ``count`` distinct procedural caves from the stored generation
        params. ``seed_offset`` carves out a disjoint seed range so the training
        pool (offset 0) and the held-out eval set never overlap."""
        p = self._proc_params
        assert p is not None, "procedural generation params not initialised"
        themes, fams = p["themes"], p["families"]
        return tuple(
            p["generate"](
                p["base"] * 1000 + seed_offset + i,
                themes[i % len(themes)],
                fams[i % len(fams)],
                difficulty=p["difficulty"],
            )
            for i in range(count)
        )

    def use_eval_levels(self, count: int) -> None:
        """Switch this instance into evaluation mode: reset() will deterministically
        cycle a fixed HELD-OUT set of ``count`` caves (disjoint from the training
        pool, identical across calls) so eval measures generalisation, not memory.
        No-op for authored (non-procedural) caves, which are already a fixed set."""
        if self._proc_params is None or count <= 0:
            return
        # Build once; the seed range (offset 500000) is disjoint from the training
        # pool's offset-0 range, so these levels are never seen during training.
        if len(self._eval_caves) != count:
            self._eval_caves = self._build_cave_set(count, seed_offset=500000)
        self._eval_mode = True
        self._eval_cursor = 0

    def reset_eval_cursor(self) -> None:
        """Restart the held-out cycle so every evaluation plays the same levels in
        the same order (reproducible eval across training checkpoints)."""
        self._eval_cursor = 0

    def reset(self) -> np.ndarray:
        """Reset and return the initial state. In eval mode, deterministically cycle
        the held-out caves; in training with a pool, sample a random cave; otherwise
        fall back to the legacy CAVES[level_index] behaviour."""
        if self._eval_mode and self._eval_caves:
            self.level_index = self._eval_cursor % len(self._eval_caves)
            self.level = self._eval_caves[self.level_index]
            self._eval_cursor += 1
        elif self._randomize_levels and len(self.CAVES) > 1:
            self.level_index = int(np.random.randint(len(self.CAVES)))
            self.level = self.CAVES[self.level_index]
        else:
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
        self.shake_timer = 0
        self.super_timer = 0
        self.freeze_timer = 0
        self.gravity_timer = 0
        self.gravity_dir = 1
        self.open_colors = set()
        self.exit_unlocked = False
        self.bullets.clear()
        self.visual_events.clear()

        # Completion-progress tracker (deepest row reached + PBRS potential).
        self._max_depth_row = self._player_tile()[1]
        self._progress = self._progress_potential()[0]
        self._progress_initial = self._progress
        # Invalidate the geodesic cache for the new level before computing closeness.
        self._geodesic_field = None
        self._geodesic_field_key = None
        # Capture initial closeness BEFORE the first PBRS potential so the geodesic
        # term (like the base term) telescopes to exactly 0 over a full episode.
        self._closeness_initial = self._target_closeness()
        self._progress_phi = self._progress_pbrs_potential(raw_progress=self._progress)
        self._end_reason = "running"
        self._visited_obj_cells = set()
        self._obj_region_total = 0.0
        self._visited_novelty_cells = set()
        self._novelty_bonus_total = 0.0
        self._remember_current_novelty_cell()
        self._target_best_distances: Dict[Tuple[str, int, int], float] = {}
        self._anti_loop_tile = None
        self._anti_loop_same_tile_steps = 0
        self._anti_loop_no_approach_steps = 0
        self._anti_loop_recent_tiles.clear()
        self._anti_loop_total = 0.0
        self._invalid_interact_count = 0
        self._invalid_interact_total = 0.0
        self._invalid_shoot_count = 0
        self._invalid_shoot_total = 0.0
        self._reset_history_state()

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance the cave simulation by one frame."""
        action = validate_action(action, self.action_size, "CrystalCaves")

        if self.game_over:
            return self.get_state(), 0.0, True, self._info()
        reward = -0.01
        self.steps += 1
        self.steps_since_progress += 1
        previous_progress_phi = self._progress_phi
        self.invuln_timer = max(0, self.invuln_timer - 1)
        self.shake_timer = max(0, self.shake_timer - 1)
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.super_timer = max(0, self.super_timer - 1)
        self.freeze_timer = max(0, self.freeze_timer - 1)
        self._update_visual_events()

        previous_target, previous_distance = self._current_target()

        if self.gravity_timer > 0:
            self.gravity_timer -= 1
            if self.gravity_timer == 0:
                self.gravity_dir = 1

        self._update_elevators()

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
        reward += self._anti_loop_penalty(previous_target, previous_distance)

        self._max_depth_row = max(self._max_depth_row, self._player_tile()[1])

        # AI-2: reward reaching a new region that holds a known uncollected objective
        region_bonus = self._objective_region_reward()
        if region_bonus:
            reward += region_bonus
            self._mark_progress()
        novelty_bonus = self._novelty_region_reward()
        if novelty_bonus:
            reward += novelty_bonus
            self._mark_progress()

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            self._end_reason = "timeout"
            reward -= 8.0
        elif self.steps_since_progress >= self.MAX_STEPS_WITHOUT_PROGRESS and not self.game_over:
            self.game_over = True
            self._end_reason = "stalled"
            reward -= 6.0

        reward += self._progress_shaping_reward(previous_progress_phi)

        self._record_history_step(action, previous_target, previous_distance)
        return self.get_state(), float(reward), self.game_over, self._info()

    def _progress_potential(self) -> Tuple[float, Dict[str, float]]:
        """Monotonic 0..1 completion potential and its components — how close the
        player is to clearing the cave (collect all crystals, throw the switch,
        reach the exit). Drives progress reward shaping and info['progress']."""
        total = max(1, self.initial_crystals)
        crystal_frac = (self.initial_crystals - len(self.crystals)) / total
        switch_done = 0.0 if (self.switches - self.used_switches) else 1.0
        span = max(1, self.level_rows - self.sky_rows - 1)
        depth_frac = float(np.clip((self._max_depth_row - self.sky_rows) / span, 0.0, 1.0))
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

    def _progress_pbrs_potential(
        self,
        raw_progress: float | None = None,
        *,
        terminal: bool | None = None,
    ) -> float:
        is_terminal = self.game_over if terminal is None else terminal
        if is_terminal:
            return 0.0
        progress = self._progress_potential()[0] if raw_progress is None else raw_progress
        phi = self.PROGRESS_REWARD_SCALE * (float(progress) - self._progress_initial)
        if getattr(self.config, "CRYSTAL_CAVES_GEODESIC_POTENTIAL", False):
            weight = float(getattr(self.config, "CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT", 0.3))
            closeness = self._target_closeness()
            phi += self.PROGRESS_REWARD_SCALE * weight * (closeness - self._closeness_initial)
        return phi

    def _active_target_tiles(self) -> frozenset:
        """The phase-ordered objective tiles the closeness potential aims at:
        unthrown switches first (they gate crystals), then remaining crystals, then
        the exit — mirroring the compass in ``_current_target``."""
        unused_switches = self.switches - self.used_switches
        if unused_switches and self.crystals:
            return frozenset(unused_switches)
        if self.crystals:
            return frozenset(self.crystals)
        if unused_switches:
            return frozenset(unused_switches)
        return frozenset({self.exit_pos})

    def _geodesic_distance_field(self) -> Dict[Tuple[int, int], int]:
        """Multi-source BFS tile distance to the nearest active objective over
        traversable tiles (4-connected), honouring walls and *locked* doors via
        ``_solid_at`` — a true route distance, not straight-line through walls.

        Cached and only recomputed when the objective set or open-door state
        changes (collecting a crystal, throwing a switch), so the per-step cost is
        a dict lookup. Vertical adjacency approximates the agent's jump/fall, which
        is enough for a smooth shaping gradient along the real corridors."""
        key = (self._active_target_tiles(), frozenset(self.open_colors))
        if self._geodesic_field is not None and self._geodesic_field_key == key:
            return self._geodesic_field

        field: Dict[Tuple[int, int], int] = {}
        queue: Deque[Tuple[int, int]] = deque()
        for col, row in key[0]:
            if not self._solid_at(col, row):
                field[(col, row)] = 0
                queue.append((col, row))
        while queue:
            col, row = queue.popleft()
            dist = field[(col, row)] + 1
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = col + dc, row + dr
                if (nc, nr) in field or self._solid_at(nc, nr):
                    continue
                field[(nc, nr)] = dist
                queue.append((nc, nr))

        self._geodesic_field = field
        self._geodesic_field_key = key
        return field

    def _target_closeness(self) -> float:
        """Normalized 0..1 geodesic closeness to the current objective (1 = on it).

        Re-aims automatically as objectives are cleared, and is a deterministic
        function of state, which keeps the closeness term PBRS-valid. Uses true
        route distance (``_geodesic_distance_field``) so the gradient follows
        traversable corridors instead of rewarding pushing toward a walled-off
        shortcut. Returns 0 when the objective is unreachable from the player's
        tile (e.g. behind a still-locked door)."""
        field = self._geodesic_distance_field()
        if not field:
            return 0.0
        dist = field.get(self._player_tile())
        if dist is None:
            return 0.0
        norm = max(field.values())
        if norm <= 0:
            return 1.0
        return float(1.0 - dist / norm)

    def _progress_shaping_reward(self, previous_phi: float) -> float:
        raw_progress = self._progress_potential()[0]
        if raw_progress > self._progress + 1e-9:
            self._mark_progress()
        self._progress = raw_progress
        next_phi = self._progress_pbrs_potential(raw_progress=raw_progress)
        self._progress_phi = next_phi
        gamma = float(getattr(self.config, "GAMMA", 0.99))
        return gamma * next_phi - previous_phi

    def get_state(self) -> np.ndarray:
        """Return a normalized local state vector for DQN training."""
        idx = 0
        player_col = int((self.player_x + self.PLAYER_WIDTH / 2) // self.TILE_SIZE)
        player_row = int((self.player_y + self.PLAYER_HEIGHT / 2) // self.TILE_SIZE)
        n_window = self.WINDOW_ROWS * self.WINDOW_COLS
        self._state_array[:n_window] = self._fill_window(player_col, player_row).reshape(-1)
        idx = n_window

        center_idx = (self.WINDOW_ROWS * self.WINDOW_COLS) // 2
        self._state_array[center_idx] = self.TILE_CODES[self.PLAYER]

        # coarse global objective map (where the remaining crystals/switches/exit
        # are across the whole cave) so the agent isn't blind beyond its window
        self._fill_global_map(idx)
        idx += self.GLOBAL_MAP_COLS * self.GLOBAL_MAP_ROWS

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
        if self._history_state_enabled:
            metadata.extend(self._history_metadata())
        self._state_array[idx:] = np.array(metadata, dtype=np.float32)
        return self._state_array.copy()

    def _reset_history_state(self) -> None:
        self._action_history.clear()

    def _record_history_step(
        self,
        action: int,
        previous_target: Optional[Tuple[str, int, int]],
        previous_distance: float,
    ) -> None:
        if not self._history_state_enabled or self._history_steps <= 0:
            return
        progress_delta = 0.5
        current_target, current_distance = self._current_target()
        if (
            current_target == previous_target
            and previous_target is not None
            and np.isfinite(previous_distance)
            and np.isfinite(current_distance)
        ):
            delta_tiles = (previous_distance - current_distance) / self.TILE_SIZE
            progress_delta = self._normalize_signed(delta_tiles, 1.0)
        self._action_history.append((int(action), progress_delta))

    def _history_metadata(self) -> list[float]:
        features: list[float] = []
        missing = self._history_steps - len(self._action_history)
        for _ in range(max(0, missing)):
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        for action, progress_delta in self._action_history:
            features.extend(self._history_action_features(action, progress_delta))
        return features

    def _history_action_features(self, action: int, progress_delta: float) -> list[float]:
        return [
            1.0 if action == self.IDLE else 0.0,
            1.0 if action in (self.LEFT, self.LEFT_JUMP, self.LEFT_SHOOT) else 0.0,
            1.0 if action in (self.RIGHT, self.RIGHT_JUMP, self.RIGHT_SHOOT) else 0.0,
            1.0 if action in (self.JUMP, self.LEFT_JUMP, self.RIGHT_JUMP) else 0.0,
            1.0 if action in (self.SHOOT, self.LEFT_SHOOT, self.RIGHT_SHOOT) else 0.0,
            1.0 if action == self.INTERACT else 0.0,
            float(np.clip(progress_delta, 0.0, 1.0)),
        ]

    def _fill_global_map(self, start: int) -> None:
        """Write a coarse global map of remaining objectives into the state."""
        gc, gr = self.GLOBAL_MAP_COLS, self.GLOBAL_MAP_ROWS
        if gc * gr == 0:  # legacy state: no global map
            return
        cw = max(1.0, self.level_cols / gc)
        ch = max(1.0, self.level_rows / gr)
        cells = [0.0] * (gc * gr)

        def mark(col: int, row: int, value: float) -> None:
            cx = min(gc - 1, int(col / cw))
            cy = min(gr - 1, int(row / ch))
            i = cy * gc + cx
            if value > cells[i]:
                cells[i] = value

        for c, r in self.crystals:
            mark(c, r, 0.4)
        for c, r in self.switches - self.used_switches:
            mark(c, r, 0.6)
        if self.exit_unlocked:
            mark(self.exit_pos[0], self.exit_pos[1], 0.9)
        elif getattr(self.config, "CRYSTAL_CAVES_SHOW_LOCKED_EXIT", False):
            # Reveal the still-locked exit at a distinct, lower value so the agent can
            # learn its route before the last crystal is collected. mark() keeps the
            # higher value per cell, so a co-located crystal/switch still dominates.
            mark(self.exit_pos[0], self.exit_pos[1], 0.2)

        self._state_array[start : start + gc * gr] = np.array(cells, dtype=np.float32)

    def _objective_region_reward(self) -> float:
        """One-time bonus for entering a coarse region with a remaining objective."""
        gc, gr = self.GLOBAL_MAP_COLS, self.GLOBAL_MAP_ROWS
        if gc * gr == 0 or self._obj_region_total >= self.OBJECTIVE_REGION_CAP:
            return 0.0  # legacy state has no map -> no region bonus
        cw = max(1.0, self.level_cols / gc)
        ch = max(1.0, self.level_rows / gr)
        pcol, prow = self._player_tile()
        cell = (min(gc - 1, int(pcol / cw)), min(gr - 1, int(prow / ch)))
        if cell in self._visited_obj_cells:
            return 0.0

        def in_cell(c: int, r: int) -> bool:
            return (min(gc - 1, int(c / cw)), min(gr - 1, int(r / ch))) == cell

        has_obj = any(in_cell(c, r) for c, r in self.crystals) or any(
            in_cell(c, r) for c, r in (self.switches - self.used_switches)
        )
        if not has_obj:
            return 0.0
        self._visited_obj_cells.add(cell)
        self._obj_region_total += self.OBJECTIVE_REGION_BONUS
        return self.OBJECTIVE_REGION_BONUS

    def _novelty_cell(self) -> tuple[int, int]:
        gc, gr = self.GLOBAL_MAP_COLS, self.GLOBAL_MAP_ROWS
        cw = max(1.0, self.level_cols / max(1, gc))
        ch = max(1.0, self.level_rows / max(1, gr))
        pcol, prow = self._player_tile()
        return (
            min(max(0, gc - 1), max(0, int(pcol / cw))),
            min(max(0, gr - 1), max(0, int(prow / ch))),
        )

    def _remember_current_novelty_cell(self) -> None:
        if self.GLOBAL_MAP_COLS * self.GLOBAL_MAP_ROWS == 0:
            return
        self._visited_novelty_cells.add(self._novelty_cell())

    def _novelty_region_reward(self) -> float:
        """Opt-in one-time bonus for entering new coarse regions."""
        if not getattr(self.config, "CRYSTAL_CAVES_NOVELTY_BONUS", False):
            return 0.0
        if self.game_over or self.GLOBAL_MAP_COLS * self.GLOBAL_MAP_ROWS == 0:
            return 0.0
        if self._novelty_bonus_total >= self.NOVELTY_REGION_CAP:
            return 0.0
        target, distance = self._current_target()
        if target is None or not np.isfinite(distance):
            return 0.0
        cell = self._novelty_cell()
        if cell in self._visited_novelty_cells:
            return 0.0
        self._visited_novelty_cells.add(cell)
        bonus = min(
            self.NOVELTY_REGION_BONUS,
            self.NOVELTY_REGION_CAP - self._novelty_bonus_total,
        )
        self._novelty_bonus_total += bonus
        return bonus

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
        self.door_color.clear()
        self.switch_color.clear()
        self.hazards.clear()
        self.hazard_kinds.clear()
        self.ammo_pickups.clear()
        self.treasures.clear()
        self.powerups.clear()
        self.air_tanks.clear()
        self.enemies.clear()
        self.elevators.clear()

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
                elif char in (self.DOOR, self.DOOR2):
                    self.doors.add((col, row))
                    self.door_color[(col, row)] = self.DOOR_COLOR_OF[char]
                elif char in (self.SWITCH, self.SWITCH2):
                    self.switches.add((col, row))
                    self.switch_color[(col, row)] = self.SWITCH_COLOR_OF[char]
                elif char in (self.SPIKE, self.ACID):
                    self.hazards.add((col, row))
                    self.hazard_kinds[(col, row)] = char
                elif char == self.ELEVATOR:
                    self.grid[row][col] = self.ELEVATOR
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

        # group ELEVATOR cells into vertical-shaft lifts; the platform starts at
        # the top and oscillates down to the bottom of each run
        for col in range(self.level_cols):
            row = 0
            while row < self.level_rows:
                if self.grid[row][col] == self.ELEVATOR:
                    top = row
                    while row < self.level_rows and self.grid[row][col] == self.ELEVATOR:
                        row += 1
                    self.elevators.append(
                        Elevator(col=col, top=top, bottom=row - 1, pos=float(top))
                    )
                else:
                    row += 1
        self._refresh_elevator_rects()

        # Cache static terrain masks for the vectorized state window. After load,
        # self.grid only ever holds SOLID / ELEVATOR chars, and both are static for
        # the level's lifetime — so these masks never need rebuilding mid-episode.
        grid_arr = np.array(self.grid, dtype="<U1")
        self._wall_mask = grid_arr == self.SOLID
        self._elevator_mask = grid_arr == self.ELEVATOR

    def _info(self) -> dict:
        return {
            "score": self.score,
            "health": self.health,
            "ammo": self.ammo,
            "crystals_remaining": len(self.crystals),
            "initial_crystals": self.initial_crystals,
            "switches_total": len(self.switches),
            "switches_used": len(self.used_switches),
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
            "anti_loop_penalty_total": round(self._anti_loop_total, 3),
            "invalid_interact_count": self._invalid_interact_count,
            "invalid_interact_penalty_total": round(self._invalid_interact_total, 3),
            "invalid_shoot_count": self._invalid_shoot_count,
            "invalid_shoot_penalty_total": round(self._invalid_shoot_total, 3),
            "novelty_bonus_total": round(self._novelty_bonus_total, 3),
        }
