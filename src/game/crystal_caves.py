"""
Crystal Caves-style game implementation.

This is a clean-room, DOS-era puzzle platformer inspired by the 1991
Crystal Caves formula: collect every crystal, open doors with switches,
avoid hazards, shoot enemies with limited ammo, and escape through the exit.

The implementation is designed for both human play and DQN training. It uses
deterministic tile maps, a compact local state window, and discrete compound
actions that let a neural network learn platforming without a separate input
combiner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame

from config import Config

from .base_game import BaseGame, validate_action
from .crystal_caves_art import EGA, CrystalCavesArt
from .crystal_caves_audio import CrystalCavesAudio


@dataclass(frozen=True)
class CaveSpec:
    """Static cave layout and display metadata."""

    name: str
    layout: Tuple[str, ...]
    background: Tuple[int, int, int]
    accent: Tuple[int, int, int]


@dataclass
class Bullet:
    """Projectile fired by the player."""

    x: float
    y: float
    vx: float
    ttl: int
    powered: bool = False

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), 10, 4)


@dataclass
class Enemy:
    """Simple enemy with either ground patrol or hovering movement."""

    x: float
    y: float
    vx: float
    kind: str = "crawler"
    alive: bool = True

    @property
    def width(self) -> int:
        return 24

    @property
    def height(self) -> int:
        return 24

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)


@dataclass(frozen=True)
class DressingPiece:
    """Authored visual-only prop placed in a cave room."""

    kind: str
    col: int
    row: int
    label: str = ""


@dataclass
class VisualEvent:
    """Short-lived arcade feedback effect."""

    kind: str
    x: float
    y: float
    ttl: int
    max_ttl: int
    text: str = ""
    color: Tuple[int, int, int] = (255, 255, 255)


class CrystalCaves(BaseGame):
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

    CAVES: Tuple[CaveSpec, ...] = (
        CaveSpec(
            name="Trouble with Twinkles",
            background=(9, 12, 22),
            accent=(80, 190, 255),
            layout=(
                "############################################",
                "#..........................................#",
                "#..P......*.......A......*............E....#",
                "#.....########...........#..#......####D####",
                "#..................M.....#..#..s..........#",
                "#.........####...........#..########.......#",
                "#..*......#..#.....O.....#.........*......#",
                "#.........#..#...........#....^^^^........#",
                "#.....A...#..#....####...#................#",
                "#.........####....#..#...#####.....M......#",
                "#.................#..#.............####...#",
                "#....*.......p....#..#.....*..............#",
                "#........#####....#..###########..........#",
                "#..M..........#................#.....*....#",
                "#.........*...#..........A.....#..........#",
                "#.............#................#..........#",
                "#^^^^^........#~~~~~...........#.....^^^^^#",
                "############################################",
            ),
        ),
        CaveSpec(
            name="Slugging It Out",
            background=(12, 10, 18),
            accent=(255, 188, 80),
            layout=(
                "############################################",
                "#..........................................#",
                "#..P....A.....*.........................E..#",
                "#........#########....................DD####",
                "#..*....................F.............DD...#",
                "#...............#####......................#",
                "#.....M.............#...........*..........#",
                "#..########.........#......##########......#",
                "#.........#.....*...#..............#.......#",
                "#.........#.........#..s...........#..*....#",
                "#.........#####.....########.......#.......#",
                "#..g.......................#.......#.......#",
                "#.............^^^^......O...M......#.......#",
                "#....*.................#########...#..A....#",
                "#...........########.......................#",
                "#...................*..........p...........#",
                "#~~~~~..............................^^^^...#",
                "############################################",
            ),
        ),
        CaveSpec(
            name="Mylo and the Supernova",
            background=(8, 15, 13),
            accent=(120, 255, 155),
            layout=(
                "############################################",
                "#..........................................#",
                "#..P........*..................A........E..#",
                "#........###########..............##########",
                "#..*...............#.......F.....O.....D...#",
                "#..........M.......#...................D...#",
                "#.....########.....#....*.....#########D...#",
                "#...........#......#.................#.....#",
                "#...........#..A...#######..........#..*..#",
                "#..z........#.............#....s.....#.....#",
                "#...........#####.........########...#.....#",
                "#....................M...............#.....#",
                "#....*.....########...........^^^^...#.....#",
                "#................#...................#.....#",
                "#..........p.....#....*..............#.....#",
                "#................#............*............#",
                "#^^^^^...........#~~~~~~~..............^^^^#",
                "############################################",
            ),
        ),
    )

    CAVE_DRESSING: Dict[int, Tuple[DressingPiece, ...]] = {
        0: (
            DressingPiece("beacon", 2, 2),
            DressingPiece("mine_sign", 6, 2, "MINE"),
            DressingPiece("cable_h", 13, 3, "7"),
            DressingPiece("generator", 17, 4),
            DressingPiece("terminal", 20, 4),
            DressingPiece("clear_blocks", 25, 3, "4"),
            DressingPiece("crystal_light", 11, 5),
            DressingPiece("pipe_stack", 4, 8),
            DressingPiece("room_label", 2, 9, "LANDING"),
            DressingPiece("warning_post", 27, 7),
            DressingPiece("eye_turret", 30, 8),
            DressingPiece("mushroom", 29, 16),
            DressingPiece("hammer_marker", 32, 16),
            DressingPiece("vacuum", 34, 10),
            DressingPiece("zapper", 36, 16),
            DressingPiece("bat_perch", 38, 8),
            DressingPiece("elevator_frame", 37, 2, "EXIT"),
        ),
        1: (
            DressingPiece("mine_sign", 4, 2, "SLUG"),
            DressingPiece("pipe_stack", 11, 4),
            DressingPiece("generator", 20, 6),
            DressingPiece("cable_h", 19, 5, "6"),
            DressingPiece("mushroom", 11, 12),
            DressingPiece("slug_enemy", 18, 13),
            DressingPiece("clear_blocks", 26, 7, "3"),
            DressingPiece("vacuum", 30, 12),
            DressingPiece("room_label", 3, 15, "SLUG PIT"),
            DressingPiece("elevator_frame", 39, 2, "EXIT"),
        ),
        2: (
            DressingPiece("beacon", 4, 2),
            DressingPiece("terminal", 12, 5),
            DressingPiece("cable_h", 22, 4, "7"),
            DressingPiece("zapper", 30, 12),
            DressingPiece("generator", 36, 4),
            DressingPiece("clear_blocks", 18, 8, "5"),
            DressingPiece("eye_turret", 24, 10),
            DressingPiece("warning_post", 34, 15),
            DressingPiece("room_label", 7, 15, "MOON MINE"),
            DressingPiece("elevator_frame", 39, 2, "EXIT"),
        ),
    }

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

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward -= 8.0
        elif (
            self.steps_since_progress >= self.MAX_STEPS_WITHOUT_PROGRESS
            and not self.game_over
        ):
            self.game_over = True
            reward -= 6.0

        return self.get_state(), float(reward), self.game_over, self._info()

    def render(self, screen) -> None:
        """Render the cave, HUD, and DOS-era pixel art overlays."""
        camera_x, camera_y = self._camera()
        screen.fill((0, 0, 0))
        self._draw_background(screen, camera_x, camera_y)
        self._draw_tiles(screen, camera_x, camera_y)
        self._draw_switch_wires(screen, camera_x, camera_y)
        self._draw_level_dressing(screen, camera_x, camera_y)
        self._draw_pickups(screen, camera_x, camera_y)
        self._draw_enemies(screen, camera_x, camera_y)
        self._draw_bullets(screen, camera_x, camera_y)
        self._draw_player(screen, camera_x, camera_y)
        self._draw_visual_events(screen, camera_x, camera_y)
        self._draw_gravity_overlay(screen)
        self._draw_hud(screen)

        if self.game_over and self._font:
            title = "CAVE CLEARED" if self.won else "MYLO DOWN"
            color = (88, 255, 88) if self.won else (255, 80, 80)
            text = self._font.render(title, True, color)
            rect = text.get_rect(center=(self.width // 2, self.height // 2 - 18))
            screen.blit(text, rect)
            if self._small_font:
                detail = self._small_font.render(
                    f"Score {self.score} | Crystals left {len(self.crystals)}",
                    True,
                    (230, 230, 230),
                )
                detail_rect = detail.get_rect(
                    center=(self.width // 2, self.height // 2 + 18)
                )
                screen.blit(detail, detail_rect)

    def render_title_screen(self, screen) -> None:
        """Render a Crystal Caves-style title and instruction screen."""
        art = self._art or CrystalCavesArt()
        width, height = screen.get_size()
        palette = self._episode_palette()
        screen.fill((0, 0, 0))

        for y in range(0, height, self.TILE_SIZE):
            for x in range(0, width, self.TILE_SIZE):
                edge = x < 64 or x > width - 96 or y < 48 or y > height - 92
                if not edge and (x // 32 + y // 32) % 5:
                    continue
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(screen, palette["rock_dark"], rect)
                pygame.draw.rect(screen, palette["rock"], rect.inflate(-3, -3))
                pygame.draw.line(
                    screen,
                    palette["rock_mid"],
                    (rect.x + 5, rect.y + 20),
                    (rect.x + 24, rect.y + 11),
                    2,
                )
                if (x + y) % 96 == 0:
                    pygame.draw.rect(
                        screen, palette["rock_light"], (rect.x + 10, rect.y + 7, 4, 4)
                    )

        title = "CRYSTAL CAVES"
        title_surface = art.text(title, EGA["C"], scale=5)
        title_x = (width - title_surface.get_width()) // 2
        art.draw_text(screen, title, title_x, 78, EGA["C"], scale=5)
        art.draw_text(
            screen,
            "TROUBLE WITH TWINKLES",
            (width - art.text("TROUBLE WITH TWINKLES", EGA["Y"], scale=2).get_width())
            // 2,
            136,
            EGA["Y"],
            scale=2,
        )

        panel = pygame.Rect(104, 184, width - 208, 230)
        pygame.draw.rect(screen, EGA["K"], panel.inflate(8, 8))
        pygame.draw.rect(screen, (6, 6, 24), panel)
        pygame.draw.rect(screen, EGA["w"], panel, 2)
        pygame.draw.rect(screen, EGA["G"], (panel.x, panel.y, panel.w, 5))
        pygame.draw.rect(screen, EGA["G"], (panel.x, panel.bottom - 5, panel.w, 5))

        art.draw_sprite(screen, "mylo_shoot", panel.x + 28, panel.y + 36, scale=3)
        art.draw_sprite(screen, "bat_enemy", panel.x + 32, panel.y + 142, scale=2)
        art.draw_sprite(screen, "crystal_blue", panel.x + 136, panel.y + 38, scale=3)
        art.draw_sprite(screen, "door_locked", panel.x + 146, panel.y + 130, scale=2)

        instructions = (
            ("COLLECT EVERY CRYSTAL", EGA["C"]),
            ("FIND SWITCHES AND ELEVATORS", EGA["G"]),
            ("Z FIRES  E USES  SPACE JUMPS", EGA["Y"]),
            ("ARROWS OR A D MOVE MYLO", EGA["W"]),
        )
        art.draw_text(screen, "MISSION", panel.x + 256, panel.y + 22, EGA["Y"], scale=2)
        for index, (line, color) in enumerate(instructions):
            art.draw_text(
                screen,
                line,
                panel.x + 256,
                panel.y + 58 + index * 25,
                color,
                scale=1,
            )

        score_panel = pygame.Rect(panel.right - 150, panel.y + 26, 118, 166)
        pygame.draw.rect(screen, EGA["K"], score_panel.inflate(4, 4))
        pygame.draw.rect(screen, (18, 28, 54), score_panel)
        pygame.draw.rect(screen, EGA["w"], score_panel, 1)
        art.draw_text(
            screen,
            "HIGH SCORES",
            score_panel.x + 8,
            score_panel.y + 10,
            EGA["Y"],
            scale=1,
        )
        for index, (name, score) in enumerate(
            (("MYLO", "12500"), ("NOVA", "09000"), ("BOT", "05000"))
        ):
            y = score_panel.y + 38 + index * 31
            art.draw_text(screen, name, score_panel.x + 10, y, EGA["G"], scale=1)
            art.draw_text(screen, score, score_panel.x + 64, y, EGA["C"], scale=1)

        menu_panel = pygame.Rect(panel.x + 248, panel.bottom - 48, 190, 27)
        pygame.draw.rect(screen, EGA["K"], menu_panel.inflate(3, 3))
        pygame.draw.rect(screen, (16, 18, 42), menu_panel)
        pygame.draw.rect(screen, EGA["G"], menu_panel, 1)
        art.draw_text(
            screen,
            "1 START  2 EPISODE  3 HELP",
            menu_panel.x + 7,
            menu_panel.y + 8,
            EGA["G"],
            scale=1,
        )

        episodes = ("EP1 TWINKLES", "EP2 SLUGS", "EP3 SUPERNOVA")
        for index, episode in enumerate(episodes):
            box = pygame.Rect(132 + index * 178, 436, 158, 54)
            selected = index == self.level_index % len(self.CAVES)
            pygame.draw.rect(screen, EGA["K"], box.inflate(4, 4))
            pygame.draw.rect(screen, (16, 18, 42), box)
            pygame.draw.rect(screen, EGA["Y"] if selected else EGA["w"], box, 2)
            art.draw_text(
                screen,
                episode,
                box.x + 11,
                box.y + 18,
                EGA["Y"] if selected else EGA["w"],
                scale=1,
            )

        art.draw_text(
            screen,
            "PRESS ANY KEY",
            (width - art.text("PRESS ANY KEY", EGA["G"], scale=2).get_width()) // 2,
            height - 62,
            EGA["G"],
            scale=2,
        )

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
        jump = bool(
            keys.get(pygame.K_SPACE) or keys.get(pygame.K_UP) or keys.get(pygame.K_w)
        )
        shoot = bool(
            keys.get(pygame.K_z) or keys.get(pygame.K_LCTRL) or keys.get(pygame.K_RCTRL)
        )
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

    def _decode_action(self, action: int) -> Tuple[int, bool, bool, bool]:
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

    def _apply_player_input(self, move_dir: int, wants_jump: bool) -> None:
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

    def _move_player(self) -> None:
        was_airborne = not self.grounded
        falling_speed = abs(self.vy)
        self._move_axis(self.vx, 0.0)
        self._move_axis(0.0, self.vy)
        self.grounded = self._is_on_surface()
        if was_airborne and self.grounded and falling_speed > 3.0:
            self.audio.play("land")

    def _move_axis(self, dx: float, dy: float) -> None:
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

    def _try_shoot(self) -> float:
        if self.shoot_cooldown > 0 or self.ammo <= 0:
            return -0.03

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

        return -0.01

    def _try_interact(self) -> float:
        reward = 0.0
        player_col, player_row = self._player_tile()
        for switch in self.switches:
            col, row = switch
            if abs(col - player_col) <= 1 and abs(row - player_row) <= 1:
                if switch not in self.used_switches:
                    self.used_switches.add(switch)
                    self.doors_open = True
                    self.score += 250
                    reward += 3.0
                    self._add_tile_event(
                        switch, "score", "+250", EGA["G"], ttl=34, y_offset=-10
                    )
                    for door in self.doors:
                        self._add_tile_event(door, "sparkle", "OPEN", EGA["G"], ttl=42)
                    self._mark_progress()
                    self.audio.play("switch")
                else:
                    reward += 0.05
                break
        return reward

    def _collect_pickups(self) -> float:
        reward = 0.0
        touched_tiles = self._tiles_for_rect(self._player_rect())

        for tile in list(self.crystals.intersection(touched_tiles)):
            self.crystals.remove(tile)
            self.score += 100
            reward += 5.0
            self._add_tile_event(tile, "sparkle", "+100", EGA["Y"], ttl=34)
            self._mark_progress()
            self.audio.play("gem")

        if not self.crystals and not self.exit_unlocked:
            self.exit_unlocked = True
            self.score += 500
            reward += 10.0
            self._add_tile_event(
                self.exit_pos, "sparkle", "EXIT OPEN", EGA["G"], ttl=58
            )
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

    def _update_bullets(self) -> float:
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

    def _update_enemies(self) -> float:
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
                if self._rect_collides_solid(enemy.rect) or not self._solid_at(
                    ahead_col, foot_row
                ):
                    enemy.x -= enemy.vx
                    enemy.vx *= -1

        return reward

    def _check_player_danger(self) -> float:
        reward = 0.0
        player_rect = self._player_rect()
        danger = False

        for tile in self.hazards:
            if player_rect.colliderect(self._tile_rect(tile)):
                danger = True
                break

        if not danger:
            for enemy in self.enemies:
                if enemy.alive and player_rect.colliderect(enemy.rect):
                    danger = True
                    break

        if danger:
            reward += self._damage_player()

        return reward

    def _damage_player(self) -> float:
        if self.invuln_timer > 0:
            return 0.0

        self.health -= 1
        self.invuln_timer = self.INVULN_FRAMES
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
            self.audio.play("lose")
            return -12.0

        self.audio.play("damage")
        return -3.0

    def _check_exit(self) -> float:
        if not self.exit_unlocked:
            return 0.0

        exit_rect = self._tile_rect(self.exit_pos).inflate(-6, -2)
        if self._player_rect().colliderect(exit_rect):
            self.game_over = True
            self.won = True
            self.score += 1000 + self.health * 250 + self.ammo * 10
            self.level_index = (self.level_index + 1) % len(self.CAVES)
            self.audio.play("door")
            return 25.0

        return 0.0

    def _current_target(self) -> Tuple[Optional[Tuple[str, int, int]], float]:
        player_x, player_y = self._player_center()
        candidates: List[Tuple[str, int, int]] = []

        unused_switches = self.switches - self.used_switches
        if self.exit_unlocked and not self.doors_open and unused_switches:
            candidates = [("switch", col, row) for col, row in unused_switches]
        elif self.crystals:
            candidates = [("crystal", col, row) for col, row in self.crystals]
        elif self.exit_unlocked:
            candidates = [("exit", self.exit_pos[0], self.exit_pos[1])]
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

    def _target_features(self) -> Tuple[float, float, float, float]:
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
        self,
        previous_target: Optional[Tuple[str, int, int]],
        previous_distance: float,
    ) -> float:
        if self.game_over or previous_target is None:
            return 0.0

        current_target, current_distance = self._current_target()
        if current_target != previous_target or not np.isfinite(previous_distance):
            return 0.0

        tile_progress = (previous_distance - current_distance) / self.TILE_SIZE
        if tile_progress > 0.03:
            self._mark_progress()
        return float(np.clip(tile_progress * 0.08, -0.03, 0.06))

    def _mark_progress(self) -> None:
        self.steps_since_progress = 0

    def _add_visual_event(
        self,
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
        self,
        tile: Tuple[int, int],
        kind: str,
        text: str = "",
        color: Tuple[int, int, int] = (255, 255, 255),
        ttl: int = 24,
        y_offset: float = 0.0,
    ) -> None:
        x, y = self._tile_center(tile)
        self._add_visual_event(kind, x, y + y_offset, ttl, text, color)

    def _update_visual_events(self) -> None:
        if not self.visual_events:
            return
        updated = []
        for event in self.visual_events:
            event.ttl -= 1
            if event.ttl > 0:
                updated.append(event)
        self.visual_events = updated

    def _air_tank_for_rect(self, rect: pygame.Rect) -> Optional[Tuple[int, int]]:
        for tank in self.air_tanks:
            if rect.colliderect(self._tile_rect(tank).inflate(-8, -6)):
                return tank
        return None

    def _damage_from_air_tank(self, tank: Tuple[int, int]) -> None:
        tank_x, tank_y = self._tile_center(tank)
        player_x, player_y = self._player_center()
        if np.hypot(tank_x - player_x, tank_y - player_y) <= self.TILE_SIZE * 2.2:
            self._damage_player()

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
        }

    def _player_rect(
        self, x: Optional[float] = None, y: Optional[float] = None
    ) -> pygame.Rect:
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
        return {
            (col, row)
            for row in range(top, bottom + 1)
            for col in range(left, right + 1)
        }

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
                return self.TILE_CODES[
                    self.FLYER if enemy.kind == "flyer" else self.CRAWLER
                ]

        if self._solid_at(col, row):
            return (
                self.TILE_CODES[self.DOOR]
                if tile in self.doors
                else self.TILE_CODES[self.SOLID]
            )
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

    def _camera(self) -> Tuple[int, int]:
        visible_height = self.height - self.HUD_HEIGHT
        target_x = self.player_x + self.PLAYER_WIDTH / 2 - self.width / 2
        target_y = self.player_y + self.PLAYER_HEIGHT / 2 - visible_height / 2
        camera_x = int(np.clip(target_x, 0, max(0, self.level_width - self.width)))
        camera_y = int(np.clip(target_y, 0, max(0, self.level_height - visible_height)))
        return camera_x, camera_y

    def _world_to_screen(
        self, x: float, y: float, camera_x: int, camera_y: int
    ) -> Tuple[int, int]:
        return int(x - camera_x), int(y - camera_y)

    def _draw_background(self, screen, camera_x: int, camera_y: int) -> None:
        play_bottom = self.height - self.HUD_HEIGHT
        palette = self._episode_palette()
        # Dense theme-colored cave back-wall fill replaces the old black void so
        # terrain reads as carved out of a cave room (backlog V001/V019).
        self._draw_wall_fill(screen, camera_x, camera_y, play_bottom, palette)
        self._draw_cave_depth(screen, camera_x, camera_y, play_bottom, palette)

        # Later episodes keep sparse background machinery. Episode 1 now relies
        # on authored pipes and props so the first screen does not become a grid.
        if self.level_index % len(self.CAVES) != 0:
            for x in range(76 - (camera_x // 5) % 260, self.width + 96, 260):
                pygame.draw.rect(
                    screen, palette["pipe_shadow"], (x + 4, 0, 7, play_bottom)
                )
                pygame.draw.rect(screen, palette["pipe_dark"], (x, 0, 7, play_bottom))
                pygame.draw.line(
                    screen,
                    palette["pipe_light"],
                    (x + 2, 0),
                    (x + 2, play_bottom),
                    1,
                )
                for y in range(38 - (camera_y // 6) % 128, play_bottom + 36, 128):
                    pygame.draw.circle(screen, palette["pipe_light"], (x + 3, y), 4)

            for y in range(42 - (camera_y // 5) % 172, play_bottom, 172):
                for x in range(24 - (camera_x // 6) % 248, self.width + 80, 248):
                    segment = pygame.Rect(x, y, 86, 5)
                    pygame.draw.rect(screen, palette["pipe_shadow"], segment.move(4, 4))
                    pygame.draw.rect(screen, palette["pipe_dark"], segment)
                    pygame.draw.line(
                        screen,
                        palette["pipe_light"],
                        (segment.left, segment.y + 1),
                        (segment.right, segment.y + 1),
                        1,
                    )
                    pygame.draw.circle(
                        screen, palette["pipe_light"], (segment.left + 8, y + 2), 3
                    )
                    pygame.draw.circle(
                        screen, palette["pipe_light"], (segment.right - 8, y + 2), 3
                    )

        for i in range(8):
            panel_x = (i * 173 - camera_x // 4) % (self.width + 96) - 48
            panel_y = (i * 89 - camera_y // 5) % max(1, play_bottom - 80)
            panel = pygame.Rect(panel_x, panel_y, 46, 30)
            pygame.draw.rect(screen, (0, 0, 0), panel.inflate(4, 4))
            pygame.draw.rect(screen, (18, 28, 54), panel)
            pygame.draw.rect(screen, palette["pipe_dark"], panel, 2)
            pygame.draw.rect(
                screen,
                palette["pipe_light"],
                (panel.x + 6, panel.y + 8, 10, 4),
            )
            pygame.draw.rect(
                screen,
                palette["spark"],
                (panel.x + 25, panel.y + 17, 12, 4),
            )

    def _draw_wall_fill(
        self,
        screen,
        camera_x: int,
        camera_y: int,
        play_bottom: int,
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        """Paint a dense, dim, theme-colored masonry back-wall behind gameplay.

        This is the single biggest visual-identity change: the play area is no
        longer a black void with floating platforms, but a carved cave room.
        Tones sit well below the foreground palette so solid tiles, bright lips
        and bolts still pop in front of the fill (backlog V001/V004/V019/V082).
        """

        def dim(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
            return (
                int(color[0] * factor),
                int(color[1] * factor),
                int(color[2] * factor),
            )

        style = self.level_index % len(self.CAVES)
        # Dedicated back-wall role colors (CCV-02) keep the fill intentional and
        # clearly dimmer than the foreground rock palette.
        mortar = palette["edge_dark"]
        block = palette["wall_fill"]
        block_dark = dim(palette["wall_fill"], 0.66)
        block_hi = dim(palette["wall_accent"], 0.82)
        speck = dim(palette["ledge_lip"], 0.70)

        # Flat mortar base so any sub-pixel seam never reveals pure black.
        pygame.draw.rect(screen, mortar, (0, 0, self.width, play_bottom))

        ts = self.TILE_SIZE
        # Parallax: the wall scrolls at half camera speed for a sense of depth.
        px = camera_x // 2
        py = camera_y // 2
        first_col = px // ts - 1
        last_col = (px + self.width) // ts + 1
        first_row = py // ts - 1
        last_row = (py + play_bottom) // ts + 1

        for row in range(first_row, last_row + 1):
            # Masonry stagger: every other course shifts half a tile.
            stagger = (ts // 2) if (row & 1) else 0
            for col in range(first_col, last_col + 1):
                sx = col * ts - px + stagger
                sy = row * ts - py
                if sx > self.width or sx < -ts or sy > play_bottom or sy < -ts:
                    continue
                seed = (col * 73 + row * 151 + style * 17) & 0xFFFF
                body = pygame.Rect(sx + 1, sy + 1, ts - 2, ts - 2)
                tone = block if (seed % 5) else block_dark
                pygame.draw.rect(screen, tone, body, border_radius=6)
                # Top-left bevel highlight + bottom-right shadow = rounded relief.
                pygame.draw.line(
                    screen, block_hi,
                    (body.left + 3, body.top + 2),
                    (body.right - 6, body.top + 2), 2,
                )
                pygame.draw.line(
                    screen, block_hi,
                    (body.left + 2, body.top + 3),
                    (body.left + 2, body.bottom - 6), 2,
                )
                pygame.draw.line(
                    screen, mortar,
                    (body.left + 4, body.bottom - 2),
                    (body.right - 2, body.bottom - 2), 2,
                )
                # Occasional embedded crystal speck or pit for organic variation.
                if seed % 9 == 0:
                    pygame.draw.rect(
                        screen, speck, (body.centerx - 2, body.centery - 2, 4, 4)
                    )
                elif seed % 13 == 0:
                    pygame.draw.rect(
                        screen, mortar, (body.centerx, body.centery, 3, 3)
                    )

    def _draw_cave_depth(
        self,
        screen,
        camera_x: int,
        camera_y: int,
        play_bottom: int,
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        """Paint distant cave cuts and machinery shadows behind gameplay tiles."""
        depth = pygame.Surface((self.width, play_bottom), pygame.SRCALPHA)
        style = self.level_index % len(self.CAVES)
        # Boosted alphas (CCV-01) so carved recesses read against the dim wall
        # fill instead of being black-on-black texture.
        alpha = 86 if style == 0 else 68
        dark = (*palette["edge_dark"], alpha)
        mid = (*palette["wall_accent"], 60)
        accent = (*palette["pipe_light"], 74 if style else 58)
        rim = (*palette["rock_light"], 70)

        for i in range(6):
            base_x = (i * 211 - camera_x // 7) % (self.width + 180) - 90
            base_y = 40 + ((i * 83 - camera_y // 8) % max(1, play_bottom - 120))
            points = [
                (base_x, base_y + 72),
                (base_x + 34, base_y + 18),
                (base_x + 91, base_y),
                (base_x + 138, base_y + 37),
                (base_x + 154, base_y + 96),
                (base_x + 54, base_y + 116),
            ]
            pygame.draw.polygon(depth, dark, points)
            pygame.draw.lines(depth, mid, False, points, 2)
            # A brighter rim on every third recess gives a clear carved silhouette.
            if i % 3 == 0:
                pygame.draw.lines(depth, rim, True, points, 2)
            if i % 2 == 0:
                pygame.draw.rect(depth, accent, (base_x + 42, base_y + 45, 54, 5))
                pygame.draw.rect(depth, accent, (base_x + 48, base_y + 62, 5, 36))

        for i in range(4):
            x = (i * 257 - camera_x // 9) % (self.width + 160) - 80
            y = 70 + ((i * 121 - camera_y // 10) % max(1, play_bottom - 150))
            pygame.draw.rect(depth, (*palette["pipe_shadow"], 60), (x, y, 78, 42))
            pygame.draw.rect(depth, (*palette["rock_mid"], 45), (x + 8, y + 9, 28, 5))
            pygame.draw.rect(depth, (*palette["spark"], 45), (x + 51, y + 25, 12, 4))

        screen.blit(depth, (0, 0))

    def _episode_palette(self) -> Dict[str, Tuple[int, int, int]]:
        palettes = (
            {
                "rock_dark": (0, 0, 72),
                "rock": (18, 34, 150),
                "rock_mid": (38, 72, 204),
                "rock_light": (106, 150, 255),
                "platform_dark": (0, 50, 64),
                "platform": (28, 84, 118),
                "platform_light": (92, 232, 162),
                "pipe_dark": (0, 76, 34),
                "pipe_light": (34, 238, 80),
                "pipe_shadow": (0, 28, 10),
                "spark": (94, 120, 255),
                "wall_fill": (8, 28, 80),
                "wall_accent": (28, 72, 150),
                "ledge_lip": (106, 150, 255),
                "edge_dark": (0, 0, 40),
                "grass": (96, 220, 132),
            },
            {
                "rock_dark": (70, 24, 0),
                "rock": (150, 54, 18),
                "rock_mid": (204, 88, 28),
                "rock_light": (255, 158, 52),
                "platform_dark": (0, 70, 20),
                "platform": (22, 164, 48),
                "platform_light": (180, 255, 72),
                "pipe_dark": (86, 54, 0),
                "pipe_light": (255, 190, 52),
                "pipe_shadow": (40, 20, 0),
                "spark": (255, 190, 72),
                "wall_fill": (64, 32, 16),
                "wall_accent": (140, 70, 28),
                "ledge_lip": (255, 96, 72),
                "edge_dark": (40, 16, 0),
                "grass": (150, 230, 64),
            },
            {
                "rock_dark": (40, 40, 48),
                "rock": (92, 96, 108),
                "rock_mid": (138, 144, 162),
                "rock_light": (218, 226, 255),
                "platform_dark": (36, 50, 74),
                "platform": (92, 108, 170),
                "platform_light": (210, 224, 255),
                "pipe_dark": (54, 38, 80),
                "pipe_light": (194, 128, 255),
                "pipe_shadow": (22, 18, 36),
                "spark": (188, 220, 255),
                "wall_fill": (40, 48, 64),
                "wall_accent": (90, 110, 140),
                "ledge_lip": (120, 240, 255),
                "edge_dark": (20, 24, 34),
            },
        )
        return palettes[self.level_index % len(palettes)]

    def _draw_ledge_growth(
        self,
        screen,
        rect: pygame.Rect,
        col: int,
        row: int,
        palette: Dict[str, Tuple[int, int, int]],
        under_open: bool,
    ) -> None:
        """Bright moss fringe + tufts on a walkable ledge, with hanging vines
        under exposed edges (CCV-17). Only natural-cave themes define ``grass``;
        industrial/tech episodes skip it for a clean metal look.
        """
        grass = palette.get("grass")
        if grass is None:
            return
        grass_dark = (grass[0] * 6 // 10, grass[1] * 6 // 10, grass[2] * 6 // 10)
        seed = col * 31 + row * 17 + self.level_index * 5

        # Mossy fringe sitting on the very top lip.
        pygame.draw.rect(screen, grass_dark, (rect.x, rect.y, rect.w, 4))
        pygame.draw.rect(screen, grass, (rect.x, rect.y, rect.w, 2))

        # Blades poking up above the ledge, with a gentle idle sway.
        for i, x in enumerate(range(rect.left + 2, rect.right - 1, 7)):
            blade = 4 + ((seed + i * 5) % 4)
            sway = -1 if (self.steps // 16 + i) % 2 else 0
            pygame.draw.rect(screen, grass_dark, (x + sway, rect.y - blade, 2, blade))
            pygame.draw.rect(
                screen, grass, (x + sway, rect.y - blade, 1, max(1, blade - 1))
            )

        # Vines drooping from an exposed underside.
        if under_open and seed % 3 == 0:
            for i, x in enumerate(range(rect.left + 5, rect.right - 4, 12)):
                vine = 6 + ((seed + i * 7) % 8)
                pygame.draw.rect(screen, grass_dark, (x, rect.bottom - 1, 2, vine))
                pygame.draw.rect(screen, grass, (x, rect.bottom - 1, 1, vine - 2))
                pygame.draw.rect(
                    screen, grass, (x - 1, rect.bottom - 1 + vine, 3, 2)
                )

    def _draw_solid_tile(self, screen, rect: pygame.Rect, col: int, row: int) -> None:
        palette = self._episode_palette()
        is_surface = not self._solid_at(col, row - 1)
        pygame.draw.rect(screen, (0, 0, 0), rect)

        if is_surface:
            left_edge = not self._solid_at(col - 1, row)
            right_edge = not self._solid_at(col + 1, row)
            under_open = not self._solid_at(col, row + 1)
            seed = col * 19 + row * 23 + self.level_index * 7
            pygame.draw.rect(screen, palette["rock_dark"], rect)
            pygame.draw.rect(screen, palette["platform_dark"], rect.inflate(-1, -1))
            pygame.draw.rect(
                screen, palette["platform"], (rect.x + 1, rect.y, rect.w - 2, 10)
            )
            pygame.draw.rect(screen, EGA["K"], (rect.x, rect.y + 10, rect.w, 3))
            pygame.draw.rect(
                screen,
                palette["platform_light"],
                (rect.x + 1, rect.y + 1, rect.w - 2, 3),
            )
            pygame.draw.line(
                screen,
                EGA["W"] if self.level_index % len(self.CAVES) == 2 else EGA["C"],
                (rect.left + 3, rect.y + 5),
                (rect.right - 4, rect.y + 5),
                2,
            )
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (rect.left, rect.y + 12),
                (rect.right - 1, rect.y + 12),
                2,
            )
            pygame.draw.rect(
                screen, palette["rock_mid"], (rect.x + 2, rect.y + 13, rect.w - 4, 8)
            )
            pygame.draw.rect(
                screen,
                palette["rock_dark"],
                (rect.x + 2, rect.y + 22, rect.w - 4, 8),
            )
            if left_edge:
                pygame.draw.rect(
                    screen, palette["platform_light"], (rect.x, rect.y, 5, 15)
                )
            if right_edge:
                pygame.draw.rect(screen, (0, 0, 0), (rect.right - 5, rect.y, 5, 15))
            for x in range(rect.left + 5, rect.right - 2, 12):
                bolt_color = (
                    palette["platform_light"]
                    if (seed + x) % 2
                    else palette["rock_light"]
                )
                pygame.draw.rect(screen, bolt_color, (x, rect.y + 15, 3, 3))
                if under_open:
                    pygame.draw.rect(screen, (0, 0, 0), (x + 1, rect.y + 23, 2, 7))
            if seed % 5 == 0:
                pygame.draw.rect(
                    screen, palette["pipe_dark"], (rect.x + 7, rect.y + 22, 18, 4)
                )
                pygame.draw.rect(
                    screen, palette["pipe_light"], (rect.x + 9, rect.y + 22, 5, 2)
                )
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)
            self._draw_ledge_growth(screen, rect, col, row, palette, under_open)
            return

        open_left = not self._solid_at(col - 1, row)
        open_right = not self._solid_at(col + 1, row)
        open_top = not self._solid_at(col, row - 1)
        open_bottom = not self._solid_at(col, row + 1)
        seed = col * 17 + row * 31 + self.level_index * 11
        variant = seed % 11

        pygame.draw.rect(screen, palette["rock_dark"], rect)
        inner = rect.inflate(-2, -2)
        pygame.draw.rect(screen, palette["rock"], inner)

        if open_left:
            pygame.draw.rect(screen, palette["rock_light"], (rect.x, rect.y, 4, rect.h))
            pygame.draw.rect(
                screen, palette["rock_mid"], (rect.x + 4, rect.y, 3, rect.h)
            )
        if open_right:
            pygame.draw.rect(screen, (0, 0, 0), (rect.right - 4, rect.y, 4, rect.h))
            pygame.draw.rect(
                screen, palette["rock_mid"], (rect.right - 7, rect.y, 3, rect.h)
            )
        if open_top:
            pygame.draw.rect(screen, palette["rock_light"], (rect.x, rect.y, rect.w, 4))
            pygame.draw.rect(
                screen, palette["rock_mid"], (rect.x, rect.y + 4, rect.w, 3)
            )
        if open_bottom:
            pygame.draw.rect(screen, (0, 0, 0), (rect.x, rect.bottom - 4, rect.w, 4))
            pygame.draw.rect(
                screen, palette["rock_mid"], (rect.x, rect.bottom - 7, rect.w, 3)
            )

        if variant == 0 and not (open_left or open_right or open_top or open_bottom):
            plate = rect.inflate(-8, -8)
            pygame.draw.rect(screen, palette["pipe_shadow"], plate)
            pygame.draw.rect(screen, (110, 116, 132), plate, 2)
            for px, py in (
                (plate.left + 4, plate.top + 4),
                (plate.right - 7, plate.top + 4),
                (plate.left + 4, plate.bottom - 7),
                (plate.right - 7, plate.bottom - 7),
            ):
                pygame.draw.rect(screen, palette["rock_light"], (px, py, 3, 3))
        elif variant == 1 and not open_top:
            pipe_y = rect.y + 13
            pygame.draw.rect(
                screen, palette["pipe_shadow"], (rect.x, pipe_y + 4, rect.w, 7)
            )
            pygame.draw.rect(screen, palette["pipe_dark"], (rect.x, pipe_y, rect.w, 7))
            pygame.draw.line(
                screen,
                palette["pipe_light"],
                (rect.x + 2, pipe_y + 1),
                (rect.right - 3, pipe_y + 1),
                1,
            )
        else:
            veins = (
                ((3 + seed % 8, 6), (16 + seed % 7, 4)),
                ((8, 17 + seed % 6), (25, 21 + seed % 4)),
                ((4 + seed % 5, 26), (14 + seed % 8, 23)),
            )
            for start, end in veins:
                pygame.draw.line(
                    screen,
                    palette["rock_mid"],
                    (rect.x + start[0], rect.y + start[1]),
                    (rect.x + end[0], rect.y + end[1]),
                    2,
                )
            for x_off, y_off in ((5, 5), (21, 11), (12, 25)):
                if (seed + x_off + y_off) % 3:
                    pygame.draw.rect(
                        screen,
                        palette["rock_light"],
                        (rect.x + x_off, rect.y + y_off, 3, 3),
                    )

        if self.level_index % len(self.CAVES) == 1 and variant in (3, 7):
            pygame.draw.rect(screen, (255, 188, 80), (rect.x + 8, rect.y + 9, 5, 5))
            pygame.draw.rect(screen, (150, 54, 18), (rect.x + 16, rect.y + 18, 7, 4))
        elif self.level_index % len(self.CAVES) == 2 and variant in (2, 8):
            pygame.draw.rect(screen, (188, 220, 255), (rect.x + 7, rect.y + 7, 4, 4))
            pygame.draw.line(
                screen,
                (194, 128, 255),
                (rect.x + 11, rect.y + 21),
                (rect.x + 24, rect.y + 13),
                1,
            )

    def _draw_spike_tile(self, screen, rect: pygame.Rect, col: int, row: int) -> None:
        phase = (self.steps + col * 5 + row * 3) % 36
        # Full-tile hazard volume (CCV-06): dark base, warning crust, six teeth
        # spanning most of the tile, framed by a black outline like solid tiles.
        pygame.draw.rect(screen, (44, 8, 8), rect)
        pygame.draw.rect(screen, (96, 0, 0), (rect.x, rect.y + 16, rect.w, rect.h - 16))
        pygame.draw.line(
            screen, EGA["Y"], (rect.left, rect.y + 15), (rect.right, rect.y + 15), 2
        )
        teeth = 6
        step = rect.w / teeth
        for i in range(teeth):
            x = rect.x + int(i * step)
            jut = 1 if (phase < 6 and i == phase % teeth) else 0
            tip_x = x + int(step / 2)
            tip_y = rect.y + 3 + jut
            base_y = rect.bottom - 2
            pygame.draw.polygon(
                screen,
                EGA["K"],
                [(x - 1, base_y), (tip_x, tip_y - 2), (x + int(step) + 1, base_y)],
            )
            pygame.draw.polygon(
                screen,
                EGA["W"],
                [(x + 1, base_y), (tip_x, tip_y), (x + int(step) - 1, base_y)],
            )
            pygame.draw.line(
                screen, EGA["w"], (tip_x, tip_y + 3), (tip_x, base_y - 3), 1
            )
        pygame.draw.rect(screen, EGA["K"], rect, 2)

    def _draw_acid_tile(self, screen, rect: pygame.Rect, col: int, row: int) -> None:
        wave = (self.steps // 4 + col * 3) % 8
        # Full-tile molten pool (CCV-07): maroon body, bright animated crust,
        # rising bubbles, framed by a black outline.
        pygame.draw.rect(screen, (28, 0, 0), rect)
        pygame.draw.rect(screen, (130, 0, 0), (rect.x, rect.y + 8, rect.w, rect.h - 8))
        pygame.draw.rect(screen, EGA["t"], (rect.x, rect.y + 12, rect.w, rect.h - 12))
        for i in range(-4, rect.w + 4, 7):
            x = rect.x + i
            y = rect.y + 8 + ((i + wave) % 4)
            pygame.draw.rect(screen, EGA["O"], (x, y, 6, 4))
            pygame.draw.rect(screen, EGA["L"], (x + 1, y, 3, 2))
        for i in range(3):
            bubble_x = rect.x + 4 + ((self.steps + col * 11 + i * 13) % (rect.w - 8))
            bubble_y = rect.y + 16 + ((self.steps // 3 + row + i * 5) % (rect.h - 18))
            pygame.draw.circle(screen, EGA["Y"], (bubble_x, bubble_y), 2)
        pygame.draw.rect(screen, EGA["K"], rect, 2)

    def _draw_locked_door(self, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "door_locked", rect.x, rect.y, scale=2)
            return
        pygame.draw.rect(screen, (8, 8, 16), rect)
        pygame.draw.rect(screen, (160, 0, 0), rect.inflate(-8, -2))
        pygame.draw.rect(screen, (255, 78, 78), rect.inflate(-8, -2), 2)
        pygame.draw.rect(screen, (210, 210, 220), (rect.x + 11, rect.y + 6, 10, 20))
        pygame.draw.rect(screen, (0, 0, 0), (rect.x + 14, rect.y + 9, 4, 14))

    def _draw_exit_airlock(self, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(
                screen,
                "door_open" if self.exit_unlocked else "door_locked",
                rect.x,
                rect.y,
                scale=2,
            )
            if self.exit_unlocked:
                pulse = 1 + (self.steps // 8) % 3
                pygame.draw.rect(screen, (98, 255, 98), rect.inflate(pulse, pulse), 2)
            return
        glow = (70, 255, 82) if self.exit_unlocked else (120, 120, 130)
        door = (0, 112, 44) if self.exit_unlocked else (68, 68, 82)
        pygame.draw.rect(screen, (0, 0, 0), rect)
        pygame.draw.rect(screen, (210, 210, 220), rect.inflate(-2, 0), 2)
        pygame.draw.rect(
            screen, (118, 118, 132), (rect.x + 3, rect.y + 2, 5, rect.h - 4)
        )
        pygame.draw.rect(screen, door, rect.inflate(-10, -6))
        pygame.draw.rect(screen, glow, rect.inflate(-16, -14))
        pygame.draw.rect(screen, (0, 0, 0), rect.inflate(-21, -21))
        pygame.draw.rect(screen, (255, 255, 255), (rect.x + 7, rect.y + 5, 3, 4))
        if self._tiny_font:
            label = self._tiny_font.render("EXIT", True, glow)
            screen.blit(label, label.get_rect(center=(rect.centerx, rect.y + 6)))

    def _draw_switch_wires(self, screen, camera_x: int, camera_y: int) -> None:
        """Draw a taut cable from each switch to the nearest door it controls
        (CCV-18). The core glows green once thrown, amber while still armed, so
        the switch->target relationship is readable like the DOS references.
        """
        if not self.switches or not self.doors:
            return
        half = self.TILE_SIZE // 2
        for switch in self.switches:
            sc, sr = switch
            door = min(
                self.doors,
                key=lambda d: (d[0] - sc) ** 2 + (d[1] - sr) ** 2,
            )
            sx, sy = self._world_to_screen(
                sc * self.TILE_SIZE + half, sr * self.TILE_SIZE + half,
                camera_x, camera_y,
            )
            dx, dy = self._world_to_screen(
                door[0] * self.TILE_SIZE + half, door[1] * self.TILE_SIZE + half,
                camera_x, camera_y,
            )
            core = EGA["G"] if switch in self.used_switches else EGA["A"]
            pygame.draw.line(screen, EGA["K"], (sx, sy), (dx, dy), 3)
            pygame.draw.line(screen, core, (sx, sy), (dx, dy), 1)
            # Small anchor bolts at each end seat the cable into the world.
            pygame.draw.circle(screen, EGA["m"], (sx, sy), 2)
            pygame.draw.circle(screen, EGA["m"], (dx, dy), 2)

    def _draw_lever_switch(self, screen, rect: pygame.Rect, used: bool) -> None:
        if self._art:
            self._art.draw_sprite(
                screen,
                "switch_on" if used else "switch_off",
                rect.x,
                rect.y,
                scale=2,
            )
            return
        pygame.draw.rect(screen, (0, 0, 0), rect)
        pygame.draw.rect(screen, (166, 166, 176), (rect.x + 7, rect.y + 20, 18, 8))
        base_color = (70, 255, 80) if used else (255, 80, 48)
        pivot = (rect.x + 16, rect.y + 20)
        lever_end = (rect.x + 23, rect.y + 8) if used else (rect.x + 9, rect.y + 7)
        pygame.draw.line(screen, (230, 230, 230), pivot, lever_end, 3)
        pygame.draw.circle(screen, base_color, lever_end, 5)
        pygame.draw.rect(screen, (0, 0, 0), (rect.x + 9, rect.y + 24, 14, 3))

    def _draw_tiles(self, screen, camera_x: int, camera_y: int) -> None:
        first_col = max(0, camera_x // self.TILE_SIZE)
        last_col = min(self.level_cols, (camera_x + self.width) // self.TILE_SIZE + 2)
        first_row = max(0, camera_y // self.TILE_SIZE)
        last_row = min(
            self.level_rows,
            (camera_y + self.height - self.HUD_HEIGHT) // self.TILE_SIZE + 2,
        )

        for row in range(first_row, last_row):
            for col in range(first_col, last_col):
                x, y = self._world_to_screen(
                    col * self.TILE_SIZE,
                    row * self.TILE_SIZE,
                    camera_x,
                    camera_y,
                )
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                tile = (col, row)

                if self.grid[row][col] == self.SOLID:
                    self._draw_solid_tile(screen, rect, col, row)
                elif tile in self.hazards:
                    if self.hazard_kinds.get(tile) == self.ACID:
                        self._draw_acid_tile(screen, rect, col, row)
                    else:
                        self._draw_spike_tile(screen, rect, col, row)
                elif tile in self.doors and not self.doors_open:
                    self._draw_locked_door(screen, rect)
                elif tile == self.exit_pos:
                    self._draw_exit_airlock(screen, rect)
                elif tile in self.switches:
                    self._draw_lever_switch(screen, rect, tile in self.used_switches)

    def _draw_level_dressing(self, screen, camera_x: int, camera_y: int) -> None:
        first_col = max(0, camera_x // self.TILE_SIZE)
        last_col = min(self.level_cols, (camera_x + self.width) // self.TILE_SIZE + 2)
        first_row = max(0, camera_y // self.TILE_SIZE)
        last_row = min(
            self.level_rows,
            (camera_y + self.height - self.HUD_HEIGHT) // self.TILE_SIZE + 2,
        )

        # Vertical rails hanging from platforms.
        for col in range(first_col, last_col):
            for row in range(first_row, last_row):
                if not self._should_draw_support_rail(col, row):
                    continue
                x, y = self._world_to_screen(
                    col * self.TILE_SIZE,
                    row * self.TILE_SIZE,
                    camera_x,
                    camera_y,
                )
                self._draw_support_rail(
                    screen, pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                )

        self._draw_authored_dressing(
            screen, camera_x, camera_y, first_col, last_col, first_row, last_row
        )

        for tile in self._visible_tiles(
            self.hazards, first_col, last_col, first_row, last_row
        ):
            if self.hazard_kinds.get(tile) == self.ACID:
                if not self._same_hazard_at(tile[0] - 1, tile[1], self.ACID):
                    self._draw_sign_for_tile(
                        screen, camera_x, camera_y, tile, "ACID", (255, 230, 64)
                    )
            elif not self._same_hazard_at(tile[0] - 1, tile[1], self.SPIKE):
                self._draw_sign_for_tile(
                    screen, camera_x, camera_y, tile, "DANGER", (255, 72, 72)
                )

        for tile, power in self.powerups.items():
            if not (
                first_col <= tile[0] < last_col and first_row <= tile[1] < last_row
            ):
                continue
            label = {
                self.POWER_SHOT: "POWER",
                self.GRAVITY_POWER: "LOW G",
                self.FREEZE_POWER: "STOP",
            }[power]
            self._draw_sign_for_tile(
                screen, camera_x, camera_y, tile, label, (255, 216, 64)
            )

        for tank in self._visible_tiles(
            self.air_tanks, first_col, last_col, first_row, last_row
        ):
            self._draw_sign_for_tile(
                screen, camera_x, camera_y, tank, "AIR", (88, 240, 255)
            )

        if self.level_index % len(self.CAVES) != 0:
            # Mine props on empty floor-adjacent cells. Deterministic so
            # screenshots and tests remain stable.
            for row in range(first_row, last_row):
                for col in range(first_col, last_col):
                    if not self._empty_dressing_cell(col, row):
                        continue
                    selector = (col * 13 + row * 17 + self.level_index * 5) % 37
                    x, y = self._world_to_screen(
                        col * self.TILE_SIZE,
                        row * self.TILE_SIZE,
                        camera_x,
                        camera_y,
                    )
                    rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                    if selector == 0:
                        self._draw_crate(screen, rect)
                    elif selector == 7:
                        self._draw_pickaxe(screen, rect)
                    elif selector == 13:
                        self._draw_lamp(screen, rect)

    @staticmethod
    def _visible_tiles(
        tiles: Set[Tuple[int, int]],
        first_col: int,
        last_col: int,
        first_row: int,
        last_row: int,
    ) -> List[Tuple[int, int]]:
        return [
            tile
            for tile in tiles
            if first_col <= tile[0] < last_col and first_row <= tile[1] < last_row
        ]

    def _draw_authored_dressing(
        self,
        screen,
        camera_x: int,
        camera_y: int,
        first_col: int,
        last_col: int,
        first_row: int,
        last_row: int,
    ) -> None:
        pieces = self.CAVE_DRESSING.get(self.level_index % len(self.CAVES), ())
        for piece in pieces:
            length = self._dressing_length(piece)
            if (
                piece.col + length < first_col
                or piece.col >= last_col
                or piece.row < first_row - 1
                or piece.row >= last_row + 1
            ):
                continue
            x, y = self._world_to_screen(
                piece.col * self.TILE_SIZE,
                piece.row * self.TILE_SIZE,
                camera_x,
                camera_y,
            )
            rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
            self._draw_dressing_piece(screen, rect, piece)

    @staticmethod
    def _dressing_length(piece: DressingPiece) -> int:
        if piece.kind not in {"cable_h", "clear_blocks"}:
            return 1
        try:
            return max(1, int(piece.label))
        except ValueError:
            return 1

    def _draw_dressing_piece(
        self, screen, rect: pygame.Rect, piece: DressingPiece
    ) -> None:
        if piece.kind == "cable_h":
            self._draw_cable_run(screen, rect, self._dressing_length(piece))
            return
        if piece.kind == "elevator_frame":
            self._draw_elevator_frame(screen, rect, piece.label)
            return
        if piece.kind == "crystal_light":
            self._draw_crystal_light(screen, rect)
            return
        if piece.kind == "clear_blocks":
            self._draw_clear_block_run(screen, rect, self._dressing_length(piece))
            return
        if piece.kind == "room_label":
            self._draw_room_label(screen, rect, piece.label)
            return

        sprite_name = {
            "beacon": "beacon",
            "mine_sign": "mine_sign",
            "generator": "generator",
            "terminal": "terminal",
            "pipe_stack": "pipe_stack",
            "warning_post": "warning_post",
            "mushroom": "mushroom",
            "hammer_marker": "hammer_marker",
            "zapper": "zapper",
            "vacuum": "vacuum",
            "eye_turret": "eye_turret",
            "slug_enemy": "slug_enemy",
            "bat_perch": "bat_enemy",
        }.get(piece.kind)
        if sprite_name and self._art:
            self._art.draw_sprite(screen, sprite_name, rect.x, rect.y, scale=2)
            if piece.kind == "mine_sign":
                label = piece.label or "MINE"
                self._art.draw_text(
                    screen,
                    label[:4],
                    rect.x + 5,
                    rect.y + 7,
                    EGA["K"],
                    scale=1,
                    shadow=False,
                )
            return

        pygame.draw.rect(screen, (0, 0, 0), rect.inflate(-4, -8))
        pygame.draw.rect(screen, (116, 116, 132), rect.inflate(-8, -12))
        pygame.draw.rect(screen, (255, 224, 64), (rect.x + 11, rect.y + 10, 5, 5))

    def _draw_clear_block_run(self, screen, rect: pygame.Rect, length: int) -> None:
        phase = (self.steps // 12) % 2
        for index in range(length):
            x = rect.x + index * self.TILE_SIZE
            y = rect.y
            block = pygame.Rect(x + 2, y + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
            pygame.draw.rect(screen, EGA["K"], block.inflate(4, 4))
            glass = pygame.Surface((block.w, block.h), pygame.SRCALPHA)
            pygame.draw.rect(glass, (88, 232, 255, 44), (0, 0, block.w, block.h))
            pygame.draw.rect(glass, (255, 255, 255, 110), (0, 0, block.w, block.h), 2)
            pygame.draw.line(glass, (255, 255, 255, 125), (4, 5), (block.w - 7, 5), 2)
            pygame.draw.line(
                glass,
                (88, 232, 255, 120),
                (block.w - 7, 8),
                (7, block.h - 8),
                2,
            )
            screen.blit(glass, block.topleft)
            if phase:
                pygame.draw.rect(screen, EGA["W"], (x + 8, y + 7, 9, 2))
                pygame.draw.rect(screen, EGA["C"], (x + 20, y + 21, 5, 2))

    def _draw_room_label(self, screen, rect: pygame.Rect, label: str) -> None:
        if not self._art or not label:
            return
        width = max(74, len(label) * 8 + 14)
        sign = pygame.Rect(rect.x, rect.y + 7, width, 18)
        pygame.draw.rect(screen, EGA["K"], sign.inflate(4, 4))
        pygame.draw.rect(screen, (28, 28, 54), sign)
        pygame.draw.rect(screen, EGA["Y"], sign, 1)
        self._art.draw_text(screen, label, sign.x + 7, sign.y + 5, EGA["Y"], scale=1)

    def _draw_cable_run(self, screen, rect: pygame.Rect, length: int) -> None:
        palette = self._episode_palette()
        run = pygame.Rect(rect.x, rect.y + 13, self.TILE_SIZE * length, 9)
        pygame.draw.rect(screen, (0, 0, 0), run.inflate(4, 4))
        pygame.draw.rect(screen, palette["pipe_shadow"], run.move(3, 4))
        pygame.draw.rect(screen, palette["pipe_dark"], run)
        pygame.draw.line(
            screen,
            palette["pipe_light"],
            (run.left + 2, run.top + 2),
            (run.right - 3, run.top + 2),
            2,
        )
        for x in range(run.left + 11, run.right, 28):
            pygame.draw.rect(screen, (0, 0, 0), (x - 2, run.top - 3, 7, run.h + 6))
            pygame.draw.rect(
                screen, palette["pipe_light"], (x, run.top - 1, 3, run.h + 2)
            )

    def _draw_elevator_frame(
        self, screen, rect: pygame.Rect, label: str = "EXIT"
    ) -> None:
        palette = self._episode_palette()
        frame = pygame.Rect(rect.x - 13, rect.y - 18, 58, 58)
        pygame.draw.rect(screen, (0, 0, 0), frame.inflate(6, 6))
        pygame.draw.rect(screen, palette["pipe_shadow"], frame)
        pygame.draw.rect(screen, (176, 176, 190), (frame.x, frame.y, 8, frame.h))
        pygame.draw.rect(screen, (86, 86, 104), (frame.right - 8, frame.y, 8, frame.h))
        pygame.draw.rect(screen, (176, 176, 190), (frame.x, frame.y, frame.w, 8))
        pygame.draw.rect(screen, (86, 86, 104), (frame.x, frame.bottom - 8, frame.w, 8))
        pygame.draw.rect(
            screen, palette["pipe_light"], (frame.x + 4, frame.y + 4, 8, 8)
        )
        pygame.draw.rect(
            screen,
            EGA["G"] if self.exit_unlocked else EGA["A"],
            (frame.right - 12, frame.y + 4, 8, 8),
        )
        if self._art:
            self._art.draw_text(
                screen,
                label or "EXIT",
                frame.x + 15,
                frame.y + 5,
                EGA["G"] if self.exit_unlocked else EGA["Y"],
                scale=1,
            )

    def _draw_crystal_light(self, screen, rect: pygame.Rect) -> None:
        glow = pygame.Surface((64, 64), pygame.SRCALPHA)
        pygame.draw.circle(glow, (88, 232, 255, 45), (32, 32), 30)
        pygame.draw.circle(glow, (255, 255, 255, 36), (32, 32), 13)
        screen.blit(glow, (rect.x - 16, rect.y - 16))
        if self._art:
            self._art.draw_sprite(screen, "lamp", rect.x, rect.y, scale=2)
        else:
            self._draw_lamp(screen, rect)

    def _should_draw_support_rail(self, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.level_cols or row >= self.level_rows:
            return False
        if self.level_index % len(self.CAVES) == 0:
            return False
        if self._solid_at(col, row):
            return False
        if not self._solid_at(col, row - 1):
            return False
        if self._solid_at(col, row + 1):
            return False
        return (col * 5 + row * 2 + self.level_index) % 19 == 0

    def _empty_dressing_cell(self, col: int, row: int) -> bool:
        tile = (col, row)
        if (
            col <= 0
            or row <= 0
            or col >= self.level_cols - 1
            or row >= self.level_rows - 1
        ):
            return False
        if self._solid_at(col, row) or not self._solid_at(col, row + 1):
            return False
        occupied = (
            self.crystals
            | self.switches
            | self.hazards
            | self.ammo_pickups
            | self.treasures
            | self.air_tanks
            | set(self.powerups.keys())
            | {self.exit_pos}
        )
        return tile not in occupied

    def _same_hazard_at(self, col: int, row: int, kind: str) -> bool:
        return (col, row) in self.hazards and self.hazard_kinds.get((col, row)) == kind

    def _draw_support_rail(self, screen, rect: pygame.Rect) -> None:
        palette = self._episode_palette()
        x = rect.centerx - 2
        pygame.draw.rect(screen, (0, 0, 0), (x - 2, rect.y, 8, rect.h))
        pygame.draw.rect(screen, palette["pipe_dark"], (x, rect.y, 4, rect.h))
        for y in range(rect.y + 2, rect.bottom, 8):
            pygame.draw.line(
                screen, palette["pipe_light"], (x - 2, y), (x + 6, y + 5), 1
            )

    def _draw_sign_for_tile(
        self,
        screen,
        camera_x: int,
        camera_y: int,
        tile: Tuple[int, int],
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        col, row = tile
        x, y = self._world_to_screen(
            col * self.TILE_SIZE,
            (row - 1) * self.TILE_SIZE,
            camera_x,
            camera_y,
        )
        if y < -self.TILE_SIZE or y > self.height - self.HUD_HEIGHT:
            return
        text_width = 0
        if self._art:
            text_width = self._art.text(label, color, scale=1).get_width()
        sign_width = max(34, min(74, text_width + 18))
        sign = pygame.Rect(x + 1, y + 8, sign_width, 18)
        pygame.draw.rect(screen, (0, 0, 0), sign.inflate(4, 4))
        fill = (86, 0, 0) if label == "DANGER" else (72, 46, 0)
        pygame.draw.rect(screen, fill, sign)
        pygame.draw.rect(screen, color, sign, 2)
        if label == "DANGER":
            pygame.draw.polygon(
                screen,
                EGA["Y"],
                [
                    (sign.x + 5, sign.y + 14),
                    (sign.x + 11, sign.y + 4),
                    (sign.x + 17, sign.y + 14),
                ],
            )
            pygame.draw.rect(screen, EGA["K"], (sign.x + 10, sign.y + 8, 2, 4))
            pygame.draw.rect(screen, EGA["K"], (sign.x + 10, sign.y + 13, 2, 2))
            text_x = sign.x + 22
        else:
            text_x = sign.x + 6
        if self._art:
            self._art.draw_text(
                screen,
                label,
                text_x,
                sign.y + 6,
                color,
                scale=1,
                shadow=False,
            )
        elif self._tiny_font:
            text = self._tiny_font.render(label, True, color)
            screen.blit(text, text.get_rect(center=sign.center))

    def _draw_crate(self, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "crate", rect.x, rect.y, scale=2)
            return
        crate = pygame.Rect(rect.x + 6, rect.y + 10, 20, 18)
        pygame.draw.rect(screen, (0, 0, 0), crate.inflate(2, 2))
        pygame.draw.rect(screen, (156, 82, 26), crate)
        pygame.draw.rect(screen, (236, 150, 52), crate, 2)
        pygame.draw.line(screen, (92, 44, 14), crate.topleft, crate.bottomright, 2)
        pygame.draw.line(screen, (92, 44, 14), crate.topright, crate.bottomleft, 2)

    def _draw_pickaxe(self, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "pickaxe", rect.x, rect.y, scale=2)
            return
        pygame.draw.line(
            screen,
            (176, 118, 62),
            (rect.x + 9, rect.y + 25),
            (rect.x + 21, rect.y + 10),
            3,
        )
        pygame.draw.line(
            screen,
            (220, 220, 230),
            (rect.x + 10, rect.y + 10),
            (rect.x + 27, rect.y + 7),
            3,
        )
        pygame.draw.line(
            screen,
            (220, 220, 230),
            (rect.x + 12, rect.y + 10),
            (rect.x + 5, rect.y + 16),
            3,
        )
        pygame.draw.line(
            screen, (0, 0, 0), (rect.x + 9, rect.y + 25), (rect.x + 21, rect.y + 10), 1
        )

    def _draw_lamp(self, screen, rect: pygame.Rect) -> None:
        if self._art:
            self._art.draw_sprite(screen, "lamp", rect.x, rect.y, scale=2)
            return
        pygame.draw.rect(screen, (0, 0, 0), (rect.x + 8, rect.y + 10, 16, 18))
        pygame.draw.rect(screen, (90, 90, 105), (rect.x + 11, rect.y + 13, 10, 13))
        pygame.draw.rect(screen, (255, 230, 64), (rect.x + 13, rect.y + 15, 6, 6))
        pygame.draw.line(
            screen,
            (200, 200, 210),
            (rect.x + 16, rect.y + 8),
            (rect.x + 16, rect.y + 13),
            2,
        )

    def _draw_pickups(self, screen, camera_x: int, camera_y: int) -> None:
        for col, row in self.crystals:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                crystal_key = (
                    "crystal_blue",
                    "crystal_green",
                    "crystal_yellow",
                    "crystal_red",
                )[(col + row) % len(self.GEM_COLORS)]
                glow = pygame.Surface((40, 40), pygame.SRCALPHA)
                pulse = 16 + ((self.steps + col * 3 + row) // 8) % 4
                pygame.draw.circle(glow, (88, 232, 255, 30 + pulse), (20, 20), pulse)
                screen.blit(glow, (x - 4, y - 4))
                bob = -1 if ((self.steps + col * 7) // 16) % 2 else 0
                self._art.draw_sprite(screen, crystal_key, x, y + bob, scale=2)
                if (self.steps + col + row) % 48 < 6:
                    pygame.draw.rect(screen, EGA["W"], (x + 10, y + 7 + bob, 5, 2))
                continue
            color = self.GEM_COLORS[(col + row) % len(self.GEM_COLORS)]
            shine = tuple(min(255, c + 70) for c in color)
            points = [
                (x + 16, y + 4),
                (x + 27, y + 16),
                (x + 16, y + 29),
                (x + 5, y + 16),
            ]
            pygame.draw.polygon(
                screen,
                (0, 0, 0),
                [(x + 16, y + 1), (x + 30, y + 16), (x + 16, y + 31), (x + 2, y + 16)],
            )
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(
                screen, shine, [(x + 16, y + 4), (x + 27, y + 16), (x + 16, y + 16)]
            )
            pygame.draw.polygon(
                screen,
                (255, 255, 255),
                [(x + 11, y + 13), (x + 16, y + 7), (x + 14, y + 15)],
            )
            pygame.draw.polygon(screen, (0, 0, 0), points, 1)

        for col, row in self.ammo_pickups:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_sprite(screen, "ammo", x, y, scale=2)
                if (self.steps // 10) % 2 == 0:
                    pygame.draw.rect(screen, EGA["Y"], (x + 16, y + 13, 9, 3))
                continue
            pygame.draw.rect(screen, (0, 0, 0), (x + 6, y + 8, 20, 14))
            pygame.draw.rect(screen, (190, 190, 205), (x + 7, y + 9, 18, 12))
            pygame.draw.rect(screen, (255, 255, 255), (x + 10, y + 11, 5, 3))
            pygame.draw.rect(screen, (255, 220, 72), (x + 13, y + 14, 9, 4))
            if self._tiny_font:
                label = self._tiny_font.render("R", True, (0, 0, 0))
                screen.blit(label, label.get_rect(center=(x + 16, y + 15)))

        for col, row in self.air_tanks:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_sprite(screen, "air_tank", x, y, scale=2)
                if (self.steps // 14) % 2 == 0:
                    pygame.draw.rect(screen, EGA["C"], (x + 20, y + 6, 3, 18))
                continue
            pygame.draw.rect(screen, (0, 0, 0), (x + 7, y + 2, 18, 28))
            pygame.draw.rect(screen, (50, 210, 240), (x + 9, y + 6, 14, 22))
            pygame.draw.rect(screen, (220, 255, 255), (x + 9, y + 6, 14, 22), 2)
            pygame.draw.rect(screen, (250, 250, 255), (x + 12, y + 9, 3, 14))
            pygame.draw.rect(screen, (160, 160, 170), (x + 12, y + 2, 8, 5))
            if self._tiny_font:
                label = self._tiny_font.render("AIR", True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=(x + 16, y + 17)))

        for col, row in self.treasures:
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_text(screen, "$", x + 9, y + 8, EGA["Y"], scale=2)
                continue
            pygame.draw.circle(screen, (255, 215, 95), (x + 16, y + 16), 8)

        for (col, row), power in self.powerups.items():
            x, y = self._world_to_screen(
                col * self.TILE_SIZE, row * self.TILE_SIZE, camera_x, camera_y
            )
            if self._art:
                self._art.draw_sprite(screen, "power", x, y, scale=2)
                if (self.steps // 8) % 2 == 0:
                    pygame.draw.rect(screen, EGA["W"], (x + 5, y + 5, 22, 2))
                self._art.draw_text(
                    screen,
                    power.upper(),
                    x + 11,
                    y + 10,
                    EGA["K"],
                    scale=1,
                    shadow=False,
                )
                continue
            color = {
                self.POWER_SHOT: (255, 95, 95),
                self.GRAVITY_POWER: (150, 130, 255),
                self.FREEZE_POWER: (100, 220, 255),
            }[power]
            pygame.draw.rect(screen, (0, 0, 0), (x + 5, y + 5, 22, 22))
            pygame.draw.rect(screen, color, (x + 7, y + 7, 18, 18))
            pygame.draw.rect(screen, (255, 255, 255), (x + 10, y + 9, 5, 4))
            if self._tiny_font:
                label = self._tiny_font.render(power.upper(), True, (20, 20, 25))
                screen.blit(label, label.get_rect(center=(x + 16, y + 16)))

    def _draw_enemies(self, screen, camera_x: int, camera_y: int) -> None:
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            x, y = self._world_to_screen(enemy.x, enemy.y, camera_x, camera_y)
            if self._art:
                if enemy.kind == "flyer":
                    bob = 2 if int((self.steps + enemy.x) / 10) % 2 else 0
                    sprite = (
                        "bat_enemy"
                        if (self.level_index + int(enemy.x // self.TILE_SIZE)) % 2
                        else "eye_flyer"
                    )
                    self._art.draw_sprite(screen, sprite, x - 4, y + bob, scale=2)
                    if (self.steps // 9) % 2 == 0:
                        pygame.draw.rect(screen, EGA["A"], (x + 11, y + 18 + bob, 5, 2))
                else:
                    sprite = (
                        "slug_enemy"
                        if (self.level_index + int(enemy.x // self.TILE_SIZE)) % 2
                        else "walking_rock"
                    )
                    step_bob = 1 if int((self.steps + enemy.x) / 12) % 2 else 0
                    self._art.draw_sprite(
                        screen,
                        sprite,
                        x - 4,
                        y + 2 + step_bob,
                        scale=2,
                        flip_x=enemy.vx < 0,
                    )
                continue
            if enemy.kind == "flyer":
                wing_phase = int((self.steps + int(enemy.x)) / 8) % 2
                wing_y = y + (0 if wing_phase else 4)
                pygame.draw.rect(screen, (0, 0, 0), (x - 2, y, 29, 22))
                pygame.draw.polygon(
                    screen,
                    (190, 42, 210),
                    [(x + 3, y + 12), (x + 9, wing_y + 3), (x + 15, y + 12)],
                )
                pygame.draw.polygon(
                    screen,
                    (250, 88, 255),
                    [(x + 11, y + 12), (x + 19, wing_y + 2), (x + 24, y + 12)],
                )
                pygame.draw.rect(screen, (120, 250, 88), (x + 9, y + 12, 12, 7))
                pygame.draw.rect(screen, (255, 255, 255), (x + 16, y + 14, 3, 3))
            else:
                foot_offset = 2 if int((self.steps + enemy.x) / 12) % 2 else 0
                pygame.draw.rect(screen, (0, 0, 0), (x - 2, y + 2, enemy.width + 4, 24))
                pygame.draw.rect(screen, (52, 202, 60), (x + 1, y + 9, 20, 12))
                pygame.draw.rect(screen, (112, 255, 88), (x + 4, y + 6, 14, 5))
                pygame.draw.rect(screen, (0, 108, 20), (x + 3, y + 20, 8, 4))
                pygame.draw.rect(
                    screen, (0, 108, 20), (x + 13, y + 20 + foot_offset, 8, 4)
                )
                pygame.draw.rect(screen, (255, 255, 255), (x + 16, y + 10, 3, 3))
                pygame.draw.rect(screen, (255, 64, 64), (x + 2, y + 14, 4, 3))

    def _draw_bullets(self, screen, camera_x: int, camera_y: int) -> None:
        for bullet in self.bullets:
            x, y = self._world_to_screen(bullet.x, bullet.y, camera_x, camera_y)
            color = (255, 255, 120) if bullet.powered else (255, 210, 60)
            pygame.draw.rect(screen, (0, 0, 0), (x - 1, y - 1, 14, 6))
            pygame.draw.rect(screen, color, (x, y, 11, 4))
            pygame.draw.rect(
                screen, (255, 90, 40), (x - 3 if bullet.vx > 0 else x + 10, y, 3, 4)
            )

    def _draw_player(self, screen, camera_x: int, camera_y: int) -> None:
        x, y = self._world_to_screen(self.player_x, self.player_y, camera_x, camera_y)
        flash = self.invuln_timer > 0 and (self.invuln_timer // 5) % 2 == 0
        if flash:
            return

        if self._art:
            if self.invuln_timer > 0:
                sprite = "mylo_hurt"
            elif self.shoot_cooldown > self.SHOOT_COOLDOWN - 7:
                sprite = "mylo_shoot"
            elif not self.grounded:
                sprite = "mylo_jump"
            elif abs(self.vx) > 0.2:
                sprite = "mylo_walk_1" if (self.steps // 8) % 2 == 0 else "mylo_walk_2"
            else:
                sprite = "mylo_idle"

            pygame.draw.ellipse(
                screen,
                EGA["K"],
                (x - 4, y + self.PLAYER_HEIGHT - 2, 34, 7),
            )
            pygame.draw.rect(screen, EGA["K"], (x - 4, y - 4, 33, 38), 1)
            self._art.draw_sprite(
                screen,
                sprite,
                x - 1,
                y - 2,
                scale=2,
                flip_x=self.facing < 0,
            )
            pygame.draw.rect(screen, EGA["Y"], (x + 7, y - 2, 10, 2))
            if sprite != "mylo_shoot":
                gun_x = x + 16 if self.facing > 0 else x - 10
                self._art.draw_sprite(
                    screen,
                    "raygun",
                    gun_x,
                    y + 12,
                    scale=1,
                    flip_x=self.facing < 0,
                )
            if self.gravity_dir < 0:
                pygame.draw.rect(screen, EGA["C"], (x + 1, y - 5, 22, 3))
            if self.super_timer > 0:
                pygame.draw.rect(screen, EGA["Y"], (x - 1, y + 10, 4, 18))
            if self.freeze_timer > 0:
                pygame.draw.rect(screen, EGA["C"], (x + 22, y + 2, 3, 24))
            return

        pygame.draw.rect(screen, (0, 0, 0), (x - 2, y - 1, 28, 33))
        walk_frame = int(self.steps / 8) % 2 if abs(self.vx) > 0.2 else 0
        recoil = 2 if self.shoot_cooldown > self.SHOOT_COOLDOWN - 5 else 0

        # Helmet and face.
        pygame.draw.rect(screen, (255, 222, 52), (x + 5, y, 14, 7))
        pygame.draw.rect(screen, (255, 172, 40), (x + 4, y + 5, 17, 4))
        pygame.draw.rect(screen, (255, 190, 132), (x + 6, y + 9, 13, 8))
        eye_x = x + 15 if self.facing > 0 else x + 8
        pygame.draw.rect(screen, (0, 0, 0), (eye_x, y + 11, 2, 2))

        # Pink shirt, blue overalls, red boots: intentionally loud EGA.
        pygame.draw.rect(screen, (255, 76, 172), (x + 4, y + 17, 17, 6))
        pygame.draw.rect(screen, (68, 84, 255), (x + 7, y + 20, 11, 8))
        pygame.draw.rect(screen, (255, 64, 52), (x + 4, y + 28 + walk_frame, 8, 4))
        pygame.draw.rect(screen, (255, 64, 52), (x + 15, y + 29 - walk_frame, 8, 4))

        if self.gravity_dir < 0:
            pygame.draw.rect(screen, (100, 190, 255), (x + 2, y - 4, 20, 3))
        if self.super_timer > 0:
            pygame.draw.rect(screen, (255, 255, 90), (x + 2, y + 15, 3, 9))
        if self.freeze_timer > 0:
            pygame.draw.rect(screen, (80, 220, 255), (x + 20, y + 3, 3, 20))

        gun_x = x + 19 - recoil if self.facing > 0 else x - 7 + recoil
        pygame.draw.rect(screen, (225, 225, 232), (gun_x, y + 17, 11, 4))
        pygame.draw.rect(
            screen,
            (255, 236, 80),
            (gun_x + (9 if self.facing > 0 else -2), y + 18, 3, 2),
        )

    def _draw_visual_events(self, screen, camera_x: int, camera_y: int) -> None:
        if not self.visual_events:
            return

        for event in self.visual_events:
            x, y = self._world_to_screen(event.x, event.y, camera_x, camera_y)
            age = event.max_ttl - event.ttl
            rise = age // 2
            if event.kind == "sparkle":
                if self._art:
                    self._art.draw_sprite(
                        screen, "sparkle", x - 16, y - 20 - rise, scale=2
                    )
                else:
                    pygame.draw.circle(screen, event.color, (x, y - rise), 10, 2)
                if event.text and self._art:
                    self._art.draw_text(
                        screen,
                        event.text,
                        x - 20,
                        y - 31 - rise,
                        event.color,
                        scale=1,
                    )
            elif event.kind == "poof":
                if self._art:
                    self._art.draw_sprite(
                        screen, "poof", x - 16, y - 18 - rise, scale=2
                    )
                else:
                    pygame.draw.circle(screen, (210, 210, 220), (x, y - rise), 12)
                if event.text and self._art:
                    self._art.draw_text(
                        screen,
                        event.text,
                        x - 14,
                        y - 34 - rise,
                        event.color,
                        scale=1,
                    )
            elif event.kind == "spark":
                radius = max(2, 10 - age // 2)
                pygame.draw.line(
                    screen, event.color, (x - radius, y), (x + radius, y), 2
                )
                pygame.draw.line(
                    screen, event.color, (x, y - radius), (x, y + radius), 2
                )
                pygame.draw.rect(screen, EGA["W"], (x - 1, y - 1, 3, 3))
                if event.text and self._art:
                    self._art.draw_text(
                        screen,
                        event.text,
                        x - 12,
                        y - 28 - rise,
                        event.color,
                        scale=1,
                    )
            elif event.text and self._art:
                self._art.draw_text(
                    screen,
                    event.text,
                    x - 16,
                    y - 24 - rise,
                    event.color,
                    scale=1,
                )

    def _draw_gravity_overlay(self, screen) -> None:
        """Full-screen treatment while the gravity field is inverted (CCV-19):
        a violet edge vignette, debris floating upward, and a period-style
        REVERSE GRAVITY banner — so the altered field reads at a glance.
        """
        if self.gravity_timer <= 0:
            return
        play_bottom = self.height - self.HUD_HEIGHT
        fade = min(1.0, self.gravity_timer / 60.0)
        overlay = pygame.Surface((self.width, play_bottom), pygame.SRCALPHA)
        tint = (150, 60, 230)

        band = 56
        for i in range(band):
            alpha = int(64 * fade * (1 - i / band))
            pygame.draw.line(overlay, (*tint, alpha), (0, i), (self.width, i))
            pygame.draw.line(
                overlay,
                (*tint, alpha),
                (0, play_bottom - 1 - i),
                (self.width, play_bottom - 1 - i),
            )

        # Debris floats upward to sell the inverted field.
        for k in range(26):
            sx = (k * 137 + (self.steps * 2)) % self.width
            sy = play_bottom - ((self.steps * 3 + k * 53) % play_bottom)
            pygame.draw.rect(overlay, (*EGA["C"], 150), (sx, sy, 2, 4))
        screen.blit(overlay, (0, 0))

        if self._art:
            label = "REVERSE GRAVITY"
            text_w = self._art.text(label, EGA["Y"], scale=2).get_width()
            bx = (self.width - text_w) // 2
            sign = pygame.Rect(bx - 8, 10, text_w + 16, 24)
            pygame.draw.rect(screen, EGA["A"], sign)
            pygame.draw.rect(screen, EGA["K"], sign, 2)
            self._art.draw_text(screen, label, bx, 16, EGA["Y"], scale=2)

    def _draw_heart(self, screen, x: int, y: int, alive: bool) -> None:
        """Draw a small pixel heart pip (bright-red alive, dark husk when lost)."""
        body = (255, 85, 85) if alive else (62, 30, 30)
        pygame.draw.circle(screen, body, (x + 4, y + 4), 4)
        pygame.draw.circle(screen, body, (x + 12, y + 4), 4)
        pygame.draw.polygon(
            screen, body, [(x, y + 5), (x + 16, y + 5), (x + 8, y + 15)]
        )
        if alive:
            pygame.draw.rect(screen, (255, 190, 190), (x + 2, y + 2, 2, 2))

    def _draw_hud(self, screen) -> None:
        hud_y = self.height - self.HUD_HEIGHT
        # Period-authentic footer (CCV-04/05): one thin black bar with a bright
        # top rule and label-less data clusters — score, crystals, ammo, hearts.
        # No CAVE/EXIT/MYLO labels, no compartment dividers, no controls line.
        pygame.draw.rect(screen, EGA["K"], (0, hud_y, self.width, self.HUD_HEIGHT))
        pygame.draw.line(screen, EGA["G"], (0, hud_y), (self.width, hud_y), 1)
        pygame.draw.line(screen, EGA["g"], (0, hud_y + 2), (self.width, hud_y + 2), 1)

        collected = self.initial_crystals - len(self.crystals)
        row = hud_y + 10

        if self._art:
            # Score — gold "$" + chunky green numerals.
            self._art.draw_text(screen, "$", 12, row, EGA["Y"], scale=2)
            self._art.draw_text(
                screen, f"{self.score:06d}", 34, row, EGA["G"], scale=2
            )

            # Crystals collected toward the exit goal.
            self._art.draw_sprite(screen, "crystal_blue", 198, row - 1, scale=1)
            self._art.draw_text(
                screen,
                f"{collected:02d}/{self.initial_crystals:02d}",
                226,
                row,
                EGA["Y"],
                scale=2,
            )

            # Ammo — raygun icon + green count.
            self._art.draw_sprite(screen, "raygun", 380, row + 1, scale=1)
            self._art.draw_text(
                screen, f"{self.ammo:02d}", 418, row, EGA["G"], scale=2
            )

            # Exit lock state — a single iconographic cue, no text label.
            self._art.draw_sprite(
                screen,
                "door_open" if self.exit_unlocked else "door_locked",
                494,
                row - 5,
                scale=1,
            )

            # Hearts — right-aligned; lost hearts drop to a dark husk.
            for i in range(self.MAX_HEALTH):
                hx = self.width - 26 - (self.MAX_HEALTH - 1 - i) * 26
                self._draw_heart(screen, hx, row, i < self.health)
            return

        if not self._small_font:
            return

        # Font fallback (no pixel-art atlas) — same label-less layout.
        score_text = self._small_font.render(f"$ {self.score:06d}", True, (64, 236, 80))
        screen.blit(score_text, (12, hud_y + 10))

        gem_color = self.GEM_COLORS[collected % len(self.GEM_COLORS)]
        gem_x = 210
        pygame.draw.polygon(
            screen,
            gem_color,
            [
                (gem_x + 9, hud_y + 6),
                (gem_x + 18, hud_y + 16),
                (gem_x + 9, hud_y + 28),
                (gem_x, hud_y + 16),
            ],
        )
        gem_text = self._small_font.render(
            f"{collected}/{self.initial_crystals}", True, (255, 224, 64)
        )
        screen.blit(gem_text, (gem_x + 26, hud_y + 10))

        ammo_x = 360
        pygame.draw.rect(screen, (220, 220, 230), (ammo_x, hud_y + 14, 18, 5))
        pygame.draw.polygon(
            screen,
            (255, 230, 78),
            [
                (ammo_x + 18, hud_y + 12),
                (ammo_x + 25, hud_y + 16),
                (ammo_x + 18, hud_y + 20),
            ],
        )
        ammo_text = self._small_font.render(str(self.ammo), True, (64, 236, 80))
        screen.blit(ammo_text, (ammo_x + 34, hud_y + 10))

        for i in range(self.MAX_HEALTH):
            hx = self.width - 26 - (self.MAX_HEALTH - 1 - i) * 26
            self._draw_heart(screen, hx, hud_y + 10, i < self.health)


class VecCrystalCaves:
    """
    Vectorized Crystal Caves environment for parallel training.
    """

    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        self.num_envs = num_envs
        self.config = config
        self.headless = headless
        self.envs = [CrystalCaves(config, headless=headless) for _ in range(num_envs)]
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        self._states: np.ndarray = np.empty(
            (num_envs, self.state_size), dtype=np.float32
        )
        self._rewards: np.ndarray = np.empty(num_envs, dtype=np.float32)
        self._dones: np.ndarray = np.empty(num_envs, dtype=np.bool_)
        self._pending_resets: np.ndarray = np.zeros(num_envs, dtype=np.bool_)
        self._last_infos: List[dict] = []

    def reset(self) -> np.ndarray:
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        self._step_into_buffers(actions)
        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        for i, done in enumerate(self._dones):
            if done:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        return states_to_return, rewards_to_return, dones_to_return, self._last_infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        for i in range(self.num_envs):
            if self._pending_resets[i]:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        self._step_into_buffers(actions)
        return self._states, self._rewards, self._dones, self._last_infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seeds: List[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def _step_into_buffers(self, actions: np.ndarray) -> None:
        infos = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))
            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)
            if done:
                env.reset()
                self._pending_resets[i] = True
        self._last_infos = infos
