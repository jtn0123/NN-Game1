"""Dataclasses and level data for the Crystal Caves game mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pygame


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


CAVES: Tuple[CaveSpec, ...] = (
    CaveSpec(
        name="Trouble with Twinkles",
        background=(9, 12, 22),
        accent=(80, 190, 255),
        layout=(
            "############################################",
            "##...#######################################",
            "##.P......*.......A......*.....##.....E#####",
            "##...######################.##....#####D####",
            "####.##############M#######.###s###########",
            "####.#########...........#..##########..####",
            "#..*.....#####.....O.....#..#######*.....##",
            "#.........####...........##...^^^^........#",
            "#.....A...####....####...###..............#",
            "#.........####....####...#####.....M......#",
            "#.........####....#####..#####.....####...#",
            "#....*...####p....#########*.......####...#",
            "#........######...#..###########..........#",
            "#..M.....######...#..###########.....*....#",
            "#.........*.###..........A...###..........#",
            "#............##...............###.........#",
            "#^^^^^......###~~~~~.........#####...^^^^^#",
            "############################################",
        ),
    ),
    CaveSpec(
        name="Slugging It Out",
        background=(12, 10, 18),
        accent=(255, 188, 80),
        layout=(
            "############################################",
            "##...#######################################",
            "##.P....A.....*.....###################.E###",
            "##...##############..#################DD####",
            "###*####.###########....F.............DD####",
            "#.............#######....................###",
            "#.....M........######...........*.........##",
            "#..########.......###......##########......#",
            "#..#########....*..###.....##########......#",
            "#.......#####.......###s..###....####.*....#",
            "#........######.....########......###......#",
            "#..g......#####.....########.......#.......#",
            "#.............^^^^....##O###M......#.......#",
            "#....*.................#########...#..A....#",
            "#...........########...#########...#.......#",
            "#...........########*..........p..........##",
            "#~~~~~.....##########...............^^^^.###",
            "############################################",
        ),
    ),
    CaveSpec(
        name="Mylo and the Supernova",
        background=(8, 15, 13),
        accent=(120, 255, 155),
        layout=(
            "############################################",
            "##...#######################################",
            "##.P.###....*........##########A#.......E###",
            "##...###.###########.############.##########",
            "###*.....###########.......F.....O#####D####",
            "#.......###M###..###...................D.###",
            "#.....########....###...*.....#########D..##",
            "#.....########.....###........#########...##",
            "#.........####.A...#########......#####*..#",
            "#..z.......####....##########..s...####...##",
            "#..........#######......##########..##....##",
            "#..........########..M...#########..##....##",
            "#....*.....########...........^^^^...#.....#",
            "#..........########..................#.....#",
            "#..........p...####...*..............#.....#",
            "#...............###...........*......#.....#",
            "#^^^^^.........###~~~~~~~...........###^^^^#",
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
