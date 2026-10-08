"""Native 32-pixel retro art for the Unity Crystal Caves presentation.

EGA color anchors, compact arcade silhouettes and discrete animation poses.
Small hand-selected shade ramps add fidelity without filtering or painted art.
All terrain remains aligned to the authoritative simulation grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pygame

from .visual_creatures import creature_sprites
from .visual_doors import door_headroom_cells, door_sprites
from .visual_equipment import equipment_sprites
from .visual_fidelity import connected_pipes, pickup_sprites, service_pipe_cells
from .visual_materials import (
    BLUE,
    CYAN,
    GEM_RAMPS,
    GOLD,
    GREEN,
    INK,
    MAGENTA,
    PINK,
    RED,
    RUST,
    STEEL,
    TILE,
    WHITE,
    YELLOW,
    Color,
    bolt,
    canvas,
    lettering,
    metal,
    scratch,
)
from .visual_mechanisms import mechanism_sprites
from .visual_projectiles import projectile_sprites
from .visual_props import apparatus, crate, extra_sprites, pipe_segment, sign, warning_plate
from .visual_rooms import (
    blue_cobble,
    green_platform,
    pale_blocks,
    pipe_wall,
    purple_planks,
    ribbed_wall,
    rust_brick_wall,
    silver_girder,
    slate_platform,
)
from .visual_scenery import DressingPlacement, placement_cells, scenery_placements, scenery_sprites
from .visual_secrets import secret_sprites


@dataclass(frozen=True)
class Theme:
    body: Color
    shade: Color
    mid: Color
    light: Color
    lip: Color
    wall: Color
    wall_light: Color
    wall_shadow: Color
    name: str


THEMES: tuple[Theme, ...] = (
    Theme(
        (79, 105, 134),
        (41, 57, 87),
        (73, 94, 122),
        (81, 136, 147),
        (118, 205, 192),
        (162, 38, 60),
        (224, 104, 75),
        (61, 27, 54),
        "RED PANEL MINE",
    ),
    Theme(
        (28, 43, 184),
        (12, 22, 88),
        (21, 34, 148),
        (75, 92, 229),
        (136, 166, 255),
        (67, 77, 96),
        (113, 133, 153),
        (26, 35, 54),
        "BLUE STONE WORKS",
    ),
    Theme(
        (163, 89, 32),
        (72, 37, 27),
        (130, 65, 28),
        (204, 130, 45),
        (247, 192, 83),
        (7, 12, 23),
        (35, 44, 61),
        (12, 21, 36),
        "TIMBER OUTPOST",
    ),
    Theme(
        (62, 123, 74),
        (30, 69, 47),
        (46, 95, 62),
        (122, 169, 83),
        (205, 227, 122),
        (124, 64, 47),
        (168, 102, 65),
        (34, 31, 42),
        "GREEN STEEL WORKS",
    ),
)
THEMES += (
    Theme(
        (23, 37, 79),
        (8, 14, 39),
        (36, 58, 112),
        (73, 115, 174),
        (85, 211, 100),
        (79, 85, 83),
        (130, 142, 130),
        (42, 51, 47),
        "BLUE COBBLE MINE",
    ),
    Theme(
        (77, 103, 133),
        (33, 45, 67),
        (57, 78, 106),
        (94, 140, 151),
        (113, 207, 206),
        (77, 81, 80),
        (116, 126, 127),
        (39, 46, 57),
        "RIBBED MACHINERY WORKS",
    ),
    Theme(
        (166, 165, 194),
        (66, 65, 99),
        (127, 121, 163),
        (216, 217, 231),
        (233, 236, 239),
        (3, 5, 9),
        (170, 86, 52),
        (39, 42, 48),
        "PALE BLOCK VAULT",
    ),
    Theme(
        (159, 158, 188),
        (49, 48, 80),
        (105, 104, 143),
        (222, 228, 232),
        (250, 253, 239),
        (78, 27, 68),
        (128, 56, 111),
        (31, 19, 36),
        "PURPLE GIRDER WORKS",
    ),
)
CAVE_THEMES = (0, 6, 7, 3, 4, 0, 3, 7, 5, 3, 0, 2, 3, 4, 7, 0)


def theme_index(level: int) -> int:
    return CAVE_THEMES[level % len(CAVE_THEMES)]


def stone(theme: Theme, col: int, row: int, edges: tuple[bool, ...]) -> pygame.Surface:
    if theme == THEMES[3]:
        return green_platform(col, row, edges)
    if theme == THEMES[4]:
        return blue_cobble(col, row, edges)
    if theme == THEMES[5]:
        return slate_platform(col, row, edges)
    if theme == THEMES[6]:
        return pale_blocks(col, row)
    if theme == THEMES[7]:
        return silver_girder(col, row, edges)
    image = canvas()
    image.fill(theme.body)
    top, left, right, bottom = edges
    # Large quiet faces preserve the original's continuous platform masses.
    if top:
        pygame.draw.rect(image, theme.light, (0, 3, 32, 3))
        if theme != THEMES[0]:
            pygame.draw.rect(image, theme.mid, (0, 8, 32, 2))
        pygame.draw.line(image, INK, (0, 0), (31, 0))
        pygame.draw.line(image, theme.lip, (1, 1), (30, 1))
        pygame.draw.line(image, theme.light, (1, 2), (30, 2))
        if theme != THEMES[0]:
            for x in range(3, 31, 8):
                pygame.draw.line(image, theme.lip, (x, 4), (x + 3, 4))
    if left:
        pygame.draw.rect(image, INK, (0, 0, 1, 32))
        pygame.draw.rect(image, theme.lip, (1, 2 if top else 0, 1, 30 if top else 32))
        pygame.draw.rect(image, theme.light, (2, 5 if top else 0, 2, 27 if top else 32))
    if right:
        pygame.draw.rect(image, theme.mid, (27, 0, 2, 32))
        pygame.draw.rect(image, theme.shade, (29, 0, 2, 32))
        pygame.draw.rect(image, INK, (31, 0, 1, 32))
    if bottom:
        pygame.draw.rect(image, theme.mid, (0, 27, 32, 2))
        pygame.draw.rect(image, theme.shade, (0, 29, 32, 2))
        pygame.draw.rect(image, INK, (0, 31, 32, 1))
    if theme == THEMES[0]:
        # The reference's slate-blue platforms read as continuous quiet masses.
        # Keep wear on occasional exposed edges, rather than detailing every cell.
        if top and (col + row) % 6 == 0:
            pygame.draw.line(image, theme.body, (21, 1), (24, 1))
            pygame.draw.line(image, theme.mid, (22, 2), (24, 2))
        if bottom and (col * 3 + row) % 11 == 0:
            pygame.draw.line(image, theme.shade, (9, 26), (14, 27))
            pygame.draw.line(image, theme.mid, (10, 25), (14, 26))
        return image
    if theme == THEMES[2]:
        # Long grain runs across adjacent tiles, rather than noisy little blocks.
        for y in (12, 20, 25):
            pygame.draw.line(image, theme.mid, (0, y), (31, y))
            pygame.draw.line(image, theme.light, (3 + col % 3, y + 1), (23 + col % 5, y + 1))
        if (col + row) % 5 == 0:
            pygame.draw.ellipse(image, theme.shade, (8, 16, 8, 3), 1)
            pygame.draw.line(image, theme.light, (9, 19), (14, 19))
    elif (col + row * 3) % 7 == 0:
        # Small localized seams and wear leave most of each colored mass quiet.
        pygame.draw.line(image, theme.mid, (8, 17), (21, 17))
        pygame.draw.line(image, theme.light, (8, 18), (14, 18))
        if theme == THEMES[3]:
            pygame.draw.rect(image, theme.shade, (11, 15, 11, 8))
            for x in (13, 16, 19):
                pygame.draw.line(image, theme.mid, (x, 16), (x, 21))
            pygame.draw.line(image, theme.light, (11, 14), (21, 14))
        else:
            image.set_at((22, 17), theme.shade)
    variant = (col * 17 + row * 11) % 8
    if theme == THEMES[2]:
        # Knots, nail holes and torn grain describe wood, not metal corrosion.
        pygame.draw.line(image, theme.shade, (3, 13), (13 + col % 7, 13))
        pygame.draw.line(image, theme.light, (4, 14), (10, 14))
        if variant in (0, 3, 6):
            pygame.draw.ellipse(image, theme.shade, (18, 17, 7, 4), 1)
            pygame.draw.line(image, theme.mid, (19, 19), (23, 19))
            pygame.draw.line(image, theme.light, (18, 21), (25, 21))
        for xx in (5, 27):
            image.set_at((xx, 7), INK)
            image.set_at((xx - 1, 6), STEEL[2])
        if top and variant % 2 == 0:
            pygame.draw.line(image, theme.shade, (20, 2), (24, 4))
            image.set_at((23, 3), theme.body)
    elif theme == THEMES[1]:
        if variant in (0, 2, 5):
            pygame.draw.lines(image, theme.shade, False, [(7, 13), (11, 16), (10, 20), (16, 23)])
            pygame.draw.line(image, theme.light, (8, 13), (11, 15))
            pygame.draw.line(image, theme.mid, (15, 22), (22, 22))
            image.set_at((21, 21), theme.light)
        else:
            pygame.draw.line(image, theme.mid, (8, 18), (14, 17))
            pygame.draw.line(image, theme.shade, (14, 18), (17, 20))
            image.set_at((9, 19), theme.light)
        if top:
            pygame.draw.line(image, theme.mid, (12, 2), (17, 2))
            pygame.draw.line(image, theme.shade, (14, 3), (17, 3))
            image.set_at((17, 4), theme.light)
    else:
        # Wear gathers at exposed/traffic edges; broad interior faces stay quiet.
        exposed = any(edges)
        if exposed or variant == 3:
            xx, yy = 7 + variant % 3 * 4, 19 + variant % 2
            pygame.draw.polygon(
                image, theme.mid, [(xx, yy), (xx + 8, yy - 1), (xx + 6, yy + 2), (xx + 1, yy + 3)]
            )
            pygame.draw.line(image, theme.shade, (xx + 2, yy + 1), (xx + 6, yy))
            pygame.draw.line(image, STEEL[2], (xx + 2, yy + 2), (xx + 4, yy + 1))
        if variant in (0, 2, 4, 7) and (exposed or variant == 2):
            scratch(image, 6 + variant, 15 + variant % 3, 5, theme.shade, theme.light)
            pygame.draw.line(image, STEEL[0], (22, 22), (25, 21))
            pygame.draw.line(image, STEEL[2], (23, 23), (25, 22))
        if variant in (1, 6) and (exposed or variant == 1):
            pygame.draw.lines(image, theme.shade, False, [(9, 16), (10, 24), (22, 24)])
            pygame.draw.line(image, theme.mid, (11, 16), (21, 16))
            image.set_at((11, 18), STEEL[2])
            image.set_at((21, 22), theme.lip)
        if top:
            xx = 7 + col * 7 % 18
            pygame.draw.line(image, STEEL[1], (xx, 1), (xx + 3, 1))
            pygame.draw.line(image, STEEL[3], (xx + 1, 2), (xx + 3, 2))
            if theme == THEMES[3] and variant % 3 == 0:
                pygame.draw.line(image, RUST[0], (xx, 5), (xx + 3, 6))
                pygame.draw.line(image, RUST[1], (xx + 1, 5), (xx + 2, 6))
        if bottom and variant % 3 == 1:
            pygame.draw.line(image, theme.shade, (4, 26), (10, 26))
            image.set_at((6, 27), RUST[1])
            image.set_at((7, 28), RUST[0])
    return image


def ladder() -> pygame.Surface:
    image = canvas()
    for x in (8, 23):
        pygame.draw.rect(image, INK, (x - 1, 0, 4, 32))
        pygame.draw.line(image, STEEL[3], (x, 0), (x, 31))
        pygame.draw.line(image, STEEL[1], (x + 1, 0), (x + 1, 31))
    for y in (3, 11, 19, 27):
        pygame.draw.rect(image, INK, (8, y, 17, 4))
        pygame.draw.line(image, STEEL[4], (9, y), (23, y))
        pygame.draw.line(image, STEEL[2], (9, y + 1), (23, y + 1))
        pygame.draw.line(image, STEEL[1], (9, y + 2), (23, y + 2))
        pygame.draw.line(image, STEEL[3], (13, y), (18, y))
    pygame.draw.line(image, RUST[0], (9, 21), (9, 25))
    image.set_at((10, 23), RUST[1])
    pygame.draw.line(image, STEEL[0], (18, 12), (20, 12))
    image.set_at((19, 13), STEEL[3])
    return image


def hazard(kind: str) -> pygame.Surface:
    image = canvas()
    if kind == "^":
        metal(image, (0, 25, 32, 7))
        for x in range(0, 32, 8):
            pygame.draw.polygon(image, INK, [(x, 25), (x + 4, 2), (x + 8, 25)])
            pygame.draw.polygon(image, STEEL[2], [(x + 1, 24), (x + 4, 5), (x + 7, 24)])
            pygame.draw.polygon(image, STEEL[4], [(x + 2, 21), (x + 4, 6), (x + 4, 22)])
            pygame.draw.line(image, STEEL[0], (x + 5, 13), (x + 6, 23))
        for x in range(0, 32, 8):
            pygame.draw.line(image, GOLD[2], (x, 29), (x + 3, 29))
            pygame.draw.line(image, RUST[0], (x + 5, 22), (x + 6, 24))
            image.set_at((x + 5, 23), RUST[1])
        scratch(image, 15, 27, 4)
    else:
        image.fill((11, 57, 38))
        pygame.draw.rect(image, (22, 151, 70), (0, 3, 32, 12))
        pygame.draw.rect(image, (11, 91, 48), (0, 15, 32, 7))
        pygame.draw.line(image, GREEN, (0, 1), (31, 1))
        pygame.draw.line(image, (148, 255, 138), (0, 2), (31, 2))
        for x, y in ((5, 10), (18, 6), (27, 15), (13, 21)):
            pygame.draw.circle(image, (11, 57, 38), (x, y), 3)
            pygame.draw.circle(image, GREEN, (x, y - 1), 2, 1)
            image.set_at((x - 1, y - 2), (218, 255, 201))
        pygame.draw.line(image, INK, (0, 31), (31, 31))
    return image


def terrain(layout: Sequence[str], level: int) -> pygame.Surface:
    cols, rows = max(map(len, layout)), len(layout)
    image = canvas((cols * TILE, rows * TILE))
    theme = THEMES[theme_index(level)]

    def solid(x: int, y: int) -> bool:
        return 0 <= y < rows and 0 <= x < len(layout[y]) and layout[y][x] == "#"

    for row, cells in enumerate(layout):
        for col, symbol in enumerate(cells):
            tile = None
            if symbol == "#":
                tile = stone(
                    theme,
                    col,
                    row,
                    (
                        not solid(col, row - 1),
                        not solid(col - 1, row),
                        not solid(col + 1, row),
                        not solid(col, row + 1),
                    ),
                )
            elif symbol == "H":
                tile = ladder()
            elif symbol in ("^", "~"):
                tile = hazard(symbol)
            if tile is not None:
                image.blit(tile, (col * TILE, row * TILE))
    return image


def explorer(pose: str = "idle", frame: int = 0) -> pygame.Surface:
    image = canvas((24, 32))
    body = canvas((24, 26))
    # Red helmet, yellow trim/skin, magenta overalls and red work boots.
    # More pixels describe those forms; the character keeps its small silhouette.
    pygame.draw.polygon(
        body,
        INK,
        [(7, 1), (16, 1), (19, 4), (19, 7), (22, 7), (22, 10), (3, 10), (3, 7), (5, 7), (5, 4)],
    )
    pygame.draw.polygon(body, (165, 32, 59), [(7, 2), (16, 2), (18, 4), (18, 8), (6, 8), (6, 4)])
    pygame.draw.rect(body, (208, 48, 73), (8, 2, 8, 4))
    pygame.draw.line(body, (242, 104, 115), (9, 2), (14, 2))
    pygame.draw.line(body, (111, 22, 52), (6, 6), (18, 6))
    pygame.draw.line(body, YELLOW, (4, 8), (20, 8))
    pygame.draw.line(body, GOLD[1], (4, 9), (20, 9))
    # Stable equipment wear across poses avoids flickering dirt during animation.
    pygame.draw.line(body, (111, 22, 52), (14, 5), (17, 5))
    body.set_at((15, 6), (242, 104, 115))
    body.set_at((6, 8), GOLD[0])
    pygame.draw.line(body, GOLD[2], (11, 8), (13, 8))
    pygame.draw.rect(body, INK, (16, 1, 5, 5))
    pygame.draw.rect(body, STEEL[3], (17, 2, 3, 3))
    body.set_at((18, 2), WHITE)
    pygame.draw.polygon(
        body, INK, [(7, 10), (18, 10), (18, 12), (20, 12), (20, 14), (18, 14), (17, 16), (7, 16)]
    )
    pygame.draw.rect(body, (250, 194, 110), (8, 10, 10, 5))
    pygame.draw.rect(body, (255, 220, 147), (10, 10, 7, 2))
    pygame.draw.line(body, GOLD[1], (8, 10), (8, 13))
    pygame.draw.line(body, GOLD[4], (18, 12), (19, 12))
    pygame.draw.rect(body, INK, (16 if pose != "idle_look" else 14, 10, 1, 2))
    pygame.draw.line(body, STEEL[2], (15, 14), (18, 14))
    pygame.draw.line(body, WHITE, (11, 15), (16, 15))
    pygame.draw.polygon(body, INK, [(7, 16), (18, 16), (20, 19), (19, 24), (6, 24), (5, 19)])
    pygame.draw.rect(body, PINK[1], (7, 17, 11, 6))
    pygame.draw.rect(body, PINK[2], (10, 17, 8, 4))
    pygame.draw.line(body, PINK[4], (9, 17), (16, 17))
    pygame.draw.line(body, MAGENTA, (8, 18), (8, 21))
    pygame.draw.lines(body, PINK[0], False, [(10, 17), (11, 21), (15, 21), (16, 17)])
    body.set_at((11, 20), GOLD[3])
    body.set_at((15, 20), GOLD[3])
    pygame.draw.line(body, PINK[0], (15, 21), (17, 20))
    body.set_at((16, 22), PINK[2])
    body.set_at((8, 22), GOLD[1])
    pygame.draw.line(body, INK, (7, 23), (18, 23))
    pygame.draw.rect(body, GOLD[2], (13, 23, 3, 1))
    if pose == "climb":
        for x, y in ((4, 12 + frame % 2 * 4), (19, 16 - frame % 2 * 4)):
            pygame.draw.rect(body, INK, (x - 1, y - 1, 4, 7))
            pygame.draw.rect(body, PINK[2], (x, y + 2, 2, 3))
            pygame.draw.rect(body, GOLD[4], (x, y, 2, 2))
    elif pose in ("fall", "hurt"):
        for x in (2, 19):
            pygame.draw.rect(body, INK, (x, 15, 3, 7))
            pygame.draw.line(body, MAGENTA, (x + 1, 16), (x + 1, 19))
            pygame.draw.line(body, GOLD[4], (x + 1, 20), (x + 1, 21))
    else:
        arm = 1 if pose == "walk" and frame % 4 == 3 else 0
        pygame.draw.rect(body, INK, (4 + arm, 18, 4, 7))
        pygame.draw.line(body, PINK[2], (5 + arm, 19), (5 + arm, 22))
        pygame.draw.rect(body, GOLD[4], (5 + arm, 23, 2, 1))
    if pose == "hurt":
        pygame.draw.line(body, INK, (16, 11), (18, 13))
        pygame.draw.line(body, INK, (18, 11), (16, 13))
    bounce = 1 if pose == "walk" and frame % 4 in (1, 3) else 0
    if pose == "land":
        bounce = 2
    image.blit(body, (0, bounce))
    if pose in ("jump", "shoot_air"):
        legs = ((6, 24, -2), (16, 22, 2))
    elif pose == "climb":
        legs = ((6, 24 + frame % 2, -2), (16, 24 - frame % 2, 2))
    elif pose == "land":
        legs = ((5, 27, -2), (17, 27, 1))
    else:
        offsets = ((0, 0), (-2, 2), (0, 0), (2, -2))
        left, right = offsets[frame % 4] if pose == "walk" else (0, 0)
        legs = ((8 + left, 25, -1), (16 + right, 25, 1))
        if pose == "walk" and frame % 4 == 2:
            legs = ((10, 24, -1), (14, 25, 1))
    for x, y, boot in legs:
        height = 5 if pose not in ("jump", "climb", "shoot_air") else 3
        pygame.draw.rect(image, INK, (x - 1, y, 5, height + 2))
        pygame.draw.rect(image, PINK[1], (x, y, 3, height))
        pygame.draw.line(image, PINK[2], (x, y), (x, y + height - 1))
        image.set_at((x + 1, min(29, y + height - 1)), PINK[0])
        pygame.draw.rect(image, (102, 20, 44), (x + boot - 1, min(30, y + height), 5, 2))
        pygame.draw.line(
            image,
            (208, 48, 73),
            (x + boot, min(30, y + height)),
            (x + boot + 2, min(30, y + height)),
        )
        if pose not in ("shoot_air", "jump", "climb"):
            image.set_at((x + boot, 31), (111, 22, 52))
    # Keep the carried raygun readable while standing, walking and airborne.
    # Raise it to the projectile's height to fire; stow it to climb or recover.
    if pose not in ("climb", "hurt"):
        gun_y = 16 if pose in ("shoot", "shoot_air") else 20 + bounce
        pygame.draw.rect(image, INK, (13, gun_y, 11, 5))
        pygame.draw.rect(image, INK, (14, gun_y + 3, 4, 4))
        pygame.draw.rect(image, STEEL[2], (14, gun_y + 1, 10, 2))
        pygame.draw.line(image, STEEL[4], (14, gun_y + 1), (22, gun_y + 1))
        pygame.draw.rect(image, CYAN, (21, gun_y + 2, 2, 1))
        image.set_at((17, gun_y + 1), STEEL[1])
        pygame.draw.line(image, GOLD[4], (14, gun_y + 4), (17, gun_y + 4))
        image.set_at((15, gun_y + 5), STEEL[1])
    return image


def crystal(color: Color, glint: bool = False) -> pygame.Surface:
    image = canvas()
    dark, shade, body, light, gleam = GEM_RAMPS[color]
    pygame.draw.polygon(image, INK, [(7, 3), (24, 3), (31, 12), (16, 31), (0, 12)])
    pygame.draw.polygon(image, body, [(8, 4), (23, 4), (29, 12), (16, 28), (2, 12)])
    pygame.draw.polygon(image, light, [(8, 4), (15, 4), (11, 11), (3, 11)])
    pygame.draw.polygon(image, gleam, [(16, 4), (23, 4), (21, 11), (12, 11)])
    pygame.draw.polygon(image, shade, [(23, 5), (29, 12), (22, 12)])
    pygame.draw.polygon(image, light, [(3, 13), (11, 13), (16, 27)])
    pygame.draw.polygon(image, shade, [(12, 13), (21, 13), (16, 27)])
    pygame.draw.polygon(image, dark, [(22, 13), (28, 13), (17, 27)])
    pygame.draw.lines(image, WHITE, False, [(3, 10), (8, 4), (22, 4), (26, 9)])
    pygame.draw.line(image, gleam, (3, 12), (28, 12))
    pygame.draw.line(image, gleam, (4, 14), (14, 25))
    pygame.draw.line(image, light, (22, 14), (18, 22))
    # Tiny inner reflections describe crystal; the pickup stays clean and bright.
    pygame.draw.lines(image, light, False, [(15, 15), (17, 16), (16, 20)])
    image.set_at((24, 15), body)
    image.set_at((9, 7), gleam)
    if glint:
        pygame.draw.line(image, WHITE, (8, 0), (8, 8))
        pygame.draw.line(image, WHITE, (4, 4), (12, 4))
        pygame.draw.rect(image, WHITE, (7, 3, 3, 3))
    return image


def machine(kind: str, used: bool = False, key: str = "red") -> pygame.Surface:
    image = canvas()
    tint = RED if key == "red" else BLUE
    if kind == "switch":
        metal(image, (5, 15, 23, 17))
        pygame.draw.rect(image, STEEL[0], (9, 18, 14, 8))
        pygame.draw.rect(image, INK, (12, 17, 8, 10))
        end = (23, 6) if used else (9, 6)
        pygame.draw.line(image, INK, (16, 25), end, 5)
        pygame.draw.line(image, STEEL[3], (16, 24), end, 3)
        pygame.draw.line(image, STEEL[4], (15, 24), (end[0] - 1, 7))
        pygame.draw.rect(image, INK, (end[0] - 4, end[1] - 2, 9, 5))
        pygame.draw.rect(image, tint, (end[0] - 3, end[1] - 1, 7, 3))
        image.set_at((end[0] - 2, end[1] - 1), WHITE)
        pygame.draw.rect(image, GREEN if used else tint, (11, 28, 11, 2))
        for xx in (7, 24):
            bolt(image, xx, 26)
        return image
    if kind == "air":
        # A pressure vessel with a neck and external pipework, not another door.
        metal(image, (10, 3, 15, 29))
        metal(image, (13, 0, 8, 5), worn=False)
        pygame.draw.line(image, RED, (14, 1), (19, 1))
        pygame.draw.line(image, STEEL[0], (6, 10), (6, 27), 5)
        pygame.draw.line(image, STEEL[3], (5, 10), (5, 25), 2)
        pygame.draw.line(image, STEEL[2], (6, 26), (11, 26), 3)
        pygame.draw.line(image, STEEL[0], (27, 8), (27, 23), 3)
        pygame.draw.line(image, STEEL[3], (27, 8), (27, 21))
        pygame.draw.rect(image, INK, (12, 7, 11, 7))
        pygame.draw.rect(image, RED, (14, 9, 7, 3))
        pygame.draw.line(image, STEEL[4], (14, 8), (20, 8))
        pygame.draw.rect(image, INK, (6, 16, 24, 9))
        lettering(image, "AIR", 9, 17)
        for xx, color in ((12, RED), (16, GOLD[2]), (20, GREEN)):
            image.set_at((xx, 28), color)
        pygame.draw.line(image, RUST[0], (12, 30), (15, 30))
        image.set_at((13, 29), RUST[1])
        return image
    metal(image, (4, 1, 25, 31))
    for x, y in ((6, 3), (24, 3), (6, 26), (24, 26)):
        bolt(image, x, y)
    pygame.draw.rect(image, STEEL[0], (8, 7, 17, 18))
    pygame.draw.line(image, STEEL[1], (5, 11), (5, 20))
    pygame.draw.line(image, STEEL[3], (27, 13), (27, 22))
    pygame.draw.line(image, RUST[0], (24, 29), (27, 29))
    image.set_at((25, 28), RUST[1])
    if kind in ("door", "exit"):
        pygame.draw.rect(image, INK, (9, 8, 15, 24))
        pygame.draw.line(image, YELLOW if kind == "exit" else tint, (9, 6), (23, 6), 2)
        if used:
            pygame.draw.rect(image, (11, 57, 38), (11, 10, 11, 22))
            pygame.draw.lines(image, GREEN, False, [(14, 17), (18, 21), (14, 25)], 2)
            pygame.draw.line(image, WHITE, (13, 21), (18, 21))
        else:
            pygame.draw.rect(image, STEEL[1], (10, 9, 13, 23))
            pygame.draw.rect(image, STEEL[2], (11, 10, 5, 21))
            pygame.draw.line(image, STEEL[3], (11, 10), (11, 30))
            pygame.draw.line(image, STEEL[0], (17, 9), (17, 31))
            pygame.draw.rect(image, tint, (16, 19, 4, 2))
            for yy in (12, 25):
                pygame.draw.line(image, STEEL[0], (10, yy), (15, yy))
                pygame.draw.line(image, STEEL[3], (11, yy + 1), (14, yy + 1))
            scratch(image, 19, 26, 2, STEEL[0], STEEL[2])
            pygame.draw.line(image, RUST[0], (11, 30), (14, 30))
            image.set_at((12, 29), RUST[1])
        for yy in (11, 23):
            metal(image, (4, yy, 4, 5), worn=False)
    else:
        color = {"power_shot": YELLOW, "gravity": MAGENTA, "freeze": CYAN}[kind]
        pygame.draw.rect(image, INK, (10, 9, 13, 14))
        if kind == "power_shot":
            pygame.draw.polygon(
                image, color, [(16, 10), (12, 17), (16, 17), (14, 22), (21, 14), (17, 14), (19, 10)]
            )
        elif kind == "gravity":
            pygame.draw.lines(image, color, False, [(12, 16), (16, 11), (21, 16)], 2)
            pygame.draw.line(image, color, (16, 13), (16, 21), 2)
        else:
            for start, end in (((16, 10), (16, 21)), ((12, 12), (20, 20)), ((12, 20), (20, 12))):
                pygame.draw.line(image, color, start, end)
        pygame.draw.line(image, color, (11, 27), (21, 27), 2)
        pygame.draw.line(image, color, (5, 11), (5, 19))
        image.set_at((5, 17), STEEL[0])
        for xx in (10, 14, 18, 22):
            pygame.draw.line(image, STEEL[1], (xx, 25), (xx + 1, 25))
    return image


def player_hit_pose(source: pygame.Surface) -> pygame.Surface:
    """Turn an action pose white without changing its outline or alpha."""
    image = source.copy()
    for y in range(image.get_height()):
        for x in range(image.get_width()):
            color = image.get_at((x, y))
            if color.a and (color.r, color.g, color.b) != INK:
                image.set_at((x, y), WHITE)
    return image


def art_sprites() -> dict[str, pygame.Surface]:
    images = {
        f"mylo_{pose}": explorer(pose)
        for pose in (
            "idle",
            "jump",
            "fall",
            "land",
            "hurt",
            "climb",
            "shoot",
            "shoot_air",
            "idle_look",
        )
    }
    for frame in range(4):
        images[f"mylo_walk_{frame + 1}"] = explorer("walk", frame)
        images[f"mylo_run_{frame}"] = explorer("walk", frame)
    for frame in range(2):
        images[f"mylo_climb_{frame}"] = explorer("climb", frame)
    images.update({name + "_hit": player_hit_pose(image) for name, image in images.items()})
    for name, color in (
        ("blue", BLUE),
        ("green", GREEN),
        ("yellow", YELLOW),
        ("red", RED),
        ("purple", MAGENTA),
    ):
        images[f"crystal_{name}"] = crystal(color)
        images[f"crystal_{name}_glint"] = crystal(color, True)
    for key in ("red", "blue"):
        images[f"door_{key}"] = machine("door", key=key)
        for used in (False, True):
            images[f"switch_{key}_{'on' if used else 'off'}"] = machine("switch", used, key)
    images.update(
        {
            "exit_locked": machine("exit"),
            "exit_open": machine("exit", True),
            "air_tank": machine("air"),
        }
    )
    for name in ("power_shot", "gravity", "freeze"):
        images[name] = machine(name)
    ammo = canvas()
    metal(ammo, (4, 12, 24, 16))
    pygame.draw.rect(ammo, STEEL[0], (7, 16, 18, 9))
    pygame.draw.line(ammo, STEEL[0], (5, 14), (26, 14))
    for xx in (6, 24):
        bolt(ammo, xx, 25)
    for x in (9, 15, 21):
        pygame.draw.rect(ammo, GOLD[1], (x, 19, 3, 5))
        pygame.draw.line(ammo, YELLOW, (x, 18), (x + 2, 18))
        pygame.draw.line(ammo, GOLD[4], (x, 19), (x, 22))
    images["ammo"] = ammo
    treasure = canvas()
    pygame.draw.circle(treasure, INK, (16, 17), 12)
    pygame.draw.circle(treasure, GOLD[1], (16, 16), 10)
    pygame.draw.circle(treasure, GOLD[2], (15, 15), 9)
    pygame.draw.circle(treasure, YELLOW, (15, 15), 7, 1)
    lettering(treasure, "$", 13, 11, GOLD[0])
    pygame.draw.line(treasure, GOLD[4], (10, 8), (16, 8))
    pygame.draw.line(treasure, GOLD[0], (19, 22), (22, 19))
    pygame.draw.line(treasure, GOLD[1], (10, 21), (13, 23))
    treasure.set_at((12, 22), GOLD[4])
    images["treasure"] = treasure
    images.update(mechanism_sprites())
    images.update(projectile_sprites())
    for name in ("poof", "sparkle"):
        feedback = canvas()
        if name == "sparkle":
            pygame.draw.polygon(
                feedback,
                GOLD[2],
                [(16, 3), (19, 13), (29, 16), (19, 19), (16, 29), (13, 19), (3, 16), (13, 13)],
            )
            pygame.draw.line(feedback, WHITE, (16, 8), (16, 24))
            pygame.draw.line(feedback, GOLD[4], (8, 16), (24, 16))
        else:
            for x, y, radius in ((10, 16, 6), (16, 11, 7), (21, 17, 6), (14, 21, 6)):
                pygame.draw.circle(feedback, STEEL[1], (x, y), radius)
                pygame.draw.circle(feedback, STEEL[3], (x - 1, y - 1), radius - 2)
            pygame.draw.line(feedback, STEEL[4], (12, 6), (17, 6))
        images[name] = feedback
    images.update(extra_sprites())
    images.update(creature_sprites())
    images.update(door_sprites())
    images.update(
        {
            "door_locked": images["door_red"],
            "door_open": images["exit_open"],
            "switch_off": images["switch_red_off"],
            "switch_on": images["switch_red_on"],
            "spikes": hazard("^"),
            "acid": hazard("~"),
        }
    )
    images.update(pickup_sprites())
    images.update(scenery_sprites())
    images.update(equipment_sprites())
    images.update(secret_sprites())
    return images


def backdrop(theme: Theme, size: tuple[int, int] = (640, 384)) -> pygame.Surface:
    if theme == THEMES[5]:
        return ribbed_wall(size)
    if theme == THEMES[6]:
        return pipe_wall(size)
    if theme == THEMES[7]:
        return purple_planks(size)
    image = canvas(size)
    image.fill(theme.wall_shadow)
    if theme == THEMES[4]:
        image.fill(theme.wall)
        for yy in range(0, size[1], 32):
            for xx in range(0, size[0], 32):
                pygame.draw.lines(
                    image,
                    theme.wall_light,
                    True,
                    [(xx + 16, yy + 2), (xx + 30, yy + 16), (xx + 16, yy + 30), (xx + 2, yy + 16)],
                )
                pygame.draw.lines(
                    image,
                    theme.wall_shadow,
                    False,
                    [(xx + 3, yy + 17), (xx + 16, yy + 30), (xx + 29, yy + 17)],
                )
        return image
    if theme == THEMES[2]:
        # Low-contrast rock masonry behind the timber, rather than a star field.
        for row, y in enumerate(range(0, size[1], 16)):
            for col, x in enumerate(range(-16 if row % 2 else 0, size[0], 32)):
                tile = canvas((32, 16))
                pygame.draw.polygon(
                    tile,
                    theme.wall_light,
                    [(3, 1), (26, 1), (30, 4), (29, 12), (26, 14), (3, 14), (1, 11), (1, 4)],
                )
                pygame.draw.polygon(
                    tile,
                    theme.wall,
                    [(4, 3), (25, 3), (28, 5), (27, 11), (25, 12), (4, 12), (3, 10), (3, 5)],
                )
                if (col + row * 3) % 7 == 0:
                    pygame.draw.line(tile, theme.wall_shadow, (20, 4), (23, 8))
                image.blit(tile, (x, y))
        return image
    if theme == THEMES[3]:
        return rust_brick_wall(size)
    # The red room's panels span two 32px collision cells in the reference.
    # Keep that larger rhythm while actors and terrain retain their native grid.
    for y in range(0, size[1], 64):
        for x in range(0, size[0], 64):
            tile = canvas((64, 64))
            tile.fill(theme.wall_shadow)
            pygame.draw.polygon(
                tile,
                theme.wall_light,
                [(7, 1), (55, 1), (62, 8), (62, 55), (55, 62), (8, 62), (1, 55), (1, 8)],
            )
            pygame.draw.polygon(
                tile,
                theme.wall,
                [(9, 4), (53, 4), (59, 10), (59, 53), (53, 59), (10, 59), (4, 53), (4, 10)],
            )
            pygame.draw.lines(
                tile, theme.wall_shadow, False, [(4, 51), (4, 10), (10, 4), (51, 4)], 2
            )
            pygame.draw.line(tile, theme.wall_shadow, (9, 61), (53, 61))
            variant = (x // 64 * 7 + y // 64 * 13) % 11
            if theme == THEMES[1]:
                # Raised diagonal plates, with dents confined to an occasional seam.
                pygame.draw.line(tile, theme.wall_shadow, (56, 6), (6, 56), 2)
                pygame.draw.line(tile, theme.wall_light, (57, 8), (8, 57))
                if variant in (2, 7):
                    pygame.draw.lines(tile, theme.wall_shadow, False, [(9, 40), (13, 43), (11, 48)])
                    pygame.draw.line(tile, theme.wall_light, (10, 40), (13, 42))
            else:
                # Restrained paint loss at the beveled corners keeps broad red faces.
                if variant in (1, 4, 8):
                    pygame.draw.line(tile, theme.wall_shadow, (51, 58), (56, 53))
                    pygame.draw.line(tile, STEEL[1], (52, 59), (57, 54))
                    tile.set_at((55, 55), theme.wall_light)
                elif variant == 6:
                    pygame.draw.lines(tile, theme.wall_shadow, False, [(12, 8), (10, 12), (13, 14)])
                    tile.set_at((12, 13), RUST[1])
            image.blit(tile, (x, y))
    return image


def title_scene() -> pygame.Surface:
    """A pixel arcade vignette using the same miner, dinosaur and cave artwork."""
    image = backdrop(THEMES[0], (160, 112))
    for col in range(5):
        image.blit(stone(THEMES[0], col, 3, (True, False, False, False)), (col * 32, 96))
    for row in range(3):
        image.blit(pipe_segment(), (128, row * 32))
    image.blit(crate(), (4, 64))
    image.blit(warning_plate(), (113, 8))
    image.blit(crystal(YELLOW, True), (15, 14))
    image.blit(crystal(BLUE), (121, 31))
    image.blit(explorer("jump"), (44, 48))
    image.blit(pygame.transform.flip(creature_sprites()["dinosaur_enemy_1"], True, False), (91, 32))
    pygame.draw.rect(image, INK, (0, 0, 160, 112), 2)
    pygame.draw.line(image, STEEL[3], (2, 0), (157, 0))
    pygame.draw.line(image, STEEL[1], (0, 2), (0, 109))
    return image


def dressing_placements(layout: Sequence[str], level: int) -> list[DressingPlacement]:
    """Room fixtures use clear bays, with breathing room around gameplay objects."""
    priority_signs: list[DressingPlacement] = []
    if level == 0 and len(layout) > 22 and layout[22][8:10] == "t^":
        # Keep the opening trench warning above the takeoff approach. Scenery
        # otherwise occupies its clear bay before automatic warning placement.
        priority_signs.append({"sprite": "danger_sign", "col": 10, "row": 20})
    if level == 8 and len(layout) > 12 and layout[6][1:4] == "##H" and layout[12][4] == "#":
        # Warn about the top-vault dinosaur before decorating its small room.
        priority_signs.append({"sprite": "danger_sign", "col": 3, "row": 1})
    placements = scenery_placements(layout, theme_index(level), priority_signs)
    reserved = {
        (col + dx, row)
        for placement in placements
        for col, row in placement_cells(placement)
        for dx in (-1, 0, 1)
    }
    reserved.update(service_pipe_cells(layout))
    reserved.update(door_headroom_cells(layout))
    rows = len(layout)

    def empty(c: int, r: int) -> bool:
        return (
            0 <= r < rows
            and 0 <= c < len(layout[r])
            and layout[r][c] == "."
            and (c, r) not in reserved
        )

    def clear_bay(c: int, r: int) -> bool:
        return all(empty(c + dx, r) for dx in (-1, 0, 1))

    for row, cells in enumerate(layout):
        floor_bays = [
            col
            for col, symbol in enumerate(cells)
            if symbol == "."
            and row + 1 < rows
            and layout[row + 1][col] == "#"
            and clear_bay(col, row)
        ]
        occupied: set[int] = set()
        # One purpose-built fixture per eight clear cells in a corridor.
        for index in range(0, len(floor_bays), 8):
            group = floor_bays[index : index + 8]
            col = group[(level + row) % len(group)]
            kind = ("generator", "crate", "terminal", "mine_sign")[(level + row + index // 8) % 4]
            if kind == "crate" and theme_index(level) in (0, 3, 5, 7):
                kind = "barrel"
            placements.append({"sprite": kind, "col": col, "row": row})
            occupied.update((col - 1, col, col + 1))
        for col, symbol in enumerate(cells):
            if not empty(col, row) or col in occupied:
                continue
            above = row > 0 and layout[row - 1][col] == "#"
            if above and clear_bay(col, row) and (col + row * 3 + level) % 13 == 0:
                kind = "lamp" if (col + level) % 2 else "warning_plate"
                placements.append({"sprite": kind, "col": col, "row": row})
    return placements


def dressing(layout: Sequence[str], level: int) -> tuple[pygame.Surface, list[dict[str, int]]]:
    cols, rows = max(map(len, layout)), len(layout)
    image = canvas((cols * 32, rows * 32))
    props = {
        "pipe": pipe_segment(),
        "crate": crate(),
        "warning_plate": warning_plate(),
        "generator": apparatus("generator"),
        "terminal": apparatus("terminal"),
        "mine_sign": sign("mine_sign"),
        "lamp": sign("lamp"),
        **scenery_sprites(),
        **equipment_sprites(),
    }
    # Ceiling lamps are short inverted housings rather than floor pedestals.
    lamp = canvas()
    lamp.blit(pygame.transform.flip(props["lamp"], False, True), (0, -8))
    props["lamp"] = lamp
    for placement in dressing_placements(layout, level):
        image.blit(
            props[placement["sprite"]],
            (placement["col"] * 32, placement["row"] * 32),
        )
    connected_pipes(layout, image)
    return image, []


def vignette() -> pygame.Surface:
    return canvas((1, 1))
