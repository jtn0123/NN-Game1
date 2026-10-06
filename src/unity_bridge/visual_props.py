"""Native pixel props: battered equipment, timber, glass and organic creatures."""

from __future__ import annotations

import pygame

from .visual_materials import (
    BLUE,
    CYAN,
    GOLD,
    GREEN,
    INK,
    MAGENTA,
    RED,
    RUST,
    STEEL,
    WHITE,
    YELLOW,
    bolt,
    canvas,
    lettering,
    metal,
    raygun,
    scratch,
)


def crate() -> pygame.Surface:
    image = canvas()
    pygame.draw.rect(image, INK, (3, 12, 26, 20))
    pygame.draw.rect(image, GOLD[1], (4, 13, 24, 18))
    pygame.draw.rect(image, GOLD[0], (6, 15, 20, 15))
    for yy in (15, 20, 25):
        pygame.draw.rect(image, (130, 65, 28), (7, yy + 1, 18, 3))
        pygame.draw.line(image, GOLD[2], (7, yy), (24, yy))
        pygame.draw.line(image, GOLD[0], (11, yy + 2), (20, yy + 2))
    pygame.draw.line(image, GOLD[2], (4, 13), (26, 13))
    pygame.draw.line(image, GOLD[1], (5, 14), (26, 29), 3)
    pygame.draw.line(image, GOLD[2], (5, 14), (26, 29))
    scratch(image, 8, 27, 4, GOLD[0], GOLD[1])
    pygame.draw.lines(image, GOLD[0], False, [(19, 15), (17, 18), (18, 19)])
    pygame.draw.line(image, GOLD[0], (4, 25), (4, 29))
    image.set_at((5, 26), GOLD[2])
    for xx, yy in ((5, 14), (26, 14), (5, 29), (25, 28)):
        pygame.draw.rect(image, INK, (xx, yy, 2, 2))
        image.set_at((xx, yy), STEEL[3])
    return image


def pipe_segment() -> pygame.Surface:
    image = canvas()
    pygame.draw.rect(image, INK, (21, 0, 9, 32))
    for offset, color in enumerate((STEEL[0], STEEL[1], STEEL[3], STEEL[4], STEEL[2], STEEL[1])):
        pygame.draw.line(image, color, (22 + offset, 0), (22 + offset, 31))
    for yy in range(1, 32, 4):
        pygame.draw.line(image, STEEL[3], (22, yy), (27, yy))
        pygame.draw.line(image, STEEL[0], (22, yy + 1), (27, yy + 1))
    for yy in (7, 24):
        pygame.draw.rect(image, STEEL[0], (20, yy, 11, 4))
        pygame.draw.line(image, STEEL[3], (21, yy), (29, yy))
        pygame.draw.line(image, STEEL[1], (21, yy + 2), (29, yy + 2))
        image.set_at((29, yy + 1), GOLD[1])
    pygame.draw.line(image, RUST[0], (22, 14), (23, 18))
    pygame.draw.line(image, RUST[1], (23, 15), (24, 17))
    image.set_at((24, 16), RUST[2])
    return image


def warning_plate() -> pygame.Surface:
    image = canvas()
    pygame.draw.rect(image, INK, (3, 4, 26, 13))
    pygame.draw.rect(image, (142, 22, 37), (4, 5, 24, 11))
    pygame.draw.line(image, RED, (4, 5), (27, 5))
    pygame.draw.line(image, (87, 17, 37), (4, 15), (27, 15))
    lettering(image, "!", 7, 7, YELLOW)
    pygame.draw.lines(image, GOLD[2], False, [(17, 8), (22, 10), (17, 12)])
    for xx in (4, 26):
        image.set_at((xx, 6), STEEL[3])
        image.set_at((xx, 14), STEEL[1])
    pygame.draw.line(image, STEEL[1], (13, 5), (15, 5))
    pygame.draw.line(image, RUST[1], (23, 15), (25, 15))
    scratch(image, 13, 13, 3, (87, 17, 37), (216, 65, 69))
    return image


def _wood_handle(image: pygame.Surface, start: tuple[int, int], end: tuple[int, int]) -> None:
    pygame.draw.line(image, INK, start, end, 5)
    pygame.draw.line(image, GOLD[0], start, end, 3)
    pygame.draw.line(image, GOLD[1], start, end)


def tool(kind: str) -> pygame.Surface:
    image = canvas()
    if kind == "pickaxe":
        _wood_handle(image, (8, 29), (20, 10))
        pygame.draw.polygon(
            image, INK, [(8, 7), (15, 3), (25, 5), (30, 10), (25, 8), (15, 8), (8, 13)]
        )
        pygame.draw.lines(image, STEEL[3], False, [(9, 9), (16, 5), (24, 6), (28, 9)], 2)
        pygame.draw.line(image, STEEL[1], (14, 8), (24, 8), 2)
        pygame.draw.line(image, RUST[1], (18, 7), (21, 7))
        pygame.draw.line(image, GOLD[2], (10, 26), (12, 23))
    elif kind == "hammer_marker":
        _wood_handle(image, (14, 30), (14, 8))
        metal(image, (4, 3, 23, 9))
        pygame.draw.rect(image, STEEL[0], (4, 6, 5, 4))
        pygame.draw.line(image, STEEL[4], (4, 4), (6, 4))
        pygame.draw.line(image, GOLD[2], (13, 21), (15, 21))
        pygame.draw.line(image, GOLD[0], (12, 26), (16, 26))
    else:
        metal(image, (3, 10, 26, 12))
        pygame.draw.rect(image, (20, 45, 52), (5, 13, 21, 6))
        pygame.draw.line(image, STEEL[4], (7, 11), (19, 11))
        pygame.draw.line(image, CYAN, (23, 14), (26, 14))
        scratch(image, 7, 16, 5)
        pygame.draw.rect(image, GOLD[0], (10, 22, 7, 8))
        for yy in (23, 26):
            pygame.draw.line(image, GOLD[1], (11, yy), (15, yy))
    return image


def sign(kind: str) -> pygame.Surface:
    image = canvas()
    if kind == "mine_sign":
        _wood_handle(image, (16, 30), (16, 14))
        pygame.draw.rect(image, INK, (1, 4, 30, 16))
        pygame.draw.rect(image, GOLD[1], (2, 5, 28, 14))
        pygame.draw.rect(image, GOLD[2], (3, 6, 26, 11))
        pygame.draw.line(image, YELLOW, (4, 6), (24, 6))
        pygame.draw.lines(image, INK, False, [(12, 8), (17, 11), (12, 14)], 2)
        pygame.draw.line(image, INK, (6, 11), (16, 11), 2)
        scratch(image, 20, 15, 5, GOLD[0], GOLD[1])
        pygame.draw.line(image, GOLD[0], (3, 6), (3, 9))
        image.set_at((4, 8), GOLD[1])
        bolt(image, 3, 15)
    elif kind == "warning_post":
        pygame.draw.rect(image, INK, (13, 16, 5, 15))
        pygame.draw.line(image, STEEL[3], (14, 17), (14, 30))
        metal(image, (7, 28, 17, 4))
        pygame.draw.polygon(image, INK, [(15, 1), (30, 21), (1, 21)])
        pygame.draw.polygon(image, GOLD[2], [(15, 4), (27, 19), (4, 19)])
        pygame.draw.line(image, YELLOW, (15, 5), (7, 16))
        lettering(image, "!", 13, 10, INK)
        pygame.draw.line(image, GOLD[0], (23, 17), (25, 18))
        image.set_at((7, 18), RUST[1])
    else:
        metal(image, (5, 26, 22, 6))
        metal(image, (10, 19, 12, 9))
        pygame.draw.polygon(image, INK, [(10, 4), (21, 4), (25, 12), (22, 20), (9, 20), (6, 12)])
        pygame.draw.polygon(
            image, GOLD[1], [(11, 5), (20, 5), (23, 12), (21, 18), (10, 18), (8, 12)]
        )
        pygame.draw.rect(image, YELLOW, (11, 7, 9, 7))
        pygame.draw.line(image, GOLD[4], (12, 7), (17, 7))
        pygame.draw.line(image, GOLD[0], (9, 15), (22, 15))
        if kind == "lamp":
            pygame.draw.lines(image, STEEL[2], False, [(10, 5), (11, 2), (20, 2), (21, 5)], 2)
            pygame.draw.line(image, STEEL[0], (15, 5), (15, 18))
    return image


def apparatus(kind: str) -> pygame.Surface:
    image = canvas()
    if kind == "pipe_stack":
        for yy in (3, 13, 23):
            metal(image, (2, yy, 29, 7))
            pygame.draw.line(image, (17, 126, 51), (5, yy + 2), (23, yy + 2))
            pygame.draw.line(image, (12, 67, 42), (5, yy + 4), (23, yy + 4))
            metal(image, (24, yy - 1, 6, 9), worn=False)
            pygame.draw.line(image, RUST[1], (23, yy + 5), (26, yy + 5))
        return image
    metal(image, (3, 4, 27, 28))
    for xx, yy in ((5, 6), (25, 6), (5, 27), (25, 27)):
        bolt(image, xx, yy)
    if kind == "generator":
        pygame.draw.rect(image, INK, (8, 8, 17, 12))
        for xx in (10, 14, 18, 22):
            pygame.draw.line(image, GOLD[1], (xx, 9), (xx, 18), 2)
            pygame.draw.line(image, GOLD[4], (xx, 10), (xx, 13))
        pygame.draw.line(image, GOLD[0], (9, 18), (24, 18))
        pygame.draw.rect(image, INK, (8, 23, 17, 5))
        pygame.draw.line(image, GREEN, (10, 24), (15, 24), 2)
        pygame.draw.line(image, RED, (19, 24), (22, 24))
        pygame.draw.rect(image, STEEL[0], (7, 1, 9, 5))
    elif kind == "terminal":
        pygame.draw.rect(image, INK, (6, 8, 21, 14))
        pygame.draw.rect(image, (10, 67, 84), (8, 10, 17, 10))
        pygame.draw.lines(
            image, CYAN, False, [(9, 17), (11, 17), (13, 12), (15, 17), (18, 14), (23, 14)]
        )
        pygame.draw.line(image, (46, 208, 208), (9, 11), (14, 11))
        for xx in (9, 13, 17, 21):
            pygame.draw.line(image, STEEL[0], (xx, 25), (xx + 1, 25), 2)
        image.set_at((23, 27), GREEN)
    elif kind == "vacuum":
        pygame.draw.circle(image, INK, (16, 16), 10)
        pygame.draw.circle(image, STEEL[3], (16, 16), 8)
        pygame.draw.circle(image, (8, 82, 107), (16, 16), 6)
        pygame.draw.circle(image, CYAN, (16, 16), 4, 1)
        pygame.draw.line(image, STEEL[0], (15, 11), (17, 20), 2)
        pygame.draw.line(image, STEEL[0], (11, 17), (20, 15), 2)
        for xx in (8, 13, 18, 23):
            pygame.draw.line(image, STEEL[0], (xx, 28), (xx + 1, 28))
    elif kind == "power":
        pygame.draw.rect(image, (87, 17, 37), (7, 9, 19, 17))
        pygame.draw.rect(image, RED, (8, 10, 17, 12))
        pygame.draw.rect(image, WHITE, (14, 12, 5, 9))
        pygame.draw.rect(image, WHITE, (11, 14, 11, 5))
        pygame.draw.line(image, (184, 35, 61), (8, 23), (24, 23))
        pygame.draw.line(image, YELLOW, (11, 28), (20, 28))
    else:
        pygame.draw.rect(image, INK, (8, 8, 17, 17))
        pygame.draw.line(image, CYAN, (10, 8), (22, 8))
        pygame.draw.line(image, GOLD[2], (10, 24), (22, 24))
        pygame.draw.lines(image, YELLOW, False, [(19, 9), (12, 16), (19, 16), (12, 23)], 2)
        for xx in (9, 22):
            pygame.draw.line(image, STEEL[3], (xx, 4), (xx, 0), 2)
    scratch(image, 19, 30, 5)
    return image


def organic(kind: str) -> pygame.Surface:
    image = canvas((32, 24) if kind in ("crawler", "flyer") else (32, 32))
    if kind == "mushroom":
        pygame.draw.polygon(image, INK, [(13, 14), (20, 14), (22, 29), (10, 29)])
        pygame.draw.polygon(image, GOLD[4], [(14, 15), (18, 15), (20, 28), (12, 28)])
        pygame.draw.line(image, GOLD[1], (16, 18), (18, 27))
        pygame.draw.polygon(image, INK, [(1, 16), (3, 8), (10, 3), (21, 3), (29, 9), (31, 17)])
        pygame.draw.polygon(
            image, (184, 35, 61), [(3, 15), (5, 9), (11, 5), (20, 5), (27, 10), (29, 15)]
        )
        pygame.draw.polygon(image, RED, [(5, 12), (10, 6), (18, 5), (24, 8), (25, 11)])
        for xx, yy in ((9, 9), (17, 7), (23, 12)):
            pygame.draw.rect(image, GOLD[4], (xx, yy, 3, 2))
            image.set_at((xx, yy), WHITE)
        pygame.draw.line(image, (87, 17, 37), (4, 16), (28, 16))
        pygame.draw.line(image, (22, 151, 70), (8, 30), (22, 30), 2)
    elif kind == "walking_rock":
        pygame.draw.polygon(
            image, INK, [(4, 11), (10, 4), (20, 3), (28, 11), (29, 23), (22, 28), (6, 27), (2, 20)]
        )
        pygame.draw.polygon(
            image,
            (39, 49, 176),
            [(5, 12), (11, 6), (19, 5), (26, 12), (26, 23), (20, 26), (7, 25), (4, 20)],
        )
        pygame.draw.polygon(image, BLUE, [(6, 12), (12, 7), (19, 6), (22, 10), (16, 14)])
        pygame.draw.lines(image, (20, 23, 85), False, [(21, 9), (18, 14), (21, 18), (19, 24)])
        for xx in (9, 19):
            pygame.draw.rect(image, WHITE, (xx, 15, 3, 3))
            image.set_at((xx + 1, 16), INK)
        pygame.draw.line(image, STEEL[0], (7, 30), (11, 30), 2)
        pygame.draw.line(image, STEEL[0], (21, 30), (25, 30), 2)
        scratch(image, 7, 22, 3, (20, 23, 85), (142, 183, 255))
    elif kind == "crawler":
        pygame.draw.ellipse(image, INK, (1, 5, 30, 16))
        pygame.draw.ellipse(image, (22, 151, 70), (3, 6, 26, 13))
        pygame.draw.ellipse(image, GREEN, (6, 7, 17, 6))
        for xx in (7, 13, 19):
            pygame.draw.line(image, (11, 57, 38), (xx, 12), (xx + 2, 17))
            pygame.draw.line(image, STEEL[0], (xx, 19), (xx - 1, 22), 2)
        pygame.draw.rect(image, WHITE, (23, 9, 4, 3))
        image.set_at((26, 10), INK)
    else:
        center_y = 12 if kind == "flyer" else 16
        pygame.draw.polygon(
            image, INK, [(1, 5), (11, 10), (21, 10), (30, 5), (26, 18), (20, 24), (11, 24), (5, 18)]
        )
        pygame.draw.polygon(image, (64, 24, 84), [(3, 8), (12, 13), (10, 20), (6, 16)])
        pygame.draw.polygon(image, (142, 42, 163), [(29, 8), (20, 13), (22, 20), (26, 16)])
        pygame.draw.line(image, MAGENTA, (4, 8), (11, 13))
        pygame.draw.line(image, (230, 153, 255), (28, 8), (21, 13))
        pygame.draw.ellipse(image, INK, (10, center_y - 4, 13, 15))
        pygame.draw.ellipse(image, MAGENTA, (11, center_y - 3, 11, 12))
        pygame.draw.ellipse(image, WHITE, (12, center_y, 9, 6))
        pygame.draw.rect(image, (8, 82, 107), (16, center_y + 1, 3, 4))
        image.set_at((17, center_y + 1), INK)
        pygame.draw.line(image, YELLOW, (15, center_y + 7), (17, center_y + 7))
    return image


def eye_turret() -> pygame.Surface:
    image = canvas()
    metal(image, (9, 25, 16, 7))
    metal(image, (13, 16, 8, 12))
    pygame.draw.ellipse(image, INK, (2, 4, 29, 17))
    pygame.draw.ellipse(image, STEEL[2], (3, 5, 27, 14))
    pygame.draw.line(image, STEEL[4], (8, 6), (23, 6))
    pygame.draw.ellipse(image, WHITE, (9, 7, 15, 9))
    pygame.draw.rect(image, CYAN, (15, 8, 5, 7))
    pygame.draw.rect(image, INK, (17, 9, 2, 5))
    scratch(image, 5, 15, 4)
    pygame.draw.line(image, RUST[1], (23, 17), (26, 17))
    bolt(image, 5, 9)
    return image


def clear_block() -> pygame.Surface:
    image = canvas()
    pygame.draw.rect(image, INK, (1, 1, 30, 30))
    pygame.draw.rect(image, (8, 82, 107), (2, 2, 28, 28))
    pygame.draw.rect(image, (46, 208, 208), (3, 3, 26, 25))
    pygame.draw.lines(image, CYAN, False, [(3, 27), (3, 3), (27, 3)], 2)
    pygame.draw.polygon(image, (0, 142, 155), [(28, 7), (28, 28), (7, 28)])
    pygame.draw.lines(image, WHITE, False, [(6, 12), (6, 6), (12, 6)])
    pygame.draw.lines(image, (8, 82, 107), False, [(25, 20), (21, 23), (23, 28)])
    pygame.draw.line(image, CYAN, (24, 21), (21, 24))
    return image


def extra_sprites() -> dict[str, pygame.Surface]:
    images = {
        name: apparatus(name)
        for name in ("generator", "terminal", "vacuum", "power", "zapper", "pipe_stack")
    }
    images.update({name: sign(name) for name in ("mine_sign", "warning_post", "beacon", "lamp")})
    images.update(
        {
            name: organic(name)
            for name in ("mushroom", "walking_rock", "crawler", "flyer", "eye_flyer")
        }
    )
    images.update({name: tool(name) for name in ("pickaxe", "hammer_marker", "raygun")})
    images.update({"crate": crate(), "eye_turret": eye_turret(), "clear_block": clear_block()})
    # The standalone weapon and HUD share the same worn metal and grip detail.
    images["raygun"] = canvas()
    images["raygun"].blit(raygun(), (4, 10))
    return images
