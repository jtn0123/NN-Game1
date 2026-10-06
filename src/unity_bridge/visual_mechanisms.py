"""Discrete pixel poses for the reference's hover pads and green floor thorns."""

import pygame

from .visual_materials import CYAN, GOLD, GREEN, INK, RED, STEEL, WHITE, canvas


def hover_lift(frame: int = 0) -> pygame.Surface:
    image = canvas()
    # Square instrument housing, yellow landing lip and two cyan underside struts.
    pygame.draw.rect(image, INK, (0, 0, 32, 24))
    pygame.draw.rect(image, STEEL[1], (2, 2, 28, 20))
    pygame.draw.rect(image, STEEL[0], (4, 4, 24, 16))
    pygame.draw.line(image, STEEL[4], (2, 1), (29, 1))
    pygame.draw.rect(image, GOLD[1], (5, 3, 22, 3))
    pygame.draw.line(image, GOLD[3], (6, 3), (25, 3))
    landing_strip = image.subsurface((4, 3, 24, 4)).copy()
    # A cool bevel surrounds the reference's broad, split instrument face.
    pygame.draw.rect(image, (105, 104, 143), (2, 2, 28, 20))
    pygame.draw.rect(image, (49, 48, 80), (4, 4, 24, 16))
    pygame.draw.line(image, (222, 228, 232), (2, 2), (29, 2))
    pygame.draw.line(image, (169, 169, 194), (2, 3), (2, 21))
    pygame.draw.line(image, (49, 48, 80), (3, 21), (28, 21))
    image.blit(landing_strip, (4, 3))
    pygame.draw.rect(image, INK, (5, 7, 22, 12))
    pygame.draw.line(image, WHITE, (5, 7), (26, 7))
    pygame.draw.line(image, (222, 228, 232), (17, 8), (17, 17))
    pygame.draw.line(image, STEEL[2], (8, 16), (14, 9))
    pygame.draw.line(image, RED, (9, 16), (14 + frame % 2, 10))
    pygame.draw.rect(image, GOLD[2], (20, 9, 5, 3))
    pygame.draw.line(image, GOLD[4], (20, 9), (24, 9))
    pygame.draw.line(image, (105, 104, 143), (6, 19), (25, 19))
    for x in (5, 23):
        pygame.draw.rect(image, INK, (x, 23, 6, 9))
        pygame.draw.rect(image, CYAN, (x + 1, 24, 4, 7))
        pygame.draw.line(image, STEEL[4], (x + 1, 24), (x + 1, 30))
        for y in (25 + frame % 2, 28 + frame % 2):
            pygame.draw.line(image, STEEL[1], (x + 2, y), (x + 4, y))
    # Small chips/oil marks retain the working-mine wear at native resolution.
    pygame.draw.line(image, STEEL[2], (4, 19), (8, 19))
    image.set_at((26, 18), STEEL[3])
    image.set_at((27, 19), INK)
    return image


def green_thorn(frame: int) -> pygame.Surface:
    image = canvas()
    if frame == 0:
        pygame.draw.line(image, (29, 70, 34), (11, 31), (21, 31))
        image.set_at((15, 30), GREEN)
        return image
    height = frame * 8
    # A single crooked plant spike, rather than the separate silver spike clusters.
    points = [
        (15, 0),
        (16, 1),
        (17, 5),
        (17, 10),
        (19, 12),
        (18, 17),
        (21, 21),
        (20, 26),
        (22, 31),
        (10, 31),
        (11, 26),
        (13, 23),
        (12, 18),
        (14, 15),
        (13, 10),
        (15, 7),
    ]
    # Compress the lower silhouette into each rising pose; keep the base fixed.
    points = [(x, min(31, 32 - height + round(y * height / 32))) for x, y in points]
    pygame.draw.polygon(image, (24, 57, 28), points)
    pygame.draw.lines(
        image,
        (58, 117, 39),
        False,
        [(16, 36 - height), (15, 39 - height), (17, 45 - height), (15, 30)],
        3,
    )
    pygame.draw.lines(
        image,
        (120, 184, 61),
        False,
        [(15, 32 - height), (15, 38 - height), (14, 42 - height), (16, 46 - height), (14, 30)],
        1,
    )
    image.set_at((15, 32 - height), (170, 225, 81))
    pygame.draw.line(image, (46, 98, 36), (11, 31), (22, 31))
    return image


def stalactite() -> pygame.Surface:
    image = canvas()
    # Narrow orange cone with a bright left facet and a crooked pointed tip.
    pygame.draw.polygon(
        image, INK, [(9, 0), (23, 0), (22, 9), (19, 21), (16, 31), (13, 24), (10, 10)]
    )
    pygame.draw.polygon(
        image, (141, 48, 37), [(11, 1), (21, 1), (20, 10), (17, 26), (14, 22), (12, 10)]
    )
    pygame.draw.polygon(
        image, (236, 124, 44), [(11, 2), (17, 2), (17, 17), (16, 27), (14, 22), (12, 9)]
    )
    pygame.draw.lines(image, (255, 216, 76), False, [(12, 2), (13, 9), (15, 20)])
    pygame.draw.line(image, (255, 243, 155), (12, 2), (12, 5))
    pygame.draw.lines(image, (189, 74, 37), False, [(19, 3), (18, 9), (19, 13)])
    return image


def mechanism_sprites() -> dict[str, pygame.Surface]:
    return {
        "stalactite": stalactite(),
        "elevator": hover_lift(),
        **{f"elevator_{frame}": hover_lift(frame) for frame in range(4)},
        **{f"green_thorn_{frame}": green_thorn(frame) for frame in range(5)},
    }
