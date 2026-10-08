"""Transparent, deliberately readable clues over the existing solid terrain."""

import pygame

from .visual_materials import CYAN, GOLD, INK, STEEL, WHITE, canvas


def secret_sprites() -> dict[str, pygame.Surface]:
    images = {}
    for armed in (True, False):
        image = canvas()
        # Keep the accepted platform material visible around a single seam.
        seam = [(9, 5), (12, 10), (10, 15), (15, 21), (14, 29)]
        pygame.draw.lines(image, INK, False, seam, 3)
        pygame.draw.lines(image, STEEL[4], False, [(x + 2, y) for x, y in seam], 1)
        if armed:
            pygame.draw.rect(image, INK, (16, 8, 13, 15))
            pygame.draw.rect(image, GOLD[2], (17, 9, 11, 13), 1)
            pygame.draw.polygon(image, CYAN, [(22, 11), (26, 15), (22, 20), (18, 15)])
            pygame.draw.line(image, WHITE, (22, 12), (19, 15))
            pygame.draw.lines(image, GOLD[3], False, [(18, 29), (22, 25), (26, 29)], 2)
        else:
            pygame.draw.rect(image, INK, (19, 12, 7, 7))
            pygame.draw.rect(image, STEEL[2], (19, 12, 7, 7), 1)
        images[f"secret_cache_{'armed' if armed else 'empty'}"] = image
    return images
