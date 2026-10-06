"""Room materials observed in the gameplay reference, drawn at native pixel size."""

import pygame

from .visual_materials import INK, canvas


def blue_cobble_field(size: tuple[int, int], origin: tuple[int, int] = (0, 0)) -> pygame.Surface:
    """Rounded blue rocks use world coordinates, so crops share the same joins."""
    # Rasterize whole stones outside the requested crop first. Clipping a thick
    # angled facet directly at a tile edge can change pygame's endpoint pixels.
    field_size = (size[0] + 64, size[1] + 64)
    image = canvas(field_size)
    image.fill((8, 14, 39))
    origin_x, origin_y = origin[0] - 32, origin[1] - 32
    faces = ((37, 58, 114), (43, 66, 126), (45, 73, 133), (34, 53, 104))
    outlines = (
        ((4, 1), (10, 1), (13, 3), (15, 6), (14, 11), (12, 14), (7, 15), (3, 13), (1, 10), (1, 5)),
        ((3, 0), (10, 0), (14, 3), (15, 9), (11, 13), (7, 15), (2, 13), (0, 8), (1, 3)),
        ((6, 1), (12, 0), (15, 4), (14, 11), (10, 15), (4, 14), (0, 11), (1, 6)),
        ((2, 2), (12, 1), (15, 4), (15, 10), (13, 13), (8, 14), (2, 12), (0, 7)),
        ((5, 0), (12, 1), (14, 5), (15, 10), (12, 14), (4, 15), (1, 11), (0, 5)),
    )
    for course in range(origin_y // 16 - 1, (origin_y + field_size[1] - 1) // 16 + 2):
        shift = 8 if course % 2 else 0
        widths: tuple[int, ...] = (18, 14, 20, 12)
        turn = course % len(widths)
        widths = widths[turn:] + widths[:turn]
        for block in range(
            (origin_x - shift) // 64 - 1, (origin_x + field_size[0] - 1 - shift) // 64 + 2
        ):
            x = block * 64 + shift - origin_x
            for slot, width in enumerate(widths):
                variant = ((block * 4 + slot) * 13 + course * 5) % 7
                y = course * 16 - origin_y

                def points(coords: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
                    return [(x + dx * (width - 1) // 15, y + dy) for dx, dy in coords]

                # Unequal widths and stepped silhouettes give broad rounded
                # rubble faces instead of a regular diamond or button pattern.
                pygame.draw.polygon(
                    image,
                    faces[variant % len(faces)],
                    points(outlines[variant % 5]),
                )
                facet = (
                    ((3, 6), (5, 2), (10, 2), (7, 6))
                    if variant in (0, 2, 5)
                    else ((3, 4), (5, 2), (9, 2), (8, 5))
                )
                pygame.draw.polygon(image, (73, 111, 174), points(facet))
                pygame.draw.lines(image, (93, 146, 188), False, points(((4, 3), (6, 2))))
                pygame.draw.lines(
                    image, (17, 29, 72), False, points(((13, 6), (12, 11), (8, 14))), 2
                )
                x += width
    return image.subsurface((32, 32, size[0], size[1])).copy()


def blue_cobble(col: int, row: int, edges: tuple[bool, ...]) -> pygame.Surface:
    """Native opaque collision cells, with pale worn rims and sparse grass."""
    image = blue_cobble_field((32, 32), (col * 32, row * 32))
    top, left, right, bottom = edges
    if top:
        pygame.draw.line(image, INK, (0, 0), (31, 0))
        pygame.draw.line(image, (177, 213, 136), (0, 1), (31, 1))
        pygame.draw.line(image, (85, 157, 142), (0, 2), (31, 2))
        for x in range(2, 31, 3):
            if (col * 32 + x + row * 5) % 11 < 3:
                pygame.draw.line(image, (58, 121, 60), (x, 2), (x, 4))
                image.set_at((x, 2), (150, 193, 87))
        if (col + row * 3) % 9 == 0:
            pygame.draw.line(image, (76, 125, 119), (22, 1), (24, 1))
    for exposed, x, rim in ((left, 0, (95, 180, 174)), (right, 31, (62, 127, 156))):
        if exposed:
            pygame.draw.line(image, INK, (x, 0), (x, 31))
            inner = 1 if left else 30
            pygame.draw.line(image, rim, (inner, 3 if top else 1), (inner, 30))
    if bottom:
        pygame.draw.line(image, (12, 22, 53), (0, 30), (31, 30))
        pygame.draw.line(image, INK, (0, 31), (31, 31))
    return image


def green_platform(col: int, row: int, edges: tuple[bool, ...]) -> pygame.Surface:
    """Broad muted-green masses with pale industrial rims and occasional chips."""
    image = canvas()
    image.fill((62, 123, 74))
    top, left, right, bottom = edges
    if top:
        pygame.draw.line(image, INK, (0, 0), (31, 0))
        pygame.draw.line(image, (205, 227, 122), (0, 1), (31, 1))
        pygame.draw.line(image, (154, 188, 88), (0, 2), (31, 2))
        for x in range(3, 32, 8):
            pygame.draw.line(image, (75, 146, 68), (x, 3), (x + 3, 3))
    for exposed, x in ((left, 0), (right, 31)):
        if exposed:
            pygame.draw.line(image, INK, (x, 0), (x, 31))
            pygame.draw.line(image, (122, 169, 83), (1 if left else 30, 1), (1 if left else 30, 30))
    if bottom:
        pygame.draw.line(image, (30, 69, 47), (0, 30), (31, 30))
        pygame.draw.line(image, INK, (0, 31), (31, 31))
    if top and (col + row * 3) % 9 == 0:
        pygame.draw.line(image, (66, 102, 62), (22, 1), (25, 1))
        pygame.draw.line(image, (164, 179, 102), (23, 2), (25, 2))
    if bottom and (col * 3 + row) % 13 == 0:
        pygame.draw.line(image, (46, 95, 62), (6, 25), (10, 26))
        pygame.draw.line(image, (73, 135, 77), (7, 24), (10, 25))
    return image


def rust_brick_wall(size: tuple[int, int]) -> pygame.Surface:
    image = canvas(size)
    image.fill((34, 31, 42))
    colors = ((112, 60, 47), (124, 64, 47), (132, 72, 51), (116, 66, 51))
    for row, y in enumerate(range(0, size[1], 16)):
        for col, x in enumerate(range(-24 if row % 2 else 0, size[0], 48)):
            color = colors[(col * 3 + row * 5) % len(colors)]
            pygame.draw.polygon(
                image,
                color,
                [
                    (x + 3, y + 1),
                    (x + 44, y + 1),
                    (x + 46, y + 3),
                    (x + 46, y + 13),
                    (x + 43, y + 14),
                    (x + 2, y + 14),
                    (x + 1, y + 11),
                    (x + 1, y + 3),
                ],
            )
            pygame.draw.line(image, (168, 102, 65), (x + 4, y + 2), (x + 43, y + 2))
            pygame.draw.line(image, (140, 81, 54), (x + 2, y + 4), (x + 2, y + 10))
            pygame.draw.line(image, (76, 42, 41), (x + 4, y + 13), (x + 42, y + 13))
            if (col + row * 3) % 7 == 0:
                pygame.draw.rect(image, (87, 47, 40), (x + 33, y + 4, 4, 3))
                pygame.draw.line(image, (153, 89, 60), (x + 32, y + 3), (x + 36, y + 3))
    return image


def pale_blocks(col: int, row: int) -> pygame.Surface:
    image = canvas()
    image.fill((66, 65, 99))
    variant = (col * 3 + row * 5) % 8
    blocks: tuple[tuple[int, int, int, int], ...]
    if variant in (1, 4):
        blocks = ((0, 0, 16, 32), (16, 0, 16, 32))
    elif variant == 3:
        blocks = ((0, 0, 32, 16), (0, 16, 32, 16))
    elif variant == 6:
        blocks = tuple((x, y, 16, 16) for y in (0, 16) for x in (0, 16))
    else:
        blocks = ((0, 0, 32, 32),)
    for index, (x, y, w, h) in enumerate(blocks):
        pygame.draw.rect(image, (166, 165, 194), (x + 1, y + 1, w - 2, h - 2))
        pygame.draw.line(image, (233, 236, 239), (x + 1, y + 1), (x + w - 2, y + 1))
        pygame.draw.line(image, (216, 217, 231), (x + 1, y + 2), (x + 1, y + h - 3))
        pygame.draw.line(image, (110, 106, 145), (x + w - 3, y + 3), (x + w - 3, y + h - 3), 2)
        pygame.draw.line(image, (127, 121, 163), (x + 3, y + h - 3), (x + w - 4, y + h - 3), 2)
        pygame.draw.rect(image, (224, 226, 236), (x + 3, y + 3, 3, 4))
        # Small scuffed corners, rather than cracks across every clean block face.
        if (col + row * 3 + index) % 5 == 0:
            pygame.draw.lines(
                image,
                (137, 134, 175),
                False,
                [(x + w - 6, y + h - 9), (x + w - 6, y + h - 6), (x + w - 9, y + h - 6)],
                2,
            )
    return image


def pipe_wall(size: tuple[int, int]) -> pygame.Surface:
    image = canvas(size)
    image.fill((3, 5, 9))
    for row, y in enumerate(range(0, size[1], 64)):
        for col, x in enumerate(range(0, size[0], 64)):
            # Quiet black panels with worn copper seams and rounded upper corners.
            pygame.draw.lines(
                image,
                (83, 45, 43),
                False,
                [(x + 1, y + 61), (x + 1, y + 6), (x + 6, y + 1), (x + 61, y + 1)],
            )
            pygame.draw.lines(
                image,
                (170, 86, 52),
                False,
                [(x + 3, y + 22), (x + 3, y + 8), (x + 8, y + 3), (x + 27, y + 3)],
            )
            pygame.draw.lines(
                image, (39, 42, 48), False, [(x + 2, y + 57), (x + 7, y + 62), (x + 59, y + 62)]
            )
            for dx in range(10, 61, 9):
                image.set_at((min(size[0] - 1, x + dx), min(size[1] - 1, y + 1)), (43, 45, 50))
            if (col * 3 + row) % 5 == 0:
                pygame.draw.line(image, (57, 45, 41), (x + 1, y + 42), (x + 1, y + 49))
    return image


def ribbed_wall(size: tuple[int, int]) -> pygame.Surface:
    image = canvas(size)
    image.fill((77, 81, 80))
    for x in range(0, size[0], 32):
        pygame.draw.rect(image, (39, 46, 57), (x, 0, 4, size[1]))
        pygame.draw.line(image, (116, 126, 127), (x + 4, 0), (x + 4, size[1] - 1))
        pygame.draw.line(image, (92, 101, 103), (x + 5, 0), (x + 5, size[1] - 1))
    # Sparse pits keep the large vertical rhythm of the reference room intact.
    for row, y in enumerate(range(24, size[1], 96)):
        for col, x in enumerate(range(20, size[0], 128)):
            if (row + col) % 3 == 0:
                pygame.draw.rect(image, (52, 62, 70), (x, y, 2, 3))
                pygame.draw.line(image, (137, 144, 140), (x - 1, y), (x - 1, y + 1))
    return image


def slate_platform(col: int, row: int, edges: tuple[bool, ...]) -> pygame.Surface:
    image = canvas()
    image.fill((77, 103, 133))
    top, left, right, bottom = edges
    if top:
        pygame.draw.line(image, INK, (0, 0), (31, 0))
        pygame.draw.line(image, (94, 140, 151), (1, 1), (30, 1))
        for x in range(2, 31, 8):
            pygame.draw.line(image, (113, 207, 206), (x, 2), (x + 4, 2))
    for exposed, x in ((left, 0), (right, 31)):
        if exposed:
            pygame.draw.line(image, (33, 45, 67), (x, 0), (x, 31))
            for y in range(3, 31, 8):
                pygame.draw.line(
                    image, (98, 185, 195), (1 if left else 30, y), (1 if left else 30, y + 4)
                )
    if bottom:
        pygame.draw.line(image, (36, 48, 73), (0, 30), (31, 30))
        pygame.draw.line(image, INK, (0, 31), (31, 31))
    if any(edges) and (col + row * 3) % 11 == 0:
        pygame.draw.line(image, (57, 78, 106), (21, 26), (25, 26))
    return image


def purple_planks(size: tuple[int, int]) -> pygame.Surface:
    image = canvas(size)
    image.fill(INK)
    for row, y in enumerate(range(0, size[1], 32)):
        for col, x in enumerate(range(-64 if row % 2 else 0, size[0], 128)):
            pygame.draw.rect(image, (78, 27, 68), (x + 2, y + 2, 124, 28))
            pygame.draw.line(image, (128, 56, 111), (x + 3, y + 2), (x + 125, y + 2))
            pygame.draw.line(image, (49, 19, 48), (x + 3, y + 28), (x + 125, y + 28), 2)
            for dx in (16, 48, 80, 112):
                pygame.draw.rect(image, (31, 19, 36), (x + dx, y + 11, 2, 3))
                pygame.draw.line(image, (106, 43, 93), (x + dx - 1, y + 10), (x + dx, y + 10))
            if (col + row * 3) % 7 == 0:
                pygame.draw.line(image, (55, 27, 54), (x + 55, y + 23), (x + 61, y + 22))
                pygame.draw.line(image, (101, 51, 91), (x + 56, y + 24), (x + 60, y + 23))
    return image


def silver_girder(col: int, row: int, edges: tuple[bool, ...]) -> pygame.Surface:
    """Small recessed slots fit inside the unchanged full-tile collision shape."""
    image = canvas()
    image.fill((159, 158, 188))
    top, left, right, bottom = edges
    # A continuous lip across each platform, with the dark slot inside the web.
    if top:
        pygame.draw.rect(image, (198, 199, 219), (0, 1, 32, 5))
        pygame.draw.line(image, (247, 251, 239), (0, 1), (31, 1))
        pygame.draw.line(image, (89, 88, 127), (0, 5), (31, 5))
        pygame.draw.line(image, INK, (0, 0), (31, 0))
    if bottom:
        pygame.draw.rect(image, (187, 186, 211), (0, 26, 32, 5))
        pygame.draw.line(image, (225, 230, 231), (0, 26), (31, 26))
        pygame.draw.line(image, (101, 98, 135), (0, 30), (31, 30))
        pygame.draw.line(image, INK, (0, 31), (31, 31))
    pygame.draw.rect(image, (47, 28, 48), (8, 11, 16, 9))
    pygame.draw.line(image, INK, (8, 11), (23, 11), 2)
    pygame.draw.line(image, (94, 46, 85), (9, 14), (22, 14))
    pygame.draw.line(image, (224, 224, 236), (8, 20), (23, 20))
    pygame.draw.line(image, (91, 86, 125), (7, 12), (7, 19))
    for exposed, x in ((left, 0), (right, 31)):
        if exposed:
            pygame.draw.line(image, INK, (x, 0), (x, 31))
            pygame.draw.line(
                image,
                (215, 218, 230) if left else (104, 101, 138),
                (1 if left else 30, 1),
                (1 if left else 30, 30),
            )
    if (col + row * 3) % 5 == 0:
        pygame.draw.line(image, (110, 105, 143), (3, 28), (5, 28))
        image.set_at((4, 29), (211, 214, 226))
    return image
