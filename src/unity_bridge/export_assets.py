"""Export native pixel art, terrain and classic speaker effects into Unity Resources."""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from src.game.crystal_caves_art import SPRITES, CrystalCavesArt

from .classic_audio import SAMPLE_RATE, TONE_RATE, sound_bank
from .mine import MINE_SPEC
from .session import CaveSession
from .visual_fidelity import mine_door, mine_surfaces, torch
from .visual_materials import raygun
from .visuals import (
    THEMES,
    art_sprites,
    backdrop,
    dressing,
    dressing_placements,
    terrain,
    theme_index,
    title_scene,
    vignette,
)


def export(destination: Path) -> None:
    pygame.font.init()
    for directory in ("Sprites", "Terrain", "Audio", "Interface", "Environment"):
        (destination / directory).mkdir(parents=True, exist_ok=True)
    art = CrystalCavesArt()
    # Retire the rejected painted direction from the build's managed resources.
    for pattern in (
        "Environment/painted_*.png",
        "Environment/near_*.png",
        "Interface/destination_*.png",
        "Interface/explorer_portrait*.png",
        "Interface/title_ledge.png",
        "Sprites/mylo_run_[4-7].png",
    ):
        for obsolete in destination.glob(pattern):
            obsolete.unlink()
            obsolete.with_suffix(".png.meta").unlink(missing_ok=True)
    font = pygame.Surface((96, 48), pygame.SRCALPHA)
    for code in range(32, 128):
        glyph = art.text(chr(code), (255, 255, 255), scale=1)
        font.blit(glyph, (((code - 32) % 16) * 6, ((code - 32) // 16) * 8))
    pygame.image.save(font, destination / "Interface/pixel_font.png")
    heart = pygame.Surface((18, 16), pygame.SRCALPHA)
    pygame.draw.polygon(
        heart,
        (69, 19, 55),
        [(2, 2), (6, 1), (9, 4), (12, 1), (16, 2), (17, 6), (15, 10), (9, 15), (3, 10), (1, 6)],
    )
    pygame.draw.polygon(
        heart,
        (164, 31, 76),
        [(3, 3), (6, 2), (9, 5), (12, 2), (15, 3), (16, 6), (13, 10), (9, 13), (4, 9), (2, 6)],
    )
    # Warm lit lobe and magenta shadow remain legible at the native HUD size.
    pygame.draw.polygon(heart, (229, 49, 56), [(3, 3), (6, 2), (8, 4), (8, 10), (5, 9), (2, 6)])
    pygame.draw.polygon(heart, (255, 140, 36), [(3, 4), (6, 3), (7, 4), (7, 8), (5, 9), (3, 7)])
    pygame.draw.rect(heart, (255, 211, 66), (4, 4, 2, 4))
    pygame.draw.line(heart, (255, 184, 78), (3, 3), (6, 3))
    pygame.draw.line(heart, (198, 52, 91), (11, 3), (14, 3))
    pygame.draw.polygon(heart, (117, 25, 79), [(10, 7), (15, 5), (15, 8), (9, 13)])
    pygame.image.save(heart, destination / "Interface/heart.png")
    pygame.image.save(title_scene(), destination / "Interface/title_scene.png")
    for word in ("CRYSTAL", "CAVES"):
        pygame.image.save(
            art.text(word, (255, 255, 255), scale=4),
            destination / "Interface" / f"{word.lower()}.png",
        )
    for name in SPRITES:
        pygame.image.save(art.sprite(name), destination / "Sprites" / f"{name}.png")
    session = CaveSession()
    game = session.game
    game._art = art
    # New art follows the exact collision grid and is sliced into native Tilemap cells.
    # Environmental dressing is a separate, non-colliding layer.
    catalog = []
    for index in range(len(game.CAVES)):
        session.reset(index)
        game.width = game.level_width
        game.height = game.level_height + game.HUD_HEIGHT
        surface = terrain(session.terrain_layout(), index)
        wall = backdrop(THEMES[theme_index(index)], surface.get_size())
        pygame.image.save(wall, destination / "Environment" / f"wall_{index}.png")
        pygame.image.save(surface, destination / "Terrain" / f"level_{index}.png")
        decoration, lights = dressing(game.level.layout, index)
        pygame.image.save(decoration, destination / "Environment" / f"dressing_{index}.png")
        catalog.append(
            {
                "name": game.level.name,
                "crystals": game.initial_crystals,
                "cols": game.level_cols,
                "rows": game.level_rows,
                "theme": theme_index(index),
                "region": THEMES[theme_index(index)].name,
                "lights": lights,
                "decorations": dressing_placements(game.level.layout, index),
            }
        )
    (destination / "CaveCatalog.json").write_text(
        json.dumps({"caves": catalog}, separators=(",", ":"))
    )
    for index, theme in enumerate(THEMES):
        pygame.image.save(backdrop(theme), destination / "Environment" / f"backdrop_{index}.png")
    pygame.image.save(vignette(), destination / "Environment/vignette.png")
    session.reset(0)
    for name in (
        "exit_locked",
        "exit_open",
        "door_red",
        "door_blue",
        "elevator",
        "bullet",
        "treasure",
    ):
        surface = pygame.Surface((32, 32), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, 32, 32)
        if name.startswith("exit"):
            game.exit_unlocked = name == "exit_open"
            game._draw_exit_airlock(surface, rect)
        elif name.startswith("door"):
            game._draw_locked_door(surface, rect, name.removeprefix("door_"))
        elif name == "elevator":
            surface = pygame.Surface((32, 10), pygame.SRCALPHA)
            pygame.draw.rect(surface, (19, 29, 48), (0, 0, 32, 10))
            pygame.draw.rect(surface, (99, 222, 239), (1, 1, 30, 3))
            for x in range(4, 30, 8):
                pygame.draw.rect(surface, (255, 196, 92), (x, 5, 4, 3))
        elif name == "bullet":
            surface = pygame.Surface((10, 4), pygame.SRCALPHA)
            surface.fill((255, 221, 112))
        else:
            pygame.draw.ellipse(surface, (104, 61, 22), (7, 7, 20, 20))
            pygame.draw.ellipse(surface, (255, 201, 74), (7, 5, 20, 20))
            pygame.draw.ellipse(surface, (255, 242, 146), (10, 7, 12, 12), 2)
        pygame.image.save(surface, destination / "Sprites" / f"{name}.png")
    for name, tint in (
        ("power_shot", (255, 212, 100)),
        ("gravity", (198, 144, 255)),
        ("freeze", (100, 230, 255)),
    ):
        surface = art.sprite("power").copy()
        surface.fill((*tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
        pygame.image.save(surface, destination / "Sprites" / f"{name}.png")
    for name, surface in art_sprites().items():
        pygame.image.save(surface, destination / "Sprites" / f"{name}.png")
    pygame.image.save(raygun(), destination / "Interface/raygun.png")
    mine_wall, mine_terrain, mine_props = mine_surfaces(MINE_SPEC.layout)
    pygame.image.save(mine_wall, destination / "Environment/wall_mine.png")
    pygame.image.save(mine_terrain, destination / "Terrain/level_mine.png")
    pygame.image.save(mine_props, destination / "Environment/dressing_mine.png")
    for name, surface in {
        "mine_door": mine_door(),
        "mine_door_cleared": mine_door(True),
        **{f"mine_torch_{frame}": torch(frame) for frame in range(4)},
    }.items():
        pygame.image.save(surface, destination / "Sprites" / f"{name}.png")
    # Retire the unrelated two-second loop; classic play uses speaker effects.
    for obsolete in destination.glob("Audio/music.wav*"):
        obsolete.unlink()
    clips, sound_info = sound_bank()
    (destination / "ClassicAudio.json").write_text(
        json.dumps(
            {
                "sample_rate": SAMPLE_RATE,
                "tone_rate": TONE_RATE,
                "sounds": [{"name": name, **info} for name, info in sound_info.items()],
            }
        )
    )
    for name, samples in clips.items():
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        with wave.open(str(destination / "Audio" / f"{name}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
    sprites = art_sprites()
    for index in range(len(game.CAVES)):
        session.reset(index)
        snapshot = session.snapshot()
        scene = backdrop(THEMES[theme_index(index)], (game.level_cols * 32, game.level_rows * 32))
        decoration, _ = dressing(game.level.layout, index)
        scene.blit(decoration, (0, 0))
        scene.blit(terrain(session.terrain_layout(), index), (0, 0))
        for entity in snapshot["entities"]:
            name = entity["sprite"]
            if entity["id"].startswith("crystal_"):
                name = (
                    "crystal_"
                    + ("blue", "green", "yellow", "red")[
                        (int(entity["x"]) // 32 + int(entity["y"]) // 32 * 3) % 4
                    ]
                )
            sprite = sprites.get(name)
            if sprite is not None:
                if entity["flip"]:
                    sprite = pygame.transform.flip(sprite, True, False)
                vertical = (
                    entity["h"] - sprite.get_height()
                    if name == "dinosaur_enemy"
                    else (entity["h"] - sprite.get_height()) / 2
                )
                scene.blit(
                    sprite,
                    (
                        round(entity["x"] + (entity["w"] - sprite.get_width()) / 2),
                        round(entity["y"] + vertical),
                    ),
                )
        # Pixel scene previews use actual art and actors, rather than painted destination art.
        crop = pygame.Rect(0, max(0, int(snapshot["player"]["y"]) - 220), 640, 384)
        crop.clamp_ip(scene.get_rect())
        pygame.image.save(scene.subsurface(crop), destination / "Interface" / f"cave_{index}.png")
    session.reset(0)
    (destination / "PilotPreview.json").write_text(
        json.dumps(session.snapshot(), separators=(",", ":"))
    )
    game.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "unity/Assets/Resources",
    )
    args = parser.parse_args()
    export(args.output)
    print(f"Exported Crystal Caves assets to {args.output}")


if __name__ == "__main__":
    main()
