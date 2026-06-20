"""
Render deterministic Crystal Caves review screenshots.

This is intentionally lightweight so visual work can be reviewed without
starting the full training app.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from src.game.crystal_caves import CrystalCaves


def _render(game: CrystalCaves, path: Path) -> None:
    surface = pygame.Surface((game.width, game.height))
    game.render(surface)
    pygame.image.save(surface, path)


def _render_title(game: CrystalCaves, path: Path) -> None:
    surface = pygame.Surface((game.width, game.height))
    game.render_title_screen(surface)
    pygame.image.save(surface, path)


def _place_at_tile(game: CrystalCaves, tile: tuple[int, int]) -> None:
    game.player_x = tile[0] * game.TILE_SIZE + 5
    game.player_y = tile[1] * game.TILE_SIZE + 1
    game.vx = 0.0
    game.vy = 0.0


def render_gallery(output_dir: Path) -> list[Path]:
    pygame.init()
    config = Config()
    game = CrystalCaves(config, headless=False)
    game.show_controls = True
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    title = output_dir / "01_title.png"
    _render_title(game, title)
    paths.append(title)

    start = output_dir / "02_start.png"
    _render(game, start)
    paths.append(start)

    crystal = sorted(game.crystals, key=lambda tile: (tile[0], tile[1]))[1]
    _place_at_tile(game, crystal)
    crystal_path = output_dir / "03_crystal_pocket.png"
    _render(game, crystal_path)
    paths.append(crystal_path)

    game.step(CrystalCaves.IDLE)
    pickup_path = output_dir / "04_pickup_sparkle.png"
    _render(game, pickup_path)
    paths.append(pickup_path)

    switch = sorted(game.switches)[0]
    _place_at_tile(game, switch)
    switch_path = output_dir / "05_switch_room.png"
    _render(game, switch_path)
    paths.append(switch_path)

    hazard = sorted(game.hazards, key=lambda tile: (tile[1], tile[0]))[0]
    _place_at_tile(game, hazard)
    hazard_path = output_dir / "06_hazard_corridor.png"
    _render(game, hazard_path)
    paths.append(hazard_path)

    tank = sorted(game.air_tanks)[0]
    _place_at_tile(game, tank)
    tank_path = output_dir / "07_air_tank.png"
    _render(game, tank_path)
    paths.append(tank_path)

    enemy = next(enemy for enemy in game.enemies if enemy.alive)
    _place_at_tile(
        game,
        (
            max(1, int(enemy.x // game.TILE_SIZE) - 2),
            max(1, int(enemy.y // game.TILE_SIZE)),
        ),
    )
    enemy_path = output_dir / "08_enemy_room.png"
    _render(game, enemy_path)
    paths.append(enemy_path)

    game.facing = 1
    game.ammo = 5
    game.step(CrystalCaves.SHOOT)
    action_path = output_dir / "09_shooting_frame.png"
    _render(game, action_path)
    paths.append(action_path)

    game.crystals.clear()
    game.exit_unlocked = True
    game.open_colors.update(game.door_color.values())  # open every colour-keyed door
    _place_at_tile(game, game.exit_pos)
    exit_path = output_dir / "10_exit_open.png"
    _render(game, exit_path)
    paths.append(exit_path)

    game.level_index = 1
    game.reset()
    _place_at_tile(game, (7, 2))
    episode2_path = output_dir / "11_episode2_amber.png"
    _render(game, episode2_path)
    paths.append(episode2_path)

    game.level_index = 2
    game.reset()
    _place_at_tile(game, (8, 2))
    episode3_path = output_dir / "12_episode3_moon.png"
    _render(game, episode3_path)
    paths.append(episode3_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".Codex/artifacts/crystal_caves",
        help="Directory for rendered PNG files.",
    )
    args = parser.parse_args()

    for path in render_gallery(Path(args.output_dir)):
        print(path)


if __name__ == "__main__":
    main()
