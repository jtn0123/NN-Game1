"""Screenshot the hand-crafted levels through the REAL game renderer.

Unlike render_levels.py (schematic tile markers), this boots the actual
CrystalCaves engine and calls its render() — real tile art, ladders, doors,
switch wires, enemies, pickups, player sprite, and HUD. Two shot types:

  full/  one PNG per level with the viewport sized to the whole 40x24 cave
         (camera pinned at origin), for level-design validation
  view/  the normal 800x600 in-game player view at spawn, for UX validation

Run:  python -m experiments.cc_status.screenshot_levels [output_dir]
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pygame  # noqa: E402

from config import Config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402


def _game_for(spec, width: int, height: int) -> CrystalCaves:
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    cfg.SCREEN_WIDTH = width
    cfg.SCREEN_HEIGHT = height
    game = CrystalCaves(cfg, headless=False)
    game.CAVES = (spec,)
    game._eval_caves = (spec,)
    game._randomize_levels = False
    game.reset()
    return game


def main(argv) -> int:
    out_dir = argv[1] if len(argv) > 1 else os.path.join(_REPO_ROOT, "scratchpad", "cc_shots")
    full_dir = os.path.join(out_dir, "full")
    view_dir = os.path.join(out_dir, "view")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(view_dir, exist_ok=True)
    pygame.init()

    hud = CrystalCaves.HUD_HEIGHT
    ts = CrystalCaves.TILE_SIZE
    for spec in HANDCRAFTED_LEVELS:
        cols = len(spec.layout[0])
        rows = len(spec.layout)
        name = spec.name.replace(" ", "_")

        # whole cave through the real renderer (camera clamps to 0,0)
        game = _game_for(spec, cols * ts, rows * ts + hud)
        surf = pygame.Surface((game.width, game.height))
        game.render(surf)
        pygame.image.save(surf, os.path.join(full_dir, f"{name}.png"))

        # the player's actual 800x600 spawn view
        game = _game_for(spec, 800, 600)
        surf = pygame.Surface((800, 600))
        game.render(surf)
        pygame.image.save(surf, os.path.join(view_dir, f"{name}.png"))
        print(f"shot {spec.name}")

    print(
        f"wrote {len(HANDCRAFTED_LEVELS)} full + {len(HANDCRAFTED_LEVELS)} view shots -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
