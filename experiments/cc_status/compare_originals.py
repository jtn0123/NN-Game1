"""Side-by-side comparison: original CC1 level reconstructions vs the hand-crafted set.

Left of each pair: the best-effort byte->tile reconstruction of the real Episode 1
level, recovered from git history (commit 2d4134b, later removed because the
mapping is provably imperfect — trustworthy for STRUCTURE/DENSITY on ~10/16
levels, garbage on the rest, and it drops all object types). Right: the
hand-crafted level at the same index. Both drawn by the real game renderer.

Writes pair_NN.png files and prints a per-level structural metrics table.

Run:  python -m experiments.cc_status.compare_originals [output_dir]
"""

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pygame  # noqa: E402

from config import Config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402

_ORIGINALS_COMMIT = "2d4134b"
_ORIGINALS_PATH = "src/game/crystal_caves_cc1_levels.py"

# Reconstructions whose byte->tile decode produced credible terrain (solid mass in
# the plausible range). The rest decoded with ~zero solids (terrain misread as
# ladders/air) and are labelled as unreliable in the output.
CREDIBLE = {1, 2, 4, 5, 7, 8, 11, 12, 13, 15}


def _load_originals():
    try:
        src = subprocess.check_output(
            ["git", "show", f"{_ORIGINALS_COMMIT}:{_ORIGINALS_PATH}"], cwd=_REPO_ROOT, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"could not read {_ORIGINALS_PATH} from commit {_ORIGINALS_COMMIT} "
            "(shallow clone missing history?)"
        ) from exc
    src = src.replace(
        "from .crystal_caves_entities import CaveSpec",
        "from src.game.crystal_caves_entities import CaveSpec",
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cc1_originals.py")
        with open(path, "w") as f:
            f.write(src)
        namespace: dict = {}
        exec(compile(src, path, "exec"), namespace)  # noqa: S102 - our own historic file
        return namespace["CC1_LEVELS"]


def _shot(spec) -> pygame.Surface:
    ts = CrystalCaves.TILE_SIZE
    hud = CrystalCaves.HUD_HEIGHT
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    cfg.SCREEN_WIDTH = len(spec.layout[0]) * ts
    cfg.SCREEN_HEIGHT = len(spec.layout) * ts + hud
    game = CrystalCaves(cfg, headless=False)
    game.CAVES = (spec,)
    game._eval_caves = (spec,)
    game._randomize_levels = False
    game.reset()
    surf = pygame.Surface((game.width, game.height))
    game.render(surf)
    return surf


def _metrics(layout):
    flat = "".join(layout)
    n = {ch: flat.count(ch) for ch in "#*HsSDd^~MFA$O="}
    return n, len(flat) - flat.count(".")


def main(argv) -> int:
    out = argv[1] if len(argv) > 1 else os.path.join(_REPO_ROOT, "scratchpad", "cc_compare")
    os.makedirs(out, exist_ok=True)
    originals = _load_originals()
    pygame.init()
    font = pygame.font.SysFont("monospace", 20, bold=True)
    small = pygame.font.SysFont("monospace", 14, bold=True)

    print(
        f"{'PAIR':<34} {'side':>5} {'fill':>5} {'#':>4} {'cry':>4} {'lad':>4} "
        f"{'elev':>4} {'sw':>3} {'dr':>3} {'haz':>4} {'enemy':>5} {'loot':>4}"
    )
    for i, (orig, mine) in enumerate(zip(originals, HANDCRAFTED_LEVELS, strict=True), 1):
        a, b = _shot(orig), _shot(mine)
        pad, cap = 8, 30
        sheet = pygame.Surface(
            (
                a.get_width() + b.get_width() + pad * 3,
                max(a.get_height(), b.get_height()) + cap + pad * 2,
            )
        )
        sheet.fill((10, 10, 14))
        tag = "reconstruction" if i in CREDIBLE else "reconstruction UNRELIABLE (decode failed)"
        sheet.blit(font.render(f"ORIGINAL {orig.name} ({tag})", True, (255, 200, 120)), (pad, pad))
        sheet.blit(
            font.render(f"MINE: {mine.name}", True, (140, 230, 255)),
            (a.get_width() + pad * 2, pad),
        )
        sheet.blit(
            small.render("byte->tile decode, known imperfect", True, (170, 140, 100)),
            (pad + 640, pad + 4),
        )
        sheet.blit(a, (pad, cap + pad))
        sheet.blit(b, (a.get_width() + pad * 2, cap + pad))
        pygame.image.save(sheet, os.path.join(out, f"pair_{i:02d}.png"))

        for side, spec in (("orig", orig), ("mine", mine)):
            n, fill = _metrics(spec.layout)
            print(
                f"{('%02d ' % i) + orig.name + ' vs ' + mine.name:<34} {side:>5} {fill:>5} "
                f"{n['#']:>4} {n['*']:>4} {n['H']:>4} {n['=']:>4} {n['s'] + n['S']:>3} "
                f"{n['D'] + n['d']:>3} {n['^'] + n['~']:>4} {n['M'] + n['F']:>5} "
                f"{n['A'] + n['$'] + n['O']:>4}"
            )
    print("done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
