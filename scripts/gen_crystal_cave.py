"""CLI for the Crystal Caves procedural level generator.

The generation algorithm + rubric live in ``src.game.crystal_caves_gen``; this is
a thin wrapper that grades, renders, or batch-evaluates generated levels.

    python scripts/gen_crystal_cave.py --seed 3 --theme rust --render /tmp/gen.png
    python scripts/gen_crystal_cave.py --batch 200
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game.crystal_caves_gen import (  # noqa: E402
    THEME_NAMES,
    THEMES,
    generate_cave,
    grade_cave,
)


def render_png(spec, out: str, theme_index: int) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from config import Config
    from src.game.crystal_caves import CrystalCaves

    pygame.init()
    cfg = Config()
    game = CrystalCaves(cfg, headless=False)
    game.level_index = theme_index  # palette/theme for the renderer
    game.level = spec
    game._load_level(spec)
    # Generated caves carry no authored CAVE_DRESSING yet — suppress the episode
    # dressing that would otherwise bleed in at hard-coded coordinates.
    game._draw_authored_dressing = lambda *a, **k: None  # type: ignore[method-assign]
    surface = pygame.Surface((cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT))
    game.render(surface)
    pygame.image.save(surface, out)


def _theme_for(seed: int, theme: str | None) -> str:
    return theme or THEME_NAMES[seed % len(THEME_NAMES)]


def run_batch(count: int, theme: str | None) -> None:
    scores = []
    accepted = 0
    missed: Counter = Counter()
    for seed in range(count):
        spec = generate_cave(seed, theme)
        report = grade_cave(spec)
        scores.append(report["score"])
        if report["score"] >= 85 and report["solvable"]:
            accepted += 1
        for key in ("density_ok", "top_entrance", "fully_connected", "door_gates_exit"):
            if not report.get(key):
                missed[key] += 1
    mean = sum(scores) / len(scores)
    print(
        f"batch={count} theme={theme or 'mixed'} "
        f"mean={mean:.1f} min={min(scores)} max={max(scores)} "
        f"accept(>=85 & solvable)={accepted}/{count}"
    )
    print("most-missed criteria:", missed.most_common(5) or "none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--theme", choices=list(THEME_NAMES), default=None)
    parser.add_argument("--render", type=str, default="")
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch, args.theme)
        return

    theme = _theme_for(args.seed, args.theme)
    spec = generate_cave(args.seed, args.theme)
    report = grade_cave(spec)
    print(f"seed={args.seed}  theme={theme}  score={report['score']}/100")
    for key, value in report.items():
        if key != "score":
            print(f"  {key}: {value}")
    if args.render:
        render_png(spec, args.render, THEMES[theme]["index"])
        print(f"rendered -> {args.render}")


if __name__ == "__main__":
    main()
