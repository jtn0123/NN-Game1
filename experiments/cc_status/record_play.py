"""Record a trained policy PLAYING held-out Crystal Caves levels, as labeled GIFs.

The diagnose_gap runs never persist the trained weights to disk (the agent lives in memory
only, and parallel per-seed workers would clobber a shared checkpoint), so recording has to
happen in-process right after training while the agent is still live. Because the runs are
deterministic (fixed seeds + torch determinism), re-running a past experiment with
``--record-play`` reproduces the identical policy and the identical held-out eval episodes,
so the GIFs show exactly the behaviour the stall/death traces measured — not a different run.

Rendering is headless (SDL dummy driver) and uses Pillow to assemble GIFs (imageio isn't
available in this environment). Frames are downscaled and stride-sampled to keep GIFs small
and memory bounded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cc_status.evals import (  # noqa: E402
    _enter_greedy_agent_eval,
    _restore_greedy_agent_eval,
)
from src.game.crystal_caves import CrystalCaves  # noqa: E402

# Failure modes we most want to SEE, in save-priority order (stalls first: that's the
# dominant, least-understood failure after RUN-18/19).
_REASON_PRIORITY = {
    "stalled": 0,
    "won": 1,
    "killed": 2,
    "first_crystal_goal": 3,
    "timeout": 4,
}


def _frame(surface: pygame.Surface, size: tuple[int, int]) -> Image.Image:
    """Snapshot the rendered surface as a downscaled RGB Pillow image."""
    arr = np.ascontiguousarray(pygame.surfarray.array3d(surface).swapaxes(0, 1))
    img = Image.fromarray(arr, mode="RGB")
    if img.size != size:
        img = img.resize(size)
    return img


def record_policy_play(
    agent: Any,
    config: Any,
    *,
    games: int,
    out_dir: str | Path,
    max_gifs: int = 6,
    capture_games: int = 8,
    max_frames: int = 160,
    scale: float = 0.5,
    duration_ms: int = 80,
) -> list[dict[str, Any]]:
    """Greedy-play the first ``capture_games`` held-out levels, capturing frames, then save
    up to ``max_gifs`` of them as GIFs (preferring stalls, then wins, then kills). Returns a
    list of {path, end_reason, frames, steps} for the saved GIFs.

    Only the first ``capture_games`` episodes are rendered/held in memory (bounded RAM); with
    a ~0.49 stall rate that reliably captures several stalls to watch. Frames are sampled at
    a fixed stride so no episode exceeds ``max_frames`` frames regardless of length."""
    pygame.init()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    # headless=False so the renderer's art is available; physics/observations are identical
    # to the headless eval game, so the greedy trajectory matches what the traces measured.
    game = CrystalCaves(config, headless=False)
    n_capture = min(games, max(1, capture_games))
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    step_limit = int(config.EVAL_MAX_STEPS)
    capture_stride = max(1, step_limit // max_frames)
    size = (max(1, int(game.width * scale)), max(1, int(game.height * scale)))
    surface = pygame.Surface((game.width, game.height))

    episodes: list[dict[str, Any]] = []
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for _ in range(n_capture):
            state = game.reset()
            frames: list[Image.Image] = []
            done = False
            steps = 0
            info: dict[str, Any] = {}
            while not done and steps < step_limit:
                if steps % capture_stride == 0:
                    game.render(surface)
                    frames.append(_frame(surface, size))
                action = agent.select_action(state, training=False)
                state, _, done, info = game.step(action)
                steps += 1
            # capture the final frame so the terminal state is visible
            game.render(surface)
            frames.append(_frame(surface, size))
            reason = str(info.get("end_reason", "timeout")) if done else "timeout"
            episodes.append({"end_reason": reason, "frames": frames, "steps": steps})
    finally:
        _restore_greedy_agent_eval(agent, agent_state)

    order = sorted(
        range(len(episodes)),
        key=lambda i: (_REASON_PRIORITY.get(episodes[i]["end_reason"], 9), i),
    )
    saved: list[dict[str, Any]] = []
    for i in order[:max_gifs]:
        ep = episodes[i]
        frames = ep["frames"]
        if not frames:
            continue
        gif_path = out_path / f"ep{i:02d}_{ep['end_reason']}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        saved.append(
            {
                "path": str(gif_path),
                "end_reason": ep["end_reason"],
                "frames": len(frames),
                "steps": ep["steps"],
            }
        )
    return saved
