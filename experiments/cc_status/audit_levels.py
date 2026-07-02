"""Audit the hand-crafted Crystal Caves levels for accuracy / playability.

Two independent checks per level:

  1. ORACLE reachability (experiments/cc_status/level_reach): from the player
     start, using the engine's exact walk/jump/climb physics, is every crystal,
     switch, and the exit physically reachable? (geometric ground truth)

  2. REAL-ENGINE playthrough: load the level in the live CrystalCaves engine and
     drive it with a simple greedy controller that follows the engine's own
     geodesic navigation compass (the same signal the trained agent sees) —
     jumping to climb, interacting at switches. Reports whether that policy WINS
     and how much of the objective it clears. This confirms the level is not just
     geometrically reachable but actually beatable by an in-engine policy.

Plus a per-level property table (size, crystals, switches, doors, ladders,
spikes) so the set can be eyeballed for variety.

Run:  python -m experiments.cc_status.audit_levels
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import Config  # noqa: E402
from experiments.cc_status.level_reach import analyze  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402


def _controller_action(game: CrystalCaves) -> int:
    """Simple greedy policy: throw switches, follow the engine's corridor compass
    toward the objective, jump when it points up. This is a deliberately DUMB
    baseline — it clears flat levels but has no real ladder/spike navigation
    (that is what training is for), so treat its win count as a lower bound."""
    target, _dist = game._current_target()
    pcol, prow = game._player_tile()

    if target is not None and target[0] == "switch":
        if abs(target[1] - pcol) <= 1 and abs(target[2] - prow) <= 1:
            return game.INTERACT

    dx, dy, _reach, _ = game._geodesic_next_step_compass()
    on_ladder = game._is_on_ladder()
    want_up = dy < 0 or (target is not None and target[2] < prow)

    if on_ladder:
        if want_up:
            return game.JUMP
        if dx > 0:
            return game.RIGHT
        if dx < 0:
            return game.LEFT
        return game.JUMP if target and target[2] < prow else game.IDLE

    if dx > 0:
        return game.RIGHT_JUMP if want_up else game.RIGHT
    if dx < 0:
        return game.LEFT_JUMP if want_up else game.LEFT
    if want_up:
        return game.JUMP
    return game.RIGHT


def playthrough(game: CrystalCaves, max_steps: int = 6000) -> Dict:
    """Greedy playthrough; returns win + best objective fraction reached."""
    game.reset()
    total_crystals = max(1, game.initial_crystals)
    total_switches = max(0, len(game.switches))
    best_frac = 0.0
    stuck_tile = None
    stuck_for = 0
    for step in range(max_steps):
        action = _controller_action(game)
        # anti-stuck: if the player hasn't changed tile for a while, force a hop
        tile = game._player_tile()
        if tile == stuck_tile:
            stuck_for += 1
        else:
            stuck_for = 0
            stuck_tile = tile
        if stuck_for > 12:
            action = game.LEFT_JUMP if (step // 6) % 2 == 0 else game.RIGHT_JUMP
            stuck_for = 0

        _state, _reward, done, info = game.step(action)

        crystals_got = total_crystals - len(game.crystals)
        switches_got = total_switches - len(game.switches - game.used_switches)
        objs = crystals_got + switches_got + (1 if game.won else 0)
        denom = total_crystals + total_switches + 1
        best_frac = max(best_frac, objs / denom)

        if game.won:
            return {
                "won": True,
                "steps": step + 1,
                "best_frac": 1.0,
                "crystals": (crystals_got, total_crystals),
                "switches": (switches_got, total_switches),
            }
        if done:
            return {
                "won": False,
                "steps": step + 1,
                "best_frac": best_frac,
                "end": info.get("end_reason", game._end_reason),
                "crystals": (crystals_got, total_crystals),
                "switches": (switches_got, total_switches),
            }
    return {
        "won": False,
        "steps": max_steps,
        "best_frac": best_frac,
        "end": "step_cap",
        "crystals": (total_crystals - len(game.crystals), total_crystals),
        "switches": (len(game.used_switches), total_switches),
    }


def properties(layout) -> Dict:
    flat = "".join(layout)
    return {
        "size": f"{len(layout[0])}x{len(layout)}",
        "crystals": flat.count("*"),
        "switches": flat.count("s") + flat.count("S"),
        "doors": flat.count("D") + flat.count("d"),
        "ladders": flat.count("H"),
        "spikes": flat.count("^"),
    }


def main() -> int:
    # A dedicated engine per level (CRYSTAL_CAVES_IMPORTED selects the set, but we
    # drive one specific level at a time by pinning the cave list to a single spec).
    print("=" * 92)
    print(
        f"{'LEVEL':<16} {'SIZE':>7} {'CRY':>4} {'SW':>3} {'DR':>3} {'LAD':>4} "
        f"{'SPK':>4} | {'ORACLE':>7} | {'ENGINE PLAYTHROUGH':>26}"
    )
    print("=" * 92)

    oracle_win = 0
    engine_win = 0
    for lv in HANDCRAFTED_LEVELS:
        prop = properties(lv.layout)
        ores = analyze(lv.layout)
        oracle_ok = ores["winnable"]
        oracle_win += oracle_ok

        cfg = Config()
        cfg.CRYSTAL_CAVES_IMPORTED = True
        cfg.CRYSTAL_CAVES_GEODESIC_POTENTIAL = True  # enable the corridor compass
        game = CrystalCaves(cfg, headless=True)
        game.CAVES = (lv,)  # pin to this one level
        game._eval_caves = (lv,)
        game._randomize_levels = False
        pres = playthrough(game)
        engine_win += pres["won"]

        if pres["won"]:
            play = f"WON in {pres['steps']} steps"
        else:
            cc, ct = pres["crystals"]
            play = f"{pres.get('end','?')} {int(pres['best_frac']*100)}% (cry {cc}/{ct})"
        print(
            f"{lv.name:<16} {prop['size']:>7} {prop['crystals']:>4} {prop['switches']:>3} "
            f"{prop['doors']:>3} {prop['ladders']:>4} {prop['spikes']:>4} | "
            f"{'WIN' if oracle_ok else 'FAIL':>7} | {play:>26}"
        )

    print("=" * 92)
    print(
        f"Oracle winnable: {oracle_win}/{len(HANDCRAFTED_LEVELS)}   "
        f"Greedy engine playthrough won: {engine_win}/{len(HANDCRAFTED_LEVELS)}"
    )
    print("\nNote: the oracle proves geometric winnability (all objectives reachable).")
    print("The greedy controller is a SIMPLE heuristic policy; a level it doesn't win")
    print("is still winnable if the oracle passed — it just needs smarter play.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
