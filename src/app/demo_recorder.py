"""Record human play as replayable action-sequence demos.

Every episode played in ``--human`` mode (with ``--record-demos``) is captured as
the exact list of discrete action ids fed to ``game.step``. Because the engine is
deterministic given a pinned level and an action list, a WON episode is a
complete, verifiable demonstration — the same format the offline planner
(``experiments/cc_status/demo_extract.py``) produces, consumable by
demo-seeding / backward-curriculum training and checkable open-loop with
``demo_extract.verify_stored``.

Lost episodes are saved too: attempts-per-level is the ground-truth difficulty
calibration the levels have never had.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HumanDemoRecorder:
    """Buffers per-frame actions and writes one JSON file per finished episode.

    Robust to mid-episode resets (the R key) and cave advancement: the game's own
    step counter is the episode clock — when it rewinds to the first step, any
    unfinished buffer is discarded as an aborted attempt.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._actions: List[int] = []
        self.saved: List[Path] = []

    @staticmethod
    def _level_identity(game: Any) -> Tuple[Optional[int], str]:
        """(index into HANDCRAFTED_LEVELS or None, level name) for the live cave."""
        level = getattr(game, "level", None)
        name = str(getattr(level, "name", f"level{getattr(game, 'level_index', 0)}"))
        try:
            from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS
        except ImportError:
            return None, name
        for i, spec in enumerate(HANDCRAFTED_LEVELS):
            if spec is level or getattr(spec, "name", None) == name:
                return i, name
        return None, name

    def after_step(self, game: Any, action: int, done: bool, info: Dict[str, Any]) -> None:
        """Call once after every ``game.step(action)`` in the human-play loop."""
        if int(getattr(game, "steps", 0)) <= 1:
            # A new episode just began (reset / R key / next cave): the previous
            # buffer never reached done, so it was an aborted attempt — drop it.
            self._actions = []
        self._actions.append(int(action))
        if done:
            self._save(game, info)
            self._actions = []

    def _save(self, game: Any, info: Dict[str, Any]) -> None:
        index, name = self._level_identity(game)
        won = bool(info.get("won", getattr(game, "won", False)))
        record = {
            "source": "human",
            "level_index": index,
            "level_name": name,
            "won": won,
            "end_reason": str(info.get("end_reason", "unknown")),
            "steps": len(self._actions),
            "actions": self._actions,
        }
        slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "level"
        prefix = f"L{index:02d}" if index is not None else "Lxx"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        outcome = "won" if won else "lost"
        path = self.out_dir / f"{prefix}_{slug}_{outcome}_{stamp}.json"
        n = 1
        while path.exists():  # several attempts can finish within one second
            path = self.out_dir / f"{prefix}_{slug}_{outcome}_{stamp}_{n}.json"
            n += 1
        path.write_text(json.dumps(record))
        self.saved.append(path)
        print(f"   🎬 demo saved: {path.name} ({record['end_reason']}, {record['steps']} steps)")
