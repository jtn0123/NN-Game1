"""DQfD-lite demonstration learning: a fixed demo buffer + margin loss.

Loads verified winning demos (action-sequence JSONs from the human recorder or
the offline planner), replays them in a pinned engine to regenerate exact
transitions under the RUN's observation/reward config, precomputes n-step
returns matching the agent's replay semantics, and provides the two DQfD
ingredients the completion literature says actually matter:

  1. a NEVER-overwritten demo buffer sampled alongside regular replay, and
  2. a large-margin supervised loss that pushes the demonstrated action's
     Q-value above all alternatives by a margin (ablations show naive
     buffer-dumping without the margin loss fails).

Usage (wired by the training harness when config.DEMO_DIR is set):
    store = DemoStore.from_dir(config.DEMO_DIR, config)
    agent.attach_demo_store(store)
    agent.pretrain_on_demos(config.DEMO_PRETRAIN_STEPS)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


def _load_demo_files(demo_dir: Path) -> List[dict]:
    """All winning, level-indexed demo records in ``demo_dir`` (both the human
    recorder's schema and the planner's ``{"level": .., "actions": ..}`` schema)."""
    records = []
    for path in sorted(demo_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        level = rec.get("level_index", rec.get("level"))
        actions = rec.get("actions")
        won = rec.get("won", True)  # planner files only store verified wins
        if level is None or not actions or not won:
            continue
        try:
            record = {"level": int(level), "actions": [int(a) for a in actions]}
        except (TypeError, ValueError):
            continue  # one malformed demo file must not abort the whole store
        records.append(record)
    return records


def _pinned_game(config: Any, level: int) -> Any:
    """A deterministic engine pinned to one hand-crafted level (same pinning the
    open-loop demo verifier uses, so replays here match verified replays)."""
    from src.game.crystal_caves import CrystalCaves
    from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS

    spec = HANDCRAFTED_LEVELS[level]
    game = CrystalCaves(config, headless=True)
    game.CAVES = (spec,)
    game._randomize_levels = False
    game.use_eval_levels(1)
    game.reset_eval_cursor()
    return game


class DemoStore:
    """Fixed demonstration transitions with n-step returns, ready for TD + margin."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        n_step_lengths: np.ndarray,
        n_episodes: int,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.n_step_lengths = n_step_lengths
        self.n_episodes = n_episodes

    def __len__(self) -> int:
        return len(self.actions)

    def sample(self, k: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, len(self.actions), size=min(k, len(self.actions)))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.n_step_lengths[idx],
        )

    @classmethod
    def from_dir(cls, demo_dir: str, config: Any) -> Optional["DemoStore"]:
        """Replay every winning demo under the run's config; None if none exist."""
        records = _load_demo_files(Path(demo_dir))
        if not records:
            return None

        n_step = (
            int(getattr(config, "N_STEP_SIZE", 1))
            if getattr(config, "USE_N_STEP_RETURNS", False)
            else 1
        )
        gamma = float(config.GAMMA)
        opening = int(getattr(config, "DEMO_OPENING_ONLY_STEPS", 0))

        all_s: List[np.ndarray] = []
        all_a: List[int] = []
        all_r: List[float] = []
        all_ns: List[np.ndarray] = []
        all_d: List[float] = []
        all_n: List[int] = []
        for rec in records:
            game = _pinned_game(config, rec["level"])
            state = game.reset()
            episode: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
            for action in rec["actions"]:
                next_state, reward, done, _info = game.step(action)
                episode.append((state, action, float(reward), next_state, bool(done)))
                state = next_state
                if done:
                    break
            # n-step returns with the same semantics as the agent's replay buffer:
            # R = sum_{i<n} gamma^i r_{t+i}; next_state/dones taken at t+n (or the
            # terminal), n_step_lengths = the actual horizon used.
            length = len(episode)
            # Opening-only mode keeps just the route's first `opening` transitions;
            # n-step tails may still look past the cutoff into the full episode.
            kept = min(length, opening) if opening > 0 else length
            for t in range(kept):
                horizon = min(n_step, length - t)
                ret = 0.0
                for i in range(horizon):
                    ret += (gamma**i) * episode[t + i][2]
                tail = episode[t + horizon - 1]
                all_s.append(episode[t][0])
                all_a.append(episode[t][1])
                all_r.append(ret)
                all_ns.append(tail[3])
                all_d.append(1.0 if tail[4] else 0.0)
                all_n.append(horizon)

        if not all_a:
            return None
        return cls(
            states=np.asarray(all_s, dtype=np.float32),
            actions=np.asarray(all_a, dtype=np.int64),
            rewards=np.asarray(all_r, dtype=np.float32),
            next_states=np.asarray(all_ns, dtype=np.float32),
            dones=np.asarray(all_d, dtype=np.float32),
            n_step_lengths=np.asarray(all_n, dtype=np.int64),
            n_episodes=len(records),
        )


def demo_prefix_registry(demo_dir: str) -> dict:
    """level_index -> list of action sequences, for demo-prefix episode starts
    (the cheap half of the backward-curriculum technique). Actions only — no
    engine replay needed at load time."""
    registry: dict = {}
    for rec in _load_demo_files(Path(demo_dir)):
        registry.setdefault(rec["level"], []).append(rec["actions"])
    return registry
