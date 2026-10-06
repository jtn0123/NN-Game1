"""Full training-state snapshots: pause/resume without progress loss.

A weights-only checkpoint loses the replay buffer, optimizer moments, and
schedule counters — resumed runs then destroy their own policy (the phase-1
"resume law"). A snapshot bundles EVERYTHING the live process carries:

  - policy + target network state_dicts, optimizer (and scheduler) state
  - the complete replay buffer arrays (incl. PER priorities and the n-step
    pending queue), write position and fill size
  - agent schedule state: epsilon, demo-margin scale
  - the backward-ladder shared dicts (offsets + win credits)
  - RNG states (numpy / torch / python), best-effort for approximate replay

Bundles are ~1.4GB (buffer-dominated). ``rotate_snapshots`` keeps the newest
``keep`` bundles so steady-state disk stays under ~3GB per device (5GB cap
with write-transient, per operating agreement). Writes are atomic
(tmp + os.replace) so a crash mid-write never corrupts the latest bundle.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

SNAPSHOT_PREFIX = "snapshot_ep"


def _buffer_payload(mem: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "class": type(mem).__name__,
        "capacity": mem.capacity,
        "states": mem.states,
        "actions": mem.actions,
        "rewards": mem.rewards,
        "next_states": mem.next_states,
        "dones": mem.dones,
        "position": mem._position,
        "size": mem._size,
        "initialized": getattr(mem, "_initialized", True),
    }
    if hasattr(mem, "priorities"):  # PER family
        payload["priorities"] = mem.priorities
        payload["max_priority"] = mem.max_priority
        payload["beta"] = mem.beta
        payload["frame_count"] = getattr(mem, "_frame_count", 0)
    if hasattr(mem, "_n_step_buffer"):  # n-step family
        payload["n_step_pending"] = list(mem._n_step_buffer)
    return payload


def _restore_buffer(mem: Any, payload: Dict[str, Any]) -> None:
    if payload["capacity"] != mem.capacity:
        raise ValueError(
            f"snapshot capacity {payload['capacity']} != buffer capacity {mem.capacity}"
        )
    if payload["states"].shape != mem.states.shape:
        raise ValueError(
            f"snapshot state shape {payload['states'].shape} != buffer {mem.states.shape}"
        )
    np.copyto(mem.states, payload["states"])
    np.copyto(mem.actions, payload["actions"])
    np.copyto(mem.rewards, payload["rewards"])
    np.copyto(mem.next_states, payload["next_states"])
    np.copyto(mem.dones, payload["dones"])
    mem._position = int(payload["position"])
    mem._size = int(payload["size"])
    if hasattr(mem, "_initialized"):
        mem._initialized = bool(payload["initialized"])
    if hasattr(mem, "priorities") and "priorities" in payload:
        np.copyto(mem.priorities, payload["priorities"])
        mem.max_priority = float(payload["max_priority"])
        mem.beta = float(payload["beta"])
        if hasattr(mem, "_frame_count"):
            mem._frame_count = int(payload["frame_count"])
    if hasattr(mem, "_n_step_buffer") and "n_step_pending" in payload:
        mem._n_step_buffer = list(payload["n_step_pending"])


def save_training_snapshot(
    agent: Any,
    directory: str | Path,
    episode: int,
    ladder_offsets: Optional[Dict[int, int]] = None,
    ladder_wins: Optional[Dict[int, int]] = None,
    keep: int = 2,
) -> Path:
    """Write a complete training-state bundle atomically; prune to ``keep``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle: Dict[str, Any] = {
        "meta": {
            "episode": int(episode),
            "saved_at": time.time(),
            "state_size": agent.state_size,
            "action_size": agent.action_size,
        },
        "policy": agent.policy_net.state_dict(),
        "target": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict() if agent.scheduler else None,
        "epsilon": float(agent.epsilon),
        "demo_margin_scale": float(getattr(agent, "_demo_margin_scale", 1.0)),
        "buffer": _buffer_payload(agent.memory),
        "ladder_offsets": dict(ladder_offsets or {}),
        "ladder_wins": dict(ladder_wins or {}),
        "rng": {
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        },
    }
    final = directory / f"{SNAPSHOT_PREFIX}{episode}.pt"
    tmp = directory / f"{SNAPSHOT_PREFIX}{episode}.pt.tmp"
    torch.save(bundle, tmp)
    os.replace(tmp, final)
    rotate_snapshots(directory, keep=keep)
    return final


def rotate_snapshots(directory: str | Path, keep: int = 2) -> None:
    """Delete all but the ``keep`` highest-episode bundles (and stray tmps)."""
    directory = Path(directory)
    for stray in directory.glob(f"{SNAPSHOT_PREFIX}*.pt.tmp"):
        stray.unlink(missing_ok=True)
    bundles = sorted(
        directory.glob(f"{SNAPSHOT_PREFIX}*.pt"),
        key=lambda p: int(p.stem.removeprefix(SNAPSHOT_PREFIX)),
    )
    for old in bundles[:-keep] if keep > 0 else bundles:
        old.unlink(missing_ok=True)


def latest_snapshot(directory: str | Path) -> Optional[Path]:
    directory = Path(directory)
    bundles = sorted(
        directory.glob(f"{SNAPSHOT_PREFIX}*.pt"),
        key=lambda p: int(p.stem.removeprefix(SNAPSHOT_PREFIX)),
    )
    return bundles[-1] if bundles else None


def load_training_snapshot(agent: Any, path: str | Path) -> Dict[str, Any]:
    """Restore the complete training state into a freshly-built agent.

    Returns the bundle's meta dict (incl. the episode number the snapshot was
    taken at — the harness uses it to offset schedule counters so epsilon and
    the demo-margin decay continue where they left off).
    """
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    agent.policy_net.load_state_dict(bundle["policy"])
    agent.target_net.load_state_dict(bundle["target"])
    agent.optimizer.load_state_dict(bundle["optimizer"])
    if agent.scheduler and bundle.get("scheduler"):
        agent.scheduler.load_state_dict(bundle["scheduler"])
    agent.epsilon = float(bundle["epsilon"])
    agent._demo_margin_scale = float(bundle["demo_margin_scale"])
    _restore_buffer(agent.memory, bundle["buffer"])
    try:
        np.random.set_state(bundle["rng"]["numpy"])
        torch.set_rng_state(bundle["rng"]["torch"])
        random.setstate(bundle["rng"]["python"])
    except (ValueError, TypeError):  # cross-platform RNG mismatch: not fatal
        pass
    meta = dict(bundle["meta"])
    meta["ladder_offsets"] = dict(bundle.get("ladder_offsets", {}))
    meta["ladder_wins"] = dict(bundle.get("ladder_wins", {}))
    return meta
