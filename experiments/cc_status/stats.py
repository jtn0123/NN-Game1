"""Small aggregation helpers for status-session metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in keys}


def counter_tail(values: list[str], n: int = 100) -> dict[str, int]:
    return dict(Counter(values[-n:]))


def mean_tail(values: list[float], n: int = 100) -> float:
    return float(np.mean(values[-n:])) if values else 0.0


def max_or_zero(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def trainer_source_stats(trainer: Any) -> dict[str, Any]:
    vec_env = getattr(trainer, "vec_env", None)
    stats_fn = getattr(vec_env, "source_stats", None)
    if callable(stats_fn):
        return stats_fn()
    return {}


def contact_interleave_metric_aliases(source_stats: dict[str, Any]) -> dict[str, float]:
    """Flatten the B6 contact/full lane stats into stable top-level metric names."""

    contact = source_stats.get("contact") or {}
    full = source_stats.get("full") or {}
    if not contact and not full:
        return {}
    return {
        "contact_lane_win_rate_100": float(contact.get("win_rate_100", 0.0) or 0.0),
        "contact_lane_crystal_rate_100": float(contact.get("crystal_rate_100", 0.0) or 0.0),
        "contact_lane_exit_rate_100": float(contact.get("exit_rate_100", 0.0) or 0.0),
        "full_lane_progress_100": float(full.get("avg_progress_100", 0.0) or 0.0),
        "full_lane_first_crystal_rate_100": float(full.get("crystal_rate_100", 0.0) or 0.0),
    }


def trainer_reverse_start_stats(trainer: Any) -> dict[str, Any]:
    vec_env = getattr(trainer, "vec_env", None)
    stats_fn = getattr(vec_env, "reverse_start_stats", None)
    if callable(stats_fn):
        return stats_fn()
    return {}


def trainer_archive_stats(trainer: Any) -> dict[str, Any]:
    vec_env = getattr(trainer, "vec_env", None)
    stats_fn = getattr(vec_env, "archive_stats", None)
    if callable(stats_fn):
        return stats_fn()
    return {}
