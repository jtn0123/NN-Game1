"""Shared training performance presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PerformanceMode:
    """Training settings applied by dashboard performance-mode controls."""

    mode: str
    label: str
    learn_every: int
    batch_size: int
    gradient_steps: int
    description: str


PERFORMANCE_MODES: Dict[str, PerformanceMode] = {
    "normal": PerformanceMode(
        mode="normal",
        label="Normal",
        learn_every=1,
        batch_size=128,
        gradient_steps=1,
        description="Learn every step",
    ),
    "fast": PerformanceMode(
        mode="fast",
        label="Fast",
        learn_every=4,
        batch_size=128,
        gradient_steps=1,
        description="Learn every 4 steps",
    ),
    "turbo": PerformanceMode(
        mode="turbo",
        label="Turbo",
        learn_every=8,
        batch_size=128,
        gradient_steps=2,
        description="Learn every 8 steps + 2 gradient updates",
    ),
    "ultra": PerformanceMode(
        mode="ultra",
        label="Ultra",
        learn_every=32,
        batch_size=128,
        gradient_steps=2,
        description="Learn every 32 steps + 2 gradient updates",
    ),
}


def get_performance_mode(mode: str) -> PerformanceMode:
    """Return a known performance mode or raise KeyError."""
    return PERFORMANCE_MODES[mode]


def apply_performance_mode(config: Any, mode: str) -> PerformanceMode:
    """Apply a performance preset to a mutable config object."""
    preset = get_performance_mode(mode)
    config.LEARN_EVERY = preset.learn_every
    config.BATCH_SIZE = preset.batch_size
    config.GRADIENT_STEPS = preset.gradient_steps
    return preset


def performance_mode_payload() -> Dict[str, Dict[str, Any]]:
    """Return a serializable representation for UI/API use."""
    return {
        mode: {
            "label": preset.label,
            "learn_every": preset.learn_every,
            "batch_size": preset.batch_size,
            "gradient_steps": preset.gradient_steps,
            "description": preset.description,
        }
        for mode, preset in PERFORMANCE_MODES.items()
    }
