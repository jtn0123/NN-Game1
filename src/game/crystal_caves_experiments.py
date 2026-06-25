"""Optional Crystal Caves experiment defaults.

These knobs are intentionally kept out of the global Config surface because
they represent targeted probes, rejected ideas, or training-only diagnostics.
Experiment runners can install them on a Config instance when they need exact
reproducibility for old runs.
"""

from __future__ import annotations

from typing import Any

CRYSTAL_CAVES_EXPERIMENT_DEFAULTS: dict[str, Any] = {
    "CRYSTAL_CAVES_BRIDGES": False,
    "CRYSTAL_CAVES_ANTI_LOOP_REWARD": False,
    "CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL": False,
    "CRYSTAL_CAVES_NOVELTY_BONUS": False,
    "CRYSTAL_CAVES_INVALID_INTERACT_PENALTY": False,
    "CRYSTAL_CAVES_INVALID_SHOOT_PENALTY": False,
    "CRYSTAL_CAVES_ROUTE_AUX_LOSS": False,
    "CRYSTAL_CAVES_ROUTE_AUX_WEIGHT": 0.05,
    "CRYSTAL_CAVES_ROUTE_AUX_DEADBAND": 0.01,
    "CRYSTAL_CAVES_DEMO_ACTION_LOSS": False,
    "CRYSTAL_CAVES_DEMO_ACTION_WEIGHT": 0.05,
    "CRYSTAL_CAVES_DEMO_ACTION_MARGIN": 0.8,
    "CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE": 64,
    "CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT": 0.0,
    "CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE": 1.0,
    "CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS": False,
    "CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT": 0.03,
    "CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE": 64,
    "CRYSTAL_CAVES_CORRECTION_ACTION_LOSS": False,
    "CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT": 0.02,
    "CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN": 0.6,
    "CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE": 64,
}


def install_crystal_caves_experiment_defaults(config: object) -> None:
    """Attach optional Crystal Caves experiment defaults if absent."""

    for name, value in CRYSTAL_CAVES_EXPERIMENT_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, value)
