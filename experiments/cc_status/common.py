# ruff: noqa: F401,F403,F405,I001
"""Shared imports and constants for Crystal Caves status sessions."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from config import Config  # noqa: E402
from src.ai.evaluator import Evaluator  # noqa: E402
from src.app.cli import create_parser  # noqa: E402
from src.app.crystal_curriculum import stage_epsilon_decay  # noqa: E402
from src.app.headless import HeadlessTrainer  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_drills import (  # noqa: E402
    BRIDGE_CAVES,
    CONTACT_CAVES,
    DRILL_CAVES,
    contact_pool_caves,
)
from src.game.crystal_caves_experiments import (  # noqa: E402
    install_crystal_caves_experiment_defaults,
)


class CrystalCavesExperimentConfig(Protocol):
    """Typed view of Crystal Caves flags installed only for status-session runs."""

    CRYSTAL_CAVES_BRIDGES: bool
    CRYSTAL_CAVES_CONTACT_LEVELS: bool
    CRYSTAL_CAVES_CONTACT_POOL_SIZE: int
    CRYSTAL_CAVES_CONTACT_POOL_SEED: int
    CRYSTAL_CAVES_HISTORY_STATE: bool
    CRYSTAL_CAVES_HISTORY_STEPS: int
    CRYSTAL_CAVES_ANTI_LOOP_REWARD: bool
    CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL: bool
    CRYSTAL_CAVES_NOVELTY_BONUS: bool
    CRYSTAL_CAVES_INVALID_INTERACT_PENALTY: bool
    CRYSTAL_CAVES_INVALID_SHOOT_PENALTY: bool
    CRYSTAL_CAVES_ROUTE_AUX_LOSS: bool
    CRYSTAL_CAVES_ROUTE_AUX_WEIGHT: float
    CRYSTAL_CAVES_ROUTE_AUX_DEADBAND: float
    CRYSTAL_CAVES_DEMO_ACTION_LOSS: bool
    CRYSTAL_CAVES_DEMO_ACTION_WEIGHT: float
    CRYSTAL_CAVES_DEMO_ACTION_MARGIN: float
    CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE: int
    CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT: float
    CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE: float
    CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS: bool
    CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT: float
    CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE: int
    CRYSTAL_CAVES_CORRECTION_ACTION_LOSS: bool
    CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT: float
    CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN: float
    CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE: int
    CRYSTAL_CAVES_CONTACT_ACTION_HEAD: bool
    CRYSTAL_CAVES_CONTACT_ACTION_WEIGHT: float
    CRYSTAL_CAVES_CONTACT_ACTION_BATCH_SIZE: int
    CRYSTAL_CAVES_CONTACT_ACTION_DISTANCE_NORM: float
    CRYSTAL_CAVES_POLICY_ANCHOR_LOSS: bool
    CRYSTAL_CAVES_POLICY_ANCHOR_WEIGHT: float
    CRYSTAL_CAVES_POLICY_ANCHOR_TEMPERATURE: float
    CRYSTAL_CAVES_POLICY_ANCHOR_MIN_TARGET_DISTANCE_NORM: float


def cc_experiment_config(config: Config) -> CrystalCavesExperimentConfig:
    """Return the typed experiment-only view installed by status-session setup."""

    return cast(CrystalCavesExperimentConfig, config)


TUTORIAL_MIN_EPSILON = 0.35
TUTORIAL_EPSILON_END_FRACTION = 0.3
NEAR_MISS_DISTANCE_BANDS = (10.0, 5.0, 3.0, 1.5)
CLOSE_ZONE_DISTANCE_TILES = 3.0
TUTORIAL_LEVEL_DIAGONAL_TILES = float(np.hypot(44.0, 18.0))
ROUTE_DEMO_VARIANTS = {"direct", "recovery", "sweep", "beam"}
ROUTE_BEAM_COMMIT_STEPS = 8
SELECTED_WEIGHT_SNAPSHOT_KIND = "cc_status_selected_weights_v1"
DEMO_SELECTION_MODES = {"all", "filtered-weighted"}
DEMO_JUMP_ACTIONS = {"JUMP", "LEFT_JUMP", "RIGHT_JUMP"}
DEMO_IDLE_INTERACT_ACTIONS = {"IDLE", "INTERACT"}
DEMO_CLOSE_ZONE_EXTRA_LABEL_SOURCES = {"scripted", "oracle"}
