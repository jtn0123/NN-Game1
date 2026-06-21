"""Short, controlled A/B harness for Crystal Caves tutorial-tier reward/exploration
experiments.

Runs ONE tutorial curriculum stage (vec8 + CNN, the production path) for a fixed
episode budget with seeded RNG, then prints the held-out eval trajectory so two
code states can be compared on identical (deterministically-generated) caves.

Usage:
    python experiments/ab_tutorial.py --label baseline --episodes 250 --seed 0
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.app import crystal_curriculum as cc  # noqa: E402
from src.app.cli import create_parser  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    # Limit the curriculum to a single tutorial stage so the run stays on the
    # tutorial tier and finishes after one gate eval (index == len(stages)).
    cc.DEFAULT_CRYSTAL_CURRICULUM = (
        cc.CrystalCurriculumStage(
            stage_id="tutorial_platform",
            name=f"AB:{opts.label}",
            difficulty="tutorial",
            families="platform_network",
            default_episodes=opts.episodes,
            min_epsilon=0.35,
            gate="ab tutorial probe",
        ),
    )

    # A complete args namespace from the real parser (all defaults present).
    args = create_parser().parse_args(
        ["--crystal-curriculum", "--episodes", str(opts.episodes), "--headless"]
    )

    config = Config()
    config.GAME_NAME = "crystal_caves"

    print(f"\n===== A/B RUN: {opts.label} (episodes={opts.episodes}, seed={opts.seed}) =====")
    cc.run_crystal_curriculum(config, args)
    print(f"===== A/B RUN COMPLETE: {opts.label} =====\n")


if __name__ == "__main__":
    main()
