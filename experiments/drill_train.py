"""Train on the single-skill drill levels and report per-skill mastery.

This is the skill diagnostic: train on the hand-authored drills, then greedily eval
each drill on its own to see which motor skills the agent can learn in isolation
(walk, jump-up, jump-gap, drop-and-climb, staircase, collect-then-reach-exit). A skill
that stays at ~0% even on its dedicated drill is the real wall.

Usage:
    python experiments/drill_train.py --episodes 600 --seed 0 --eval-k 16
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import Config  # noqa: E402
from src.app.cli import create_parser  # noqa: E402
from src.app.headless import HeadlessTrainer  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_drills import DRILL_CAVES  # noqa: E402


def _make_config(episodes: int) -> Config:
    config = Config()
    config.GAME_NAME = "crystal_caves"
    config.CRYSTAL_CAVES_DRILLS = True
    config.CRYSTAL_CAVES_PROCEDURAL = False
    config.USE_CNN_STATE = True
    config.FORCE_CPU = True
    config.MAX_EPISODES = episodes
    config.EVAL_EVERY = 0  # we run our own per-drill eval below
    return config


def per_skill_eval(agent, config: Config, k: int) -> None:
    """Greedily run each drill k times and print a per-skill mastery table."""
    game = CrystalCaves(config, headless=True)
    game._randomize_levels = False
    print(f"\n{'drill':<36}{'win%':>7}{'crystal%':>10}{'reached-exit%':>15}")
    print("-" * 68)
    for i, spec in enumerate(DRILL_CAVES):
        wins = collected = reached_exit = 0
        for _ in range(k):
            game.level_index = i
            state = game.reset()
            initial = max(1, game.initial_crystals)
            done = False
            info: dict = {}
            steps = 0
            while not done and steps < config.EVAL_MAX_STEPS:
                action = agent.select_action(state, training=False)
                state, _, done, info = game.step(action)
                steps += 1
            wins += int(info.get("won", False))
            collected += int(len(game.crystals) < initial)  # grabbed at least one
            reached_exit += int(game.exit_unlocked and info.get("won", False))
        print(
            f"{spec.name[:35]:<36}{100*wins/k:>6.0f}%{100*collected/k:>9.0f}%{100*reached_exit/k:>14.0f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-k", type=int, default=16)
    opts = ap.parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    config = _make_config(opts.episodes)
    args = create_parser().parse_args(
        ["--headless", "--game", "crystal_caves", "--cnn", "--vec-envs", "8"]
    )
    args.vec_envs = 8
    args.episodes = opts.episodes

    print(f"===== DRILL TRAIN: {opts.episodes} episodes, seed {opts.seed} =====")
    trainer = HeadlessTrainer(config, args)
    trainer.train()
    per_skill_eval(trainer.agent, config, opts.eval_k)
    print("===== DRILL TRAIN COMPLETE =====")


if __name__ == "__main__":
    main()
