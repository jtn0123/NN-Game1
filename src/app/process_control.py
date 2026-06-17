"""Process-level helpers for restarting game modes."""

from __future__ import annotations

import argparse
import os
import sys
import time


def restart_with_game(game_name: str, args: argparse.Namespace) -> None:
    """Restart the current process with a different selected game."""
    new_args = [sys.executable, sys.argv[0]]
    new_args.extend(["--game", game_name])

    if args.headless:
        new_args.append("--headless")
    if args.web:
        new_args.extend(["--web", "--port", str(args.port)])
    if hasattr(args, "turbo") and args.turbo:
        new_args.append("--turbo")
    if hasattr(args, "vec_envs") and args.vec_envs and args.vec_envs > 1:
        new_args.extend(["--vec-envs", str(args.vec_envs)])
    if args.episodes:
        new_args.extend(["--episodes", str(args.episodes)])
    if hasattr(args, "cpu") and args.cpu:
        new_args.append("--cpu")

    print(f"\nRestarting with {game_name}...")
    print(f"Command: {' '.join(new_args)}\n")
    time.sleep(0.3)

    os.execv(sys.executable, new_args)
