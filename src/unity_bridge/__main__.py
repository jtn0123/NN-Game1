"""Run with python -m src.unity_bridge. No training or checkpoint writes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from config import Config
from src.ai.agent import Agent

from .server import BridgeServer
from .session import CaveSession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--level", type=int, default=0, help="handcrafted cave index, 0-15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model", type=Path, help="compatible Crystal Caves checkpoint for Watch AI"
    )
    parser.add_argument(
        "--cnn-state", action="store_true", help="use SpatialDQN for a CNN checkpoint"
    )
    parser.add_argument(
        "--legacy-state", action="store_true", help="use the older 119-feature observation"
    )
    parser.add_argument("--record-demos", help="save completed human episodes in this directory")
    args = parser.parse_args()
    if not 0 <= args.level < 16 or not 1 <= args.port <= 65535:
        parser.error("level must be 0-15 and port must be 1-65535")
    config = Config(GAME_NAME="crystal_caves", CRYSTAL_CAVES_IMPORTED=True)
    config.FORCE_CPU = True
    config.USE_CNN_STATE = args.cnn_state
    config.CRYSTAL_CAVES_RICH_STATE = not args.legacy_state
    config.USE_TORCH_COMPILE = False
    config.MEMORY_SIZE = 1  # Inference only; there is no replay or training here.
    probe = CaveSession(level=args.level, seed=args.seed, config=config)
    policy = None
    if args.model:
        agent = Agent(probe.game.state_size, probe.game.action_size, config=config)
        if not agent.load_weights_only(str(args.model), quiet=False):
            parser.error(
                "checkpoint is missing or incompatible with the pilot's observation/network configuration"
            )
        agent.policy_net.eval()
        policy = agent.get_q_values
    probe.game.close()
    with BridgeServer(
        ("127.0.0.1", args.port),
        lambda: CaveSession(
            level=args.level,
            seed=args.seed,
            config=config,
            policy=policy,
            policy_name=args.model.name if args.model else "",
            record_dir=args.record_demos,
        ),
    ) as server:
        print(f"Crystal Caves Unity bridge ready on 127.0.0.1:{args.port}", flush=True)
        print(
            "Unity owns presentation; Python owns every simulation step. Ctrl-C stops the bridge.",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
