"""Command-line interface construction for NN-Game1."""

from __future__ import annotations

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create the application argument parser."""
    from src.game import list_games

    available_games = list_games()

    parser = argparse.ArgumentParser(
        description="DQN Game AI - Train neural networks to play classic arcade games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
EXAMPLES
========

Getting Started:
    python main.py                    Train with visual display (shows game selection)
    python main.py --human            Play a game yourself to test it
    python main.py --game pong        Train a specific game directly

Fast Training (Recommended):
    python main.py --headless --turbo          ~5000 steps/sec on M4 Mac
    python main.py --headless --turbo --web    With web dashboard at localhost:5000
    python main.py --headless --vec-envs 8     Parallel training (~12,000 steps/sec)

Watch Trained Agent:
    python main.py --play --model models/pong_best.pth

Model Management:
    python main.py --list-models               Show all saved models
    python main.py --inspect models/best.pth   Inspect model metadata

AVAILABLE GAMES: {', '.join(available_games)}

TIPS
====
- Use --headless for 10x faster training (no display overhead)
- Add --turbo for optimized batch settings (~4x faster)
- Add --vec-envs 8 for parallel environments (~2-3x faster)
- Use --web to monitor training in browser at http://localhost:5000
- Press Ctrl+C to gracefully stop training (model auto-saves)
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--play",
        action="store_true",
        help="Play mode: watch trained agent without training",
    )
    mode_group.add_argument(
        "--human", action="store_true", help="Human mode: play the game yourself"
    )
    mode_group.add_argument(
        "--headless",
        action="store_true",
        help="Headless training: no visualization (faster)",
    )
    mode_group.add_argument(
        "--inspect",
        type=str,
        metavar="MODEL_PATH",
        help="Inspect a model file and show its metadata",
    )
    mode_group.add_argument(
        "--list-models",
        action="store_true",
        help="List all saved models with their metadata",
    )

    parser.add_argument(
        "--game",
        type=str,
        default=None,
        choices=available_games,
        help=f'Game to train/play. If not specified, shows game selection. Available: {", ".join(available_games)}',
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Show game selection menu on launch (interactive game picker)",
    )
    parser.add_argument(
        "--random-caves",
        action="store_true",
        help="Crystal Caves: play procedurally generated caves instead of the authored ones",
    )
    parser.add_argument(
        "--cave-families",
        type=str,
        default=None,
        help="Crystal Caves: restrict generated level families (comma-separated, "
        "e.g. platform_network,snake_bands). Used for curriculum training.",
    )
    parser.add_argument(
        "--legacy-state",
        action="store_true",
        help="Crystal Caves: use the legacy 11x9 perception window (119-feature "
        "state) instead of the AI-1 rich state (19x11 + global objective map). "
        "For head-to-head state-representation comparisons.",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Use a convolutional Q-network that reads the perception window as a "
        "2D grid (exploits spatial structure; recommended for the rich state).",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="End a run once eval win rate/score plateaus, instead of training the "
        "live policy past its peak into collapse (keeps frequent evals).",
    )
    parser.add_argument(
        "--cave-difficulty",
        type=str,
        choices=["tutorial", "easy", "normal"],
        default=None,
        help="Crystal Caves: objective/threat budget for generated caves. "
        "'tutorial' is the simplest winnable level (1 open crystal, no lock); "
        "'easy' is a learnable curriculum floor (few crystals, no threats); "
        "'normal' is the full game (default).",
    )
    parser.add_argument(
        "--imported",
        action="store_true",
        help="Crystal Caves: play/train on the 16 hand-crafted Episode-1 levels "
        "(sets CRYSTAL_CAVES_IMPORTED).",
    )
    parser.add_argument(
        "--record-demos",
        type=str,
        nargs="?",
        const="experiments/cc_status/demos/human",
        default=None,
        metavar="DIR",
        help="Human mode: record every finished episode as a replayable "
        "action-sequence JSON in DIR (default experiments/cc_status/demos/human). "
        "Wins double as training demonstrations.",
    )
    parser.add_argument(
        "--crystal-curriculum",
        action="store_true",
        help="Crystal Caves: run the staged curriculum instead of one fixed "
        "difficulty/family. Stages warm-start from eval-best checkpoints and "
        "advance from tutorial/easy to normal mixed caves. In this mode "
        "--episodes is the total curriculum budget.",
    )
    parser.add_argument(
        "--curriculum-stage-episodes",
        type=int,
        default=None,
        help="Crystal Caves curriculum: override the episode budget for every "
        "stage. If omitted, --episodes is spread across stage weights.",
    )

    parser.add_argument("--model", type=str, default=None, help="Path to model file to load")

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (default: unlimited, trains until stopped)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use for training",
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web dashboard for remote monitoring (http://localhost:5000)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for web dashboard (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for web dashboard (default: 127.0.0.1; use 0.0.0.0 only for trusted networks)",
    )

    parser.add_argument(
        "--learn-every",
        type=int,
        default=None,
        help="Learn every N steps (default: 1, try 4 for ~4x speedup)",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Number of gradient updates per learning call (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 128, try 256 for M4)",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Turbo mode preset: learn-every 8, batch 128, 2 grad steps (~5000 steps/sec on M4)",
    )
    parser.add_argument(
        "--vec-envs",
        type=int,
        default=1,
        help="Number of parallel environments for vectorized training (default: 1, try 8 for ~3x speedup)",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile() for ~20-50%% speedup (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--lr-decay",
        action="store_true",
        help="Cosine-decay the learning rate to LR_MIN over the run's episodes "
        "(freezes the policy near its peak; stabilizes late-training win rate)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (faster than MPS for small models on M4)",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return create_parser().parse_args()
