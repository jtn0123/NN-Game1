#!/usr/bin/env python3
"""
Neural Network Game AI - Main Entry Point
==========================================

This is the main script that runs the AI training with live visualization.

Usage:
    # Train with visualization (default)
    python main.py

    # Train without visualization (faster)
    python main.py --headless

    # TURBO MODE: Maximum speed training (~5000 steps/sec on M4!)
    python main.py --headless --turbo --episodes 5000

    # TURBO + Web Dashboard: Best of both worlds
    python main.py --headless --turbo --web --port 5001

    # VECTORIZED: 8 parallel games for ~3x additional speedup
    python main.py --headless --turbo --vec-envs 8

    # Custom performance tuning
    python main.py --headless --learn-every 4 --batch-size 256

    # Play with a trained model
    python main.py --play --model models/breakout_best.pth

    # Human play mode (for testing game)
    python main.py --human

    # Custom training parameters
    python main.py --episodes 5000 --lr 0.0001

Performance Options:
    --headless        Skip pygame entirely for max throughput
    --turbo           Preset: learn-every=8, batch=128, grad-steps=2 (~5000 steps/sec)
    --learn-every N   Learn every N steps (default: 1, try 4 for ~4x speedup)
    --batch-size N    Training batch size (default: 128)
    --torch-compile   Enable torch.compile() for ~20-50% extra speedup

The visualization shows:
    - Left side: The game (Breakout)
    - Right side: Neural network with live activations
    - Bottom: Training metrics dashboard

Press:
    - ESC or Q: Quit
    - P: Pause/Resume training
    - S: Save current model
    - R: Reset episode
    - +/-: Adjust game speed
    - F: Toggle fullscreen
"""

# Suppress pygame's pkg_resources deprecation warning (pygame issue #4557)
# This is a known issue - pygame hasn't migrated to importlib.resources yet
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import os
import sys
from typing import Optional

import numpy as np
import pygame

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.app.cli import parse_args
from src.app.headless import HeadlessTrainer
from src.app.interactive import GameApp
from src.app.model_commands import inspect_model, list_models
from src.app.web_launcher import run_web_mode
from src.game import (
    GameMenu,
    get_game_info,
    list_games,
)
from src.utils.logger import LogLevel, setup_logging


def terminal_game_selector() -> Optional[str]:
    """Terminal-based game selector for headless mode.

    Returns:
        Selected game name, or None if cancelled
    """
    available = list_games()

    print("\n" + "=" * 60)
    print("   SELECT A GAME TO TRAIN")
    print("=" * 60)

    # Difficulty color codes (for terminals that support it)
    difficulty_indicators = {"Easy": "(Easy)", "Medium": "(Medium)", "Hard": "(Hard)"}

    for i, game_name in enumerate(available, 1):
        game_info = get_game_info(game_name)
        if game_info:
            difficulty = game_info.get("difficulty", "Medium")
            difficulty_str = difficulty_indicators.get(difficulty, "")
            actions = game_info.get("actions", [])
            action_str = ", ".join(actions) if actions else "N/A"

            print(f"\n  [{i}] {game_info['icon']} {game_info['name']} {difficulty_str}")
            print(f"      {game_info['description']}")
            print(f"      Actions: {action_str}")
        else:
            print(f"\n  [{i}] {game_name}")

    print("\n  [0] Exit")
    print("\n" + "=" * 60)

    while True:
        try:
            choice = input("\nEnter number (1-{0}): ".format(len(available)))
            if choice.strip() == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx]
            print(f"Please enter a number between 1 and {len(available)}")
        except ValueError:
            print("Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print()
            return None


def print_startup_banner() -> None:
    """Print a welcome banner for the application."""
    print()
    print("=" * 60)
    print("       DQN GAME AI - Deep Q-Learning Trainer")
    print("=" * 60)
    print("   Train neural networks to play classic arcade games!")
    print()
    print("   Quick Start:")
    print("   - python main.py              # Visual training (default)")
    print("   - python main.py --human      # Play a game yourself")
    print("   - python main.py --headless   # Fast training (no display)")
    print("   - python main.py --help       # See all options")
    print("=" * 60)


def main():
    """Main entry point."""
    # Show banner for non-help invocations
    if "--help" not in sys.argv and "-h" not in sys.argv:
        print_startup_banner()

    args = parse_args()

    # Handle --inspect command (no pygame needed)
    if args.inspect:
        inspect_model(args.inspect)
        return

    # Handle --list-models command (no pygame needed)
    if args.list_models:
        list_models()
        return

    # Load config
    config = Config()

    # Initialize logging from config
    log_level = LogLevel[config.LOG_LEVEL.upper()]
    setup_logging(
        log_dir=config.LOG_DIR,
        level=log_level,
        console_output=config.LOG_TO_CONSOLE,
        file_output=config.LOG_TO_FILE,
    )

    # Web mode: use web interface for everything
    if hasattr(args, "web") and args.web:
        run_web_mode(config, args)
        return

    # Show game selection menu if requested OR if no game specified (visual mode only)
    show_menu = (hasattr(args, "menu") and args.menu) or (args.game is None and not args.headless)

    if show_menu:
        pygame.init()
        menu_screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("🧠 Neural Network AI - Game Selection")
        menu_clock = pygame.time.Clock()

        menu = GameMenu(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        selected_game = menu.run(menu_screen, menu_clock)

        if selected_game is None:
            print("No game selected. Exiting.")
            pygame.quit()
            return

        args.game = selected_game
        pygame.quit()  # Quit and reinitialize for proper game window
        print(f"🎮 Selected: {selected_game}")

    # Set game from CLI argument
    if hasattr(args, "game") and args.game:
        config.GAME_NAME = args.game
        game_info = get_game_info(args.game)
        if game_info:
            print(f"🎮 Game: {game_info['icon']} {game_info['name']}")

    # Crystal Caves: procedurally generated caves
    if getattr(args, "random_caves", False):
        config.CRYSTAL_CAVES_PROCEDURAL = True
        print("🎲 Crystal Caves: procedurally generated caves")
    if getattr(args, "cave_families", None):
        config.CRYSTAL_CAVES_FAMILIES = args.cave_families
        print(f"🎓 Crystal Caves families: {args.cave_families}")
    if getattr(args, "cave_difficulty", None):
        config.CRYSTAL_CAVES_DIFFICULTY = args.cave_difficulty
        print(f"🎚️  Crystal Caves difficulty: {args.cave_difficulty}")
    if getattr(args, "legacy_state", False):
        config.CRYSTAL_CAVES_RICH_STATE = False
        print("🔭 Crystal Caves: legacy 11x9 state (rich state OFF)")
    if getattr(args, "cnn", False):
        config.USE_CNN_STATE = True
        print("🧠 CNN Q-network: reading the perception window as a 2D grid")
    if getattr(args, "lr_decay", False):
        config.LR_DECAY = True
        print("📉 LR decay: cosine to LR_MIN over the run (stabilizes late training)")

    # Force CPU if specified (faster for small models on M4)
    if hasattr(args, "cpu") and args.cpu:
        config.FORCE_CPU = True
        print("💻 CPU mode: Using CPU (faster for small models on M4)")

    # Set seed if specified
    if args.seed:
        np.random.seed(args.seed)
        import torch

        torch.manual_seed(args.seed)
        config.SEED = args.seed

    # Handle headless mode separately (no pygame)
    if args.headless:
        # If no game specified and web mode, run in launcher mode
        if args.game is None and args.web:
            run_web_mode(config, args)
            return

        # If no game specified, show terminal game selector
        if args.game is None:
            selected = terminal_game_selector()
            if selected is None:
                print("No game selected. Exiting.")
                return
            args.game = selected
            config.GAME_NAME = selected
            game_info = get_game_info(selected)
            if game_info:
                print(f"🎮 Selected: {game_info['icon']} {game_info['name']}")
            else:
                print(f"🎮 Selected: {selected}")

        trainer = HeadlessTrainer(config, args)
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            trainer._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
        finally:
            # Clean up web dashboard if running
            if trainer.web_dashboard:
                trainer.web_dashboard.stop()
        return

    # Apply CLI overrides to config for visualized mode
    if args.learn_every:
        config.LEARN_EVERY = args.learn_every
    if args.gradient_steps:
        config.GRADIENT_STEPS = args.gradient_steps
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.torch_compile:
        config.USE_TORCH_COMPILE = True

    # Apply turbo preset - optimized for M4 CPU based on benchmarks
    if args.turbo:
        config.LEARN_EVERY = 8
        config.BATCH_SIZE = 128
        config.GRADIENT_STEPS = 2
        config.USE_TORCH_COMPILE = False
        config.FORCE_CPU = True
        print("🚀 Turbo mode: CPU, B=128, LE=8, GS=2 (~5000 steps/sec on M4)")

    # Create application (with pygame) and run - supports returning to menu
    app = None
    while True:
        try:
            app = GameApp(config, args)

            # Run appropriate mode
            if args.human:
                app.run_human_mode()
            elif args.play:
                app.run_play_mode()
            else:
                app.run_training()

            # Check if user wants to return to game selector
            if not app.return_to_menu:
                break  # Normal exit

            # Return to game selector
            print("\n🏠 Returning to game selector...")
            pygame.quit()  # Close current window

            selected = terminal_game_selector()
            if not selected:
                print("No game selected. Exiting.")
                break

            # Update config for new game
            config.GAME_NAME = selected
            args.game = selected
            print(f"\n🎮 Starting {selected}...")

            # Reinitialize pygame for new game
            pygame.init()

        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            if app:
                app._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
                if app.web_dashboard:
                    app.web_dashboard.log("⛔ Training interrupted by user", "warning")
            break

    # Clean up resources
    if app and app.web_dashboard:
        app.web_dashboard.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
