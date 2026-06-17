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

import argparse
import os
import sys
import time
from typing import Any, Optional, Type

import numpy as np
import pygame

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.ai.agent import Agent
from src.app.cli import parse_args
from src.app.headless import HeadlessTrainer
from src.app.interactive import GameApp
from src.game import (
    GameMenu,
    get_game_info,
    list_games,
)
from src.utils.logger import LogLevel, setup_logging

# Optional web dashboard
WEB_AVAILABLE: bool
WebDashboard: Optional[Type[Any]]
try:
    from src.web import WebDashboard as _WebDashboard

    WebDashboard = _WebDashboard
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    WebDashboard = None


def inspect_model(filepath: str) -> None:
    """Inspect a model file and display its metadata."""

    info = Agent.inspect_model(filepath)
    if not info:
        return

    print("\n" + "=" * 60)
    print(f"🔍 Model Inspection: {info['filename']}")
    print("=" * 60)
    print(f"   File Size: {info['file_size_mb']:.2f} MB")
    print(f"   Modified:  {info['file_modified']}")
    print(
        f"\n   Steps:     {info['steps']:,}"
        if isinstance(info["steps"], int)
        else f"\n   Steps:     {info['steps']}"
    )
    print(
        f"   Epsilon:   {info['epsilon']:.4f}"
        if isinstance(info["epsilon"], float)
        else f"   Epsilon:   {info['epsilon']}"
    )
    print(f"   State Size: {info['state_size']}")
    print(f"   Action Size: {info['action_size']}")

    if info["has_metadata"] and info["metadata"]:
        meta = info["metadata"]
        print("\n   📊 Training Metadata:")
        print("   ─────────────────────")
        print(f"   Save Reason:    {meta.get('save_reason', 'unknown')}")
        print(
            f"   Episode:        {meta.get('episode', 'unknown'):,}"
            if isinstance(meta.get("episode"), int)
            else f"   Episode:        {meta.get('episode', 'unknown')}"
        )
        print(f"   Best Score:     {meta.get('best_score', 'unknown')}")
        print(f"   Avg Score(100): {meta.get('avg_score_last_100', 0):.1f}")
        print(f"   Win Rate:       {meta.get('win_rate', 0)*100:.1f}%")
        print(f"   Avg Loss:       {meta.get('avg_loss', 0):.4f}")

        training_time = meta.get("total_training_time_seconds", 0)
        if training_time > 0:
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            print(f"   Training Time:  {hours}h {minutes}m")

        print("\n   ⚙️ Config Snapshot:")
        print("   ─────────────────────")
        print(f"   Learning Rate:  {meta.get('learning_rate', 'unknown')}")
        print(f"   Gamma:          {meta.get('gamma', 'unknown')}")
        print(f"   Batch Size:     {meta.get('batch_size', 'unknown')}")
        print(f"   Hidden Layers:  {meta.get('hidden_layers', 'unknown')}")
        print(f"   Dueling DQN:    {meta.get('use_dueling', 'unknown')}")
    else:
        print("\n   ⚠️ No detailed metadata (legacy save format)")

    print("=" * 60 + "\n")


def list_models(model_dir: str = "models") -> None:
    """List all model files in the models directory."""

    models = Agent.list_models(model_dir)

    if not models:
        print(f"\n❌ No model files found in '{model_dir}/'")
        return

    print("\n" + "=" * 80)
    print(f"📁 Saved Models in '{model_dir}/' ({len(models)} files)")
    print("=" * 80)
    print(f"{'Filename':<35} {'Episode':>8} {'Steps':>12} {'Best':>6} {'Epsilon':>8} {'Size':>8}")
    print("-" * 80)

    for model in models:
        filename = (
            model["filename"][:33] + ".." if len(model["filename"]) > 35 else model["filename"]
        )

        # Get metadata if available
        if model["has_metadata"] and model["metadata"]:
            meta = model["metadata"]
            episode = meta.get("episode", "?")
            steps = meta.get("total_steps", model.get("steps", "?"))
            best = meta.get("best_score", "?")
            epsilon = meta.get("epsilon", model.get("epsilon", "?"))
        else:
            episode = "?"
            steps = model.get("steps", "?")
            best = "?"
            epsilon = model.get("epsilon", "?")

        size_mb = f"{model['file_size_mb']:.1f}MB"

        # Format values
        ep_str = f"{episode:,}" if isinstance(episode, int) else str(episode)
        steps_str = f"{steps:,}" if isinstance(steps, int) else str(steps)
        best_str = str(best)
        eps_str = f"{epsilon:.3f}" if isinstance(epsilon, float) else str(epsilon)

        print(f"{filename:<35} {ep_str:>8} {steps_str:>12} {best_str:>6} {eps_str:>8} {size_mb:>8}")

    print("=" * 80)
    print("\nUse --inspect <path> to see detailed info about a specific model.\n")


def run_web_mode(config: Config, args: argparse.Namespace) -> None:
    """
    Run with web interface for game selection and monitoring.
    Works for both headless and visual modes.

    This function handles all --web scenarios:
    - If game specified: starts immediately with web dashboard
    - If no game: shows web launcher for game selection, then starts training
    """
    import socket
    import threading

    try:
        from src.web.server import WebDashboard
    except ImportError:
        print("❌ Web dashboard requires Flask. Install with:")
        print("   pip install flask flask-socketio eventlet")
        return

    # Find available port (auto-increment if busy)
    port = args.port
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
            break  # Port is available
        except OSError:
            if attempt == 0:
                print(f"⚠️  Port {port} is busy, finding available port...")
            port += 1
    else:
        print(f"❌ Could not find available port after {max_attempts} attempts")
        return

    if port != args.port:
        print(f"✓ Using port {port}")

    # Track game and mode selection
    selected_game = None
    selected_mode = "ai"  # 'ai' or 'human'
    selection_event = threading.Event()

    def on_game_selected(game_name: str, mode: str) -> None:
        """Called when user selects a game from web UI."""
        nonlocal selected_game, selected_mode
        selected_game = game_name
        selected_mode = mode
        selection_event.set()

    # Start web dashboard in launcher mode
    dashboard = WebDashboard(
        config, port=port, host=getattr(args, "host", "127.0.0.1"), launcher_mode=True
    )
    dashboard.on_game_selected_callback = on_game_selected  # Set callback BEFORE start
    dashboard.start()

    # Wait for server to print its startup message

    time.sleep(0.3)

    # If game already specified, skip selection
    if args.game:
        selected_game = args.game
        # Check if human mode was specified via CLI
        if hasattr(args, "human") and args.human:
            selected_mode = "human"
        print(f"\n🎮 Starting {selected_game}...")
    else:
        # Wait for game selection from web UI
        print("\n⏳ Open browser to select a game...")
        print("   Press Ctrl+C to exit\n")

        try:
            # Wait for game selection or keyboard interrupt
            while not selection_event.is_set():
                selection_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            print("\n\n👋 Closed by user")
            dashboard.stop()
            return

        if not selected_game:
            print("No game selected. Exiting.")
            dashboard.stop()
            return

        mode_text = "🎮 Playing" if selected_mode == "human" else "🤖 Training"
        print(f"\n{mode_text} {selected_game}...")

    # Update config and args based on selection
    config.GAME_NAME = selected_game
    args.game = selected_game

    # Set mode based on web selection
    if selected_mode == "human":
        args.human = True

    dashboard.launcher_mode = False
    dashboard.socketio.emit("game_ready", {"game": selected_game, "mode": selected_mode})

    # Game loop - supports returning to menu
    while True:
        return_to_menu = False

        # Start appropriate trainer based on mode
        try:
            if args.headless and selected_mode != "human":
                # Headless training (only for AI mode)
                trainer = HeadlessTrainer(config, args, existing_dashboard=dashboard)
                trainer.train()
            else:
                # Visual mode (required for human play)
                app = GameApp(config, args, existing_dashboard=dashboard)

                # Run appropriate mode
                if selected_mode == "human" or args.human:
                    app.run_human_mode()
                elif args.play:
                    app.run_play_mode()
                else:
                    app.run_training()

                # Check if user wants to return to game selector
                return_to_menu = app.return_to_menu

        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            if args.headless:
                if "trainer" in locals():
                    trainer._save_model(
                        f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted"
                    )
            else:
                if "app" in locals():
                    app._save_model(
                        f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted"
                    )
                    if app.web_dashboard:
                        app.web_dashboard.log("⛔ Training interrupted by user", "warning")
            break  # Exit on keyboard interrupt

        if not return_to_menu:
            break  # Normal exit

        # Return to game selector
        print("\n🏠 Returning to game selector...")

        # Reset selection state
        selected_game = None
        selected_mode = "ai"
        selection_event.clear()

        # Switch dashboard back to launcher mode
        dashboard.launcher_mode = True

        print("\n⏳ Open browser to select a game...")
        print("   Press Ctrl+C to exit\n")

        try:
            # Wait for new game selection
            while not selection_event.is_set():
                selection_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            print("\n\n👋 Closed by user")
            break

        if not selected_game:
            print("No game selected. Exiting.")
            break

        # Update config for new game
        mode_text = "🎮 Playing" if selected_mode == "human" else "🤖 Training"
        print(f"\n{mode_text} {selected_game}...")

        config.GAME_NAME = selected_game
        args.game = selected_game
        args.human = selected_mode == "human"

        dashboard.launcher_mode = False
        dashboard.socketio.emit("game_ready", {"game": selected_game, "mode": selected_mode})

    # Clean up resources
    if dashboard:
        dashboard.stop()
    if not args.headless:
        pygame.quit()
    print("\n👋 Done")


def run_web_launcher(config: Config, args: argparse.Namespace) -> None:
    """Compatibility wrapper for the canonical web-mode launcher flow."""
    run_web_mode(config, args)


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
    if hasattr(args, "web") and args.web and WEB_AVAILABLE:
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
