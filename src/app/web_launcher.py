"""Web launcher orchestration for game selection and dashboard-backed runs."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from typing import Any

import pygame

from config import Config
from src.app.crystal_curriculum import run_crystal_curriculum
from src.app.headless import HeadlessTrainer
from src.app.interactive import GameApp


def run_web_mode(config: Config, args: argparse.Namespace) -> None:
    """
    Run with the web interface for game selection and monitoring.

    Supports both headless and visual modes:
    - If a game is specified, training starts immediately with the dashboard.
    - If no game is specified, the browser launcher chooses game and mode.
    """
    try:
        from src.web.server import WebDashboard
    except ImportError:
        print("❌ Web dashboard requires Flask. Install with:")
        print("   pip install flask flask-socketio eventlet")
        return

    port = _available_port(args.port)
    if port is None:
        print("❌ Could not find available port after 10 attempts")
        return
    if port != args.port:
        print(f"✓ Using port {port}")

    selected_game = None
    selected_mode = "ai"
    selection_event = threading.Event()

    def on_game_selected(game_name: str, mode: str) -> None:
        nonlocal selected_game, selected_mode
        selected_game = game_name
        selected_mode = mode
        selection_event.set()

    dashboard = WebDashboard(
        config,
        port=port,
        host=getattr(args, "host", "127.0.0.1"),
        launcher_mode=True,
    )
    dashboard.on_game_selected_callback = on_game_selected
    dashboard.start()
    time.sleep(0.3)

    if args.game:
        selected_game = args.game
        if getattr(args, "human", False):
            selected_mode = "human"
        print(f"\n🎮 Starting {selected_game}...")
    else:
        print("\n⏳ Open browser to select a game...")
        print("   Press Ctrl+C to exit\n")
        if not _wait_for_selection(selection_event, dashboard):
            return
        if not selected_game:
            print("No game selected. Exiting.")
            dashboard.stop()
            return
        _print_selected_mode(selected_game, selected_mode)

    config.GAME_NAME = selected_game
    args.game = selected_game
    if selected_mode == "human":
        args.human = True

    dashboard.launcher_mode = False
    dashboard.socketio.emit("game_ready", {"game": selected_game, "mode": selected_mode})

    while True:
        return_to_menu = False
        try:
            if getattr(args, "crystal_curriculum", False) and selected_game == "crystal_caves":
                args.headless = True
                run_crystal_curriculum(config, args, existing_dashboard=dashboard)
            elif args.headless and selected_mode != "human":
                trainer = HeadlessTrainer(config, args, existing_dashboard=dashboard)
                trainer.train()
            else:
                app = GameApp(config, args, existing_dashboard=dashboard)
                if selected_mode == "human" or args.human:
                    app.run_human_mode()
                elif args.play:
                    app.run_play_mode()
                else:
                    app.run_training()
                return_to_menu = app.return_to_menu
        except KeyboardInterrupt:
            print("\n\n⛔ Training interrupted by user")
            _save_interrupted_run(config, args, locals())
            break

        if not return_to_menu:
            break

        print("\n🏠 Returning to game selector...")
        selected_game = None
        selected_mode = "ai"
        selection_event.clear()
        dashboard.launcher_mode = True

        print("\n⏳ Open browser to select a game...")
        print("   Press Ctrl+C to exit\n")

        try:
            while not selection_event.is_set():
                selection_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            print("\n\n👋 Closed by user")
            break

        if not selected_game:
            print("No game selected. Exiting.")
            break

        _print_selected_mode(selected_game, selected_mode)
        config.GAME_NAME = selected_game
        args.game = selected_game
        args.human = selected_mode == "human"
        dashboard.launcher_mode = False
        dashboard.socketio.emit("game_ready", {"game": selected_game, "mode": selected_mode})

    dashboard.stop()
    if not args.headless:
        pygame.quit()
    print("\n👋 Done")


def run_web_launcher(config: Config, args: argparse.Namespace) -> None:
    """Compatibility wrapper for the canonical web-mode launcher flow."""
    run_web_mode(config, args)


def _available_port(start_port: int, max_attempts: int = 10) -> int | None:
    port = start_port
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_ref:
                socket_ref.bind(("localhost", port))
            return port
        except OSError:
            if attempt == 0:
                print(f"⚠️  Port {port} is busy, finding available port...")
            port += 1
    return None


def _wait_for_selection(selection_event: threading.Event, dashboard: Any) -> bool:
    try:
        while not selection_event.is_set():
            selection_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        print("\n\n👋 Closed by user")
        dashboard.stop()
        return False
    return True


def _print_selected_mode(selected_game: str, selected_mode: str) -> None:
    mode_text = "🎮 Playing" if selected_mode == "human" else "🤖 Training"
    print(f"\n{mode_text} {selected_game}...")


def _save_interrupted_run(config: Config, args: argparse.Namespace, local_values: dict) -> None:
    if args.headless:
        trainer = local_values.get("trainer")
        if trainer is not None:
            trainer._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
        return

    app = local_values.get("app")
    if app is not None:
        app._save_model(f"{config.GAME_NAME}_interrupted.pth", save_reason="interrupted")
        if app.web_dashboard:
            app.web_dashboard.log("⛔ Training interrupted by user", "warning")
