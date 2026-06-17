"""Launch a disposable dashboard server for Playwright smoke tests."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import Config
from src.web.server import WebDashboard


def build_config(tmp_root: str) -> Config:
    """Return a dashboard config that cannot touch real training artifacts."""
    config = Config()
    config.GAME_NAME = "breakout"
    config.MODEL_DIR = os.path.join(tmp_root, "models")
    config.LOG_DIR = os.path.join(tmp_root, "logs")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    with tempfile.TemporaryDirectory(prefix="nn-game-dashboard-e2e-") as tmp_root:
        dashboard = WebDashboard(
            config=build_config(tmp_root),
            port=args.port,
            host=args.host,
        )

        dashboard.publisher.set_system_info(
            device="cpu",
            torch_compiled=False,
            target_episodes=0,
            headless=True,
        )
        dashboard.on_save_callback = lambda: (
            dashboard.publisher.record_save(
                filename="e2e-smoke.pth",
                reason="manual",
                episode=dashboard.publisher.state.episode,
                best_score=dashboard.publisher.state.best_score,
            )
            or True
        )
        dashboard.on_pause_callback = lambda: (
            dashboard.publisher.set_paused(not dashboard.publisher.state.is_paused) or True
        )
        dashboard.on_reset_callback = lambda: dashboard.log("E2E reset requested", "action") or True
        dashboard.on_start_fresh_callback = (
            lambda: dashboard.publisher.reset_training_state() or True
        )
        dashboard.on_config_change_callback = (
            lambda config: dashboard.log(f"E2E config changed: {config}", "action") or True
        )
        dashboard.on_performance_mode_callback = (
            lambda mode: dashboard.log(f"E2E performance mode: {mode}", "action") or True
        )

        dashboard.start()
        dashboard.log("E2E dashboard ready", "success")
        print(f"DASHBOARD_URL={dashboard.dashboard_url()}", flush=True)

        stop_event.wait()
        dashboard.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
