from __future__ import annotations

from types import SimpleNamespace

from src.app.dashboard_bindings import DashboardCallbacks, bind_dashboard_callbacks


def test_bind_dashboard_callbacks_assigns_runtime_hooks():
    calls = []
    dashboard = SimpleNamespace()

    bind_dashboard_callbacks(
        dashboard,
        DashboardCallbacks(
            pause=lambda: calls.append("pause"),
            save=lambda: calls.append("save"),
            save_as=lambda filename: calls.append(("save_as", filename)),
            speed=lambda speed: calls.append(("speed", speed)),
            reset=lambda: calls.append("reset"),
            start_fresh=lambda: calls.append("fresh"),
            load_model=lambda path: calls.append(("load", path)),
            config_change=lambda config: calls.append(("config", config)),
            performance_mode=lambda mode: calls.append(("mode", mode)),
            restart_with_game=lambda game: calls.append(("restart", game)),
            save_and_quit=lambda: calls.append("quit"),
        ),
    )

    dashboard.on_pause_callback()
    dashboard.on_save_as_callback("demo.pth")
    dashboard.on_speed_callback(2.0)
    dashboard.on_restart_with_game_callback("snake")

    assert calls == ["pause", ("save_as", "demo.pth"), ("speed", 2.0), ("restart", "snake")]
