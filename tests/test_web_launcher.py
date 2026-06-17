"""Tests for web launcher orchestration helpers."""

from __future__ import annotations

import argparse
import socket
from types import SimpleNamespace

from config import Config
from src.app import web_launcher


def test_available_port_finds_open_port_after_busy_one():
    busy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        busy_socket.bind(("localhost", 0))
        busy_port = busy_socket.getsockname()[1]

        assert web_launcher._available_port(busy_port, max_attempts=1) is None
        assert web_launcher._available_port(busy_port, max_attempts=2) == busy_port + 1
    finally:
        busy_socket.close()


def test_wait_for_selection_stops_dashboard_on_keyboard_interrupt(capsys):
    class InterruptingEvent:
        def is_set(self):
            return False

        def wait(self, timeout):
            raise KeyboardInterrupt

    dashboard = SimpleNamespace(stopped=False)
    dashboard.stop = lambda: setattr(dashboard, "stopped", True)

    assert web_launcher._wait_for_selection(InterruptingEvent(), dashboard) is False
    assert dashboard.stopped is True
    assert "Closed by user" in capsys.readouterr().out


def test_wait_for_selection_returns_true_when_event_is_set():
    event = SimpleNamespace(is_set=lambda: True, wait=lambda timeout: None)

    assert web_launcher._wait_for_selection(event, SimpleNamespace(stop=lambda: None)) is True


def test_print_selected_mode_outputs_human_and_ai_modes(capsys):
    web_launcher._print_selected_mode("breakout", "human")
    web_launcher._print_selected_mode("space_invaders", "ai")

    output = capsys.readouterr().out
    assert "Playing breakout" in output
    assert "Training space_invaders" in output


def test_save_interrupted_run_handles_headless_and_visual_modes():
    config = Config()
    config.GAME_NAME = "breakout"
    saved = []

    trainer = SimpleNamespace(
        _save_model=lambda filename, save_reason: saved.append((filename, save_reason))
    )
    web_launcher._save_interrupted_run(
        config,
        argparse.Namespace(headless=True),
        {"trainer": trainer},
    )

    dashboard = SimpleNamespace(log=lambda message, level: saved.append((message, level)))
    app = SimpleNamespace(
        web_dashboard=dashboard,
        _save_model=lambda filename, save_reason: saved.append((filename, save_reason)),
    )
    web_launcher._save_interrupted_run(
        config,
        argparse.Namespace(headless=False),
        {"app": app},
    )

    assert ("breakout_interrupted.pth", "interrupted") in saved
    assert ("Training interrupted by user", "warning") in [
        (message.replace("⛔ ", ""), level) for message, level in saved if isinstance(message, str)
    ]


def test_run_web_mode_starts_headless_training_for_preselected_game(monkeypatch):
    from src.web import server

    events = []

    class FakeSocket:
        def emit(self, event, payload):
            events.append((event, payload))

    class FakeDashboard:
        instances = []

        def __init__(self, config, port, host, launcher_mode):
            self.config = config
            self.port = port
            self.host = host
            self.launcher_mode = launcher_mode
            self.on_game_selected_callback = None
            self.socketio = FakeSocket()
            self.started = False
            self.stopped = False
            FakeDashboard.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    class FakeTrainer:
        instances = []

        def __init__(self, config, args, existing_dashboard):
            self.config = config
            self.args = args
            self.existing_dashboard = existing_dashboard
            self.trained = False
            FakeTrainer.instances.append(self)

        def train(self):
            self.trained = True

    config = Config()
    args = argparse.Namespace(
        port=5000,
        host="127.0.0.1",
        game="space_invaders",
        human=False,
        headless=True,
        play=False,
    )
    monkeypatch.setattr(web_launcher, "_available_port", lambda _port: 5050)
    monkeypatch.setattr(web_launcher.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(web_launcher, "HeadlessTrainer", FakeTrainer)
    monkeypatch.setattr(server, "WebDashboard", FakeDashboard)

    web_launcher.run_web_mode(config, args)

    dashboard = FakeDashboard.instances[0]
    trainer = FakeTrainer.instances[0]
    assert dashboard.started is True
    assert dashboard.stopped is True
    assert dashboard.launcher_mode is False
    assert trainer.trained is True
    assert trainer.existing_dashboard is dashboard
    assert config.GAME_NAME == "space_invaders"
    assert args.game == "space_invaders"
    assert events == [("game_ready", {"game": "space_invaders", "mode": "ai"})]


def test_run_web_mode_exits_when_no_port_available(monkeypatch, capsys):
    config = Config()
    args = argparse.Namespace(port=5000, host="127.0.0.1", game="breakout")
    monkeypatch.setattr(web_launcher, "_available_port", lambda _port: None)

    web_launcher.run_web_mode(config, args)

    assert "Could not find available port" in capsys.readouterr().out


def test_run_web_mode_stops_dashboard_when_selection_has_no_game(monkeypatch, capsys):
    from src.web import server

    class FakeDashboard:
        instances = []

        def __init__(self, config, port, host, launcher_mode):
            self.config = config
            self.port = port
            self.host = host
            self.launcher_mode = launcher_mode
            self.on_game_selected_callback = None
            self.started = False
            self.stopped = False
            self.socketio = SimpleNamespace(emit=lambda _event, _payload: None)
            FakeDashboard.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    args = argparse.Namespace(
        port=5000,
        host="127.0.0.1",
        game=None,
        human=False,
        headless=True,
        play=False,
    )
    monkeypatch.setattr(web_launcher, "_available_port", lambda port: port)
    monkeypatch.setattr(web_launcher.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(web_launcher, "_wait_for_selection", lambda _event, _dashboard: True)
    monkeypatch.setattr(server, "WebDashboard", FakeDashboard)

    web_launcher.run_web_mode(Config(), args)

    assert FakeDashboard.instances[0].started is True
    assert FakeDashboard.instances[0].stopped is True
    assert "No game selected" in capsys.readouterr().out


def test_run_web_mode_runs_visual_human_mode_and_quits_pygame(monkeypatch):
    from src.web import server

    events = []
    quit_called = []

    class FakeDashboard:
        instances = []

        def __init__(self, config, port, host, launcher_mode):
            self.config = config
            self.port = port
            self.host = host
            self.launcher_mode = launcher_mode
            self.on_game_selected_callback = None
            self.socketio = SimpleNamespace(
                emit=lambda event, payload: events.append((event, payload))
            )
            self.stopped = False
            FakeDashboard.instances.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class FakeApp:
        instances = []

        def __init__(self, config, args, existing_dashboard):
            self.config = config
            self.args = args
            self.existing_dashboard = existing_dashboard
            self.return_to_menu = False
            self.mode = None
            FakeApp.instances.append(self)

        def run_human_mode(self):
            self.mode = "human"

        def run_play_mode(self):
            self.mode = "play"

        def run_training(self):
            self.mode = "training"

    args = argparse.Namespace(
        port=5000,
        host="127.0.0.1",
        game="breakout",
        human=True,
        headless=False,
        play=False,
    )
    monkeypatch.setattr(web_launcher, "_available_port", lambda port: port)
    monkeypatch.setattr(web_launcher.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(server, "WebDashboard", FakeDashboard)
    monkeypatch.setattr(web_launcher, "GameApp", FakeApp)
    monkeypatch.setattr(web_launcher.pygame, "quit", lambda: quit_called.append(True))

    web_launcher.run_web_mode(Config(), args)

    assert FakeApp.instances[0].mode == "human"
    assert FakeDashboard.instances[0].stopped is True
    assert quit_called == [True]
    assert events == [("game_ready", {"game": "breakout", "mode": "human"})]


def test_run_web_launcher_delegates_to_web_mode(monkeypatch):
    called = []
    config = Config()
    args = argparse.Namespace()
    monkeypatch.setattr(web_launcher, "run_web_mode", lambda cfg, ns: called.append((cfg, ns)))

    web_launcher.run_web_launcher(config, args)

    assert called == [(config, args)]
