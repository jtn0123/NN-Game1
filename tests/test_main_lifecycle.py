"""Tests for application lifecycle helpers in main.py."""

from types import SimpleNamespace

import main


class FakeDashboard:
    def __init__(self):
        self.logs = []

    def log(self, message, level="info", data=None):
        self.logs.append((message, level, data))


def test_game_app_save_and_quit_requests_loop_shutdown(monkeypatch):
    """Visual save-and-quit should stop the loop without hard-exiting the process."""
    app = main.GameApp.__new__(main.GameApp)
    app.config = SimpleNamespace(GAME_NAME="breakout")
    app.web_dashboard = FakeDashboard()
    app.running = True
    app._save_model = lambda *args, **kwargs: True
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    app._save_and_quit()

    assert app.running is False
    assert any(
        "Shutting down" in message for message, _level, _data in app.web_dashboard.logs
    )


def test_headless_save_and_quit_requests_loop_shutdown(monkeypatch):
    """Headless save-and-quit should stop training loops without os._exit()."""
    trainer = main.HeadlessTrainer.__new__(main.HeadlessTrainer)
    trainer.config = SimpleNamespace(GAME_NAME="breakout")
    trainer.web_dashboard = FakeDashboard()
    trainer.running = True
    trainer._save_model = lambda *args, **kwargs: True
    monkeypatch.setattr(main.time, "sleep", lambda _seconds: None)

    trainer._save_and_quit()

    assert trainer.running is False
    assert any(
        "Shutting down" in message
        for message, _level, _data in trainer.web_dashboard.logs
    )
