from __future__ import annotations

import argparse
import sys

from src.app.process_control import restart_with_game


def test_restart_with_game_preserves_runtime_flags(monkeypatch, capsys):
    executed = []

    monkeypatch.setattr("src.app.process_control.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("src.app.process_control.sys.argv", ["main.py", "--game", "breakout"])
    monkeypatch.setattr(
        "src.app.process_control.os.execv",
        lambda executable, args: executed.append((executable, args)),
    )

    restart_with_game(
        "snake",
        argparse.Namespace(
            headless=True,
            web=True,
            port=5123,
            turbo=True,
            vec_envs=4,
            episodes=25,
            cpu=True,
        ),
    )

    assert executed == [
        (
            sys.executable,
            [
                sys.executable,
                "main.py",
                "--game",
                "snake",
                "--headless",
                "--web",
                "--port",
                "5123",
                "--turbo",
                "--vec-envs",
                "4",
                "--episodes",
                "25",
                "--cpu",
            ],
        )
    ]
    assert "Restarting with snake" in capsys.readouterr().out
