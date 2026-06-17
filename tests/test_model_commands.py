"""Tests for CLI model listing and inspection helpers."""

from __future__ import annotations

from src.app import model_commands


def test_inspect_model_prints_detailed_metadata(monkeypatch, capsys):
    def inspect_model(_filepath):
        return {
            "filename": "demo.pth",
            "file_size_mb": 1.25,
            "file_modified": "2026-06-16",
            "steps": 1234,
            "epsilon": 0.1234,
            "state_size": 8,
            "action_size": 3,
            "has_metadata": True,
            "metadata": {
                "save_reason": "best_score",
                "episode": 42,
                "best_score": 99,
                "avg_score_last_100": 12.3,
                "win_rate": 0.5,
                "avg_loss": 0.0123,
                "total_training_time_seconds": 3660,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "batch_size": 64,
                "hidden_layers": [32, 16],
                "use_dueling": True,
            },
        }

    monkeypatch.setattr(model_commands.Agent, "inspect_model", staticmethod(inspect_model))

    model_commands.inspect_model("models/demo.pth")

    output = capsys.readouterr().out
    assert "Model Inspection: demo.pth" in output
    assert "Steps:     1,234" in output
    assert "Epsilon:   0.1234" in output
    assert "Save Reason:    best_score" in output
    assert "Training Time:  1h 1m" in output
    assert "Dueling DQN:    True" in output


def test_inspect_model_handles_legacy_and_missing_metadata(monkeypatch, capsys):
    responses = [
        None,
        {
            "filename": "legacy.pth",
            "file_size_mb": 0.5,
            "file_modified": "yesterday",
            "steps": "unknown",
            "epsilon": "unknown",
            "state_size": 4,
            "action_size": 2,
            "has_metadata": False,
            "metadata": {},
        },
    ]

    monkeypatch.setattr(
        model_commands.Agent,
        "inspect_model",
        staticmethod(lambda _filepath: responses.pop(0)),
    )

    model_commands.inspect_model("missing.pth")
    assert capsys.readouterr().out == ""

    model_commands.inspect_model("legacy.pth")

    output = capsys.readouterr().out
    assert "Model Inspection: legacy.pth" in output
    assert "Steps:     unknown" in output
    assert "No detailed metadata" in output


def test_list_models_prints_empty_and_metadata_rows(monkeypatch, capsys):
    monkeypatch.setattr(
        model_commands.Agent,
        "list_models",
        staticmethod(lambda _model_dir: []),
    )

    model_commands.list_models("empty")

    assert "No model files found in 'empty/'" in capsys.readouterr().out

    long_filename = "breakout_" + ("very_long_name_" * 3) + ".pth"
    monkeypatch.setattr(
        model_commands.Agent,
        "list_models",
        staticmethod(
            lambda _model_dir: [
                {
                    "filename": long_filename,
                    "file_size_mb": 3.2,
                    "steps": 500,
                    "epsilon": 0.2,
                    "has_metadata": True,
                    "metadata": {
                        "episode": 1234,
                        "total_steps": 98765,
                        "best_score": 321,
                        "epsilon": 0.12345,
                    },
                },
                {
                    "filename": "legacy.pth",
                    "file_size_mb": 0.7,
                    "steps": "unknown",
                    "epsilon": "unknown",
                    "has_metadata": False,
                    "metadata": {},
                },
            ]
        ),
    )

    model_commands.list_models("models")

    output = capsys.readouterr().out
    assert "Saved Models in 'models/' (2 files)" in output
    assert "breakout_very_long_name" in output
    assert ".." in output
    assert "1,234" in output
    assert "98,765" in output
    assert "0.123" in output
    assert "legacy.pth" in output
    assert "unknown" in output
