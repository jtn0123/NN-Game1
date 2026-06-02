"""Tests for command-line parser behavior."""

import pytest

from src.app.cli import create_parser


def test_cli_defaults_to_local_dashboard_host():
    args = create_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 5000
    assert args.web is False
    assert args.vec_envs == 1
    assert args.game is None


def test_cli_parses_headless_web_training_options():
    args = create_parser().parse_args(
        [
            "--headless",
            "--web",
            "--game",
            "pong",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--episodes",
            "12",
            "--turbo",
            "--vec-envs",
            "4",
            "--seed",
            "99",
        ]
    )

    assert args.headless is True
    assert args.web is True
    assert args.game == "pong"
    assert args.host == "0.0.0.0"
    assert args.port == 8765
    assert args.episodes == 12
    assert args.turbo is True
    assert args.vec_envs == 4
    assert args.seed == 99


def test_cli_enforces_mutually_exclusive_modes():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--headless", "--human"])


def test_cli_rejects_unknown_game():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--game", "not-a-game"])


def test_cli_parses_model_inspection_and_list_modes():
    inspect_args = create_parser().parse_args(["--inspect", "models/demo.pth"])
    list_args = create_parser().parse_args(["--list-models"])

    assert inspect_args.inspect == "models/demo.pth"
    assert list_args.list_models is True
