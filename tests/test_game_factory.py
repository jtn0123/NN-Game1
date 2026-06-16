"""Tests for runtime game construction helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.app import game_factory


class FakeGame:
    state_size = 3
    action_size = 2

    def __init__(self, config, headless=False):
        self.config = config
        self.headless = headless

    def reset(self):
        return np.zeros(self.state_size)

    def step(self, action):
        return np.zeros(self.state_size), 0.0, False, {"score": 0}

    def render(self, screen):
        return None

    def get_state(self):
        return np.zeros(self.state_size)


class FakeVecGame:
    def __init__(self, num_envs, config, headless=True):
        self.envs = [FakeGame(config, headless=headless) for _ in range(num_envs)]


def test_create_training_environment_uses_registered_vector_class(monkeypatch):
    """Vectorized training should come from registry metadata."""
    monkeypatch.setattr(game_factory, "get_game", lambda _name: FakeGame)
    monkeypatch.setattr(game_factory, "get_vec_game", lambda _name: FakeVecGame)

    environment = game_factory.create_training_environment(
        "fake",
        SimpleNamespace(),
        num_envs=4,
        headless=True,
    )

    assert isinstance(environment.vec_env, FakeVecGame)
    assert len(environment.vec_env.envs) == 4
    assert environment.game is environment.vec_env.envs[0]
    assert environment.num_envs == 4


def test_create_training_environment_falls_back_to_single_game(monkeypatch):
    """Unsupported vector mode should still return a single registered game."""
    monkeypatch.setattr(game_factory, "get_game", lambda _name: FakeGame)
    monkeypatch.setattr(game_factory, "get_vec_game", lambda _name: None)

    environment = game_factory.create_training_environment(
        "fake",
        SimpleNamespace(),
        num_envs=4,
        headless=True,
    )

    assert environment.vec_env is None
    assert isinstance(environment.game, FakeGame)
    assert environment.num_envs == 1


def test_resolve_game_class_rejects_unknown_game(monkeypatch):
    """Unknown game errors should include the available registry names."""
    monkeypatch.setattr(game_factory, "get_game", lambda _name: None)
    monkeypatch.setattr(game_factory, "list_games", lambda: ["breakout"])

    with pytest.raises(ValueError, match="Available games: breakout"):
        game_factory.resolve_game_class("missing")
