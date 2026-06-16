"""Registry-level contracts for playable games."""

import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pygame")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from config import Config
from src.game import BaseGame, get_game, get_game_info, get_vec_game, list_games

pytestmark = [pytest.mark.pygame, pytest.mark.torch]


@pytest.mark.parametrize("game_id", list_games())
def test_registered_games_satisfy_base_contract(game_id):
    """Every registered game should reset and step through the BaseGame contract."""
    config = Config()
    config.GAME_NAME = game_id
    game_class = get_game(game_id)
    assert game_class is not None

    game = game_class(config, headless=True)  # type: ignore[call-arg]
    try:
        assert isinstance(game, BaseGame)
        initial_state = game.reset()
        assert isinstance(initial_state, np.ndarray)
        assert initial_state.shape == (game.state_size,)
        assert game.action_size > 0

        next_state, reward, done, info = game.step(0)
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (game.state_size,)
        assert isinstance(reward, (int, float, np.integer, np.floating))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "score" in info
        assert "won" in info
    finally:
        game.close()


@pytest.mark.parametrize("game_id", list_games())
def test_registered_vector_games_satisfy_batch_contract(game_id):
    """Every registered vector game should expose the shared batched contract."""
    config = Config()
    config.GAME_NAME = game_id
    vec_game_class = get_vec_game(game_id)
    assert vec_game_class is not None

    vec_env = vec_game_class(2, config, headless=True)  # type: ignore[call-arg]
    try:
        states = vec_env.reset()
        assert isinstance(states, np.ndarray)
        assert states.shape[0] == 2
        assert len(vec_env.envs) == 2

        actions = np.zeros(2, dtype=np.int64)
        next_states, rewards, dones, infos = vec_env.step_no_copy(actions)
        assert isinstance(next_states, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert len(infos) == 2
        assert next_states.shape[0] == 2
    finally:
        vec_env.close()


@pytest.mark.parametrize("game_id", list_games())
def test_registered_games_define_human_control_help(game_id):
    """Human mode should be able to show controls from registry metadata."""
    info = get_game_info(game_id)

    assert info is not None
    assert isinstance(info["controls"], list)
    assert info["controls"]
    assert all(isinstance(control, str) and control for control in info["controls"])
