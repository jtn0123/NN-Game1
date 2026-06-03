"""Cross-game contract and invariant tests."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from config import Config
from src.game import GAME_REGISTRY, get_game, get_game_info, list_games
from src.game.asteroids import VecAsteroids
from src.game.breakout import VecBreakout
from src.game.pong import VecPong
from src.game.snake import VecSnake
from src.game.space_invaders import VecSpaceInvaders


def _make_config(game_id: str) -> Config:
    config = Config()
    config.GAME_NAME = game_id
    return config


def _assert_valid_state(state: np.ndarray, expected_size: int) -> None:
    assert isinstance(state, np.ndarray)
    assert state.shape == (expected_size,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()


@pytest.mark.parametrize("game_id", list_games())
def test_registered_games_reset_and_step_valid_actions(game_id):
    """Every registered game should honor the BaseGame reset/step contract."""
    config = _make_config(game_id)
    game_cls = get_game(game_id)
    assert game_cls is not None
    game = game_cls(config, headless=True)

    initial_state = game.reset()
    _assert_valid_state(initial_state, game.state_size)

    info = get_game_info(game_id)
    assert info is not None
    assert len(info["actions"]) == game.action_size

    for action in range(game.action_size):
        game.reset()
        next_state, reward, done, step_info = game.step(action)

        _assert_valid_state(next_state, game.state_size)
        assert isinstance(float(reward), float)
        assert np.isfinite(reward)
        assert isinstance(done, bool)
        assert isinstance(step_info, dict)


def test_game_registry_metadata_is_complete_and_unique():
    """Registry metadata drives the CLI and dashboard game pickers."""
    game_ids = list_games()

    assert game_ids == list(GAME_REGISTRY.keys())
    assert len(game_ids) == len(set(game_ids))
    for game_id in game_ids:
        info = get_game_info(game_id)
        assert info is not None
        assert info["name"]
        assert info["description"]
        assert info["actions"]
        assert info["difficulty"]
        assert len(info["color"]) == 3


@pytest.mark.parametrize(
    ("game_id", "vec_cls"),
    [
        ("breakout", VecBreakout),
        ("space_invaders", VecSpaceInvaders),
        ("pong", VecPong),
        ("snake", VecSnake),
        ("asteroids", VecAsteroids),
    ],
)
def test_vectorized_game_envs_preserve_batch_contract(game_id, vec_cls):
    """Vectorized envs should return finite batched states/rewards/dones."""
    config = _make_config(game_id)
    num_envs = 2
    env = vec_cls(num_envs=num_envs, config=config, headless=True)

    states = env.reset()
    assert states.shape == (num_envs, env.state_size)
    assert states.dtype == np.float32
    assert np.isfinite(states).all()

    actions = np.zeros(num_envs, dtype=np.int64)
    next_states, rewards, dones, infos = env.step(actions)

    assert next_states.shape == (num_envs, env.state_size)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    assert isinstance(infos, list)
    assert len(infos) == num_envs
    assert np.isfinite(next_states).all()
    assert np.isfinite(rewards).all()
    assert dones.dtype == np.bool_
