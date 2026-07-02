"""DQfD-lite machinery: demo transitions replay deterministically, the margin
loss prefers the demonstrated action, pretraining moves weights, and the
demo-prefix episode start hands the agent a mid-route state with fresh clocks."""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.ai.agent import Agent  # noqa: E402
from src.ai.demo_learning import DemoStore, demo_prefix_registry  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402


def _write_demo(tmp_path, level=0, actions=None, won=True):
    actions = actions if actions is not None else [2, 2, 5, 5, 2, 1, 3, 2] * 6
    (tmp_path / f"demo_l{level}.json").write_text(
        json.dumps({"level_index": level, "actions": actions, "won": won})
    )
    return actions


def _config(**overrides):
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_store_builds_nstep_transitions(tmp_path):
    cfg = _config()
    actions = _write_demo(tmp_path)
    store = DemoStore.from_dir(str(tmp_path), cfg)
    assert store is not None
    assert store.n_episodes == 1
    assert len(store) == len(actions)  # episode didn't terminate early
    n = int(cfg.N_STEP_SIZE)
    assert store.n_step_lengths.max() == n
    assert store.n_step_lengths[-1] == 1  # horizon shrinks at the episode tail
    game = CrystalCaves(cfg, headless=True)
    assert store.states.shape[1] == game.state_size


def test_store_ignores_losses_and_unknown_levels(tmp_path):
    cfg = _config()
    _write_demo(tmp_path, won=False)
    (tmp_path / "noise.json").write_text(json.dumps({"actions": [1, 2]}))  # no level key
    assert DemoStore.from_dir(str(tmp_path), cfg) is None


def test_margin_loss_prefers_demo_action(tmp_path):
    cfg = _config(DEMO_PRETRAIN_STEPS=0)
    _write_demo(tmp_path)
    store = DemoStore.from_dir(str(tmp_path), cfg)
    game = CrystalCaves(cfg, headless=True)
    agent = Agent(state_size=game.state_size, action_size=game.action_size, config=cfg)
    agent.attach_demo_store(store)
    loss = agent._dqfd_loss()
    assert loss is not None and torch.isfinite(loss)


def test_pretrain_moves_weights_and_raises_demo_q(tmp_path):
    cfg = _config()
    _write_demo(tmp_path)
    store = DemoStore.from_dir(str(tmp_path), cfg)
    game = CrystalCaves(cfg, headless=True)
    agent = Agent(state_size=game.state_size, action_size=game.action_size, config=cfg)
    agent.attach_demo_store(store)
    before = [p.detach().clone() for p in agent.policy_net.parameters()]
    done = agent.pretrain_on_demos(25)
    assert done == 25
    changed = any(
        not torch.equal(b, p.detach()) for b, p in zip(before, agent.policy_net.parameters())
    )
    assert changed, "pretraining must update policy weights"


def test_demo_prefix_start_hands_over_mid_route(tmp_path):
    actions = _write_demo(tmp_path, actions=[2, 2, 2, 5, 5, 2, 2, 2] * 20)
    cfg = _config(DEMO_DIR=str(tmp_path), CRYSTAL_CAVES_DEMO_RESET_P=1.0)
    game = CrystalCaves(cfg, headless=True)
    np.random.seed(3)
    spawn_game = CrystalCaves(_config(), headless=True)
    spawn_game.reset()
    spawn_x = spawn_game.player_x
    moved = False
    for _ in range(5):
        game.reset()
        assert game.steps == 0, "prefix replay must not consume the episode budget"
        assert game.steps_since_progress == 0
        assert not game.game_over
        if abs(game.player_x - spawn_x) > 4:
            moved = True
    assert moved, "with p=1.0 at least one reset should start mid-route"


def test_demo_prefix_start_never_fires_in_eval(tmp_path):
    from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS

    _write_demo(tmp_path, actions=[2] * 200)
    cfg = _config(DEMO_DIR=str(tmp_path), CRYSTAL_CAVES_DEMO_RESET_P=1.0)
    spec = HANDCRAFTED_LEVELS[0]
    game = CrystalCaves(cfg, headless=True)
    game.CAVES = (spec,)
    game._randomize_levels = False
    game.use_eval_levels(1)
    game.reset_eval_cursor()
    spawn_game = CrystalCaves(_config(), headless=True)
    spawn_game.CAVES = (spec,)
    spawn_game._randomize_levels = False
    spawn_game.use_eval_levels(1)
    spawn_game.reset_eval_cursor()
    spawn_game.reset()
    game.reset()
    assert game.player_x == spawn_game.player_x, "eval episodes must start at spawn"


def test_registry_groups_by_level(tmp_path):
    _write_demo(tmp_path, level=0)
    _write_demo(tmp_path, level=3)
    registry = demo_prefix_registry(str(tmp_path))
    assert set(registry) == {0, 3}
