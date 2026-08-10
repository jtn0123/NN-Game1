"""Tests for full training-state snapshots (pause/resume without progress loss)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.ai.agent import Agent  # noqa: E402
from src.ai.snapshot import (  # noqa: E402
    latest_snapshot,
    load_training_snapshot,
    rotate_snapshots,
    save_training_snapshot,
)


def _agent(**overrides) -> Agent:
    config = Config()
    config.FORCE_CPU = True
    config.MEMORY_SIZE = 512
    for key, value in overrides.items():
        setattr(config, key, value)
    return Agent(state_size=8, action_size=3, config=config)


def _fill(agent: Agent, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = rng.random(8, dtype=np.float32)
        ns = rng.random(8, dtype=np.float32)
        agent.remember(s, int(rng.integers(0, 3)), float(rng.random()), ns, False)


def test_roundtrip_restores_everything(tmp_path):
    agent = _agent()
    _fill(agent, 200)
    agent.epsilon = 0.123
    agent._demo_margin_scale = 0.456
    path = save_training_snapshot(
        agent, tmp_path, episode=777, ladder_offsets={12: 1400}, ladder_wins={12: 1}
    )
    assert path.exists()

    fresh = _agent()
    _fill(fresh, 37, seed=99)  # dirty state that must be overwritten
    fresh.epsilon = 0.9
    meta = load_training_snapshot(fresh, path)

    assert meta["episode"] == 777
    assert meta["ladder_offsets"] == {12: 1400}
    assert fresh.epsilon == pytest.approx(0.123)
    assert fresh._demo_margin_scale == pytest.approx(0.456)
    assert fresh.memory._size == agent.memory._size
    assert fresh.memory._position == agent.memory._position
    np.testing.assert_array_equal(fresh.memory.states, agent.memory.states)
    np.testing.assert_array_equal(fresh.memory.actions, agent.memory.actions)
    for k, v in agent.policy_net.state_dict().items():
        assert (fresh.policy_net.state_dict()[k] == v).all()
    # optimizer moments survive (the thing weights-only checkpoints lose)
    assert len(fresh.optimizer.state_dict()["state"]) == len(agent.optimizer.state_dict()["state"])


def test_per_priorities_roundtrip(tmp_path):
    agent = _agent(USE_PRIORITIZED_REPLAY=True)
    if not hasattr(agent.memory, "priorities"):
        pytest.skip("config did not build a PER buffer")
    _fill(agent, 150)
    agent.memory.priorities[:150] = np.linspace(0.1, 2.0, 150, dtype=np.float32)
    agent.memory.max_priority = 2.0
    path = save_training_snapshot(agent, tmp_path, episode=5)
    fresh = _agent(USE_PRIORITIZED_REPLAY=True)
    load_training_snapshot(fresh, path)
    np.testing.assert_array_equal(fresh.memory.priorities, agent.memory.priorities)
    assert fresh.memory.max_priority == pytest.approx(2.0)


def test_capacity_mismatch_rejected(tmp_path):
    agent = _agent()
    _fill(agent, 10)
    path = save_training_snapshot(agent, tmp_path, episode=1)
    other = _agent(MEMORY_SIZE=256)
    with pytest.raises(ValueError):
        load_training_snapshot(other, path)


def test_rotation_keeps_newest_two(tmp_path):
    agent = _agent()
    _fill(agent, 20)
    for ep in (100, 200, 300):
        save_training_snapshot(agent, tmp_path, episode=ep, keep=2)
    names = sorted(p.name for p in tmp_path.glob("snapshot_ep*.pt"))
    assert names == ["snapshot_ep200.pt", "snapshot_ep300.pt"]
    assert latest_snapshot(tmp_path).name == "snapshot_ep300.pt"


def test_rotation_removes_stray_tmp(tmp_path):
    (tmp_path / "snapshot_ep50.pt.tmp").write_bytes(b"junk")
    rotate_snapshots(tmp_path, keep=2)
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_config_knobs():
    config = Config()
    assert config.SNAPSHOT_EVERY_EPISODES == 0
    assert config.SNAPSHOT_KEEP == 2
    with pytest.raises(Exception):
        Config(SNAPSHOT_EVERY_EPISODES=-1)
    with pytest.raises(Exception):
        Config(SNAPSHOT_KEEP=0)


def test_diagnose_gap_exposes_snapshot_levers():
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--snapshot-every"' in src
    assert '"--resume-snapshot"' in src
    assert "save_training_snapshot" in src
    assert "load_training_snapshot" in src
