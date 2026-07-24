"""Tests for phase-2 opening-focused imitation: DEMO_OPENING_ONLY_STEPS restricts
the DQfD demo store to each route's first N transitions so the margin loss becomes
a pure route-opening prior."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.ai.demo_learning import DemoStore  # noqa: E402


def test_opening_steps_config_default_and_validation():
    config = Config()
    assert config.DEMO_OPENING_ONLY_STEPS == 0
    with pytest.raises(Exception):
        Config(DEMO_OPENING_ONLY_STEPS=-1)


def _write_demo(tmp_path, n_actions: int, level: int = 0) -> None:
    (tmp_path / "demo.json").write_text(
        json.dumps({"level": level, "actions": [0] * n_actions, "won": True})
    )


def _cc_config(**overrides) -> Config:
    config = Config()
    config.CRYSTAL_CAVES_IMPORTED = True
    config.FORCE_CPU = True
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_store_keeps_full_route_by_default(tmp_path):
    _write_demo(tmp_path, 40)
    store = DemoStore.from_dir(str(tmp_path), _cc_config())
    assert store is not None
    assert len(store) == 40


def test_store_keeps_only_opening_when_configured(tmp_path):
    _write_demo(tmp_path, 40)
    store = DemoStore.from_dir(str(tmp_path), _cc_config(DEMO_OPENING_ONLY_STEPS=15))
    assert store is not None
    assert len(store) == 15


def test_opening_longer_than_route_keeps_whole_route(tmp_path):
    _write_demo(tmp_path, 40)
    store = DemoStore.from_dir(str(tmp_path), _cc_config(DEMO_OPENING_ONLY_STEPS=500))
    assert store is not None
    assert len(store) == 40


def test_opening_nstep_tails_may_cross_the_cutoff(tmp_path):
    """The cutoff caps which transitions are KEPT, not the n-step horizon: the
    last kept transitions still bootstrap into the route beyond the cutoff."""
    _write_demo(tmp_path, 40)
    config = _cc_config(DEMO_OPENING_ONLY_STEPS=15, USE_N_STEP_RETURNS=True, N_STEP_SIZE=6)
    store = DemoStore.from_dir(str(tmp_path), config)
    assert store is not None
    assert len(store) == 15
    # transition 14 has 25 route steps after it, so its horizon is the full n=6 —
    # a hard-truncated episode would have shown a shrinking tail (1 at the end)
    assert int(store.n_step_lengths[-1]) == 6


def test_diagnose_gap_exposes_demo_opening_lever():
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--demo-opening-steps"' in src
    assert 'overrides["DEMO_OPENING_ONLY_STEPS"] = demo_opening_steps' in src


# --- margin-weight decay (RUN-62 iteration) ---------------------------------


def test_margin_decay_config_default_and_validation():
    config = Config()
    assert config.DEMO_MARGIN_DECAY_EPISODES == 0
    with pytest.raises(Exception):
        Config(DEMO_MARGIN_DECAY_EPISODES=-1)


def _cpu_agent(**overrides):
    from src.ai.agent import Agent

    config = Config()
    config.FORCE_CPU = True
    for key, value in overrides.items():
        setattr(config, key, value)
    return Agent(state_size=8, action_size=3, config=config)


def test_margin_decay_scale_tracks_episodes():
    agent = _cpu_agent(DEMO_MARGIN_DECAY_EPISODES=100)
    assert getattr(agent, "_demo_margin_scale", 1.0) == 1.0
    agent.decay_epsilon(episode=0)
    assert agent._demo_margin_scale == 1.0
    agent.decay_epsilon(episode=50)
    assert agent._demo_margin_scale == pytest.approx(0.5)
    agent.decay_epsilon(episode=100)
    assert agent._demo_margin_scale == 0.0
    agent.decay_epsilon(episode=250)  # past the horizon: clamps, never negative
    assert agent._demo_margin_scale == 0.0


def test_margin_decay_updates_during_epsilon_warmup():
    """The scale must track episodes even while epsilon decay is warmup-gated."""
    agent = _cpu_agent(DEMO_MARGIN_DECAY_EPISODES=100, EPSILON_WARMUP=1000)
    agent.decay_epsilon(episode=50)
    assert agent._demo_margin_scale == pytest.approx(0.5)


def test_margin_decay_off_leaves_scale_untouched():
    agent = _cpu_agent()
    agent.decay_epsilon(episode=50)
    assert not hasattr(agent, "_demo_margin_scale")


def test_fully_decayed_margin_skips_demo_forward_pass():
    """Once the scale hits zero (with td weight 0) the demo store must not even
    be sampled — the demo gradient is gone, not just multiplied by zero."""
    agent = _cpu_agent(DEMO_MARGIN_DECAY_EPISODES=10, DEMO_TD_WEIGHT=0.0, DEMO_MARGIN_WEIGHT=0.3)

    class _ExplodingStore:
        def __len__(self):
            return 4

        def sample(self, k):  # pragma: no cover - must never be called
            raise AssertionError("demo store sampled after full decay")

    agent.attach_demo_store(_ExplodingStore())
    agent.decay_epsilon(episode=10)
    assert agent._dqfd_loss() is None


def test_diagnose_gap_exposes_margin_decay_lever():
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--demo-margin-decay"' in src
    assert 'overrides["DEMO_MARGIN_DECAY_EPISODES"] = demo_margin_decay' in src


# --- re-ignition (RUN-63 iteration) -----------------------------------------


def test_reignite_config_defaults_and_validation():
    config = Config()
    assert config.DEMO_MARGIN_REIGNITE_EPISODE == 0
    assert config.DEMO_MARGIN_REIGNITE_SCALE == 0.5
    with pytest.raises(Exception):
        Config(DEMO_MARGIN_REIGNITE_EPISODE=-1)
    with pytest.raises(Exception):
        Config(DEMO_MARGIN_REIGNITE_SCALE=1.5)


def test_reignite_schedule_v_shape():
    """Scale must decay to zero, stay there, then floor at the re-ignite scale."""
    agent = _cpu_agent(
        DEMO_MARGIN_DECAY_EPISODES=100,
        DEMO_MARGIN_REIGNITE_EPISODE=300,
        DEMO_MARGIN_REIGNITE_SCALE=0.4,
    )
    agent.decay_epsilon(episode=50)
    assert agent._demo_margin_scale == pytest.approx(0.5)  # decaying
    agent.decay_epsilon(episode=200)
    assert agent._demo_margin_scale == 0.0  # dead zone between decay and reignite
    agent.decay_epsilon(episode=300)
    assert agent._demo_margin_scale == pytest.approx(0.4)  # re-ignited
    agent.decay_epsilon(episode=5000)
    assert agent._demo_margin_scale == pytest.approx(0.4)  # holds


def test_reignite_floor_never_lowers_live_decay():
    """If re-ignition overlaps the decay window, it floors — never clips — the scale."""
    agent = _cpu_agent(
        DEMO_MARGIN_DECAY_EPISODES=100,
        DEMO_MARGIN_REIGNITE_EPISODE=10,
        DEMO_MARGIN_REIGNITE_SCALE=0.2,
    )
    agent.decay_epsilon(episode=10)
    assert agent._demo_margin_scale == pytest.approx(0.9)  # decay still above floor


def test_diagnose_gap_exposes_reignite_lever():
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--demo-margin-reignite"' in src
    assert 'overrides["DEMO_MARGIN_REIGNITE_EPISODE"] = demo_margin_reignite' in src
    assert 'overrides["DEMO_MARGIN_REIGNITE_SCALE"] = demo_margin_reignite_scale' in src
