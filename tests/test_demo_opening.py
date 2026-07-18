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
