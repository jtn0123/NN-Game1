"""Tests for the RUN-26 prep tooling: eval objective override, contact-head
checkpoint persistence, and the configurable stall window."""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from experiments.cc_status.cli_args import add_status_session_arguments  # noqa: E402
from experiments.cc_status.cli_helpers import tutorial_demo_bc_kwargs  # noqa: E402
from experiments.cc_status.config_helpers import (  # noqa: E402
    apply_reward_shaping_override,
    cc_experiment_config,
)
from experiments.cc_status.reports import (  # noqa: E402
    load_selected_weight_snapshot,
    save_selected_weight_snapshot,
)
from experiments.cc_status.runs_transfer import config_from_selected_checkpoint  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402

# --- stall window -----------------------------------------------------------


def test_stall_window_default_unchanged():
    config = Config()
    assert config.CRYSTAL_CAVES_STALL_WINDOW_STEPS == 0
    game = CrystalCaves(config, headless=True)
    assert game.MAX_STEPS_WITHOUT_PROGRESS == 720
    assert CrystalCaves.MAX_STEPS_WITHOUT_PROGRESS == 720


def test_stall_window_config_overrides_game():
    config = Config()
    config.CRYSTAL_CAVES_STALL_WINDOW_STEPS = 1440
    game = CrystalCaves(config, headless=True)
    assert game.MAX_STEPS_WITHOUT_PROGRESS == 1440
    # class constant untouched — other instances keep the default
    assert CrystalCaves.MAX_STEPS_WITHOUT_PROGRESS == 720


def test_stall_window_config_validation():
    with pytest.raises(Exception):
        Config(CRYSTAL_CAVES_STALL_WINDOW_STEPS=-1)


def test_stall_window_override_helper():
    config = Config()
    apply_reward_shaping_override(
        config,
        geodesic_potential=False,
        geodesic_potential_weight=0.3,
        show_locked_exit=False,
        reverse_curriculum_p=0.0,
        stall_window=1440,
    )
    assert config.CRYSTAL_CAVES_STALL_WINDOW_STEPS == 1440
    with pytest.raises(ValueError):
        apply_reward_shaping_override(
            config,
            geodesic_potential=False,
            geodesic_potential_weight=0.3,
            show_locked_exit=False,
            reverse_curriculum_p=0.0,
            stall_window=0,
        )


def test_stall_window_cli_flag_and_kwargs():
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)
    defaults = parser.parse_args(["tutorial-demo-conservative"])
    assert defaults.stall_window is None
    opts = parser.parse_args(["tutorial-demo-conservative", "--stall-window", "1440"])
    assert opts.stall_window == 1440
    assert tutorial_demo_bc_kwargs(opts, ("direct",))["stall_window"] == 1440


def test_config_snapshot_records_stall_window():
    from experiments.cc_status.reports import config_snapshot
    from src.game.crystal_caves_experiments import (
        install_crystal_caves_experiment_defaults,
    )

    config = Config()
    install_crystal_caves_experiment_defaults(config)
    config.GAME_NAME = "crystal_caves"
    config.CRYSTAL_CAVES_STALL_WINDOW_STEPS = 1440
    assert config_snapshot(config)["stall_window"] == 1440


# --- eval objective override ------------------------------------------------


def _snapshot_with(first_crystal_goal: bool) -> dict:
    return {
        "config": {
            "first_crystal_goal": first_crystal_goal,
            "cave_difficulty": "tutorial",
        }
    }


def test_objective_override_full_beats_saved_first_crystal(tmp_path):
    config = config_from_selected_checkpoint(
        tmp_path,
        snapshot=_snapshot_with(True),
        seed=0,
        log_every=100,
        report_seconds=0.0,
        objective="full",
    )
    assert cc_experiment_config(config).CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL is False


def test_objective_override_first_crystal_beats_saved_full(tmp_path):
    config = config_from_selected_checkpoint(
        tmp_path,
        snapshot=_snapshot_with(False),
        seed=0,
        log_every=100,
        report_seconds=0.0,
        objective="first-crystal",
    )
    assert cc_experiment_config(config).CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL is True


def test_objective_none_keeps_saved(tmp_path):
    config = config_from_selected_checkpoint(
        tmp_path,
        snapshot=_snapshot_with(True),
        seed=0,
        log_every=100,
        report_seconds=0.0,
    )
    assert cc_experiment_config(config).CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL is True


def test_objective_invalid_rejected(tmp_path):
    with pytest.raises(ValueError):
        config_from_selected_checkpoint(
            tmp_path,
            snapshot=_snapshot_with(True),
            seed=0,
            log_every=100,
            report_seconds=0.0,
            objective="everything",
        )


def test_objective_cli_flag():
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)
    assert parser.parse_args(["eval-checkpoint"]).objective is None
    opts = parser.parse_args(["eval-checkpoint", "--objective", "full"])
    assert opts.objective == "full"
    with pytest.raises(SystemExit):
        parser.parse_args(["eval-checkpoint", "--objective", "everything"])


# --- contact-head checkpoint persistence -------------------------------------


def test_contact_head_snapshot_roundtrips_head_weights(tmp_path):
    weights = {
        "policy": {
            "trunk.weight": torch.ones(2, 2),
            "contact_action_head.weight": torch.full((2, 2), 3.0),
        },
        "target": {
            "trunk.weight": torch.ones(2, 2),
            "contact_action_head.weight": torch.full((2, 2), 3.0),
        },
    }
    path = tmp_path / "with_head.pth"
    save_selected_weight_snapshot(
        path,
        label="unit_with_head",
        config_payload={"contact_action_head": True, "first_crystal_goal": True},
        state_size=295,
        action_size=10,
        selected_episode=300,
        source_eval={"win_rate": 0.33},
        weights=weights,
    )
    loaded = load_selected_weight_snapshot(path)
    assert loaded["config"]["contact_action_head"] is True
    assert "contact_action_head.weight" in loaded["weights"]["policy"]
    assert torch.equal(
        loaded["weights"]["policy"]["contact_action_head.weight"],
        weights["policy"]["contact_action_head.weight"],
    )


def test_contact_head_offline_saves_standalone_checkpoint_source():
    """run_contact_head_offline must persist the combined trunk+head snapshot and
    record its path — guards against regressing to the B21 empty-models/ state."""
    import inspect

    from experiments.cc_status.corrections import run_contact_head_offline

    src = inspect.getsource(run_contact_head_offline)
    assert "save_selected_weight_snapshot" in src
    assert "contact_head_checkpoint" in src
    assert "capture_weight_snapshot" in src


def test_diagnose_gap_exposes_stall_window_lever():
    """The Track B harness must expose --stall-window and wire it to the config
    override, or the RUN-26 fidelity arm cannot run."""
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--stall-window"' in src
    assert 'overrides["CRYSTAL_CAVES_STALL_WINDOW_STEPS"] = stall_window' in src


def test_max_steps_override():
    config = Config()
    assert config.CRYSTAL_CAVES_MAX_STEPS_OVERRIDE == 0
    game = CrystalCaves(config, headless=True)
    assert game.MAX_STEPS == 3000
    config2 = Config()
    config2.CRYSTAL_CAVES_MAX_STEPS_OVERRIDE = 4500
    game2 = CrystalCaves(config2, headless=True)
    assert game2.MAX_STEPS == 4500
    assert CrystalCaves.MAX_STEPS == 3000  # class default untouched
    with pytest.raises(Exception):
        Config(CRYSTAL_CAVES_MAX_STEPS_OVERRIDE=-1)


def test_diagnose_gap_exposes_max_steps_lever():
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--max-steps"' in src
    assert 'overrides["CRYSTAL_CAVES_MAX_STEPS_OVERRIDE"] = max_steps' in src


def test_diagnose_gap_exposes_demo_td_weight_lever():
    """RUN-26c ablation lever: --demo-td-weight must wire to the DEMO_TD_WEIGHT
    override (0 = margin-only DQfD-lite) so the demo-TD Q-inflation hypothesis
    is testable."""
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--demo-td-weight"' in src
    assert 'overrides["DEMO_TD_WEIGHT"] = demo_td_weight' in src


def test_demo_td_weight_zero_drops_td_term():
    """With DEMO_TD_WEIGHT=0 the demo loss must be the margin term only —
    finite, and independent of the demo transitions' (large) rewards."""
    import numpy as np
    import torch

    from src.ai.agent import Agent

    config = Config()
    config.DEMO_TD_WEIGHT = 0.0
    config.FORCE_CPU = True
    agent = Agent(state_size=8, action_size=3, config=config)

    class _TinyStore:
        def __len__(self):
            return 4

        def sample(self, k):
            rng = np.random.default_rng(0)
            s = rng.random((k, 8), dtype=np.float32)
            a = np.zeros(k, dtype=np.int64)
            r = np.full(k, 1e6, dtype=np.float32)  # absurd reward: must not matter
            ns = rng.random((k, 8), dtype=np.float32)
            d = np.zeros(k, dtype=np.float32)
            nl = np.ones(k, dtype=np.float32)
            return s, a, r, ns, d, nl

    agent.attach_demo_store(_TinyStore())
    loss = agent._dqfd_loss()
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss.item()) < 1e3  # TD on 1e6 rewards would dwarf this


def test_diagnose_gap_exposes_demo_margin_weight_lever():
    """RUN-26d lever: --demo-margin-weight 0 with --demo-td-weight 0 must leave
    demo-prefix starts as the only demo mechanism."""
    import inspect

    import experiments.cc_status.diagnose_gap as dg

    src = inspect.getsource(dg)
    assert '"--demo-margin-weight"' in src
    assert 'overrides["DEMO_MARGIN_WEIGHT"] = demo_margin_weight' in src


def test_demo_loss_skipped_when_both_weights_zero():
    """Both demo-loss weights at 0 must skip the demo gradient entirely
    (prefix-starts-only arms attach a store but want no demo loss)."""
    from src.ai.agent import Agent

    config = Config()
    config.DEMO_TD_WEIGHT = 0.0
    config.DEMO_MARGIN_WEIGHT = 0.0
    config.FORCE_CPU = True
    agent = Agent(state_size=8, action_size=3, config=config)

    class _ExplodingStore:
        def __len__(self):
            return 4

        def sample(self, k):  # pragma: no cover - must never be called
            raise AssertionError("demo store sampled despite zero weights")

    agent.attach_demo_store(_ExplodingStore())
    assert agent._dqfd_loss() is None
