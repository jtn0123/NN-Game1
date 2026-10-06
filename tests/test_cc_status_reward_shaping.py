"""Tests for the PR #35/#36 shaping/curriculum lever plumbing in cc_status."""

import argparse
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from experiments.cc_status.cli_args import add_status_session_arguments  # noqa: E402
from experiments.cc_status.cli_helpers import tutorial_demo_bc_kwargs  # noqa: E402
from experiments.cc_status.config_helpers import (  # noqa: E402
    apply_reward_shaping_override,
)


def test_reward_shaping_override_defaults_leave_levers_off():
    config = Config()
    apply_reward_shaping_override(
        config,
        geodesic_potential=False,
        geodesic_potential_weight=0.3,
        show_locked_exit=False,
        reverse_curriculum_p=0.0,
    )
    assert config.CRYSTAL_CAVES_GEODESIC_POTENTIAL is False
    assert config.CRYSTAL_CAVES_SHOW_LOCKED_EXIT is False
    assert config.CRYSTAL_CAVES_REVERSE_CURRICULUM is False
    assert config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P == 0.0


def test_reward_shaping_override_enables_levers():
    config = Config()
    apply_reward_shaping_override(
        config,
        geodesic_potential=True,
        geodesic_potential_weight=0.45,
        show_locked_exit=True,
        reverse_curriculum_p=0.5,
    )
    assert config.CRYSTAL_CAVES_GEODESIC_POTENTIAL is True
    assert config.CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT == 0.45
    assert config.CRYSTAL_CAVES_SHOW_LOCKED_EXIT is True
    assert config.CRYSTAL_CAVES_REVERSE_CURRICULUM is True
    assert config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P == 0.5


def test_reward_shaping_override_reward_clip():
    config = Config()
    default_clip = config.REWARD_CLIP
    apply_reward_shaping_override(
        config,
        geodesic_potential=False,
        geodesic_potential_weight=0.3,
        show_locked_exit=False,
        reverse_curriculum_p=0.0,
    )
    assert config.REWARD_CLIP == default_clip

    apply_reward_shaping_override(
        config,
        geodesic_potential=False,
        geodesic_potential_weight=0.3,
        show_locked_exit=False,
        reverse_curriculum_p=0.0,
        reward_clip=0.0,
    )
    assert config.REWARD_CLIP == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"geodesic_potential_weight": -0.1, "reverse_curriculum_p": 0.0},
        {"geodesic_potential_weight": float("nan"), "reverse_curriculum_p": 0.0},
        {"geodesic_potential_weight": 0.3, "reverse_curriculum_p": -0.01},
        {"geodesic_potential_weight": 0.3, "reverse_curriculum_p": 1.01},
        {
            "geodesic_potential_weight": 0.3,
            "reverse_curriculum_p": 0.0,
            "reward_clip": -1.0,
        },
    ],
)
def test_reward_shaping_override_rejects_invalid_values(kwargs):
    config = Config()
    with pytest.raises(ValueError):
        apply_reward_shaping_override(
            config,
            geodesic_potential=True,
            show_locked_exit=False,
            **kwargs,
        )


def test_geo_compass_override_sets_and_validates():
    from experiments.cc_status.config_helpers import apply_geo_compass_override

    config = Config()
    apply_geo_compass_override(config, geo_compass=True, hazard_aware=True)
    assert config.CRYSTAL_CAVES_GEO_COMPASS is True
    assert config.CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE is True

    with pytest.raises(ValueError):
        apply_geo_compass_override(config, geo_compass=False, hazard_aware=True)


def test_status_session_parser_exposes_geo_compass_flags():
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)

    defaults = parser.parse_args(["tutorial-demo-conservative"])
    assert defaults.geo_compass is False
    assert defaults.geo_compass_hazard_aware is False

    args = parser.parse_args(
        ["tutorial-demo-conservative", "--geo-compass", "--geo-compass-hazard-aware"]
    )
    assert args.geo_compass is True
    assert args.geo_compass_hazard_aware is True

    kwargs = tutorial_demo_bc_kwargs(args, ("direct",))
    assert kwargs["geo_compass"] is True
    assert kwargs["geo_compass_hazard_aware"] is True


def test_status_session_parser_exposes_reward_shaping_flags():
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)

    defaults = parser.parse_args(["tutorial-demo-conservative"])
    assert defaults.geodesic_potential is False
    assert defaults.geodesic_potential_weight == 0.3
    assert defaults.show_locked_exit is False
    assert defaults.reverse_curriculum_p == 0.0

    args = parser.parse_args(
        [
            "tutorial-demo-conservative",
            "--geodesic-potential",
            "--geodesic-potential-weight",
            "0.4",
            "--show-locked-exit",
            "--reverse-curriculum-p",
            "0.5",
        ]
    )
    assert args.geodesic_potential is True
    assert args.geodesic_potential_weight == 0.4
    assert args.show_locked_exit is True
    assert args.reverse_curriculum_p == 0.5


def test_tutorial_demo_bc_kwargs_forwards_reward_shaping_flags():
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)
    opts = parser.parse_args(
        [
            "tutorial-demo-conservative",
            "--geodesic-potential",
            "--reverse-curriculum-p",
            "0.25",
        ]
    )
    kwargs = tutorial_demo_bc_kwargs(opts, ("direct",))
    assert kwargs["geodesic_potential"] is True
    assert kwargs["geodesic_potential_weight"] == 0.3
    assert kwargs["show_locked_exit"] is False
    assert kwargs["reverse_curriculum_p"] == 0.25


def test_config_snapshot_records_reward_shaping_levers():
    from experiments.cc_status.reports import config_snapshot
    from src.game.crystal_caves_experiments import (
        install_crystal_caves_experiment_defaults,
    )

    config = Config()
    install_crystal_caves_experiment_defaults(config)
    config.GAME_NAME = "crystal_caves"
    apply_reward_shaping_override(
        config,
        geodesic_potential=True,
        geodesic_potential_weight=0.3,
        show_locked_exit=False,
        reverse_curriculum_p=0.5,
    )
    snapshot = config_snapshot(config)
    assert snapshot["geodesic_potential"] is True
    assert snapshot["reverse_curriculum"] is True
    assert snapshot["reverse_curriculum_p"] == 0.5
    assert snapshot["show_locked_exit"] is False
    assert snapshot["reward_clip"] == config.REWARD_CLIP


def test_run_tutorial_demo_bc_signature_accepts_reward_shaping_kwargs():
    """The kwargs helper and the run function must stay in sync."""
    import inspect

    from experiments.cc_status.runs_demo import run_tutorial_demo_bc

    params = inspect.signature(run_tutorial_demo_bc).parameters
    for name in (
        "geodesic_potential",
        "geodesic_potential_weight",
        "show_locked_exit",
        "reverse_curriculum_p",
    ):
        assert name in params, f"run_tutorial_demo_bc missing {name}"

    opts_keys = set(
        tutorial_demo_bc_kwargs(
            _minimal_opts(),
            ("direct",),
        ).keys()
    )
    missing = opts_keys - set(params.keys())
    assert not missing, f"tutorial_demo_bc_kwargs passes unknown kwargs: {missing}"


def _minimal_opts() -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)
    return parser.parse_args(["tutorial-demo-conservative"])
