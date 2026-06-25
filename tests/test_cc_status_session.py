"""Tests for Crystal Caves status-session experiment helpers."""

import argparse
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.cc_status.cli as status_cli  # noqa: E402
from experiments.cc_status.cli import _requires_live_metrics  # noqa: E402
from experiments.cc_status.cli_args import (  # noqa: E402
    STATUS_SESSION_MODES,
    add_status_session_arguments,
)
from experiments.cc_status_session import (  # noqa: E402
    ArchiveStartCrystalCavesVec,
    InterleavedCrystalCavesVec,
    ReverseStartCrystalCavesVec,
    apply_close_zone_demo_action_override,
    apply_correction_action_override,
    apply_demo_action_override,
    apply_reverse_start,
    apply_route_aux_override,
    archive_milestone_key,
    archive_start_counts,
    behavior_clone_from_demonstrations,
    bridge_config,
    close_zone_oracle_action,
    collect_scripted_route_demonstrations,
    demo_action_arrays,
    first_crystal_config,
    full_tutorial_config,
    interleave_counts,
    level_eval_rollup,
    live_status_line,
    load_selected_weight_snapshot,
    make_interleaved_drill_config,
    parse_route_demo_variants,
    reverse_start_counts,
    reverse_start_modes,
    route_beam_plan,
    save_selected_weight_snapshot,
    seed_replay_from_demonstrations,
    select_route_demo_trajectories,
    selected_bridge_snapshot,
)
from src.game.crystal_caves import CrystalCaves  # noqa: E402


def _status_cli_dispatch_modes() -> set[str]:
    assert status_cli.__file__ is not None
    tree = ast.parse(Path(status_cli.__file__).read_text(encoding="utf-8"))
    modes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Attribute) or node.left.attr != "mode":
            continue
        for op, comparator in zip(node.ops, node.comparators):
            if (
                isinstance(op, ast.Eq)
                and isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, str)
            ):
                modes.add(comparator.value)
    return modes


def test_status_session_parser_modes_match_cli_dispatch():
    """Every parser mode should have an explicit dispatch branch."""
    assert set(STATUS_SESSION_MODES) == _status_cli_dispatch_modes()


def test_live_metrics_requirement_tracks_training_runs_and_heartbeat():
    opts = argparse.Namespace(heartbeat_seconds=30.0)

    assert _requires_live_metrics(opts, {"runs": [{"train_seconds": 1.0}]}) is True
    assert _requires_live_metrics(opts, {"runs": [{"train_seconds": 0.0}]}) is False

    opts.heartbeat_seconds = 0.0
    assert _requires_live_metrics(opts, {"runs": [{"train_seconds": 1.0}]}) is False


def test_status_session_parser_preserves_representative_experiment_flags():
    """CLI argument groups should still expose every major experiment surface."""
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)

    args = parser.parse_args(
        [
            "correction-finetune",
            "--episodes",
            "12",
            "--route-demo-variants",
            "direct,recovery",
            "--demo-selection-mode",
            "filtered-weighted",
            "--demo-action-weight",
            "0.12",
            "--demo-conservative-weight",
            "0.03",
            "--close-zone-extra-label-source",
            "oracle",
            "--eval-games",
            "9",
            "--trace-eval-games",
            "2",
            "--interleave-bridge-envs",
            "1",
            "--archive-replay-prob",
            "0.5",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
            "--correction-action-weight",
            "0.04",
            "--save-selected-checkpoint",
            "--no-artifact-validation",
        ]
    )

    assert args.mode == "correction-finetune"
    assert args.episodes == 12
    assert args.route_demo_variants == "direct,recovery"
    assert args.demo_selection_mode == "filtered-weighted"
    assert args.demo_action_weight == 0.12
    assert args.demo_conservative_weight == 0.03
    assert args.close_zone_extra_label_source == "oracle"
    assert args.eval_games == 9
    assert args.trace_eval_games == 2
    assert args.interleave_bridge_envs == 1
    assert args.archive_replay_prob == 0.5
    assert args.checkpoint == "selected.pth"
    assert args.correction_dataset == "correction_examples.npz"
    assert args.correction_action_weight == 0.04
    assert args.save_selected_checkpoint is True
    assert args.no_artifact_validation is True


def test_interleave_counts_defaults_to_two_drill_lanes_for_eight_envs():
    """The recommended 25% drill mix should become 6 full + 2 drill lanes."""
    assert interleave_counts(vec_envs=8, skill_ratio=0.25, skill_envs=None) == (6, 2)


def test_interleave_counts_allows_explicit_drill_lane_count():
    """Explicit drill lane counts should override the ratio and stay clamped."""
    assert interleave_counts(vec_envs=8, skill_ratio=0.25, skill_envs=3) == (5, 3)
    assert interleave_counts(vec_envs=8, skill_ratio=0.25, skill_envs=99) == (1, 7)


def test_interleave_counts_requires_vector_mixing():
    """Interleaving needs at least one full lane and one drill lane."""
    with pytest.raises(ValueError, match="--vec-envs >= 2"):
        interleave_counts(vec_envs=1, skill_ratio=0.25, skill_envs=None)


def test_reverse_start_counts_defaults_to_two_reverse_lanes_for_eight_envs():
    """The recommended 25% reverse mix should become 6 full + 2 reverse lanes."""
    assert reverse_start_counts(vec_envs=8, reverse_ratio=0.25, reverse_envs=None) == (6, 2)
    assert reverse_start_modes(2) == ["reverse_objective", "reverse_exit"]


def test_archive_start_counts_defaults_to_two_archive_lanes_for_eight_envs():
    """The recommended 25% archive mix should become 6 full + 2 archive lanes."""
    assert archive_start_counts(vec_envs=8, archive_ratio=0.25, archive_envs=None) == (6, 2)


def test_interleaved_vec_env_mixes_full_and_drill_sources(tmp_path):
    """The mixed vector env should expose normal vectorized batches and source stats."""
    full_config = full_tutorial_config(
        tmp_path / "full",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    full_config.CRYSTAL_CAVES_POOL_SIZE = 2
    drill_config = make_interleaved_drill_config(full_config)

    vec = InterleavedCrystalCavesVec(
        full_config=full_config,
        skill_config=drill_config,
        full_envs=2,
        skill_envs=1,
        skill_source="drill",
        headless=True,
    )
    try:
        states = vec.reset()
        assert states.shape == (3, vec.state_size)
        assert vec.sources == ["full", "full", "drill"]

        # Force one full and one drill lane to complete so stats update without
        # running thousands of platformer frames.
        vec.envs[0].game_over = True
        vec.envs[2].game_over = True
        _, _, dones, infos = vec.step_no_copy(np.zeros(vec.num_envs, dtype=np.int64))

        assert dones.tolist() == [True, False, True]
        assert infos[0]["training_source"] == "full"
        assert infos[2]["training_source"] == "drill"
        stats = vec.source_stats()
        assert stats["full"]["episodes"] == 1
        assert stats["drill"]["episodes"] == 1
    finally:
        vec.close()


def test_reverse_start_vec_env_mixes_full_and_near_objective_sources(tmp_path):
    """Reverse-start lanes should reset near objectives without changing eval defaults."""
    cfg = full_tutorial_config(
        tmp_path / "reverse",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    cfg.CRYSTAL_CAVES_PROCEDURAL = False

    vec = ReverseStartCrystalCavesVec(
        full_config=cfg,
        full_envs=1,
        reverse_envs=2,
        headless=True,
    )
    try:
        states = vec.reset()
        assert states.shape == (3, vec.state_size)
        assert vec.sources == ["full", "reverse_objective", "reverse_exit"]

        reverse_stats = vec.reverse_start_stats()
        assert reverse_stats["reverse_objective"]["attempts"] == 1
        assert reverse_stats["reverse_exit"]["attempts"] == 1
        assert reverse_stats["reverse_objective"]["applied"] == 1
        assert reverse_stats["reverse_exit"]["applied"] == 1
        assert vec.envs[2].exit_unlocked is True
        assert len(vec.envs[2].crystals) == 0

        vec.envs[1].game_over = True
        _, _, dones, infos = vec.step_no_copy(np.zeros(vec.num_envs, dtype=np.int64))

        assert dones.tolist() == [False, True, False]
        assert infos[1]["training_source"] == "reverse_objective"
        stats = vec.source_stats()
        assert stats["reverse_objective"]["episodes"] == 1
    finally:
        vec.close()


def test_apply_reverse_exit_start_unlocks_exit_near_target(tmp_path):
    """Reverse exit starts should make the terminal skill immediately trainable."""
    cfg = full_tutorial_config(
        tmp_path / "reverse-exit",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    cfg.CRYSTAL_CAVES_PROCEDURAL = False
    game = ReverseStartCrystalCavesVec(
        full_config=cfg,
        full_envs=1,
        reverse_envs=1,
        headless=True,
    ).envs[0]

    try:
        assert apply_reverse_start(game, "reverse_exit") is True
        target, distance = game._current_target()

        assert target is not None and target[0] == "exit"
        assert game.exit_unlocked is True
        assert len(game.crystals) == 0
        assert distance / game.TILE_SIZE <= 6
    finally:
        game.close()


def test_archive_start_vec_env_stores_and_replays_full_milestones(tmp_path):
    """Archive lanes should replay deep-copied milestones discovered by full lanes."""
    cfg = full_tutorial_config(
        tmp_path / "archive",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    cfg.CRYSTAL_CAVES_PROCEDURAL = False

    vec = ArchiveStartCrystalCavesVec(
        full_config=cfg,
        full_envs=1,
        archive_envs=1,
        replay_prob=1.0,
        max_size=4,
        min_steps=0,
        headless=True,
    )
    try:
        states = vec.reset()
        assert states.shape == (2, vec.state_size)
        assert vec.sources == ["full", "archive"]

        key = archive_milestone_key(vec.envs[0])
        assert key[0] in {"crystal", "switch", "exit", "none"}
        vec.envs[0].steps = 5
        vec._maybe_archive(vec.envs[0])

        stats = vec.archive_stats()
        assert stats["size"] == 1
        assert stats["stores"] == 1
        assert stats["seen_milestones"] == 1

        vec._reset_env(1)
        replay_stats = vec.archive_stats()
        assert replay_stats["replays"] == 1
        assert replay_stats["replay_attempts"] == 2
        assert vec.envs[1].steps == 5

        vec.envs[1].game_over = True
        _, _, dones, infos = vec.step_no_copy(np.zeros(vec.num_envs, dtype=np.int64))
        assert dones.tolist() == [False, True]
        assert infos[1]["training_source"] == "archive"
        assert vec.source_stats()["archive"]["episodes"] == 1
    finally:
        vec.close()


def test_bridge_config_enables_bridges_without_procedural_or_drills(tmp_path):
    """Bridge runs should load the compositional bridge cave set only."""
    cfg = bridge_config(
        tmp_path / "bridge",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    assert cfg.CRYSTAL_CAVES_BRIDGES is True
    assert cfg.CRYSTAL_CAVES_DRILLS is False
    assert cfg.CRYSTAL_CAVES_PROCEDURAL is False


def test_first_crystal_config_keeps_full_caves_but_changes_terminal_objective(tmp_path):
    """Route pretraining should use procedural tutorial caves, not hand-authored drills."""
    cfg = first_crystal_config(
        tmp_path / "first-crystal",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    assert cfg.CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL is True
    assert cfg.CRYSTAL_CAVES_PROCEDURAL is True
    assert cfg.CRYSTAL_CAVES_DRILLS is False
    assert cfg.CRYSTAL_CAVES_BRIDGES is False
    assert cfg.CRYSTAL_CAVES_DIFFICULTY == "tutorial"


def test_first_crystal_config_allows_route_floor_difficulty(tmp_path):
    """B1 uses a training-only route floor while preserving the same objective."""
    cfg = first_crystal_config(
        tmp_path / "route-floor",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
        difficulty="route_floor",
    )

    assert cfg.CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL is True
    assert cfg.CRYSTAL_CAVES_PROCEDURAL is True
    assert cfg.CRYSTAL_CAVES_DIFFICULTY == "route_floor"


def test_apply_route_aux_override_is_opt_in(tmp_path):
    """B3f route supervision should stay disabled unless a positive weight is set."""
    cfg = first_crystal_config(
        tmp_path / "route-aux",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    apply_route_aux_override(cfg, route_aux_weight=0.0, route_aux_deadband=0.01)
    assert cfg.CRYSTAL_CAVES_ROUTE_AUX_LOSS is False

    apply_route_aux_override(cfg, route_aux_weight=0.05, route_aux_deadband=0.02)
    assert cfg.CRYSTAL_CAVES_ROUTE_AUX_LOSS is True
    assert cfg.CRYSTAL_CAVES_ROUTE_AUX_WEIGHT == 0.05
    assert cfg.CRYSTAL_CAVES_ROUTE_AUX_DEADBAND == 0.02


def test_apply_demo_action_override_is_opt_in(tmp_path):
    """B3h demo action margin loss should stay disabled unless weight is positive."""
    cfg = first_crystal_config(
        tmp_path / "demo-action",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    apply_demo_action_override(
        cfg,
        demo_action_weight=0.0,
        demo_action_margin=0.8,
        demo_action_batch_size=64,
        demo_conservative_weight=0.0,
        demo_conservative_temperature=1.0,
    )
    assert cfg.CRYSTAL_CAVES_DEMO_ACTION_LOSS is False

    apply_demo_action_override(
        cfg,
        demo_action_weight=0.05,
        demo_action_margin=0.7,
        demo_action_batch_size=32,
        demo_conservative_weight=0.02,
        demo_conservative_temperature=0.75,
    )
    assert cfg.CRYSTAL_CAVES_DEMO_ACTION_LOSS is True
    assert cfg.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT == 0.05
    assert cfg.CRYSTAL_CAVES_DEMO_ACTION_MARGIN == 0.7
    assert cfg.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE == 32
    assert cfg.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT == 0.02
    assert cfg.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE == 0.75


def test_apply_close_zone_demo_action_override_is_opt_in(tmp_path):
    """B3t close-zone extra action pressure should stay disabled unless requested."""
    cfg = first_crystal_config(
        tmp_path / "close-zone-demo-action",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    apply_close_zone_demo_action_override(
        cfg,
        close_zone_demo_action_weight=0.0,
        close_zone_demo_action_batch_size=64,
    )
    assert cfg.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS is False

    apply_close_zone_demo_action_override(
        cfg,
        close_zone_demo_action_weight=0.03,
        close_zone_demo_action_batch_size=32,
    )
    assert cfg.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS is True
    assert cfg.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT == 0.03
    assert cfg.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE == 32


def test_apply_correction_action_override_is_opt_in(tmp_path):
    """Correction action supervision should stay disabled unless weight is positive."""
    cfg = first_crystal_config(
        tmp_path / "correction-action",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    apply_correction_action_override(
        cfg,
        correction_action_weight=0.0,
        correction_action_margin=0.6,
        correction_action_batch_size=64,
    )
    assert cfg.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS is False

    apply_correction_action_override(
        cfg,
        correction_action_weight=0.02,
        correction_action_margin=0.5,
        correction_action_batch_size=32,
    )
    assert cfg.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS is True
    assert cfg.CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT == 0.02
    assert cfg.CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN == 0.5
    assert cfg.CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE == 32


def test_live_status_line_shows_enabled_zero_loss_correction():
    """Correction live status should show active supervision even after zero hinge loss."""
    line = live_status_line(
        {
            "label": "correction_finetune",
            "episode": 1,
            "total_episodes": 10,
            "episode_pct": 0.1,
            "avg_score_100": 0.0,
            "avg_progress_100": 0.0,
            "best_progress": 0.0,
            "win_rate_100": 0.0,
            "avg_loss_100": 0.0,
            "avg_q_100": 0.0,
            "steps_per_second": 100.0,
            "correction_action_enabled": True,
            "correction_action_transitions": 12,
            "correction_action_samples_100": 0,
            "avg_correction_action_loss_100": 0.0,
            "avg_correction_action_accuracy_100": 0.0,
        }
    )

    assert "| corr 0.000/0% n=0" in line


def test_level_eval_rollup_summarizes_skill_rows():
    """Bridge history should have stable aggregate fields for early-stop decisions."""
    rollup = level_eval_rollup(
        [
            {
                "win_rate": 1.0,
                "collected_any_rate": 1.0,
                "all_crystals_rate": 1.0,
                "mean_progress": 1.0,
            },
            {
                "win_rate": 0.0,
                "collected_any_rate": 0.5,
                "all_crystals_rate": 0.0,
                "mean_progress": 0.25,
            },
        ]
    )

    assert rollup == {
        "mean_win_rate": 0.5,
        "mean_any_crystal_rate": 0.75,
        "mean_all_crystals_rate": 0.5,
        "mean_progress": 0.625,
        "solved_levels": 1,
        "levels": 2,
    }


def test_selected_bridge_snapshot_prefers_greedy_wins_over_later_episode():
    """Bridge transfer should keep the best greedy policy, not just final weights."""
    history = [
        {
            "episode": 300,
            "rollup": {
                "mean_win_rate": 0.2,
                "mean_all_crystals_rate": 0.2,
                "mean_any_crystal_rate": 0.8,
                "mean_progress": 0.525,
            },
        },
        {
            "episode": 350,
            "rollup": {
                "mean_win_rate": 0.6,
                "mean_all_crystals_rate": 0.6,
                "mean_any_crystal_rate": 1.0,
                "mean_progress": 0.803,
            },
        },
        {
            "episode": 400,
            "rollup": {
                "mean_win_rate": 0.4,
                "mean_all_crystals_rate": 0.8,
                "mean_any_crystal_rate": 1.0,
                "mean_progress": 0.899,
            },
        },
    ]

    assert selected_bridge_snapshot(history) is history[1]


def test_seed_replay_from_demonstrations_repeats_trajectories():
    """Demo replay seeding should push every transition the requested number of times."""

    class FakeMemory:
        def __len__(self):
            return 6

    class FakeAgent:
        def __init__(self):
            self.memory = FakeMemory()
            self.calls = []

        def remember(self, state, action, reward, next_state, done):
            self.calls.append((state, action, reward, next_state, done))

    transition = (
        np.zeros(3, dtype=np.float32),
        2,
        1.5,
        np.ones(3, dtype=np.float32),
        False,
    )
    agent = FakeAgent()

    summary = seed_replay_from_demonstrations(agent, [[transition, transition]], repeat=3)

    assert len(agent.calls) == 6
    assert summary == {
        "trajectories": 1,
        "repeat": 3,
        "pushed_transitions": 6,
        "memory_size_after_seed": 6,
    }


def test_demo_action_arrays_flatten_trajectories():
    """Online demo supervision should train on every successful demo state/action."""
    trajectory = [
        (
            np.array([1.0, 2.0], dtype=np.float32),
            2,
            0.0,
            np.array([3.0, 4.0], dtype=np.float32),
            False,
        ),
        (
            np.array([5.0, 6.0], dtype=np.float32),
            1,
            1.0,
            np.array([7.0, 8.0], dtype=np.float32),
            True,
        ),
    ]

    states, actions = demo_action_arrays([trajectory])

    assert states.shape == (2, 2)
    assert actions.tolist() == [2, 1]


def test_selected_weight_snapshot_round_trips_without_replay_memory(tmp_path):
    """Selected snapshots should stay small and contain only weights plus metadata."""
    path = tmp_path / "selected_ep10.pth"
    weights = {
        "policy": {"layer.weight": torch.ones((2, 2))},
        "target": {"layer.weight": torch.zeros((2, 2))},
    }

    saved = save_selected_weight_snapshot(
        path,
        label="tutorial_demo_bc",
        config_payload={"first_crystal_goal": True, "cave_difficulty": "tutorial"},
        state_size=4,
        action_size=2,
        selected_episode=10,
        source_eval={"wins": 2, "num_games": 16},
        weights=weights,
    )
    assert saved == str(path)
    loaded = load_selected_weight_snapshot(path)

    assert loaded["episode"] == 10
    assert loaded["source_eval"] == {"wins": 2, "num_games": 16}
    assert torch.equal(loaded["weights"]["policy"]["layer.weight"], torch.ones((2, 2)))
    assert "memory" not in loaded
    assert "replay" not in loaded
    assert path.stat().st_size < 100_000


def test_parse_route_demo_variants_accepts_beam_controller():
    """B3p planner-assisted demos should be opt-in without breaking old defaults."""
    assert parse_route_demo_variants("direct,recovery,beam") == (
        "direct",
        "recovery",
        "beam",
    )


def test_route_beam_plan_is_bounded_and_side_effect_free(tmp_path):
    """The lookahead controller should simulate futures without mutating the live game."""
    cfg = first_crystal_config(
        tmp_path / "beam-demo",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
        difficulty="tutorial",
    )
    cfg.CRYSTAL_CAVES_POOL_SIZE = 4
    game = CrystalCaves(cfg, headless=True)
    try:
        game._randomize_levels = False
        game.level_index = 0
        game.reset()
        before = (game._player_tile(), len(game.crystals), game.steps, game.score)

        plan = route_beam_plan(game, stale_steps=40)

        assert 1 <= len(plan) <= 8
        assert all(0 <= action < game.action_size for action in plan)
        assert (game._player_tile(), len(game.crystals), game.steps, game.score) == before
    finally:
        game.close()


def test_close_zone_oracle_action_is_side_effect_free(tmp_path):
    """B3u oracle labels should be scored on copied game states only."""
    cfg = first_crystal_config(
        tmp_path / "oracle-label",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
        difficulty="tutorial",
    )
    cfg.CRYSTAL_CAVES_POOL_SIZE = 4
    game = CrystalCaves(cfg, headless=True)
    try:
        game._randomize_levels = False
        game.level_index = 0
        game.reset()
        before = (
            game._player_tile(),
            len(game.crystals),
            game.steps,
            game.score,
            round(game.player_x, 3),
            round(game.player_y, 3),
        )

        action, meta = close_zone_oracle_action(game, stale_steps=40)

        assert 0 <= action < game.action_size
        assert meta["reason"] in {"ok", "no_target"}
        assert (
            game._player_tile(),
            len(game.crystals),
            game.steps,
            game.score,
            round(game.player_x, 3),
            round(game.player_y, 3),
        ) == before
    finally:
        game.close()


def test_scripted_route_floor_demonstrations_keep_successes(tmp_path):
    """The B2 demonstrator should produce successful route-floor examples."""
    cfg = first_crystal_config(
        tmp_path / "route-demo",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
        difficulty="route_floor",
    )
    cfg.CRYSTAL_CAVES_POOL_SIZE = 4

    demos = collect_scripted_route_demonstrations(cfg, max_levels=4, max_steps=800)
    summary = demos["summary"]

    assert summary["attempts"] == 4
    assert summary["kept_trajectories"] >= 1
    assert summary["kept_transitions"] > 0
    assert len(demos["trajectories"]) == summary["kept_trajectories"]
    assert all(trajectory[-1][4] is True for trajectory in demos["trajectories"])


def test_scripted_tutorial_demonstrations_keep_on_distribution_successes(tmp_path):
    """B3g should collect successful demos directly from normal tutorial maps."""
    cfg = first_crystal_config(
        tmp_path / "tutorial-demo",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
        difficulty="tutorial",
    )
    cfg.CRYSTAL_CAVES_POOL_SIZE = 64

    demos = collect_scripted_route_demonstrations(cfg, max_levels=16, max_steps=800)
    summary = demos["summary"]

    assert summary["attempts"] == 16
    assert summary["kept_trajectories"] >= 1
    assert summary["kept_transitions"] > 0
    assert summary["close_zone_kept_transitions"] > 0
    assert "failure_mode_counts" in summary
    assert "mean_failed_min_target_distance_tiles" in summary
    assert "mean_kept_close_zone_jump_rate" in summary
    assert "mean_kept_idle_interact_rate" in summary
    assert len(summary["kept_rows"]) == summary["kept_trajectories"]
    assert all("failure_modes" in row for row in summary["rows"])
    assert all("close_zone_jump_rate" in row for row in summary["rows"])
    assert all("idle_interact_rate" in row for row in summary["rows"])
    assert all("max_tile_visit_frac" in row for row in summary["rows"])
    assert len(demos["close_zone_trajectories"]) == summary["kept_trajectories"]
    assert (
        sum(len(traj) for traj in demos["close_zone_trajectories"])
        == summary["close_zone_kept_transitions"]
    )
    assert len(demos["trajectories"]) == summary["kept_trajectories"]
    assert all(trajectory[-1][4] is True for trajectory in demos["trajectories"])


def test_scripted_tutorial_demonstrations_can_oracle_label_close_zone(tmp_path):
    """B3u should add bounded oracle labels only when explicitly requested."""
    cfg = first_crystal_config(
        tmp_path / "tutorial-oracle-demo",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
        difficulty="tutorial",
    )
    cfg.CRYSTAL_CAVES_POOL_SIZE = 64

    demos = collect_scripted_route_demonstrations(
        cfg,
        max_levels=8,
        max_steps=800,
        controller_variants=("direct", "recovery"),
        oracle_close_zone_labels=True,
        oracle_close_zone_stride=4,
        oracle_close_zone_max_per_trajectory=4,
    )
    summary = demos["summary"]

    assert summary["kept_trajectories"] >= 1
    assert summary["oracle_close_zone_enabled"] is True
    assert summary["oracle_close_zone_stride"] == 4
    assert summary["oracle_close_zone_max_per_trajectory"] == 4
    assert summary["oracle_close_zone_kept_transitions"] > 0
    assert "oracle_close_zone_action_counts" in summary
    assert "mean_kept_oracle_close_zone_relabel_rate" in summary
    assert len(demos["oracle_close_zone_trajectories"]) == summary["kept_trajectories"]
    assert (
        sum(len(traj) for traj in demos["oracle_close_zone_trajectories"])
        == summary["oracle_close_zone_kept_transitions"]
    )
    assert all(len(traj) <= 4 for traj in demos["oracle_close_zone_trajectories"] if len(traj) > 0)


def test_filtered_weighted_demo_selection_filters_messy_beam():
    """B3r should keep clean demos while down-weighting noisy beam coverage."""
    transition = (
        np.zeros(2, dtype=np.float32),
        CrystalCaves.RIGHT,
        0.0,
        np.ones(2, dtype=np.float32),
        True,
    )
    demos = {
        "trajectories": [
            [transition] * 10,
            [transition] * 80,
            [transition] * 20,
        ],
        "close_zone_trajectories": [
            [transition] * 3,
            [transition] * 2,
            [transition] * 5,
        ],
        "oracle_close_zone_trajectories": [
            [transition] * 3,
            [transition] * 2,
            [transition] * 5,
        ],
        "summary": {
            "kept_rows": [
                {
                    "trajectory_index": 0,
                    "variant": "direct",
                    "steps": 10,
                    "close_zone_steps": 3,
                    "close_zone_jump_rate": 0.0,
                    "close_zone_idle_interact_rate": 0.0,
                    "max_tile_visit_frac": 0.1,
                },
                {
                    "trajectory_index": 1,
                    "variant": "beam",
                    "steps": 80,
                    "close_zone_steps": 2,
                    "close_zone_jump_rate": 0.0,
                    "close_zone_idle_interact_rate": 1.0,
                    "max_tile_visit_frac": 0.9,
                },
                {
                    "trajectory_index": 2,
                    "variant": "beam",
                    "steps": 20,
                    "close_zone_steps": 5,
                    "close_zone_jump_rate": 0.2,
                    "close_zone_idle_interact_rate": 0.0,
                    "max_tile_visit_frac": 0.2,
                },
            ]
        },
    }

    selected = select_route_demo_trajectories(demos, mode="filtered-weighted")
    summary = selected["summary"]

    assert summary["selected_unique_trajectories"] == 2
    assert summary["selected_weighted_trajectories"] == 3
    assert summary["selected_by_variant"] == {"direct": 1, "beam": 1}
    assert summary["weighted_by_variant"] == {"direct": 2, "beam": 1}
    assert summary["excluded_reasons"] == {"beam_quality_filter": 1}
    assert summary["selected_transitions"] == 40
    assert summary["selected_close_zone_transitions"] == 11
    assert summary["selected_oracle_close_zone_transitions"] == 11


def test_behavior_clone_from_demonstrations_allows_empty_demo_set():
    """A failed demo collection should be reported as a no-op, not crash the runner."""
    summary = behavior_clone_from_demonstrations(
        object(),
        [],
        epochs=2,
        batch_size=8,
    )

    assert summary == {
        "epochs": 2,
        "batch_size": 8,
        "transitions": 0,
        "updates": 0,
        "final_loss": 0.0,
    }
