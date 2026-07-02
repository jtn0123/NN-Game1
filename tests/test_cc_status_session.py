"""Tests for Crystal Caves status-session experiment helpers."""

import argparse
import ast
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.cc_status.cli as status_cli  # noqa: E402
import experiments.cc_status.cli_label_modes as label_modes  # noqa: E402
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
    apply_contact_action_head_override,
    apply_correction_action_override,
    apply_demo_action_override,
    apply_distributional_dqn_override,
    apply_policy_anchor_override,
    apply_reverse_start,
    apply_route_aux_override,
    archive_milestone_key,
    archive_start_counts,
    behavior_clone_from_demonstrations,
    bridge_config,
    close_zone_oracle_action,
    collect_scripted_route_demonstrations,
    config_snapshot,
    contact_config,
    demo_action_arrays,
    final_contact_option_action,
    first_crystal_config,
    full_tutorial_config,
    interleave_counts,
    level_eval_rollup,
    live_status_line,
    load_selected_weight_snapshot,
    make_interleaved_contact_config,
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
from src.game.crystal_caves_drills import CONTACT_CAVES  # noqa: E402


def _status_cli_dispatch_modes() -> set[str]:
    modes: set[str] = set()
    for module in (status_cli, label_modes):
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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
    assert (
        _requires_live_metrics(
            opts,
            {
                "runs": [
                    {
                        "train_seconds": 1.0,
                        "contact_action_head_calibration": {"decision": {"passed": True}},
                    }
                ]
            },
        )
        is False
    )

    opts.heartbeat_seconds = 0.0
    assert _requires_live_metrics(opts, {"runs": [{"train_seconds": 1.0}]}) is False


def test_interrupted_status_session_payload_uses_live_and_source_history(tmp_path):
    """KeyboardInterrupt fallback should leave comparable partial artifacts behind."""
    out_dir = tmp_path / "session"
    run_dir = out_dir / "contact_interleaved"
    run_dir.mkdir(parents=True)
    live_payload = {
        "label": "contact_interleaved",
        "status": "interrupted",
        "episode": 150,
        "total_episodes": 300,
        "elapsed_seconds": 12.5,
        "steps_per_second": 1000.0,
        "total_steps": 12500,
        "epsilon": 0.2,
        "memory_size": 4096,
        "avg_loss_100": 0.1,
        "avg_q_100": 3.0,
        "avg_score_100": 10.0,
        "win_rate_100": 0.25,
        "avg_progress_100": 0.5,
        "best_progress": 0.9,
        "end_reason_counts_100": {"timeout": 3},
        "latest_eval": {
            "wins": 0,
            "num_games": 8,
            "win_rate": 0.0,
            "mean_crystal_frac": 0.0,
            "mean_depth_frac": 0.2,
            "end_reason_counts": {"timeout": 8},
        },
        "source_stats": {"contact": {"win_rate_100": 1.0}},
        "contact_lane_win_rate_100": 1.0,
    }
    (run_dir / "live_metrics.json").write_text(json.dumps(live_payload), encoding="utf-8")
    (run_dir / "live_metrics.jsonl").write_text(json.dumps(live_payload) + "\n", encoding="utf-8")
    source_rows = [
        {
            "episode": 50,
            "source_eval": {
                "wins": 0,
                "num_games": 16,
                "win_rate": 0.0,
                "mean_crystal_frac": 0.1,
                "mean_depth_frac": 0.3,
                "mean_score": 40.0,
                "end_reason_counts": {"timeout": 16},
            },
        },
        {
            "episode": 100,
            "source_eval": {
                "wins": 0,
                "num_games": 16,
                "win_rate": 0.0,
                "mean_crystal_frac": 0.2,
                "mean_depth_frac": 0.4,
                "mean_score": 80.0,
                "end_reason_counts": {"timeout": 16},
            },
        },
    ]
    (run_dir / "source_eval_history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in source_rows),
        encoding="utf-8",
    )
    payload = {"runs": []}

    assert status_cli._append_interrupted_run_from_live_metrics(out_dir, payload) is True

    run = payload["runs"][0]
    assert payload["interrupted"] is True
    assert run["partial"] is True
    assert run["episodes"] == 150
    assert run["final_eval"]["mean_crystal_frac"] == 0.2
    assert run["selected_source_episode"] == 100
    assert run["contact_lane_win_rate_100"] == 1.0
    assert run["route_contact_scorecard"]["eval_source"] == "selected_source_eval"
    assert run["route_contact_scorecard"]["metrics"]["first_crystal_rate"] == 0.2


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
            "--contact-pool-size",
            "128",
            "--contact-eval-pool-size",
            "32",
            "--history-state",
            "--history-steps",
            "4",
            "--distributional-dqn",
            "--c51-atoms",
            "51",
            "--c51-v-min",
            "-20",
            "--c51-v-max",
            "120",
            "--archive-replay-prob",
            "0.5",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
            "--correction-datasets",
            "a.npz,b.npz",
            "--correction-action-weight",
            "0.04",
            "--policy-anchor-weight",
            "0.02",
            "--policy-anchor-temperature",
            "1.5",
            "--policy-anchor-min-distance-tiles",
            "3.0",
            "--contact-action-weight",
            "0.03",
            "--contact-action-batch-size",
            "32",
            "--contact-action-distance",
            "2.5",
            "--contact-head-offline-steps",
            "123",
            "--contact-head-learning-rate",
            "0.002",
            "--contact-head-confidence",
            "0.8",
            "--contact-head-jump-confidence",
            "0.85",
            "--contact-head-balance-classes",
            "--contact-head-calibration-frac",
            "0.2",
            "--contact-head-calibration-seed",
            "19",
            "--contact-head-min-calibration-accuracy",
            "0.72",
            "--contact-head-min-class-examples",
            "12",
            "--contact-label-state-decimals",
            "2",
            "--contact-label-adjacent-step-window",
            "3",
            "--contact-label-top-groups",
            "7",
            "--contact-label-filter-majority-threshold",
            "0.75",
            "--final-contact-distance",
            "2.5",
            "--final-contact-commit-steps",
            "6",
            "--final-contact-cancel-outside",
            "--final-contact-policy-advantage-gate",
            "--final-contact-min-option-advantage",
            "250",
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
    assert args.interleave_contact_ratio == 0.25
    assert args.contact_pool_size == 128
    assert args.contact_eval_pool_size == 32
    assert args.history_state is True
    assert args.history_steps == 4
    assert args.distributional_dqn is True
    assert args.c51_atoms == 51
    assert args.c51_v_min == -20.0
    assert args.c51_v_max == 120.0
    assert args.archive_replay_prob == 0.5
    assert args.checkpoint == "selected.pth"
    assert args.correction_dataset == "correction_examples.npz"
    assert args.correction_datasets == "a.npz,b.npz"
    assert args.correction_action_weight == 0.04
    assert args.policy_anchor_weight == 0.02
    assert args.policy_anchor_temperature == 1.5
    assert args.policy_anchor_min_distance_tiles == 3.0
    assert args.contact_action_weight == 0.03
    assert args.contact_action_batch_size == 32
    assert args.contact_action_distance == 2.5
    assert args.contact_head_offline_steps == 123
    assert args.contact_head_learning_rate == 0.002
    assert args.contact_head_confidence == 0.8
    assert args.contact_head_jump_confidence == 0.85
    assert args.contact_head_balance_classes is True
    assert args.contact_head_calibration_frac == 0.2
    assert args.contact_head_calibration_seed == 19
    assert args.contact_head_min_calibration_accuracy == 0.72
    assert args.contact_head_min_class_examples == 12
    assert args.contact_label_state_decimals == 2
    assert args.contact_label_adjacent_step_window == 3
    assert args.contact_label_top_groups == 7
    assert args.contact_label_filter_majority_threshold == 0.75
    assert args.final_contact_distance == 2.5
    assert args.final_contact_commit_steps == 6
    assert args.final_contact_cancel_outside is True
    assert args.final_contact_policy_advantage_gate is True
    assert args.final_contact_min_option_advantage == 250.0
    assert args.save_selected_checkpoint is True
    assert args.no_artifact_validation is True


def test_eval_final_contact_option_cli_uses_requested_label(tmp_path, monkeypatch):
    """The outer session label should also name the inner run summary."""
    captured: dict[str, object] = {}

    def fake_run_eval_final_contact_option(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return {"label": kwargs["label"], "train_seconds": 0.0}

    monkeypatch.setattr(
        status_cli,
        "run_eval_final_contact_option",
        fake_run_eval_final_contact_option,
    )

    parser = argparse.ArgumentParser()
    opts = argparse.Namespace(
        mode="eval-final-contact-option",
        checkpoint=str(tmp_path / "selected.pth"),
        seed=3,
        eval_games=5,
        log_every=1,
        report_seconds=1.0,
        final_contact_distance=1.5,
        final_contact_commit_steps=4,
        final_contact_cancel_outside=False,
        final_contact_policy_advantage_gate=True,
        final_contact_min_option_advantage=250.0,
        label="narrow_option_eval",
    )
    payload = {"runs": []}

    handled = status_cli._run_checkpoint_correction_mode(
        parser,
        opts,
        tmp_path,
        payload,
    )

    assert handled is True
    assert captured["label"] == "narrow_option_eval"
    assert captured["cancel_option_outside_close_zone"] is False
    assert captured["gate_policy_advantage"] is True
    assert captured["min_option_advantage"] == 250.0
    assert payload["runs"] == [{"label": "narrow_option_eval", "train_seconds": 0.0}]


def test_collect_corrections_cli_uses_requested_label(tmp_path, monkeypatch):
    """Dataset runs should preserve labels in the inner run summary."""
    captured: dict[str, object] = {}

    def fake_run_collect_corrections(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return {"label": kwargs["label"], "train_seconds": 0.0}

    monkeypatch.setattr(status_cli, "run_collect_corrections", fake_run_collect_corrections)

    parser = argparse.ArgumentParser()
    opts = argparse.Namespace(
        mode="collect-corrections",
        checkpoint=str(tmp_path / "selected.pth"),
        seed=3,
        correction_games=5,
        correction_max_steps=1200,
        correction_max_examples=512,
        correction_sample_every=4,
        correction_max_examples_per_game=64,
        correction_stale_steps=999999,
        correction_loop_tile_visits=999999,
        correction_keep_agreements=False,
        final_contact_distance=3.0,
        final_contact_commit_steps=8,
        final_contact_policy_advantage_gate=True,
        final_contact_min_option_advantage=250.0,
        log_every=1,
        report_seconds=1.0,
        label="contact_only_collect",
    )
    payload = {"runs": []}

    handled = status_cli._run_checkpoint_correction_mode(
        parser,
        opts,
        tmp_path,
        payload,
    )

    assert handled is True
    assert captured["label"] == "contact_only_collect"
    assert captured["correction_label_mode"] == "advantage_gate"
    assert captured["final_contact_min_option_advantage"] == 250.0
    assert payload["runs"] == [{"label": "contact_only_collect", "train_seconds": 0.0}]


def test_collect_contact_head_corrections_cli_uses_contact_head_rollout(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_collect_contact_head_corrections(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return {"label": kwargs["label"], "train_seconds": 0.0}

    monkeypatch.setattr(
        status_cli,
        "run_collect_contact_head_corrections",
        fake_run_collect_contact_head_corrections,
    )

    parser = argparse.ArgumentParser()
    opts = argparse.Namespace(
        mode="collect-contact-head-corrections",
        checkpoint=str(tmp_path / "selected.pth"),
        correction_dataset=str(tmp_path / "contact_labels.npz"),
        seed=1,
        correction_games=5,
        correction_max_steps=1200,
        correction_max_examples=512,
        correction_sample_every=1,
        correction_max_examples_per_game=12,
        correction_stale_steps=999999,
        correction_loop_tile_visits=999999,
        correction_keep_agreements=False,
        final_contact_distance=3.0,
        final_contact_commit_steps=8,
        final_contact_policy_advantage_gate=True,
        final_contact_min_option_advantage=250.0,
        contact_action_batch_size=32,
        contact_action_distance=3.0,
        contact_head_offline_steps=500,
        contact_head_learning_rate=0.001,
        contact_head_confidence=0.75,
        contact_head_jump_confidence=0.0,
        contact_head_action_thresholds="LEFT_JUMP:0.90",
        contact_head_balance_classes=True,
        log_every=1,
        report_seconds=1.0,
        label="b25_collect",
    )
    payload = {"runs": []}

    handled = status_cli._run_checkpoint_correction_mode(
        parser,
        opts,
        tmp_path,
        payload,
    )

    assert handled is True
    assert captured["label"] == "b25_collect"
    assert captured["correction_label_mode"] == "advantage_gate"
    assert captured["contact_head_action_thresholds"] == {"LEFT_JUMP": 0.9}
    assert captured["contact_head_balance_classes"] is True
    assert payload["runs"] == [{"label": "b25_collect", "train_seconds": 0.0}]


def test_contact_head_calibrate_cli_uses_dataset_list(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_contact_head_calibration(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return {"label": kwargs["label"], "train_seconds": 0.0}

    monkeypatch.setattr(
        status_cli,
        "run_contact_head_calibration",
        fake_run_contact_head_calibration,
    )

    parser = argparse.ArgumentParser()
    opts = argparse.Namespace(
        mode="contact-head-calibrate",
        checkpoint=str(tmp_path / "selected.pth"),
        correction_datasets=f"{tmp_path / 'a.npz'},{tmp_path / 'b.npz'}",
        seed=3,
        log_every=1,
        report_seconds=1.0,
        contact_action_batch_size=32,
        contact_action_distance=3.0,
        contact_head_offline_steps=123,
        contact_head_learning_rate=0.002,
        contact_head_balance_classes=True,
        contact_head_calibration_frac=0.2,
        contact_head_calibration_seed=19,
        contact_head_min_calibration_accuracy=0.72,
        contact_head_min_class_examples=12,
        label="combined_calibration",
    )
    payload = {"runs": []}

    handled = status_cli._run_checkpoint_correction_mode(
        parser,
        opts,
        tmp_path,
        payload,
    )

    assert handled is True
    assert captured["label"] == "combined_calibration"
    assert [path.name for path in captured["correction_dataset_paths"]] == ["a.npz", "b.npz"]
    assert captured["calibration_frac"] == 0.2
    assert captured["min_calibration_accuracy"] == 0.72
    assert captured["min_class_examples"] == 12
    assert payload["runs"] == [{"label": "combined_calibration", "train_seconds": 0.0}]


def test_contact_label_audit_cli_uses_dataset_list(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_contact_label_audit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return {"label": kwargs["label"], "train_seconds": 0.0}

    monkeypatch.setattr(label_modes, "run_contact_label_audit", fake_run_contact_label_audit)

    parser = argparse.ArgumentParser()
    opts = argparse.Namespace(
        mode="contact-label-audit",
        correction_datasets=f"{tmp_path / 'a.npz'},{tmp_path / 'b.npz'}",
        contact_label_state_decimals=2,
        contact_label_adjacent_step_window=3,
        contact_label_top_groups=7,
        label="label_audit",
    )
    payload = {"runs": []}

    handled = label_modes.run_label_dataset_mode(
        parser,
        opts,
        tmp_path,
        payload,
    )

    assert handled is True
    assert captured["label"] == "label_audit"
    assert [path.name for path in captured["correction_dataset_paths"]] == ["a.npz", "b.npz"]
    assert captured["state_round_decimals"] == 2
    assert captured["adjacent_step_window"] == 3
    assert captured["top_groups"] == 7
    assert payload["runs"] == [{"label": "label_audit", "train_seconds": 0.0}]


def test_contact_label_filter_cli_uses_dataset_list(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_contact_label_filter(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.update(kwargs)
        return {"label": kwargs["label"], "train_seconds": 0.0}

    monkeypatch.setattr(label_modes, "run_contact_label_filter", fake_run_contact_label_filter)

    parser = argparse.ArgumentParser()
    opts = argparse.Namespace(
        mode="contact-label-filter",
        correction_datasets=f"{tmp_path / 'a.npz'},{tmp_path / 'b.npz'}",
        contact_label_filter_majority_threshold=0.75,
        contact_label_adjacent_step_window=3,
        label="label_filter",
    )
    payload = {"runs": []}

    handled = label_modes.run_label_dataset_mode(
        parser,
        opts,
        tmp_path,
        payload,
    )

    assert handled is True
    assert captured["label"] == "label_filter"
    assert [path.name for path in captured["correction_dataset_paths"]] == ["a.npz", "b.npz"]
    assert captured["semantic_majority_threshold"] == 0.75
    assert captured["adjacent_step_window"] == 3
    assert payload["runs"] == [{"label": "label_filter", "train_seconds": 0.0}]


def test_final_contact_option_can_cancel_queue_outside_close_zone(tmp_path):
    """Committed local macros should be cancellable when the target is no longer close."""

    class FixedAgent:
        def __init__(self, action: int) -> None:
            self.action = action

        def select_action(self, state, training=False):  # noqa: ANN001, ANN202
            return self.action

    cfg = first_crystal_config(
        tmp_path / "cancel-option",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    game = CrystalCaves(cfg, headless=True)
    state = game.reset()
    info = game._info()
    queued_actions = deque([game.LEFT, game.LEFT_JUMP])

    action, decision = final_contact_option_action(
        FixedAgent(game.RIGHT),
        state,
        game,
        info,
        action_labels=list(game.ACTION_LABELS),
        close_zone_distance_tiles=0.01,
        option_queue=queued_actions,
        cancel_option_outside_close_zone=True,
    )

    assert action == game.RIGHT
    assert not queued_actions
    assert decision["source"] == "policy"
    assert decision["option_meta"]["cancelled_committed_actions"] == 2


def test_final_contact_option_advantage_gate_can_keep_policy_action(tmp_path):
    """The option should not take over when its simulated advantage is below the gate."""

    class ShootAgent:
        def select_action(self, state, training=False):  # noqa: ANN001, ANN202
            return 6

    cfg = first_crystal_config(
        tmp_path / "advantage-gate",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    game = CrystalCaves(cfg, headless=True)
    state = game.reset()
    info = game._info()

    action, decision = final_contact_option_action(
        ShootAgent(),
        state,
        game,
        info,
        action_labels=list(game.ACTION_LABELS),
        close_zone_distance_tiles=99.0,
        final_contact_commit_steps=2,
        gate_policy_advantage=True,
        min_option_advantage=1_000_000.0,
    )

    assert action == game.SHOOT
    assert decision["source"] == "policy"
    assert decision["option_meta"]["gate_policy_advantage"] is True
    assert decision["option_meta"]["rejected_by_policy_gate"] is True
    assert decision["option_meta"]["gate_min_option_advantage"] == 1_000_000.0


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


def test_interleaved_vec_env_mixes_full_and_contact_sources(tmp_path):
    """Contact lanes should reuse the generic source-stats machinery."""
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
    contact_source_config = make_interleaved_contact_config(full_config)

    vec = InterleavedCrystalCavesVec(
        full_config=full_config,
        skill_config=contact_source_config,
        full_envs=6,
        skill_envs=2,
        skill_source="contact",
        headless=True,
    )
    try:
        states = vec.reset()
        assert states.shape == (8, vec.state_size)
        assert vec.sources == ["full"] * 6 + ["contact"] * 2
        assert vec.envs[6].CAVES == CONTACT_CAVES

        vec.envs[6].game_over = True
        vec.envs[6].exit_unlocked = True
        _, _, dones, infos = vec.step_no_copy(np.zeros(vec.num_envs, dtype=np.int64))

        assert dones.tolist()[6] is True
        assert infos[6]["training_source"] == "contact"
        stats = vec.source_stats()
        assert stats["contact"]["episodes"] == 1
        assert stats["contact"]["exit_rate_100"] == 1.0
        assert "crystal_rate_100" in stats["contact"]
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


def test_apply_reverse_exit_far_start_is_distant_and_reachable(tmp_path):
    """The FAR reverse-exit start clears+unlocks like the near one, but drops the agent
    a real distance from the exit (long-range navigation) with the exit still oracle-
    reachable — so the leg-2 probe measures routing, not the trivial final hop."""
    cfg = full_tutorial_config(
        tmp_path / "reverse-exit-far",
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
        assert apply_reverse_start(game, "reverse_exit_far") is True
        assert game.exit_unlocked is True
        assert len(game.crystals) == 0
        # Same post-collection world state, but a genuinely distant, reachable start.
        col, row = game._player_tile()
        assert game.exit_pos in game._oracle_reachable((col, row))
        exit_col, exit_row = game.exit_pos
        assert abs(col - exit_col) + abs(row - exit_row) >= 4
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


def test_contact_config_enables_contact_levels_without_procedural_drills_or_bridges(tmp_path):
    """Contact runs should load only the compact final-objective contact cave set."""
    cfg = contact_config(
        tmp_path / "contact",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    assert cfg.CRYSTAL_CAVES_CONTACT_LEVELS is True
    assert cfg.CRYSTAL_CAVES_CONTACT_POOL_SIZE == 0
    assert cfg.CRYSTAL_CAVES_BRIDGES is False
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


def test_apply_policy_anchor_override_is_opt_in(tmp_path):
    """Policy anchoring should stay disabled unless weight is positive."""
    cfg = first_crystal_config(
        tmp_path / "policy-anchor",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    apply_policy_anchor_override(
        cfg,
        policy_anchor_weight=0.0,
        policy_anchor_temperature=1.0,
        policy_anchor_min_distance_tiles=3.0,
    )
    assert cfg.CRYSTAL_CAVES_POLICY_ANCHOR_LOSS is False

    apply_policy_anchor_override(
        cfg,
        policy_anchor_weight=0.02,
        policy_anchor_temperature=1.5,
        policy_anchor_min_distance_tiles=3.0,
    )
    assert cfg.CRYSTAL_CAVES_POLICY_ANCHOR_LOSS is True
    assert cfg.CRYSTAL_CAVES_POLICY_ANCHOR_WEIGHT == 0.02
    assert cfg.CRYSTAL_CAVES_POLICY_ANCHOR_TEMPERATURE == 1.5
    assert cfg.CRYSTAL_CAVES_POLICY_ANCHOR_MIN_TARGET_DISTANCE_NORM > 0.0

    with pytest.raises(ValueError, match="temperature"):
        apply_policy_anchor_override(
            cfg,
            policy_anchor_weight=0.02,
            policy_anchor_temperature=0.0,
        )

    with pytest.raises(ValueError, match="distance"):
        apply_policy_anchor_override(
            cfg,
            policy_anchor_weight=0.02,
            policy_anchor_temperature=1.0,
            policy_anchor_min_distance_tiles=-1.0,
        )


def test_apply_distributional_dqn_override_is_opt_in(tmp_path):
    cfg = first_crystal_config(
        tmp_path / "c51",
        episodes=4,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    apply_distributional_dqn_override(
        cfg,
        distributional_dqn=True,
        c51_atoms=51,
        c51_v_min=-20.0,
        c51_v_max=120.0,
    )
    snapshot = config_snapshot(cfg)

    assert cfg.USE_DISTRIBUTIONAL_DQN is True
    assert snapshot["use_distributional_dqn"] is True
    assert snapshot["c51_num_atoms"] == 51
    assert snapshot["c51_v_min"] == -20.0
    assert snapshot["c51_v_max"] == 120.0

    with pytest.raises(ValueError, match="c51_atoms"):
        apply_distributional_dqn_override(
            cfg,
            distributional_dqn=True,
            c51_atoms=1,
            c51_v_min=-20.0,
            c51_v_max=120.0,
        )
    with pytest.raises(ValueError, match="c51_v_min"):
        apply_distributional_dqn_override(
            cfg,
            distributional_dqn=True,
            c51_atoms=51,
            c51_v_min=120.0,
            c51_v_max=120.0,
        )


def test_apply_contact_action_head_override_is_opt_in(tmp_path):
    config = full_tutorial_config(
        tmp_path / "contact-head",
        episodes=10,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )

    assert not config.CRYSTAL_CAVES_CONTACT_ACTION_HEAD

    apply_contact_action_head_override(
        config,
        contact_action_weight=0.02,
        contact_action_batch_size=32,
        contact_action_distance_tiles=3.0,
    )

    assert config.CRYSTAL_CAVES_CONTACT_ACTION_HEAD
    assert config.CRYSTAL_CAVES_CONTACT_ACTION_WEIGHT == 0.02
    assert config.CRYSTAL_CAVES_CONTACT_ACTION_BATCH_SIZE == 32
    assert config.CRYSTAL_CAVES_CONTACT_ACTION_DISTANCE_NORM > 0.0

    with pytest.raises(ValueError, match="contact_action_weight"):
        apply_contact_action_head_override(
            config,
            contact_action_weight=-0.01,
            contact_action_batch_size=32,
            contact_action_distance_tiles=3.0,
        )
    with pytest.raises(ValueError, match="batch_size"):
        apply_contact_action_head_override(
            config,
            contact_action_weight=0.02,
            contact_action_batch_size=0,
            contact_action_distance_tiles=3.0,
        )
    with pytest.raises(ValueError, match="distance"):
        apply_contact_action_head_override(
            config,
            contact_action_weight=0.02,
            contact_action_batch_size=32,
            contact_action_distance_tiles=0.0,
        )


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


def test_live_status_line_shows_enabled_policy_anchor():
    """Anchor live status should show active teacher matching when enabled."""
    line = live_status_line(
        {
            "label": "anchored_correction",
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
            "policy_anchor_enabled": True,
            "policy_anchor_samples_100": 0,
            "avg_policy_anchor_loss_100": 0.0,
            "avg_policy_anchor_accuracy_100": 1.0,
        }
    )

    assert "| anchor 0.000/100% n=0" in line


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
