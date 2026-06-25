"""Report and diagnostic tests for Crystal Caves status sessions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cc_status_session import (  # noqa: E402
    classify_trace_failure,
    first_crystal_config,
    first_objective_near_miss_eval,
    first_objective_near_miss_rollup,
    trace_rollup,
    trainer_archive_stats,
    trainer_reverse_start_stats,
    trainer_source_stats,
    write_markdown_report,
)


class _StatsVecEnv:
    def source_stats(self):
        return {"full": {"episodes": 2}}

    def reverse_start_stats(self):
        return {"reverse_exit": {"applied": 1}}

    def archive_stats(self):
        return {"archive": {"replayed": 3}}


class _StatsTrainer:
    vec_env = _StatsVecEnv()


def test_trainer_stats_helpers_are_available_to_reports():
    trainer = _StatsTrainer()

    assert trainer_source_stats(trainer) == {"full": {"episodes": 2}}
    assert trainer_reverse_start_stats(trainer) == {"reverse_exit": {"applied": 1}}
    assert trainer_archive_stats(trainer) == {"archive": {"replayed": 3}}


def test_markdown_report_includes_bridge_eval_history(tmp_path):
    """Reports should show milestone bridge diagnostics, not just final rows."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "bridge_pretrain",
                    "episodes": 50,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "bridge_eval_history": [
                        {
                            "episode": 50,
                            "rollup": {
                                "mean_win_rate": 0.2,
                                "mean_any_crystal_rate": 0.4,
                                "mean_all_crystals_rate": 0.2,
                                "mean_progress": 0.33,
                                "solved_levels": 1,
                                "levels": 5,
                            },
                        }
                    ],
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "| Bridge Eval Ep | Win | Any crystal | All crystals | Progress | Solved |" in report
    assert "| 50 | 20% | 40% | 20% | 0.330 | 1/5 |" in report


def test_markdown_report_includes_selected_bridge_policy(tmp_path):
    """Bridge-transfer reports should identify which bridge checkpoint was used."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "bridge_pretrain",
                    "episodes": 400,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "selected_bridge_episode": 350,
                    "selected_bridge_rollup": {
                        "mean_win_rate": 0.6,
                        "solved_levels": 3,
                        "levels": 5,
                    },
                },
                {
                    "label": "bridge_transfer_full",
                    "episodes": 300,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "transfer_source": {
                        "kind": "bridge",
                        "selected_bridge_episode": 350,
                        "selected_bridge_rollup": {"mean_win_rate": 0.6},
                    },
                },
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Selected bridge policy: ep 350 (60% win, 3/5 solved)" in report
    assert "- Transfer source: bridge ep 350 (60% bridge win)" in report


def test_markdown_report_marks_first_crystal_transfer_source(tmp_path):
    """First-crystal transfer reports should distinguish source wins from real wins."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "first_crystal_pretrain",
                    "episodes": 100,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "config": {"first_crystal_goal": True},
                    "selected_source_episode": 100,
                    "selected_source_eval": {
                        "win_rate": 0.75,
                        "mean_crystal_frac": 0.75,
                        "mean_depth_frac": 0.25,
                    },
                    "final_eval": {
                        "wins": 6,
                        "num_games": 8,
                        "win_rate": 0.75,
                        "mean_crystal_frac": 0.75,
                        "mean_depth_frac": 0.25,
                        "mean_score": 100.0,
                        "end_reason_counts": {"first_crystal_goal": 6, "timeout": 2},
                    },
                },
                {
                    "label": "first_crystal_transfer_full",
                    "episodes": 150,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "transfer_source": {
                        "kind": "first_crystal",
                        "source_episode": 100,
                        "source_win_rate": 0.75,
                        "source_wins": 6,
                        "source_games": 8,
                    },
                },
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Objective mode: first crystal terminal success" in report
    assert "- Selected source policy: ep 100 (75% win, 75% crystals, 25% depth)" in report
    assert (
        "- Transfer source: first-crystal route policy ep 100 (75% source held-out win)" in report
    )


def test_markdown_report_includes_route_curriculum_source(tmp_path):
    """First-crystal route reports should show the training-only route floor."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "first_crystal_route",
                    "episodes": 150,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "config": {"first_crystal_goal": True},
                    "route_curriculum": {
                        "route_floor_episodes": 75,
                        "route_scaffold_difficulty": "route_catch",
                        "tutorial_route_episodes": 150,
                        "route_floor_source_win_rate": 0.875,
                        "route_floor_source_wins": 7,
                        "route_floor_source_games": 8,
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Objective mode: first crystal terminal success" in report
    assert (
        "- Route curriculum: 75 route_catch episodes -> 150 tutorial route episodes "
        "(floor source 88% win, 7/8)"
    ) in report


def test_markdown_report_includes_direct_route_training(tmp_path):
    """Direct first-crystal reports should identify the matching training pool."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "first_crystal_direct",
                    "episodes": 300,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "config": {"first_crystal_goal": True},
                    "selected_checkpoint_eval": {
                        "wins": 6,
                        "num_games": 30,
                        "win_rate": 0.2,
                        "mean_crystal_frac": 0.2,
                        "mean_depth_frac": 0.31,
                        "end_reason_counts": {"first_crystal_goal": 6, "timeout": 24},
                    },
                    "selected_checkpoint_failure_diagnostics": {
                        "trace_dir": "selected_trace",
                        "rollup": {
                            "wins": 1,
                            "games": 4,
                            "any_crystal_rate": 0.25,
                            "mean_depth_frac": 0.19,
                            "failure_mode_counts": {"tile_loop": 3},
                        },
                    },
                    "route_direct": {"difficulty": "tutorial", "cave_pool_size": 512},
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Objective mode: first crystal terminal success" in report
    assert "- Direct route training: tutorial, pool 512" in report
    assert (
        "- Selected checkpoint expanded eval: 6/30 wins (20.0%), " "20.0% crystals, 31.0% depth"
    ) in report
    assert "- Selected checkpoint trace: 1/4 wins, 25.0% any crystal, 19.0% depth" in report
    assert "- Selected checkpoint trace files: `selected_trace`" in report


def test_markdown_report_includes_correction_dataset(tmp_path):
    """Correction collection runs should be readable from the markdown report."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "collect_corrections",
                    "episodes": 0,
                    "device": "cpu",
                    "steps_per_second": 0.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "correction_dataset": {
                        "kept_examples": 12,
                        "games_completed": 3,
                        "disagreement_rate": 0.75,
                        "trigger_counts": {"close_zone": 8, "stale": 4},
                        "label_action_counts": {"RIGHT_JUMP": 6, "LEFT": 6},
                        "states_path": "/tmp/corrections.npz",
                        "rows_path": "/tmp/corrections.jsonl",
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert (
        "- Correction dataset: 12 kept states from 3 games (75.0% policy/label disagreement)"
        in report
    )
    assert "- Correction triggers: `{'close_zone': 8, 'stale': 4}`" in report
    assert "- Correction arrays: `/tmp/corrections.npz`" in report


def test_markdown_report_includes_correction_training(tmp_path):
    """Correction fine-tune runs should show supervision strength and live metrics."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "correction_finetune",
                    "episodes": 10,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "avg_correction_action_loss_100": 0.1234,
                    "avg_correction_action_accuracy_100": 0.75,
                    "correction_action_samples_100": 5,
                    "correction_training": {
                        "dataset_path": "/tmp/correction_examples.npz",
                        "correction_action_transitions": 12,
                        "weight": 0.02,
                        "margin": 0.6,
                        "batch_size": 8,
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert (
        "- Correction action supervision: 12 states, weight 0.020, margin 0.60, "
        "batch 8, dataset `/tmp/correction_examples.npz`" in report
    )
    assert "- Correction action metrics avg100: loss 0.1234, accuracy 75.0%, samples 5" in report


def test_markdown_report_includes_demo_replay_source(tmp_path):
    """Demo replay reports should show collection and replay-seeding counts."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "bridge_demo_replay_full",
                    "episodes": 50,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "demo_replay": {
                        "kind": "bridge",
                        "selected_bridge_episode": 200,
                        "selected_bridge_rollup": {"mean_win_rate": 0.4},
                        "demo_summary": {
                            "kept_trajectories": 6,
                            "kept_transitions": 1200,
                            "wins": 6,
                            "attempts": 20,
                        },
                        "seeded": {
                            "pushed_transitions": 4800,
                            "repeat": 4,
                            "memory_size_after_seed": 4795,
                        },
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Demo replay source: bridge ep 200 (40% bridge win)" in report
    assert (
        "- Demo replay collected: 6 trajectories, 1200 transitions from 6/20 source wins" in report
    )
    assert "- Demo replay seeded: 4800 pushed transitions (4x repeat), memory size 4795" in report


def test_markdown_report_includes_checkpoint_eval_source(tmp_path):
    """Checkpoint re-eval reports should identify the selected source policy."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "eval_checkpoint",
                    "episodes": 0,
                    "device": "cpu",
                    "steps_per_second": 0.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "checkpoint_eval": {
                        "source_label": "tutorial_demo_bc",
                        "source_episode": 300,
                        "source_eval": {"wins": 3, "num_games": 16},
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Checkpoint eval source: tutorial_demo_bc ep 300 (3/16 source wins)" in report


def test_markdown_report_includes_route_demo_bc_source(tmp_path):
    """Route-demo BC reports should show scripted, supervised, and replay counts."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "route_demo_floor",
                    "episodes": 75,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "selected_checkpoint_path": "models/tutorial_demo_bc_selected_ep100.pth",
                    "route_demo": {
                        "demo_summary": {
                            "attempts": 32,
                            "win_rate": 0.5,
                            "kept_trajectories": 16,
                            "kept_transitions": 4200,
                            "close_zone_kept_transitions": 640,
                            "oracle_close_zone_kept_transitions": 640,
                            "mean_kept_oracle_close_zone_relabel_rate": 0.25,
                            "oracle_close_zone_action_counts": {"RIGHT_JUMP": 320, "RIGHT": 320},
                        },
                        "demo_selection": {
                            "mode": "filtered-weighted",
                            "input_trajectories": 16,
                            "selected_unique_trajectories": 14,
                            "selected_weighted_trajectories": 24,
                            "selected_transitions": 5200,
                            "excluded_reasons": {"beam_quality_filter": 2},
                        },
                        "behavior_cloning": {
                            "updates": 198,
                            "transitions": 4200,
                            "final_loss": 0.1234,
                        },
                        "seeded": {
                            "pushed_transitions": 16800,
                            "repeat": 4,
                            "memory_size_after_seed": 16790,
                        },
                        "after_bc_eval": {
                            "wins": 3,
                            "num_games": 8,
                            "win_rate": 0.375,
                            "mean_crystal_frac": 0.375,
                        },
                        "online_action_supervision": {
                            "enabled": True,
                            "demo_action_source": "close_zone",
                            "close_zone_distance_tiles": 3.0,
                            "close_zone_available_transitions": 640,
                            "weight": 0.12,
                            "margin": 0.8,
                            "conservative_weight": 0.02,
                            "conservative_temperature": 0.7,
                            "batch_size": 64,
                            "demo_action_transitions": 640,
                            "close_zone_extra_enabled": True,
                            "close_zone_extra_weight": 0.03,
                            "close_zone_extra_batch_size": 64,
                            "close_zone_extra_source": "scripted",
                            "close_zone_extra_transitions": 640,
                        },
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert (
        "- Route demos collected: 16 kept / 32 attempts (50% scripted win), 4200 transitions"
        in report
    )
    assert (
        "- Route demo selection: filtered-weighted, 14/16 unique kept, "
        "24 weighted trajectories, 5200 training transitions, excluded "
        "`{'beam_quality_filter': 2}`" in report
    )
    assert (
        "- Route demo oracle close-zone labels: 640 transitions, relabel rate 25.0%, "
        "actions `{'RIGHT_JUMP': 320, 'RIGHT': 320}`" in report
    )
    assert "- Behavior cloning: 198 updates over 4200 transitions, final CE 0.1234" in report
    assert (
        "- Route demo replay seeded: 16800 pushed transitions (4x repeat), memory size 16790"
        in report
    )
    assert "- After-BC source eval: 3/8 wins (38%), 38% crystals" in report
    assert (
        "- Online demo action supervision: close_zone, <= 3.0 tiles, 640 available, "
        "weight 0.120, margin 0.80, conservative 0.020 @T 0.70, batch 64, "
        "640 active transitions" in report
    )
    assert (
        "- Close-zone extra action supervision: scripted, weight 0.030, batch 64, "
        "640 active transitions within <= 3.0 tiles" in report
    )
    assert "- Selected checkpoint path: `models/tutorial_demo_bc_selected_ep100.pth`" in report


def test_markdown_report_includes_reverse_start_mix(tmp_path):
    """Reverse-start reports should show lane mix and spawn application counts."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "reverse_start",
                    "episodes": 50,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "reverse_start": {
                        "full_envs": 6,
                        "reverse_envs": 2,
                        "reverse_ratio": 0.25,
                        "modes": {"reverse_objective": 1, "reverse_exit": 1},
                    },
                    "reverse_start_stats": {
                        "reverse_objective": {"attempts": 12, "applied": 12},
                        "reverse_exit": {"attempts": 10, "applied": 9},
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Reverse-start mix: 6 full / 2 reverse envs (25% reverse lanes)" in report
    assert "- Reverse-start modes: `{'reverse_objective': 1, 'reverse_exit': 1}`" in report
    assert "- Reverse-start applied: reverse_exit: 9/10, reverse_objective: 12/12" in report


def test_markdown_report_includes_archive_start_mix(tmp_path):
    """Archive-start reports should show lane mix and archive replay counts."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "archive_start",
                    "episodes": 50,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "archive_start": {
                        "full_envs": 6,
                        "archive_envs": 2,
                        "archive_ratio": 0.25,
                        "replay_prob": 0.7,
                        "max_size": 64,
                        "min_steps": 30,
                    },
                    "archive_stats": {
                        "size": 12,
                        "max_size": 64,
                        "stores": 14,
                        "replay_attempts": 20,
                        "replays": 9,
                        "replay_rate": 0.45,
                        "seen_milestones": 14,
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Archive-start mix: 6 full / 2 archive envs (25% archive lanes)" in report
    assert "- Archive-start replay: 70% reset probability, max 64 states, min 30 steps" in report
    assert (
        "- Archive-start stats: size 12/64, stores 14, replays 9/20 (45%), seen milestones 14"
        in report
    )


def test_trace_rollup_counts_failure_modes():
    """Held-out traces should produce an aggregate view of why games failed."""
    rows = [
        {
            "won": False,
            "crystals_collected": 0,
            "exit_unlocked": False,
            "final_progress": 0.2,
            "final_depth_frac": 0.4,
            "unique_tiles": 12,
            "max_tile_visit_frac": 0.25,
            "idle_action_frac": 0.1,
            "mean_q_margin": 0.5,
            "target_distance_delta_tiles": 2.0,
            "target_distance_best_delta_tiles": 4.0,
            "anti_loop_penalty_total": -1.5,
            "interact_action_frac": 0.2,
            "invalid_interact_count": 5,
            "invalid_interact_penalty_total": -0.25,
            "novelty_bonus_total": 0.4,
            "end_reason": "timeout",
            "failure_modes": ["no_crystal", "tile_loop"],
        },
        {
            "won": False,
            "crystals_collected": 1,
            "exit_unlocked": False,
            "final_progress": 0.5,
            "final_depth_frac": 0.6,
            "unique_tiles": 30,
            "max_tile_visit_frac": 0.05,
            "idle_action_frac": 0.4,
            "mean_q_margin": 0.2,
            "target_distance_delta_tiles": -1.0,
            "target_distance_best_delta_tiles": 3.0,
            "anti_loop_penalty_total": -0.5,
            "interact_action_frac": 0.0,
            "invalid_interact_count": 0,
            "invalid_interact_penalty_total": 0.0,
            "novelty_bonus_total": 0.2,
            "end_reason": "stalled",
            "failure_modes": ["partial_crystals", "idle_heavy"],
        },
    ]

    rollup = trace_rollup(rows)

    assert rollup["games"] == 2
    assert rollup["wins"] == 0
    assert rollup["any_crystal_rate"] == 0.5
    assert rollup["mean_target_distance_delta_tiles"] == 0.5
    assert rollup["mean_target_distance_best_delta_tiles"] == 3.5
    assert rollup["mean_anti_loop_penalty_total"] == -1.0
    assert rollup["mean_interact_action_frac"] == 0.1
    assert rollup["mean_invalid_interact_count"] == 2.5
    assert rollup["mean_invalid_interact_penalty_total"] == -0.125
    assert rollup["mean_novelty_bonus_total"] == pytest.approx(0.3)
    assert rollup["failure_mode_counts"] == {
        "no_crystal": 1,
        "tile_loop": 1,
        "partial_crystals": 1,
        "idle_heavy": 1,
    }


def test_first_objective_near_miss_rollup_counts_distance_bands():
    """Near-miss rollups should expose progress that win/crystal counts hide."""
    rows = [
        {
            "won": False,
            "crystals_collected": 0,
            "final_progress": 0.2,
            "final_depth_frac": 0.4,
            "initial_target_distance_tiles": 10.0,
            "min_target_distance_tiles": 2.5,
            "final_target_distance_tiles": 4.0,
            "target_distance_best_delta_tiles": 7.5,
            "target_distance_final_delta_tiles": 6.0,
            "step_of_best_approach": 100,
            "stuck_after_close": True,
            "loop_after_close": True,
            "close_zone_steps": 20,
            "close_zone_jump_rate": 0.25,
            "close_zone_idle_or_interact_rate": 0.5,
            "end_reason": "stalled",
        },
        {
            "won": True,
            "crystals_collected": 1,
            "final_progress": 1.0,
            "final_depth_frac": 0.8,
            "initial_target_distance_tiles": 12.0,
            "min_target_distance_tiles": 1.0,
            "final_target_distance_tiles": 0.0,
            "target_distance_best_delta_tiles": 11.0,
            "target_distance_final_delta_tiles": 12.0,
            "step_of_best_approach": 50,
            "stuck_after_close": False,
            "loop_after_close": False,
            "close_zone_steps": 5,
            "close_zone_jump_rate": 0.6,
            "close_zone_idle_or_interact_rate": 0.0,
            "end_reason": "first_crystal_goal",
        },
        {
            "won": False,
            "crystals_collected": 0,
            "final_progress": 0.1,
            "final_depth_frac": 0.2,
            "initial_target_distance_tiles": 11.0,
            "min_target_distance_tiles": 6.0,
            "final_target_distance_tiles": 9.0,
            "target_distance_best_delta_tiles": 5.0,
            "target_distance_final_delta_tiles": 2.0,
            "step_of_best_approach": 25,
            "stuck_after_close": False,
            "loop_after_close": False,
            "close_zone_steps": 0,
            "close_zone_jump_rate": 0.0,
            "close_zone_idle_or_interact_rate": 0.0,
            "end_reason": "timeout",
        },
    ]

    rollup = first_objective_near_miss_rollup(rows)

    assert rollup["games"] == 3
    assert rollup["wins"] == 1
    assert rollup["any_crystal_rate"] == pytest.approx(1 / 3)
    assert rollup["near_miss_rate_10"] == 1.0
    assert rollup["near_miss_rate_5"] == pytest.approx(2 / 3)
    assert rollup["near_miss_rate_3"] == pytest.approx(2 / 3)
    assert rollup["near_miss_rate_1_5"] == pytest.approx(1 / 3)
    assert rollup["mean_min_target_distance_tiles"] == pytest.approx((2.5 + 1.0 + 6.0) / 3)
    assert rollup["mean_target_distance_best_delta_tiles"] == pytest.approx((7.5 + 11.0 + 5.0) / 3)
    assert rollup["stuck_after_close_rate"] == pytest.approx(1 / 3)
    assert rollup["loop_after_close_rate"] == pytest.approx(1 / 3)
    assert rollup["mean_close_zone_jump_rate"] == pytest.approx((0.25 + 0.6) / 3)
    assert rollup["end_reason_counts"] == {
        "stalled": 1,
        "first_crystal_goal": 1,
        "timeout": 1,
    }


def test_first_objective_near_miss_eval_writes_per_level_rows(tmp_path):
    """The detailed eval should write a compact per-level JSONL matrix."""

    class IdleAgent:
        def __init__(self) -> None:
            self.epsilon = 0.0

        def select_action(self, state, training=False):  # noqa: ANN001, ANN202
            return 0

    cfg = first_crystal_config(
        tmp_path / "run",
        episodes=1,
        seed=0,
        eval_every=0,
        train_eval_games=0,
        log_every=1,
        report_seconds=1.0,
    )
    cfg.CRYSTAL_CAVES_PROCEDURAL = False
    cfg.EVAL_MAX_STEPS = 5

    summary = first_objective_near_miss_eval(
        cfg,
        IdleAgent(),
        out_dir=tmp_path,
        label="smoke",
        episode=0,
        games=1,
        max_steps=5,
    )

    assert summary is not None
    assert summary["rollup"]["games"] == 1
    rows_path = tmp_path / "near_miss_eval" / "smoke" / "per_level_eval.jsonl"
    summary_path = tmp_path / "near_miss_eval" / "smoke" / "summary.json"
    assert rows_path.exists()
    assert summary_path.exists()
    row = rows_path.read_text(encoding="utf-8").strip()
    assert '"min_target_distance_tiles"' in row
    assert '"close_zone_action_counts"' in row


def test_classify_trace_failure_identifies_exit_unlock_timeout():
    """A policy that collects every crystal but times out at the exit gets its own mode."""
    modes = classify_trace_failure(
        {
            "won": False,
            "crystals_collected": 2,
            "initial_crystals": 2,
            "exit_unlocked": True,
            "end_reason": "timeout",
            "final_steps_since_progress": 100,
            "max_tile_visit_frac": 0.05,
            "idle_action_frac": 0.1,
        }
    )

    assert modes == ["exit_unlocked_no_exit"]


def test_markdown_report_includes_failure_diagnostics(tmp_path):
    """Diagnostic reports should show the compact held-out failure table."""
    report_path = tmp_path / "report.md"
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "diagnostic_baseline",
                    "episodes": 50,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "failure_diagnostics": {
                        "trace_dir": "diag",
                        "rollup": {
                            "wins": 0,
                            "games": 1,
                            "any_crystal_rate": 0.0,
                            "mean_depth_frac": 0.25,
                            "mean_target_distance_delta_tiles": 3.5,
                            "mean_target_distance_best_delta_tiles": 4.0,
                            "mean_anti_loop_penalty_total": -2.25,
                            "mean_interact_action_frac": 0.2,
                            "mean_invalid_interact_count": 8.0,
                            "mean_invalid_interact_penalty_total": -0.4,
                            "mean_novelty_bonus_total": 0.32,
                            "failure_mode_counts": {"no_crystal": 1},
                        },
                        "games_summary": [
                            {
                                "game_index": 0,
                                "end_reason": "timeout",
                                "crystals_collected": 0,
                                "initial_crystals": 1,
                                "final_progress": 0.2,
                                "final_depth_frac": 0.25,
                                "target_distance_delta_tiles": 3.5,
                                "max_tile_visit_frac": 0.2,
                                "idle_action_frac": 0.1,
                                "anti_loop_penalty_total": -2.25,
                                "interact_action_frac": 0.2,
                                "invalid_interact_count": 8,
                                "invalid_interact_penalty_total": -0.4,
                                "novelty_bonus_total": 0.32,
                                "top_actions": {"RIGHT": 10},
                                "failure_modes": ["no_crystal"],
                            }
                        ],
                    },
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Held-out trace: 0/1 wins, 0.0% any crystal, 25.0% depth" in report
    assert "- Trace target distance delta: final 3.50 tiles, best 4.00 tiles" in report
    assert "- Trace anti-loop penalty avg: -2.25" in report
    assert "- Trace invalid interact avg: 8.0 presses, -0.40 reward, 20.0% actions" in report
    assert "- Trace novelty bonus avg: 0.32" in report
    assert (
        "| Trace Game | End | Crystals | Progress | Depth | Target delta | Loop | Idle | Loop penalty | Top actions | Modes |"
        in report
    )
    assert (
        "| 0 | timeout | 0/1 | 0.200 | 25% | 3.50 | 20% | 10% | -2.25 | `{'RIGHT': 10}` | `['no_crystal']` |"
        in report
    )


def test_markdown_report_includes_near_miss_eval(tmp_path):
    """Reports should surface near-miss metrics without manual JSON inspection."""
    report_path = tmp_path / "report.md"
    near_miss = {
        "rows_path": "near_miss/per_level_eval.jsonl",
        "rollup": {
            "near_miss_rate_3": 0.5,
            "near_miss_rate_1_5": 0.25,
            "mean_min_target_distance_tiles": 2.75,
            "mean_target_distance_best_delta_tiles": 6.5,
            "mean_target_distance_final_delta_tiles": 4.0,
            "mean_step_of_best_approach": 123.0,
            "mean_close_zone_steps": 12.0,
            "mean_close_zone_jump_rate": 0.4,
            "mean_close_zone_idle_or_interact_rate": 0.2,
            "stuck_after_close_rate": 0.3,
            "loop_after_close_rate": 0.6,
        },
    }
    write_markdown_report(
        report_path,
        {
            "run_id": "test",
            "seed": 0,
            "created_at": "now",
            "runs": [
                {
                    "label": "diagnostic_baseline",
                    "episodes": 50,
                    "device": "cpu",
                    "steps_per_second": 1.0,
                    "avg_score_100": 0.0,
                    "avg_progress_100": 0.0,
                    "best_progress": 0.0,
                    "final_eval": {
                        "wins": 0,
                        "num_games": 2,
                        "win_rate": 0.0,
                        "mean_crystal_frac": 0.0,
                        "mean_depth_frac": 0.25,
                        "mean_score": 0.0,
                        "end_reason_counts": {"timeout": 2},
                    },
                    "near_miss_eval": near_miss,
                    "selected_checkpoint_eval": {
                        "wins": 1,
                        "num_games": 2,
                        "win_rate": 0.5,
                        "mean_crystal_frac": 0.5,
                        "mean_depth_frac": 0.5,
                        "end_reason_counts": {"first_crystal_goal": 1, "timeout": 1},
                    },
                    "selected_checkpoint_near_miss_eval": near_miss,
                }
            ],
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Final held-out near-miss: <=3 tiles 50.0%, <=1.5 tiles 25.0%" in report
    assert "- Final held-out approach: best delta 6.50 tiles, final delta 4.00 tiles" in report
    assert "- Selected checkpoint near-miss: <=3 tiles 50.0%, <=1.5 tiles 25.0%" in report
    assert (
        "- Selected checkpoint per-level near-miss rows: `near_miss/per_level_eval.jsonl`" in report
    )
