"""Small helpers used by the Crystal Caves status-session CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .scorecard import build_route_contact_scorecard, route_contact_source_snapshot_score


def append_interrupted_run_from_live_metrics(
    out_dir: Path,
    payload: dict[str, Any],
    *,
    reason: str = "KeyboardInterrupt",
) -> bool:
    live_paths = sorted(
        out_dir.glob("*/live_metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not live_paths:
        payload["interrupted"] = True
        payload["interruption"] = {
            "reason": reason,
            "partial_summary": False,
        }
        return False

    live_path = live_paths[0]
    live = json.loads(live_path.read_text(encoding="utf-8"))
    run = _partial_run_from_live_metrics(live_path, live, reason=reason)
    existing_labels = {
        str(item.get("label", "")) for item in payload.get("runs", []) if isinstance(item, dict)
    }
    if run["label"] not in existing_labels:
        payload.setdefault("runs", []).append(run)
    payload["interrupted"] = True
    payload["interruption"] = {
        "reason": reason,
        "partial_summary": True,
        "live_metrics_path": str(live_path),
    }
    return True


def tutorial_demo_bc_kwargs(
    opts: argparse.Namespace,
    route_demo_variants: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "episodes": opts.episodes,
        "seed": opts.seed,
        "eval_games": opts.eval_games,
        "trace_games": opts.trace_eval_games,
        "trace_max_steps": opts.trace_max_steps,
        "trace_sample_every": opts.trace_sample_every,
        "trace_tail_steps": opts.trace_tail_steps,
        "train_eval_games": opts.train_eval_games,
        "eval_every": opts.eval_every,
        "log_every": opts.log_every,
        "report_seconds": opts.report_seconds,
        "heartbeat_seconds": opts.heartbeat_seconds,
        "vec_envs": opts.vec_envs,
        "save_checkpoints": opts.save_checkpoints,
        "save_selected_checkpoint": opts.save_selected_checkpoint,
        "cave_pool_size": opts.cave_pool_size,
        "selected_eval_games": opts.selected_eval_games,
        "history_state": opts.history_state,
        "history_steps": opts.history_steps,
        "geo_compass": opts.geo_compass,
        "geo_compass_hazard_aware": opts.geo_compass_hazard_aware,
        "geodesic_potential": opts.geodesic_potential,
        "geodesic_potential_weight": opts.geodesic_potential_weight,
        "show_locked_exit": opts.show_locked_exit,
        "reverse_curriculum_p": opts.reverse_curriculum_p,
        "reward_clip": opts.reward_clip,
        "distributional_dqn": opts.distributional_dqn,
        "c51_atoms": opts.c51_atoms,
        "c51_v_min": opts.c51_v_min,
        "c51_v_max": opts.c51_v_max,
        "route_demo_levels": opts.route_demo_levels,
        "route_demo_max_steps": opts.route_demo_max_steps,
        "route_demo_variants": route_demo_variants,
        "demo_selection_mode": opts.demo_selection_mode,
        "bc_epochs": opts.bc_epochs,
        "bc_batch_size": opts.bc_batch_size,
        "demo_repeat": opts.demo_repeat,
    }


def close_zone_route_demo_variants(
    opts: argparse.Namespace,
    route_demo_variants: tuple[str, ...],
) -> tuple[str, ...]:
    if opts.route_demo_variants == "direct":
        return ("direct", "recovery")
    return route_demo_variants


def _partial_run_from_live_metrics(
    live_path: Path,
    live: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    run_dir = live_path.parent
    source_history = _read_jsonl_objects(run_dir / "source_eval_history.jsonl")
    latest_source_eval = (
        source_history[-1].get("source_eval")
        if source_history and isinstance(source_history[-1], dict)
        else None
    )
    final_eval = latest_source_eval or live.get("latest_eval")
    elapsed = float(live.get("elapsed_seconds", 0.0) or 0.0)
    label = str(live.get("label") or run_dir.name)
    run: dict[str, Any] = {
        "label": label,
        "partial": True,
        "interrupted": True,
        "interrupt_reason": reason,
        "episodes": int(live.get("episode", 0) or 0),
        "target_episodes": int(live.get("total_episodes", 0) or 0),
        "total_steps": int(live.get("total_steps", 0) or 0),
        "train_seconds": elapsed,
        "steps_per_second": float(live.get("steps_per_second", 0.0) or 0.0),
        "device": "unknown",
        "final_epsilon": float(live.get("epsilon", 0.0) or 0.0),
        "memory_size": int(live.get("memory_size", 0) or 0),
        "avg_loss_100": float(live.get("avg_loss_100", 0.0) or 0.0),
        "avg_q_100": float(live.get("avg_q_100", 0.0) or 0.0),
        "avg_score_100": float(live.get("avg_score_100", 0.0) or 0.0),
        "win_rate_100": float(live.get("win_rate_100", 0.0) or 0.0),
        "avg_reward_100": 0.0,
        "avg_progress_100": float(live.get("avg_progress_100", 0.0) or 0.0),
        "best_progress": float(live.get("best_progress", 0.0) or 0.0),
        "end_reason_counts_100": live.get("end_reason_counts_100", {}),
        "mean_phi_parts_100": live.get("mean_phi_parts_100", {}),
        "config": {},
        "eval_history": _eval_history_from_live(live),
        "source_stats": live.get("source_stats", {}),
        "reverse_start_stats": live.get("reverse_start_stats", {}),
        "archive_stats": live.get("archive_stats", {}),
        "source_eval_history": source_history,
        "live_metrics_path": str(live_path),
        "live_metrics_history_path": str(run_dir / "live_metrics.jsonl"),
    }
    for key in (
        "contact_lane_win_rate_100",
        "contact_lane_crystal_rate_100",
        "contact_lane_exit_rate_100",
        "full_lane_progress_100",
        "full_lane_first_crystal_rate_100",
    ):
        if key in live:
            run[key] = float(live.get(key, 0.0) or 0.0)
    if final_eval:
        run["final_eval"] = final_eval
    if source_history:
        best_source = max(source_history, key=route_contact_source_snapshot_score)
        run["selected_source_episode"] = int(best_source.get("episode", 0) or 0)
        run["selected_source_eval"] = best_source.get("source_eval") or {}
    run["route_contact_scorecard"] = build_route_contact_scorecard(run)
    return run


def _eval_history_from_live(live: dict[str, Any]) -> list[dict[str, Any]]:
    latest_eval = live.get("latest_eval")
    return [latest_eval] if isinstance(latest_eval, dict) else []


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows
