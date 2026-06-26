# ruff: noqa: F401,F403,F405,I001
"""Route/contact scorecard helpers for Crystal Caves status sessions."""

from __future__ import annotations

from typing import Any

SELECTION_EVAL_KEYS = (
    "selected_checkpoint_eval",
    "selected_source_eval",
    "final_eval",
)

SCORE_WEIGHTS = {
    "first_crystal_rate": 3.0,
    "crystal_frac": 1.5,
    "depth_frac": 1.0,
    "close_zone_rate": 0.5,
    "loop_after_close_rate": -1.0,
    "stall_rate": -0.5,
}


def build_route_contact_scorecard(run: dict[str, Any]) -> dict[str, Any]:
    """Summarize route-preservation and final-contact evidence for one run."""

    eval_source, eval_payload = _select_eval_payload(run)
    if not eval_payload:
        return {
            "has_eval": False,
            "eval_source": "",
            "score": 0.0,
            "verdict": "insufficient evidence",
            "risks": ["no selected/source/final eval payload available"],
            "weights": dict(SCORE_WEIGHTS),
        }

    near_miss_source, near_miss_rollup = _select_near_miss_rollup(run, eval_source)
    diagnostics_source, diagnostics_rollup = _select_diagnostics_rollup(run, eval_source)
    games = _eval_games(eval_payload)
    wins = _int(eval_payload.get("wins"))
    end_reason_counts = _dict(eval_payload.get("end_reason_counts"))

    win_rate = _eval_rate(eval_payload, wins, games, "win_rate")
    crystal_frac = _float(eval_payload.get("mean_crystal_frac"))
    depth_frac = _float(eval_payload.get("mean_depth_frac"))
    first_crystal_rate, first_crystal_source = _first_crystal_rate(
        run,
        eval_payload,
        diagnostics_rollup,
        win_rate,
        crystal_frac,
    )
    close_zone_rate = _float(near_miss_rollup.get("near_miss_rate_3"))
    loop_after_close_rate = _float(near_miss_rollup.get("loop_after_close_rate"))
    stuck_after_close_rate = _float(near_miss_rollup.get("stuck_after_close_rate"))
    stall_rate = _end_reason_rate(end_reason_counts, games, "stalled")
    timeout_rate = _end_reason_rate(end_reason_counts, games, "timeout")

    score = route_contact_score(
        first_crystal_rate=first_crystal_rate,
        crystal_frac=crystal_frac,
        depth_frac=depth_frac,
        close_zone_rate=close_zone_rate,
        loop_after_close_rate=loop_after_close_rate,
        stall_rate=stall_rate,
    )
    metrics = {
        "wins": wins,
        "games": games,
        "win_rate": win_rate,
        "first_crystal_rate": first_crystal_rate,
        "crystal_frac": crystal_frac,
        "depth_frac": depth_frac,
        "close_zone_rate": close_zone_rate,
        "near_miss_rate_1_5": _float(near_miss_rollup.get("near_miss_rate_1_5")),
        "mean_min_target_distance_tiles": _optional_float(
            near_miss_rollup.get("mean_min_target_distance_tiles")
        ),
        "target_distance_best_delta_tiles": _optional_float(
            near_miss_rollup.get("mean_target_distance_best_delta_tiles")
        ),
        "target_distance_final_delta_tiles": _optional_float(
            near_miss_rollup.get("mean_target_distance_final_delta_tiles")
        ),
        "close_zone_jump_rate": _float(near_miss_rollup.get("mean_close_zone_jump_rate")),
        "close_zone_idle_or_interact_rate": _float(
            near_miss_rollup.get("mean_close_zone_idle_or_interact_rate")
        ),
        "stuck_after_close_rate": stuck_after_close_rate,
        "loop_after_close_rate": loop_after_close_rate,
        "stall_rate": stall_rate,
        "timeout_rate": timeout_rate,
    }
    risks = _route_contact_risks(metrics, run)
    return {
        "has_eval": True,
        "eval_source": eval_source,
        "near_miss_source": near_miss_source,
        "diagnostics_source": diagnostics_source,
        "score": score,
        "verdict": route_contact_verdict(metrics, risks),
        "risks": risks,
        "metrics": metrics,
        "end_reason_counts": end_reason_counts,
        "first_crystal_rate_source": first_crystal_source,
        "weights": dict(SCORE_WEIGHTS),
        "contact_lane": _contact_lane_metrics(run),
    }


def route_contact_score(
    *,
    first_crystal_rate: float,
    crystal_frac: float,
    depth_frac: float,
    close_zone_rate: float,
    loop_after_close_rate: float,
    stall_rate: float,
) -> float:
    return (
        SCORE_WEIGHTS["first_crystal_rate"] * first_crystal_rate
        + SCORE_WEIGHTS["crystal_frac"] * crystal_frac
        + SCORE_WEIGHTS["depth_frac"] * depth_frac
        + SCORE_WEIGHTS["close_zone_rate"] * close_zone_rate
        + SCORE_WEIGHTS["loop_after_close_rate"] * loop_after_close_rate
        + SCORE_WEIGHTS["stall_rate"] * stall_rate
    )


def route_contact_source_snapshot_score(
    snapshot: dict[str, Any],
) -> tuple[float, float, float, int]:
    """Composite ordering for future source checkpoint selection."""

    eval_payload = _dict(snapshot.get("source_eval"))
    scorecard = build_route_contact_scorecard({"selected_source_eval": eval_payload})
    metrics = _dict(scorecard.get("metrics"))
    return (
        _float(scorecard.get("score")),
        _float(metrics.get("first_crystal_rate")),
        _float(metrics.get("depth_frac")),
        int(snapshot.get("episode", 0) or 0),
    )


def route_contact_verdict(metrics: dict[str, Any], risks: list[str]) -> str:
    if not metrics:
        return "insufficient evidence"
    if any(risk.startswith("route regression") for risk in risks):
        return "route regression"
    if any(risk.startswith("depth regression") for risk in risks):
        return "depth regression"
    if any(risk.startswith("contact regression") for risk in risks):
        return "contact regression"
    if any(risk.startswith("stall pressure") for risk in risks):
        return "candidate keeper with stall risk"
    return "candidate keeper"


def _route_contact_risks(metrics: dict[str, Any], run: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    first_crystal_rate = _float(metrics.get("first_crystal_rate"))
    crystal_frac = _float(metrics.get("crystal_frac"))
    depth_frac = _float(metrics.get("depth_frac"))
    close_zone_rate = _float(metrics.get("close_zone_rate"))
    loop_rate = _float(metrics.get("loop_after_close_rate"))
    stuck_rate = _float(metrics.get("stuck_after_close_rate"))
    stall_rate = _float(metrics.get("stall_rate"))
    jump_rate = _float(metrics.get("close_zone_jump_rate"))

    if first_crystal_rate < 0.20 and crystal_frac < 0.20:
        risks.append(
            "route regression: first objective below 20%, so contact training is not transferring"
        )
    if depth_frac < 0.55:
        risks.append("depth regression: mean depth below the 55% route-preservation guardrail")
    if close_zone_rate >= 0.25 and (loop_rate >= 0.25 or stuck_rate >= 0.25 or jump_rate < 0.05):
        risks.append("contact regression: reaches close-zone but loop/stuck/jump behavior is weak")
    if stall_rate >= 0.30:
        risks.append("stall pressure: stalled end states are at or above 30%")
    contact_lane = _contact_lane_metrics(run)
    if contact_lane.get("contact_lane_win_rate_100", 0.0) >= 0.90 and first_crystal_rate < 0.20:
        risks.append(
            "lane-transfer gap: contact lanes are mastered but held-out route/contact is low"
        )
    return risks


def _select_eval_payload(run: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key in SELECTION_EVAL_KEYS:
        payload = _dict(run.get(key))
        if payload:
            return key, payload
    return "", {}


def _select_near_miss_rollup(run: dict[str, Any], eval_source: str) -> tuple[str, dict[str, Any]]:
    keys = (
        ("selected_checkpoint_near_miss_eval",)
        if eval_source == "selected_checkpoint_eval"
        else ("near_miss_eval", "selected_checkpoint_near_miss_eval")
    )
    for key in keys:
        payload = _dict(run.get(key))
        rollup = _dict(payload.get("rollup"))
        if rollup:
            return key, rollup
    return "", {}


def _select_diagnostics_rollup(run: dict[str, Any], eval_source: str) -> tuple[str, dict[str, Any]]:
    keys = (
        ("selected_checkpoint_failure_diagnostics",)
        if eval_source == "selected_checkpoint_eval"
        else ("failure_diagnostics", "selected_checkpoint_failure_diagnostics")
    )
    for key in keys:
        payload = _dict(run.get(key))
        rollup = _dict(payload.get("rollup"))
        if rollup:
            return key, rollup
    return "", {}


def _first_crystal_rate(
    run: dict[str, Any],
    eval_payload: dict[str, Any],
    diagnostics_rollup: dict[str, Any],
    win_rate: float,
    crystal_frac: float,
) -> tuple[float, str]:
    end_reason_counts = _dict(eval_payload.get("end_reason_counts"))
    config = _dict(run.get("config"))
    if config.get("first_crystal_goal") or "first_crystal_goal" in end_reason_counts:
        return win_rate, "win_rate:first_crystal_goal"
    any_crystal_rate = diagnostics_rollup.get("any_crystal_rate")
    if any_crystal_rate is not None:
        return _float(any_crystal_rate), "trace:any_crystal_rate"
    return max(win_rate, crystal_frac), "mean_crystal_frac_proxy"


def _contact_lane_metrics(run: dict[str, Any]) -> dict[str, float]:
    keys = (
        "contact_lane_win_rate_100",
        "contact_lane_crystal_rate_100",
        "contact_lane_exit_rate_100",
        "full_lane_progress_100",
        "full_lane_first_crystal_rate_100",
    )
    return {key: _float(run.get(key)) for key in keys if key in run}


def _eval_games(eval_payload: dict[str, Any]) -> int:
    return _int(eval_payload.get("num_games", eval_payload.get("games")))


def _eval_rate(eval_payload: dict[str, Any], wins: int, games: int, key: str) -> float:
    if eval_payload.get(key) is not None:
        return _float(eval_payload.get(key))
    return wins / games if games else 0.0


def _end_reason_rate(end_reason_counts: dict[str, Any], games: int, key: str) -> float:
    if games <= 0:
        return 0.0
    return _float(end_reason_counts.get(key)) / games


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
