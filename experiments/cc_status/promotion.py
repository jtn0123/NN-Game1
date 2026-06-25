"""Promotion gate for Crystal Caves status-session artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

DECISION_PROMOTE = "PROMOTE"
DECISION_HOLD = "HOLD"
DECISION_REGRESS = "REGRESS"
WIN_RATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PromotionSnapshot:
    label: str
    artifact: str
    wins: int
    games: int
    win_rate: float
    crystal_frac: float
    depth_frac: float
    near_miss_rate_3: float | None = None
    near_miss_rate_1_5: float | None = None
    mean_min_target_distance_tiles: float | None = None
    close_zone_jump_rate: float | None = None
    stuck_after_close_rate: float | None = None
    loop_after_close_rate: float | None = None
    validation_wins: int | None = None
    validation_games: int | None = None
    validation_win_rate: float | None = None
    validation_crystal_frac: float | None = None
    validation_depth_frac: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionGateResult:
    decision: str
    candidate: PromotionSnapshot
    baseline: PromotionSnapshot
    reasons: tuple[str, ...]
    support_improvements: tuple[str, ...]
    support_regressions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
            "reasons": list(self.reasons),
            "support_improvements": list(self.support_improvements),
            "support_regressions": list(self.support_regressions),
        }


B3S_PROMOTED_BASELINE = PromotionSnapshot(
    label="B3s conservative demo-Q",
    artifact="frozen-b3s-reference",
    wins=10,
    games=30,
    win_rate=10 / 30,
    crystal_frac=0.333,
    depth_frac=0.605,
    near_miss_rate_3=0.600,
    near_miss_rate_1_5=0.400,
    mean_min_target_distance_tiles=3.51,
    close_zone_jump_rate=0.000,
    stuck_after_close_rate=0.200,
    loop_after_close_rate=0.333,
    validation_wins=19,
    validation_games=60,
    validation_win_rate=19 / 60,
    validation_crystal_frac=0.317,
    validation_depth_frac=0.600,
)


def compare_promotion_candidate(
    candidate_artifact: Path,
    *,
    baseline_artifact: Path | None = None,
    validation_artifact: Path | None = None,
) -> PromotionGateResult:
    candidate = promotion_snapshot_from_artifact(candidate_artifact)
    if validation_artifact is not None:
        validation = promotion_snapshot_from_artifact(validation_artifact)
        candidate = _with_validation(candidate, validation)
    baseline = (
        promotion_snapshot_from_artifact(baseline_artifact)
        if baseline_artifact is not None
        else B3S_PROMOTED_BASELINE
    )
    return gate_candidate(candidate, baseline)


def gate_candidate(
    candidate: PromotionSnapshot,
    baseline: PromotionSnapshot = B3S_PROMOTED_BASELINE,
) -> PromotionGateResult:
    reasons: list[str] = []
    improvements, regressions = support_metric_changes(candidate, baseline)
    candidate_win_rate = _safe_rate(candidate.wins, candidate.games, candidate.win_rate)
    baseline_win_rate = _safe_rate(baseline.wins, baseline.games, baseline.win_rate)

    if candidate.games < baseline.games:
        reasons.append(
            f"selected eval sample is too small: {candidate.wins}/{candidate.games} "
            f"vs baseline {baseline.wins}/{baseline.games}"
        )
        return PromotionGateResult(
            DECISION_HOLD,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    if candidate_win_rate < baseline_win_rate - WIN_RATE_TOLERANCE:
        reasons.append(
            f"selected win rate trails baseline: {candidate.wins}/{candidate.games} "
            f"({100*candidate_win_rate:.1f}%) vs {baseline.wins}/{baseline.games} "
            f"({100*baseline_win_rate:.1f}%)"
        )
        return PromotionGateResult(
            DECISION_REGRESS,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    selected_ties_baseline = abs(candidate_win_rate - baseline_win_rate) <= WIN_RATE_TOLERANCE
    if selected_ties_baseline and len(improvements) < 2:
        reasons.append(
            f"selected win rate ties baseline at {candidate.wins}/{candidate.games}, "
            "but fewer than two support metrics improved"
        )
        return PromotionGateResult(
            DECISION_REGRESS,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    if candidate.validation_wins is None:
        reasons.append(
            "selected checkpoint is promising but needs expanded validation before promotion"
        )
        return PromotionGateResult(
            DECISION_HOLD,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    baseline_validation_wins = baseline.validation_wins or baseline.wins
    baseline_validation_games = baseline.validation_games or baseline.games
    baseline_validation_rate = _safe_rate(
        baseline_validation_wins,
        baseline_validation_games,
        _fallback_float(baseline.validation_win_rate, 0.0),
    )
    candidate_validation_games = candidate.validation_games or 0
    if candidate_validation_games < baseline_validation_games:
        reasons.append(
            f"expanded validation sample is too small: "
            f"{candidate.validation_wins}/{candidate.validation_games} vs "
            f"{baseline_validation_wins}/{baseline_validation_games}"
        )
        return PromotionGateResult(
            DECISION_HOLD,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    candidate_validation_rate = _safe_rate(
        candidate.validation_wins or 0,
        candidate_validation_games,
        _fallback_float(candidate.validation_win_rate, 0.0),
    )
    if candidate_validation_rate < baseline_validation_rate - WIN_RATE_TOLERANCE:
        reasons.append(
            f"expanded validation win rate trails baseline: "
            f"{candidate.validation_wins}/{candidate.validation_games} "
            f"({100*candidate_validation_rate:.1f}%) vs "
            f"{baseline_validation_wins}/{baseline_validation_games} "
            f"({100*baseline_validation_rate:.1f}%)"
        )
        return PromotionGateResult(
            DECISION_REGRESS,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    min_depth = _fallback_float(baseline.validation_depth_frac, baseline.depth_frac) - 0.03
    validation_depth = _fallback_float(candidate.validation_depth_frac, candidate.depth_frac)
    if validation_depth < min_depth:
        reasons.append(
            f"expanded validation depth regressed: {validation_depth:.3f} "
            f"below required {min_depth:.3f}"
        )
        return PromotionGateResult(
            DECISION_REGRESS,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    if selected_ties_baseline and len(improvements) < 3:
        reasons.append(
            "tie on selected wins needs at least three support improvements after validation"
        )
        return PromotionGateResult(
            DECISION_HOLD,
            candidate,
            baseline,
            tuple(reasons),
            tuple(improvements),
            tuple(regressions),
        )

    reasons.append("selected and expanded validation gates clear the B3s baseline")
    return PromotionGateResult(
        DECISION_PROMOTE,
        candidate,
        baseline,
        tuple(reasons),
        tuple(improvements),
        tuple(regressions),
    )


def support_metric_changes(
    candidate: PromotionSnapshot,
    baseline: PromotionSnapshot,
) -> tuple[list[str], list[str]]:
    improvements: list[str] = []
    regressions: list[str] = []
    _compare_higher(
        "crystal_frac", candidate.crystal_frac, baseline.crystal_frac, improvements, regressions
    )
    _compare_higher(
        "depth_frac", candidate.depth_frac, baseline.depth_frac, improvements, regressions
    )
    _compare_higher(
        "near_miss_rate_3",
        candidate.near_miss_rate_3,
        baseline.near_miss_rate_3,
        improvements,
        regressions,
    )
    _compare_higher(
        "near_miss_rate_1_5",
        candidate.near_miss_rate_1_5,
        baseline.near_miss_rate_1_5,
        improvements,
        regressions,
    )
    _compare_lower(
        "mean_min_target_distance_tiles",
        candidate.mean_min_target_distance_tiles,
        baseline.mean_min_target_distance_tiles,
        improvements,
        regressions,
    )
    _compare_higher(
        "close_zone_jump_rate",
        candidate.close_zone_jump_rate,
        baseline.close_zone_jump_rate,
        improvements,
        regressions,
    )
    _compare_lower(
        "stuck_after_close_rate",
        candidate.stuck_after_close_rate,
        baseline.stuck_after_close_rate,
        improvements,
        regressions,
    )
    _compare_lower(
        "loop_after_close_rate",
        candidate.loop_after_close_rate,
        baseline.loop_after_close_rate,
        improvements,
        regressions,
    )
    return improvements, regressions


def promotion_snapshot_from_artifact(path: Path) -> PromotionSnapshot:
    summary_path = path / "summary.json" if path.is_dir() else path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path} root must be a JSON object")
    run = _select_run(payload)
    eval_payload = run.get("selected_checkpoint_eval") or run.get("final_eval")
    if not isinstance(eval_payload, dict):
        raise ValueError(f"{summary_path} has no selected_checkpoint_eval or final_eval")
    near_miss = run.get("selected_checkpoint_near_miss_eval") or run.get("near_miss_eval") or {}
    rollup = near_miss.get("rollup") if isinstance(near_miss, dict) else {}
    if not isinstance(rollup, dict):
        rollup = {}
    wins = int(eval_payload.get("wins", 0) or 0)
    games = int(eval_payload.get("num_games", eval_payload.get("games", 0)) or 0)
    return PromotionSnapshot(
        label=str(run.get("label") or payload.get("mode") or summary_path.parent.name),
        artifact=str(summary_path.parent),
        wins=wins,
        games=games,
        win_rate=_eval_win_rate(eval_payload, wins, games),
        crystal_frac=float(eval_payload.get("mean_crystal_frac", 0.0) or 0.0),
        depth_frac=float(eval_payload.get("mean_depth_frac", 0.0) or 0.0),
        near_miss_rate_3=_optional_float(rollup.get("near_miss_rate_3")),
        near_miss_rate_1_5=_optional_float(rollup.get("near_miss_rate_1_5")),
        mean_min_target_distance_tiles=_optional_float(
            rollup.get("mean_min_target_distance_tiles")
        ),
        close_zone_jump_rate=_optional_float(rollup.get("mean_close_zone_jump_rate")),
        stuck_after_close_rate=_optional_float(rollup.get("stuck_after_close_rate")),
        loop_after_close_rate=_optional_float(rollup.get("loop_after_close_rate")),
    )


def format_promotion_result(result: PromotionGateResult) -> str:
    c = result.candidate
    b = result.baseline
    lines = [
        f"Decision: {result.decision}",
        f"Candidate: {c.label} ({c.wins}/{c.games}, crystals {100*c.crystal_frac:.1f}%, depth {100*c.depth_frac:.1f}%)",
        f"Baseline:  {b.label} ({b.wins}/{b.games}, crystals {100*b.crystal_frac:.1f}%, depth {100*b.depth_frac:.1f}%)",
    ]
    if c.validation_wins is not None:
        lines.append(
            f"Validation: {c.validation_wins}/{c.validation_games}, "
            f"crystals {100*(c.validation_crystal_frac or 0):.1f}%, "
            f"depth {100*(c.validation_depth_frac or 0):.1f}%"
        )
    lines.append("Reasons:")
    lines.extend(f"- {reason}" for reason in result.reasons)
    if result.support_improvements:
        lines.append("Support improvements: " + ", ".join(result.support_improvements))
    if result.support_regressions:
        lines.append("Support regressions: " + ", ".join(result.support_regressions))
    return "\n".join(lines)


def promotion_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a Crystal Caves run artifact to B3s.")
    parser.add_argument("candidate", help="Candidate status-session folder or summary.json")
    parser.add_argument(
        "--baseline", help="Optional baseline status-session folder or summary.json"
    )
    parser.add_argument(
        "--validation",
        help="Optional expanded-validation status-session folder or summary.json for candidate",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    opts = parser.parse_args(argv)
    result = compare_promotion_candidate(
        Path(opts.candidate),
        baseline_artifact=Path(opts.baseline) if opts.baseline else None,
        validation_artifact=Path(opts.validation) if opts.validation else None,
    )
    if opts.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_promotion_result(result))
    return 0 if result.decision in {DECISION_PROMOTE, DECISION_HOLD} else 1


def _select_run(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("summary has no runs")
    for run in runs:
        if isinstance(run, dict) and "selected_checkpoint_eval" in run:
            return run
    for run in reversed(runs):
        if isinstance(run, dict) and "final_eval" in run:
            return run
    raise ValueError("summary has no run with selected_checkpoint_eval or final_eval")


def _with_validation(
    candidate: PromotionSnapshot,
    validation: PromotionSnapshot,
) -> PromotionSnapshot:
    return PromotionSnapshot(
        **{
            **candidate.to_dict(),
            "validation_wins": validation.wins,
            "validation_games": validation.games,
            "validation_win_rate": validation.win_rate,
            "validation_crystal_frac": validation.crystal_frac,
            "validation_depth_frac": validation.depth_frac,
        }
    )


def _compare_higher(
    name: str,
    candidate: float | None,
    baseline: float | None,
    improvements: list[str],
    regressions: list[str],
    *,
    tolerance: float = 0.005,
) -> None:
    if candidate is None or baseline is None:
        return
    if candidate > baseline + tolerance:
        improvements.append(name)
    elif candidate < baseline - tolerance:
        regressions.append(name)


def _compare_lower(
    name: str,
    candidate: float | None,
    baseline: float | None,
    improvements: list[str],
    regressions: list[str],
    *,
    tolerance: float = 0.005,
) -> None:
    if candidate is None or baseline is None:
        return
    if candidate < baseline - tolerance:
        improvements.append(name)
    elif candidate > baseline + tolerance:
        regressions.append(name)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _eval_win_rate(eval_payload: dict[str, Any], wins: int, games: int) -> float:
    value = eval_payload.get("win_rate")
    if value is None:
        return _safe_rate(wins, games)
    return float(value)


def _safe_rate(wins: int, games: int, fallback: float = 0.0) -> float:
    if games:
        return wins / games
    return fallback


def _fallback_float(primary: float | None, fallback: float) -> float:
    return fallback if primary is None else primary
