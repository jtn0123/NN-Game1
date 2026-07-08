# ruff: noqa: F401,F403,F405,I001
"""Paired multi-seed A/B evaluation for Crystal Caves checkpoints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .common import *
from .config_helpers import set_seed, timestamp_id
from .evals import _enter_greedy_agent_eval, _resolved_end_reason, _restore_greedy_agent_eval
from .io_utils import append_jsonl, write_json
from .reports import load_selected_weight_snapshot
from .runs_transfer import config_from_selected_checkpoint
from .training import load_weight_snapshot, prepare_trainer
from src.ai.evaluator import EvalResults

DEFAULT_AB_METRIC = "selection_score"


@dataclass(frozen=True)
class ArmSpec:
    label: str
    checkpoint: Path


def interquartile_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) < 4:
        return float(np.mean(ordered))
    lo = int(np.floor(len(ordered) * 0.25))
    hi = int(np.ceil(len(ordered) * 0.75))
    core = ordered[lo:hi]
    return float(np.mean(core if len(core) else ordered))


def pipeline_mean(values: list[float]) -> float:
    """Empty-safe PLAIN mean — the estimator the metrics/lever pipeline must use.

    NOT interquartile_mean: trimming the outer 50% floored bottom-heavy sparse-RL
    distributions to 0 and even flipped the SIGN of near-zero lever deltas (audit B1), so
    a helpful lever could read as harmful. interquartile_mean is kept only as a generic
    utility; nothing in the metric/A-B path should call it."""
    return float(np.mean(values)) if values else 0.0


def stratified_bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    point = pipeline_mean(values)
    if not rows or n_bootstrap <= 0:
        return {"mean": point, "ci_low": point, "ci_high": point, "n": float(len(rows))}

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row.get("seed", 0) or 0)].append(row)
    group_keys = sorted(groups)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    alpha = (1.0 - confidence) / 2.0
    for _ in range(n_bootstrap):
        sampled_rows: list[dict[str, Any]] = []
        sampled_keys = rng.choice(group_keys, size=len(group_keys), replace=True)
        for key in sampled_keys:
            group = groups[int(key)]
            indices = rng.integers(0, len(group), size=len(group))
            sampled_rows.extend(group[int(index)] for index in indices)
        samples.append(pipeline_mean([float(row.get(metric, 0.0) or 0.0) for row in sampled_rows]))

    return {
        "mean": point,
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "n": float(len(rows)),
    }


def pair_level_rows(
    arm_a_rows: list[dict[str, Any]],
    arm_b_rows: list[dict[str, Any]],
    *,
    metric: str = DEFAULT_AB_METRIC,
) -> list[dict[str, Any]]:
    a_by_key = {(int(row["seed"]), int(row["level_index"])): row for row in arm_a_rows}
    b_by_key = {(int(row["seed"]), int(row["level_index"])): row for row in arm_b_rows}
    paired: list[dict[str, Any]] = []
    for key in sorted(a_by_key.keys() & b_by_key.keys()):
        a_row = a_by_key[key]
        b_row = b_by_key[key]
        a_value = float(a_row.get(metric, 0.0) or 0.0)
        b_value = float(b_row.get(metric, 0.0) or 0.0)
        paired.append(
            {
                "seed": key[0],
                "level_index": key[1],
                "level_name": a_row.get("level_name") or b_row.get("level_name"),
                f"a_{metric}": a_value,
                f"b_{metric}": b_value,
                f"delta_{metric}": b_value - a_value,
                "a_end_reason": a_row.get("end_reason"),
                "b_end_reason": b_row.get("end_reason"),
                "a_won": bool(a_row.get("won", False)),
                "b_won": bool(b_row.get("won", False)),
            }
        )
    return paired


def aggregate_paired_ab(
    arm_a_rows: list[dict[str, Any]],
    arm_b_rows: list[dict[str, Any]],
    *,
    metric: str = DEFAULT_AB_METRIC,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    paired = pair_level_rows(arm_a_rows, arm_b_rows, metric=metric)
    return {
        "metric": metric,
        "arm_a": stratified_bootstrap_ci(
            arm_a_rows,
            metric=metric,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "arm_b": stratified_bootstrap_ci(
            arm_b_rows,
            metric=metric,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 1,
        ),
        "paired_delta_b_minus_a": stratified_bootstrap_ci(
            paired,
            metric=f"delta_{metric}",
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 2,
        ),
        "paired_rows": len(paired),
    }


def evaluate_checkpoint_rows(
    arm: ArmSpec,
    *,
    out_dir: Path,
    seed: int,
    games: int,
    max_steps: int | None,
    log_every: int,
    report_seconds: float,
) -> list[dict[str, Any]]:
    set_seed(seed)
    snapshot = load_selected_weight_snapshot(arm.checkpoint)
    config = config_from_selected_checkpoint(
        out_dir / arm.label / f"seed_{seed}",
        snapshot=snapshot,
        seed=seed,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    trainer = prepare_trainer(config, episodes=1, vec_envs=1, save_checkpoints=False)
    _validate_checkpoint_shape(trainer, snapshot, arm.checkpoint)
    load_weight_snapshot(trainer.agent, snapshot["weights"])
    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    evaluator = Evaluator(
        game=game,
        agent=trainer.agent,
        config=config,
        log_dir=out_dir / arm.label / f"seed_{seed}" / "eval",
    )
    step_limit = int(max_steps or config.EVAL_MAX_STEPS)
    rows: list[dict[str, Any]] = []
    agent_state = _enter_greedy_agent_eval(trainer.agent)
    try:
        for level_index in range(games):
            rows.append(
                _evaluate_one_level(
                    game,
                    trainer.agent,
                    evaluator,
                    arm=arm.label,
                    seed=seed,
                    level_index=level_index,
                    max_steps=step_limit,
                )
            )
    finally:
        _restore_greedy_agent_eval(trainer.agent, agent_state)
    return rows


def _validate_checkpoint_shape(trainer: Any, snapshot: dict[str, Any], checkpoint: Path) -> None:
    saved_state_size = int(snapshot.get("state_size", trainer.agent.state_size) or 0)
    saved_action_size = int(snapshot.get("action_size", trainer.agent.action_size) or 0)
    if saved_state_size and saved_state_size != trainer.agent.state_size:
        raise ValueError(
            f"{checkpoint} state size {saved_state_size} does not match "
            f"environment state size {trainer.agent.state_size}"
        )
    if saved_action_size and saved_action_size != trainer.agent.action_size:
        raise ValueError(
            f"{checkpoint} action size {saved_action_size} does not match "
            f"environment action size {trainer.agent.action_size}"
        )


def _evaluate_one_level(
    game: CrystalCaves,
    agent: Any,
    evaluator: Evaluator,
    *,
    arm: str,
    seed: int,
    level_index: int,
    max_steps: int,
) -> dict[str, Any]:
    state = game.reset()
    done = False
    steps = 0
    info: dict[str, Any] = {"score": 0, "level": 1, "won": False}
    target_distances: list[float] = []
    initial_distance = evaluator._target_distance_tiles()
    if initial_distance is not None:
        target_distances.append(initial_distance)
    while not done and steps < max_steps:
        action = agent.select_action(state, training=False)
        state, _, done, info = game.step(action)
        steps += 1
        distance = evaluator._target_distance_tiles()
        if distance is not None:
            target_distances.append(distance)

    parts = info.get("progress_parts") or {}
    if not isinstance(parts, dict):
        parts = {}
    target_progress = _target_distance_progress(target_distances)
    single_result = EvalResults(
        timestamp=datetime.now().isoformat(),
        episode=0,
        num_games=1,
        mean_score=float(info.get("score", 0) or 0.0),
        median_score=float(info.get("score", 0) or 0.0),
        std_score=0.0,
        min_score=int(info.get("score", 0) or 0),
        max_score=int(info.get("score", 0) or 0),
        q25_score=float(info.get("score", 0) or 0.0),
        q75_score=float(info.get("score", 0) or 0.0),
        mean_level=float(info.get("level", 1) or 1),
        max_level=int(info.get("level", 1) or 1),
        level_distribution={int(info.get("level", 1) or 1): 1},
        wins=int(bool(info.get("won", False))),
        win_rate=1.0 if info.get("won", False) else 0.0,
        mean_steps=float(steps),
        max_steps=steps,
        mean_crystal_frac=float(parts.get("crystal_frac", 0.0) or 0.0),
        mean_switch_rate=float(parts.get("switch_done", 0.0) or 0.0),
        mean_depth_frac=float(parts.get("depth_frac", 0.0) or 0.0),
        mean_target_distance_progress=target_progress,
        mean_exit_unlocked_rate=1.0 if info.get("exit_unlocked", False) else 0.0,
        end_reason_counts={_resolved_end_reason(info, steps=steps, max_steps=max_steps): 1},
    )
    return {
        "arm": arm,
        "seed": seed,
        "level_index": level_index,
        "level_name": info.get("level_name"),
        "score": single_result.mean_score,
        "won": bool(info.get("won", False)),
        "crystal_frac": single_result.mean_crystal_frac,
        "switch_rate": single_result.mean_switch_rate,
        "depth_frac": single_result.mean_depth_frac,
        "target_distance_progress": target_progress,
        "exit_unlocked_rate": single_result.mean_exit_unlocked_rate,
        "selection_score": evaluator._selection_score(single_result),
        "steps": steps,
        "damage_taken": float(info.get("damage_taken", 0.0) or 0.0),
        "tiles_visited": float(info.get("tiles_visited", 0.0) or 0.0),
        "idle_frac": float(info.get("idle_frac", 0.0) or 0.0),
        "end_reason": next(iter(single_result.end_reason_counts)),
        "last_damage_source": str(info.get("last_damage_source", "none") or "none"),
    }


def _target_distance_progress(distances: list[float]) -> float:
    """Closeness to the agent's CURRENT objective at the FINAL step, normalized by the
    initial distance. Uses the terminal distance, NOT the best-ever minimum: the old min
    form saturated to ~1.0 the instant the agent touched its first objective (the target
    then switches and the distance jumps), badly overstating competence. 1.0 = ended on the
    objective; 0.0 = ended at/beyond the starting distance."""
    if not distances:
        return 0.0
    initial = distances[0]
    final = distances[-1]
    if initial > 1e-6:
        return float(np.clip(1.0 - final / initial, 0.0, 1.0))
    return 1.0 if final <= 1e-6 else 0.0


def paired_ab_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paired Crystal Caves checkpoint A/B eval.")
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--b-checkpoint", required=True)
    parser.add_argument("--a-label", default="a")
    parser.add_argument("--b-label", default="b")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--metric", default=DEFAULT_AB_METRIC)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--label", default="paired_ab")
    parser.add_argument("--out-dir", default=".Codex/artifacts/cc_sessions")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--report-seconds", type=float, default=15.0)
    opts = parser.parse_args(argv)
    seeds = _parse_seeds(opts.seeds)
    if opts.games <= 0:
        parser.error("--games must be positive")
    if not seeds:
        parser.error("--seeds must include at least one integer")

    out_dir = Path(opts.out_dir) / timestamp_id(opts.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_a = ArmSpec(opts.a_label, Path(opts.a_checkpoint))
    arm_b = ArmSpec(opts.b_label, Path(opts.b_checkpoint))
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in (arm_a, arm_b):
            rows = evaluate_checkpoint_rows(
                arm,
                out_dir=out_dir,
                seed=seed,
                games=opts.games,
                max_steps=opts.max_steps or None,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
            )
            for row in rows:
                append_jsonl(out_dir / "rows.jsonl", row)
            all_rows.extend(rows)

    a_rows = [row for row in all_rows if row["arm"] == arm_a.label]
    b_rows = [row for row in all_rows if row["arm"] == arm_b.label]
    paired_rows = pair_level_rows(a_rows, b_rows, metric=opts.metric)
    for row in paired_rows:
        append_jsonl(out_dir / "paired_rows.jsonl", row)
    aggregate = aggregate_paired_ab(
        a_rows,
        b_rows,
        metric=opts.metric,
        n_bootstrap=opts.bootstrap_samples,
        bootstrap_seed=opts.bootstrap_seed,
    )
    summary = {
        "mode": "paired-ab",
        "created_at": datetime.now().isoformat(),
        "out_dir": str(out_dir),
        "arms": {
            "a": {"label": arm_a.label, "checkpoint": str(arm_a.checkpoint)},
            "b": {"label": arm_b.label, "checkpoint": str(arm_b.checkpoint)},
        },
        "seeds": seeds,
        "games": opts.games,
        "metric": opts.metric,
        "bootstrap_samples": opts.bootstrap_samples,
        "aggregate": aggregate,
        "rows_path": str(out_dir / "rows.jsonl"),
        "paired_rows_path": str(out_dir / "paired_rows.jsonl"),
    }
    write_json(out_dir / "summary.json", summary)
    _write_report(out_dir / "report.md", summary)
    print(_summary_line(summary))
    print(f"Wrote paired A/B artifacts to {out_dir}")
    return 0


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _summary_line(summary: dict[str, Any]) -> str:
    delta = summary["aggregate"]["paired_delta_b_minus_a"]
    return (
        f"paired-ab {summary['arms']['b']['label']} - {summary['arms']['a']['label']} "
        f"{summary['metric']} mean delta {delta['mean']:.4f} "
        f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]"
    )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    delta = aggregate["paired_delta_b_minus_a"]
    lines = [
        "# Paired Crystal Caves A/B",
        "",
        f"- Metric: `{summary['metric']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Games per seed: `{summary['games']}`",
        f"- A: `{summary['arms']['a']['label']}`",
        f"- B: `{summary['arms']['b']['label']}`",
        "",
        "## Mean",
        "",
        f"- A mean: `{aggregate['arm_a']['mean']:.4f}`",
        f"- B mean: `{aggregate['arm_b']['mean']:.4f}`",
        (
            f"- Paired delta B-A mean: `{delta['mean']:.4f}` "
            f"95% CI `[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]`"
        ),
        "",
        f"- Rows: `{summary['rows_path']}`",
        f"- Paired rows: `{summary['paired_rows_path']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "aggregate_paired_ab",
    "interquartile_mean",
    "pipeline_mean",
    "paired_ab_main",
    "pair_level_rows",
    "stratified_bootstrap_ci",
]
