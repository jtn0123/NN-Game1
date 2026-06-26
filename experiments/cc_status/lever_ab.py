"""Multi-seed paired A/B harness for the Crystal Caves experiment levers.

This upgrades ``scratchpad/lever_smoke.py`` (single-seed directional smoke) into a
trustworthy tool: for every (arm, seed) it trains a short vec-8 run, greedy-evals
each held-out level, and emits PER-(arm, seed, level_index) rows that mirror the
shape produced by :func:`experiments.cc_status.paired_ab._evaluate_one_level`.
That lets us reuse the rigorous aggregation that already lives in ``paired_ab``
(:func:`aggregate_paired_ab`, :func:`stratified_bootstrap_ci`,
:func:`interquartile_mean`, :func:`pair_level_rows`) instead of reimplementing it.

Each non-baseline arm is compared against the baseline arm with a paired bootstrap
(IQM point estimate + 95% CI on the paired B-A delta), and per-arm IQMs of the
surrogate components are reported alongside a completion / timeout summary.

Difficulty note: the default difficulty is ``easy`` (2-3 crystals). The
``tutorial`` difficulty generates caves with only a single crystal, so levers that
need more than one objective to engage -- notably the reverse-curriculum arms,
which relocate / re-seed around already-collected crystals -- effectively become
no-ops there. Pick ``easy`` (or harder) when A/B-testing those levers.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import Config  # noqa: E402
from experiments.cc_status.config_helpers import set_seed  # noqa: E402
from experiments.cc_status.evals import (  # noqa: E402
    _enter_greedy_agent_eval,
    _restore_greedy_agent_eval,
)
from experiments.cc_status.io_utils import append_jsonl, write_json  # noqa: E402
from experiments.cc_status.paired_ab import (  # noqa: E402
    _evaluate_one_level,
    aggregate_paired_ab,
    interquartile_mean,
)
from experiments.cc_status.training import (  # noqa: E402
    TUTORIAL_MIN_EPSILON,
    prepare_trainer,
    run_training,
)
from src.ai.evaluator import Evaluator  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402

# Default lever arms. ``baseline`` MUST be present (it is the A in every paired A/B).
ARMS: dict[str, dict[str, object]] = {
    "baseline": {},
    "geodesic": {"CRYSTAL_CAVES_GEODESIC_POTENTIAL": True},
    "locked_exit": {"CRYSTAL_CAVES_SHOW_LOCKED_EXIT": True},
    # Annealed reverse curriculum: p decays 0.5 -> 0 over the first 150 episodes so
    # late training is full-from-spawn (a fixed p hurt held-out perf). Pair with
    # --episodes ~200 so the final ~25% of training is on the full task.
    "reverse": {
        "CRYSTAL_CAVES_REVERSE_CURRICULUM": True,
        "CRYSTAL_CAVES_REVERSE_CURRICULUM_P": 0.5,
        "CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES": 150,
    },
    # Fixed-p reverse curriculum (no annealing) -- kept to isolate the annealing effect.
    "reverse_fixed": {
        "CRYSTAL_CAVES_REVERSE_CURRICULUM": True,
        "CRYSTAL_CAVES_REVERSE_CURRICULUM_P": 0.5,
    },
    "relocate": {
        "CRYSTAL_CAVES_REVERSE_CURRICULUM": True,
        "CRYSTAL_CAVES_REVERSE_CURRICULUM_P": 0.5,
        "CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES": 150,
        "CRYSTAL_CAVES_REVERSE_CURRICULUM_RELOCATE": True,
    },
    "ngu": {"CRYSTAL_CAVES_NGU_BONUS": True, "CRYSTAL_CAVES_NGU_BETA": 0.02},
}

BASELINE_ARM = "baseline"

# Surrogate component metrics reported per-arm as IQMs.
PER_ARM_METRICS = (
    "selection_score",
    "crystal_frac",
    "depth_frac",
    "target_distance_progress",
    "exit_unlocked_rate",
    "win",
    "steps",
)

# The paired-delta metric reported with bootstrap CIs.
DELTA_METRIC = "selection_score"


def make_config(overrides: dict[str, object], *, difficulty: str) -> Config:
    """Build a procedural-tutorial config plus the arm's lever overrides.

    Turbo throughput knobs (LEARN_EVERY=8, GRADIENT_STEPS=2, BATCH_SIZE=128) are
    applied identically to every arm, so they cannot bias the A/B comparison.
    """

    config = Config()
    config.GAME_NAME = "crystal_caves"
    config.CRYSTAL_CAVES_PROCEDURAL = True
    config.CRYSTAL_CAVES_DIFFICULTY = difficulty
    config.CRYSTAL_CAVES_FAMILIES = "platform_network"
    config.CRYSTAL_CAVES_POOL_SIZE = 24
    # Turbo-ish throughput, applied equally to every arm to keep the A/B fair.
    config.LEARN_EVERY = 8
    config.GRADIENT_STEPS = 2
    config.BATCH_SIZE = 128
    for key, value in overrides.items():
        setattr(config, key, value)
    config.__post_init__()
    return config


def evaluate_arm_seed_rows(
    trainer: Any,
    config: Config,
    *,
    arm: str,
    seed: int,
    games: int,
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Greedy-eval each held-out level, returning PER-(level) surrogate rows.

    Row shape mirrors ``paired_ab._evaluate_one_level`` exactly (it is reused here)
    so ``aggregate_paired_ab`` can consume the rows directly.
    """

    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    evaluator = Evaluator(
        game=game,
        agent=trainer.agent,
        config=config,
        log_dir=str(run_dir / "eval"),
    )
    step_limit = int(config.EVAL_MAX_STEPS)
    rows: list[dict[str, Any]] = []
    agent_state = _enter_greedy_agent_eval(trainer.agent)
    try:
        for level_index in range(games):
            rows.append(
                _evaluate_one_level(
                    game,
                    trainer.agent,
                    evaluator,
                    arm=arm,
                    seed=seed,
                    level_index=level_index,
                    max_steps=step_limit,
                )
            )
    finally:
        _restore_greedy_agent_eval(trainer.agent, agent_state)
    return rows


def train_and_evaluate_arm_seed(
    arm: str,
    overrides: dict[str, object],
    *,
    seed: int,
    episodes: int,
    games: int,
    difficulty: str,
    vec_envs: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Train one short run for (arm, seed) and return its per-level eval rows.

    Returns an empty list (rather than raising) if anything goes wrong, so a single
    bad arm/seed cannot sink the whole sweep.
    """

    run_dir = out_dir / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        set_seed(seed)
        config = make_config(overrides, difficulty=difficulty)
        config.CRYSTAL_CAVES_SEED = seed
        trainer = prepare_trainer(config, episodes=episodes, vec_envs=vec_envs)
        if trainer.agent.epsilon < TUTORIAL_MIN_EPSILON:
            trainer.agent.epsilon = TUTORIAL_MIN_EPSILON
        run_training(
            trainer,
            run_dir=run_dir,
            label=f"{arm}/seed_{seed}",
            total_episodes=episodes,
            heartbeat_seconds=0.0,
        )
        rows = evaluate_arm_seed_rows(
            trainer,
            config,
            arm=arm,
            seed=seed,
            games=games,
            run_dir=run_dir,
        )
    except Exception as exc:  # noqa: BLE001 - robustness: never crash the sweep
        print(f"[lever-ab] arm={arm} seed={seed} FAILED: {exc}", flush=True)
        return []
    return rows


def per_arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """IQM of each surrogate component plus a completion / timeout breakdown."""

    if not rows:
        return {
            "rows": 0,
            "iqm": {metric: 0.0 for metric in PER_ARM_METRICS},
            "win_rate": 0.0,
            "exit_unlock_rate": 0.0,
            "timeout_frac": 0.0,
            "end_reason_counts": {},
        }
    iqm = {
        metric: interquartile_mean([float(row.get(metric, 0.0) or 0.0) for row in rows])
        for metric in PER_ARM_METRICS
    }
    end_reason_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("end_reason", "unknown") or "unknown")
        end_reason_counts[reason] = end_reason_counts.get(reason, 0) + 1
    total = len(rows)
    return {
        "rows": total,
        "iqm": iqm,
        "win_rate": float(np.mean([bool(row.get("won", False)) for row in rows])),
        "exit_unlock_rate": float(
            np.mean([float(row.get("exit_unlocked_rate", 0.0) or 0.0) for row in rows])
        ),
        "timeout_frac": float(end_reason_counts.get("timeout", 0) / max(1, total)),
        "end_reason_counts": end_reason_counts,
    }


def build_summary(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    arms: list[str],
    seeds: list[int],
    episodes: int,
    games: int,
    difficulty: str,
    vec_envs: int,
    out_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_rows = rows_by_arm.get(BASELINE_ARM, [])
    per_arm = {arm: per_arm_summary(rows_by_arm.get(arm, [])) for arm in arms}

    comparisons: dict[str, Any] = {}
    for arm in arms:
        if arm == BASELINE_ARM:
            continue
        arm_rows = rows_by_arm.get(arm, [])
        if not baseline_rows or not arm_rows:
            comparisons[arm] = {
                "metric": DELTA_METRIC,
                "skipped": True,
                "reason": "no rows for baseline and/or arm",
                "paired_rows": 0,
            }
            continue
        comparisons[arm] = aggregate_paired_ab(
            baseline_rows,
            arm_rows,
            metric=DELTA_METRIC,
            n_bootstrap=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )

    return {
        "mode": "lever-ab",
        "created_at": datetime.now().isoformat(),
        "out_dir": str(out_dir),
        "baseline_arm": BASELINE_ARM,
        "arms": arms,
        "seeds": seeds,
        "episodes": episodes,
        "games": games,
        "difficulty": difficulty,
        "vec_envs": vec_envs,
        "delta_metric": DELTA_METRIC,
        "bootstrap_samples": bootstrap_samples,
        "per_arm": per_arm,
        "comparisons_vs_baseline": comparisons,
        "rows_path": str(out_dir / "rows.jsonl"),
    }


def _fmt_ci(stat: dict[str, Any]) -> str:
    return (
        f"{stat['iqm']:+.4f} [{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}]"
        f" (n={int(stat.get('n', 0))})"
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Crystal Caves Lever A/B (multi-seed, paired)",
        "",
        f"- Created: `{summary['created_at']}`",
        f"- Baseline arm: `{summary['baseline_arm']}`",
        f"- Arms: `{', '.join(summary['arms'])}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Episodes/run: `{summary['episodes']}`  Games/eval: `{summary['games']}`",
        f"- Difficulty: `{summary['difficulty']}`  Vec envs: `{summary['vec_envs']}`",
        f"- Paired delta metric: `{summary['delta_metric']}`",
        f"- Bootstrap samples: `{summary['bootstrap_samples']}`",
        "",
        "> Difficulty note: `tutorial` caves have a single crystal, so reverse-"
        "curriculum levers can't engage. Use `easy`+ for those.",
        "",
        "## Paired delta vs baseline (B - A, IQM + 95% CI)",
        "",
        f"Metric: `{summary['delta_metric']}` -- positive favors the arm over baseline.",
        "",
        "| arm | paired delta (IQM, 95% CI) | arm IQM | baseline IQM | paired rows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in summary["arms"]:
        if arm == summary["baseline_arm"]:
            continue
        comp = summary["comparisons_vs_baseline"].get(arm, {})
        if comp.get("skipped"):
            lines.append(f"| {arm} | _skipped: {comp.get('reason', '')}_ | - | - | 0 |")
            continue
        delta = comp["paired_delta_b_minus_a"]
        arm_iqm = comp["arm_b"]["iqm"]
        base_iqm = comp["arm_a"]["iqm"]
        lines.append(
            f"| {arm} | {_fmt_ci(delta)} | {arm_iqm:.4f} | {base_iqm:.4f} "
            f"| {comp.get('paired_rows', 0)} |"
        )

    lines += [
        "",
        "## Per-arm IQM surrogate components",
        "",
        "| arm | rows | sel_score | crystal_frac | depth_frac | target_prog "
        "| exit_unlock | win | timeout_frac |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in summary["arms"]:
        info = summary["per_arm"].get(arm, {})
        iqm = info.get("iqm", {})
        lines.append(
            f"| {arm} | {info.get('rows', 0)} "
            f"| {iqm.get('selection_score', 0.0):.4f} "
            f"| {iqm.get('crystal_frac', 0.0):.4f} "
            f"| {iqm.get('depth_frac', 0.0):.4f} "
            f"| {iqm.get('target_distance_progress', 0.0):.4f} "
            f"| {info.get('exit_unlock_rate', 0.0):.4f} "
            f"| {info.get('win_rate', 0.0):.4f} "
            f"| {info.get('timeout_frac', 0.0):.4f} |"
        )

    lines += ["", "## End-reason breakdown", ""]
    for arm in summary["arms"]:
        info = summary["per_arm"].get(arm, {})
        counts = info.get("end_reason_counts", {})
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
        lines.append(f"- `{arm}`: {rendered}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(summary: dict[str, Any]) -> None:
    print("\n==== LEVER A/B (multi-seed, paired vs baseline) ====", flush=True)
    print(
        f"baseline={summary['baseline_arm']} seeds={summary['seeds']} "
        f"episodes={summary['episodes']} games={summary['games']} "
        f"difficulty={summary['difficulty']}",
        flush=True,
    )
    header = "arm".ljust(12) + "rows".rjust(6) + "selIQM".rjust(10) + "dSel(B-A) [95% CI]".rjust(34)
    print(header, flush=True)
    for arm in summary["arms"]:
        info = summary["per_arm"].get(arm, {})
        sel = info.get("iqm", {}).get("selection_score", 0.0)
        line = arm.ljust(12) + str(info.get("rows", 0)).rjust(6) + f"{sel:10.4f}"
        if arm != summary["baseline_arm"]:
            comp = summary["comparisons_vs_baseline"].get(arm, {})
            if comp.get("skipped"):
                line += "   (skipped)".rjust(34)
            else:
                d = comp["paired_delta_b_minus_a"]
                line += f"   {d['iqm']:+.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"
        print(line, flush=True)


def run_lever_ab(
    *,
    arms: list[str],
    seeds: list[int],
    episodes: int,
    games: int,
    difficulty: str,
    vec_envs: int,
    out_dir: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for seed in seeds:
        for arm in arms:
            print(
                f"\n===== ARM {arm} seed={seed} " f"(episodes={episodes}, games={games}) =====",
                flush=True,
            )
            rows = train_and_evaluate_arm_seed(
                arm,
                ARMS[arm],
                seed=seed,
                episodes=episodes,
                games=games,
                difficulty=difficulty,
                vec_envs=vec_envs,
                out_dir=out_dir,
            )
            for row in rows:
                append_jsonl(rows_path, row)
            rows_by_arm[arm].extend(rows)
            print(f"[{arm} seed={seed}] {len(rows)} rows", flush=True)

    summary = build_summary(
        rows_by_arm,
        arms=arms,
        seeds=seeds,
        episodes=episodes,
        games=games,
        difficulty=difficulty,
        vec_envs=vec_envs,
        out_dir=out_dir,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    print_table(summary)
    print(f"\nWrote lever A/B artifacts to {out_dir}", flush=True)
    return summary


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-seed paired A/B for Crystal Caves levers.")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--vec-envs", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out", default="scratchpad/lever_ab_results")
    opts = parser.parse_args(argv)

    arms = [a.strip() for a in opts.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {unknown}. Known: {sorted(ARMS)}")
    if BASELINE_ARM not in arms:
        arms = [BASELINE_ARM] + arms
    else:
        # Ensure baseline is first so it's trained/printed before comparisons.
        arms = [BASELINE_ARM] + [a for a in arms if a != BASELINE_ARM]

    seeds = _parse_seeds(opts.seeds)
    if not seeds:
        parser.error("--seeds must include at least one integer")
    if opts.games <= 0:
        parser.error("--games must be positive")
    if opts.episodes <= 0:
        parser.error("--episodes must be positive")

    run_lever_ab(
        arms=arms,
        seeds=seeds,
        episodes=opts.episodes,
        games=opts.games,
        difficulty=opts.difficulty,
        vec_envs=opts.vec_envs,
        out_dir=Path(opts.out),
        bootstrap_samples=opts.bootstrap_samples,
        bootstrap_seed=opts.bootstrap_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
