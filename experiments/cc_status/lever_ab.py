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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    pair_level_rows,
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
    # Truncation-aware bootstrapping: timeouts/stalls stored as non-terminal so the
    # value still bootstraps the final state (Pardo et al. 2018). Cheap, budget-free.
    "truncation_fix": {"CRYSTAL_CAVES_TRUNCATION_BOOTSTRAP": True},
    # --- 2x2 representation x diversity (baseline reward only) ---
    # 'baseline' above is the control: flat MLP, pool=24 (make_config default).
    "mlp_p256": {"CRYSTAL_CAVES_POOL_SIZE": 256},
    "cnn_p24": {"USE_CNN_STATE": True, "CRYSTAL_CAVES_CNN_GLOBAL_POOL": True},
    "cnn_p256": {
        "USE_CNN_STATE": True,
        "CRYSTAL_CAVES_CNN_GLOBAL_POOL": True,
        "CRYSTAL_CAVES_POOL_SIZE": 256,
    },
}

BASELINE_ARM = "baseline"

# Surrogate component metrics reported per-arm as IQMs.
PER_ARM_METRICS = (
    "selection_score",
    "crystal_frac",
    "depth_frac",
    "target_distance_progress",
    "exit_unlocked_rate",
    "won",
    "steps",
)

# The paired-delta metric reported with bootstrap CIs. Primary is a CONTINUOUS,
# non-saturated surrogate (selection_score's crystal/win components are ~0 at this
# budget, so it carries little signal); we also report deltas on these others.
DELTA_METRIC = "target_distance_progress"
SURROGATE_DELTA_METRICS = ("target_distance_progress", "depth_frac", "selection_score")


def paired_row_ci(
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """IQM + 95% CI of the paired (B-A) delta, resampling the PAIRED (seed,level)
    ROWS with replacement -- not the 3 seed-groups. The seed-grouped bootstrap in
    paired_ab effectively has n=3 (inflated CIs); pairing by level gives ~seeds*games
    units, the correct resample population for a paired design."""
    paired = pair_level_rows(a_rows, b_rows, metric=metric)
    deltas = [float(p.get(f"delta_{metric}", 0.0) or 0.0) for p in paired]
    if not deltas:
        return {"iqm": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    point = interquartile_mean(deltas)
    rng = np.random.default_rng(seed)
    n = len(deltas)
    samples = []
    for _ in range(max(1, n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        samples.append(interquartile_mean([deltas[i] for i in idx]))
    return {
        "iqm": float(point),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n": n,
    }


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
        comp = aggregate_paired_ab(
            baseline_rows,
            arm_rows,
            metric=DELTA_METRIC,
            n_bootstrap=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if not comp.get("paired_rows"):
            # Both arms have rows but no (seed, level_index) overlap -> nothing to pair;
            # mark skipped rather than emit bogus zero-valued CIs.
            comparisons[arm] = {
                "metric": DELTA_METRIC,
                "skipped": True,
                "reason": "no matched seed/level rows for baseline and arm",
                "paired_rows": 0,
            }
            continue
        # Correct (paired-row) bootstrap on each surrogate metric, in addition to the
        # seed-grouped one above (kept for reference / back-compat).
        comp["paired_row"] = {
            metric: paired_row_ci(
                baseline_rows,
                arm_rows,
                metric=metric,
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed + i,
            )
            for i, metric in enumerate(SURROGATE_DELTA_METRICS)
        }
        comparisons[arm] = comp

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
        "## Paired delta vs baseline (B - A, IQM + 95% CI, paired-row bootstrap)",
        "",
        "Resamples paired (seed, level) rows -- positive favors the arm. Columns are "
        "the surrogate metrics; `target_distance_progress` is the primary "
        "(non-saturated) signal.",
        "",
        "| arm | target_progress Δ | depth_frac Δ | selection_score Δ |",
        "| --- | --- | --- | --- |",
    ]
    for arm in summary["arms"]:
        if arm == summary["baseline_arm"]:
            continue
        comp = summary["comparisons_vs_baseline"].get(arm, {})
        if comp.get("skipped"):
            lines.append(f"| {arm} | _skipped: {comp.get('reason', '')}_ | - | - |")
            continue
        pr = comp.get("paired_row", {})
        cells = [
            _fmt_ci(pr[m]) if m in pr else "-"
            for m in ("target_distance_progress", "depth_frac", "selection_score")
        ]
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} |")

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


def _format_duration(seconds: float) -> str:
    """Render a seconds duration as a compact ``2h44m`` / ``37m`` / ``45s`` string."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def print_running_ab(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    arms: list[str],
    completed: int,
    total: int,
    elapsed: float,
    last_label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    """Print a partial paired A/B snapshot after each (arm, seed) run completes.

    Lets a long sweep be watched live instead of blind: the per-arm IQMs and the
    paired delta + 95% CI vs baseline appear immediately and tighten as more seeds
    land. The numbers are PARTIAL until the sweep finishes -- early rows are noisy
    and the CI is wide -- so treat them as a progress indicator, not a verdict.
    """
    eta = (elapsed / completed) * (total - completed) if completed else 0.0
    print(
        f"\n[progress {completed}/{total} runs | elapsed {_format_duration(elapsed)}"
        f" | eta ~{_format_duration(eta)} | last: {last_label}]",
        flush=True,
    )
    print(f"  running paired A/B on '{DELTA_METRIC}' (partial, B - baseline):", flush=True)
    header = (
        "  " + "arm".ljust(16) + "rows".rjust(6) + "tgtIQM".rjust(10) + "crystIQM".rjust(10)
    ) + "   dTgt(B-A) [95% CI]"
    print(header, flush=True)
    baseline_rows = rows_by_arm.get(BASELINE_ARM, [])
    for arm in arms:
        arm_rows = rows_by_arm.get(arm, [])
        tgt = (
            interquartile_mean([float(r.get(DELTA_METRIC, 0.0) or 0.0) for r in arm_rows])
            if arm_rows
            else 0.0
        )
        cryst = (
            interquartile_mean([float(r.get("crystal_frac", 0.0) or 0.0) for r in arm_rows])
            if arm_rows
            else 0.0
        )
        line = (
            "  " + arm.ljust(16) + str(len(arm_rows)).rjust(6) + f"{tgt:10.4f}" + f"{cryst:10.4f}"
        )
        if arm != BASELINE_ARM:
            if baseline_rows and arm_rows:
                ci = paired_row_ci(
                    baseline_rows,
                    arm_rows,
                    metric=DELTA_METRIC,
                    n_bootstrap=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                if ci["n"]:
                    line += (
                        f"   {ci['iqm']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]"
                        f"  (n_pairs={ci['n']})"
                    )
                else:
                    line += "   (no paired rows yet)"
            else:
                line += "   (waiting for rows)"
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
    # Start each sweep from a clean rows file so a reused --out can't mix in stale
    # rows (the summary only reflects this run's in-memory rows).
    if rows_path.exists():
        rows_path.unlink()

    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    total_runs = len(seeds) * len(arms)
    completed_runs = 0
    sweep_start = time.monotonic()
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
            completed_runs += 1
            print(f"[{arm} seed={seed}] {len(rows)} rows", flush=True)
            # Live, partial paired snapshot so a multi-hour sweep is never blind.
            print_running_ab(
                rows_by_arm,
                arms=arms,
                completed=completed_runs,
                total=total_runs,
                elapsed=time.monotonic() - sweep_start,
                last_label=f"{arm}/seed_{seed}",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )

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
    # Dedupe (preserve order) so a repeated arm can't append rows twice and bias it.
    seen_arms: set[str] = set()
    arms = [a for a in arms if not (a in seen_arms or seen_arms.add(a))]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {unknown}. Known: {sorted(ARMS)}")
    if BASELINE_ARM not in arms:
        arms = [BASELINE_ARM, *arms]
    else:
        # Ensure baseline is first so it's trained/printed before comparisons.
        arms = [BASELINE_ARM, *[a for a in arms if a != BASELINE_ARM]]

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
