"""Aggregate per-seed diagnose_gap results into one seed-averaged report.

Use this when `diagnose_gap` was run as **one process per seed** (multicore
parallelism on a many-core box) instead of one process with `--seeds 0,1,2`. Each
single-seed run writes its own `diagnosis.json`; this reproduces the exact
seed-averaging that `diagnose_gap`'s built-in multi-seed path does:

    python -m experiments.cc_status.aggregate_diag \
        scratchpad/m4_learnceiling_parallel/seed_0/diagnosis.json \
        scratchpad/m4_learnceiling_parallel/seed_1/diagnosis.json \
        scratchpad/m4_learnceiling_parallel/seed_2/diagnosis.json

It concatenates each run's per-(seed, episode) curve, averages per checkpoint
across seeds, picks the best checkpoint on seed-averaged TRAINING competence, and
prints the same LEARNING CURVE + PHASE 0 DIAGNOSIS blocks (now also covering the
meanQ column). Identical method to the single-process output, so results are
directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cc_status.diagnose_gap import (  # noqa: E402
    _MEAN_SURROGATE_METRICS,
    _RATE_METRICS,
    _average_curve,
    _print_curve,
    _print_leg2,
    _print_report,
)


def aggregate(paths: list[str]) -> dict[str, Any]:
    """Load per-seed diagnosis.json files and seed-average their curves."""
    summaries: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as handle:
            summaries.append(json.load(handle))
    if not summaries:
        raise SystemExit("no diagnosis.json inputs given")

    curve: list[dict[str, Any]] = []
    for summary in summaries:
        curve.extend(summary.get("curve", []))
    if not curve:
        raise SystemExit(
            "inputs contain no per-checkpoint 'curve' data — were these run with "
            "--checkpoint-every > 0?"
        )

    curve_avg = _average_curve(curve)
    best = max(curve_avg, key=lambda pt: (pt["train"]["won"], pt["train"]["crystal_frac"]))
    final = curve_avg[-1]
    seeds = sorted({int(pt["seed"]) for pt in curve})
    metrics = (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS)

    base = summaries[0]
    agg = {
        "difficulty": base.get("difficulty"),
        "episodes": base.get("episodes"),
        "seeds": seeds,
        "games": base.get("games"),
        "pool_size": base.get("pool_size"),
        "truncation_bootstrap": base.get("truncation_bootstrap"),
        "train": final["train"],
        "test": final["test"],
        "gap_train_minus_test": {
            m: round(final["train"][m] - final["test"][m], 4) for m in metrics
        },
        "curve": curve,
        "curve_avg": curve_avg,
        "best": best,
    }
    # Seed-average the leg-2 probe across per-seed workers (each single-seed run stored
    # its own leg2_reach_rate), so the aggregated report carries the same route-to-exit
    # number the single-process --seeds path would have.
    leg2_per_seed = [float(s["leg2_reach_rate"]) for s in summaries if "leg2_reach_rate" in s]
    if leg2_per_seed:
        agg["leg2_reach_rate"] = sum(leg2_per_seed) / len(leg2_per_seed)
        agg["leg2_reach_rate_per_seed"] = leg2_per_seed
    far_per_seed = [
        float(s["leg2_far_reach_rate"]) for s in summaries if "leg2_far_reach_rate" in s
    ]
    if far_per_seed:
        agg["leg2_far_reach_rate"] = sum(far_per_seed) / len(far_per_seed)
        agg["leg2_far_reach_rate_per_seed"] = far_per_seed
        dists = [float(s["leg2_far_mean_dist"]) for s in summaries if "leg2_far_mean_dist" in s]
        if dists:
            agg["leg2_far_mean_dist"] = sum(dists) / len(dists)
    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed-average per-seed diagnose_gap diagnosis.json files."
    )
    parser.add_argument("paths", nargs="+", help="diagnosis.json files, one per seed run")
    parser.add_argument(
        "--out", default=None, help="optional path to write aggregated summary JSON"
    )
    args = parser.parse_args(argv)

    agg = aggregate(args.paths)
    _print_curve(agg["curve_avg"])
    _print_report(agg)
    _print_leg2(agg)
    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"\nWrote aggregated summary to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
