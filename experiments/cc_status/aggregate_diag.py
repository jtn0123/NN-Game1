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
    _print_death_trace,
    _print_leg2,
    _print_report,
    _print_stall_trace,
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
    seeds = sorted({int(pt["seed"]) for pt in curve})
    metrics = (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS)
    # Audit R2-C: only pick the best checkpoint from buckets that have ALL seeds. Under the
    # multi-process aggregate path a crashed/straggling seed leaves ragged buckets; without
    # this guard max() could report a single lucky seed as the cross-seed result.
    full_buckets = [pt for pt in curve_avg if pt.get("n_seeds", 1) == len(seeds)]
    if not full_buckets:
        max_seeds = max(pt.get("n_seeds", 1) for pt in curve_avg)
        full_buckets = [pt for pt in curve_avg if pt.get("n_seeds", 1) == max_seeds]
        print(
            f"[aggregate_diag] WARNING: no checkpoint bucket contains all {len(seeds)} "
            f"seeds; using buckets with n_seeds={max_seeds} only",
            flush=True,
        )
    # The FINAL bucket gets the same guard: the last milestone of a crashed seed is
    # missing, so curve_avg[-1] could silently be a partial-seed bucket while the
    # header still claims all seeds (metric-audit finding #1).
    final = full_buckets[-1]
    if final is not curve_avg[-1]:
        print(
            "[aggregate_diag] WARNING: final checkpoint bucket is ragged; reporting the "
            f"last bucket with n_seeds={final.get('n_seeds')} "
            f"(episode {final.get('episode')}) as FINAL",
            flush=True,
        )
    # Homogeneity check: the aggregate copies run params from the FIRST summary; if the
    # per-seed runs disagree, the header would misdescribe the data (finding #1b).
    for key in ("difficulty", "episodes", "games", "pool_size"):
        values = {repr(s.get(key)) for s in summaries}
        if len(values) > 1:
            print(
                f"[aggregate_diag] WARNING: per-seed runs disagree on {key!r}: "
                f"{sorted(values)} — header reports the first",
                flush=True,
            )
    best = max(full_buckets, key=lambda pt: (pt["train"]["won"], pt["train"]["crystal_frac"]))

    base = summaries[0]
    gap_final = {m: round(final["train"][m] - final["test"][m], 4) for m in metrics}
    agg = {
        "difficulty": base.get("difficulty"),
        "episodes": base.get("episodes"),
        "seeds": seeds,
        "games": base.get("games"),
        "pool_size": base.get("pool_size"),
        "truncation_bootstrap": base.get("truncation_bootstrap"),
        "train": final["train"],
        "test": final["test"],
        "gap_train_minus_test": gap_final,
        # Audit B5 parity: emit final + best gaps like the single-process path.
        "gap_train_minus_test_final": gap_final,
        "gap_train_minus_test_best": {
            m: round(best["train"].get(m, 0.0) - best["test"].get(m, 0.0), 4)
            for m in metrics
            if m in best["train"] and m in best["test"]
        },
        # Audit R2-C: carry the B3 regenerate/holdout gate so the aggregate report doesn't
        # re-run the (meaningless) memorisation verdict when the train split is held-out.
        "train_split_is_holdout": any(bool(s.get("train_split_is_holdout")) for s in summaries),
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
    # Seed-average the death trace across per-seed workers. Each single-seed run stores its
    # own `death_trace` (RUN-16 survival lever picker); without this the multi-process path
    # dropped it and the trace had to be averaged by hand. Mirror diagnose_gap's mean-over-
    # union-of-keys so the aggregated trace matches the single-process --seeds output.
    death_traces = [s["death_trace"] for s in summaries if isinstance(s.get("death_trace"), dict)]
    if death_traces:
        keys = sorted({k for dt in death_traces for k in dt})
        agg["death_trace"] = {
            k: sum(float(dt.get(k, 0.0)) for dt in death_traces) / len(death_traces) for k in keys
        }
        agg["death_trace_per_seed"] = death_traces
    # Same seed-averaging for the stall trace (RUN-19 stall diagnostic).
    stall_traces = [s["stall_trace"] for s in summaries if isinstance(s.get("stall_trace"), dict)]
    if stall_traces:
        keys = sorted({k for st in stall_traces for k in st})
        agg["stall_trace"] = {
            k: sum(float(st.get(k, 0.0)) for st in stall_traces) / len(stall_traces) for k in keys
        }
        agg["stall_trace_per_seed"] = stall_traces
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
    _print_death_trace(agg)
    _print_stall_trace(agg)
    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"\nWrote aggregated summary to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
