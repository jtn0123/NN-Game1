"""How do episodes END across learning? Percentages of stalled / timeout /
killed / won at every training milestone, per run and arm — the data for the
fidelity question: the 1991 game has NO episode clock and NO stall executioner;
both are training-harness inventions, so if a large, non-shrinking share of
episodes die to those artificial timers, the harness (not the agent or the
levels) is capping performance.

Consumes the per-milestone evaluation rows the canonical runner persists
(``per_level_rows.jsonl``, one JSON row per level per milestone per seed with
an ``end_reason`` field). Point it at one or many run directories:

    python -m experiments.cc_status.analyze_end_reasons runs/RUN-25 runs/RUN-24

Output per (run, arm, split): a milestone table of end-reason percentages plus
a first-vs-last trend verdict per reason (SHRINKING / FLAT / GROWING).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REASONS = ("won", "killed", "stalled", "timeout", "first_crystal_goal", "unknown")


def load_rows(root: Path) -> Dict[Tuple[str, str], List[dict]]:
    """(arm-dir, split) -> rows. The arm is the path segment under the given root
    (A/B/C/D for lever runs; '.' for single-arm runs)."""
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for path in sorted(root.rglob("per_level_rows.jsonl")):
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        arm = rel.parts[0] if len(rel.parts) > 2 else "."
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            grouped[(arm, str(row.get("split", "?")))].append(row)
    return grouped


def milestone_table(rows: List[dict]) -> List[dict]:
    by_ms: Dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        reason = str(row.get("end_reason", "unknown"))
        if reason not in REASONS:
            reason = "unknown"
        by_ms[int(row.get("episode", 0))][reason] += 1
    table = []
    for episode in sorted(by_ms):
        counts = by_ms[episode]
        total = sum(counts.values())
        table.append(
            {
                "episode": episode,
                "n": total,
                **{r: counts.get(r, 0) / total for r in REASONS},
            }
        )
    return table


def trend(first: float, last: float) -> str:
    if last < first - 0.05:
        return "SHRINKING"
    if last > first + 0.05:
        return "GROWING"
    return "FLAT"


def report(root: Path) -> None:
    grouped = load_rows(root)
    if not grouped:
        print(f"{root}: no per_level_rows.jsonl found")
        return
    for (arm, split), rows in sorted(grouped.items()):
        table = milestone_table(rows)
        if not table:
            continue
        print(f"\n== {root.name} arm={arm} split={split} ==")
        print(f"{'episode':>8} {'n':>5} " + " ".join(f"{r[:7]:>8}" for r in REASONS))
        for entry in table:
            print(
                f"{entry['episode']:>8} {entry['n']:>5} "
                + " ".join(f"{entry[r]:>8.3f}" for r in REASONS)
            )
        first, last = table[0], table[-1]
        verdicts = ", ".join(
            f"{r}={trend(first[r], last[r])}({first[r]:.2f}->{last[r]:.2f})"
            for r in ("killed", "stalled", "timeout", "won")
        )
        print(f"  trend first->last: {verdicts}")
        harness_share = last["stalled"] + last["timeout"]
        print(
            f"  HARNESS-TIMER share at final (stalled+timeout): {harness_share:.3f} "
            "(episodes ended by rules the 1991 game does not have)"
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for root in args.roots:
        report(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
