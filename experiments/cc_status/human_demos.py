"""Verify and summarize human-recorded demos.

Reads the JSON episodes written by ``src/app/demo_recorder.py`` (via
``python main.py --human --imported --record-demos``) and produces the two
Phase-0 deliverables:

1. Difficulty ground truth — attempts / wins / loss reasons per level: any level
   a human can't beat in a handful of tries is objectively over-tuned.
2. A verified demo dataset — every WON episode is replayed open-loop with
   ``demo_extract.verify_stored``; only reproducible wins count as
   demonstrations for demo-seeded training.

Run:  python -m experiments.cc_status.human_demos [--dir DIR] [--no-verify]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_DIR = _REPO_ROOT / "experiments" / "cc_status" / "demos" / "human"


def load_demos(demo_dir: Path) -> List[Dict]:
    records = []
    for path in sorted(demo_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"SKIP unreadable {path.name}: {exc}")
            continue
        rec["_path"] = path
        records.append(rec)
    return records


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the open-loop replay of won episodes"
    )
    args = parser.parse_args(argv)

    records = load_demos(args.dir)
    if not records:
        print(f"no demos found in {args.dir}")
        return 1

    by_level: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        key = (
            f"L{rec['level_index']:02d} {rec['level_name']}"
            if rec.get("level_index") is not None
            else f"Lxx {rec.get('level_name', '?')}"
        )
        by_level[key].append(rec)

    verified = 0
    failed: List[str] = []
    print(f"{'level':<32} {'tries':>5} {'wins':>4}  loss reasons")
    for key in sorted(by_level):
        recs = by_level[key]
        wins = [r for r in recs if r.get("won")]
        losses = defaultdict(int)
        for r in recs:
            if not r.get("won"):
                losses[r.get("end_reason", "?")] += 1
        loss_txt = ", ".join(f"{k}×{v}" for k, v in sorted(losses.items())) or "-"
        print(f"{key:<32} {len(recs):>5} {len(wins):>4}  {loss_txt}")

    won_records = [r for r in records if r.get("won") and r.get("level_index") is not None]
    if not args.no_verify and won_records:
        from experiments.cc_status.demo_extract import verify_stored

        print(f"\nreplaying {len(won_records)} won episode(s) open-loop:")
        for rec in won_records:
            ok = verify_stored(int(rec["level_index"]), [int(a) for a in rec["actions"]])
            tag = "VERIFIED" if ok else "REPLAY-MISMATCH"
            if ok:
                verified += 1
            else:
                failed.append(rec["_path"].name)
            print(f"  {tag}  {rec['_path'].name}")
        print(f"\n{verified}/{len(won_records)} wins verified as replayable demonstrations")
        if failed:
            print("replay mismatches (non-deterministic setup? curriculum flags on during play?):")
            for name in failed:
                print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
