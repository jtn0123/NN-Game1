"""Phase 0 diagnostic: measure the TRAIN vs HELD-OUT generalisation gap.

The whole point: a 0% held-out score has two very different causes that need
opposite fixes, and we cannot currently tell them apart because the harness only
ever scores held-out levels. This script scores the SAME trained agent on:

  * its TRAINING levels  (the caves it learned on), and
  * fresh HELD-OUT levels (disjoint, never seen).

Reading the result:
  * train HIGH, held-out LOW   -> MEMORISATION (generalisation problem)
  * train LOW,  held-out LOW   -> it never learned the skill (learning problem)
  * train HIGH, held-out HIGH  -> it generalises here; push to harder settings

Run (defaults are tuned to be fast + high-signal on this box):

    python -m experiments.cc_status.diagnose_gap \
        --difficulty tutorial --episodes 600 --seeds 0,1 --games 20 --out scratchpad/diag

Nothing here changes training; it only adds a second evaluation pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cc_status.config_helpers import set_seed  # noqa: E402
from experiments.cc_status.evals import (  # noqa: E402
    _enter_greedy_agent_eval,
    _restore_greedy_agent_eval,
)
from experiments.cc_status.io_utils import write_json  # noqa: E402
from experiments.cc_status.lever_ab import make_config  # noqa: E402
from experiments.cc_status.paired_ab import _evaluate_one_level, interquartile_mean  # noqa: E402
from experiments.cc_status.training import (  # noqa: E402
    TUTORIAL_MIN_EPSILON,
    prepare_trainer,
    run_training,
)
from src.ai.evaluator import Evaluator  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402

# Metrics summarised per split. win/exit are rates (means); the rest are IQMs.
_RATE_METRICS = ("won", "exit_unlocked_rate")
_IQM_METRICS = ("crystal_frac", "depth_frac", "target_distance_progress", "selection_score")


def _eval_split(
    trainer: Any,
    config: Any,
    *,
    split: str,
    games: int,
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Greedy-eval the trained agent on one split ('train' or 'test'), one row/level."""
    game = CrystalCaves(config, headless=True)
    if split == "train":
        game.use_train_levels(games)
    else:
        game.use_eval_levels(games)
    game.reset_eval_cursor()
    evaluator = Evaluator(
        game=game,
        agent=trainer.agent,
        config=config,
        log_dir=str(run_dir / f"eval_{split}"),
    )
    step_limit = int(config.EVAL_MAX_STEPS)
    rows: list[dict[str, Any]] = []
    agent_state = _enter_greedy_agent_eval(trainer.agent)
    try:
        for level_index in range(games):
            row = _evaluate_one_level(
                game,
                trainer.agent,
                evaluator,
                arm="baseline",
                seed=0,
                level_index=level_index,
                max_steps=step_limit,
            )
            row["split"] = split
            rows.append(row)
    finally:
        _restore_greedy_agent_eval(trainer.agent, agent_state)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Per-split summary: rates for win/exit, IQM for the continuous surrogates."""
    out: dict[str, float] = {"n": float(len(rows))}
    if not rows:
        for metric in (*_RATE_METRICS, *_IQM_METRICS):
            out[metric] = 0.0
        return out
    for metric in _RATE_METRICS:
        out[metric] = float(np.mean([float(bool(r.get(metric, False))) for r in rows]))
    for metric in _IQM_METRICS:
        out[metric] = interquartile_mean([float(r.get(metric, 0.0) or 0.0) for r in rows])
    return out


def _milestones(episodes: int, checkpoint_every: int) -> list[int]:
    """Episode counts at which to evaluate; always includes the final episode."""
    if checkpoint_every <= 0:
        return [episodes]
    ms = list(range(checkpoint_every, episodes + 1, checkpoint_every))
    if not ms or ms[-1] != episodes:
        ms.append(episodes)
    return ms


def run_diagnosis(
    *,
    difficulty: str,
    episodes: int,
    seeds: list[int],
    games: int,
    vec_envs: int,
    pool_size: int | None,
    out_dir: Path,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, object] = {}
    if pool_size is not None:
        overrides["CRYSTAL_CAVES_POOL_SIZE"] = pool_size

    train_rows_all: list[dict[str, Any]] = []
    test_rows_all: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    for seed in seeds:
        print(
            f"\n===== baseline seed={seed} (difficulty={difficulty}, episodes={episodes}) =====",
            flush=True,
        )
        set_seed(seed)
        config = make_config(overrides, difficulty=difficulty)
        config.CRYSTAL_CAVES_SEED = seed
        run_dir = out_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer = prepare_trainer(config, episodes=episodes, vec_envs=vec_envs)
        if trainer.agent.epsilon < TUTORIAL_MIN_EPSILON:
            trainer.agent.epsilon = TUTORIAL_MIN_EPSILON

        # Train in segments to each milestone, grading both splits at each so we can
        # see whether competence is rising, flat, or never starts (slow vs stuck).
        train_rows: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        for milestone in _milestones(episodes, checkpoint_every):
            run_training(
                trainer,
                run_dir=run_dir,
                label=f"diag/seed_{seed}",
                total_episodes=episodes,
                heartbeat_seconds=0.0,
                target_episodes=milestone if checkpoint_every > 0 else None,
            )
            train_rows = _eval_split(trainer, config, split="train", games=games, run_dir=run_dir)
            test_rows = _eval_split(trainer, config, split="test", games=games, run_dir=run_dir)
            tr, te = _aggregate(train_rows), _aggregate(test_rows)
            if checkpoint_every > 0:
                curve.append({"seed": seed, "episode": milestone, "train": tr, "test": te})
                print(
                    f"[seed {seed} @ep{milestone}] "
                    f"TRAIN win={tr['won']:.2f} cryst={tr['crystal_frac']:.3f} "
                    f"tgt={tr['target_distance_progress']:.3f} | "
                    f"TEST win={te['won']:.2f} cryst={te['crystal_frac']:.3f} "
                    f"tgt={te['target_distance_progress']:.3f}",
                    flush=True,
                )

        train_rows_all.extend(train_rows)
        test_rows_all.extend(test_rows)
        tr, te = _aggregate(train_rows), _aggregate(test_rows)
        print(
            f"[seed {seed}] FINAL TRAIN win={tr['won']:.2f} cryst={tr['crystal_frac']:.3f} "
            f"tgt={tr['target_distance_progress']:.3f} | "
            f"TEST win={te['won']:.2f} cryst={te['crystal_frac']:.3f} "
            f"tgt={te['target_distance_progress']:.3f}",
            flush=True,
        )

    train_agg, test_agg = _aggregate(train_rows_all), _aggregate(test_rows_all)
    summary = {
        "difficulty": difficulty,
        "episodes": episodes,
        "seeds": seeds,
        "games": games,
        "vec_envs": vec_envs,
        "pool_size": pool_size,
        "train": train_agg,
        "test": test_agg,
        "gap_train_minus_test": {
            m: round(train_agg[m] - test_agg[m], 4) for m in (*_RATE_METRICS, *_IQM_METRICS)
        },
        "curve": curve,
    }
    write_json(out_dir / "diagnosis.json", summary)
    if curve:
        _print_curve(curve)
    _print_report(summary)
    return summary


def _print_curve(curve: list[dict[str, Any]]) -> None:
    """Learning curve: key metrics over training, to separate 'slow' from 'stuck'."""
    print("\n==== LEARNING CURVE (train split over time) ====", flush=True)
    header = (
        "episode".rjust(8)
        + "win".rjust(8)
        + "crystals".rjust(10)
        + "closeness".rjust(11)
        + "exitUnlk".rjust(10)
    )
    print(header, flush=True)
    for point in curve:
        tr = point["train"]
        print(
            f"{point['episode']:8d}"
            + f"{tr['won']:8.2f}"
            + f"{tr['crystal_frac']:10.3f}"
            + f"{tr['target_distance_progress']:11.3f}"
            + f"{tr['exit_unlocked_rate']:10.2f}",
            flush=True,
        )


def _print_report(summary: dict[str, Any]) -> None:
    tr, te = summary["train"], summary["test"]
    gap = summary["gap_train_minus_test"]
    print("\n==== PHASE 0 DIAGNOSIS: train vs held-out ====", flush=True)
    print(
        f"difficulty={summary['difficulty']} seeds={summary['seeds']} "
        f"episodes={summary['episodes']} games/split={summary['games']}",
        flush=True,
    )
    header = "metric".ljust(26) + "TRAIN".rjust(10) + "HELD-OUT".rjust(11) + "GAP".rjust(11)
    print(header, flush=True)
    labels = {
        "won": "win rate",
        "crystal_frac": "crystals collected",
        "depth_frac": "depth reached",
        "target_distance_progress": "closeness to goal",
        "exit_unlocked_rate": "exit unlocked rate",
        "selection_score": "overall score",
    }
    for metric in (
        "won",
        "crystal_frac",
        "depth_frac",
        "target_distance_progress",
        "exit_unlocked_rate",
        "selection_score",
    ):
        print(
            labels[metric].ljust(26)
            + f"{tr[metric]:10.3f}"
            + f"{te[metric]:11.3f}"
            + f"{gap[metric]:+11.3f}",
            flush=True,
        )
    # Plain-English verdict heuristic on the primary learnability signals.
    train_learns = tr["crystal_frac"] > 0.15 or tr["won"] > 0.1
    test_learns = te["crystal_frac"] > 0.15 or te["won"] > 0.1
    print("\nread:", flush=True)
    if not train_learns:
        print(
            "  -> The agent barely solves even its TRAINING levels: this is a LEARNING"
            " problem, not a generalisation one. Fix learnability first.",
            flush=True,
        )
    elif train_learns and not test_learns:
        print(
            "  -> The agent solves TRAINING levels but fails HELD-OUT ones: MEMORISATION."
            " Attack generalisation (more level variety + regularisation).",
            flush=True,
        )
    else:
        print(
            "  -> The agent generalises at this difficulty. Escalate difficulty to find"
            " where it breaks.",
            flush=True,
        )


def _parse_seeds(raw: str) -> list[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 train-vs-test gap diagnostic.")
    parser.add_argument("--difficulty", default="tutorial")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--vec-envs", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, grade train+test every N episodes to plot a learning curve.",
    )
    parser.add_argument("--out", default="scratchpad/diag")
    args = parser.parse_args(argv)
    run_diagnosis(
        difficulty=args.difficulty,
        episodes=args.episodes,
        seeds=_parse_seeds(args.seeds),
        games=args.games,
        vec_envs=args.vec_envs,
        pool_size=args.pool_size,
        out_dir=Path(args.out),
        checkpoint_every=args.checkpoint_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
