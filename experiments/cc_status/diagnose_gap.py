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
from experiments.cc_status.paired_ab import _evaluate_one_level  # noqa: E402
from experiments.cc_status.training import (  # noqa: E402
    TUTORIAL_MIN_EPSILON,
    capture_weight_snapshot,
    prepare_trainer,
    run_training,
)
from experiments.cc_status.vec_envs import apply_reverse_start  # noqa: E402
from src.ai.evaluator import Evaluator  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402

# Metrics summarised per split. `won` and `exit_unlocked_rate` are per-level binary
# outcomes → mean of the bool = a true rate. `crystal_frac` is a 0..1 collection FRACTION
# → aggregated as a TRUE MEAN in _aggregate (NOT bool: the bool form reported a
# collected-≥1 rate, e.g. normal 0.83 while the real mean is ~0.20; RUN-16 death-trace).
# The continuous chain surrogates use a PLAIN MEAN (previously an interquartile mean that
# discarded the outer 50% and floored bottom-heavy distributions — a true mean is honest
# and matches what a reader assumes). target_distance_progress is now a FINAL-step closeness
# (see paired_ab), not best-ever, which used to saturate to ~1.0 on the first objective.
_RATE_METRICS = ("won", "exit_unlocked_rate", "crystal_frac")
_MEAN_SURROGATE_METRICS = ("depth_frac", "target_distance_progress", "selection_score")


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


def _mean_q(trainer: Any, config: Any, *, games: int) -> float:
    """Mean of max-Q over the first `games` training-level start states. Tracks value
    magnitude over training: a steady drift toward large-negative is the Q-divergence
    signature the instability investigation flagged."""
    game = CrystalCaves(config, headless=True)
    game.use_train_levels(games)
    game.reset_eval_cursor()
    agent = trainer.agent
    qmax: list[float] = []
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for _ in range(games):
            state = game.reset()
            qmax.append(float(np.max(agent.get_q_values(state))))
    finally:
        _restore_greedy_agent_eval(agent, agent_state)
    return float(np.mean(qmax)) if qmax else 0.0


def _eval_leg2(
    trainer: Any, config: Any, *, games: int, mode: str = "reverse_exit"
) -> dict[str, float]:
    """Leg-2 probe: on each HELD-OUT level, drop the (greedy) trained agent in the
    post-collection state (all crystals cleared, exit unlocked), then measure whether it
    reaches the exit. 'reached' == won, since touching the open exit with no crystals left
    wins.

    mode="reverse_exit" places the agent right NEXT TO the exit — this only isolates the
    trivial final hop. mode="reverse_exit_far" places it at a RANDOM reachable standing
    tile a real distance away, isolating the genuine long-range route-to-exit skill (the
    actual leg-2 wall). Comparing the two separates "can't do the last step" (both low)
    from "can't navigate to it" (near high, far low) from "ceiling is elsewhere/leg-1"
    (both high while full-play win stays low). Also returns the mean start->exit distance.
    """
    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    agent = trainer.agent
    step_limit = int(config.EVAL_MAX_STEPS)
    reached: list[float] = []
    distances: list[float] = []
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for _ in range(games):
            game.reset()
            if not apply_reverse_start(game, mode):
                continue  # no oracle-verified placement on this level; skip (not counted)
            col, row = game._player_tile()
            ex, ey = game.exit_pos
            distances.append(float(abs(col - ex) + abs(row - ey)))
            state = game.get_state()
            done = False
            steps = 0
            info: dict[str, Any] = {}
            while not done and steps < step_limit:
                action = agent.select_action(state, training=False)
                state, _, done, info = game.step(action)
                steps += 1
            reached.append(1.0 if info.get("won") else 0.0)
    finally:
        _restore_greedy_agent_eval(agent, agent_state)
    return {
        "leg2_reach_rate": float(np.mean(reached)) if reached else 0.0,
        "n": float(len(reached)),
        "mean_start_dist": float(np.mean(distances)) if distances else 0.0,
    }


def _eval_death_trace(trainer: Any, config: Any, *, games: int) -> dict[str, float]:
    """Behavioral failure trace (RUN-16): greedy-play held-out levels to termination and
    record HOW each episode ends — won / killed (by hazard vs enemy vs air) / timeout /
    stalled — plus crystals collected at the end. Pins whether normal's 0 wins are deaths
    (and to what) vs running out of time, which picks the survival lever."""
    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    agent = trainer.agent
    step_limit = int(config.EVAL_MAX_STEPS)
    reasons: dict[str, int] = {}
    kill_sources: dict[str, int] = {}
    crystal_fracs: list[float] = []
    steps_used: list[float] = []
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for _ in range(games):
            state = game.reset()
            done = False
            steps = 0
            info: dict[str, Any] = {}
            while not done and steps < step_limit:
                action = agent.select_action(state, training=False)
                state, _, done, info = game.step(action)
                steps += 1
            reason = info.get("end_reason", "unknown") if done else "timeout"
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "killed":
                src = str(info.get("last_damage_source", "none"))
                kill_sources[src] = kill_sources.get(src, 0) + 1
            init_c = max(1, int(info.get("initial_crystals", 1)))
            rem = int(info.get("crystals_remaining", 0))
            crystal_fracs.append((init_c - rem) / init_c)
            steps_used.append(float(steps))
    finally:
        _restore_greedy_agent_eval(agent, agent_state)
    n = max(1, games)
    out: dict[str, float] = {
        "n": float(games),
        "crystal_frac_mean": float(np.mean(crystal_fracs)) if crystal_fracs else 0.0,
        "steps_mean": float(np.mean(steps_used)) if steps_used else 0.0,
    }
    for r in ("won", "killed", "timeout", "stalled", "first_crystal_goal"):
        out[f"reason_{r}"] = reasons.get(r, 0) / n
    for s in ("hazard", "enemy", "air"):
        out[f"killed_by_{s}"] = kill_sources.get(s, 0) / n
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Per-split summary: rates for win/exit, IQM for the continuous surrogates."""
    out: dict[str, float] = {"n": float(len(rows))}
    if not rows:
        for metric in (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS):
            out[metric] = 0.0
        return out
    for metric in _RATE_METRICS:
        if metric == "crystal_frac":
            # crystal_frac is a 0..1 collection FRACTION → TRUE MEAN, not a bool
            # "collected-any" rate. The bool form over-reports multi-crystal difficulties:
            # on normal it read 0.83 (= 83% of levels collected ≥1 crystal) while the actual
            # mean fraction is ~0.20 (RUN-16 death-trace). Also expose the collected-≥1 rate
            # separately so the distribution signal isn't lost.
            out[metric] = float(np.mean([float(r.get(metric, 0.0) or 0.0) for r in rows]))
            out["crystal_any_rate"] = float(
                np.mean([float(bool(r.get(metric, False))) for r in rows])
            )
        else:
            out[metric] = float(np.mean([float(bool(r.get(metric, False))) for r in rows]))
    for metric in _MEAN_SURROGATE_METRICS:
        out[metric] = float(np.mean([float(r.get(metric, 0.0) or 0.0) for r in rows]))
    return out


def _average_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-(seed, episode) curve into per-episode means across seeds, so
    the best checkpoint and the verdict reflect the typical run, not one lucky seed."""
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for point in curve:
        by_episode.setdefault(int(point["episode"]), []).append(point)
    metrics = (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS)
    averaged: list[dict[str, Any]] = []
    for episode in sorted(by_episode):
        points = by_episode[episode]
        train = {m: float(np.mean([p["train"][m] for p in points])) for m in metrics}
        test = {m: float(np.mean([p["test"][m] for p in points])) for m in metrics}
        mean_q = float(np.mean([p.get("mean_q", 0.0) for p in points]))
        averaged.append(
            {
                "episode": episode,
                "n_seeds": len(points),
                "train": train,
                "test": test,
                "mean_q": mean_q,
            }
        )
    return averaged


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
    truncation_bootstrap: bool = False,
    force_cpu: bool = False,
    weight_decay: float = 0.0,
    regenerate_each_episode: bool = False,
    drop_leak_features: bool = False,
    use_cnn: bool = False,
    geodesic: bool = False,
    geodesic_weight: float | None = None,
    geodesic_after_unlock: bool = False,
    reverse_exit_curriculum_p: float | None = None,
    reverse_exit_curriculum_far: bool = False,
    geo_compass: bool = False,
    route_aux: bool = False,
    curriculum_easy_episodes: int = 0,
    curriculum_pretrain_difficulty: str = "easy",
    leg2_probe: bool = False,
    death_trace: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, object] = {}
    if geodesic:
        # Geodesic (BFS-distance) potential shaping toward the active objective —
        # targets the collect->exit conversion wall (leg 2: route to the open exit).
        overrides["CRYSTAL_CAVES_GEODESIC_POTENTIAL"] = True
    if geodesic_after_unlock:
        # Engage geodesic shaping ONLY after the exit unlocks (leg 2 only).
        overrides["CRYSTAL_CAVES_GEODESIC_POTENTIAL"] = True
        overrides["CRYSTAL_CAVES_GEODESIC_AFTER_UNLOCK"] = True
    if geodesic_weight is not None:
        # Strength of the geodesic shaping (default 0.3). Lower = gentler, to avoid
        # the dense-shaping learnability hit seen in RUN-06.
        overrides["CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT"] = geodesic_weight
    if pool_size is not None:
        overrides["CRYSTAL_CAVES_POOL_SIZE"] = pool_size
    if weight_decay > 0:
        overrides["WEIGHT_DECAY"] = weight_decay
    if use_cnn:
        # Position-preserving spatial CNN (SpatialDQN, flatten — NOT global-average-pool,
        # which was disconfirmed). Tests whether a conv inductive bias beats the flat MLP.
        overrides["USE_CNN_STATE"] = True
        overrides["CRYSTAL_CAVES_CNN_GLOBAL_POOL"] = False
    if regenerate_each_episode:
        overrides["CRYSTAL_CAVES_REGENERATE_EACH_EPISODE"] = True
    if drop_leak_features:
        overrides["CRYSTAL_CAVES_DROP_LEAK_FEATURES"] = True
    if force_cpu:
        # On Apple Silicon (M-series) CPU beats MPS for this small model; force it.
        overrides["FORCE_CPU"] = True
    if geo_compass:
        # Corridor compass: append metadata pointing down the real traversable route to
        # the active objective (the RUN-11 nav fix). Observation, not reward.
        overrides["CRYSTAL_CAVES_GEO_COMPASS"] = True
    if route_aux:
        # Op 2 (learnable route): supervise an aux head to PREDICT the geodesic route
        # direction (carried in trailing label slots, sliced off the policy input) — so the
        # net learns route-awareness instead of being fed the compass. Oracle off at eval.
        overrides["CRYSTAL_CAVES_ROUTE_AUX_LOSS"] = True
        overrides["CRYSTAL_CAVES_ROUTE_AUX_GEODESIC"] = True
    if truncation_bootstrap:
        # Treat timeout/stalled cutoffs as non-terminal so the TD target bootstraps
        # instead of learning value = raw -8/-6 (which n-step(6) compounds backwards).
        overrides["CRYSTAL_CAVES_TRUNCATION_BOOTSTRAP"] = True
    if reverse_exit_curriculum_p is not None:
        # Reverse-EXIT curriculum: on a fraction of training resets, start already in the
        # post-collection state next to the open exit, drilling the leg-2 route-to-exit
        # skill in isolation (the documented collect->exit wall). RUN-10 lever.
        overrides["CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM"] = True
        overrides["CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P"] = reverse_exit_curriculum_p
        if reverse_exit_curriculum_far:
            # FAR placement: drill long-range route-to-exit (the RUN-11 wall), not the hop.
            overrides["CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_FAR"] = True

    train_rows_all: list[dict[str, Any]] = []
    test_rows_all: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    leg2_rates: list[float] = []
    leg2_far_rates: list[float] = []
    leg2_far_dists: list[float] = []
    death_traces: list[dict[str, float]] = []
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

        # Difficulty curriculum (RUN-15): warm-start the target-difficulty phase from an
        # agent first trained on an easier tier, so it learns the full collect->exit chain
        # progressively instead of from scratch (RUN-15a showed normal's wall is behavioral
        # completion, not structural). state_size is identical across tiers (difficulty
        # changes crystal count, not window/gmap/meta), so the weights transfer strictly.
        transfer_weights = None
        if curriculum_easy_episodes > 0:
            print(
                f"  [curriculum] phase 1: pretrain {curriculum_easy_episodes} ep at "
                f"difficulty={curriculum_pretrain_difficulty}, then {episodes} ep at "
                f"difficulty={difficulty}",
                flush=True,
            )
            set_seed(seed)
            pre_config = make_config(overrides, difficulty=curriculum_pretrain_difficulty)
            pre_config.CRYSTAL_CAVES_SEED = seed
            pre_dir = run_dir / "pretrain"
            pre_dir.mkdir(parents=True, exist_ok=True)
            pre_trainer = prepare_trainer(
                pre_config, episodes=curriculum_easy_episodes, vec_envs=vec_envs
            )
            run_training(
                pre_trainer,
                run_dir=pre_dir,
                label=f"diag/seed_{seed}/pretrain",
                total_episodes=curriculum_easy_episodes,
                heartbeat_seconds=0.0,
                target_episodes=None,
            )
            transfer_weights = capture_weight_snapshot(pre_trainer.agent)
            set_seed(seed)  # realign RNG so the target phase mirrors the control's stream

        trainer = prepare_trainer(
            config, episodes=episodes, vec_envs=vec_envs, transfer_weights=transfer_weights
        )
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
            mean_q = _mean_q(trainer, config, games=games)
            if checkpoint_every > 0:
                curve.append(
                    {"seed": seed, "episode": milestone, "train": tr, "test": te, "mean_q": mean_q}
                )
                print(
                    f"[seed {seed} @ep{milestone}] "
                    f"TRAIN win={tr['won']:.2f} cryst={tr['crystal_frac']:.3f} "
                    f"tgt={tr['target_distance_progress']:.3f} | "
                    f"TEST win={te['won']:.2f} cryst={te['crystal_frac']:.3f} "
                    f"tgt={te['target_distance_progress']:.3f} | meanQ={mean_q:+.2f}",
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

        if leg2_probe:
            leg2 = _eval_leg2(trainer, config, games=games, mode="reverse_exit")
            leg2_far = _eval_leg2(trainer, config, games=games, mode="reverse_exit_far")
            leg2_rates.append(leg2["leg2_reach_rate"])
            leg2_far_rates.append(leg2_far["leg2_reach_rate"])
            leg2_far_dists.append(leg2_far["mean_start_dist"])
            print(
                f"[seed {seed}] LEG-2 PROBE: held-out reach-exit rate "
                f"NEAR (next to exit) = {leg2['leg2_reach_rate']:.3f} (n={int(leg2['n'])}) | "
                f"FAR (random start, mean dist {leg2_far['mean_start_dist']:.1f}) = "
                f"{leg2_far['leg2_reach_rate']:.3f} (n={int(leg2_far['n'])})",
                flush=True,
            )

        if death_trace:
            dt = _eval_death_trace(trainer, config, games=games)
            death_traces.append(dt)
            print(
                f"[seed {seed}] DEATH-TRACE (held-out, greedy play): "
                f"won={dt['reason_won']:.2f} killed={dt['reason_killed']:.2f} "
                f"(hazard={dt['killed_by_hazard']:.2f} enemy={dt['killed_by_enemy']:.2f} "
                f"air={dt['killed_by_air']:.2f}) timeout={dt['reason_timeout']:.2f} "
                f"stalled={dt['reason_stalled']:.2f} | crystals={dt['crystal_frac_mean']:.2f} "
                f"steps={dt['steps_mean']:.0f}",
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
            m: round(train_agg[m] - test_agg[m], 4)
            for m in (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS)
        },
        "curve": curve,
        "truncation_bootstrap": truncation_bootstrap,
        # Audit B3: under --regenerate-each-episode the agent trains on freshly generated
        # caves (seed offset 1e6+), NOT on the offset-0 CAVES[:n] that use_train_levels grades
        # as the "train" split — so that split is a SECOND held-out set and the
        # train-vs-test gap / memorisation verdict do NOT apply (no training-distribution
        # score was ever measured). Consumers must check this flag before reading the gap.
        "regenerate_each_episode": regenerate_each_episode,
        "train_split_is_holdout": regenerate_each_episode,
    }
    if leg2_rates:
        summary["leg2_reach_rate"] = float(np.mean(leg2_rates))
        summary["leg2_reach_rate_per_seed"] = leg2_rates
    if leg2_far_rates:
        summary["leg2_far_reach_rate"] = float(np.mean(leg2_far_rates))
        summary["leg2_far_reach_rate_per_seed"] = leg2_far_rates
        summary["leg2_far_mean_dist"] = float(np.mean(leg2_far_dists))
    if death_traces:
        keys = sorted({k for dt in death_traces for k in dt})
        summary["death_trace"] = {
            k: float(np.mean([dt.get(k, 0.0) for dt in death_traces])) for k in keys
        }
        print("\n==== DEATH-TRACE (held-out, greedy play, seed-avg) ====", flush=True)
        dts = summary["death_trace"]
        print(
            f"  ended: won={dts['reason_won']:.2f}  killed={dts['reason_killed']:.2f}  "
            f"timeout={dts['reason_timeout']:.2f}  stalled={dts['reason_stalled']:.2f}",
            flush=True,
        )
        print(
            f"  of deaths: hazard={dts['killed_by_hazard']:.2f}  enemy={dts['killed_by_enemy']:.2f}  "
            f"air={dts['killed_by_air']:.2f}  (fractions of all episodes)",
            flush=True,
        )
        print(
            f"  crystals collected at end={dts['crystal_frac_mean']:.2f}  "
            f"mean steps={dts['steps_mean']:.0f}",
            flush=True,
        )
        print(
            "  read: killed≫timeout & enemy-dominated => moving-enemy survival (maybe an "
            "enemy-velocity obs gap); killed≫timeout & hazard-dominated => static-hazard "
            "pathing; timeout-dominated => wander/exploration, not death.",
            flush=True,
        )
    # Best checkpoint = the milestone with the strongest TRAINING competence (win, then
    # crystals), chosen on the SEED-AVERAGED curve. Prior experiments graded the FINAL
    # net, which can be the collapsed one; grading the best checkpoint de-confounds
    # "collapsed" from "doesn't generalise", and averaging across seeds avoids picking
    # one lucky run.
    curve_avg = _average_curve(curve) if curve else []
    if curve_avg:
        summary["curve_avg"] = curve_avg
        summary["best"] = max(
            curve_avg, key=lambda p: (p["train"]["won"], p["train"]["crystal_frac"])
        )
    write_json(out_dir / "diagnosis.json", summary)
    if curve_avg:
        _print_curve(curve_avg)
    _print_report(summary)
    _print_leg2(summary)
    return summary


def _print_leg2(summary: dict[str, Any]) -> None:
    """Print the held-out LEG-2 PROBE block (route-to-exit reach rate). Shared by the
    single-process path and the per-seed aggregator so both render it identically."""
    if "leg2_reach_rate" not in summary:
        return
    per_seed = summary.get("leg2_reach_rate_per_seed", [])
    near = summary["leg2_reach_rate"]
    lines = [
        "\n==== LEG-2 PROBE (held-out, post-collection start, reach the open exit) ====",
        f"NEAR (dropped next to exit, trivial final hop) = {near:.3f}  "
        f"(per-seed {[round(r, 3) for r in per_seed]})",
    ]
    if "leg2_far_reach_rate" in summary:
        far = summary["leg2_far_reach_rate"]
        far_seed = summary.get("leg2_far_reach_rate_per_seed", [])
        dist = summary.get("leg2_far_mean_dist", 0.0)
        lines.append(
            f"FAR  (random reachable start, mean dist {dist:.1f}, real navigation) = "
            f"{far:.3f}  (per-seed {[round(r, 3) for r in far_seed]})"
        )
        lines.append(
            "read: NEAR high & FAR low => the WALL is long-range route-to-exit navigation "
            "(a corrected far-start curriculum has upside). NEAR≈FAR high while full-play "
            "win stays low => leg-2 is NOT the wall; the bottleneck is leg-1 collect-all "
            "or full-chain credit -> ceiling for this lever family. Both low => fundamental."
        )
    else:
        lines.append(
            "read: HIGH (>=0.5) => route-to-exit skill EXISTS but is under-reps'd in normal "
            "play. LOW => fundamental bottleneck -> ceiling. (No FAR probe in this run.)"
        )
    print("\n".join(lines), flush=True)


def _print_curve(curve: list[dict[str, Any]]) -> None:
    """Learning curve: key metrics over training, to separate 'slow' from 'stuck'."""
    print("\n==== LEARNING CURVE (train split over time) ====", flush=True)
    header = (
        "episode".rjust(8)
        + "win".rjust(8)
        + "crystals".rjust(10)
        + "closeness".rjust(11)
        + "exitUnlk".rjust(10)
        + "meanQ".rjust(10)
    )
    print(header, flush=True)
    for point in curve:
        tr = point["train"]
        print(
            f"{point['episode']:8d}"
            + f"{tr['won']:8.2f}"
            + f"{tr['crystal_frac']:10.3f}"
            + f"{tr['target_distance_progress']:11.3f}"
            + f"{tr['exit_unlocked_rate']:10.2f}"
            + f"{point.get('mean_q', 0.0):+10.2f}",
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
    if summary.get("train_split_is_holdout"):
        print(
            "  [!] --regenerate-each-episode: the TRAIN column is a 2nd HELD-OUT set "
            "(agent never trained on it); the GAP column is NOT a generalisation gap.",
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

    # Collect->win conversion: of the levels where the crystal was collected, what
    # fraction were finished? A low value = the agent solves leg 1 (find/collect) but
    # fails leg 2 (route to the now-open exit). This is the metric the IQM bug hid.
    def _conv(split: dict[str, float]) -> float:
        c = split.get("crystal_frac", 0.0)
        return (split.get("won", 0.0) / c) if c > 1e-9 else 0.0

    print(
        "collect->win conversion".ljust(26)
        + f"{_conv(tr):10.2f}"
        + f"{_conv(te):11.2f}"
        + f"{_conv(tr) - _conv(te):+11.2f}",
        flush=True,
    )

    # Best-checkpoint readout + collapse indicator. The verdict is based on the agent's
    # BEST training competence, not the (possibly collapsed) final episode.
    best = summary.get("best")
    if best:
        btr, bte = best["train"], best["test"]
        print(
            f"\nbest checkpoint: ep{best['episode']} "
            f"(TRAIN win={btr['won']:.2f} cryst={btr['crystal_frac']:.3f} | "
            f"TEST win={bte['won']:.2f} cryst={bte['crystal_frac']:.3f})",
            flush=True,
        )
        collapse = btr["won"] - tr["won"]
        if collapse > 0.1:
            print(
                f"  ⚠ COLLAPSE: train win fell {btr['won']:.2f} (ep{best['episode']}) "
                f"-> {tr['won']:.2f} (final). The agent unlearned; training past the peak hurts.",
                flush=True,
            )
        tr, te = btr, bte  # base the verdict below on the best checkpoint

    # Audit B3: under --regenerate-each-episode the "train" split is a 2nd held-out set, so
    # the gap and the memorisation-vs-learning heuristic below are meaningless — both legs
    # are out-of-distribution. Emit a regenerate-appropriate read instead.
    if summary.get("train_split_is_holdout"):
        test_learns = te["crystal_frac"] > 0.15 or te["won"] > 0.1
        print("\nread:", flush=True)
        print(
            "  -> --regenerate-each-episode: BOTH splits are held-out (the agent trained on "
            "freshly generated caves, not the 'train' split). The train-vs-test GAP and the "
            "memorisation/generalisation verdict DO NOT APPLY here. Judge absolute held-out: "
            + (
                "held-out is non-trivial — a real (if data-hungry) signal."
                if test_learns
                else "held-out is ~0 — it is not solving unseen caves."
            ),
            flush=True,
        )
        return

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
    parser.add_argument(
        "--truncation-bootstrap",
        action="store_true",
        help="Treat timeout/stalled as non-terminal (bootstrap) to test the collapse fix.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (recommended on Apple Silicon: faster than MPS for this model).",
    )
    parser.add_argument(
        "--regenerate-each-episode",
        action="store_true",
        help="Infinite-levels: generate a fresh procedural cave every training episode.",
    )
    parser.add_argument(
        "--drop-leak-features",
        action="store_true",
        help="Zero level_index + absolute player_x/y from the observation (anti-memorization).",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Use the position-preserving spatial CNN (flatten, not global-pool).",
    )
    parser.add_argument(
        "--geodesic",
        action="store_true",
        help="Enable geodesic (BFS-distance) potential shaping toward the active objective.",
    )
    parser.add_argument(
        "--geodesic-weight",
        type=float,
        default=None,
        help="Override geodesic shaping strength (default 0.3); use e.g. 0.1 for a gentler nudge.",
    )
    parser.add_argument(
        "--geodesic-after-unlock",
        action="store_true",
        help="Apply geodesic shaping ONLY after the exit unlocks (leg-2/route-to-exit only).",
    )
    parser.add_argument(
        "--reverse-exit-curriculum-p",
        type=float,
        default=None,
        help="Fraction of TRAINING resets that start in the post-collection state next to "
        "the open exit, drilling the leg-2 route-to-exit skill (RUN-10 lever); e.g. 0.5.",
    )
    parser.add_argument(
        "--geo-compass",
        action="store_true",
        help="Append a geodesic next-step corridor compass to the state: scalars pointing "
        "down the real traversable route to the active objective (the RUN-11 nav fix). An "
        "observation, not a reward — does not re-trigger the disconfirmed geodesic shaping.",
    )
    parser.add_argument(
        "--route-aux",
        action="store_true",
        help="Op 2 (learnable route): train an auxiliary head to PREDICT the geodesic route "
        "direction (carried in trailing label slots, sliced off the policy input) instead of "
        "feeding it. The net learns route-awareness from raw observation; oracle off at eval.",
    )
    parser.add_argument(
        "--curriculum-easy-episodes",
        type=int,
        default=0,
        help="Difficulty curriculum (RUN-15): pretrain this many episodes at "
        "--curriculum-pretrain-difficulty, then warm-start the main --difficulty run from "
        "those weights. Targets the behavioral completion wall (RUN-15a). For a fair A/B, "
        "set control --episodes = curriculum (pretrain + main) episodes.",
    )
    parser.add_argument(
        "--curriculum-pretrain-difficulty",
        default="easy",
        help="Difficulty tier for the curriculum pretrain phase (default easy).",
    )
    parser.add_argument(
        "--death-trace",
        action="store_true",
        help="RUN-16 behavioral failure trace: after training, greedy-play held-out levels "
        "and report HOW episodes end (won/killed-by-hazard/killed-by-enemy/timeout/stalled) "
        "+ crystals at end. Pins whether normal's 0 wins are deaths (and to what) vs timeouts.",
    )
    parser.add_argument(
        "--reverse-exit-curriculum-far",
        action="store_true",
        help="With --reverse-exit-curriculum-p: drop the player a real distance from the "
        "exit (random reachable tile) instead of next to it, drilling long-range "
        "route-to-exit navigation (the RUN-11 wall) rather than the trivial final hop.",
    )
    parser.add_argument(
        "--leg2-probe",
        action="store_true",
        help="After training, probe held-out route-to-exit: drop the agent next to the open "
        "exit (oracle-verified) and measure reach-exit rate (isolates leg-2 from leg-1).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam L2 weight decay (e.g. 1e-4) as a regularization lever; 0 = off.",
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
        truncation_bootstrap=args.truncation_bootstrap,
        force_cpu=args.cpu,
        weight_decay=args.weight_decay,
        regenerate_each_episode=args.regenerate_each_episode,
        drop_leak_features=args.drop_leak_features,
        use_cnn=args.cnn,
        geodesic=args.geodesic,
        geodesic_weight=args.geodesic_weight,
        geodesic_after_unlock=args.geodesic_after_unlock,
        reverse_exit_curriculum_p=args.reverse_exit_curriculum_p,
        reverse_exit_curriculum_far=args.reverse_exit_curriculum_far,
        geo_compass=args.geo_compass,
        route_aux=args.route_aux,
        curriculum_easy_episodes=args.curriculum_easy_episodes,
        curriculum_pretrain_difficulty=args.curriculum_pretrain_difficulty,
        leg2_probe=args.leg2_probe,
        death_trace=args.death_trace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
