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
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Tuple

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
_MEAN_SURROGATE_METRICS = (
    "depth_frac",
    "target_distance_progress",
    "selection_score",
    "damage_taken",
    "tiles_visited",
    "idle_frac",
)


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
    # Audit R2-D: the train pool can hold fewer DISTINCT caves than `games` (when
    # pool_size < games); cycling would silently duplicate caves and double-weight the
    # early ones in the mean. Grade only the distinct caves actually available.
    n_levels = min(games, len(getattr(game, "_eval_caves", ()) or range(games)))
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
        for level_index in range(n_levels):
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
        for probe_index in range(games):
            game.reset()
            # Deterministic probe placement: apply_reverse_start draws from the GLOBAL
            # np.random stream; unseeded it inherited whatever state training left, so
            # the same checkpoint gave different FAR start tiles per invocation
            # (eval-hygiene audit finding #4).
            np.random.seed(913_000 + probe_index)
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
            init_c = int(info.get("initial_crystals", 1))
            rem = int(info.get("crystals_remaining", 0))
            # zero-crystal cave = vacuously fully collected (matches the engine metric)
            crystal_fracs.append(1.0 if init_c <= 0 else (init_c - rem) / init_c)
            steps_used.append(float(steps))
    finally:
        _restore_greedy_agent_eval(agent, agent_state)
    n = max(1, games)
    out: dict[str, float] = {
        "n": float(games),
        "crystal_frac_mean": float(np.mean(crystal_fracs)) if crystal_fracs else 0.0,
        "steps_mean": float(np.mean(steps_used)) if steps_used else 0.0,
    }
    # Emit the standard taxonomy PLUS anything unexpected: the old fixed key list
    # silently dropped unknown end reasons, hiding taxonomy breaks (audit finding #8).
    for r in sorted({"won", "killed", "timeout", "stalled", "first_crystal_goal"} | set(reasons)):
        out[f"reason_{r}"] = reasons.get(r, 0) / n
    for s in ("hazard", "enemy", "both", "air"):
        out[f"killed_by_{s}"] = kill_sources.get(s, 0) / n
    return out


def _eval_stall_trace(trainer: Any, config: Any, *, games: int) -> dict[str, float]:
    """Stall diagnostic (RUN-19): for held-out greedy episodes that end in a STALL (the
    no-net-progress timer fires), characterise WHY, the way death-trace did for kills.
    RUN-18 showed 'stalled' is the dominant failure (~0.56) and that removing deaths just
    converts them into stalls, so the stall mechanism is now the binding constraint. For
    each stalled episode we record, at the stuck position:
      - trapped: the nearest active objective is NOT physically reachable (oracle/jump-aware)
        from the stuck tile -> the agent is wedged where no policy could finish (a control /
        exploration dead-end, or the BFS-compass routed it through a jump physics forbids);
      - near_objective: geodesic route distance to the objective is small (<=3 tiles) -> a
        last-mile failure (basically there, can't close it);
      - oscillating: <=3 distinct tiles visited in the trailing window -> bouncing in place
        rather than searching;
    plus mean geodesic distance and crystals collected at the stall. This tells whether the
    fix is control/jump skill (trapped/last-mile), exploration (far+drifting), or the route
    signal being physically unexecutable."""
    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    agent = trainer.agent
    step_limit = int(config.EVAL_MAX_STEPS)
    trail_window = 120
    near_tiles = 3
    n_stalled = 0
    trapped = osc = near = far = clock_mislabel = 0
    geo_dists: list[float] = []
    crystal_fracs: list[float] = []
    agent_state = _enter_greedy_agent_eval(agent)
    stall_clock = int(getattr(game, "MAX_STEPS_WITHOUT_PROGRESS", 720))
    try:
        for _ in range(games):
            state = game.reset()
            done = False
            steps = 0
            info: dict[str, Any] = {}
            trail: Deque[Tuple[int, int]] = deque(maxlen=trail_window)
            geo_hist: list[float] = []
            while not done and steps < step_limit:
                trail.append(game._player_tile())
                gd = game._geodesic_distance_field().get(game._player_tile())
                geo_hist.append(float(gd) if gd is not None else float("inf"))
                action = agent.select_action(state, training=False)
                state, _, done, info = game.step(action)
                steps += 1
            reason = info.get("end_reason", "unknown") if done else "timeout"
            if reason != "stalled":
                continue
            n_stalled += 1
            # Red-team check #4: the 720-step clock resets on EUCLIDEAN new-best approach,
            # but the compass the agent follows descends a GEODESIC field. If a new
            # geodesic minimum was set inside the very window the clock declared dead,
            # the episode was making real route progress and "stalled" is a mislabel.
            window, before = geo_hist[-stall_clock:], geo_hist[:-stall_clock]
            if before and window and min(window) < min(before):
                clock_mislabel += 1
            ptile = game._player_tile()
            targets = game._active_target_tiles()
            reachable = game._oracle_reachable(ptile) if targets else set()
            if targets and not any(t in reachable for t in targets):
                trapped += 1
            gdist = game._geodesic_distance_field().get(ptile)
            if gdist is not None:
                geo_dists.append(float(gdist))
                if gdist <= near_tiles:
                    near += 1
                else:
                    far += 1
            if len(set(trail)) <= 3:
                osc += 1
            init_c = int(info.get("initial_crystals", 1))
            rem = int(info.get("crystals_remaining", 0))
            # zero-crystal cave = vacuously fully collected (matches the engine metric)
            crystal_fracs.append(1.0 if init_c <= 0 else (init_c - rem) / init_c)
    finally:
        _restore_greedy_agent_eval(agent, agent_state)
    s = max(1, n_stalled)
    return {
        "n_games": float(games),
        "n_stalled": float(n_stalled),
        "stalled_rate": n_stalled / max(1, games),
        # fractions of STALLED episodes (overlapping buckets, like kill sources)
        "trapped_frac": trapped / s,
        "near_objective_frac": near / s,
        "far_from_objective_frac": far / s,
        "oscillating_frac": osc / s,
        "clock_mislabel_frac": clock_mislabel / s,
        "geo_dist_mean": float(np.mean(geo_dists)) if geo_dists else 0.0,
        "crystal_frac_mean": float(np.mean(crystal_fracs)) if crystal_fracs else 0.0,
    }


def _present_values(rows: list[dict[str, Any]], metric: str) -> tuple[list[Any], int]:
    """Values for ``metric`` from rows that HAVE it, plus the count that don't.
    A missing key used to silently score 0.0, making a pipeline break read as a
    bad agent (metric-audit finding #3) — now missing rows are excluded from the
    mean and surfaced loudly."""
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    return vals, len(rows) - len(vals)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-split summary: rates for win/exit, true means for the surrogates."""
    out: dict[str, Any] = {"n": float(len(rows))}
    if not rows:
        for metric in (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS):
            out[metric] = 0.0
        out["end_reason_counts"] = {}
        out["kill_source_counts"] = {}
        return out
    out["end_reason_counts"] = dict(
        Counter(str(r.get("end_reason", "unknown") or "unknown") for r in rows)
    )
    out["kill_source_counts"] = dict(
        Counter(
            str(r.get("last_damage_source", "none") or "none")
            for r in rows
            if str(r.get("end_reason", "")) == "killed"
        )
    )
    for metric in (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS):
        vals, missing = _present_values(rows, metric)
        if missing:
            out[f"missing_{metric}"] = float(missing)
            print(
                f"[diagnose_gap] WARNING: {missing}/{len(rows)} rows missing metric "
                f"{metric!r} — excluded from the mean (broken rows are NOT zeros)",
                flush=True,
            )
        if metric == "crystal_frac":
            # crystal_frac is a 0..1 collection FRACTION → TRUE MEAN, not a bool
            # "collected-any" rate. The bool form over-reports multi-crystal difficulties:
            # on normal it read 0.83 (= 83% of levels collected ≥1 crystal) while the actual
            # mean fraction is ~0.20 (RUN-16 death-trace). Also expose the collected-≥1 rate
            # separately so the distribution signal isn't lost.
            out[metric] = float(np.mean([float(v) for v in vals])) if vals else 0.0
            out["crystal_any_rate"] = (
                float(np.mean([float(bool(v)) for v in vals])) if vals else 0.0
            )
        elif metric in _RATE_METRICS:
            out[metric] = float(np.mean([float(bool(v)) for v in vals])) if vals else 0.0
        else:
            out[metric] = float(np.mean([float(v) for v in vals])) if vals else 0.0
    return out


def _average_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-(seed, episode) curve into per-episode means across seeds, so
    the best checkpoint and the verdict reflect the typical run, not one lucky seed."""
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for point in curve:
        by_episode.setdefault(int(point["episode"]), []).append(point)
    metrics = (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS)
    averaged: list[dict[str, Any]] = []

    def _split_means(points: list[dict[str, Any]], split: str) -> dict[str, float]:
        # Tolerate metrics absent from older rows (e.g. RUN-<=22 curves predate the
        # damage/tiles/idle telemetry): average what exists, omit what doesn't —
        # hard-indexing KeyError'd on any pre-telemetry curve, including reused
        # baseline arms.
        out: dict[str, float] = {}
        for m in metrics:
            vals = [p[split][m] for p in points if m in p[split]]
            if vals:
                out[m] = float(np.mean(vals))
        return out

    for episode in sorted(by_episode):
        points = by_episode[episode]
        train = _split_means(points, "train")
        test = _split_means(points, "test")
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
    reward_clip: float | None = None,
    stall_window: int | None = None,
    max_steps: int | None = None,
    death_penalty: float | None = None,
    hit_penalty: float | None = None,
    ngu_bonus: bool = False,
    ngu_beta: float | None = None,
    enemy_motion: bool = False,
    win_at_k: int = 0,
    win_at_k_ramp: int = 0,
    win_at_k_ramp_delay: int = 0,
    save_weights: bool = False,
    demo_dir: str | None = None,
    demo_pretrain: int = 0,
    demo_reset_p: float = 0.0,
    demo_td_weight: float | None = None,
    demo_margin_weight: float | None = None,
    demo_opening_steps: int = 0,
    demo_margin_decay: int = 0,
    demo_margin_reignite: int = 0,
    demo_margin_reignite_scale: float = 0.5,
    demo_backward: bool = False,
    demo_backward_retreat: int = 0,
    demo_backward_wins: int = 0,
    demo_level_bias: float = 0.0,
    demo_backward_window: int = 0,
    demo_backward_deep: int = 0,
    demo_heal: bool = False,
    resume_weights: str | None = None,
    ladder_init: str | None = None,
    regenerate_each_episode: bool = False,
    drop_leak_features: bool = False,
    use_cnn: bool = False,
    geodesic: bool = False,
    geodesic_weight: float | None = None,
    geodesic_after_unlock: bool = False,
    reverse_exit_curriculum_p: float | None = None,
    reverse_exit_curriculum_far: bool = False,
    geo_compass: bool = False,
    geo_compass_hazard_aware: bool = False,
    route_aux: bool = False,
    curriculum_easy_episodes: int = 0,
    curriculum_pretrain_difficulty: str = "easy",
    leg2_probe: bool = False,
    death_trace: bool = False,
    stall_trace: bool = False,
    record_play: int = 0,
    imported: bool = False,
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
    if reward_clip is not None:
        # RUN-23 finding: REWARD_CLIP=5 clamps sampled (n-step) rewards to min=-5, so
        # death -12/-30, timeout -8 and stall -6 all trained as the SAME -5 — death-scale
        # levers are invisible unless the clip is raised (or 0 = disabled; the agent code
        # guards `if REWARD_CLIP > 0`, so 0 is a safe off-switch, not a clamp-to-zero).
        overrides["REWARD_CLIP"] = reward_clip
    if max_steps is not None:
        # 1991-fidelity lever: the original has no level timer; see PR #39
        # level-validity audit before using in comparable runs.
        overrides["CRYSTAL_CAVES_MAX_STEPS_OVERRIDE"] = max_steps
    if stall_window is not None:
        # RUN-26 fidelity arm: widen the no-progress stall window (game default 720).
        # DATA-1: harness timers own 35-54% of endings, flat across learning.
        overrides["CRYSTAL_CAVES_STALL_WINDOW_STEPS"] = stall_window
    if death_penalty is not None:
        overrides["CRYSTAL_CAVES_DEATH_PENALTY"] = death_penalty
    if hit_penalty is not None:
        overrides["CRYSTAL_CAVES_HIT_PENALTY"] = hit_penalty
    if ngu_bonus:
        overrides["CRYSTAL_CAVES_NGU_BONUS"] = True
    if ngu_beta is not None:
        overrides["CRYSTAL_CAVES_NGU_BETA"] = ngu_beta
    if enemy_motion:
        # FOV-limited per-enemy relative (dx, dy, vx, vy) for the 3 nearest visible
        # enemies — a single-frame tile window cannot dodge movers without it.
        overrides["CRYSTAL_CAVES_ENEMY_MOTION"] = True
    if win_at_k > 0:
        # Training-only win tier: exit opens at K crystals; eval keeps the real rule.
        overrides["CRYSTAL_CAVES_WIN_AT_K"] = win_at_k
        if win_at_k_ramp > 0:
            # CLI takes GLOBAL episodes; the game ramps on per-instance episodes.
            # Divide by the TRAINING env count (vec_envs) — NOT `games`, which is
            # the eval split size (RUN-34/35 bug: dividing by 48 instead of 8
            # finished the whole ramp by global ep~2000).
            overrides["CRYSTAL_CAVES_WIN_AT_K_RAMP_EPISODES"] = max(
                1, win_at_k_ramp // max(1, vec_envs)
            )
            if win_at_k_ramp_delay > 0:
                overrides["CRYSTAL_CAVES_WIN_AT_K_RAMP_DELAY"] = max(
                    1, win_at_k_ramp_delay // max(1, vec_envs)
                )
    if demo_dir:
        # DQfD-lite: fixed demo buffer + margin loss on every gradient step, plus
        # optional demo-only pre-training and demo-prefix (backward-curriculum) starts.
        overrides["DEMO_DIR"] = demo_dir
        overrides["DEMO_PRETRAIN_STEPS"] = demo_pretrain
        overrides["CRYSTAL_CAVES_DEMO_RESET_P"] = demo_reset_p
        if demo_backward:
            # Salimans & Chen backward algorithm: start each demo episode near the
            # WIN and retreat on competence — the bottom rungs the random-cut
            # prefix curriculum (10-85%, never near the win) was missing.
            overrides["CRYSTAL_CAVES_DEMO_BACKWARD"] = True
            if demo_backward_retreat > 0:
                overrides["CRYSTAL_CAVES_DEMO_BACKWARD_RETREAT"] = demo_backward_retreat
            if demo_backward_wins > 0:
                overrides["CRYSTAL_CAVES_DEMO_BACKWARD_WINS"] = demo_backward_wins
        if demo_level_bias > 0:
            # Concentrate training episodes on demoed levels (ladder throughput).
            overrides["CRYSTAL_CAVES_DEMO_LEVEL_BIAS"] = demo_level_bias
        if demo_backward_window > 0:
            overrides["CRYSTAL_CAVES_DEMO_BACKWARD_WINDOW"] = demo_backward_window
        if demo_backward_deep > 0:
            overrides["CRYSTAL_CAVES_DEMO_BACKWARD_DEEP"] = demo_backward_deep
        if demo_heal:
            overrides["CRYSTAL_CAVES_DEMO_HEAL_ON_HANDOFF"] = True
        if demo_td_weight is not None:
            # RUN-26c ablation: the per-step demo TD term drills large winning-return
            # targets from a tiny fixed set thousands of times (Q-inflation suspect);
            # 0.0 keeps only the large-margin supervised term.
            overrides["DEMO_TD_WEIGHT"] = demo_td_weight
        if demo_margin_weight is not None:
            # RUN-26d ablation: 0.0 together with --demo-td-weight 0 disables the
            # demo gradient entirely, leaving demo-prefix starts (backward
            # curriculum) as the only demo mechanism.
            overrides["DEMO_MARGIN_WEIGHT"] = demo_margin_weight
        if demo_opening_steps:
            # Phase-2 opening imitation: demo store keeps only each route's first
            # N transitions, making the margin loss a pure route-opening prior.
            overrides["DEMO_OPENING_ONLY_STEPS"] = demo_opening_steps
        if demo_margin_decay:
            # RUN-62 iteration: linear-decay the margin weight to zero over this
            # many GLOBAL episodes — keep the early accelerant, drop the anchor.
            overrides["DEMO_MARGIN_DECAY_EPISODES"] = demo_margin_decay
        if demo_margin_reignite:
            # RUN-63 iteration: floor the scale again from this episode, once the
            # frontier reaches the demo-covered opening where imitation aligns.
            overrides["DEMO_MARGIN_REIGNITE_EPISODE"] = demo_margin_reignite
            overrides["DEMO_MARGIN_REIGNITE_SCALE"] = demo_margin_reignite_scale
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
    if geo_compass_hazard_aware:
        # RUN-17 survival lever: route the compass AROUND static hazards (the largest
        # death source in the trusted RUN-17 trace) instead of through them. Same dims,
        # observation-only; implies the compass.
        overrides["CRYSTAL_CAVES_GEO_COMPASS"] = True
        overrides["CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE"] = True
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
    stall_traces: list[dict[str, float]] = []
    for seed in seeds:
        print(
            f"\n===== baseline seed={seed} (difficulty={difficulty}, episodes={episodes}) =====",
            flush=True,
        )
        set_seed(seed)
        config = make_config(overrides, difficulty=difficulty, imported=imported)
        config.CRYSTAL_CAVES_SEED = seed
        run_dir = out_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Fresh sidecar per invocation: appends below are per-milestone, but a rerun
        # into the same directory must not mix stale rows into trend analysis.
        (run_dir / "per_level_rows.jsonl").unlink(missing_ok=True)

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
            pre_config = make_config(
                overrides, difficulty=curriculum_pretrain_difficulty, imported=imported
            )
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
        if resume_weights:
            # Warm-start from a persisted policy checkpoint (policy_seedX_epN.pth =
            # raw policy_net state_dict). Target net starts as a copy — standard
            # for resuming; the first target sync realigns it anyway.
            import torch as _torch

            sd = _torch.load(resume_weights, map_location="cpu")
            transfer_weights = {"policy": sd, "target": dict(sd)}
            print(f"  [resume] warm-started from {resume_weights}", flush=True)

        if ladder_init:
            # Pin backward-ladder frontiers at run start ("14:2600,12:350") so a
            # resumed brain practices from-spawn immediately instead of paying
            # the re-climb tax (RUN-40: only reached 950/2600 re-climbing).
            from src.game.crystal_caves import CrystalCaves as _CC

            _CC._BC_SHARED_OFFSET.clear()
            _CC._BC_SHARED_WINS.clear()
            for pair in ladder_init.split(","):
                lvl, off = pair.split(":")
                _CC._BC_SHARED_OFFSET[int(lvl)] = int(off)
            print(f"  [ladder-init] {_CC._BC_SHARED_OFFSET}", flush=True)
        trainer = prepare_trainer(
            config, episodes=episodes, vec_envs=vec_envs, transfer_weights=transfer_weights
        )
        if trainer.agent.epsilon < TUTORIAL_MIN_EPSILON:
            trainer.agent.epsilon = TUTORIAL_MIN_EPSILON

        if getattr(config, "DEMO_DIR", None):
            from src.ai.demo_learning import DemoStore

            store = DemoStore.from_dir(config.DEMO_DIR, config)
            if store is None:
                print(f"[seed {seed}] WARNING: no usable demos in {config.DEMO_DIR}", flush=True)
            else:
                trainer.agent.attach_demo_store(store)
                pre_steps = trainer.agent.pretrain_on_demos(
                    int(getattr(config, "DEMO_PRETRAIN_STEPS", 0))
                )
                print(
                    f"[seed {seed}] demos: {store.n_episodes} episodes, "
                    f"{len(store)} transitions, pretrain steps={pre_steps}, "
                    f"demo_reset_p={getattr(config, 'CRYSTAL_CAVES_DEMO_RESET_P', 0.0)}",
                    flush=True,
                )

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
            # Eval must not perturb the training RNG stream: constructing eval games
            # consumes global np.random draws, so milestone cadence used to change the
            # trained policy itself (eval-hygiene audit finding #3).
            rng_state = np.random.get_state()
            try:
                train_rows = _eval_split(
                    trainer, config, split="train", games=games, run_dir=run_dir
                )
                test_rows = _eval_split(trainer, config, split="test", games=games, run_dir=run_dir)
                tr, te = _aggregate(train_rows), _aggregate(test_rows)
                mean_q = _mean_q(trainer, config, games=games)
            finally:
                np.random.set_state(rng_state)
            # RUN-24 artifact gap: per-level rows were evaluated but never persisted,
            # so a "per-level table at the best checkpoint" could not be reconstructed
            # after the run. Append every milestone's rows to a JSONL sidecar.
            rows_path = run_dir / "per_level_rows.jsonl"
            with open(rows_path, "a") as handle:
                for row in (*train_rows, *test_rows):
                    handle.write(json.dumps({"seed": seed, "episode": milestone, **row}) + "\n")
            # Winner's-curse guard (RUN-24 lesson): the "best checkpoint" could never be
            # re-evaluated because no weights were persisted, so a 3/48 win blip at one
            # milestone was unverifiable. Persist per-milestone policy weights so any
            # claimed winner can be replayed on fresh seeds before promotion.
            if save_weights:
                import torch

                weights_path = run_dir / f"policy_seed{seed}_ep{milestone}.pth"
                torch.save(trainer.agent.policy_net.state_dict(), weights_path)
                from src.game.crystal_caves import CrystalCaves as _CC

                if _CC._BC_SHARED_OFFSET:
                    (run_dir / f"ladder_seed{seed}.json").write_text(
                        json.dumps({str(k): v for k, v in _CC._BC_SHARED_OFFSET.items()})
                    )
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
                f"both={dt['killed_by_both']:.2f} air={dt['killed_by_air']:.2f}) "
                f"timeout={dt['reason_timeout']:.2f} "
                f"stalled={dt['reason_stalled']:.2f} | crystals={dt['crystal_frac_mean']:.2f} "
                f"steps={dt['steps_mean']:.0f}",
                flush=True,
            )

        if stall_trace:
            st = _eval_stall_trace(trainer, config, games=games)
            stall_traces.append(st)
            print(
                f"[seed {seed}] STALL-TRACE (held-out, greedy play): "
                f"stalled={st['stalled_rate']:.2f} (n={int(st['n_stalled'])}) | "
                f"trapped={st['trapped_frac']:.2f} near={st['near_objective_frac']:.2f} "
                f"far={st['far_from_objective_frac']:.2f} osc={st['oscillating_frac']:.2f} | "
                f"geoDist={st['geo_dist_mean']:.1f} crystals={st['crystal_frac_mean']:.2f}",
                flush=True,
            )

        if record_play > 0:
            from experiments.cc_status.record_play import record_policy_play

            replay_dir = run_dir / "replays"
            saved = record_policy_play(
                trainer.agent, config, games=games, out_dir=replay_dir, max_gifs=record_play
            )
            print(
                f"[seed {seed}] RECORD-PLAY: saved {len(saved)} gif(s) to {replay_dir} "
                f"({', '.join(s['end_reason'] for s in saved)})",
                flush=True,
            )

    train_agg, test_agg = _aggregate(train_rows_all), _aggregate(test_rows_all)
    summary = {
        "imported_fixed_set": bool(imported),
        "config_overrides": {k: v for k, v in sorted(overrides.items())},
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
        summary["death_trace_per_seed"] = death_traces
        _print_death_trace(summary)
        print(
            "  read: killed≫timeout & enemy-dominated => moving-enemy survival (maybe an "
            "enemy-velocity obs gap); killed≫timeout & hazard-dominated => static-hazard "
            "pathing; timeout-dominated => wander/exploration, not death.",
            flush=True,
        )
    if stall_traces:
        keys = sorted({k for st in stall_traces for k in st})
        summary["stall_trace"] = {
            k: float(np.mean([st.get(k, 0.0) for st in stall_traces])) for k in keys
        }
        summary["stall_trace_per_seed"] = stall_traces
        _print_stall_trace(summary)
    # Best checkpoint = the milestone with the strongest TRAINING competence (win, then
    # crystals), chosen on the SEED-AVERAGED curve. Prior experiments graded the FINAL
    # net, which can be the collapsed one; grading the best checkpoint de-confounds
    # "collapsed" from "doesn't generalise", and averaging across seeds avoids picking
    # one lucky run.
    curve_avg = _average_curve(curve) if curve else []
    if curve_avg:
        summary["curve_avg"] = curve_avg
        # Audit R2-C: pick the best checkpoint only from buckets that have ALL seeds, so a
        # ragged/straggling seed can't make a single-seed bucket the reported cross-seed best.
        n_all = len(seeds)
        full_buckets = [p for p in curve_avg if p.get("n_seeds", 1) == n_all]
        if not full_buckets:
            # No bucket has all seeds (crashed seeds / mismatched --checkpoint-every).
            # Fall back to the buckets with the MOST seeds — loudly, not silently: the
            # old `or curve_avg` fallback quietly re-admitted single-seed buckets,
            # exactly the failure the guard exists for (metric-audit finding #6).
            max_seeds = max(p.get("n_seeds", 1) for p in curve_avg)
            full_buckets = [p for p in curve_avg if p.get("n_seeds", 1) == max_seeds]
            print(
                f"[diagnose_gap] WARNING: no checkpoint bucket contains all {n_all} seeds; "
                f"selecting best from buckets with n_seeds={max_seeds} only",
                flush=True,
            )
        best = max(full_buckets, key=lambda p: (p["train"]["won"], p["train"]["crystal_frac"]))
        summary["best"] = best
        # Audit B5: the top-level gap_train_minus_test is the FINAL net; the verdict uses the
        # BEST checkpoint. On a collapsing run these disagree (final gap says "never learned
        # train" while the verdict says MEMORISATION). Emit the best-checkpoint gap too so the
        # JSON carries both, and label the printed one FINAL.
        summary["gap_train_minus_test_final"] = summary["gap_train_minus_test"]
        summary["gap_train_minus_test_best"] = {
            m: round(best["train"].get(m, 0.0) - best["test"].get(m, 0.0), 4)
            for m in (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS)
            if m in best["train"] and m in best["test"]
        }
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


def _print_death_trace(summary: dict[str, Any]) -> None:
    """Print the held-out DEATH-TRACE block (how greedy episodes end: won/killed/timeout/
    stalled + kill source). Shared by the single-process path and the per-seed aggregator so
    both render the seed-averaged trace identically — the aggregator no longer needs the
    trace computed by hand."""
    if "death_trace" not in summary:
        return
    dts = summary["death_trace"]
    print("\n==== DEATH-TRACE (held-out, greedy play, seed-avg) ====", flush=True)
    print(
        f"  ended: won={dts.get('reason_won', 0.0):.2f}  "
        f"killed={dts.get('reason_killed', 0.0):.2f}  "
        f"timeout={dts.get('reason_timeout', 0.0):.2f}  "
        f"stalled={dts.get('reason_stalled', 0.0):.2f}",
        flush=True,
    )
    print(
        f"  of deaths: hazard={dts.get('killed_by_hazard', 0.0):.2f}  "
        f"enemy={dts.get('killed_by_enemy', 0.0):.2f}  "
        f"both={dts.get('killed_by_both', 0.0):.2f}  "
        f"air={dts.get('killed_by_air', 0.0):.2f}  "
        f"(fractions of all episodes)",
        flush=True,
    )
    print(
        f"  crystals collected at end={dts.get('crystal_frac_mean', 0.0):.2f}  "
        f"mean steps={dts.get('steps_mean', 0.0):.0f}",
        flush=True,
    )


def _print_stall_trace(summary: dict[str, Any]) -> None:
    """Print the held-out STALL-TRACE block (why greedy episodes stall: trapped / last-mile /
    far / oscillating). Shared by the single-process path and the per-seed aggregator so both
    render the seed-averaged trace identically."""
    if "stall_trace" not in summary:
        return
    st = summary["stall_trace"]
    print("\n==== STALL-TRACE (held-out, greedy play, seed-avg) ====", flush=True)
    print(
        f"  stalled rate={st.get('stalled_rate', 0.0):.3f}  "
        f"(buckets below are fractions of STALLED episodes, overlapping)",
        flush=True,
    )
    print(
        f"  trapped (objective physically unreachable from stuck tile)="
        f"{st.get('trapped_frac', 0.0):.2f}",
        flush=True,
    )
    print(
        f"  near-objective (<=3 tiles, last-mile)={st.get('near_objective_frac', 0.0):.2f}  "
        f"far={st.get('far_from_objective_frac', 0.0):.2f}  "
        f"oscillating (<=3 tiles in trail)={st.get('oscillating_frac', 0.0):.2f}",
        flush=True,
    )
    print(
        f"  mean geodesic dist to objective={st.get('geo_dist_mean', 0.0):.1f} tiles  "
        f"crystals at stall={st.get('crystal_frac_mean', 0.0):.2f}",
        flush=True,
    )
    print(
        f"  clock-mislabel (new GEODESIC minimum inside the euclidean stall window)="
        f"{st.get('clock_mislabel_frac', 0.0):.2f}  -> that fraction of 'stalls' were "
        "making real route progress when the clock killed them",
        flush=True,
    )
    print(
        "  read: trapped-heavy => control/jump dead-ends or a route the physics can't execute; "
        "near-heavy => last-mile precision (jump/landing skill); far+oscillating => the policy "
        "isn't following the route (credit/exploration), not a perception gap.",
        flush=True,
    )


def _print_curve(curve: list[dict[str, Any]]) -> None:
    """Learning curve: key metrics over training, to separate 'slow' from 'stuck'."""
    print("\n==== LEARNING CURVE (train split over time) ====", flush=True)
    header = (
        "episode".rjust(8)
        + "win".rjust(8)
        + "crystals".rjust(10)
        + "closeness".rjust(11)
        + "exitUnlk".rjust(10)
        + "damage".rjust(9)
        + "tiles".rjust(8)
        + "idle".rjust(8)
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
            + f"{tr.get('damage_taken', 0.0):9.2f}"
            + f"{tr.get('tiles_visited', 0.0):8.1f}"
            + f"{tr.get('idle_frac', 0.0):8.2f}"
            + f"{point.get('mean_q', 0.0):+10.2f}",
            flush=True,
        )


def _wilson95(rate: float, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial rate over n trials (n<=0 -> full span)."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.96
    denom = 1.0 + z * z / n
    centre = rate + z * z / (2 * n)
    half = z * ((rate * (1.0 - rate) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


def _print_report(summary: dict[str, Any]) -> None:
    tr, te = summary["train"], summary["test"]
    gap = summary["gap_train_minus_test"]
    print("\n==== PHASE 0 DIAGNOSIS: train vs held-out ====", flush=True)
    print(
        f"difficulty={summary['difficulty']} seeds={summary['seeds']} "
        f"episodes={summary['episodes']} games/split={summary['games']}",
        flush=True,
    )
    if summary.get("best"):
        # Audit B5: the table/GAP below are the FINAL net; the verdict uses the BEST
        # checkpoint (printed lower). On a collapsing run they differ — see
        # gap_train_minus_test_best in the JSON.
        print(
            "  [note] table = FINAL net; GAP can differ from the best-checkpoint verdict "
            "below (JSON has gap_train_minus_test_final/_best).",
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
        "damage_taken": "damage taken",
        "tiles_visited": "tiles visited",
        "idle_frac": "idle fraction",
    }
    for metric in (*_RATE_METRICS, *_MEAN_SURROGATE_METRICS):
        if metric not in tr or metric not in te:
            continue  # telemetry absent from pre-RUN-23 summaries
        print(
            labels.get(metric, metric).ljust(26)
            + f"{tr[metric]:10.3f}"
            + f"{te[metric]:11.3f}"
            + f"{gap.get(metric, tr[metric] - te[metric]):+11.3f}",
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
        n_eval = int(summary.get("games", 0)) * max(1, len(summary.get("seeds", [1])))
        tr_lo, tr_hi = _wilson95(btr["won"], n_eval)
        te_lo, te_hi = _wilson95(bte["won"], n_eval)
        print(
            f"\nbest checkpoint: ep{best['episode']} "
            f"(TRAIN win={btr['won']:.2f} [95% CI {tr_lo:.2f}-{tr_hi:.2f}] "
            f"cryst={btr['crystal_frac']:.3f} | "
            f"TEST win={bte['won']:.2f} [95% CI {te_lo:.2f}-{te_hi:.2f}] "
            f"cryst={bte['crystal_frac']:.3f}) n={n_eval} eval episodes/split",
            flush=True,
        )
        print(
            "  note: best checkpoint is the argmax over checkpoints of these same "
            "episodes, so its TRAIN numbers carry winner's-curse bias; the CI bounds "
            "chance, not selection.",
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

    if summary.get("imported_fixed_set"):
        print(
            "\nnote: --imported runs train and eval on the SAME fixed 16-level set; the "
            "train-vs-test gap and the memorisation verdict below do not apply — judge "
            "absolute win/crystal numbers.",
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
    parser.add_argument(
        "--imported",
        action="store_true",
        help="Train/eval on the hand-crafted fixed 16-level set (CRYSTAL_CAVES_IMPORTED) "
        "instead of procedural generation. Train and test splits are then the SAME fixed "
        "set (memorisation allowed); the train-vs-test gap read does not apply.",
    )
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
        "--geo-compass-hazard-aware",
        action="store_true",
        help="Make the corridor compass route AROUND static hazards instead of through them "
        "(implies --geo-compass). RUN-17 survival lever: hazards are the largest death source "
        "in the trusted trace and likely drive much of the stall mass. Same dims, no net-shape "
        "change — an observation that points down a hazard-avoiding route.",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the Crystal Caves episode step cap (game default 3000). "
        "The 1991 original has no level timer; perfect tours on the rebalanced "
        "set already use 0.55-0.92 of the default cap.",
    )
    parser.add_argument(
        "--stall-window",
        type=int,
        default=None,
        help="Override the no-progress stall window in steps (game default 720). "
        "RUN-26 fidelity arm widens it to ~1440 so mid-route journeys stop being "
        "executed by the harness timer.",
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=None,
        help="Override REWARD_CLIP (negative-reward clamp before learning). Default 5.0 "
        "clips ALL penalties to -5, masking death-scale levers; set >= |death penalty| "
        "when testing CRYSTAL_CAVES_DEATH_PENALTY, or 0 to disable clipping entirely.",
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=None,
        help="Override CRYSTAL_CAVES_DEATH_PENALTY, e.g. --death-penalty=-30.",
    )
    parser.add_argument(
        "--hit-penalty",
        type=float,
        default=None,
        help="Override CRYSTAL_CAVES_HIT_PENALTY.",
    )
    parser.add_argument(
        "--ngu-bonus",
        action="store_true",
        help="Enable the Crystal Caves NGU-style episodic novelty bonus.",
    )
    parser.add_argument(
        "--ngu-beta",
        type=float,
        default=None,
        help="Override CRYSTAL_CAVES_NGU_BETA; implies the configured beta value only.",
    )
    parser.add_argument(
        "--enemy-motion",
        action="store_true",
        help="Enable CRYSTAL_CAVES_ENEMY_MOTION: FOV-limited relative position/velocity "
        "of the 3 nearest visible enemies in the observation. A single-frame tile window "
        "cannot dodge movers; enemy deaths dominate the RUN-24 kill trace.",
    )
    parser.add_argument(
        "--win-at-k",
        type=int,
        default=0,
        metavar="K",
        help="Training-only win tier: the exit opens once K crystals are held "
        "(CRYSTAL_CAVES_WIN_AT_K). Eval keeps the real all-crystals rule, so reported "
        "win rates stay canonical. 0 = off.",
    )
    parser.add_argument(
        "--win-at-k-ramp",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Ramp the win-at-K tier: K climbs linearly to the full crystal count "
        "over this many GLOBAL training episodes (converted to per-instance "
        "episodes internally), converging on the real win rule. 0 = static K.",
    )
    parser.add_argument(
        "--win-at-k-ramp-delay",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Hold K at the floor for this many GLOBAL episodes before the ramp "
        "starts (win-consolidation phase). 0 = ramp immediately.",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of verified winning demo JSONs (human recorder / planner). "
        "Enables DQfD-lite: fixed demo buffer + large-margin loss each gradient step.",
    )
    parser.add_argument(
        "--demo-pretrain",
        type=int,
        default=0,
        metavar="N",
        help="Demo-only gradient steps before environment interaction (DQfD pre-training).",
    )
    parser.add_argument(
        "--demo-reset-p",
        type=float,
        default=0.0,
        metavar="P",
        help="Probability a TRAINING episode starts mid-route from a random 10-85%% "
        "prefix of a winning demo (backward curriculum). Eval unaffected.",
    )
    parser.add_argument(
        "--demo-td-weight",
        type=float,
        default=None,
        metavar="W",
        help="Weight of the n-step TD term in the per-step demo minibatch loss "
        "(default: config DEMO_TD_WEIGHT). 0 = margin-only DQfD-lite.",
    )
    parser.add_argument(
        "--demo-backward",
        action="store_true",
        help="Backward demo curriculum (Salimans & Chen): episodes start near the "
        "demo's WIN and the start point retreats as the agent banks wins. "
        "Replaces the random 10-85%% prefix cuts. Needs --demo-dir and --demo-reset-p.",
    )
    parser.add_argument(
        "--demo-backward-retreat",
        type=int,
        default=0,
        metavar="STEPS",
        help="Backward-ladder retreat per rung in steps (0 = game default 40).",
    )
    parser.add_argument(
        "--demo-backward-wins",
        type=int,
        default=0,
        metavar="N",
        help="Wins required per backward-ladder rung (0 = game default 3).",
    )
    parser.add_argument(
        "--demo-backward-window",
        type=int,
        default=0,
        metavar="STEPS",
        help="Sample backward starts uniformly from [frontier-WINDOW, frontier] "
        "(rehearsal keeps deep rungs learnable); only exact-frontier attempts "
        "bank rung credit. 0 = frontier-only.",
    )
    parser.add_argument(
        "--demo-backward-deep",
        type=int,
        default=0,
        metavar="STEPS",
        help="Deep-rung easing threshold: past this steps-from-win, rungs cost "
        "1 win and retreat half-steps. 0 = off.",
    )
    parser.add_argument(
        "--resume-weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Warm-start the policy from a persisted checkpoint "
        "(policy_seedX_epN.pth). Continues training an existing brain instead "
        "of relearning from scratch.",
    )
    parser.add_argument(
        "--ladder-init",
        type=str,
        default=None,
        metavar="LVL:OFF,...",
        help='Pin backward-ladder frontiers at run start, e.g. "14:2600,12:350".',
    )
    parser.add_argument(
        "--demo-heal",
        action="store_true",
        help="Restore full health at demo-prefix handoff (training only) — "
        "corrects the HP-1 bias of tank-and-grab harvester route suffixes.",
    )
    parser.add_argument(
        "--demo-level-bias",
        type=float,
        default=0.0,
        metavar="P",
        help="Probability a TRAINING episode resamples its level among DEMOED "
        "levels (ladder-focus). Eval unaffected. 0 = uniform.",
    )
    parser.add_argument(
        "--demo-margin-weight",
        type=float,
        default=None,
        metavar="W",
        help="Weight of the large-margin term in the per-step demo minibatch loss "
        "(default: config DEMO_MARGIN_WEIGHT). 0 with --demo-td-weight 0 = "
        "demo-prefix starts only, no demo gradient.",
    )
    parser.add_argument(
        "--demo-opening-steps",
        type=int,
        default=0,
        metavar="N",
        help="Keep only the first N transitions of each demo route in the demo "
        "store (opening-focused imitation; pair with --demo-margin-weight > 0 "
        "and --demo-td-weight 0). 0 = full routes.",
    )
    parser.add_argument(
        "--demo-margin-decay",
        type=int,
        default=0,
        metavar="EPISODES",
        help="Linearly decay the demo margin weight to zero over this many "
        "GLOBAL episodes (0 = constant weight for the whole run).",
    )
    parser.add_argument(
        "--demo-margin-reignite",
        type=int,
        default=0,
        metavar="EPISODE",
        help="From this GLOBAL episode on, floor the decayed margin scale at "
        "--demo-margin-reignite-scale (imitation re-fires once the ladder "
        "frontier reaches the demo-covered opening). 0 = off.",
    )
    parser.add_argument(
        "--demo-margin-reignite-scale",
        type=float,
        default=0.5,
        metavar="S",
        help="Scale floor applied from --demo-margin-reignite onward.",
    )
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Persist per-milestone policy weights (policy_seed<S>_ep<N>.pth) so any "
        "'best checkpoint' can be re-evaluated on fresh seeds before promotion — the "
        "RUN-24 winner's-curse guard.",
    )
    parser.add_argument(
        "--stall-trace",
        action="store_true",
        help="On held-out greedy play, characterise WHY episodes STALL (the dominant failure "
        "after RUN-18): trapped (objective physically unreachable from the stuck tile), "
        "last-mile (near the objective), far, and oscillating-in-place. Picks whether the "
        "stall fix is control/jump skill, exploration, or an unexecutable route signal.",
    )
    parser.add_argument(
        "--record-play",
        type=int,
        default=0,
        metavar="N",
        help="After training, record the greedy policy PLAYING N held-out levels as labeled "
        "GIFs (under <out>/seed_<s>/replays/), preferring stalls. Lets us watch the behaviour "
        "the traces measure. Deterministic, so re-running a past experiment reproduces it.",
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
        reward_clip=args.reward_clip,
        stall_window=args.stall_window,
        max_steps=args.max_steps,
        death_penalty=args.death_penalty,
        hit_penalty=args.hit_penalty,
        ngu_bonus=args.ngu_bonus,
        ngu_beta=args.ngu_beta,
        enemy_motion=args.enemy_motion,
        win_at_k=args.win_at_k,
        win_at_k_ramp=args.win_at_k_ramp,
        win_at_k_ramp_delay=args.win_at_k_ramp_delay,
        save_weights=args.save_weights,
        demo_dir=args.demo_dir,
        demo_pretrain=args.demo_pretrain,
        demo_reset_p=args.demo_reset_p,
        demo_td_weight=args.demo_td_weight,
        demo_margin_weight=args.demo_margin_weight,
        demo_opening_steps=args.demo_opening_steps,
        demo_margin_decay=args.demo_margin_decay,
        demo_margin_reignite=args.demo_margin_reignite,
        demo_margin_reignite_scale=args.demo_margin_reignite_scale,
        demo_backward=args.demo_backward,
        demo_backward_retreat=args.demo_backward_retreat,
        demo_backward_wins=args.demo_backward_wins,
        demo_level_bias=args.demo_level_bias,
        demo_backward_window=args.demo_backward_window,
        demo_backward_deep=args.demo_backward_deep,
        demo_heal=args.demo_heal,
        resume_weights=args.resume_weights,
        ladder_init=args.ladder_init,
        regenerate_each_episode=args.regenerate_each_episode,
        drop_leak_features=args.drop_leak_features,
        use_cnn=args.cnn,
        geodesic=args.geodesic,
        geodesic_weight=args.geodesic_weight,
        geodesic_after_unlock=args.geodesic_after_unlock,
        reverse_exit_curriculum_p=args.reverse_exit_curriculum_p,
        reverse_exit_curriculum_far=args.reverse_exit_curriculum_far,
        geo_compass=args.geo_compass,
        geo_compass_hazard_aware=args.geo_compass_hazard_aware,
        route_aux=args.route_aux,
        curriculum_easy_episodes=args.curriculum_easy_episodes,
        curriculum_pretrain_difficulty=args.curriculum_pretrain_difficulty,
        leg2_probe=args.leg2_probe,
        death_trace=args.death_trace,
        stall_trace=args.stall_trace,
        record_play=args.record_play,
        imported=args.imported,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
