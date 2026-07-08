# ruff: noqa: F401,F403,F405,I001
from dataclasses import dataclass

from .common import *
from .config_helpers import *
from .io_utils import *
from .stats import *


@dataclass(frozen=True)
class _GreedyAgentState:
    epsilon: float | None
    had_epsilon: bool
    policy_net: Any
    was_training: bool


def _action_labels(game: CrystalCaves) -> list[str]:
    labels = list(getattr(game, "ACTION_LABELS", []))
    if labels:
        return labels
    return [str(index) for index in range(game.action_size)]


def _action_label(action_labels: list[str], action: int) -> str:
    return action_labels[action] if action < len(action_labels) else str(action)


def _resolved_end_reason(
    info: dict[str, Any],
    *,
    steps: int,
    max_steps: int,
) -> str:
    reason = str(info.get("end_reason", "") or "")
    if reason and reason != "running":
        return reason
    if info.get("won", False):
        return "won"
    return "timeout" if steps >= max_steps else "ended"


def _enter_greedy_agent_eval(agent: Any) -> _GreedyAgentState:
    had_epsilon = hasattr(agent, "epsilon")
    original_epsilon = float(agent.epsilon) if had_epsilon else None
    if had_epsilon:
        agent.epsilon = 0.0

    policy_net = getattr(agent, "policy_net", None)
    was_training = bool(getattr(policy_net, "training", False))
    if policy_net is not None and hasattr(policy_net, "eval"):
        policy_net.eval()

    return _GreedyAgentState(
        epsilon=original_epsilon,
        had_epsilon=had_epsilon,
        policy_net=policy_net,
        was_training=was_training,
    )


def _restore_greedy_agent_eval(agent: Any, state: _GreedyAgentState) -> None:
    if state.had_epsilon:
        agent.epsilon = state.epsilon
    if state.was_training and state.policy_net is not None and hasattr(state.policy_net, "train"):
        state.policy_net.train()


def final_eval(
    config: Config,
    agent: Any,
    *,
    out_dir: Path,
    label: str,
    episode: int,
    games: int,
) -> dict[str, Any]:
    eval_game = CrystalCaves(config, headless=True)
    evaluator = Evaluator(
        game=eval_game,
        agent=agent,
        config=config,
        log_dir=str(out_dir / "eval"),
    )
    results = evaluator.evaluate(
        num_episodes=games,
        max_steps=config.EVAL_MAX_STEPS,
        episode_num=episode,
    )
    evaluator.log_results(results)
    payload = results.to_dict()
    payload["label"] = label
    return payload


def q_value_snapshot(q_values: np.ndarray, action_labels: list[str]) -> dict[str, Any]:
    order = np.argsort(q_values)[::-1]
    top = [
        {
            "action": int(index),
            "label": _action_label(action_labels, int(index)),
            "q": round(float(q_values[int(index)]), 4),
        }
        for index in order[:3]
    ]
    margin = float(q_values[int(order[0])] - q_values[int(order[1])]) if len(order) > 1 else 0.0
    return {
        "top": top,
        "margin": round(margin, 4),
        "mean": round(float(np.mean(q_values)), 4),
        "std": round(float(np.std(q_values)), 4),
    }


def objective_snapshot(game: CrystalCaves) -> dict[str, Any]:
    target, distance = game._current_target()
    player_tile = game._player_tile()
    player_center = game._player_center()
    payload: dict[str, Any] = {
        "player_tile": [int(player_tile[0]), int(player_tile[1])],
        "player_center": [round(float(player_center[0]), 2), round(float(player_center[1]), 2)],
        "target_kind": None,
        "target_tile": None,
        "target_distance_tiles": None,
    }
    if target is not None:
        kind, col, row = target
        payload.update(
            {
                "target_kind": kind,
                "target_tile": [int(col), int(row)],
                "target_distance_tiles": round(float(distance / game.TILE_SIZE), 3),
            }
        )
    return payload


def compact_trace_step(
    *,
    step: int,
    action: int,
    action_labels: list[str],
    reward: float,
    info: dict[str, Any],
    objective: dict[str, Any],
    q_values: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": int(step),
        "action": int(action),
        "action_label": _action_label(action_labels, action),
        "reward": round(float(reward), 4),
        "progress": float(info.get("progress", 0.0) or 0.0),
        "crystals_remaining": int(info.get("crystals_remaining", 0) or 0),
        "exit_unlocked": bool(info.get("exit_unlocked", False)),
        "steps_since_progress": int(info.get("steps_since_progress", 0) or 0),
        "objective": objective,
    }
    if q_values is not None:
        row["q"] = q_value_snapshot(q_values, action_labels)
    return row


def classify_trace_failure(row: dict[str, Any]) -> list[str]:
    if row.get("won"):
        return ["won"]

    modes: list[str] = []
    collected = int(row.get("crystals_collected", 0) or 0)
    initial = int(row.get("initial_crystals", 0) or 0)
    if collected <= 0:
        modes.append("no_crystal")
    elif collected < initial:
        modes.append("partial_crystals")
    elif row.get("exit_unlocked"):
        modes.append("exit_unlocked_no_exit")

    if (
        row.get("end_reason") == "stalled"
        or int(row.get("final_steps_since_progress", 0) or 0) >= 600
    ):
        modes.append("stalled")
    if float(row.get("max_tile_visit_frac", 0.0) or 0.0) >= 0.18:
        modes.append("tile_loop")
    if float(row.get("idle_action_frac", 0.0) or 0.0) >= 0.35:
        modes.append("idle_heavy")
    if float(row.get("interact_action_frac", 0.0) or 0.0) >= 0.20:
        modes.append("interact_heavy")
    if float(row.get("shoot_action_frac", 0.0) or 0.0) >= 0.20:
        modes.append("shoot_heavy")
    if not modes:
        modes.append("timeout_navigation")
    return modes


def _mean_crystal_frac(rows: list[dict[str, Any]]) -> float:
    """Audit R2-B: TRUE mean collection fraction = mean(crystals_collected / initial_crystals).
    Distinct from any_crystal_rate (= fraction of games collecting >=1), which over-reports
    multi-crystal difficulties (e.g. 0.83 collected-any vs ~0.20 mean fraction)."""
    if not rows:
        return 0.0
    fracs = [
        (
            int(row.get("crystals_collected", 0) or 0)
            / max(1, int(row.get("initial_crystals", 1) or 1))
        )
        for row in rows
    ]
    return float(np.mean(fracs))


def trace_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "games": 0,
            "wins": 0,
            "win_rate": 0.0,
            "any_crystal_rate": 0.0,
            "mean_crystal_frac": 0.0,
            "all_crystals_rate": 0.0,
            "mean_progress": 0.0,
            "mean_depth_frac": 0.0,
            "mean_target_distance_delta_tiles": 0.0,
            "mean_target_distance_best_delta_tiles": 0.0,
            "mean_anti_loop_penalty_total": 0.0,
            "mean_interact_action_frac": 0.0,
            "mean_invalid_interact_count": 0.0,
            "mean_invalid_interact_penalty_total": 0.0,
            "mean_shoot_action_frac": 0.0,
            "mean_invalid_shoot_count": 0.0,
            "mean_invalid_shoot_penalty_total": 0.0,
            "mean_novelty_bonus_total": 0.0,
            "failure_mode_counts": {},
            "end_reason_counts": {},
        }

    failure_modes = Counter(mode for row in rows for mode in row.get("failure_modes", []))
    end_reasons = Counter(str(row.get("end_reason", "unknown")) for row in rows)
    return {
        "games": len(rows),
        "wins": int(sum(1 for row in rows if row.get("won"))),
        "win_rate": float(np.mean([bool(row.get("won")) for row in rows])),
        "any_crystal_rate": float(np.mean([row.get("crystals_collected", 0) > 0 for row in rows])),
        # Audit R2-B: TRUE mean collection fraction (NOT the collected-≥1 rate above). The
        # near-miss payload mislabeled any_crystal_rate as mean_crystal_frac, over-reporting
        # multi-crystal collection (e.g. 0.83 vs a real ~0.20) into promotion/scorecard.
        "mean_crystal_frac": _mean_crystal_frac(rows),
        "all_crystals_rate": float(np.mean([bool(row.get("exit_unlocked")) for row in rows])),
        "mean_progress": float(np.mean([row.get("final_progress", 0.0) for row in rows])),
        "mean_depth_frac": float(np.mean([row.get("final_depth_frac", 0.0) for row in rows])),
        "mean_unique_tiles": float(np.mean([row.get("unique_tiles", 0) for row in rows])),
        "mean_max_tile_visit_frac": float(
            np.mean([row.get("max_tile_visit_frac", 0.0) for row in rows])
        ),
        "mean_idle_action_frac": float(np.mean([row.get("idle_action_frac", 0.0) for row in rows])),
        "mean_q_margin": float(np.mean([row.get("mean_q_margin", 0.0) for row in rows])),
        "mean_target_distance_delta_tiles": float(
            np.mean([row.get("target_distance_delta_tiles", 0.0) for row in rows])
        ),
        "mean_target_distance_best_delta_tiles": float(
            np.mean([row.get("target_distance_best_delta_tiles", 0.0) for row in rows])
        ),
        "mean_anti_loop_penalty_total": float(
            np.mean([row.get("anti_loop_penalty_total", 0.0) for row in rows])
        ),
        "mean_interact_action_frac": float(
            np.mean([row.get("interact_action_frac", 0.0) for row in rows])
        ),
        "mean_invalid_interact_count": float(
            np.mean([row.get("invalid_interact_count", 0.0) for row in rows])
        ),
        "mean_invalid_interact_penalty_total": float(
            np.mean([row.get("invalid_interact_penalty_total", 0.0) for row in rows])
        ),
        "mean_shoot_action_frac": float(
            np.mean([row.get("shoot_action_frac", 0.0) for row in rows])
        ),
        "mean_invalid_shoot_count": float(
            np.mean([row.get("invalid_shoot_count", 0.0) for row in rows])
        ),
        "mean_invalid_shoot_penalty_total": float(
            np.mean([row.get("invalid_shoot_penalty_total", 0.0) for row in rows])
        ),
        "mean_novelty_bonus_total": float(
            np.mean([row.get("novelty_bonus_total", 0.0) for row in rows])
        ),
        "failure_mode_counts": dict(failure_modes),
        "end_reason_counts": dict(end_reasons),
    }


def near_miss_band_key(distance: float) -> str:
    text = str(int(distance)) if float(distance).is_integer() else str(distance).replace(".", "_")
    return f"near_miss_rate_{text}"


def first_objective_near_miss_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        empty = {
            "games": 0,
            "wins": 0,
            "win_rate": 0.0,
            "any_crystal_rate": 0.0,
            "mean_crystal_frac": 0.0,
            "mean_progress": 0.0,
            "mean_depth_frac": 0.0,
            "mean_initial_target_distance_tiles": 0.0,
            "mean_min_target_distance_tiles": 0.0,
            "mean_final_target_distance_tiles": 0.0,
            "mean_target_distance_best_delta_tiles": 0.0,
            "mean_target_distance_final_delta_tiles": 0.0,
            "mean_step_of_best_approach": 0.0,
            "stuck_after_close_rate": 0.0,
            "loop_after_close_rate": 0.0,
            "mean_close_zone_steps": 0.0,
            "mean_close_zone_jump_rate": 0.0,
            "mean_close_zone_idle_or_interact_rate": 0.0,
            "mean_shoot_action_frac": 0.0,
            "mean_invalid_shoot_count": 0.0,
            "mean_invalid_shoot_penalty_total": 0.0,
            "end_reason_counts": {},
        }
        for band in NEAR_MISS_DISTANCE_BANDS:
            empty[near_miss_band_key(band)] = 0.0
        return empty

    def number(row: dict[str, Any], key: str) -> float | None:
        value = row.get(key)
        return float(value) if value is not None else None

    end_reasons = Counter(str(row.get("end_reason", "unknown")) for row in rows)
    rollup: dict[str, Any] = {
        "games": len(rows),
        "wins": int(sum(1 for row in rows if row.get("won"))),
        "win_rate": float(np.mean([bool(row.get("won")) for row in rows])),
        "any_crystal_rate": float(np.mean([row.get("crystals_collected", 0) > 0 for row in rows])),
        "mean_crystal_frac": _mean_crystal_frac(rows),  # Audit R2-B: true mean, not the rate
        "mean_progress": float(np.mean([row.get("final_progress", 0.0) for row in rows])),
        "mean_depth_frac": float(np.mean([row.get("final_depth_frac", 0.0) for row in rows])),
        "mean_target_distance_best_delta_tiles": float(
            np.mean([row.get("target_distance_best_delta_tiles", 0.0) for row in rows])
        ),
        "mean_target_distance_final_delta_tiles": float(
            np.mean([row.get("target_distance_final_delta_tiles", 0.0) for row in rows])
        ),
        "mean_step_of_best_approach": float(
            np.mean([row.get("step_of_best_approach", 0) for row in rows])
        ),
        "stuck_after_close_rate": float(
            np.mean([bool(row.get("stuck_after_close")) for row in rows])
        ),
        "loop_after_close_rate": float(
            np.mean([bool(row.get("loop_after_close")) for row in rows])
        ),
        "mean_close_zone_steps": float(np.mean([row.get("close_zone_steps", 0) for row in rows])),
        "mean_close_zone_jump_rate": float(
            np.mean([row.get("close_zone_jump_rate", 0.0) for row in rows])
        ),
        "mean_close_zone_idle_or_interact_rate": float(
            np.mean([row.get("close_zone_idle_or_interact_rate", 0.0) for row in rows])
        ),
        "mean_shoot_action_frac": float(
            np.mean([row.get("shoot_action_frac", 0.0) for row in rows])
        ),
        "mean_invalid_shoot_count": float(
            np.mean([row.get("invalid_shoot_count", 0.0) for row in rows])
        ),
        "mean_invalid_shoot_penalty_total": float(
            np.mean([row.get("invalid_shoot_penalty_total", 0.0) for row in rows])
        ),
        "end_reason_counts": dict(end_reasons),
    }
    for metric in (
        "initial_target_distance_tiles",
        "min_target_distance_tiles",
        "final_target_distance_tiles",
    ):
        values = [number(row, metric) for row in rows]
        present = [value for value in values if value is not None]
        rollup[f"mean_{metric}"] = float(np.mean(present)) if present else 0.0
    for band in NEAR_MISS_DISTANCE_BANDS:
        key = near_miss_band_key(band)
        rollup[key] = float(
            np.mean(
                [
                    (row.get("min_target_distance_tiles") is not None)
                    and float(row.get("min_target_distance_tiles", 0.0)) <= band
                    for row in rows
                ]
            )
        )
    return rollup


def first_objective_near_miss_eval(
    config: Config,
    agent: Any,
    *,
    out_dir: Path,
    label: str,
    episode: int,
    games: int,
    max_steps: int,
    action_selector: Any | None = None,
) -> dict[str, Any] | None:
    if games <= 0:
        return None
    if max_steps <= 0:
        raise ValueError("near-miss max_steps must be positive")

    eval_dir = out_dir / "near_miss_eval" / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows_path = eval_dir / "per_level_eval.jsonl"
    rows_path.unlink(missing_ok=True)

    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    action_labels = _action_labels(game)

    rows: list[dict[str, Any]] = []
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for game_index in range(games):
            state = game.reset()
            initial_crystals = int(game.initial_crystals)
            target_distances: list[float] = []
            target_kind_counts: Counter[str] = Counter()
            action_counts: Counter[str] = Counter()
            close_zone_actions: Counter[str] = Counter()
            tile_counts: Counter[tuple[int, int]] = Counter()
            best_distance: float | None = None
            step_of_best = 0
            first_close_step: int | None = None
            done = False
            info = game._info()
            steps = 0

            for step in range(max_steps):
                objective = objective_snapshot(game)
                target_distance = objective.get("target_distance_tiles")
                if target_distance is not None:
                    distance = float(target_distance)
                    target_distances.append(distance)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        step_of_best = step
                    if distance <= CLOSE_ZONE_DISTANCE_TILES:
                        if first_close_step is None:
                            first_close_step = step
                target_kind_counts[str(objective.get("target_kind") or "none")] += 1

                if action_selector is None:
                    action = agent.select_action(state, training=False)
                else:
                    action = int(action_selector(agent, state, game, info, step, action_labels))
                action_label = _action_label(action_labels, action)
                if (
                    target_distance is not None
                    and float(target_distance) <= CLOSE_ZONE_DISTANCE_TILES
                ):
                    close_zone_actions[action_label] += 1
                action_counts[action_label] += 1

                state, _, done, info = game.step(action)
                steps = step + 1
                tile_counts[game._player_tile()] += 1
                if done:
                    break

            final_objective = objective_snapshot(game)
            final_distance = final_objective.get("target_distance_tiles")
            if final_distance is not None:
                distance = float(final_distance)
                target_distances.append(distance)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    step_of_best = steps
            initial_distance = target_distances[0] if target_distances else None
            min_distance = min(target_distances) if target_distances else None
            if (
                first_close_step is None
                and min_distance is not None
                and min_distance <= CLOSE_ZONE_DISTANCE_TILES
            ):
                first_close_step = step_of_best

            reason = _resolved_end_reason(info, steps=steps, max_steps=max_steps)
            parts = info.get("progress_parts") or {}
            if not isinstance(parts, dict):
                parts = {}
            collected = initial_crystals - int(info.get("crystals_remaining", 0) or 0)
            max_tile_visits = max(tile_counts.values()) if tile_counts else 0
            close_steps = sum(close_zone_actions.values())
            close_jump_actions = sum(
                count for label_name, count in close_zone_actions.items() if "JUMP" in label_name
            )
            close_idle_or_interact = sum(
                count
                for label_name, count in close_zone_actions.items()
                if label_name in {"IDLE", "INTERACT"}
            )
            shoot_count = sum(
                count for label_name, count in action_counts.items() if "SHOOT" in label_name
            )
            max_tile_visit_frac = float(max_tile_visits / max(1, steps))
            row: dict[str, Any] = {
                "game_index": game_index,
                "episode": int(episode),
                "level": info.get("level"),
                "level_name": info.get("level_name"),
                "steps": steps,
                "end_reason": reason,
                "won": bool(info.get("won", False)),
                "score": float(info.get("score", 0) or 0),
                "initial_crystals": initial_crystals,
                "crystals_collected": collected,
                "crystals_remaining": int(info.get("crystals_remaining", 0) or 0),
                "exit_unlocked": bool(info.get("exit_unlocked", False)),
                "final_progress": float(info.get("progress", 0.0) or 0.0),
                "final_depth_frac": float(parts.get("depth_frac", 0.0) or 0.0),
                "final_steps_since_progress": int(info.get("steps_since_progress", 0) or 0),
                "initial_target_distance_tiles": initial_distance,
                "min_target_distance_tiles": min_distance,
                "final_target_distance_tiles": final_distance,
                "target_distance_best_delta_tiles": (
                    float(initial_distance - min_distance)
                    if initial_distance is not None and min_distance is not None
                    else 0.0
                ),
                "target_distance_final_delta_tiles": (
                    float(initial_distance - final_distance)
                    if initial_distance is not None and final_distance is not None
                    else 0.0
                ),
                "step_of_best_approach": int(step_of_best),
                "first_close_step": first_close_step,
                "close_zone_distance_tiles": CLOSE_ZONE_DISTANCE_TILES,
                "close_zone_steps": int(close_steps),
                "close_zone_action_counts": dict(close_zone_actions),
                "close_zone_jump_rate": float(close_jump_actions / max(1, close_steps)),
                "close_zone_idle_or_interact_rate": float(
                    close_idle_or_interact / max(1, close_steps)
                ),
                "target_kind_counts": dict(target_kind_counts),
                "unique_tiles": len(tile_counts),
                "max_tile_visit_frac": max_tile_visit_frac,
                "shoot_action_frac": float(shoot_count / max(1, steps)),
                "invalid_shoot_count": int(info.get("invalid_shoot_count", 0) or 0),
                "invalid_shoot_penalty_total": float(
                    info.get("invalid_shoot_penalty_total", 0.0) or 0.0
                ),
                "top_actions": dict(action_counts.most_common(5)),
                "final_objective": final_objective,
            }
            for band in NEAR_MISS_DISTANCE_BANDS:
                row[f"within_{near_miss_band_key(band).removeprefix('near_miss_rate_')}_tiles"] = (
                    min_distance is not None and min_distance <= band
                )
            row["stuck_after_close"] = bool(
                first_close_step is not None
                and (
                    reason == "stalled" or int(row.get("final_steps_since_progress", 0) or 0) >= 600
                )
            )
            row["loop_after_close"] = bool(
                first_close_step is not None and max_tile_visit_frac >= 0.18
            )
            rows.append(row)
            append_jsonl(rows_path, row)
    finally:
        _restore_greedy_agent_eval(agent, agent_state)

    summary = {
        "label": label,
        "episode": int(episode),
        "games": games,
        "max_steps": max_steps,
        "close_zone_distance_tiles": CLOSE_ZONE_DISTANCE_TILES,
        "rows_path": str(rows_path),
        "summary_path": str(eval_dir / "summary.json"),
        "rollup": first_objective_near_miss_rollup(rows),
        "rows": rows,
    }
    write_json(eval_dir / "summary.json", summary)
    return summary


def trace_heldout_failures(
    config: Config,
    agent: Any,
    *,
    out_dir: Path,
    label: str,
    games: int,
    max_steps: int,
    sample_every: int,
    tail_steps: int,
) -> dict[str, Any] | None:
    if games <= 0:
        return None
    if max_steps <= 0:
        raise ValueError("trace max_steps must be positive")
    if sample_every <= 0:
        raise ValueError("trace sample_every must be positive")
    if tail_steps <= 0:
        raise ValueError("trace tail_steps must be positive")

    trace_dir = out_dir / "diagnostics" / label
    trace_dir.mkdir(parents=True, exist_ok=True)
    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    action_labels = _action_labels(game)

    rows: list[dict[str, Any]] = []
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for game_index in range(games):
            state = game.reset()
            initial_crystals = int(game.initial_crystals)
            action_counts: Counter[str] = Counter()
            tile_counts: Counter[tuple[int, int]] = Counter()
            target_kind_counts: Counter[str] = Counter()
            q_margins: list[float] = []
            target_distances: list[float] = []
            samples: list[dict[str, Any]] = []
            tail: deque[dict[str, Any]] = deque(maxlen=tail_steps)
            events: list[dict[str, Any]] = []
            last_progress = float(game._progress)
            last_crystals_remaining = len(game.crystals)
            last_exit_unlocked = bool(game.exit_unlocked)
            done = False
            info = game._info()
            steps = 0
            total_reward = 0.0

            for step in range(max_steps):
                objective = objective_snapshot(game)
                target_distance = objective.get("target_distance_tiles")
                if target_distance is not None:
                    target_distances.append(float(target_distance))
                target_kind = str(objective.get("target_kind") or "none")
                target_kind_counts[target_kind] += 1
                should_sample = step < 10 or step % sample_every == 0
                q_values = agent.get_q_values(state) if should_sample else None
                if q_values is not None:
                    action = int(np.argmax(q_values))
                    q_summary = q_value_snapshot(q_values, action_labels)
                    q_margins.append(float(q_summary["margin"]))
                else:
                    action = agent.select_action(state, training=False)
                action_label = _action_label(action_labels, action)
                state, reward, done, info = game.step(action)
                steps = step + 1
                total_reward += float(reward)

                tile = game._player_tile()
                tile_counts[tile] += 1
                compact = compact_trace_step(
                    step=steps,
                    action=action,
                    action_labels=action_labels,
                    reward=reward,
                    info=info,
                    objective=objective,
                    q_values=q_values,
                )
                if should_sample or done:
                    samples.append(compact)
                tail.append(compact)
                action_counts[action_label] += 1

                progress = float(info.get("progress", 0.0) or 0.0)
                crystals_remaining = int(info.get("crystals_remaining", 0) or 0)
                exit_unlocked = bool(info.get("exit_unlocked", False))
                if progress > last_progress + 1e-6:
                    events.append({"step": steps, "kind": "progress", "progress": progress})
                    last_progress = progress
                if crystals_remaining < last_crystals_remaining:
                    events.append(
                        {
                            "step": steps,
                            "kind": "crystal_collected",
                            "crystals_remaining": crystals_remaining,
                        }
                    )
                    last_crystals_remaining = crystals_remaining
                if exit_unlocked and not last_exit_unlocked:
                    events.append({"step": steps, "kind": "exit_unlocked"})
                    last_exit_unlocked = True
                if done:
                    break

            reason = _resolved_end_reason(info, steps=steps, max_steps=max_steps)
            collected = initial_crystals - int(info.get("crystals_remaining", 0) or 0)
            max_tile_visits = max(tile_counts.values()) if tile_counts else 0
            idle_count = action_counts.get("IDLE", 0)
            interact_count = action_counts.get("INTERACT", 0)
            shoot_count = sum(
                count for label_name, count in action_counts.items() if "SHOOT" in label_name
            )
            final_objective = objective_snapshot(game)
            final_distance = final_objective.get("target_distance_tiles")
            if final_distance is not None:
                target_distances.append(float(final_distance))
            initial_distance = target_distances[0] if target_distances else None
            min_distance = min(target_distances) if target_distances else None
            parts = info.get("progress_parts") or {}
            if not isinstance(parts, dict):
                parts = {}
            row: dict[str, Any] = {
                "game_index": game_index,
                "level": info.get("level"),
                "level_name": info.get("level_name"),
                "steps": steps,
                "end_reason": reason,
                "won": bool(info.get("won", False)),
                "score": float(info.get("score", 0) or 0),
                "total_reward": round(total_reward, 4),
                "initial_crystals": initial_crystals,
                "crystals_collected": collected,
                "crystals_remaining": int(info.get("crystals_remaining", 0) or 0),
                "exit_unlocked": bool(info.get("exit_unlocked", False)),
                "final_progress": float(info.get("progress", 0.0) or 0.0),
                "final_depth_frac": float(parts.get("depth_frac", 0.0) or 0.0),
                "final_steps_since_progress": int(info.get("steps_since_progress", 0) or 0),
                "unique_tiles": len(tile_counts),
                "max_tile_visits": int(max_tile_visits),
                "max_tile_visit_frac": float(max_tile_visits / max(1, steps)),
                "idle_action_frac": float(idle_count / max(1, steps)),
                "interact_action_frac": float(interact_count / max(1, steps)),
                "shoot_action_frac": float(shoot_count / max(1, steps)),
                "top_actions": dict(action_counts.most_common(5)),
                "target_kind_counts": dict(target_kind_counts),
                "mean_q_margin": float(np.mean(q_margins)) if q_margins else 0.0,
                "initial_target_distance_tiles": initial_distance,
                "min_target_distance_tiles": min_distance,
                "final_target_distance_tiles": final_distance,
                "target_distance_delta_tiles": (
                    float(initial_distance - final_distance)
                    if initial_distance is not None and final_distance is not None
                    else 0.0
                ),
                "target_distance_best_delta_tiles": (
                    float(initial_distance - min_distance)
                    if initial_distance is not None and min_distance is not None
                    else 0.0
                ),
                "anti_loop_penalty_total": float(info.get("anti_loop_penalty_total", 0.0) or 0.0),
                "invalid_interact_count": int(info.get("invalid_interact_count", 0) or 0),
                "invalid_interact_penalty_total": float(
                    info.get("invalid_interact_penalty_total", 0.0) or 0.0
                ),
                "invalid_shoot_count": int(info.get("invalid_shoot_count", 0) or 0),
                "invalid_shoot_penalty_total": float(
                    info.get("invalid_shoot_penalty_total", 0.0) or 0.0
                ),
                "novelty_bonus_total": float(info.get("novelty_bonus_total", 0.0) or 0.0),
                "final_objective": final_objective,
                "events": events,
                "trace_file": f"game_{game_index:02d}.json",
            }
            row["failure_modes"] = classify_trace_failure(row)
            trace_payload = {
                "summary": row,
                "samples": samples,
                "tail": list(tail),
            }
            write_json(trace_dir / row["trace_file"], trace_payload)
            rows.append(row)
    finally:
        _restore_greedy_agent_eval(agent, agent_state)

    summary = {
        "label": label,
        "games": games,
        "max_steps": max_steps,
        "sample_every": sample_every,
        "tail_steps": tail_steps,
        "trace_dir": str(trace_dir),
        "rollup": trace_rollup(rows),
        "games_summary": rows,
    }
    write_json(trace_dir / "summary.json", summary)
    return summary


def per_skill_eval(
    agent: Any,
    config: Config,
    *,
    k: int,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    return level_set_eval(agent, config, specs=DRILL_CAVES, k=k, max_steps=max_steps)


def level_set_eval(
    agent: Any,
    config: Config,
    *,
    specs: tuple[Any, ...],
    k: int,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    game = CrystalCaves(config, headless=True)
    game._randomize_levels = False
    step_limit = max_steps or config.EVAL_MAX_STEPS
    max_step_reason_limit = int(step_limit)
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        wins = 0
        collected_any = 0
        all_crystals = 0
        scores: list[float] = []
        steps_list: list[int] = []
        progress: list[float] = []
        end_reasons: Counter[str] = Counter()
        progress_parts: list[dict[str, float]] = []

        for _ in range(k):
            game.level_index = index
            state = game.reset()
            initial_crystals = max(1, game.initial_crystals)
            done = False
            info: dict[str, Any] = {}
            steps = 0
            while not done and steps < step_limit:
                action = agent.select_action(state, training=False)
                state, _, done, info = game.step(action)
                steps += 1

            won = bool(info.get("won", False))
            wins += int(won)
            collected_any += int(len(game.crystals) < initial_crystals)
            all_crystals += int(game.exit_unlocked)
            scores.append(float(info.get("score", 0)))
            steps_list.append(steps)
            progress.append(float(info.get("progress", 0.0) or 0.0))
            reason = _resolved_end_reason(
                info,
                steps=steps,
                max_steps=max_step_reason_limit,
            )
            end_reasons[reason] += 1
            parts = info.get("progress_parts") or {}
            if isinstance(parts, dict):
                progress_parts.append({key: float(parts.get(key, 0.0) or 0.0) for key in parts})

        rows.append(
            {
                "index": index,
                "name": spec.name,
                "runs": k,
                "wins": wins,
                "win_rate": wins / k,
                "collected_any_rate": collected_any / k,
                "all_crystals_rate": all_crystals / k,
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
                "mean_progress": float(np.mean(progress)) if progress else 0.0,
                "end_reason_counts": dict(end_reasons),
                "mean_progress_parts": mean_dicts(progress_parts),
            }
        )
    return rows


def level_eval_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "mean_win_rate": 0.0,
            "mean_any_crystal_rate": 0.0,
            "mean_all_crystals_rate": 0.0,
            "mean_progress": 0.0,
            "solved_levels": 0,
            "levels": 0,
        }
    return {
        "mean_win_rate": float(np.mean([row.get("win_rate", 0.0) for row in rows])),
        "mean_any_crystal_rate": float(
            np.mean([row.get("collected_any_rate", 0.0) for row in rows])
        ),
        "mean_all_crystals_rate": float(
            np.mean([row.get("all_crystals_rate", 0.0) for row in rows])
        ),
        "mean_progress": float(np.mean([row.get("mean_progress", 0.0) for row in rows])),
        "solved_levels": int(sum(1 for row in rows if row.get("win_rate", 0.0) >= 1.0)),
        "levels": len(rows),
    }
