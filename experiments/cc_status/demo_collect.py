# ruff: noqa: F401,F403,F405,I001
from dataclasses import dataclass

from .common import *
from .config_helpers import *
from .demo_planners import *
from .evals import *

_DemoTransition = tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass(frozen=True)
class _RouteDemoAttempt:
    row: dict[str, Any]
    trajectory: list[_DemoTransition]
    close_zone_trajectory: list[_DemoTransition]
    oracle_close_zone_trajectory: list[_DemoTransition]

    @property
    def won(self) -> bool:
        return bool(self.row.get("won", False))


def _validate_route_demo_collection_args(
    *,
    max_levels: int,
    max_steps: int,
    close_zone_distance_tiles: float,
    controller_variants: tuple[str, ...],
    oracle_close_zone_stride: int,
    oracle_close_zone_max_per_trajectory: int,
) -> None:
    if max_levels <= 0:
        raise ValueError("demo max_levels must be positive")
    if max_steps <= 0:
        raise ValueError("demo max_steps must be positive")
    if close_zone_distance_tiles <= 0:
        raise ValueError("close_zone_distance_tiles must be positive")
    if not controller_variants:
        raise ValueError("at least one route demo controller variant is required")
    unknown_variants = set(controller_variants) - ROUTE_DEMO_VARIANTS
    if unknown_variants:
        raise ValueError(f"unknown route demo controller variants: {sorted(unknown_variants)}")
    if oracle_close_zone_stride <= 0:
        raise ValueError("oracle_close_zone_stride must be positive")
    if oracle_close_zone_max_per_trajectory < 0:
        raise ValueError("oracle_close_zone_max_per_trajectory must be non-negative")


def _route_demo_action(
    game: CrystalCaves,
    *,
    variant: str,
    stale_steps: int,
    planned_actions: list[int],
) -> int:
    if variant == "beam":
        if not planned_actions:
            planned_actions.extend(route_beam_plan(game, stale_steps=stale_steps))
        return planned_actions.pop(0)
    return route_floor_scripted_action(
        game,
        variant=variant,
        stale_steps=stale_steps,
    )


def _route_demo_end_reason(info: dict[str, Any], *, won: bool, steps: int, max_steps: int) -> str:
    reason = str(info.get("end_reason", "") or "")
    if reason and reason != "running":
        return reason
    return "won" if won else ("timeout" if steps >= max_steps else "ended")


def _maybe_oracle_close_zone_label(
    game: CrystalCaves,
    *,
    state: np.ndarray,
    action: int,
    stale_steps: int,
    should_label: bool,
    trajectory: list[_DemoTransition],
    action_counts: Counter[str],
    oracle_scores: list[float],
) -> int:
    if not should_label:
        return 0

    oracle_action, oracle_meta = close_zone_oracle_action(
        game,
        stale_steps=stale_steps,
    )
    trajectory.append((state.copy(), oracle_action, 0.0, state.copy(), False))
    oracle_label = game.ACTION_LABELS[oracle_action]
    action_counts[oracle_label] += 1
    oracle_scores.append(float(oracle_meta.get("score", 0.0) or 0.0))
    return int(oracle_action != action)


def _route_demo_attempt_row(
    game: CrystalCaves,
    *,
    index: int,
    variant: str,
    won: bool,
    steps: int,
    max_steps: int,
    info: dict[str, Any],
    trajectory: list[_DemoTransition],
    close_zone_trajectory: list[_DemoTransition],
    action_counts: Counter[str],
    close_zone_action_counts: Counter[str],
    oracle_close_zone_action_counts: Counter[str],
    tile_counts: Counter[tuple[int, int]],
    target_distances: list[float],
    oracle_scores: list[float],
    oracle_relabels: int,
    oracle_labeled_steps: int,
    step_of_best: int,
) -> dict[str, Any]:
    initial_distance = target_distances[0] if target_distances else None
    min_distance = min(target_distances) if target_distances else None
    final_distance = target_distances[-1] if target_distances else None
    close_zone_steps = sum(close_zone_action_counts.values())
    close_zone_jump_actions = sum(
        count for label, count in close_zone_action_counts.items() if label in DEMO_JUMP_ACTIONS
    )
    close_zone_idle_interact_actions = sum(
        count
        for label, count in close_zone_action_counts.items()
        if label in DEMO_IDLE_INTERACT_ACTIONS
    )
    idle_interact_actions = sum(
        count for label, count in action_counts.items() if label in DEMO_IDLE_INTERACT_ACTIONS
    )
    row = {
        "index": index,
        "variant": variant,
        "name": game.level.name,
        "won": won,
        "steps": steps,
        "end_reason": _route_demo_end_reason(info, won=won, steps=steps, max_steps=max_steps),
        "kept_transitions": len(trajectory) if won else 0,
        "close_zone_kept_transitions": (len(close_zone_trajectory) if won else 0),
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
        "action_counts": dict(action_counts),
        "close_zone_action_counts": dict(close_zone_action_counts),
        "oracle_close_zone_action_counts": dict(oracle_close_zone_action_counts),
        "close_zone_steps": int(close_zone_steps),
        "close_zone_jump_rate": (
            float(close_zone_jump_actions / close_zone_steps) if close_zone_steps else 0.0
        ),
        "close_zone_idle_interact_rate": (
            float(close_zone_idle_interact_actions / close_zone_steps) if close_zone_steps else 0.0
        ),
        "idle_interact_rate": (float(idle_interact_actions / steps) if steps else 0.0),
        "oracle_close_zone_labeled_steps": int(oracle_labeled_steps),
        "oracle_close_zone_relabels": int(oracle_relabels),
        "oracle_close_zone_relabel_rate": (
            float(oracle_relabels / oracle_labeled_steps) if oracle_labeled_steps else 0.0
        ),
        "oracle_close_zone_mean_score": float(np.mean(oracle_scores)) if oracle_scores else 0.0,
        "max_tile_visit_frac": (
            float(max(tile_counts.values()) / steps) if tile_counts and steps else 0.0
        ),
    }
    row["failure_modes"] = classify_demo_attempt(row)
    return row


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _collect_route_demo_attempt(
    game: CrystalCaves,
    *,
    index: int,
    variant: str,
    max_steps: int,
    close_zone_distance_tiles: float,
    oracle_close_zone_labels: bool,
    oracle_close_zone_stride: int,
    oracle_close_zone_max_per_trajectory: int,
) -> _RouteDemoAttempt:
    game.level_index = index
    state = game.reset()
    trajectory: list[_DemoTransition] = []
    close_zone_trajectory: list[_DemoTransition] = []
    oracle_close_zone_trajectory: list[_DemoTransition] = []
    info: dict[str, Any] = {}
    done = False
    steps = 0
    action_counts: Counter[str] = Counter()
    close_zone_action_counts: Counter[str] = Counter()
    oracle_close_zone_action_counts: Counter[str] = Counter()
    tile_counts: Counter[tuple[int, int]] = Counter()
    target_distances: list[float] = []
    oracle_scores: list[float] = []
    oracle_relabels = 0
    oracle_labeled_steps = 0
    close_zone_seen_steps = 0
    best_distance: float | None = None
    step_of_best = 0
    planned_actions: list[int] = []

    while not done and steps < max_steps:
        _, target_distance = game._current_target()
        if target_distance < float("inf"):
            distance_tiles = float(target_distance / game.TILE_SIZE)
            target_distances.append(distance_tiles)
            if best_distance is None or distance_tiles < best_distance:
                best_distance = distance_tiles
                step_of_best = steps
        is_close_zone = (
            target_distance / game.TILE_SIZE <= close_zone_distance_tiles
            if target_distance < float("inf")
            else False
        )
        stale_steps = steps - step_of_best
        tile_counts[game._player_tile()] += 1
        action = _route_demo_action(
            game,
            variant=variant,
            stale_steps=stale_steps,
            planned_actions=planned_actions,
        )

        should_label_oracle = False
        if is_close_zone:
            should_label_oracle = (
                oracle_close_zone_labels
                and close_zone_seen_steps % oracle_close_zone_stride == 0
                and (
                    oracle_close_zone_max_per_trajectory == 0
                    or oracle_labeled_steps < oracle_close_zone_max_per_trajectory
                )
            )
            close_zone_seen_steps += 1
        oracle_relabels += _maybe_oracle_close_zone_label(
            game,
            state=state,
            action=action,
            stale_steps=stale_steps,
            should_label=should_label_oracle,
            trajectory=oracle_close_zone_trajectory,
            action_counts=oracle_close_zone_action_counts,
            oracle_scores=oracle_scores,
        )
        if should_label_oracle:
            oracle_labeled_steps += 1

        next_state, reward, done, info = game.step(action)
        transition = (state.copy(), action, float(reward), next_state.copy(), done)
        trajectory.append(transition)
        if is_close_zone:
            close_zone_trajectory.append(transition)
        action_label = game.ACTION_LABELS[action]
        action_counts[action_label] += 1
        if is_close_zone:
            close_zone_action_counts[action_label] += 1
        state = next_state
        steps += 1

    won = bool(info.get("won", False))
    row = _route_demo_attempt_row(
        game,
        index=index,
        variant=variant,
        won=won,
        steps=steps,
        max_steps=max_steps,
        info=info,
        trajectory=trajectory,
        close_zone_trajectory=close_zone_trajectory,
        action_counts=action_counts,
        close_zone_action_counts=close_zone_action_counts,
        oracle_close_zone_action_counts=oracle_close_zone_action_counts,
        tile_counts=tile_counts,
        target_distances=target_distances,
        oracle_scores=oracle_scores,
        oracle_relabels=oracle_relabels,
        oracle_labeled_steps=oracle_labeled_steps,
        step_of_best=step_of_best,
    )
    return _RouteDemoAttempt(
        row=row,
        trajectory=trajectory,
        close_zone_trajectory=close_zone_trajectory,
        oracle_close_zone_trajectory=oracle_close_zone_trajectory,
    )


def _route_demo_summary(
    *,
    levels: int,
    max_steps: int,
    close_zone_distance_tiles: float,
    controller_variants: tuple[str, ...],
    oracle_close_zone_labels: bool,
    oracle_close_zone_stride: int,
    oracle_close_zone_max_per_trajectory: int,
    trajectories: list[list[_DemoTransition]],
    close_zone_trajectories: list[list[_DemoTransition]],
    oracle_close_zone_trajectories: list[list[_DemoTransition]],
    kept_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    kept_transitions = int(sum(len(traj) for traj in trajectories))
    close_zone_kept_transitions = int(sum(len(traj) for traj in close_zone_trajectories))
    oracle_close_zone_kept_transitions = int(
        sum(len(traj) for traj in oracle_close_zone_trajectories)
    )
    wins = int(sum(1 for row in rows if row["won"]))
    failure_modes = Counter(mode for row in rows for mode in row.get("failure_modes", []))
    variant_counts = Counter(str(row.get("variant", "unknown")) for row in rows if row["won"])
    oracle_counts = Counter(
        label
        for row in kept_rows
        for label, count in (row.get("oracle_close_zone_action_counts") or {}).items()
        for _ in range(int(count))
    )
    failed_rows = [row for row in rows if not row["won"]]
    failed_min_distances = [
        float(row["min_target_distance_tiles"])
        for row in failed_rows
        if row.get("min_target_distance_tiles") is not None
    ]
    return {
        "levels": levels,
        "attempts": levels,
        "controller_attempts": len(rows),
        "wins": wins,
        "win_rate": wins / levels if levels else 0.0,
        "attempt_win_rate": wins / len(rows) if rows else 0.0,
        "kept_trajectories": len(trajectories),
        "kept_transitions": kept_transitions,
        "close_zone_distance_tiles": close_zone_distance_tiles,
        "close_zone_kept_transitions": close_zone_kept_transitions,
        "oracle_close_zone_kept_transitions": oracle_close_zone_kept_transitions,
        "oracle_close_zone_enabled": oracle_close_zone_labels,
        "oracle_close_zone_stride": oracle_close_zone_stride,
        "oracle_close_zone_max_per_trajectory": oracle_close_zone_max_per_trajectory,
        "oracle_close_zone_action_counts": dict(oracle_counts),
        "mean_kept_oracle_close_zone_relabel_rate": (
            float(np.mean([row.get("oracle_close_zone_relabel_rate", 0.0) for row in kept_rows]))
            if kept_rows
            else 0.0
        ),
        "max_steps": max_steps,
        "controller_variants": list(controller_variants),
        "kept_by_variant": dict(variant_counts),
        "failure_mode_counts": dict(failure_modes),
        "mean_kept_close_zone_jump_rate": (
            float(np.mean([row.get("close_zone_jump_rate", 0.0) for row in kept_rows]))
            if kept_rows
            else 0.0
        ),
        "mean_kept_idle_interact_rate": (
            float(np.mean([row.get("idle_interact_rate", 0.0) for row in kept_rows]))
            if kept_rows
            else 0.0
        ),
        "mean_kept_max_tile_visit_frac": (
            float(np.mean([row.get("max_tile_visit_frac", 0.0) for row in kept_rows]))
            if kept_rows
            else 0.0
        ),
        "mean_failed_min_target_distance_tiles": (_mean_or_zero(failed_min_distances)),
        "mean_failed_best_delta_tiles": (
            float(
                np.mean([row.get("target_distance_best_delta_tiles", 0.0) for row in failed_rows])
            )
            if failed_rows
            else 0.0
        ),
        "kept_rows": kept_rows,
        "rows": rows,
    }


def collect_scripted_route_demonstrations(
    config: Config,
    *,
    max_levels: int,
    max_steps: int,
    close_zone_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    controller_variants: tuple[str, ...] = ("direct",),
    oracle_close_zone_labels: bool = False,
    oracle_close_zone_stride: int = 4,
    oracle_close_zone_max_per_trajectory: int = 8,
) -> dict[str, Any]:
    _validate_route_demo_collection_args(
        max_levels=max_levels,
        max_steps=max_steps,
        close_zone_distance_tiles=close_zone_distance_tiles,
        controller_variants=controller_variants,
        oracle_close_zone_stride=oracle_close_zone_stride,
        oracle_close_zone_max_per_trajectory=oracle_close_zone_max_per_trajectory,
    )

    game = CrystalCaves(config, headless=True)
    game._randomize_levels = False
    levels = min(max_levels, len(game.CAVES))
    trajectories: list[list[_DemoTransition]] = []
    close_zone_trajectories: list[list[_DemoTransition]] = []
    oracle_close_zone_trajectories: list[list[_DemoTransition]] = []
    kept_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        for index in range(levels):
            level_won = False
            for variant in controller_variants:
                attempt = _collect_route_demo_attempt(
                    game,
                    index=index,
                    variant=variant,
                    max_steps=max_steps,
                    close_zone_distance_tiles=close_zone_distance_tiles,
                    oracle_close_zone_labels=oracle_close_zone_labels,
                    oracle_close_zone_stride=oracle_close_zone_stride,
                    oracle_close_zone_max_per_trajectory=oracle_close_zone_max_per_trajectory,
                )
                row = attempt.row
                if attempt.trajectory and attempt.won:
                    row["trajectory_index"] = len(trajectories)
                    trajectories.append(attempt.trajectory)
                    close_zone_trajectories.append(attempt.close_zone_trajectory)
                    oracle_close_zone_trajectories.append(attempt.oracle_close_zone_trajectory)
                    kept_rows.append(row.copy())
                    level_won = True
                rows.append(row)
                if level_won:
                    break
    finally:
        game.close()

    return {
        "trajectories": trajectories,
        "close_zone_trajectories": close_zone_trajectories,
        "oracle_close_zone_trajectories": oracle_close_zone_trajectories,
        "summary": _route_demo_summary(
            levels=levels,
            max_steps=max_steps,
            close_zone_distance_tiles=close_zone_distance_tiles,
            controller_variants=controller_variants,
            oracle_close_zone_labels=oracle_close_zone_labels,
            oracle_close_zone_stride=oracle_close_zone_stride,
            oracle_close_zone_max_per_trajectory=oracle_close_zone_max_per_trajectory,
            trajectories=trajectories,
            close_zone_trajectories=close_zone_trajectories,
            oracle_close_zone_trajectories=oracle_close_zone_trajectories,
            kept_rows=kept_rows,
            rows=rows,
        ),
    }


def select_route_demo_trajectories(
    demos: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Select/weight successful scripted demos for BC and replay seeding."""

    mode = parse_demo_selection_mode(mode)
    trajectories = list(demos.get("trajectories") or [])
    close_zone_trajectories = list(demos.get("close_zone_trajectories") or [])
    oracle_close_zone_trajectories = list(demos.get("oracle_close_zone_trajectories") or [])
    demo_summary = demos.get("summary") or {}
    kept_rows = list(demo_summary.get("kept_rows") or [])
    if not kept_rows:
        kept_rows = [
            {**row, "trajectory_index": i}
            for i, row in enumerate(demo_summary.get("rows") or [])
            if row.get("won")
        ]

    input_transitions = int(sum(len(traj) for traj in trajectories))
    input_close_zone_transitions = int(sum(len(traj) for traj in close_zone_trajectories))
    input_oracle_close_zone_transitions = int(
        sum(len(traj) for traj in oracle_close_zone_trajectories)
    )
    selected: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]] = []
    selected_close_zone: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]] = []
    selected_oracle_close_zone: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]] = []
    selected_rows: list[dict[str, Any]] = []
    excluded_reasons: Counter[str] = Counter()
    selected_by_variant: Counter[str] = Counter()
    weighted_by_variant: Counter[str] = Counter()

    step_values = [int(row.get("steps", 0) or 0) for row in kept_rows if row.get("steps")]
    step_cutoff = float(np.percentile(step_values, 75)) if step_values else 0.0

    for row in kept_rows:
        source_index = int(row.get("trajectory_index", -1))
        if source_index < 0 or source_index >= len(trajectories):
            excluded_reasons["missing_trajectory"] += 1
            continue

        variant = str(row.get("variant", "unknown"))
        steps = int(row.get("steps", 0) or 0)
        close_zone_steps = int(row.get("close_zone_steps", 0) or 0)
        close_jump = float(row.get("close_zone_jump_rate", 0.0) or 0.0)
        close_idle = float(row.get("close_zone_idle_interact_rate", 0.0) or 0.0)
        loop_frac = float(row.get("max_tile_visit_frac", 0.0) or 0.0)

        include = True
        weight = 1
        reason = ""
        if mode == "filtered-weighted":
            if variant == "beam":
                include = (
                    (not step_cutoff or steps <= step_cutoff)
                    and close_zone_steps > 0
                    and close_jump > 0.0
                    and close_idle <= 0.35
                    and loop_frac <= 0.70
                )
                if not include:
                    reason = "beam_quality_filter"
            elif variant in {"direct", "recovery"}:
                weight = 2
                if variant == "recovery" and step_cutoff and steps > step_cutoff:
                    weight = 1

        if not include:
            excluded_reasons[reason or "filtered"] += 1
            continue

        selected_by_variant[variant] += 1
        weighted_by_variant[variant] += weight
        row_with_weight = {
            **row,
            "selection_weight": weight,
            "source_trajectory_index": source_index,
        }
        selected_rows.append(row_with_weight)
        for _ in range(weight):
            selected.append(trajectories[source_index])
            if source_index < len(close_zone_trajectories):
                selected_close_zone.append(close_zone_trajectories[source_index])
            else:
                selected_close_zone.append([])
            if source_index < len(oracle_close_zone_trajectories):
                selected_oracle_close_zone.append(oracle_close_zone_trajectories[source_index])
            else:
                selected_oracle_close_zone.append([])

    selected_transitions = int(sum(len(traj) for traj in selected))
    selected_close_zone_transitions = int(sum(len(traj) for traj in selected_close_zone))
    selected_oracle_close_zone_transitions = int(
        sum(len(traj) for traj in selected_oracle_close_zone)
    )
    return {
        "trajectories": selected,
        "close_zone_trajectories": selected_close_zone,
        "oracle_close_zone_trajectories": selected_oracle_close_zone,
        "summary": {
            "mode": mode,
            "input_trajectories": len(trajectories),
            "input_transitions": input_transitions,
            "input_close_zone_transitions": input_close_zone_transitions,
            "input_oracle_close_zone_transitions": input_oracle_close_zone_transitions,
            "selected_unique_trajectories": int(sum(selected_by_variant.values())),
            "selected_weighted_trajectories": len(selected),
            "selected_transitions": selected_transitions,
            "selected_close_zone_transitions": selected_close_zone_transitions,
            "selected_oracle_close_zone_transitions": selected_oracle_close_zone_transitions,
            "selected_by_variant": dict(selected_by_variant),
            "weighted_by_variant": dict(weighted_by_variant),
            "excluded_trajectories": int(sum(excluded_reasons.values())),
            "excluded_reasons": dict(excluded_reasons),
            "step_q75": step_cutoff,
            "rows": selected_rows,
        },
    }


def behavior_clone_from_demonstrations(
    agent: Any,
    trajectories: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]],
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("bc epochs must be positive")
    if batch_size <= 0:
        raise ValueError("bc batch_size must be positive")

    pairs = [(state, action) for trajectory in trajectories for state, action, *_ in trajectory]
    if not pairs:
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "transitions": 0,
            "updates": 0,
            "final_loss": 0.0,
        }

    states_np = np.stack([state for state, _ in pairs]).astype(np.float32, copy=False)
    actions_np = np.array([action for _, action in pairs], dtype=np.int64)
    device = agent.device
    loss_fn = torch.nn.CrossEntropyLoss()
    updates = 0
    final_loss = 0.0
    was_training = agent.policy_net.training
    agent.policy_net.train()
    for _ in range(epochs):
        order = np.random.permutation(len(actions_np))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            states = torch.from_numpy(states_np[idx]).to(device)
            actions = torch.from_numpy(actions_np[idx]).to(device)
            if hasattr(agent.policy_net, "reset_noise"):
                agent.policy_net.reset_noise()
            logits = agent.policy_net(states)
            loss = loss_fn(logits, actions)
            agent.optimizer.zero_grad()
            loss.backward()
            if agent.config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    agent.policy_net.parameters(), agent.config.GRAD_CLIP
                )
            agent.optimizer.step()
            agent._optimizer_steps_since_scheduler += 1
            final_loss = float(loss.item())
            updates += 1
            with agent._losses_lock:
                agent.losses.append(final_loss)

    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    if not was_training:
        agent.policy_net.eval()
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "transitions": int(len(actions_np)),
        "updates": updates,
        "final_loss": final_loss,
    }
