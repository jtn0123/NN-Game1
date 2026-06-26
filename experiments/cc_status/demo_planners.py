# ruff: noqa: F401,F403,F405,I001
from .common import *
from .config_helpers import *


def collect_policy_demonstrations(
    agent: Any,
    config: Config,
    *,
    specs: tuple[Any, ...],
    k: int,
    max_steps: int | None = None,
    only_wins: bool = True,
) -> dict[str, Any]:
    """Collect greedy policy trajectories from a fixed level set.

    This is intentionally small "DQfD-lite" plumbing: no supervised margin loss,
    no permanent expert buffer, just successful source-skill trajectories seeded
    into normal replay before full-cave training.
    """
    if k <= 0:
        raise ValueError("demo k must be positive")
    game = CrystalCaves(config, headless=True)
    game._randomize_levels = False
    step_limit = max_steps or config.EVAL_MAX_STEPS
    trajectories: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]] = []
    rows: list[dict[str, Any]] = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    was_training = agent.policy_net.training
    agent.policy_net.eval()
    try:
        for index, spec in enumerate(specs):
            attempts = 0
            wins = 0
            kept = 0
            transitions = 0
            steps_list: list[int] = []
            end_reasons: Counter[str] = Counter()
            for _ in range(k):
                attempts += 1
                game.level_index = index
                state = game.reset()
                done = False
                info: dict[str, Any] = {}
                trajectory: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
                steps = 0
                while not done and steps < step_limit:
                    action = agent.select_action(state, training=False)
                    next_state, reward, done, info = game.step(action)
                    trajectory.append(
                        (state.copy(), action, float(reward), next_state.copy(), done)
                    )
                    state = next_state
                    steps += 1

                won = bool(info.get("won", False))
                wins += int(won)
                steps_list.append(steps)
                reason = str(info.get("end_reason", "") or "")
                if not reason or reason == "running":
                    reason = "won" if won else ("timeout" if steps >= step_limit else "ended")
                end_reasons[reason] += 1
                if trajectory and (won or not only_wins):
                    trajectories.append(trajectory)
                    kept += 1
                    transitions += len(trajectory)

            rows.append(
                {
                    "index": index,
                    "name": spec.name,
                    "attempts": attempts,
                    "wins": wins,
                    "win_rate": wins / attempts,
                    "kept_trajectories": kept,
                    "kept_transitions": transitions,
                    "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
                    "end_reason_counts": dict(end_reasons),
                }
            )
    finally:
        agent.epsilon = original_epsilon
        if was_training:
            agent.policy_net.train()

    return {
        "trajectories": trajectories,
        "summary": {
            "levels": len(specs),
            "attempts": int(sum(row["attempts"] for row in rows)),
            "wins": int(sum(row["wins"] for row in rows)),
            "kept_trajectories": len(trajectories),
            "kept_transitions": int(sum(len(traj) for traj in trajectories)),
            "only_wins": only_wins,
            "max_steps": step_limit,
            "rows": rows,
        },
    }


def seed_replay_from_demonstrations(
    agent: Any,
    trajectories: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]],
    *,
    repeat: int,
) -> dict[str, int]:
    if repeat <= 0:
        raise ValueError("demo repeat must be positive")
    pushed = 0
    for _ in range(repeat):
        for trajectory in trajectories:
            for state, action, reward, next_state, done in trajectory:
                agent.remember(state, action, reward, next_state, done)
                pushed += 1
    return {
        "trajectories": len(trajectories),
        "repeat": repeat,
        "pushed_transitions": pushed,
        "memory_size_after_seed": len(agent.memory),
    }


def dedupe_action_sequences(sequences: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for sequence in sequences:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sequence)
    return unique


def route_beam_candidate_sequences(game: CrystalCaves) -> list[list[int]]:
    target, _ = game._current_target()
    if target is None:
        return [[game.IDLE] * ROUTE_BEAM_COMMIT_STEPS]
    _, target_col, target_row = target
    player_col, player_row = game._player_tile()

    target_right = target_col >= player_col
    forward = game.RIGHT if target_right else game.LEFT
    backward = game.LEFT if target_right else game.RIGHT
    forward_jump = game.RIGHT_JUMP if target_right else game.LEFT_JUMP
    backward_jump = game.LEFT_JUMP if target_right else game.RIGHT_JUMP
    target_above = target_row < player_row

    forward_run = [forward] * 18
    forward_hop = [forward_jump] * 8 + [forward] * 14
    jump_forward = [game.JUMP] * 4 + [forward] * 18
    backoff_hop = [backward] * 6 + [forward_jump] * 10 + [forward] * 12
    reverse_hop = [backward_jump] * 8 + [forward] * 14
    vertical_jump = [game.JUMP] * 10 + [forward] * 10
    late_jump = [forward] * 8 + [forward_jump] * 8 + [forward] * 10
    brake = [game.IDLE] * 6 + [forward] * 12

    sequences = [
        forward_hop if target_above else forward_run,
        jump_forward if target_above else forward_hop,
        late_jump,
        backoff_hop,
        reverse_hop,
        vertical_jump,
        brake,
        [backward] * 10 + [forward] * 12,
        [game.IDLE] * 12,
    ]
    return dedupe_action_sequences(sequences)


def route_beam_plan(
    game: CrystalCaves,
    *,
    stale_steps: int,
    commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
) -> list[int]:
    """Choose a short macro by simulating route actions on copied game states."""

    target, initial_distance_px = game._current_target()
    if target is None:
        return [game.IDLE] * commit_steps
    if game._player_tile()[1] <= game.sky_rows:
        return [route_floor_scripted_action(game, variant="direct")] * commit_steps

    initial_distance = (
        float(initial_distance_px / game.TILE_SIZE)
        if initial_distance_px < float("inf")
        else 1000.0
    )
    initial_crystals = len(game.crystals)
    initial_progress = float(getattr(game, "_progress", 0.0) or 0.0)
    initial_health = int(getattr(game, "health", 0) or 0)

    best_score = -float("inf")
    best_sequence: list[int] | None = None
    for sequence in route_beam_candidate_sequences(game):
        probe = copy.deepcopy(game)
        distances = [initial_distance]
        total_reward = 0.0
        info: dict[str, Any] = {}
        done = False
        for action in sequence:
            _, reward, done, info = probe.step(action)
            total_reward += float(reward)
            _, distance_px = probe._current_target()
            if distance_px < float("inf"):
                distances.append(float(distance_px / probe.TILE_SIZE))
            if done:
                break

        min_distance = min(distances)
        final_distance = distances[-1]
        crystals_collected = max(0, initial_crystals - len(probe.crystals))
        progress_gain = float(getattr(probe, "_progress", 0.0) or 0.0) - initial_progress
        health_loss = max(0, initial_health - int(getattr(probe, "health", initial_health) or 0))
        best_delta = initial_distance - min_distance
        final_delta = initial_distance - final_distance
        won = bool(info.get("won", False))

        score = 0.0
        if won:
            score += 1000.0
        score += 120.0 * crystals_collected
        score += 80.0 * progress_gain
        score += 6.0 * best_delta
        score += 2.0 * final_delta
        score -= 1.5 * min_distance
        score -= 0.5 * final_distance
        score -= 25.0 * health_loss
        score += 0.02 * total_reward
        if stale_steps >= 36:
            score += 0.5 * best_delta

        if score > best_score:
            best_score = score
            best_sequence = sequence

    if not best_sequence:
        return [route_floor_scripted_action(game, variant="recovery", stale_steps=stale_steps)]
    return best_sequence[:commit_steps]


def close_zone_oracle_candidate_sequences(game: CrystalCaves) -> list[list[int]]:
    """Short local action macros for relabeling final-contact demo states."""

    target, _ = game._current_target()
    if target is None:
        return [[game.IDLE]]

    _, target_col, target_row = target
    player_col, player_row = game._player_tile()
    target_right = target_col >= player_col
    forward = game.RIGHT if target_right else game.LEFT
    backward = game.LEFT if target_right else game.RIGHT
    forward_jump = game.RIGHT_JUMP if target_right else game.LEFT_JUMP
    backward_jump = game.LEFT_JUMP if target_right else game.RIGHT_JUMP
    target_above = target_row < player_row

    sequences = [
        [forward] * 16,
        [backward] * 6 + [forward] * 14,
        [forward_jump] * 8 + [forward] * 14,
        [game.JUMP] * 5 + [forward] * 16,
        [backward] * 5 + [forward_jump] * 10 + [forward] * 14,
        [backward_jump] * 8 + [forward] * 14,
        [game.LEFT_JUMP] * 8 + [game.LEFT] * 12,
        [game.RIGHT_JUMP] * 8 + [game.RIGHT] * 12,
        [game.IDLE] * 10,
    ]
    if target_above:
        sequences.insert(0, [forward_jump] * 12 + [forward] * 12)
        sequences.insert(1, [game.JUMP] * 12 + [forward] * 12)

    return dedupe_action_sequences(sequences)


def close_zone_sequence_score(
    game: CrystalCaves,
    sequence: list[int],
    *,
    stale_steps: int = 0,
) -> dict[str, Any]:
    """Score one local close-zone macro on a copied game state."""

    target, initial_distance_px = game._current_target()
    if target is None or initial_distance_px == float("inf"):
        return {"reason": "no_target", "score": 0.0}
    if not sequence:
        return {"reason": "empty_sequence", "score": -float("inf")}

    initial_distance = float(initial_distance_px / game.TILE_SIZE)
    initial_crystals = len(game.crystals)
    initial_progress = float(getattr(game, "_progress", 0.0) or 0.0)
    initial_health = int(getattr(game, "health", 0) or 0)
    try:
        probe = copy.deepcopy(game)
    except Exception as exc:  # pragma: no cover - defensive; deepcopy is covered elsewhere
        return {"reason": f"copy_failed:{type(exc).__name__}", "score": 0.0}

    distances = [initial_distance]
    total_reward = 0.0
    info: dict[str, Any] = {}
    done = False
    for action in sequence:
        _, reward, done, info = probe.step(int(action))
        total_reward += float(reward)
        _, distance_px = probe._current_target()
        if distance_px < float("inf"):
            distances.append(float(distance_px / probe.TILE_SIZE))
        if done:
            break

    target_after, _ = probe._current_target()
    min_distance = min(distances)
    final_distance = distances[-1]
    crystals_collected = max(0, initial_crystals - len(probe.crystals))
    progress_gain = float(getattr(probe, "_progress", 0.0) or 0.0) - initial_progress
    health_loss = max(0, initial_health - int(getattr(probe, "health", initial_health) or 0))
    best_delta = initial_distance - min_distance
    final_delta = initial_distance - final_distance
    target_completed = bool(info.get("won", False)) or target_after != target
    first_action = int(sequence[0])

    score = 0.0
    if target_completed:
        score += 2500.0
    score += 1000.0 * crystals_collected
    score += 600.0 * max(0.0, progress_gain)
    score += 80.0 * best_delta
    score += 25.0 * final_delta
    score -= 8.0 * min_distance
    score -= 2.0 * final_distance
    score -= 120.0 * health_loss
    score += 0.03 * total_reward
    if game.ACTION_LABELS[first_action] in DEMO_JUMP_ACTIONS:
        score += 0.25
    if stale_steps >= 36:
        score += 10.0 * best_delta

    return {
        "reason": "ok",
        "score": float(score),
        "target_completed": bool(target_completed),
        "best_delta_tiles": float(best_delta),
        "final_delta_tiles": float(final_delta),
        "min_distance_tiles": float(min_distance),
        "final_distance_tiles": float(final_distance),
        "first_action": game.ACTION_LABELS[first_action],
        "crystals_collected": int(crystals_collected),
        "progress_gain": float(progress_gain),
        "health_loss": int(health_loss),
        "total_reward": float(total_reward),
    }


def close_zone_oracle_plan(
    game: CrystalCaves,
    *,
    stale_steps: int = 0,
    max_actions: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Choose a local close-zone macro by simulating short candidates on copied states."""

    target, initial_distance_px = game._current_target()
    if target is None or initial_distance_px == float("inf"):
        return [game.IDLE], {"reason": "no_target", "score": 0.0}

    best_score = -float("inf")
    best_sequence: list[int] = [game.IDLE]
    best_meta: dict[str, Any] = {"reason": "no_candidate", "score": best_score}
    for sequence in close_zone_oracle_candidate_sequences(game):
        if not sequence:
            continue
        meta = close_zone_sequence_score(
            game,
            [int(action) for action in sequence],
            stale_steps=stale_steps,
        )
        reason = str(meta.get("reason", "") or "")
        if reason.startswith("copy_failed"):
            return [game.IDLE], meta
        score = float(meta.get("score", -float("inf")) or -float("inf"))

        if score > best_score:
            best_score = score
            best_sequence = [int(action) for action in sequence]
            best_meta = meta

    if max_actions is not None:
        best_sequence = best_sequence[: max(1, int(max_actions))]
    return best_sequence, best_meta


def close_zone_oracle_action(
    game: CrystalCaves,
    *,
    stale_steps: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Choose a local close-zone label by simulating short macros on copied states."""

    plan, meta = close_zone_oracle_plan(game, stale_steps=stale_steps, max_actions=1)
    return int(plan[0] if plan else game.IDLE), meta


def route_floor_scripted_action(
    game: CrystalCaves,
    *,
    variant: str = "direct",
    stale_steps: int = 0,
) -> int:
    """Small controller for collecting near-entrance route-floor crystals."""

    target, _ = game._current_target()
    if target is None:
        return game.IDLE
    _, target_col, target_row = target
    player_col, player_row = game._player_tile()

    if variant not in {"direct", "recovery", "sweep"}:
        raise ValueError(f"unknown route demo controller variant: {variant}")

    if player_row <= game.sky_rows:
        surface = game.sky_rows
        shaft_cols = [col for col, ch in enumerate(game.grid[surface]) if ch != game.SOLID]
        shaft_col = (
            min(shaft_cols, key=lambda col: abs(col - player_col)) if shaft_cols else player_col
        )
        if shaft_col > player_col:
            return game.RIGHT
        if shaft_col < player_col:
            return game.LEFT
        return game.IDLE

    target_is_above = target_row < player_row
    if target_is_above and abs(target_col - player_col) <= 1:
        if target_col > player_col:
            return game.RIGHT_JUMP
        if target_col < player_col:
            return game.LEFT_JUMP
        return game.JUMP
    if variant == "sweep":
        close_enough = abs(target_col - player_col) <= 4 and abs(target_row - player_row) <= 5
        if close_enough:
            if target_col > player_col:
                return game.RIGHT
            if target_col < player_col:
                return game.LEFT
            phase = (stale_steps // 18) % 3
            if phase == 0:
                return game.LEFT
            if phase == 1:
                return game.RIGHT
            return game.IDLE
    if variant == "recovery" and stale_steps >= 36:
        close_enough = abs(target_col - player_col) <= 4 and abs(target_row - player_row) <= 5
        if close_enough:
            phase = (stale_steps // 18) % 6
            if target_is_above:
                if phase in {0, 3}:
                    return game.LEFT_JUMP
                if phase in {1, 4}:
                    return game.RIGHT_JUMP
                return game.JUMP
            if abs(target_col - player_col) <= 1:
                return game.LEFT if phase % 2 == 0 else game.RIGHT
        if stale_steps >= 72:
            phase = (stale_steps // 24) % 5
            target_right = target_col >= player_col
            if phase == 0:
                return game.RIGHT_JUMP if target_right else game.LEFT_JUMP
            if phase == 1:
                return game.JUMP
            if phase == 2:
                return game.LEFT_JUMP if target_right else game.RIGHT_JUMP
            if phase == 3:
                return game.LEFT if target_right else game.RIGHT
    if target_col > player_col:
        return game.RIGHT_JUMP if target_is_above else game.RIGHT
    if target_col < player_col:
        return game.LEFT_JUMP if target_is_above else game.LEFT
    return game.IDLE


def classify_demo_attempt(row: dict[str, Any]) -> list[str]:
    if row.get("won"):
        return ["won"]

    modes: list[str] = []
    min_distance = row.get("min_target_distance_tiles")
    final_distance = row.get("final_target_distance_tiles")
    best_delta = float(row.get("target_distance_best_delta_tiles", 0.0) or 0.0)
    action_counts = row.get("action_counts") or {}
    steps = max(1, int(row.get("steps", 0) or 0))
    idle_frac = float(action_counts.get("IDLE", 0) / steps)
    right_frac = float(action_counts.get("RIGHT", 0) / steps)
    left_frac = float(action_counts.get("LEFT", 0) / steps)

    if min_distance is None:
        modes.append("no_target")
    elif float(min_distance) <= CLOSE_ZONE_DISTANCE_TILES:
        modes.append("reached_close_zone")
    elif float(min_distance) <= 5.0:
        modes.append("near_miss")
    elif best_delta <= 0.25:
        modes.append("no_approach")
    else:
        modes.append("partial_approach")

    if final_distance is not None and min_distance is not None:
        drift = float(final_distance) - float(min_distance)
        if drift >= 3.0:
            modes.append("drifted_after_best")

    if idle_frac >= 0.35:
        modes.append("idle_heavy")
    if right_frac >= 0.70:
        modes.append("right_wall_push")
    if left_frac >= 0.70:
        modes.append("left_wall_push")
    if row.get("end_reason") == "stalled":
        modes.append("stalled")
    elif row.get("end_reason") == "timeout":
        modes.append("timeout")

    return modes
