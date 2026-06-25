# ruff: noqa: F401,F403,F405,I001
"""Policy-visited correction dataset collection for Crystal Caves."""

from .common import *
from .config_helpers import *
from .demo_planners import *
from .evals import *
from .io_utils import *
from .reports import *
from .runs_transfer import config_from_selected_checkpoint
from .training import *

CORRECTION_DATASET_VERSION = "cc_policy_corrections_v1"
CORRECTION_REASON_BITS = {
    "close_zone": 1,
    "stale": 2,
    "loop": 4,
}


def correction_trigger_reasons(
    *,
    target_distance_tiles: float | None,
    steps_since_progress: int,
    tile_visits: int,
    close_zone_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    stale_steps: int = 90,
    loop_tile_visits: int = 8,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if target_distance_tiles is not None and target_distance_tiles <= close_zone_distance_tiles:
        reasons.append("close_zone")
    if steps_since_progress >= stale_steps:
        reasons.append("stale")
    if tile_visits >= loop_tile_visits:
        reasons.append("loop")
    return tuple(reasons)


def correction_reason_mask(reasons: tuple[str, ...]) -> int:
    mask = 0
    for reason in reasons:
        mask |= CORRECTION_REASON_BITS.get(reason, 0)
    return mask


def choose_correction_action(
    game: CrystalCaves,
    *,
    reasons: tuple[str, ...],
    stale_steps: int,
) -> tuple[int, dict[str, Any]]:
    if "close_zone" in reasons:
        action, meta = close_zone_oracle_action(game, stale_steps=stale_steps)
        return action, {"label_source": "close_zone_oracle", **meta}
    action = route_floor_scripted_action(game, variant="recovery", stale_steps=stale_steps)
    return action, {
        "label_source": "route_recovery",
        "reason": "stale_or_loop",
        "score": 0.0,
    }


def collect_policy_correction_dataset(
    config: Config,
    agent: Any,
    *,
    out_dir: Path,
    label: str,
    games: int,
    max_steps: int,
    max_examples: int,
    sample_every: int = 4,
    max_examples_per_game: int = 64,
    close_zone_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    stale_steps: int = 90,
    loop_tile_visits: int = 8,
    only_disagreements: bool = True,
) -> dict[str, Any]:
    if games <= 0:
        raise ValueError("correction games must be positive")
    if max_steps <= 0:
        raise ValueError("correction max_steps must be positive")
    if max_examples < 0:
        raise ValueError("correction max_examples must be non-negative")
    if sample_every <= 0:
        raise ValueError("correction sample_every must be positive")
    if max_examples_per_game <= 0:
        raise ValueError("correction max_examples_per_game must be positive")
    if close_zone_distance_tiles <= 0:
        raise ValueError("close_zone_distance_tiles must be positive")
    if stale_steps < 0:
        raise ValueError("correction stale_steps must be non-negative")
    if loop_tile_visits <= 0:
        raise ValueError("correction loop_tile_visits must be positive")

    correction_dir = out_dir / "corrections" / label
    correction_dir.mkdir(parents=True, exist_ok=True)
    rows_path = correction_dir / "correction_examples.jsonl"
    summary_path = correction_dir / "summary.json"
    states_path = correction_dir / "correction_examples.npz"
    rows_path.unlink(missing_ok=True)
    rows_path.touch()

    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    action_labels = list(getattr(game, "ACTION_LABELS", [])) or [
        str(index) for index in range(game.action_size)
    ]

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    policy_net = getattr(agent, "policy_net", None)
    was_training = bool(getattr(policy_net, "training", False))
    if policy_net is not None and hasattr(policy_net, "eval"):
        policy_net.eval()

    states: list[np.ndarray] = []
    label_actions: list[int] = []
    policy_actions: list[int] = []
    reason_masks: list[int] = []
    rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    trigger_counts: Counter[str] = Counter()
    candidate_trigger_counts: Counter[str] = Counter()
    label_action_counts: Counter[str] = Counter()
    policy_action_counts: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()
    end_reasons: Counter[str] = Counter()
    candidates = 0
    agreements = 0
    wins = 0
    any_crystal_games = 0

    try:
        for game_index in range(games):
            state = game.reset()
            initial_crystals = int(game.initial_crystals)
            tile_counts: Counter[tuple[int, int]] = Counter()
            examples_this_game = 0
            done = False
            info = game._info()
            steps = 0

            for step in range(max_steps):
                tile = game._player_tile()
                tile_counts[tile] += 1
                objective = objective_snapshot(game)
                target_distance = objective.get("target_distance_tiles")
                distance = float(target_distance) if target_distance is not None else None
                steps_since_progress = int(info.get("steps_since_progress", 0) or 0)
                reasons = correction_trigger_reasons(
                    target_distance_tiles=distance,
                    steps_since_progress=steps_since_progress,
                    tile_visits=tile_counts[tile],
                    close_zone_distance_tiles=close_zone_distance_tiles,
                    stale_steps=stale_steps,
                    loop_tile_visits=loop_tile_visits,
                )

                policy_action = int(agent.select_action(state, training=False))
                should_sample = bool(reasons) and (
                    "close_zone" in reasons or step % sample_every == 0
                )
                if (
                    should_sample
                    and len(states) < max_examples
                    and examples_this_game < max_examples_per_game
                ):
                    candidates += 1
                    for reason in reasons:
                        candidate_trigger_counts[reason] += 1
                    label_action, label_meta = choose_correction_action(
                        game,
                        reasons=reasons,
                        stale_steps=steps_since_progress,
                    )
                    label_action = int(label_action)
                    disagreed = label_action != policy_action
                    agreements += int(not disagreed)
                    if disagreed or not only_disagreements:
                        q_values = (
                            agent.get_q_values(state) if hasattr(agent, "get_q_values") else None
                        )
                        policy_label = _action_label(action_labels, policy_action)
                        correction_label = _action_label(action_labels, label_action)
                        row = {
                            "dataset_version": CORRECTION_DATASET_VERSION,
                            "game_index": int(game_index),
                            "step": int(step),
                            "level": info.get("level"),
                            "level_name": info.get("level_name"),
                            "trigger_reasons": list(reasons),
                            "trigger_mask": correction_reason_mask(reasons),
                            "policy_action": policy_action,
                            "policy_action_label": policy_label,
                            "label_action": label_action,
                            "label_action_label": correction_label,
                            "policy_label_disagreement": bool(disagreed),
                            "label_source": str(label_meta.get("label_source", "unknown")),
                            "label_meta": _json_safe_label_meta(label_meta),
                            "objective": objective,
                            "steps_since_progress": steps_since_progress,
                            "tile": [int(tile[0]), int(tile[1])],
                            "tile_visits": int(tile_counts[tile]),
                            "crystals_remaining": int(info.get("crystals_remaining", 0) or 0),
                            "exit_unlocked": bool(info.get("exit_unlocked", False)),
                            "progress": float(info.get("progress", 0.0) or 0.0),
                        }
                        if q_values is not None:
                            row["q"] = q_value_snapshot(q_values, action_labels)
                        states.append(state.copy())
                        label_actions.append(label_action)
                        policy_actions.append(policy_action)
                        reason_masks.append(correction_reason_mask(reasons))
                        rows.append(row)
                        append_jsonl(rows_path, row)
                        for reason in reasons:
                            trigger_counts[reason] += 1
                        label_action_counts[correction_label] += 1
                        policy_action_counts[policy_label] += 1
                        label_source_counts[str(label_meta.get("label_source", "unknown"))] += 1
                        examples_this_game += 1

                state, _, done, info = game.step(policy_action)
                steps = step + 1
                if done:
                    break

            reason = str(info.get("end_reason", "") or "")
            if not reason or reason == "running":
                reason = (
                    "won"
                    if info.get("won", False)
                    else ("timeout" if steps >= max_steps else "ended")
                )
            end_reasons[reason] += 1
            won = bool(info.get("won", False))
            wins += int(won)
            collected = initial_crystals - int(info.get("crystals_remaining", 0) or 0)
            any_crystal_games += int(collected > 0)
            game_rows.append(
                {
                    "game_index": int(game_index),
                    "steps": int(steps),
                    "won": won,
                    "end_reason": reason,
                    "crystals_collected": int(collected),
                    "examples": int(examples_this_game),
                    "unique_tiles": int(len(tile_counts)),
                }
            )
            if len(states) >= max_examples:
                break
    finally:
        agent.epsilon = original_epsilon
        if was_training and policy_net is not None and hasattr(policy_net, "train"):
            policy_net.train()
        game.close()

    states_array = (
        np.stack(states).astype(np.float32, copy=False)
        if states
        else np.empty((0, game.state_size), dtype=np.float32)
    )
    labels_array = np.asarray(label_actions, dtype=np.int64)
    policy_actions_array = np.asarray(policy_actions, dtype=np.int64)
    reason_masks_array = np.asarray(reason_masks, dtype=np.int64)
    np.savez_compressed(
        states_path,
        states=states_array,
        actions=labels_array,
        policy_actions=policy_actions_array,
        trigger_masks=reason_masks_array,
    )

    kept_examples = int(len(label_actions))
    summary = {
        "dataset_version": CORRECTION_DATASET_VERSION,
        "label": label,
        "games_requested": int(games),
        "games_completed": int(len(game_rows)),
        "max_steps": int(max_steps),
        "max_examples": int(max_examples),
        "max_examples_per_game": int(max_examples_per_game),
        "sample_every": int(sample_every),
        "close_zone_distance_tiles": float(close_zone_distance_tiles),
        "stale_steps": int(stale_steps),
        "loop_tile_visits": int(loop_tile_visits),
        "only_disagreements": bool(only_disagreements),
        "candidate_states": int(candidates),
        "agreement_states": int(agreements),
        "kept_examples": kept_examples,
        "disagreement_rate": (float((candidates - agreements) / candidates) if candidates else 0.0),
        "states_shape": list(states_array.shape),
        "actions_shape": list(labels_array.shape),
        "states_path": str(states_path),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "trigger_counts": dict(trigger_counts),
        "candidate_trigger_counts": dict(candidate_trigger_counts),
        "label_action_counts": dict(label_action_counts),
        "policy_action_counts": dict(policy_action_counts),
        "label_source_counts": dict(label_source_counts),
        "end_reason_counts": dict(end_reasons),
        "wins": int(wins),
        "win_rate": float(wins / len(game_rows)) if game_rows else 0.0,
        "any_crystal_rate": float(any_crystal_games / len(game_rows)) if game_rows else 0.0,
        "game_rows": game_rows,
        "preview_rows": rows[:20],
    }
    write_json(summary_path, summary)
    return summary


def load_correction_action_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    states = np.asarray(payload["states"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.int64)
    if states.ndim != 2:
        raise ValueError("correction states must be a 2D array")
    if actions.ndim != 1 or len(actions) != len(states):
        raise ValueError("correction actions must be a 1D array matching states")
    return states, actions


def run_collect_corrections(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    seed: int,
    correction_games: int,
    correction_max_steps: int,
    correction_max_examples: int,
    correction_sample_every: int,
    correction_max_examples_per_game: int,
    correction_stale_steps: int,
    correction_loop_tile_visits: int,
    correction_keep_agreements: bool,
    log_every: int,
    report_seconds: float,
    label: str = "collect_corrections",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    snapshot = load_selected_weight_snapshot(checkpoint_path)
    config = config_from_selected_checkpoint(
        run_dir,
        snapshot=snapshot,
        seed=seed,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    trainer = prepare_trainer(
        config,
        episodes=1,
        vec_envs=1,
        save_checkpoints=False,
    )
    selected_episode = int(snapshot.get("episode", 0) or 0)
    trainer.current_episode = selected_episode
    _validate_checkpoint_shape(trainer, snapshot, checkpoint_path)
    load_weight_snapshot(trainer.agent, snapshot["weights"])

    correction_dataset = collect_policy_correction_dataset(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        games=correction_games,
        max_steps=correction_max_steps,
        max_examples=correction_max_examples,
        sample_every=correction_sample_every,
        max_examples_per_game=correction_max_examples_per_game,
        stale_steps=correction_stale_steps,
        loop_tile_visits=correction_loop_tile_visits,
        only_disagreements=not correction_keep_agreements,
    )
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=0.0,
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_eval": {
                "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
                "source_label": snapshot.get("label", ""),
                "source_episode": selected_episode,
                "source_eval": snapshot.get("source_eval") or {},
            },
            "correction_dataset": correction_dataset,
        },
    )


def run_correction_finetune(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    correction_dataset_path: Path,
    episodes: int,
    seed: int,
    eval_games: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    save_checkpoints: bool,
    correction_action_weight: float,
    correction_action_margin: float,
    correction_action_batch_size: int,
    label: str = "correction_finetune",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    snapshot = load_selected_weight_snapshot(checkpoint_path)
    config = config_from_selected_checkpoint(
        run_dir,
        snapshot=snapshot,
        seed=seed,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    config.MAX_EPISODES = episodes
    config.EVAL_EVERY = eval_every
    config.EVAL_EPISODES = train_eval_games
    apply_correction_action_override(
        config,
        correction_action_weight=correction_action_weight,
        correction_action_margin=correction_action_margin,
        correction_action_batch_size=correction_action_batch_size,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=snapshot["weights"],
        save_checkpoints=save_checkpoints,
    )
    selected_episode = int(snapshot.get("episode", 0) or 0)
    _validate_checkpoint_shape(trainer, snapshot, checkpoint_path)
    correction_states, correction_actions = load_correction_action_dataset(correction_dataset_path)
    if correction_states.shape[0] <= 0:
        raise ValueError("correction-finetune requires at least one correction transition")
    correction_dataset_summary = trainer.agent.set_correction_action_dataset(
        correction_states,
        correction_actions,
    )

    train_seconds = run_training(
        trainer,
        run_dir=run_dir,
        label=label,
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
    )
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_final",
        episode=trainer.current_episode,
        games=eval_games,
    )
    near_miss_eval = first_objective_near_miss_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_final",
        episode=trainer.current_episode,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
    )
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_eval": {
                "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
                "source_label": snapshot.get("label", ""),
                "source_episode": selected_episode,
                "source_eval": snapshot.get("source_eval") or {},
            },
            "correction_training": {
                "dataset_path": str(correction_dataset_path),
                "dataset_states": int(correction_states.shape[0]),
                "state_size": int(correction_states.shape[1]) if correction_states.ndim == 2 else 0,
                "weight": float(correction_action_weight),
                "margin": float(correction_action_margin),
                "batch_size": int(correction_action_batch_size),
                **correction_dataset_summary,
            },
            "near_miss_eval": near_miss_eval,
        },
    )


def _validate_checkpoint_shape(
    trainer: HeadlessTrainer,
    snapshot: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    saved_state_size = int(snapshot.get("state_size", trainer.agent.state_size) or 0)
    saved_action_size = int(snapshot.get("action_size", trainer.agent.action_size) or 0)
    if saved_state_size and saved_state_size != trainer.agent.state_size:
        raise ValueError(
            f"{checkpoint_path} state size {saved_state_size} does not match "
            f"environment state size {trainer.agent.state_size}"
        )
    if saved_action_size and saved_action_size != trainer.agent.action_size:
        raise ValueError(
            f"{checkpoint_path} action size {saved_action_size} does not match "
            f"environment action size {trainer.agent.action_size}"
        )


def _action_label(action_labels: list[str], action: int) -> str:
    return action_labels[action] if 0 <= action < len(action_labels) else str(action)


def _json_safe_label_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, float):
            safe[key] = float(value)
        else:
            safe[key] = str(value)
    return safe
