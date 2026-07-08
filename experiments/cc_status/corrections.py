# ruff: noqa: F401,F403,F405,I001
"""Policy-visited correction dataset collection for Crystal Caves."""

from .common import *
from .config_helpers import *
from .contact_head import *
from .demo_planners import *
from .evals import *
from .io_utils import *
from .policy_anchor import install_policy_anchor_provider
from .reports import *
from .runs_transfer import config_from_selected_checkpoint, final_contact_option_action
from .training import *

CORRECTION_DATASET_VERSION = "cc_policy_corrections_v1"
CORRECTION_REASON_BITS = {
    "close_zone": 1,
    "stale": 2,
    "loop": 4,
}
CORRECTION_LABEL_MODES = {"standard", "advantage_gate"}


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


def choose_advantage_gate_correction_action(
    agent: Any,
    state: np.ndarray,
    game: CrystalCaves,
    info: dict[str, Any],
    *,
    action_labels: list[str],
    close_zone_distance_tiles: float,
    final_contact_commit_steps: int,
    min_option_advantage: float,
) -> tuple[int | None, dict[str, Any]]:
    action, decision = final_contact_option_action(
        agent,
        state,
        game,
        info,
        action_labels=action_labels,
        close_zone_distance_tiles=close_zone_distance_tiles,
        option_queue=None,
        final_contact_commit_steps=final_contact_commit_steps,
        cancel_option_outside_close_zone=False,
        gate_policy_advantage=True,
        min_option_advantage=min_option_advantage,
    )
    option_meta = decision.get("option_meta") or {}
    if not isinstance(option_meta, dict):
        option_meta = {}
    accepted = str(decision.get("source", "") or "") == "final_contact_option" and not bool(
        option_meta.get("rejected_by_policy_gate", False)
    )
    meta = {
        "label_source": "advantage_gate_final_contact",
        "gate_accepted": bool(accepted),
        "decision_source": str(decision.get("source", "") or ""),
        **option_meta,
    }
    if not accepted:
        return None, meta
    return int(action), meta


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
    label_mode: str = "standard",
    final_contact_commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
    min_option_advantage: float = 0.0,
    rollout_action_selector: Any | None = None,
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
    if label_mode not in CORRECTION_LABEL_MODES:
        raise ValueError(f"unknown correction label mode: {label_mode}")
    if final_contact_commit_steps <= 0:
        raise ValueError("final-contact commit steps must be positive")
    if min_option_advantage < 0:
        raise ValueError("minimum option advantage must be non-negative")

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
    rejected_label_candidates = 0
    agreements = 0
    gate_evaluations = 0
    gate_rejections = 0
    gate_advantage_total = 0.0
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

                policy_action, policy_action_meta = _select_correction_rollout_action(
                    agent,
                    state,
                    game,
                    info,
                    step,
                    action_labels,
                    rollout_action_selector=rollout_action_selector,
                )
                should_sample = bool(reasons) and (
                    "close_zone" in reasons or step % sample_every == 0
                )
                if label_mode == "advantage_gate":
                    should_sample = "close_zone" in reasons
                if (
                    should_sample
                    and len(states) < max_examples
                    and examples_this_game < max_examples_per_game
                ):
                    candidates += 1
                    for reason in reasons:
                        candidate_trigger_counts[reason] += 1
                    if label_mode == "advantage_gate":
                        label_action, label_meta = choose_advantage_gate_correction_action(
                            agent,
                            state,
                            game,
                            info,
                            action_labels=action_labels,
                            close_zone_distance_tiles=close_zone_distance_tiles,
                            final_contact_commit_steps=final_contact_commit_steps,
                            min_option_advantage=min_option_advantage,
                        )
                        if label_meta.get("gate_policy_advantage"):
                            gate_evaluations += 1
                            option_advantage = float(label_meta.get("option_advantage", 0.0) or 0.0)
                            gate_advantage_total += option_advantage
                            if not label_meta.get("gate_accepted", False):
                                gate_rejections += 1
                        if label_action is None:
                            rejected_label_candidates += 1
                            state, _, done, info = game.step(policy_action)
                            steps = step + 1
                            if done:
                                break
                            continue
                    else:
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
                            "policy_action_source": str(policy_action_meta.get("source", "policy")),
                            "policy_action_meta": _json_safe_label_meta(policy_action_meta),
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
        "label_mode": label_mode,
        "final_contact_commit_steps": int(final_contact_commit_steps),
        "min_option_advantage": float(min_option_advantage),
        "rollout_action_selector": bool(rollout_action_selector is not None),
        "candidate_states": int(candidates),
        "label_candidate_states": int(candidates - rejected_label_candidates),
        "rejected_label_candidates": int(rejected_label_candidates),
        "agreement_states": int(agreements),
        "kept_examples": kept_examples,
        "disagreement_rate": (
            float(
                (candidates - rejected_label_candidates - agreements)
                / (candidates - rejected_label_candidates)
            )
            if candidates > rejected_label_candidates
            else 0.0
        ),
        "gate_evaluations": int(gate_evaluations),
        "gate_rejections": int(gate_rejections),
        "gate_rejection_rate": float(gate_rejections / max(1, gate_evaluations)),
        "mean_gate_option_advantage": float(gate_advantage_total / max(1, gate_evaluations)),
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


def _select_correction_rollout_action(
    agent: Any,
    state: np.ndarray,
    game: CrystalCaves,
    info: dict[str, Any],
    step: int,
    action_labels: list[str],
    *,
    rollout_action_selector: Any | None,
) -> tuple[int, dict[str, Any]]:
    if rollout_action_selector is None:
        return int(agent.select_action(state, training=False)), {"source": "policy"}
    result = rollout_action_selector(agent, state, game, info, step, action_labels)
    if isinstance(result, tuple) and len(result) == 2:
        action, meta = result
        if not isinstance(meta, dict):
            meta = {"source": str(meta)}
        return int(action), dict(meta)
    return int(result), {"source": "custom"}


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
    correction_label_mode: str = "standard",
    final_contact_distance: float = CLOSE_ZONE_DISTANCE_TILES,
    final_contact_commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
    final_contact_min_option_advantage: float = 0.0,
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
        close_zone_distance_tiles=final_contact_distance,
        stale_steps=correction_stale_steps,
        loop_tile_visits=correction_loop_tile_visits,
        only_disagreements=not correction_keep_agreements,
        label_mode=correction_label_mode,
        final_contact_commit_steps=final_contact_commit_steps,
        min_option_advantage=final_contact_min_option_advantage,
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


def run_collect_contact_head_corrections(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    correction_dataset_path: Path,
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
    correction_label_mode: str = "advantage_gate",
    final_contact_distance: float = CLOSE_ZONE_DISTANCE_TILES,
    final_contact_commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
    final_contact_min_option_advantage: float = 0.0,
    contact_action_batch_size: int = 32,
    contact_action_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    contact_head_offline_steps: int = 500,
    contact_head_learning_rate: float = 0.001,
    contact_head_confidence: float = 0.75,
    contact_head_jump_confidence: float = 0.0,
    contact_head_action_thresholds: dict[str, float] | None = None,
    contact_head_balance_classes: bool = True,
    label: str = "collect_contact_head_corrections",
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
    apply_contact_action_head_override(
        config,
        contact_action_weight=0.0,
        contact_action_batch_size=contact_action_batch_size,
        contact_action_distance_tiles=contact_action_distance_tiles,
    )
    trainer = prepare_trainer(
        config,
        episodes=1,
        vec_envs=1,
        transfer_weights=snapshot["weights"],
        strict_transfer=False,
        save_checkpoints=False,
    )
    trainer.agent.epsilon = 0.0
    selected_episode = int(snapshot.get("episode", 0) or 0)
    trainer.current_episode = selected_episode
    _validate_checkpoint_shape(trainer, snapshot, checkpoint_path)

    contact_states, contact_actions = load_correction_action_dataset(correction_dataset_path)
    if contact_states.shape[0] <= 0:
        raise ValueError("contact-head correction collection requires contact labels")
    contact_dataset_summary = trainer.agent.set_contact_action_dataset(
        contact_states,
        contact_actions,
    )
    game = CrystalCaves(config, headless=True)
    try:
        action_labels = list(getattr(game, "ACTION_LABELS", [])) or [
            str(index) for index in range(trainer.agent.action_size)
        ]
    finally:
        game.close()
    action_confidence_thresholds = validate_contact_head_action_thresholds(
        contact_head_action_thresholds,
        action_labels,
    )
    dataset_stats = contact_action_dataset_stats(contact_actions, action_labels=action_labels)
    started = time.time()
    offline_training = train_contact_action_head_offline(
        trainer.agent,
        contact_states,
        contact_actions,
        steps=contact_head_offline_steps,
        batch_size=contact_action_batch_size,
        learning_rate=contact_head_learning_rate,
        balance_classes=contact_head_balance_classes,
        action_labels=action_labels,
    )

    selector_stats = new_contact_action_head_stats()

    def select_contact_head_rollout_action(
        agent: Any,
        state: np.ndarray,
        game: CrystalCaves,
        info: dict[str, Any],
        step: int,
        action_labels: list[str],
    ) -> tuple[int, dict[str, Any]]:
        del info, step
        action, decision = contact_action_head_action(
            agent,
            state,
            game,
            action_labels=action_labels,
            close_zone_distance_tiles=contact_action_distance_tiles,
            min_confidence=contact_head_confidence,
            jump_min_confidence=contact_head_jump_confidence,
            action_min_confidences=action_confidence_thresholds,
        )
        record_contact_action_head_decision(selector_stats, decision, action, action_labels)
        return action, decision

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
        close_zone_distance_tiles=final_contact_distance,
        stale_steps=correction_stale_steps,
        loop_tile_visits=correction_loop_tile_visits,
        only_disagreements=not correction_keep_agreements,
        label_mode=correction_label_mode,
        final_contact_commit_steps=final_contact_commit_steps,
        min_option_advantage=final_contact_min_option_advantage,
        rollout_action_selector=select_contact_head_rollout_action,
    )
    selector_payload = contact_action_head_stats_payload(selector_stats)
    selector_payload["min_confidence"] = float(contact_head_confidence)
    selector_payload["jump_min_confidence"] = float(contact_head_jump_confidence)
    selector_payload["action_min_confidences"] = dict(action_confidence_thresholds)
    correction_dataset["rollout_contact_action_head"] = selector_payload
    if correction_dataset.get("summary_path"):
        write_json(Path(str(correction_dataset["summary_path"])), correction_dataset)
    train_seconds = time.time() - started
    collection_live = live_snapshot(
        trainer,
        label=label,
        status="complete",
        started=started,
        total_episodes=contact_head_offline_steps,
    )
    collection_live["offline_steps"] = int(contact_head_offline_steps)
    collection_live["offline_accuracy"] = float(offline_training.get("accuracy", 0.0) or 0.0)
    collection_live["correction_games"] = int(correction_games)
    collection_live["kept_examples"] = int(correction_dataset.get("kept_examples", 0) or 0)
    collection_live["selector_head_action_rate"] = float(
        selector_payload.get("head_action_rate", 0.0) or 0.0
    )
    write_json(run_dir / "live_metrics.json", collection_live)
    append_jsonl(run_dir / "live_metrics.jsonl", collection_live)

    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_eval": {
                "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
                "source_label": snapshot.get("label", ""),
                "source_episode": selected_episode,
                "source_eval": snapshot.get("source_eval") or {},
            },
            "contact_action_head_training": {
                "mode": "offline_head_only",
                "dataset_path": str(correction_dataset_path),
                "dataset_states": int(contact_states.shape[0]),
                "state_size": int(contact_states.shape[1]) if contact_states.ndim == 2 else 0,
                "weight": 0.0,
                "batch_size": int(contact_action_batch_size),
                "distance_tiles": float(contact_action_distance_tiles),
                "learning_rate": float(contact_head_learning_rate),
                "offline_steps": int(contact_head_offline_steps),
                "confidence_threshold": float(contact_head_confidence),
                "jump_confidence_threshold": float(contact_head_jump_confidence),
                "action_confidence_thresholds": dict(action_confidence_thresholds),
                "balance_classes": bool(contact_head_balance_classes),
                "dataset_stats": dataset_stats,
                "offline_training": offline_training,
                **contact_dataset_summary,
            },
            "contact_action_head_rollout": selector_payload,
            "contact_action_head_eval": selector_payload,
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
    policy_anchor_weight: float = 0.0,
    policy_anchor_temperature: float = 1.0,
    policy_anchor_min_distance_tiles: float = 0.0,
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
    apply_policy_anchor_override(
        config,
        policy_anchor_weight=policy_anchor_weight,
        policy_anchor_temperature=policy_anchor_temperature,
        policy_anchor_min_distance_tiles=policy_anchor_min_distance_tiles,
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
    policy_anchor_summary = install_policy_anchor_provider(
        trainer.agent,
        weight=policy_anchor_weight,
        temperature=policy_anchor_temperature,
        min_target_distance_norm=getattr(
            config,
            "CRYSTAL_CAVES_POLICY_ANCHOR_MIN_TARGET_DISTANCE_NORM",
            0.0,
        ),
    )
    policy_anchor_summary["min_target_distance_tiles"] = float(policy_anchor_min_distance_tiles)

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
            "policy_anchor_training": policy_anchor_summary,
            "near_miss_eval": near_miss_eval,
        },
    )


def _near_miss_rollup_as_eval_payload(
    near_miss_eval: dict[str, Any] | None,
    *,
    label: str,
    episode: int,
) -> dict[str, Any]:
    near_miss_eval = near_miss_eval or {}
    rollup = near_miss_eval.get("rollup") or {}
    rows = near_miss_eval.get("rows") or []
    row_scores = [float(row.get("score", 0.0) or 0.0) for row in rows if isinstance(row, dict)]
    row_steps = [float(row.get("steps", 0.0) or 0.0) for row in rows if isinstance(row, dict)]
    games = int(rollup.get("games", 0) or 0)
    wins = int(rollup.get("wins", 0) or 0)
    return {
        "label": label,
        "episode": int(episode),
        "num_games": games,
        "wins": wins,
        "win_rate": float(rollup.get("win_rate", 0.0) or 0.0),
        # Audit R2-B: the TRUE mean collection fraction, not any_crystal_rate (collected-≥1),
        # which over-reported multi-crystal collection into promotion/scorecard/reports.
        "mean_crystal_frac": float(rollup.get("mean_crystal_frac", 0.0) or 0.0),
        "mean_depth_frac": float(rollup.get("mean_depth_frac", 0.0) or 0.0),
        "mean_score": float(np.mean(row_scores)) if row_scores else 0.0,
        "median_score": float(np.median(row_scores)) if row_scores else 0.0,
        "min_score": float(min(row_scores)) if row_scores else 0.0,
        "max_score": float(max(row_scores)) if row_scores else 0.0,
        "mean_steps": float(np.mean(row_steps)) if row_steps else 0.0,
        "max_steps": int(near_miss_eval.get("max_steps", 0) or 0),
        "end_reason_counts": dict(rollup.get("end_reason_counts", {}) or {}),
    }


def run_contact_head_finetune(
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
    contact_action_weight: float,
    contact_action_batch_size: int,
    contact_action_distance_tiles: float,
    label: str = "contact_head_finetune",
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
    apply_contact_action_head_override(
        config,
        contact_action_weight=contact_action_weight,
        contact_action_batch_size=contact_action_batch_size,
        contact_action_distance_tiles=contact_action_distance_tiles,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=snapshot["weights"],
        strict_transfer=False,
        save_checkpoints=save_checkpoints,
    )
    selected_episode = int(snapshot.get("episode", 0) or 0)
    _validate_checkpoint_shape(trainer, snapshot, checkpoint_path)
    contact_states, contact_actions = load_correction_action_dataset(correction_dataset_path)
    if contact_states.shape[0] <= 0:
        raise ValueError("contact-head fine-tune requires at least one contact transition")
    contact_dataset_summary = trainer.agent.set_contact_action_dataset(
        contact_states,
        contact_actions,
    )

    train_seconds = run_training(
        trainer,
        run_dir=run_dir,
        label=label,
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
    )

    selector_stats = new_contact_action_head_stats()

    def select_contact_head_action(
        agent: Any,
        state: np.ndarray,
        game: CrystalCaves,
        info: dict[str, Any],
        step: int,
        action_labels: list[str],
    ) -> int:
        del info, step
        action, decision = contact_action_head_action(
            agent,
            state,
            game,
            action_labels=action_labels,
            close_zone_distance_tiles=contact_action_distance_tiles,
        )
        record_contact_action_head_decision(selector_stats, decision, action, action_labels)
        return action

    near_miss_eval = first_objective_near_miss_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_final",
        episode=trainer.current_episode,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
        action_selector=select_contact_head_action,
    )
    selector_payload = contact_action_head_stats_payload(selector_stats)
    if near_miss_eval is not None:
        near_miss_eval["contact_action_head"] = selector_payload
    eval_payload = _near_miss_rollup_as_eval_payload(
        near_miss_eval,
        label=f"{label}_final",
        episode=trainer.current_episode,
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
            "contact_action_head_training": {
                "dataset_path": str(correction_dataset_path),
                "dataset_states": int(contact_states.shape[0]),
                "state_size": int(contact_states.shape[1]) if contact_states.ndim == 2 else 0,
                "weight": float(contact_action_weight),
                "batch_size": int(contact_action_batch_size),
                "distance_tiles": float(contact_action_distance_tiles),
                **contact_dataset_summary,
            },
            "contact_action_head_eval": selector_payload,
            "near_miss_eval": near_miss_eval,
        },
    )


def run_contact_head_offline(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    correction_dataset_path: Path,
    seed: int,
    eval_games: int,
    log_every: int,
    report_seconds: float,
    contact_action_batch_size: int,
    contact_action_distance_tiles: float,
    contact_head_offline_steps: int,
    contact_head_learning_rate: float,
    contact_head_confidence: float,
    contact_head_jump_confidence: float,
    contact_head_action_thresholds: dict[str, float] | None,
    contact_head_balance_classes: bool,
    label: str = "contact_head_offline",
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
    config.MAX_EPISODES = 1
    config.EVAL_EVERY = 0
    config.EVAL_EPISODES = 0
    apply_contact_action_head_override(
        config,
        contact_action_weight=0.0,
        contact_action_batch_size=contact_action_batch_size,
        contact_action_distance_tiles=contact_action_distance_tiles,
    )
    trainer = prepare_trainer(
        config,
        episodes=1,
        vec_envs=1,
        transfer_weights=snapshot["weights"],
        strict_transfer=False,
        save_checkpoints=False,
    )
    trainer.agent.epsilon = 0.0
    selected_episode = int(snapshot.get("episode", 0) or 0)
    _validate_checkpoint_shape(trainer, snapshot, checkpoint_path)

    contact_states, contact_actions = load_correction_action_dataset(correction_dataset_path)
    if contact_states.shape[0] <= 0:
        raise ValueError("contact-head offline training requires at least one contact transition")
    contact_dataset_summary = trainer.agent.set_contact_action_dataset(
        contact_states,
        contact_actions,
    )
    game = CrystalCaves(config, headless=True)
    action_labels = list(getattr(game, "ACTION_LABELS", [])) or [
        str(index) for index in range(trainer.agent.action_size)
    ]
    action_confidence_thresholds = validate_contact_head_action_thresholds(
        contact_head_action_thresholds,
        action_labels,
    )
    dataset_stats = contact_action_dataset_stats(contact_actions, action_labels=action_labels)

    started = time.time()
    offline_training = train_contact_action_head_offline(
        trainer.agent,
        contact_states,
        contact_actions,
        steps=contact_head_offline_steps,
        batch_size=contact_action_batch_size,
        learning_rate=contact_head_learning_rate,
        balance_classes=contact_head_balance_classes,
        action_labels=action_labels,
    )
    train_seconds = time.time() - started

    selector_stats = new_contact_action_head_stats()

    def select_contact_head_action(
        agent: Any,
        state: np.ndarray,
        game: CrystalCaves,
        info: dict[str, Any],
        step: int,
        action_labels: list[str],
    ) -> int:
        del info, step
        action, decision = contact_action_head_action(
            agent,
            state,
            game,
            action_labels=action_labels,
            close_zone_distance_tiles=contact_action_distance_tiles,
            min_confidence=contact_head_confidence,
            jump_min_confidence=contact_head_jump_confidence,
            action_min_confidences=action_confidence_thresholds,
        )
        record_contact_action_head_decision(selector_stats, decision, action, action_labels)
        return action

    near_miss_eval = first_objective_near_miss_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_final",
        episode=0,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
        action_selector=select_contact_head_action,
    )
    selector_payload = contact_action_head_stats_payload(selector_stats)
    selector_payload["min_confidence"] = float(contact_head_confidence)
    selector_payload["jump_min_confidence"] = float(contact_head_jump_confidence)
    selector_payload["action_min_confidences"] = dict(action_confidence_thresholds)
    if near_miss_eval is not None:
        near_miss_eval["contact_action_head"] = selector_payload
    eval_payload = _near_miss_rollup_as_eval_payload(
        near_miss_eval,
        label=f"{label}_final",
        episode=0,
    )

    # Persist the adapter as a standalone loadable artifact (route trunk + trained
    # head in one selected-weight snapshot). Before this, promoted adapters like
    # B21 left an empty models/ dir — the head existed only in process memory and
    # the promoted policy was not reconstructable from disk.
    head_config_payload = dict(snapshot.get("config") or {})
    head_config_payload["contact_action_head"] = True
    head_config_payload["contact_head_source_checkpoint"] = str(checkpoint_path)
    head_checkpoint_path = save_selected_weight_snapshot(
        run_dir / "models" / "crystal_caves" / f"{label}_with_head_ep{selected_episode}.pth",
        label=f"{label}_with_head",
        config_payload=head_config_payload,
        state_size=trainer.agent.state_size,
        action_size=trainer.agent.action_size,
        selected_episode=selected_episode,
        source_eval=snapshot.get("source_eval") or {},
        weights=capture_weight_snapshot(trainer.agent),
    )

    offline_live = live_snapshot(
        trainer,
        label=label,
        status="complete",
        started=started,
        total_episodes=contact_head_offline_steps,
    )
    offline_live["offline_steps"] = int(contact_head_offline_steps)
    offline_live["offline_accuracy"] = float(offline_training.get("accuracy", 0.0) or 0.0)
    offline_live["selector_head_action_rate"] = float(
        selector_payload.get("head_action_rate", 0.0) or 0.0
    )
    write_json(run_dir / "live_metrics.json", offline_live)
    append_jsonl(run_dir / "live_metrics.jsonl", offline_live)
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
            "contact_head_checkpoint": head_checkpoint_path,
            "contact_action_head_training": {
                "mode": "offline_head_only",
                "dataset_path": str(correction_dataset_path),
                "dataset_states": int(contact_states.shape[0]),
                "state_size": int(contact_states.shape[1]) if contact_states.ndim == 2 else 0,
                "weight": 0.0,
                "batch_size": int(contact_action_batch_size),
                "distance_tiles": float(contact_action_distance_tiles),
                "learning_rate": float(contact_head_learning_rate),
                "offline_steps": int(contact_head_offline_steps),
                "confidence_threshold": float(contact_head_confidence),
                "jump_confidence_threshold": float(contact_head_jump_confidence),
                "action_confidence_thresholds": dict(action_confidence_thresholds),
                "balance_classes": bool(contact_head_balance_classes),
                "dataset_stats": dataset_stats,
                "offline_training": offline_training,
                **contact_dataset_summary,
            },
            "contact_action_head_eval": selector_payload,
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
