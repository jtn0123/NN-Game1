# ruff: noqa: F401,F403,F405,I001
from .common import *
from .config_helpers import *
from .demo_collect import *
from .demo_planners import *
from .evals import *
from .io_utils import *
from .reports import *
from .snapshots import *
from .training import *
from .vec_envs import *
from .runs_baseline import *
from .runs_mixed import *
from .runs_route import *
from .runs_demo import *
from .evals import (
    _action_label,
    _action_labels,
    _enter_greedy_agent_eval,
    _resolved_end_reason,
    _restore_greedy_agent_eval,
)


def run_transfer(
    out_dir: Path,
    *,
    episodes: int,
    drill_episodes: int,
    seed: int,
    eval_games: int,
    eval_k: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    save_checkpoints: bool,
    drill_eval_max_steps: int | None,
) -> list[dict[str, Any]]:
    drill_summary, weights = run_drill_pretrain(
        out_dir,
        episodes=drill_episodes,
        seed=seed,
        eval_k=eval_k,
        train_eval_games=train_eval_games,
        eval_every=eval_every,
        log_every=log_every,
        report_seconds=report_seconds,
        heartbeat_seconds=heartbeat_seconds,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
        drill_eval_max_steps=drill_eval_max_steps,
    )

    # Reset RNG before full-level fine-tuning so transfer and baseline see the
    # same generated cave order where possible. The intended variable is the
    # pretrained weights, not a different level sequence.
    set_seed(seed)
    run_dir = out_dir / "transfer_full"
    config = full_tutorial_config(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=weights,
        save_checkpoints=save_checkpoints,
    )
    train_seconds = run_training(
        trainer,
        run_dir=run_dir,
        label="transfer_full",
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
    )
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label="transfer_full_final",
        episode=trainer.current_episode,
        games=eval_games,
    )
    transfer_summary = summarize_trainer(
        trainer,
        label="transfer_full",
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={"transfer_checkpoint": "in-memory drill weights"},
    )
    return [drill_summary, transfer_summary]


def run_first_crystal_transfer(
    out_dir: Path,
    *,
    episodes: int,
    route_episodes: int,
    seed: int,
    eval_games: int,
    trace_games: int,
    trace_max_steps: int,
    trace_sample_every: int,
    trace_tail_steps: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    save_checkpoints: bool,
) -> list[dict[str, Any]]:
    route_summary, weights = run_first_crystal_pretrain(
        out_dir,
        episodes=route_episodes,
        seed=seed,
        eval_games=eval_games,
        train_eval_games=train_eval_games,
        eval_every=eval_every,
        log_every=log_every,
        report_seconds=report_seconds,
        heartbeat_seconds=heartbeat_seconds,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )

    # Reset RNG before full-objective fine-tuning so the transfer comparison is
    # about the route-pretrained weights, not a different procedural cave order.
    set_seed(seed)
    run_dir = out_dir / "first_crystal_transfer_full"
    config = full_tutorial_config(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=weights,
        save_checkpoints=save_checkpoints,
    )
    train_seconds = run_training(
        trainer,
        run_dir=run_dir,
        label="first_crystal_transfer_full",
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
    )
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label="first_crystal_transfer_full_final",
        episode=trainer.current_episode,
        games=eval_games,
    )
    diagnostics = trace_heldout_failures(
        config,
        trainer.agent,
        out_dir=run_dir,
        label="first_crystal_transfer_heldout",
        games=trace_games,
        max_steps=trace_max_steps,
        sample_every=trace_sample_every,
        tail_steps=trace_tail_steps,
    )
    source_eval = route_summary.get("selected_source_eval") or route_summary.get("final_eval") or {}
    transfer_summary = summarize_trainer(
        trainer,
        label="first_crystal_transfer_full",
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "transfer_checkpoint": "in-memory first-crystal route weights",
            "transfer_source": {
                "kind": "first_crystal",
                "source_episode": int(route_summary.get("selected_source_episode", 0) or 0),
                "source_win_rate": float(source_eval.get("win_rate", 0.0) or 0.0),
                "source_wins": int(source_eval.get("wins", 0) or 0),
                "source_games": int(source_eval.get("num_games", 0) or 0),
            },
            "failure_diagnostics": diagnostics,
        },
    )
    return [route_summary, transfer_summary]


def run_bridge_transfer(
    out_dir: Path,
    *,
    episodes: int,
    bridge_episodes: int,
    seed: int,
    eval_games: int,
    eval_k: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    save_checkpoints: bool,
    bridge_eval_max_steps: int | None,
    bridge_eval_every: int,
) -> list[dict[str, Any]]:
    bridge_summary, weights = run_bridge_pretrain(
        out_dir,
        episodes=bridge_episodes,
        seed=seed,
        eval_k=eval_k,
        train_eval_games=train_eval_games,
        eval_every=eval_every,
        log_every=log_every,
        report_seconds=report_seconds,
        heartbeat_seconds=heartbeat_seconds,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
        bridge_eval_max_steps=bridge_eval_max_steps,
        bridge_eval_every=bridge_eval_every,
    )

    # Reset RNG before full-level fine-tuning so bridge-transfer and baseline
    # see the same generated cave order where possible. The transferred variable
    # is the bridge-selected policy, not a different level sequence.
    set_seed(seed)
    run_dir = out_dir / "bridge_transfer_full"
    config = full_tutorial_config(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=weights,
        save_checkpoints=save_checkpoints,
    )
    train_seconds = run_training(
        trainer,
        run_dir=run_dir,
        label="bridge_transfer_full",
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
    )
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label="bridge_transfer_full_final",
        episode=trainer.current_episode,
        games=eval_games,
    )
    selected_episode = bridge_summary.get("selected_bridge_episode")
    transfer_summary = summarize_trainer(
        trainer,
        label="bridge_transfer_full",
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "transfer_checkpoint": f"in-memory bridge weights from ep {selected_episode}",
            "transfer_source": {
                "kind": "bridge",
                "selected_bridge_episode": selected_episode,
                "selected_bridge_rollup": bridge_summary.get("selected_bridge_rollup"),
            },
        },
    )
    return [bridge_summary, transfer_summary]


def run_bridge_demo_replay(
    out_dir: Path,
    *,
    episodes: int,
    bridge_episodes: int,
    seed: int,
    eval_games: int,
    eval_k: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    save_checkpoints: bool,
    bridge_eval_max_steps: int | None,
    bridge_eval_every: int,
    demo_repeat: int,
) -> list[dict[str, Any]]:
    bridge_summary, weights = run_bridge_pretrain(
        out_dir,
        episodes=bridge_episodes,
        seed=seed,
        eval_k=eval_k,
        train_eval_games=train_eval_games,
        eval_every=eval_every,
        log_every=log_every,
        report_seconds=report_seconds,
        heartbeat_seconds=heartbeat_seconds,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
        bridge_eval_max_steps=bridge_eval_max_steps,
        bridge_eval_every=bridge_eval_every,
    )

    demo_dir = out_dir / "bridge_demo_collect"
    demo_config = bridge_config(
        demo_dir,
        episodes=0,
        seed=seed,
        eval_every=0,
        train_eval_games=0,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    demo_trainer = prepare_trainer(
        demo_config,
        episodes=0,
        vec_envs=1,
        transfer_weights=weights,
        save_checkpoints=False,
    )
    demos = collect_policy_demonstrations(
        demo_trainer.agent,
        demo_config,
        specs=BRIDGE_CAVES,
        k=eval_k,
        max_steps=bridge_eval_max_steps,
        only_wins=True,
    )
    demo_summary = demos["summary"]
    write_json(demo_dir / "demo_summary.json", demo_summary)
    if demo_trainer.vec_env is not None:
        demo_trainer.vec_env.close()

    # Reset RNG before full-level training so the comparison is about replay
    # seeding, not a shifted procedural cave sequence.
    set_seed(seed)
    run_dir = out_dir / "bridge_demo_replay_full"
    config = full_tutorial_config(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    seeded = seed_replay_from_demonstrations(
        trainer.agent,
        demos["trajectories"],
        repeat=demo_repeat,
    )
    train_seconds = run_training(
        trainer,
        run_dir=run_dir,
        label="bridge_demo_replay_full",
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
    )
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label="bridge_demo_replay_full_final",
        episode=trainer.current_episode,
        games=eval_games,
    )
    selected_episode = bridge_summary.get("selected_bridge_episode")
    replay_summary = summarize_trainer(
        trainer,
        label="bridge_demo_replay_full",
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "demo_replay": {
                "kind": "bridge",
                "selected_bridge_episode": selected_episode,
                "selected_bridge_rollup": bridge_summary.get("selected_bridge_rollup"),
                "demo_summary": demo_summary,
                "seeded": seeded,
                "demo_summary_path": str(demo_dir / "demo_summary.json"),
            }
        },
    )
    return [bridge_summary, replay_summary]


def config_from_selected_checkpoint(
    out_dir: Path,
    *,
    snapshot: dict[str, Any],
    seed: int,
    log_every: int,
    report_seconds: float,
) -> Config:
    saved_config = snapshot.get("config") or {}
    difficulty = str(saved_config.get("cave_difficulty", "tutorial") or "tutorial")
    first_crystal_goal = bool(saved_config.get("first_crystal_goal", True))
    if first_crystal_goal:
        config = first_crystal_config(
            out_dir,
            episodes=1,
            seed=seed,
            eval_every=0,
            train_eval_games=0,
            log_every=log_every,
            report_seconds=report_seconds,
            difficulty=difficulty,
        )
    else:
        config = full_tutorial_config(
            out_dir,
            episodes=1,
            seed=seed,
            eval_every=0,
            train_eval_games=0,
            log_every=log_every,
            report_seconds=report_seconds,
        )
        config.CRYSTAL_CAVES_DIFFICULTY = difficulty

    cave_pool = int(saved_config.get("cave_pool_size", 0) or 0)
    apply_cave_pool_override(config, cave_pool if cave_pool > 0 else None)
    exp_config = cc_experiment_config(config)
    # Audit R2-D: restore the state-SHAPE flags too. history_state/steps enlarge the env
    # state_size; if not read back, the rebuilt eval env has a smaller state than the saved
    # net and the shape guard/strict load rejects the checkpoint (it can never be graded).
    exp_config.CRYSTAL_CAVES_HISTORY_STATE = bool(saved_config.get("history_state", False))
    exp_config.CRYSTAL_CAVES_HISTORY_STEPS = int(saved_config.get("history_steps", 4) or 4)
    exp_config.CRYSTAL_CAVES_ROUTE_AUX_LOSS = bool(saved_config.get("route_aux_loss", False))
    exp_config.CRYSTAL_CAVES_ROUTE_AUX_WEIGHT = float(
        saved_config.get("route_aux_weight", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_ROUTE_AUX_DEADBAND = float(
        saved_config.get("route_aux_deadband", 0.01) or 0.01
    )
    exp_config.CRYSTAL_CAVES_DEMO_ACTION_LOSS = bool(saved_config.get("demo_action_loss", False))
    exp_config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT = float(
        saved_config.get("demo_action_weight", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_DEMO_ACTION_MARGIN = float(
        saved_config.get("demo_action_margin", 0.8) or 0.8
    )
    exp_config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE = int(
        saved_config.get("demo_action_batch_size", 64) or 64
    )
    exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT = float(
        saved_config.get("demo_conservative_weight", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE = float(
        saved_config.get("demo_conservative_temperature", 1.0) or 1.0
    )
    exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS = bool(
        saved_config.get("close_zone_demo_action_loss", False)
    )
    exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT = float(
        saved_config.get("close_zone_demo_action_weight", 0.03) or 0.03
    )
    exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE = int(
        saved_config.get("close_zone_demo_action_batch_size", 64) or 64
    )
    exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS = bool(
        saved_config.get("correction_action_loss", False)
    )
    exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT = float(
        saved_config.get("correction_action_weight", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN = float(
        saved_config.get("correction_action_margin", 0.6) or 0.6
    )
    exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE = int(
        saved_config.get("correction_action_batch_size", 64) or 64
    )
    exp_config.CRYSTAL_CAVES_POLICY_ANCHOR_LOSS = bool(
        saved_config.get("policy_anchor_loss", False)
    )
    exp_config.CRYSTAL_CAVES_POLICY_ANCHOR_WEIGHT = float(
        saved_config.get("policy_anchor_weight", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_POLICY_ANCHOR_TEMPERATURE = float(
        saved_config.get("policy_anchor_temperature", 1.0) or 1.0
    )
    exp_config.CRYSTAL_CAVES_POLICY_ANCHOR_MIN_TARGET_DISTANCE_NORM = float(
        saved_config.get("policy_anchor_min_target_distance_norm", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_CONTACT_ACTION_HEAD = bool(
        saved_config.get("contact_action_head", False)
    )
    exp_config.CRYSTAL_CAVES_CONTACT_ACTION_WEIGHT = float(
        saved_config.get("contact_action_weight", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_CONTACT_ACTION_BATCH_SIZE = int(
        saved_config.get("contact_action_batch_size", 64) or 64
    )
    exp_config.CRYSTAL_CAVES_CONTACT_ACTION_DISTANCE_NORM = float(
        saved_config.get("contact_action_distance_norm", 0.0) or 0.0
    )
    exp_config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = bool(
        saved_config.get("invalid_interact_penalty", False)
    )
    exp_config.CRYSTAL_CAVES_INVALID_SHOOT_PENALTY = bool(
        saved_config.get("invalid_shoot_penalty", False)
    )
    exp_config.CRYSTAL_CAVES_NOVELTY_BONUS = bool(saved_config.get("novelty_bonus", False))
    return config


def _policy_close_zone_plan(
    agent: Any,
    state: np.ndarray,
    game: CrystalCaves,
    *,
    max_actions: int,
) -> list[int]:
    """Roll out the greedy policy on a copied game for local option gating."""

    if max_actions <= 0:
        raise ValueError("policy close-zone plan max_actions must be positive")
    try:
        probe = copy.deepcopy(game)
    except Exception:  # pragma: no cover - defensive; normal deepcopy is covered elsewhere
        return [int(agent.select_action(state, training=False))]

    plan: list[int] = []
    probe_state = state.copy()
    for _ in range(max_actions):
        action = int(agent.select_action(probe_state, training=False))
        plan.append(action)
        probe_state, _, done, _ = probe.step(action)
        if done:
            break
    return plan or [int(agent.select_action(state, training=False))]


def final_contact_option_action(
    agent: Any,
    state: np.ndarray,
    game: CrystalCaves,
    info: dict[str, Any],
    *,
    action_labels: list[str],
    close_zone_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    option_queue: deque[int] | None = None,
    final_contact_commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
    cancel_option_outside_close_zone: bool = False,
    gate_policy_advantage: bool = False,
    min_option_advantage: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    """Route with the NN policy, but use the oracle local macro in close contact."""

    objective = objective_snapshot(game)
    target_distance = objective.get("target_distance_tiles")
    cancelled_actions = 0
    if option_queue is not None and option_queue:
        if cancel_option_outside_close_zone and (
            target_distance is None or float(target_distance) > close_zone_distance_tiles
        ):
            cancelled_actions = len(option_queue)
            option_queue.clear()
        else:
            action = int(option_queue.popleft())
            return action, {
                "source": "final_contact_option",
                "objective": objective,
                "target_distance_tiles": target_distance,
                "option_meta": {
                    "reason": "committed_plan",
                    "score": 0.0,
                    "planned": False,
                    "remaining_committed_actions": len(option_queue),
                },
                "action_label": _action_label(action_labels, action),
            }

    if target_distance is not None and float(target_distance) <= close_zone_distance_tiles:
        stale_steps = int(info.get("steps_since_progress", 0) or 0)
        plan, option_meta = close_zone_oracle_plan(
            game,
            stale_steps=stale_steps,
            max_actions=final_contact_commit_steps,
        )
        if not plan:
            plan = [game.IDLE]
        if gate_policy_advantage:
            policy_plan = _policy_close_zone_plan(
                agent,
                state,
                game,
                max_actions=final_contact_commit_steps,
            )
            policy_meta = close_zone_sequence_score(
                game,
                policy_plan,
                stale_steps=stale_steps,
            )
            option_score = float(option_meta.get("score", 0.0) or 0.0)
            policy_score = float(policy_meta.get("score", 0.0) or 0.0)
            option_advantage = option_score - policy_score
            option_meta = {
                **option_meta,
                "gate_policy_advantage": True,
                "gate_min_option_advantage": float(min_option_advantage),
                "policy_plan_score": policy_score,
                "policy_plan_first_action": _action_label(action_labels, int(policy_plan[0])),
                "option_advantage": float(option_advantage),
                "policy_plan_meta": policy_meta,
            }
            if option_advantage < min_option_advantage:
                action = int(policy_plan[0])
                return action, {
                    "source": "policy",
                    "objective": objective,
                    "target_distance_tiles": float(target_distance),
                    "option_meta": {
                        **option_meta,
                        "planned": True,
                        "rejected_by_policy_gate": True,
                        "planned_length": len(plan),
                        "commit_steps": int(final_contact_commit_steps),
                        "cancelled_committed_actions": cancelled_actions,
                    },
                    "action_label": _action_label(action_labels, action),
                }
        action = int(plan[0])
        if option_queue is not None:
            option_queue.extend(int(next_action) for next_action in plan[1:])
        option_meta = {
            **option_meta,
            "planned": True,
            "planned_length": len(plan),
            "commit_steps": int(final_contact_commit_steps),
            "cancelled_committed_actions": cancelled_actions,
            "gate_policy_advantage": bool(gate_policy_advantage),
            "gate_min_option_advantage": float(min_option_advantage),
            "rejected_by_policy_gate": False,
        }
        return int(action), {
            "source": "final_contact_option",
            "objective": objective,
            "target_distance_tiles": float(target_distance),
            "option_meta": option_meta,
            "action_label": _action_label(action_labels, int(action)),
        }

    action = int(agent.select_action(state, training=False))
    return action, {
        "source": "policy",
        "objective": objective,
        "target_distance_tiles": target_distance,
        "option_meta": {
            "cancelled_committed_actions": cancelled_actions,
        },
        "action_label": _action_label(action_labels, action),
    }


def _new_final_contact_option_stats() -> dict[str, Any]:
    return {
        "steps": 0,
        "policy_actions": 0,
        "option_triggers": 0,
        "option_target_completed": 0,
        "option_cancelled_plans": 0,
        "option_cancelled_actions": 0,
        "option_gate_evaluations": 0,
        "option_gate_rejections": 0,
        "option_gate_advantage_total": 0.0,
        "option_gate_rejected_advantage_total": 0.0,
        "option_gate_accepted_advantage_total": 0.0,
        "source_counts": Counter(),
        "action_counts": Counter(),
        "option_action_counts": Counter(),
        "option_reason_counts": Counter(),
    }


def _record_final_contact_option_decision(
    stats: dict[str, Any],
    *,
    action: int,
    action_labels: list[str],
    decision: dict[str, Any],
) -> None:
    stats["steps"] += 1
    source = str(decision.get("source", "policy") or "policy")
    action_label = _action_label(action_labels, action)
    stats["source_counts"][source] += 1
    stats["action_counts"][action_label] += 1
    option_meta = decision.get("option_meta") or {}
    if not isinstance(option_meta, dict):
        option_meta = {}
    cancelled_actions = int(option_meta.get("cancelled_committed_actions", 0) or 0)
    if cancelled_actions > 0:
        stats["option_cancelled_plans"] += 1
        stats["option_cancelled_actions"] += cancelled_actions
    if option_meta.get("gate_policy_advantage"):
        option_advantage = float(option_meta.get("option_advantage", 0.0) or 0.0)
        stats["option_gate_evaluations"] += 1
        stats["option_gate_advantage_total"] += option_advantage
        if option_meta.get("rejected_by_policy_gate"):
            stats["option_gate_rejections"] += 1
            stats["option_gate_rejected_advantage_total"] += option_advantage
        else:
            stats["option_gate_accepted_advantage_total"] += option_advantage
    if source != "final_contact_option":
        stats["policy_actions"] += 1
        return

    stats["option_triggers"] += 1
    stats["option_action_counts"][action_label] += 1
    reason = str(option_meta.get("reason", "unknown") or "unknown")
    stats["option_reason_counts"][reason] += 1
    if option_meta.get("target_completed"):
        stats["option_target_completed"] += 1


def _final_contact_option_stats_payload(
    stats: dict[str, Any],
    *,
    close_zone_distance_tiles: float,
    gate_policy_advantage: bool = False,
    min_option_advantage: float = 0.0,
) -> dict[str, Any]:
    steps = int(stats.get("steps", 0) or 0)
    triggers = int(stats.get("option_triggers", 0) or 0)
    gate_evaluations = int(stats.get("option_gate_evaluations", 0) or 0)
    gate_rejections = int(stats.get("option_gate_rejections", 0) or 0)
    gate_acceptances = max(0, gate_evaluations - gate_rejections)
    return {
        "enabled": True,
        "close_zone_distance_tiles": float(close_zone_distance_tiles),
        "policy_advantage_gate_enabled": bool(gate_policy_advantage),
        "min_option_advantage": float(min_option_advantage),
        "steps": steps,
        "policy_actions": int(stats.get("policy_actions", 0) or 0),
        "option_triggers": triggers,
        "option_trigger_rate": float(triggers / max(1, steps)),
        "option_target_completed": int(stats.get("option_target_completed", 0) or 0),
        "option_target_completed_rate": float(
            int(stats.get("option_target_completed", 0) or 0) / max(1, triggers)
        ),
        "option_cancelled_plans": int(stats.get("option_cancelled_plans", 0) or 0),
        "option_cancelled_actions": int(stats.get("option_cancelled_actions", 0) or 0),
        "option_gate_evaluations": gate_evaluations,
        "option_gate_rejections": gate_rejections,
        "option_gate_acceptances": gate_acceptances,
        "option_gate_rejection_rate": float(gate_rejections / max(1, gate_evaluations)),
        "mean_option_advantage": float(
            float(stats.get("option_gate_advantage_total", 0.0) or 0.0) / max(1, gate_evaluations)
        ),
        "mean_rejected_option_advantage": float(
            float(stats.get("option_gate_rejected_advantage_total", 0.0) or 0.0)
            / max(1, gate_rejections)
        ),
        "mean_accepted_option_advantage": float(
            float(stats.get("option_gate_accepted_advantage_total", 0.0) or 0.0)
            / max(1, gate_acceptances)
        ),
        "source_counts": dict(stats.get("source_counts", {})),
        "action_counts": dict(stats.get("action_counts", {})),
        "option_action_counts": dict(stats.get("option_action_counts", {})),
        "option_reason_counts": dict(stats.get("option_reason_counts", {})),
    }


def _final_contact_selection_score(config: Config, payload: dict[str, Any]) -> float:
    return (
        float(payload.get("win_rate", 0.0) or 0.0)
        * float(getattr(config, "EVAL_SELECTION_W_WIN", 1.0))
        + float(payload.get("mean_crystal_frac", 0.0) or 0.0)
        * float(getattr(config, "EVAL_SELECTION_W_CRYSTAL", 0.2))
        + float(payload.get("mean_score", 0.0) or 0.0)
        * float(getattr(config, "EVAL_SELECTION_W_SCORE", 0.0001))
    )


def final_contact_option_eval(
    config: Config,
    agent: Any,
    *,
    out_dir: Path,
    label: str,
    episode: int,
    games: int,
    max_steps: int | None = None,
    close_zone_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    final_contact_commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
    cancel_option_outside_close_zone: bool = False,
    gate_policy_advantage: bool = False,
    min_option_advantage: float = 0.0,
) -> dict[str, Any]:
    if games <= 0:
        raise ValueError("num option eval games must be positive")
    step_limit = int(max_steps or config.EVAL_MAX_STEPS)
    if step_limit <= 0:
        raise ValueError("option eval max_steps must be positive")
    if close_zone_distance_tiles <= 0:
        raise ValueError("close-zone option distance must be positive")
    if final_contact_commit_steps <= 0:
        raise ValueError("final-contact commit steps must be positive")
    if min_option_advantage < 0:
        raise ValueError("final-contact minimum option advantage must be non-negative")

    eval_dir = out_dir / "final_contact_option_eval" / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows_path = eval_dir / "per_level_eval.jsonl"
    rows_path.unlink(missing_ok=True)

    game = CrystalCaves(config, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()
    action_labels = _action_labels(game)

    scores: list[float] = []
    levels: list[int] = []
    steps_list: list[int] = []
    crystal_fracs: list[float] = []
    switch_rates: list[float] = []
    depth_fracs: list[float] = []
    end_reasons: list[str] = []
    wins = 0
    rows: list[dict[str, Any]] = []
    option_stats = _new_final_contact_option_stats()
    agent_state = _enter_greedy_agent_eval(agent)
    try:
        for game_index in range(games):
            state = game.reset()
            initial_crystals = int(game.initial_crystals)
            done = False
            info = game._info()
            steps = 0
            per_game_stats = _new_final_contact_option_stats()
            option_queue: deque[int] = deque()

            for step in range(step_limit):
                action, decision = final_contact_option_action(
                    agent,
                    state,
                    game,
                    info,
                    action_labels=action_labels,
                    close_zone_distance_tiles=close_zone_distance_tiles,
                    option_queue=option_queue,
                    final_contact_commit_steps=final_contact_commit_steps,
                    cancel_option_outside_close_zone=cancel_option_outside_close_zone,
                    gate_policy_advantage=gate_policy_advantage,
                    min_option_advantage=min_option_advantage,
                )
                _record_final_contact_option_decision(
                    option_stats,
                    action=action,
                    action_labels=action_labels,
                    decision=decision,
                )
                _record_final_contact_option_decision(
                    per_game_stats,
                    action=action,
                    action_labels=action_labels,
                    decision=decision,
                )
                state, _, done, info = game.step(action)
                steps = step + 1
                if done:
                    break

            won = bool(info.get("won", False))
            wins += int(won)
            reason = _resolved_end_reason(info, steps=steps, max_steps=step_limit)
            parts = info.get("progress_parts") or {}
            if not isinstance(parts, dict):
                parts = {}
            score = float(info.get("score", 0) or 0)
            level = int(info.get("level", 1) or 1)
            scores.append(score)
            levels.append(level)
            steps_list.append(steps)
            crystal_fracs.append(float(parts.get("crystal_frac", 0.0) or 0.0))
            switch_rates.append(float(parts.get("switch_done", 0.0) or 0.0))
            depth_fracs.append(float(parts.get("depth_frac", 0.0) or 0.0))
            end_reasons.append(reason)

            row = {
                "game_index": game_index,
                "episode": int(episode),
                "level": info.get("level"),
                "level_name": info.get("level_name"),
                "steps": steps,
                "end_reason": reason,
                "won": won,
                "score": score,
                "initial_crystals": initial_crystals,
                "crystals_collected": initial_crystals
                - int(info.get("crystals_remaining", 0) or 0),
                "crystals_remaining": int(info.get("crystals_remaining", 0) or 0),
                "exit_unlocked": bool(info.get("exit_unlocked", False)),
                "final_progress": float(info.get("progress", 0.0) or 0.0),
                "final_depth_frac": float(parts.get("depth_frac", 0.0) or 0.0),
                "final_objective": objective_snapshot(game),
                "final_contact_option": _final_contact_option_stats_payload(
                    per_game_stats,
                    close_zone_distance_tiles=close_zone_distance_tiles,
                    gate_policy_advantage=gate_policy_advantage,
                    min_option_advantage=min_option_advantage,
                ),
            }
            rows.append(row)
            append_jsonl(rows_path, row)
    finally:
        _restore_greedy_agent_eval(agent, agent_state)

    scores_arr = np.array(scores)
    levels_arr = np.array(levels)
    steps_arr = np.array(steps_list)
    max_level_seen = max(10, int(np.max(levels_arr)))
    level_dist = {level: int((levels_arr == level).sum()) for level in range(1, max_level_seen + 1)}
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "episode": int(episode),
        "num_games": games,
        "mean_score": float(np.mean(scores_arr)),
        "median_score": float(np.median(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "min_score": int(np.min(scores_arr)),
        "max_score": int(np.max(scores_arr)),
        "q25_score": float(np.percentile(scores_arr, 25)),
        "q75_score": float(np.percentile(scores_arr, 75)),
        "mean_level": float(np.mean(levels_arr)),
        "max_level": int(np.max(levels_arr)),
        "level_distribution": level_dist,
        "wins": int(wins),
        "win_rate": float(wins / games),
        "mean_steps": float(np.mean(steps_arr)),
        "max_steps": int(np.max(steps_arr)),
        "mean_crystal_frac": float(np.mean(crystal_fracs)) if crystal_fracs else 0.0,
        "mean_switch_rate": float(np.mean(switch_rates)) if switch_rates else 0.0,
        "mean_depth_frac": float(np.mean(depth_fracs)) if depth_fracs else 0.0,
        "end_reason_counts": dict(Counter(end_reasons)),
        "label": label,
        "eval_max_steps": step_limit,
        "rows_path": str(rows_path),
        "summary_path": str(eval_dir / "summary.json"),
        "final_contact_option": {
            **_final_contact_option_stats_payload(
                option_stats,
                close_zone_distance_tiles=close_zone_distance_tiles,
                gate_policy_advantage=gate_policy_advantage,
                min_option_advantage=min_option_advantage,
            ),
            "commit_steps": int(final_contact_commit_steps),
            "cancel_outside_close_zone": bool(cancel_option_outside_close_zone),
            "rows_path": str(rows_path),
            "summary_path": str(eval_dir / "summary.json"),
        },
        "rows": rows,
    }
    payload["selection_score"] = _final_contact_selection_score(config, payload)
    write_json(eval_dir / "summary.json", payload)
    print(
        f"Final-contact option eval {wins}/{games} wins, "
        f"{100 * payload['mean_crystal_frac']:.1f}% crystals, "
        f"{100 * payload['mean_depth_frac']:.1f}% depth, "
        f"{payload['final_contact_option']['option_triggers']} option triggers"
    )
    return payload


def run_eval_final_contact_option(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    seed: int,
    eval_games: int,
    log_every: int,
    report_seconds: float,
    close_zone_distance_tiles: float = CLOSE_ZONE_DISTANCE_TILES,
    final_contact_commit_steps: int = ROUTE_BEAM_COMMIT_STEPS,
    cancel_option_outside_close_zone: bool = False,
    gate_policy_advantage: bool = False,
    min_option_advantage: float = 0.0,
    label: str = "eval_final_contact_option",
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
    trainer.current_episode = int(snapshot.get("episode", 0) or 0)
    saved_state_size = int(snapshot.get("state_size", trainer.agent.state_size) or 0)
    saved_action_size = int(snapshot.get("action_size", trainer.agent.action_size) or 0)
    if saved_state_size and saved_state_size != trainer.agent.state_size:
        raise ValueError(
            f"checkpoint state size {saved_state_size} does not match "
            f"environment state size {trainer.agent.state_size}"
        )
    if saved_action_size and saved_action_size != trainer.agent.action_size:
        raise ValueError(
            f"checkpoint action size {saved_action_size} does not match "
            f"environment action size {trainer.agent.action_size}"
        )
    load_weight_snapshot(trainer.agent, snapshot["weights"])

    selected_episode = int(snapshot.get("episode", 0) or 0)
    eval_payload = final_contact_option_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        episode=selected_episode,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
        close_zone_distance_tiles=close_zone_distance_tiles,
        final_contact_commit_steps=final_contact_commit_steps,
        cancel_option_outside_close_zone=cancel_option_outside_close_zone,
        gate_policy_advantage=gate_policy_advantage,
        min_option_advantage=min_option_advantage,
    )
    near_miss_option_stats = _new_final_contact_option_stats()
    near_miss_option_queue: deque[int] = deque()

    def select_near_miss_option_action(
        agent_arg: Any,
        state: np.ndarray,
        game: CrystalCaves,
        info: dict[str, Any],
        step: int,
        action_labels: list[str],
    ) -> int:
        del agent_arg
        if step == 0:
            near_miss_option_queue.clear()
        action, decision = final_contact_option_action(
            trainer.agent,
            state,
            game,
            info,
            action_labels=action_labels,
            close_zone_distance_tiles=close_zone_distance_tiles,
            option_queue=near_miss_option_queue,
            final_contact_commit_steps=final_contact_commit_steps,
            cancel_option_outside_close_zone=cancel_option_outside_close_zone,
            gate_policy_advantage=gate_policy_advantage,
            min_option_advantage=min_option_advantage,
        )
        _record_final_contact_option_decision(
            near_miss_option_stats,
            action=action,
            action_labels=action_labels,
            decision=decision,
        )
        return action

    near_miss_eval = first_objective_near_miss_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        episode=selected_episode,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
        action_selector=select_near_miss_option_action,
    )
    if near_miss_eval:
        near_miss_eval["final_contact_option"] = _final_contact_option_stats_payload(
            near_miss_option_stats,
            close_zone_distance_tiles=close_zone_distance_tiles,
            gate_policy_advantage=gate_policy_advantage,
            min_option_advantage=min_option_advantage,
        )
        summary_path = near_miss_eval.get("summary_path")
        if summary_path:
            write_json(Path(str(summary_path)), near_miss_eval)

    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=0.0,
        final_eval_payload=eval_payload,
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_eval": {
                "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
                "source_label": snapshot.get("label", ""),
                "source_episode": selected_episode,
                "source_eval": snapshot.get("source_eval") or {},
            },
            "near_miss_eval": near_miss_eval,
            "final_contact_option": eval_payload.get("final_contact_option") or {},
        },
    )


def run_eval_checkpoint(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    seed: int,
    eval_games: int,
    trace_games: int,
    trace_max_steps: int,
    trace_sample_every: int,
    trace_tail_steps: int,
    log_every: int,
    report_seconds: float,
    label: str = "eval_checkpoint",
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
    trainer.current_episode = int(snapshot.get("episode", 0) or 0)
    saved_state_size = int(snapshot.get("state_size", trainer.agent.state_size) or 0)
    saved_action_size = int(snapshot.get("action_size", trainer.agent.action_size) or 0)
    if saved_state_size and saved_state_size != trainer.agent.state_size:
        raise ValueError(
            f"checkpoint state size {saved_state_size} does not match "
            f"environment state size {trainer.agent.state_size}"
        )
    if saved_action_size and saved_action_size != trainer.agent.action_size:
        raise ValueError(
            f"checkpoint action size {saved_action_size} does not match "
            f"environment action size {trainer.agent.action_size}"
        )
    load_weight_snapshot(trainer.agent, snapshot["weights"])

    selected_episode = int(snapshot.get("episode", 0) or 0)
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        episode=selected_episode,
        games=eval_games,
    )
    diagnostics = trace_heldout_failures(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_trace",
        games=trace_games,
        max_steps=trace_max_steps,
        sample_every=trace_sample_every,
        tail_steps=trace_tail_steps,
    )
    near_miss_eval = first_objective_near_miss_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        episode=selected_episode,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
    )
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=0.0,
        final_eval_payload=eval_payload,
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_eval": {
                "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
                "source_label": snapshot.get("label", ""),
                "source_episode": selected_episode,
                "source_eval": snapshot.get("source_eval") or {},
            },
            "failure_diagnostics": diagnostics,
            "near_miss_eval": near_miss_eval,
        },
    )


def compare_runs(baseline: dict[str, Any], transfer: dict[str, Any]) -> dict[str, Any]:
    b = baseline.get("final_eval", {})
    t = transfer.get("final_eval", {})

    def delta(metric: str) -> float:
        return float(t.get(metric, 0.0) or 0.0) - float(b.get(metric, 0.0) or 0.0)

    return {
        "delta_win_rate_pct": round(100 * delta("win_rate"), 3),
        "delta_wins": int(t.get("wins", 0) or 0) - int(b.get("wins", 0) or 0),
        "delta_crystal_pct": round(100 * delta("mean_crystal_frac"), 3),
        "delta_depth_pct": round(100 * delta("mean_depth_frac"), 3),
        "delta_mean_score": round(delta("mean_score"), 3),
        "baseline_final": {
            "wins": b.get("wins"),
            "num_games": b.get("num_games"),
            "win_rate": b.get("win_rate"),
            "crystal_frac": b.get("mean_crystal_frac"),
            "depth_frac": b.get("mean_depth_frac"),
            "mean_score": b.get("mean_score"),
            "ends": b.get("end_reason_counts"),
        },
        "transfer_final": {
            "wins": t.get("wins"),
            "num_games": t.get("num_games"),
            "win_rate": t.get("win_rate"),
            "crystal_frac": t.get("mean_crystal_frac"),
            "depth_frac": t.get("mean_depth_frac"),
            "mean_score": t.get("mean_score"),
            "ends": t.get("end_reason_counts"),
        },
    }
