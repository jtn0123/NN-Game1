# ruff: noqa: F401,F403,F405,I001
from .common import *
from .config_helpers import *
from .io_utils import *
from .snapshots import *
from .stats import *


def summarize_trainer(
    trainer: HeadlessTrainer,
    *,
    label: str,
    train_seconds: float,
    final_eval_payload: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = trainer.config
    progress_parts = trainer.progress_parts[-100:]
    summary: dict[str, Any] = {
        "label": label,
        "episodes": trainer.current_episode,
        "total_steps": trainer.total_steps,
        "train_seconds": train_seconds,
        "steps_per_second": trainer.total_steps / train_seconds if train_seconds > 0 else 0.0,
        "device": str(config.DEVICE),
        "state_size": trainer.game.state_size,
        "action_size": trainer.game.action_size,
        "best_training_score": trainer.best_score,
        "final_epsilon": float(trainer.agent.epsilon),
        "memory_size": len(trainer.agent.memory),
        "avg_loss_100": float(trainer.agent.get_average_loss(100)),
        "avg_route_aux_loss_100": float(trainer.agent.get_average_route_aux_loss(100)),
        "avg_route_aux_accuracy_100": float(trainer.agent.get_average_route_aux_accuracy(100)),
        "avg_demo_action_loss_100": float(trainer.agent.get_average_demo_action_loss(100)),
        "avg_demo_conservative_loss_100": float(
            trainer.agent.get_average_demo_conservative_loss(100)
        ),
        "avg_demo_action_accuracy_100": float(trainer.agent.get_average_demo_action_accuracy(100)),
        "avg_close_zone_demo_action_loss_100": float(
            trainer.agent.get_average_close_zone_demo_action_loss(100)
        ),
        "avg_close_zone_demo_action_accuracy_100": float(
            trainer.agent.get_average_close_zone_demo_action_accuracy(100)
        ),
        "avg_correction_action_loss_100": float(
            trainer.agent.get_average_correction_action_loss(100)
        ),
        "avg_correction_action_accuracy_100": float(
            trainer.agent.get_average_correction_action_accuracy(100)
        ),
        "correction_action_enabled": bool(
            getattr(config, "CRYSTAL_CAVES_CORRECTION_ACTION_LOSS", False)
        ),
        "correction_action_transitions": int(
            trainer.agent.get_correction_action_transition_count()
        ),
        "correction_action_samples_100": int(trainer.agent.get_correction_action_metric_count(100)),
        "avg_q_100": mean_tail(trainer.q_values),
        "avg_score_100": mean_tail([float(score) for score in trainer.scores]),
        "win_rate_100": float(np.mean(trainer.wins[-100:])) if trainer.wins else 0.0,
        "avg_reward_100": mean_tail(trainer.rewards),
        "avg_progress_100": mean_tail(trainer.progresses),
        "best_progress": max_or_zero(trainer.progresses),
        "end_reason_counts_100": counter_tail(trainer.end_reasons),
        "mean_phi_parts_100": mean_dicts(
            [{key: float(value or 0.0) for key, value in part.items()} for part in progress_parts]
        ),
        "exploration_actions": int(trainer.exploration_actions),
        "exploitation_actions": int(trainer.exploitation_actions),
        "target_updates": int(trainer.target_updates),
        "model_dir": config.GAME_MODEL_DIR,
        "config": config_snapshot(config),
        "eval_history": [
            result.to_dict()
            for result in (trainer.evaluator.eval_history if trainer.evaluator else [])
        ],
        "source_stats": trainer_source_stats(trainer),
        "reverse_start_stats": trainer_reverse_start_stats(trainer),
        "archive_stats": trainer_archive_stats(trainer),
    }
    if final_eval_payload is not None:
        summary["final_eval"] = final_eval_payload
    if extra:
        summary.update(extra)
    return summary


def config_snapshot(config: Config) -> dict[str, Any]:
    exp_config = cc_experiment_config(config)
    return {
        "learning_rate": config.LEARNING_RATE,
        "gamma": config.GAMMA,
        "batch_size": config.BATCH_SIZE,
        "learn_every": config.LEARN_EVERY,
        "gradient_steps": config.GRADIENT_STEPS,
        "target_update": config.TARGET_UPDATE,
        "use_soft_update": config.USE_SOFT_UPDATE,
        "target_tau": config.TARGET_TAU,
        "use_n_step_returns": config.USE_N_STEP_RETURNS,
        "n_step_size": config.N_STEP_SIZE,
        "use_prioritized_replay": config.USE_PRIORITIZED_REPLAY,
        "use_noisy_networks": config.USE_NOISY_NETWORKS,
        "epsilon_start": config.EPSILON_START,
        "epsilon_end": config.EPSILON_END,
        "epsilon_decay": config.EPSILON_DECAY,
        "epsilon_warmup": config.EPSILON_WARMUP,
        "eval_every": config.EVAL_EVERY,
        "eval_episodes": config.EVAL_EPISODES,
        "eval_max_steps": config.EVAL_MAX_STEPS,
        "cave_seed": config.CRYSTAL_CAVES_SEED,
        "cave_difficulty": config.CRYSTAL_CAVES_DIFFICULTY,
        "cave_families": config.CRYSTAL_CAVES_FAMILIES,
        "cave_pool_size": config.CRYSTAL_CAVES_POOL_SIZE,
        "procedural": config.CRYSTAL_CAVES_PROCEDURAL,
        "drills": config.CRYSTAL_CAVES_DRILLS,
        "bridges": exp_config.CRYSTAL_CAVES_BRIDGES,
        "anti_loop_reward": exp_config.CRYSTAL_CAVES_ANTI_LOOP_REWARD,
        "first_crystal_goal": exp_config.CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL,
        "invalid_interact_penalty": exp_config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY,
        "invalid_shoot_penalty": exp_config.CRYSTAL_CAVES_INVALID_SHOOT_PENALTY,
        "novelty_bonus": exp_config.CRYSTAL_CAVES_NOVELTY_BONUS,
        "route_aux_loss": exp_config.CRYSTAL_CAVES_ROUTE_AUX_LOSS,
        "route_aux_weight": exp_config.CRYSTAL_CAVES_ROUTE_AUX_WEIGHT,
        "route_aux_deadband": exp_config.CRYSTAL_CAVES_ROUTE_AUX_DEADBAND,
        "demo_action_loss": exp_config.CRYSTAL_CAVES_DEMO_ACTION_LOSS,
        "demo_action_weight": exp_config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT,
        "demo_action_margin": exp_config.CRYSTAL_CAVES_DEMO_ACTION_MARGIN,
        "demo_action_batch_size": exp_config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE,
        "demo_conservative_weight": exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT,
        "demo_conservative_temperature": (exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE),
        "close_zone_demo_action_loss": exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS,
        "close_zone_demo_action_weight": exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT,
        "close_zone_demo_action_batch_size": (
            exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE
        ),
        "correction_action_loss": exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS,
        "correction_action_weight": exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT,
        "correction_action_margin": exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN,
        "correction_action_batch_size": exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE,
        "rich_state": config.CRYSTAL_CAVES_RICH_STATE,
        "cnn_state": config.USE_CNN_STATE,
        "force_cpu": config.FORCE_CPU,
        "progress_reward_scale": CrystalCaves.PROGRESS_REWARD_SCALE,
        "approach_reward_scale": CrystalCaves.APPROACH_REWARD_SCALE,
        "approach_reward_clip_pos": CrystalCaves.APPROACH_REWARD_CLIP_POS,
        "approach_reward_clip_neg": CrystalCaves.APPROACH_REWARD_CLIP_NEG,
        "target_best_approach_scale": CrystalCaves.TARGET_BEST_APPROACH_SCALE,
        "switch_throw_bonus": CrystalCaves.SWITCH_THROW_BONUS,
        "all_crystals_collected_bonus": CrystalCaves.ALL_CRYSTALS_COLLECTED_BONUS,
        "objective_region_bonus": CrystalCaves.OBJECTIVE_REGION_BONUS,
    }


def save_selected_weight_snapshot(
    path: Path,
    *,
    label: str,
    config_payload: dict[str, Any],
    state_size: int,
    action_size: int,
    selected_episode: int,
    source_eval: dict[str, Any] | None,
    weights: dict[str, dict[str, torch.Tensor]],
) -> str:
    """Persist only selected policy weights and metadata, never replay memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
            "label": label,
            "episode": int(selected_episode),
            "source_eval": source_eval or {},
            "config": config_payload,
            "state_size": int(state_size),
            "action_size": int(action_size),
            "weights": weights,
        },
        path,
    )
    return str(path)


def load_selected_weight_snapshot(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != SELECTED_WEIGHT_SNAPSHOT_KIND:
        raise ValueError(f"{path} is not a selected-weight checkpoint")
    weights = payload.get("weights")
    if not isinstance(weights, dict) or "policy" not in weights or "target" not in weights:
        raise ValueError(f"{path} does not contain policy/target weights")
    return payload


def append_near_miss_report_lines(
    lines: list[str],
    near_miss_eval: dict[str, Any],
    *,
    prefix: str,
) -> None:
    if not near_miss_eval:
        return
    rollup = near_miss_eval.get("rollup") or {}
    lines.extend(
        [
            f"- {prefix} near-miss: <=3 tiles "
            f"{100 * rollup.get('near_miss_rate_3', 0):.1f}%, "
            f"<=1.5 tiles {100 * rollup.get('near_miss_rate_1_5', 0):.1f}%, "
            f"min distance {rollup.get('mean_min_target_distance_tiles', 0):.2f} tiles",
            f"- {prefix} approach: best delta "
            f"{rollup.get('mean_target_distance_best_delta_tiles', 0):.2f} tiles, "
            f"final delta {rollup.get('mean_target_distance_final_delta_tiles', 0):.2f} tiles, "
            f"best step {rollup.get('mean_step_of_best_approach', 0):.0f}",
            f"- {prefix} close-zone behavior: "
            f"{rollup.get('mean_close_zone_steps', 0):.1f} steps near target, "
            f"{100 * rollup.get('mean_close_zone_jump_rate', 0):.1f}% jump, "
            f"{100 * rollup.get('mean_close_zone_idle_or_interact_rate', 0):.1f}% idle/interact, "
            f"{100 * rollup.get('stuck_after_close_rate', 0):.1f}% stuck-after-close, "
            f"{100 * rollup.get('loop_after_close_rate', 0):.1f}% loop-after-close",
            f"- {prefix} per-level near-miss rows: `{near_miss_eval.get('rows_path', '')}`",
        ]
    )


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = _report_header_lines(payload)
    for run in payload.get("runs", []):
        _append_run_report(lines, run)
    _append_comparison_lines(lines, payload.get("comparison"))
    path.write_text("\n".join(lines), encoding="utf-8")


def _report_header_lines(payload: dict[str, Any]) -> list[str]:
    return [
        "# Crystal Caves NN Status Session",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Created: `{payload['created_at']}`",
        "",
        "## Results",
        "",
    ]


def _append_run_report(lines: list[str], run: dict[str, Any]) -> None:
    _append_run_summary_lines(lines, run)
    _append_final_eval_lines(lines, run)
    _append_selected_policy_lines(lines, run)
    _append_route_checkpoint_lines(lines, run)
    _append_correction_lines(lines, run)
    _append_transfer_demo_lines(lines, run)
    _append_mixed_training_lines(lines, run)
    _append_failure_diagnostics_lines(lines, run)
    _append_selected_failure_diagnostics_lines(lines, run)
    _append_source_stats_table(lines, run)
    _append_bridge_eval_history_table(lines, run)
    lines.append("")
    _append_level_eval_table(lines, run)


def _append_run_summary_lines(lines: list[str], run: dict[str, Any]) -> None:
    lines.extend(
        [
            f"### {run['label']}",
            "",
            f"- Episodes: {run.get('episodes', 0)}",
            f"- Device: `{run.get('device', '')}`",
            f"- Train steps/sec: {run.get('steps_per_second', 0):.0f}",
            f"- Training score avg100: {run.get('avg_score_100', 0):.1f}",
            f"- Training progress avg100/best: {run.get('avg_progress_100', 0):.3f} / {run.get('best_progress', 0):.3f}",
        ]
    )
    run_config = run.get("config") or {}
    if run_config.get("first_crystal_goal"):
        lines.append("- Objective mode: first crystal terminal success")


def _append_final_eval_lines(lines: list[str], run: dict[str, Any]) -> None:
    final_eval_payload = run.get("final_eval") or {}
    if final_eval_payload:
        lines.extend(
            [
                f"- Final held-out wins: {final_eval_payload.get('wins', 0)}/{final_eval_payload.get('num_games', 0)} ({100 * final_eval_payload.get('win_rate', 0):.1f}%)",
                f"- Final held-out crystals/depth: {100 * final_eval_payload.get('mean_crystal_frac', 0):.1f}% / {100 * final_eval_payload.get('mean_depth_frac', 0):.1f}%",
                f"- Final held-out mean score: {final_eval_payload.get('mean_score', 0):.1f}",
                f"- Final held-out ends: `{final_eval_payload.get('end_reason_counts', {})}`",
            ]
        )
    append_near_miss_report_lines(
        lines,
        run.get("near_miss_eval") or {},
        prefix="Final held-out",
    )


def _append_selected_policy_lines(lines: list[str], run: dict[str, Any]) -> None:
    selected_bridge_rollup = run.get("selected_bridge_rollup")
    if selected_bridge_rollup:
        lines.extend(
            [
                f"- Selected bridge policy: ep {run.get('selected_bridge_episode', 0)} "
                f"({100 * selected_bridge_rollup.get('mean_win_rate', 0):.0f}% win, "
                f"{selected_bridge_rollup.get('solved_levels', 0)}/{selected_bridge_rollup.get('levels', 0)} solved)",
            ]
        )
    selected_source_eval = run.get("selected_source_eval") or {}
    if selected_source_eval:
        lines.append(
            f"- Selected source policy: ep {run.get('selected_source_episode', 0)} "
            f"({100 * selected_source_eval.get('win_rate', 0):.0f}% win, "
            f"{100 * selected_source_eval.get('mean_crystal_frac', 0):.0f}% crystals, "
            f"{100 * selected_source_eval.get('mean_depth_frac', 0):.0f}% depth)"
        )
    selected_checkpoint_eval = run.get("selected_checkpoint_eval") or {}
    if selected_checkpoint_eval:
        lines.extend(
            [
                f"- Selected checkpoint expanded eval: "
                f"{selected_checkpoint_eval.get('wins', 0)}/"
                f"{selected_checkpoint_eval.get('num_games', 0)} wins "
                f"({100 * selected_checkpoint_eval.get('win_rate', 0):.1f}%), "
                f"{100 * selected_checkpoint_eval.get('mean_crystal_frac', 0):.1f}% crystals, "
                f"{100 * selected_checkpoint_eval.get('mean_depth_frac', 0):.1f}% depth",
                f"- Selected checkpoint ends: "
                f"`{selected_checkpoint_eval.get('end_reason_counts', {})}`",
            ]
        )
    selected_checkpoint_path = run.get("selected_checkpoint_path")
    if selected_checkpoint_path:
        lines.append(f"- Selected checkpoint path: `{selected_checkpoint_path}`")
    append_near_miss_report_lines(
        lines,
        run.get("selected_checkpoint_near_miss_eval") or {},
        prefix="Selected checkpoint",
    )


def _append_route_checkpoint_lines(lines: list[str], run: dict[str, Any]) -> None:
    route_curriculum = run.get("route_curriculum") or {}
    if route_curriculum:
        scaffold = route_curriculum.get("route_scaffold_difficulty", "route_floor")
        pool_size = route_curriculum.get("cave_pool_size", 0)
        pool_text = f", pool {pool_size}" if pool_size else ""
        lines.append(
            f"- Route curriculum: {route_curriculum.get('route_floor_episodes', 0)} "
            f"{scaffold} episodes{pool_text} -> "
            f"{route_curriculum.get('tutorial_route_episodes', 0)} tutorial route episodes "
            f"(floor source {100 * route_curriculum.get('route_floor_source_win_rate', 0):.0f}% win, "
            f"{route_curriculum.get('route_floor_source_wins', 0)}/"
            f"{route_curriculum.get('route_floor_source_games', 0)})"
        )
    checkpoint_eval = run.get("checkpoint_eval") or {}
    if checkpoint_eval:
        source_eval = checkpoint_eval.get("source_eval") or {}
        lines.append(
            f"- Checkpoint eval source: {checkpoint_eval.get('source_label', 'unknown')} "
            f"ep {checkpoint_eval.get('source_episode', 0)} "
            f"({source_eval.get('wins', 0)}/{source_eval.get('num_games', 0)} source wins)"
        )


def _append_correction_lines(lines: list[str], run: dict[str, Any]) -> None:
    correction_dataset = run.get("correction_dataset") or {}
    if correction_dataset:
        lines.extend(
            [
                f"- Correction dataset: {correction_dataset.get('kept_examples', 0)} "
                f"kept states from {correction_dataset.get('games_completed', 0)} games "
                f"({100 * correction_dataset.get('disagreement_rate', 0):.1f}% policy/label disagreement)",
                f"- Correction triggers: `{correction_dataset.get('trigger_counts', {})}`",
                f"- Correction actions: `{correction_dataset.get('label_action_counts', {})}`",
                f"- Correction arrays: `{correction_dataset.get('states_path', '')}`",
                f"- Correction rows: `{correction_dataset.get('rows_path', '')}`",
            ]
        )
    correction_training = run.get("correction_training") or {}
    if correction_training:
        lines.extend(
            [
                f"- Correction action supervision: "
                f"{correction_training.get('correction_action_transitions', 0)} states, "
                f"weight {correction_training.get('weight', 0):.3f}, "
                f"margin {correction_training.get('margin', 0):.2f}, "
                f"batch {correction_training.get('batch_size', 0)}, "
                f"dataset `{correction_training.get('dataset_path', '')}`",
                f"- Correction action metrics avg100: "
                f"loss {run.get('avg_correction_action_loss_100', 0):.4f}, "
                f"accuracy {100 * run.get('avg_correction_action_accuracy_100', 0):.1f}%, "
                f"samples {run.get('correction_action_samples_100', 0)}",
            ]
        )


def _append_transfer_demo_lines(lines: list[str], run: dict[str, Any]) -> None:
    _append_route_direct_lines(lines, run)
    _append_transfer_source_lines(lines, run)
    _append_demo_replay_lines(lines, run)
    _append_route_demo_lines(lines, run)


def _append_route_direct_lines(lines: list[str], run: dict[str, Any]) -> None:
    route_direct = run.get("route_direct") or {}
    if route_direct:
        pool_size = route_direct.get("cave_pool_size", 0)
        pool_text = f", pool {pool_size}" if pool_size else ""
        lines.append(
            f"- Direct route training: {route_direct.get('difficulty', 'tutorial')}" f"{pool_text}"
        )


def _append_transfer_source_lines(lines: list[str], run: dict[str, Any]) -> None:
    transfer_source = run.get("transfer_source") or {}
    if not transfer_source:
        return
    rollup = transfer_source.get("selected_bridge_rollup") or {}
    source_kind = transfer_source.get("kind", "unknown")
    if source_kind == "first_crystal":
        lines.append(
            f"- Transfer source: first-crystal route policy "
            f"ep {transfer_source.get('source_episode', 0)} "
            f"({100 * transfer_source.get('source_win_rate', 0):.0f}% source held-out win)"
        )
    elif source_kind == "route_demo_bc":
        lines.append(
            f"- Transfer source: route-demo BC policy "
            f"ep {transfer_source.get('source_episode', 0)} "
            f"({100 * transfer_source.get('source_win_rate', 0):.0f}% route-floor source win)"
        )
    else:
        lines.extend(
            [
                f"- Transfer source: {source_kind} ep "
                f"{transfer_source.get('selected_bridge_episode', 0)} "
                f"({100 * rollup.get('mean_win_rate', 0):.0f}% bridge win)",
            ]
        )


def _append_demo_replay_lines(lines: list[str], run: dict[str, Any]) -> None:
    demo_replay = run.get("demo_replay") or {}
    if not demo_replay:
        return
    demo_summary = demo_replay.get("demo_summary") or {}
    seeded = demo_replay.get("seeded") or {}
    rollup = demo_replay.get("selected_bridge_rollup") or {}
    lines.extend(
        [
            f"- Demo replay source: {demo_replay.get('kind', 'unknown')} ep "
            f"{demo_replay.get('selected_bridge_episode', 0)} "
            f"({100 * rollup.get('mean_win_rate', 0):.0f}% bridge win)",
            f"- Demo replay collected: {demo_summary.get('kept_trajectories', 0)} trajectories, "
            f"{demo_summary.get('kept_transitions', 0)} transitions "
            f"from {demo_summary.get('wins', 0)}/{demo_summary.get('attempts', 0)} source wins",
            f"- Demo replay seeded: {seeded.get('pushed_transitions', 0)} pushed transitions "
            f"({seeded.get('repeat', 0)}x repeat), memory size {seeded.get('memory_size_after_seed', 0)}",
        ]
    )


def _append_route_demo_lines(lines: list[str], run: dict[str, Any]) -> None:
    route_demo = run.get("route_demo") or {}
    if not route_demo:
        return
    demo_summary = route_demo.get("demo_summary") or {}
    demo_selection = route_demo.get("demo_selection") or {}
    seeded = route_demo.get("seeded") or {}
    bc = route_demo.get("behavior_cloning") or {}
    bc_eval = route_demo.get("after_bc_eval") or {}
    lines.extend(
        [
            f"- Route demos collected: {demo_summary.get('kept_trajectories', 0)} kept / "
            f"{demo_summary.get('attempts', 0)} attempts "
            f"({100 * demo_summary.get('win_rate', 0):.0f}% scripted win), "
            f"{demo_summary.get('kept_transitions', 0)} transitions",
            f"- Route demo controllers: variants "
            f"`{demo_summary.get('controller_variants', [])}`, "
            f"{demo_summary.get('controller_attempts', demo_summary.get('attempts', 0))} "
            f"controller attempts, kept by variant "
            f"`{demo_summary.get('kept_by_variant', {})}`",
        ]
    )
    _append_oracle_close_zone_lines(lines, demo_summary)
    _append_demo_selection_lines(lines, demo_selection)
    lines.extend(
        [
            f"- Behavior cloning: {bc.get('updates', 0)} updates over "
            f"{bc.get('transitions', 0)} transitions, final CE {bc.get('final_loss', 0):.4f}",
            f"- Route demo replay seeded: {seeded.get('pushed_transitions', 0)} pushed transitions "
            f"({seeded.get('repeat', 0)}x repeat), memory size {seeded.get('memory_size_after_seed', 0)}",
        ]
    )
    if bc_eval:
        lines.append(
            f"- After-BC source eval: {bc_eval.get('wins', 0)}/"
            f"{bc_eval.get('num_games', 0)} wins "
            f"({100 * bc_eval.get('win_rate', 0):.0f}%), "
            f"{100 * bc_eval.get('mean_crystal_frac', 0):.0f}% crystals"
        )
    _append_online_action_supervision_lines(
        lines,
        route_demo.get("online_action_supervision") or {},
    )


def _append_oracle_close_zone_lines(lines: list[str], demo_summary: dict[str, Any]) -> None:
    if demo_summary.get("oracle_close_zone_kept_transitions", 0):
        lines.append(
            f"- Route demo oracle close-zone labels: "
            f"{demo_summary.get('oracle_close_zone_kept_transitions', 0)} transitions, "
            f"relabel rate "
            f"{100 * demo_summary.get('mean_kept_oracle_close_zone_relabel_rate', 0):.1f}%, "
            f"actions `{demo_summary.get('oracle_close_zone_action_counts', {})}`"
        )


def _append_demo_selection_lines(lines: list[str], demo_selection: dict[str, Any]) -> None:
    if demo_selection:
        lines.append(
            f"- Route demo selection: {demo_selection.get('mode', 'all')}, "
            f"{demo_selection.get('selected_unique_trajectories', 0)}/"
            f"{demo_selection.get('input_trajectories', 0)} unique kept, "
            f"{demo_selection.get('selected_weighted_trajectories', 0)} weighted trajectories, "
            f"{demo_selection.get('selected_transitions', 0)} training transitions, "
            f"excluded `{demo_selection.get('excluded_reasons', {})}`"
        )


def _append_online_action_supervision_lines(lines: list[str], online: dict[str, Any]) -> None:
    if not online.get("enabled"):
        return
    source = online.get("demo_action_source", "all_success")
    close_text = ""
    if source == "close_zone":
        close_text = (
            f", <= {online.get('close_zone_distance_tiles', 0):.1f} tiles, "
            f"{online.get('close_zone_available_transitions', 0)} available"
        )
    lines.append(
        f"- Online demo action supervision: {source}{close_text}, "
        f"weight {online.get('weight', 0):.3f}, "
        f"margin {online.get('margin', 0):.2f}, "
        f"conservative {online.get('conservative_weight', 0):.3f} "
        f"@T {online.get('conservative_temperature', 1):.2f}, "
        f"batch {online.get('batch_size', 0)}, "
        f"{online.get('demo_action_transitions', 0)} active transitions"
    )
    if online.get("close_zone_extra_enabled"):
        lines.append(
            "- Close-zone extra action supervision: "
            f"{online.get('close_zone_extra_source', 'scripted')}, "
            f"weight {online.get('close_zone_extra_weight', 0):.3f}, "
            f"batch {online.get('close_zone_extra_batch_size', 0)}, "
            f"{online.get('close_zone_extra_transitions', 0)} active transitions "
            f"within <= {online.get('close_zone_distance_tiles', 0):.1f} tiles"
        )


def _append_mixed_training_lines(lines: list[str], run: dict[str, Any]) -> None:
    if "interleave" in run:
        mix = run["interleave"]
        source = mix.get("skill_source", "skill")
        lines.extend(
            [
                f"- Interleave mix: {mix.get('full_envs', 0)} full / "
                f"{mix.get('skill_envs', 0)} {source} envs "
                f"({100 * mix.get('skill_ratio', 0):.0f}% {source} lanes)",
            ]
        )
    _append_reverse_start_lines(lines, run)
    _append_archive_start_lines(lines, run)


def _append_reverse_start_lines(lines: list[str], run: dict[str, Any]) -> None:
    if "reverse_start" not in run:
        return
    mix = run["reverse_start"]
    lines.extend(
        [
            f"- Reverse-start mix: {mix.get('full_envs', 0)} full / "
            f"{mix.get('reverse_envs', 0)} reverse envs "
            f"({100 * mix.get('reverse_ratio', 0):.0f}% reverse lanes)",
            f"- Reverse-start modes: `{mix.get('modes', {})}`",
        ]
    )
    reverse_stats = run.get("reverse_start_stats") or {}
    if reverse_stats:
        bits = [
            f"{source}: {stats.get('applied', 0)}/{stats.get('attempts', 0)}"
            for source, stats in sorted(reverse_stats.items())
        ]
        lines.append(f"- Reverse-start applied: {', '.join(bits)}")


def _append_archive_start_lines(lines: list[str], run: dict[str, Any]) -> None:
    if "archive_start" not in run:
        return
    mix = run["archive_start"]
    lines.extend(
        [
            f"- Archive-start mix: {mix.get('full_envs', 0)} full / "
            f"{mix.get('archive_envs', 0)} archive envs "
            f"({100 * mix.get('archive_ratio', 0):.0f}% archive lanes)",
            f"- Archive-start replay: {100 * mix.get('replay_prob', 0):.0f}% reset probability, "
            f"max {mix.get('max_size', 0)} states, min {mix.get('min_steps', 0)} steps",
        ]
    )
    archive_stats = run.get("archive_stats") or {}
    if archive_stats:
        lines.append(
            f"- Archive-start stats: size {archive_stats.get('size', 0)}/"
            f"{archive_stats.get('max_size', 0)}, stores {archive_stats.get('stores', 0)}, "
            f"replays {archive_stats.get('replays', 0)}/"
            f"{archive_stats.get('replay_attempts', 0)} "
            f"({100 * archive_stats.get('replay_rate', 0):.0f}%), "
            f"seen milestones {archive_stats.get('seen_milestones', 0)}"
        )


def _append_failure_diagnostics_lines(lines: list[str], run: dict[str, Any]) -> None:
    failure_diagnostics = run.get("failure_diagnostics") or {}
    if not failure_diagnostics:
        return
    rollup = failure_diagnostics.get("rollup") or {}
    lines.extend(
        [
            f"- Held-out trace: {rollup.get('wins', 0)}/{rollup.get('games', 0)} wins, "
            f"{100 * rollup.get('any_crystal_rate', 0):.1f}% any crystal, "
            f"{100 * rollup.get('mean_depth_frac', 0):.1f}% depth",
            f"- Trace failure modes: `{rollup.get('failure_mode_counts', {})}`",
            f"- Trace target distance delta: final {rollup.get('mean_target_distance_delta_tiles', 0):.2f} tiles, "
            f"best {rollup.get('mean_target_distance_best_delta_tiles', 0):.2f} tiles",
            f"- Trace anti-loop penalty avg: {rollup.get('mean_anti_loop_penalty_total', 0):.2f}",
            f"- Trace invalid interact avg: {rollup.get('mean_invalid_interact_count', 0):.1f} presses, "
            f"{rollup.get('mean_invalid_interact_penalty_total', 0):.2f} reward, "
            f"{100 * rollup.get('mean_interact_action_frac', 0):.1f}% actions",
            f"- Trace invalid shoot avg: {rollup.get('mean_invalid_shoot_count', 0):.1f} presses, "
            f"{rollup.get('mean_invalid_shoot_penalty_total', 0):.2f} reward, "
            f"{100 * rollup.get('mean_shoot_action_frac', 0):.1f}% actions",
            f"- Trace novelty bonus avg: {rollup.get('mean_novelty_bonus_total', 0):.2f}",
            f"- Trace files: `{failure_diagnostics.get('trace_dir', '')}`",
        ]
    )
    _append_trace_games_table(lines, failure_diagnostics.get("games_summary") or [])


def _append_trace_games_table(lines: list[str], games_summary: list[dict[str, Any]]) -> None:
    if not games_summary:
        return
    lines.append("")
    lines.append(
        "| Trace Game | End | Crystals | Progress | Depth | Target delta | Loop | Idle | Loop penalty | Top actions | Modes |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in games_summary[:8]:
        lines.append(
            f"| {row.get('game_index', 0)} | {row.get('end_reason', '')} | "
            f"{row.get('crystals_collected', 0)}/{row.get('initial_crystals', 0)} | "
            f"{row.get('final_progress', 0):.3f} | "
            f"{100 * row.get('final_depth_frac', 0):.0f}% | "
            f"{row.get('target_distance_delta_tiles', 0):.2f} | "
            f"{100 * row.get('max_tile_visit_frac', 0):.0f}% | "
            f"{100 * row.get('idle_action_frac', 0):.0f}% | "
            f"{row.get('anti_loop_penalty_total', 0):.2f} | "
            f"`{row.get('top_actions', {})}` | "
            f"`{row.get('failure_modes', [])}` |"
        )


def _append_selected_failure_diagnostics_lines(lines: list[str], run: dict[str, Any]) -> None:
    selected_failure_diagnostics = run.get("selected_checkpoint_failure_diagnostics") or {}
    if not selected_failure_diagnostics:
        return
    rollup = selected_failure_diagnostics.get("rollup") or {}
    lines.extend(
        [
            f"- Selected checkpoint trace: {rollup.get('wins', 0)}/"
            f"{rollup.get('games', 0)} wins, "
            f"{100 * rollup.get('any_crystal_rate', 0):.1f}% any crystal, "
            f"{100 * rollup.get('mean_depth_frac', 0):.1f}% depth",
            f"- Selected checkpoint trace modes: " f"`{rollup.get('failure_mode_counts', {})}`",
            f"- Selected checkpoint trace files: "
            f"`{selected_failure_diagnostics.get('trace_dir', '')}`",
        ]
    )


def _append_source_stats_table(lines: list[str], run: dict[str, Any]) -> None:
    source_stats = run.get("source_stats") or {}
    if not source_stats:
        return
    lines.append("")
    lines.append("| Source | Episodes | Score100 | Win100 | Progress100 | Ends |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for source, stats in sorted(source_stats.items()):
        lines.append(
            f"| {source} | {stats.get('episodes', 0)} | "
            f"{stats.get('avg_score_100', 0):.1f} | "
            f"{100 * stats.get('win_rate_100', 0):.0f}% | "
            f"{stats.get('avg_progress_100', 0):.3f} | "
            f"`{stats.get('end_reason_counts_100', {})}` |"
        )


def _append_bridge_eval_history_table(lines: list[str], run: dict[str, Any]) -> None:
    bridge_eval_history = run.get("bridge_eval_history") or []
    if not bridge_eval_history:
        return
    lines.append("")
    lines.append("| Bridge Eval Ep | Win | Any crystal | All crystals | Progress | Solved |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for snapshot in bridge_eval_history:
        rollup = snapshot.get("rollup") or {}
        levels = rollup.get("levels", 0)
        lines.append(
            f"| {snapshot.get('episode', 0)} | "
            f"{100 * rollup.get('mean_win_rate', 0):.0f}% | "
            f"{100 * rollup.get('mean_any_crystal_rate', 0):.0f}% | "
            f"{100 * rollup.get('mean_all_crystals_rate', 0):.0f}% | "
            f"{rollup.get('mean_progress', 0):.3f} | "
            f"{rollup.get('solved_levels', 0)}/{levels} |"
        )


def _append_level_eval_table(lines: list[str], run: dict[str, Any]) -> None:
    level_eval = run.get("drill_eval") or run.get("bridge_eval")
    if not level_eval:
        return
    label = "Drill" if "drill_eval" in run else "Bridge"
    lines.append(f"| {label} | Win | Any crystal | All crystals | Progress | Ends |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in level_eval:
        lines.append(
            f"| {row['name']} | {100 * row['win_rate']:.0f}% | "
            f"{100 * row['collected_any_rate']:.0f}% | "
            f"{100 * row['all_crystals_rate']:.0f}% | "
            f"{row['mean_progress']:.3f} | `{row['end_reason_counts']}` |"
        )
    lines.append("")


def _append_comparison_lines(lines: list[str], comparison: Any) -> None:
    if comparison:
        lines.extend(["## Comparison", ""])
        for key, value in comparison.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
