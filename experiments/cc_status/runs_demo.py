# ruff: noqa: F401,F403,F405,I001
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _SourceTrainingResult:
    train_seconds: float
    history: list[dict[str, Any]]
    eval_payload: dict[str, Any]
    best_snapshot: dict[str, Any]
    best_weights: dict[str, dict[str, torch.Tensor]]


@dataclass(frozen=True)
class _SelectedCheckpointResult:
    eval_payload: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    near_miss_eval: dict[str, Any] | None = None
    checkpoint_path: str | None = None


def _run_source_snapshot_training(
    trainer: HeadlessTrainer,
    config: Config,
    *,
    run_dir: Path,
    label: str,
    total_episodes: int,
    heartbeat_seconds: float,
    source_eval_every: int,
    eval_games: int,
    initial_snapshot: dict[str, Any],
    initial_weights: dict[str, dict[str, torch.Tensor]],
) -> _SourceTrainingResult:
    train_seconds, history, training_best_snapshot, training_best_weights = (
        run_training_with_source_snapshots(
            trainer,
            config,
            run_dir=run_dir,
            label=label,
            total_episodes=total_episodes,
            heartbeat_seconds=heartbeat_seconds,
            source_eval_every=source_eval_every,
            eval_games=eval_games,
        )
    )
    best_snapshot = initial_snapshot
    best_weights = initial_weights
    if training_best_snapshot is not None and source_snapshot_score(
        training_best_snapshot
    ) > source_snapshot_score(initial_snapshot):
        best_snapshot = training_best_snapshot
        best_weights = training_best_weights or capture_weight_snapshot(trainer.agent)

    if history:
        eval_payload = history[-1]["source_eval"]
    else:
        eval_payload = initial_snapshot["source_eval"]

    return _SourceTrainingResult(
        train_seconds=train_seconds,
        history=history,
        eval_payload=eval_payload,
        best_snapshot=best_snapshot,
        best_weights=best_weights,
    )


def _run_final_diagnostics(
    config: Config,
    agent: Any,
    *,
    run_dir: Path,
    label: str,
    episode: int,
    eval_games: int,
    trace_games: int,
    trace_max_steps: int,
    trace_sample_every: int,
    trace_tail_steps: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    diagnostics = trace_heldout_failures(
        config,
        agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        games=trace_games,
        max_steps=trace_max_steps,
        sample_every=trace_sample_every,
        tail_steps=trace_tail_steps,
    )
    near_miss_eval = first_objective_near_miss_eval(
        config,
        agent,
        out_dir=run_dir,
        label=f"{label}_final",
        episode=episode,
        games=eval_games,
        max_steps=config.EVAL_MAX_STEPS,
    )
    return diagnostics, near_miss_eval


def _evaluate_selected_checkpoint(
    config: Config,
    trainer: HeadlessTrainer,
    *,
    run_dir: Path,
    label: str,
    best_snapshot: dict[str, Any],
    best_weights: dict[str, dict[str, torch.Tensor]] | None,
    selected_eval_games: int,
    trace_games: int,
    trace_max_steps: int,
    trace_sample_every: int,
    trace_tail_steps: int,
    checkpoint_path: Path | None = None,
) -> _SelectedCheckpointResult:
    if selected_eval_games <= 0 or best_weights is None:
        return _SelectedCheckpointResult()

    final_weights = capture_weight_snapshot(trainer.agent)
    selected_episode = int(best_snapshot.get("episode", trainer.current_episode) or 0)
    selected_checkpoint_path: str | None = None
    load_weight_snapshot(trainer.agent, best_weights)
    try:
        if checkpoint_path is not None:
            selected_checkpoint_path = save_selected_weight_snapshot(
                checkpoint_path,
                label=label,
                config_payload=config_snapshot(config),
                state_size=trainer.agent.state_size,
                action_size=trainer.agent.action_size,
                selected_episode=selected_episode,
                source_eval=best_snapshot.get("source_eval"),
                weights=best_weights,
            )
        selected_eval_payload = final_eval(
            config,
            trainer.agent,
            out_dir=run_dir,
            label=f"{label}_selected_ep{selected_episode}",
            episode=selected_episode,
            games=selected_eval_games,
        )
        selected_diagnostics = trace_heldout_failures(
            config,
            trainer.agent,
            out_dir=run_dir,
            label=f"{label}_selected_ep{selected_episode}_heldout",
            games=trace_games,
            max_steps=trace_max_steps,
            sample_every=trace_sample_every,
            tail_steps=trace_tail_steps,
        )
        selected_near_miss_eval = first_objective_near_miss_eval(
            config,
            trainer.agent,
            out_dir=run_dir,
            label=f"{label}_selected_ep{selected_episode}",
            episode=selected_episode,
            games=selected_eval_games,
            max_steps=config.EVAL_MAX_STEPS,
        )
        return _SelectedCheckpointResult(
            eval_payload=selected_eval_payload,
            diagnostics=selected_diagnostics,
            near_miss_eval=selected_near_miss_eval,
            checkpoint_path=selected_checkpoint_path,
        )
    finally:
        load_weight_snapshot(trainer.agent, final_weights)


def run_tutorial_demo_bc(
    out_dir: Path,
    *,
    episodes: int,
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
    save_selected_checkpoint: bool,
    cave_pool_size: int | None,
    selected_eval_games: int,
    history_state: bool,
    history_steps: int,
    geo_compass: bool = False,
    geo_compass_hazard_aware: bool = False,
    geodesic_potential: bool = False,
    geodesic_potential_weight: float = 0.3,
    show_locked_exit: bool = False,
    reverse_curriculum_p: float = 0.0,
    reward_clip: float | None = None,
    stall_window: int | None = None,
    distributional_dqn: bool,
    c51_atoms: int,
    c51_v_min: float,
    c51_v_max: float,
    route_demo_levels: int,
    route_demo_max_steps: int,
    route_demo_variants: tuple[str, ...],
    demo_selection_mode: str,
    bc_epochs: int,
    bc_batch_size: int,
    demo_repeat: int,
    demo_action_weight: float = 0.0,
    demo_action_margin: float = 0.8,
    demo_action_batch_size: int = 64,
    demo_conservative_weight: float = 0.0,
    demo_conservative_temperature: float = 1.0,
    close_zone_demo_action: bool = False,
    close_zone_extra_demo_action_weight: float = 0.0,
    close_zone_extra_label_source: str = "scripted",
    close_zone_demo_distance: float = CLOSE_ZONE_DISTANCE_TILES,
    oracle_close_zone_stride: int = 4,
    oracle_close_zone_max_per_trajectory: int = 8,
    invalid_shoot_penalty: bool = False,
    label: str = "tutorial_demo_bc",
) -> dict[str, Any]:
    if close_zone_demo_distance <= 0:
        raise ValueError("close_zone_demo_distance must be positive")
    if close_zone_extra_label_source not in DEMO_CLOSE_ZONE_EXTRA_LABEL_SOURCES:
        raise ValueError(
            f"close_zone_extra_label_source must be one of "
            f"{sorted(DEMO_CLOSE_ZONE_EXTRA_LABEL_SOURCES)}"
        )
    set_seed(seed)
    run_dir = out_dir / label
    config = first_crystal_config(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
        difficulty="tutorial",
    )
    apply_cave_pool_override(config, cave_pool_size)
    apply_history_state_override(
        config,
        history_state=history_state,
        history_steps=history_steps,
    )
    apply_distributional_dqn_override(
        config,
        distributional_dqn=distributional_dqn,
        c51_atoms=c51_atoms,
        c51_v_min=c51_v_min,
        c51_v_max=c51_v_max,
    )
    apply_reward_shaping_override(
        config,
        geodesic_potential=geodesic_potential,
        geodesic_potential_weight=geodesic_potential_weight,
        show_locked_exit=show_locked_exit,
        reverse_curriculum_p=reverse_curriculum_p,
        reward_clip=reward_clip,
        stall_window=stall_window,
    )
    apply_geo_compass_override(
        config,
        geo_compass=geo_compass,
        hazard_aware=geo_compass_hazard_aware,
    )
    exp_config = cc_experiment_config(config)
    exp_config.CRYSTAL_CAVES_INVALID_SHOOT_PENALTY = invalid_shoot_penalty
    apply_demo_action_override(
        config,
        demo_action_weight=demo_action_weight,
        demo_action_margin=demo_action_margin,
        demo_action_batch_size=demo_action_batch_size,
        demo_conservative_weight=demo_conservative_weight,
        demo_conservative_temperature=demo_conservative_temperature,
    )
    apply_close_zone_demo_action_override(
        config,
        close_zone_demo_action_weight=close_zone_extra_demo_action_weight,
        close_zone_demo_action_batch_size=demo_action_batch_size,
    )
    demos = collect_scripted_route_demonstrations(
        config,
        max_levels=route_demo_levels,
        max_steps=route_demo_max_steps,
        close_zone_distance_tiles=close_zone_demo_distance,
        controller_variants=route_demo_variants,
        oracle_close_zone_labels=close_zone_extra_label_source == "oracle",
        oracle_close_zone_stride=oracle_close_zone_stride,
        oracle_close_zone_max_per_trajectory=oracle_close_zone_max_per_trajectory,
    )
    demo_summary = demos["summary"]
    write_json(run_dir / "demo_summary.json", demo_summary)
    selected_demos = select_route_demo_trajectories(
        demos,
        mode=demo_selection_mode,
    )
    demo_selection_summary = selected_demos["summary"]
    write_json(run_dir / "demo_selection_summary.json", demo_selection_summary)

    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    bc = behavior_clone_from_demonstrations(
        trainer.agent,
        selected_demos["trajectories"],
        epochs=bc_epochs,
        batch_size=bc_batch_size,
    )
    seeded = seed_replay_from_demonstrations(
        trainer.agent,
        selected_demos["trajectories"],
        repeat=demo_repeat,
    )
    demo_action_dataset: dict[str, Any] = {
        "demo_action_transitions": 0,
        "demo_action_source": "none",
        "close_zone_distance_tiles": close_zone_demo_distance,
        "close_zone_available_transitions": int(
            demo_selection_summary.get("selected_close_zone_transitions", 0) or 0
        ),
        "close_zone_extra_enabled": exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS,
        "close_zone_extra_weight": exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT,
        "close_zone_extra_batch_size": (exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE),
        "close_zone_extra_source": close_zone_extra_label_source,
        "close_zone_extra_transitions": 0,
    }
    if exp_config.CRYSTAL_CAVES_DEMO_ACTION_LOSS:
        demo_action_source = "close_zone" if close_zone_demo_action else "all_success"
        action_trajectories = (
            selected_demos["close_zone_trajectories"]
            if close_zone_demo_action
            else selected_demos["trajectories"]
        )
        demo_states, demo_actions = demo_action_arrays(action_trajectories)
        if len(demo_actions) > 0:
            demo_action_dataset.update(
                trainer.agent.set_demo_action_dataset(demo_states, demo_actions)
            )
        demo_action_dataset["demo_action_source"] = demo_action_source
    if exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS:
        close_zone_source_key = (
            "oracle_close_zone_trajectories"
            if close_zone_extra_label_source == "oracle"
            else "close_zone_trajectories"
        )
        close_states, close_actions = demo_action_arrays(selected_demos[close_zone_source_key])
        if len(close_actions) > 0:
            close_summary = trainer.agent.set_close_zone_demo_action_dataset(
                close_states, close_actions
            )
            demo_action_dataset["close_zone_extra_transitions"] = int(
                close_summary.get("close_zone_demo_action_transitions", 0)
            )
    after_bc_eval = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_after_bc",
        episode=0,
        games=eval_games,
    )
    best_snapshot: dict[str, Any] = {"episode": 0, "source_eval": after_bc_eval}
    best_weights = capture_weight_snapshot(trainer.agent)

    print(
        f"Tutorial demos: {demo_summary.get('kept_trajectories', 0)}/"
        f"{demo_summary.get('attempts', 0)} scripted wins, "
        f"{demo_summary.get('kept_transitions', 0)} raw transitions; "
        f"selected {demo_selection_summary.get('selected_unique_trajectories', 0)} unique / "
        f"{demo_selection_summary.get('selected_weighted_trajectories', 0)} weighted, "
        f"{demo_selection_summary.get('selected_transitions', 0)} train transitions; "
        f"BC updates {bc.get('updates', 0)}, after-BC source win "
        f"{100 * after_bc_eval.get('win_rate', 0):.0f}%",
        flush=True,
    )

    source_training = _run_source_snapshot_training(
        trainer,
        config,
        run_dir=run_dir,
        label=label,
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
        source_eval_every=eval_every,
        eval_games=eval_games,
        initial_snapshot=best_snapshot,
        initial_weights=best_weights,
    )
    best_snapshot = source_training.best_snapshot
    best_weights = source_training.best_weights
    diagnostics, near_miss_eval = _run_final_diagnostics(
        config,
        trainer.agent,
        run_dir=run_dir,
        label=label,
        episode=trainer.current_episode,
        eval_games=eval_games,
        trace_games=trace_games,
        trace_max_steps=trace_max_steps,
        trace_sample_every=trace_sample_every,
        trace_tail_steps=trace_tail_steps,
    )
    selected_episode = int(best_snapshot.get("episode", trainer.current_episode) or 0)
    selected_checkpoint_target = (
        Path(config.GAME_MODEL_DIR) / f"{label}_selected_ep{selected_episode}.pth"
        if save_checkpoints or save_selected_checkpoint
        else None
    )
    selected_checkpoint = _evaluate_selected_checkpoint(
        config,
        trainer,
        run_dir=run_dir,
        label=label,
        best_snapshot=best_snapshot,
        best_weights=best_weights,
        selected_eval_games=selected_eval_games,
        trace_games=trace_games,
        trace_max_steps=trace_max_steps,
        trace_sample_every=trace_sample_every,
        trace_tail_steps=trace_tail_steps,
        checkpoint_path=selected_checkpoint_target,
    )

    checkpoint = Path(config.GAME_MODEL_DIR) / "crystal_caves_final.pth"
    extra: dict[str, Any] = {
        "source_eval_history": [{"episode": 0, "source_eval": after_bc_eval}]
        + source_training.history,
        "selected_source_episode": best_snapshot.get("episode"),
        "selected_source_eval": best_snapshot.get("source_eval"),
        "checkpoint": str(checkpoint) if save_checkpoints else "in-memory only",
        "route_direct": {
            "difficulty": "tutorial",
            "cave_pool_size": cave_pool_size
            or int(config_snapshot(config).get("cave_pool_size", 0) or 0),
            "history_state": exp_config.CRYSTAL_CAVES_HISTORY_STATE,
            "history_steps": exp_config.CRYSTAL_CAVES_HISTORY_STEPS,
            "distributional_dqn": config.USE_DISTRIBUTIONAL_DQN,
            "c51_atoms": config.C51_NUM_ATOMS,
            "c51_v_min": config.C51_V_MIN,
            "c51_v_max": config.C51_V_MAX,
        },
        "route_demo": {
            "kind": "scripted_tutorial_route",
            "demo_summary": demo_summary,
            "demo_summary_path": str(run_dir / "demo_summary.json"),
            "demo_selection": demo_selection_summary,
            "demo_selection_summary_path": str(run_dir / "demo_selection_summary.json"),
            "behavior_cloning": bc,
            "seeded": seeded,
            "online_action_supervision": {
                "enabled": exp_config.CRYSTAL_CAVES_DEMO_ACTION_LOSS,
                "weight": exp_config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT,
                "margin": exp_config.CRYSTAL_CAVES_DEMO_ACTION_MARGIN,
                "batch_size": exp_config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE,
                "conservative_weight": exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT,
                "conservative_temperature": (
                    exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE
                ),
                **demo_action_dataset,
            },
            "after_bc_eval": after_bc_eval,
        },
        "failure_diagnostics": diagnostics,
        "near_miss_eval": near_miss_eval,
    }
    if selected_checkpoint.eval_payload is not None:
        extra["selected_checkpoint_eval"] = selected_checkpoint.eval_payload
        extra["selected_checkpoint_eval_games"] = selected_eval_games
    if selected_checkpoint.checkpoint_path is not None:
        extra["selected_checkpoint_path"] = selected_checkpoint.checkpoint_path
    if selected_checkpoint.diagnostics is not None:
        extra["selected_checkpoint_failure_diagnostics"] = selected_checkpoint.diagnostics
    if selected_checkpoint.near_miss_eval is not None:
        extra["selected_checkpoint_near_miss_eval"] = selected_checkpoint.near_miss_eval
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=source_training.train_seconds,
        final_eval_payload=source_training.eval_payload,
        extra=extra,
    )


def run_tutorial_demo_bridge_finetune(
    out_dir: Path,
    *,
    episodes: int,
    bridge_finetune_episodes: int,
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
    cave_pool_size: int | None,
    selected_eval_games: int,
    route_demo_levels: int,
    route_demo_max_steps: int,
    route_demo_variants: tuple[str, ...],
    bc_epochs: int,
    bc_batch_size: int,
    demo_repeat: int,
    bridge_ratio: float,
    bridge_envs_override: int | None,
    label: str = "tutorial_demo_bridge_finetune",
) -> list[dict[str, Any]]:
    """Train B3g-style route demo BC, then lightly fine-tune with bridge lanes."""
    if bridge_finetune_episodes <= 0:
        raise ValueError("bridge_finetune_episodes must be positive")

    set_seed(seed)
    route_label = f"{label}_route"
    route_run_dir = out_dir / route_label
    route_config = first_crystal_config(
        route_run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
        difficulty="tutorial",
    )
    apply_cave_pool_override(route_config, cave_pool_size)
    demos = collect_scripted_route_demonstrations(
        route_config,
        max_levels=route_demo_levels,
        max_steps=route_demo_max_steps,
        controller_variants=route_demo_variants,
    )
    demo_summary = demos["summary"]
    write_json(route_run_dir / "demo_summary.json", demo_summary)

    route_trainer = prepare_trainer(
        route_config,
        episodes=episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    bc = behavior_clone_from_demonstrations(
        route_trainer.agent,
        demos["trajectories"],
        epochs=bc_epochs,
        batch_size=bc_batch_size,
    )
    seeded = seed_replay_from_demonstrations(
        route_trainer.agent,
        demos["trajectories"],
        repeat=demo_repeat,
    )
    after_bc_eval = final_eval(
        route_config,
        route_trainer.agent,
        out_dir=route_run_dir,
        label=f"{route_label}_after_bc",
        episode=0,
        games=eval_games,
    )
    route_best_snapshot: dict[str, Any] = {"episode": 0, "source_eval": after_bc_eval}
    route_best_weights = capture_weight_snapshot(route_trainer.agent)

    print(
        f"Two-stage route demos: {demo_summary.get('kept_trajectories', 0)}/"
        f"{demo_summary.get('attempts', 0)} scripted wins, "
        f"{demo_summary.get('kept_transitions', 0)} transitions; "
        f"BC updates {bc.get('updates', 0)}, after-BC source win "
        f"{100 * after_bc_eval.get('win_rate', 0):.0f}%",
        flush=True,
    )

    route_source_training = _run_source_snapshot_training(
        route_trainer,
        route_config,
        run_dir=route_run_dir,
        label=route_label,
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
        source_eval_every=eval_every,
        eval_games=eval_games,
        initial_snapshot=route_best_snapshot,
        initial_weights=route_best_weights,
    )
    route_best_snapshot = route_source_training.best_snapshot
    route_best_weights = route_source_training.best_weights
    route_diagnostics, route_near_miss_eval = _run_final_diagnostics(
        route_config,
        route_trainer.agent,
        run_dir=route_run_dir,
        label=route_label,
        episode=route_trainer.current_episode,
        eval_games=eval_games,
        trace_games=trace_games,
        trace_max_steps=trace_max_steps,
        trace_sample_every=trace_sample_every,
        trace_tail_steps=trace_tail_steps,
    )
    route_exp_config = cc_experiment_config(route_config)
    route_summary = summarize_trainer(
        route_trainer,
        label=route_label,
        train_seconds=route_source_training.train_seconds,
        final_eval_payload=route_source_training.eval_payload,
        extra={
            "source_eval_history": [{"episode": 0, "source_eval": after_bc_eval}]
            + route_source_training.history,
            "selected_source_episode": route_best_snapshot.get("episode"),
            "selected_source_eval": route_best_snapshot.get("source_eval"),
            "checkpoint": (
                "in-memory only"
                if not save_checkpoints
                else str(Path(route_config.GAME_MODEL_DIR) / "crystal_caves_final.pth")
            ),
            "route_direct": {
                "difficulty": "tutorial",
                "cave_pool_size": cave_pool_size
                or int(config_snapshot(route_config).get("cave_pool_size", 0) or 0),
            },
            "route_demo": {
                "kind": "scripted_tutorial_route",
                "demo_summary": demo_summary,
                "demo_summary_path": str(route_run_dir / "demo_summary.json"),
                "behavior_cloning": bc,
                "seeded": seeded,
                "online_action_supervision": {
                    "enabled": False,
                    "weight": 0.0,
                    "margin": route_exp_config.CRYSTAL_CAVES_DEMO_ACTION_MARGIN,
                    "batch_size": route_exp_config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE,
                    "demo_action_transitions": 0,
                    "demo_action_source": "none",
                    "close_zone_distance_tiles": CLOSE_ZONE_DISTANCE_TILES,
                    "close_zone_available_transitions": int(
                        demo_summary.get("close_zone_kept_transitions", 0) or 0
                    ),
                },
                "after_bc_eval": after_bc_eval,
            },
            "failure_diagnostics": route_diagnostics,
            "near_miss_eval": route_near_miss_eval,
            "stage": "route_demo_bc",
        },
    )

    set_seed(seed)
    finetune_label = f"{label}_bridge_finetune"
    finetune_run_dir = out_dir / finetune_label
    full_envs, skill_envs = interleave_counts(
        vec_envs=vec_envs,
        skill_ratio=bridge_ratio,
        skill_envs=bridge_envs_override,
    )
    finetune_config = first_crystal_config(
        finetune_run_dir,
        episodes=bridge_finetune_episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
        difficulty="tutorial",
    )
    apply_cave_pool_override(finetune_config, cave_pool_size)
    bridge_config = make_interleaved_bridge_config(finetune_config)
    cc_experiment_config(bridge_config).CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL = False
    finetune_trainer = prepare_trainer(
        finetune_config,
        episodes=bridge_finetune_episodes,
        vec_envs=vec_envs,
        transfer_weights=route_best_weights,
        save_checkpoints=save_checkpoints,
    )
    install_interleaved_vec_env(
        finetune_trainer,
        run_dir=finetune_run_dir,
        full_envs=full_envs,
        skill_envs=skill_envs,
        skill_source="bridge",
        skill_config=bridge_config,
    )
    print(
        f"Two-stage bridge fine-tune: transfer ep {route_best_snapshot.get('episode')} "
        f"with {full_envs} route envs + {skill_envs} bridge envs "
        f"({skill_envs / (full_envs + skill_envs):.0%} bridge)",
        flush=True,
    )

    initial_eval = final_eval(
        finetune_config,
        finetune_trainer.agent,
        out_dir=finetune_run_dir,
        label=f"{finetune_label}_initial",
        episode=0,
        games=eval_games,
    )
    finetune_best_snapshot: dict[str, Any] = {
        "episode": 0,
        "source_eval": initial_eval,
    }
    finetune_best_weights = capture_weight_snapshot(finetune_trainer.agent)
    finetune_source_training = _run_source_snapshot_training(
        finetune_trainer,
        finetune_config,
        run_dir=finetune_run_dir,
        label=finetune_label,
        total_episodes=bridge_finetune_episodes,
        heartbeat_seconds=heartbeat_seconds,
        source_eval_every=eval_every,
        eval_games=eval_games,
        initial_snapshot=finetune_best_snapshot,
        initial_weights=finetune_best_weights,
    )
    finetune_best_snapshot = finetune_source_training.best_snapshot
    finetune_best_weights = finetune_source_training.best_weights
    finetune_diagnostics, finetune_near_miss_eval = _run_final_diagnostics(
        finetune_config,
        finetune_trainer.agent,
        run_dir=finetune_run_dir,
        label=finetune_label,
        episode=finetune_trainer.current_episode,
        eval_games=eval_games,
        trace_games=trace_games,
        trace_max_steps=trace_max_steps,
        trace_sample_every=trace_sample_every,
        trace_tail_steps=trace_tail_steps,
    )
    selected_checkpoint = _evaluate_selected_checkpoint(
        finetune_config,
        finetune_trainer,
        run_dir=finetune_run_dir,
        label=finetune_label,
        best_snapshot=finetune_best_snapshot,
        best_weights=finetune_best_weights,
        selected_eval_games=selected_eval_games,
        trace_games=trace_games,
        trace_max_steps=trace_max_steps,
        trace_sample_every=trace_sample_every,
        trace_tail_steps=trace_tail_steps,
    )

    finetune_extra: dict[str, Any] = {
        "source_eval_history": [{"episode": 0, "source_eval": initial_eval}]
        + finetune_source_training.history,
        "selected_source_episode": finetune_best_snapshot.get("episode"),
        "selected_source_eval": finetune_best_snapshot.get("source_eval"),
        "transfer_source": {
            "kind": "tutorial_demo_bc",
            "source_episode": int(route_best_snapshot.get("episode", 0) or 0),
            "source_win_rate": float(
                (route_best_snapshot.get("source_eval") or {}).get("win_rate", 0.0) or 0.0
            ),
            "source_wins": int((route_best_snapshot.get("source_eval") or {}).get("wins", 0) or 0),
            "source_games": int(
                (route_best_snapshot.get("source_eval") or {}).get("num_games", 0) or 0
            ),
        },
        "interleave": {
            "full_envs": full_envs,
            "skill_envs": skill_envs,
            "skill_source": "bridge",
            "skill_ratio": skill_envs / (full_envs + skill_envs),
            "cave_pool_size": cave_pool_size
            or int(config_snapshot(finetune_config).get("cave_pool_size", 0) or 0),
            "full_lane_goal": "first_crystal",
            "bridge_lane_goal": "full_bridge",
        },
        "failure_diagnostics": finetune_diagnostics,
        "near_miss_eval": finetune_near_miss_eval,
        "stage": "bridge_finetune_after_tutorial_demo_bc",
        "bridge_finetune_episodes": bridge_finetune_episodes,
    }
    if selected_checkpoint.eval_payload is not None:
        finetune_extra["selected_checkpoint_eval"] = selected_checkpoint.eval_payload
        finetune_extra["selected_checkpoint_eval_games"] = selected_eval_games
    if selected_checkpoint.diagnostics is not None:
        finetune_extra["selected_checkpoint_failure_diagnostics"] = selected_checkpoint.diagnostics
    if selected_checkpoint.near_miss_eval is not None:
        finetune_extra["selected_checkpoint_near_miss_eval"] = selected_checkpoint.near_miss_eval
    finetune_summary = summarize_trainer(
        finetune_trainer,
        label=finetune_label,
        train_seconds=finetune_source_training.train_seconds,
        final_eval_payload=finetune_source_training.eval_payload,
        extra=finetune_extra,
    )
    return [route_summary, finetune_summary]
