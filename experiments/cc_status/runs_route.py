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


def run_first_crystal_route_curriculum(
    out_dir: Path,
    *,
    episodes: int,
    route_floor_episodes: int,
    route_scaffold_difficulty: str,
    cave_pool_size: int | None,
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
    selected_eval_games: int = 0,
) -> list[dict[str, Any]]:
    set_seed(seed)
    floor_run_dir = out_dir / f"{route_scaffold_difficulty}_pretrain"
    floor_config = first_crystal_config(
        floor_run_dir,
        episodes=route_floor_episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
        difficulty=route_scaffold_difficulty,
    )
    apply_cave_pool_override(floor_config, cave_pool_size)
    floor_trainer = prepare_trainer(
        floor_config,
        episodes=route_floor_episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    floor_train_seconds, floor_history, floor_best_snapshot, floor_best_weights = (
        run_training_with_source_snapshots(
            floor_trainer,
            floor_config,
            run_dir=floor_run_dir,
            label=f"{route_scaffold_difficulty}_pretrain",
            total_episodes=route_floor_episodes,
            heartbeat_seconds=heartbeat_seconds,
            source_eval_every=eval_every,
            eval_games=eval_games,
        )
    )
    if floor_history:
        floor_eval = floor_history[-1]["source_eval"]
    else:
        floor_eval = final_eval(
            floor_config,
            floor_trainer.agent,
            out_dir=floor_run_dir,
            label=f"{route_scaffold_difficulty}_pretrain_final",
            episode=floor_trainer.current_episode,
            games=eval_games,
        )
    if floor_best_snapshot is None:
        floor_best_snapshot = {
            "episode": int(floor_trainer.current_episode),
            "source_eval": floor_eval,
        }
    if floor_best_weights is None:
        floor_best_weights = capture_weight_snapshot(floor_trainer.agent)
    floor_summary = summarize_trainer(
        floor_trainer,
        label=f"{route_scaffold_difficulty}_pretrain",
        train_seconds=floor_train_seconds,
        final_eval_payload=floor_eval,
        extra={
            "source_eval_history": floor_history,
            "selected_source_episode": floor_best_snapshot.get("episode"),
            "selected_source_eval": floor_best_snapshot.get("source_eval"),
            "checkpoint": "in-memory only",
        },
    )
    weights = floor_best_weights

    set_seed(seed)
    run_dir = out_dir / "first_crystal_route"
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
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=weights,
        save_checkpoints=save_checkpoints,
    )
    train_seconds, source_eval_history, best_snapshot, best_weights = (
        run_training_with_source_snapshots(
            trainer,
            config,
            run_dir=run_dir,
            label="first_crystal_route",
            total_episodes=episodes,
            heartbeat_seconds=heartbeat_seconds,
            source_eval_every=eval_every,
            eval_games=eval_games,
        )
    )
    if source_eval_history:
        eval_payload = source_eval_history[-1]["source_eval"]
    else:
        eval_payload = final_eval(
            config,
            trainer.agent,
            out_dir=run_dir,
            label="first_crystal_route_final",
            episode=trainer.current_episode,
            games=eval_games,
        )
    if best_snapshot is None:
        best_snapshot = {
            "episode": int(trainer.current_episode),
            "source_eval": eval_payload,
        }
    selected_eval_payload: dict[str, Any] | None = None
    selected_diagnostics: dict[str, Any] | None = None
    if selected_eval_games > 0 and best_weights is not None:
        final_weights = capture_weight_snapshot(trainer.agent)
        selected_episode = int(best_snapshot.get("episode", trainer.current_episode) or 0)
        load_weight_snapshot(trainer.agent, best_weights)
        selected_eval_payload = final_eval(
            config,
            trainer.agent,
            out_dir=run_dir,
            label=f"first_crystal_route_selected_ep{selected_episode}",
            episode=selected_episode,
            games=selected_eval_games,
        )
        selected_diagnostics = trace_heldout_failures(
            config,
            trainer.agent,
            out_dir=run_dir,
            label=f"first_crystal_route_selected_ep{selected_episode}_heldout",
            games=trace_games,
            max_steps=trace_max_steps,
            sample_every=trace_sample_every,
            tail_steps=trace_tail_steps,
        )
        load_weight_snapshot(trainer.agent, final_weights)
    diagnostics = trace_heldout_failures(
        config,
        trainer.agent,
        out_dir=run_dir,
        label="first_crystal_route_heldout",
        games=trace_games,
        max_steps=trace_max_steps,
        sample_every=trace_sample_every,
        tail_steps=trace_tail_steps,
    )
    floor_eval = floor_summary.get("selected_source_eval") or floor_summary.get("final_eval") or {}
    extra: dict[str, Any] = {
        "source_eval_history": source_eval_history,
        "selected_source_episode": best_snapshot.get("episode"),
        "selected_source_eval": best_snapshot.get("source_eval"),
        "route_curriculum": {
            "route_floor_episodes": route_floor_episodes,
            "route_scaffold_difficulty": route_scaffold_difficulty,
            "cave_pool_size": cave_pool_size
            or int(floor_summary.get("config", {}).get("cave_pool_size", 0) or 0),
            "tutorial_route_episodes": episodes,
            "route_floor_source_win_rate": float(floor_eval.get("win_rate", 0.0) or 0.0),
            "route_floor_source_wins": int(floor_eval.get("wins", 0) or 0),
            "route_floor_source_games": int(floor_eval.get("num_games", 0) or 0),
        },
        "transfer_checkpoint": "in-memory route-floor first-crystal weights",
        "failure_diagnostics": diagnostics,
    }
    if selected_eval_payload is not None:
        extra["selected_checkpoint_eval"] = selected_eval_payload
        extra["selected_checkpoint_eval_games"] = selected_eval_games
    if selected_diagnostics is not None:
        extra["selected_checkpoint_failure_diagnostics"] = selected_diagnostics
    route_summary = summarize_trainer(
        trainer,
        label="first_crystal_route",
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra=extra,
    )
    return [floor_summary, route_summary]


def run_route_demo_bc(
    out_dir: Path,
    *,
    episodes: int,
    route_floor_episodes: int,
    route_scaffold_difficulty: str,
    cave_pool_size: int | None,
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
    route_demo_levels: int,
    route_demo_max_steps: int,
    route_demo_variants: tuple[str, ...],
    bc_epochs: int,
    bc_batch_size: int,
    demo_repeat: int,
) -> list[dict[str, Any]]:
    set_seed(seed)
    floor_run_dir = out_dir / "route_demo_floor"
    floor_config = first_crystal_config(
        floor_run_dir,
        episodes=route_floor_episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
        difficulty=route_scaffold_difficulty,
    )
    apply_cave_pool_override(floor_config, cave_pool_size)
    demos = collect_scripted_route_demonstrations(
        floor_config,
        max_levels=route_demo_levels,
        max_steps=route_demo_max_steps,
        controller_variants=route_demo_variants,
    )
    demo_summary = demos["summary"]
    write_json(floor_run_dir / "demo_summary.json", demo_summary)

    trainer = prepare_trainer(
        floor_config,
        episodes=route_floor_episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    bc = behavior_clone_from_demonstrations(
        trainer.agent,
        demos["trajectories"],
        epochs=bc_epochs,
        batch_size=bc_batch_size,
    )
    seeded = seed_replay_from_demonstrations(
        trainer.agent,
        demos["trajectories"],
        repeat=demo_repeat,
    )
    after_bc_eval = final_eval(
        floor_config,
        trainer.agent,
        out_dir=floor_run_dir,
        label="route_demo_floor_after_bc",
        episode=0,
        games=eval_games,
    )
    after_bc_snapshot = {"episode": 0, "source_eval": after_bc_eval}
    best_snapshot: dict[str, Any] = after_bc_snapshot
    best_weights = capture_weight_snapshot(trainer.agent)

    print(
        f"Route demos: {demo_summary.get('kept_trajectories', 0)}/"
        f"{demo_summary.get('attempts', 0)} scripted wins, "
        f"{demo_summary.get('kept_transitions', 0)} transitions; "
        f"BC updates {bc.get('updates', 0)}, after-BC source win "
        f"{100 * after_bc_eval.get('win_rate', 0):.0f}%",
        flush=True,
    )

    train_seconds, source_eval_history, training_best_snapshot, training_best_weights = (
        run_training_with_source_snapshots(
            trainer,
            floor_config,
            run_dir=floor_run_dir,
            label="route_demo_floor",
            total_episodes=route_floor_episodes,
            heartbeat_seconds=heartbeat_seconds,
            source_eval_every=eval_every,
            eval_games=eval_games,
        )
    )
    if training_best_snapshot is not None and source_snapshot_score(
        training_best_snapshot
    ) > source_snapshot_score(best_snapshot):
        best_snapshot = training_best_snapshot
        best_weights = training_best_weights or capture_weight_snapshot(trainer.agent)
    floor_history = [after_bc_snapshot] + source_eval_history
    if source_eval_history:
        floor_eval = source_eval_history[-1]["source_eval"]
    else:
        floor_eval = after_bc_eval
    checkpoint = Path(floor_config.GAME_MODEL_DIR) / "crystal_caves_final.pth"
    floor_summary = summarize_trainer(
        trainer,
        label="route_demo_floor",
        train_seconds=train_seconds,
        final_eval_payload=floor_eval,
        extra={
            "source_eval_history": floor_history,
            "selected_source_episode": best_snapshot.get("episode"),
            "selected_source_eval": best_snapshot.get("source_eval"),
            "checkpoint": str(checkpoint) if save_checkpoints else "in-memory only",
            "route_demo": {
                "kind": "scripted_route_floor",
                "demo_summary": demo_summary,
                "demo_summary_path": str(floor_run_dir / "demo_summary.json"),
                "behavior_cloning": bc,
                "seeded": seeded,
                "after_bc_eval": after_bc_eval,
            },
        },
    )

    set_seed(seed)
    run_dir = out_dir / "route_demo_tutorial_route"
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
    route_trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        transfer_weights=best_weights,
        save_checkpoints=save_checkpoints,
    )
    transfer_eval = final_eval(
        config,
        route_trainer.agent,
        out_dir=run_dir,
        label="route_demo_tutorial_route_initial",
        episode=0,
        games=eval_games,
    )
    initial_snapshot = {"episode": 0, "source_eval": transfer_eval}
    route_train_seconds, route_history, route_best_snapshot, _ = run_training_with_source_snapshots(
        route_trainer,
        config,
        run_dir=run_dir,
        label="route_demo_tutorial_route",
        total_episodes=episodes,
        heartbeat_seconds=heartbeat_seconds,
        source_eval_every=eval_every,
        eval_games=eval_games,
    )
    tutorial_history = [initial_snapshot] + route_history
    selected_route_snapshot = initial_snapshot
    if route_best_snapshot is not None and source_snapshot_score(
        route_best_snapshot
    ) > source_snapshot_score(selected_route_snapshot):
        selected_route_snapshot = route_best_snapshot
    if route_history:
        eval_payload = route_history[-1]["source_eval"]
    else:
        eval_payload = transfer_eval
    diagnostics = trace_heldout_failures(
        config,
        route_trainer.agent,
        out_dir=run_dir,
        label="route_demo_tutorial_route_heldout",
        games=trace_games,
        max_steps=trace_max_steps,
        sample_every=trace_sample_every,
        tail_steps=trace_tail_steps,
    )
    selected_floor_eval = floor_summary.get("selected_source_eval") or floor_eval
    route_summary = summarize_trainer(
        route_trainer,
        label="route_demo_tutorial_route",
        train_seconds=route_train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "source_eval_history": tutorial_history,
            "selected_source_episode": selected_route_snapshot.get("episode"),
            "selected_source_eval": selected_route_snapshot.get("source_eval"),
            "route_curriculum": {
                "route_floor_episodes": route_floor_episodes,
                "route_scaffold_difficulty": route_scaffold_difficulty,
                "cave_pool_size": cave_pool_size
                or int(floor_summary.get("config", {}).get("cave_pool_size", 0) or 0),
                "tutorial_route_episodes": episodes,
                "route_floor_source_win_rate": float(
                    selected_floor_eval.get("win_rate", 0.0) or 0.0
                ),
                "route_floor_source_wins": int(selected_floor_eval.get("wins", 0) or 0),
                "route_floor_source_games": int(selected_floor_eval.get("num_games", 0) or 0),
            },
            "transfer_checkpoint": "in-memory route-floor demo+BC weights",
            "transfer_source": {
                "kind": "route_demo_bc",
                "source_episode": int(floor_summary.get("selected_source_episode", 0) or 0),
                "source_win_rate": float(selected_floor_eval.get("win_rate", 0.0) or 0.0),
                "source_wins": int(selected_floor_eval.get("wins", 0) or 0),
                "source_games": int(selected_floor_eval.get("num_games", 0) or 0),
            },
            "failure_diagnostics": diagnostics,
        },
    )
    return [floor_summary, route_summary]


def run_first_crystal_direct(
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
    route_aux_weight: float,
    route_aux_deadband: float,
    label: str = "first_crystal_direct",
) -> dict[str, Any]:
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
    apply_route_aux_override(
        config,
        route_aux_weight=route_aux_weight,
        route_aux_deadband=route_aux_deadband,
    )
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    train_seconds, source_eval_history, best_snapshot, best_weights = (
        run_training_with_source_snapshots(
            trainer,
            config,
            run_dir=run_dir,
            label=label,
            total_episodes=episodes,
            heartbeat_seconds=heartbeat_seconds,
            source_eval_every=eval_every,
            eval_games=eval_games,
        )
    )
    if source_eval_history:
        eval_payload = source_eval_history[-1]["source_eval"]
    else:
        eval_payload = final_eval(
            config,
            trainer.agent,
            out_dir=run_dir,
            label=f"{label}_final",
            episode=trainer.current_episode,
            games=eval_games,
        )
    if best_snapshot is None:
        best_snapshot = {
            "episode": int(trainer.current_episode),
            "source_eval": eval_payload,
        }
    if best_weights is None:
        best_weights = capture_weight_snapshot(trainer.agent)
    diagnostics = trace_heldout_failures(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_heldout",
        games=trace_games,
        max_steps=trace_max_steps,
        sample_every=trace_sample_every,
        tail_steps=trace_tail_steps,
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
    selected_eval_payload: dict[str, Any] | None = None
    selected_diagnostics: dict[str, Any] | None = None
    selected_near_miss_eval: dict[str, Any] | None = None
    selected_checkpoint_path: str | None = None
    if selected_eval_games > 0 and best_weights is not None:
        final_weights = capture_weight_snapshot(trainer.agent)
        selected_episode = int(best_snapshot.get("episode", trainer.current_episode) or 0)
        load_weight_snapshot(trainer.agent, best_weights)
        if save_checkpoints:
            checkpoint_path = (
                Path(config.GAME_MODEL_DIR) / f"{label}_selected_ep{selected_episode}.pth"
            )
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
        load_weight_snapshot(trainer.agent, final_weights)
    checkpoint = Path(config.GAME_MODEL_DIR) / "crystal_caves_final.pth"
    extra: dict[str, Any] = {
        "source_eval_history": source_eval_history,
        "selected_source_episode": best_snapshot.get("episode"),
        "selected_source_eval": best_snapshot.get("source_eval"),
        "checkpoint": str(checkpoint) if save_checkpoints else "in-memory only",
        "route_direct": {
            "difficulty": "tutorial",
            "cave_pool_size": cave_pool_size
            or int(config_snapshot(config).get("cave_pool_size", 0) or 0),
            "route_aux_weight": route_aux_weight,
            "route_aux_deadband": route_aux_deadband,
        },
        "failure_diagnostics": diagnostics,
        "near_miss_eval": near_miss_eval,
    }
    if selected_eval_payload is not None:
        extra["selected_checkpoint_eval"] = selected_eval_payload
        extra["selected_checkpoint_eval_games"] = selected_eval_games
    if selected_checkpoint_path is not None:
        extra["selected_checkpoint_path"] = selected_checkpoint_path
    if selected_diagnostics is not None:
        extra["selected_checkpoint_failure_diagnostics"] = selected_diagnostics
    if selected_near_miss_eval is not None:
        extra["selected_checkpoint_near_miss_eval"] = selected_near_miss_eval
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra=extra,
    )
