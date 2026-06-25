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


def run_baseline(
    out_dir: Path,
    *,
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
    label: str = "baseline",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
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
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
    )


def run_diagnostic_baseline(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_games: int,
    cave_pool_size: int | None,
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
    label: str = "diagnostic_baseline",
    anti_loop_reward: bool = False,
    invalid_interact_penalty: bool = False,
    novelty_bonus: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    config = full_tutorial_config(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    apply_cave_pool_override(config, cave_pool_size)
    exp_config = cc_experiment_config(config)
    exp_config.CRYSTAL_CAVES_ANTI_LOOP_REWARD = anti_loop_reward
    exp_config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = invalid_interact_penalty
    exp_config.CRYSTAL_CAVES_NOVELTY_BONUS = novelty_bonus
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
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
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "failure_diagnostics": diagnostics,
            "near_miss_eval": near_miss_eval,
        },
    )


def run_interleaved(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_games: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    drill_ratio: float,
    drill_envs_override: int | None,
    save_checkpoints: bool,
    label: str = "interleaved",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    full_envs, skill_envs = interleave_counts(
        vec_envs=vec_envs,
        skill_ratio=drill_ratio,
        skill_envs=drill_envs_override,
    )
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
    install_interleaved_vec_env(
        trainer,
        run_dir=run_dir,
        full_envs=full_envs,
        skill_envs=skill_envs,
        skill_source="drill",
        skill_config=make_interleaved_drill_config(config),
    )
    print(
        f"🧪 Interleaved training: {full_envs} full tutorial envs + "
        f"{skill_envs} drill envs ({skill_envs / (full_envs + skill_envs):.0%} drill)"
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
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "interleave": {
                "full_envs": full_envs,
                "skill_envs": skill_envs,
                "skill_source": "drill",
                "skill_ratio": skill_envs / (full_envs + skill_envs),
            }
        },
    )


def run_bridge_interleaved(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_games: int,
    cave_pool_size: int | None,
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
    bridge_ratio: float,
    bridge_envs_override: int | None,
    save_checkpoints: bool,
    selected_eval_games: int,
    first_crystal_goal: bool = False,
    label: str = "bridge_interleaved",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    full_envs, skill_envs = interleave_counts(
        vec_envs=vec_envs,
        skill_ratio=bridge_ratio,
        skill_envs=bridge_envs_override,
    )
    config_factory = first_crystal_config if first_crystal_goal else full_tutorial_config
    config = config_factory(
        run_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    apply_cave_pool_override(config, cave_pool_size)
    bridge_config = make_interleaved_bridge_config(config)
    cc_experiment_config(bridge_config).CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL = False
    trainer = prepare_trainer(
        config,
        episodes=episodes,
        vec_envs=vec_envs,
        save_checkpoints=save_checkpoints,
    )
    install_interleaved_vec_env(
        trainer,
        run_dir=run_dir,
        full_envs=full_envs,
        skill_envs=skill_envs,
        skill_source="bridge",
        skill_config=bridge_config,
    )
    print(
        f"🌉 Bridge interleaved training: {full_envs} full tutorial envs + "
        f"{skill_envs} bridge envs ({skill_envs / (full_envs + skill_envs):.0%} bridge)"
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

    extra: dict[str, Any] = {
        "source_eval_history": source_eval_history,
        "selected_source_episode": best_snapshot.get("episode"),
        "selected_source_eval": best_snapshot.get("source_eval"),
        "interleave": {
            "full_envs": full_envs,
            "skill_envs": skill_envs,
            "skill_source": "bridge",
            "skill_ratio": skill_envs / (full_envs + skill_envs),
            "cave_pool_size": cave_pool_size
            or int(config_snapshot(config).get("cave_pool_size", 0) or 0),
            "full_lane_goal": "first_crystal" if first_crystal_goal else "full_level",
            "bridge_lane_goal": "full_bridge",
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
