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


def run_reverse_start(
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
    reverse_ratio: float,
    reverse_envs_override: int | None,
    save_checkpoints: bool,
    label: str = "reverse_start",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    full_envs, reverse_envs = reverse_start_counts(
        vec_envs=vec_envs,
        reverse_ratio=reverse_ratio,
        reverse_envs=reverse_envs_override,
    )
    modes = reverse_start_modes(reverse_envs)
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
    install_reverse_start_vec_env(
        trainer,
        full_envs=full_envs,
        reverse_envs=reverse_envs,
    )
    print(
        f"↩️  Reverse-start training: {full_envs} full tutorial envs + "
        f"{reverse_envs} reverse envs ({reverse_envs / (full_envs + reverse_envs):.0%} reverse)"
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
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "reverse_start": {
                "full_envs": full_envs,
                "reverse_envs": reverse_envs,
                "reverse_ratio": reverse_envs / (full_envs + reverse_envs),
                "modes": dict(Counter(modes)),
            },
            "failure_diagnostics": diagnostics,
        },
    )


def run_archive_start(
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
    archive_ratio: float,
    archive_envs_override: int | None,
    archive_replay_prob: float,
    archive_max_size: int,
    archive_min_steps: int,
    save_checkpoints: bool,
    label: str = "archive_start",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    full_envs, archive_envs = archive_start_counts(
        vec_envs=vec_envs,
        archive_ratio=archive_ratio,
        archive_envs=archive_envs_override,
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
    install_archive_start_vec_env(
        trainer,
        full_envs=full_envs,
        archive_envs=archive_envs,
        replay_prob=archive_replay_prob,
        max_size=archive_max_size,
        min_steps=archive_min_steps,
    )
    print(
        f"Archive-start training: {full_envs} full tutorial envs + "
        f"{archive_envs} archive envs ({archive_envs / (full_envs + archive_envs):.0%} archive), "
        f"replay prob {archive_replay_prob:.0%}, max {archive_max_size}"
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
    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "archive_start": {
                "full_envs": full_envs,
                "archive_envs": archive_envs,
                "archive_ratio": archive_envs / (full_envs + archive_envs),
                "replay_prob": archive_replay_prob,
                "max_size": archive_max_size,
                "min_steps": archive_min_steps,
            },
            "failure_diagnostics": diagnostics,
        },
    )


def run_drill_pretrain(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_k: int,
    train_eval_games: int,
    eval_every: int,
    log_every: int,
    report_seconds: float,
    heartbeat_seconds: float,
    vec_envs: int,
    save_checkpoints: bool,
    drill_eval_max_steps: int | None,
    label: str = "drill_pretrain",
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    set_seed(seed)
    run_dir = out_dir / label
    config = drill_config(
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
    drill_eval = per_skill_eval(
        trainer.agent,
        config,
        k=eval_k,
        max_steps=drill_eval_max_steps,
    )
    weights = capture_weight_snapshot(trainer.agent)
    checkpoint = Path(config.GAME_MODEL_DIR) / "crystal_caves_final.pth"
    summary = summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        extra={
            "drill_eval": drill_eval,
            "checkpoint": str(checkpoint) if save_checkpoints else "in-memory only",
        },
    )
    return summary, weights


def run_bridge_pretrain(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
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
    label: str = "bridge_pretrain",
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    set_seed(seed)
    run_dir = out_dir / label
    config = bridge_config(
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
    train_seconds, bridge_eval_history, best_snapshot, best_weights = (
        run_training_with_bridge_snapshots(
            trainer,
            config,
            run_dir=run_dir,
            label=label,
            total_episodes=episodes,
            heartbeat_seconds=heartbeat_seconds,
            bridge_eval_every=bridge_eval_every,
            eval_k=eval_k,
            bridge_eval_max_steps=bridge_eval_max_steps,
        )
    )
    final_bridge_eval = (
        bridge_eval_history[-1]["bridge_eval"]
        if bridge_eval_history
        else level_set_eval(
            trainer.agent,
            config,
            specs=BRIDGE_CAVES,
            k=eval_k,
            max_steps=bridge_eval_max_steps,
        )
    )
    if best_snapshot is None:
        best_snapshot = {
            "episode": int(trainer.current_episode),
            "bridge_eval": final_bridge_eval,
            "rollup": level_eval_rollup(final_bridge_eval),
        }
    if best_weights is None:
        best_weights = capture_weight_snapshot(trainer.agent)
    checkpoint = Path(config.GAME_MODEL_DIR) / "crystal_caves_final.pth"
    summary = summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        extra={
            "bridge_eval": final_bridge_eval,
            "bridge_eval_history": bridge_eval_history,
            "selected_bridge_episode": best_snapshot.get("episode"),
            "selected_bridge_rollup": best_snapshot.get("rollup"),
            "selected_bridge_eval": best_snapshot.get("bridge_eval"),
            "checkpoint": str(checkpoint) if save_checkpoints else "in-memory only",
        },
    )
    return summary, best_weights


def run_first_crystal_pretrain(
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
    difficulty: str = "tutorial",
    label: str = "first_crystal_pretrain",
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
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
        difficulty=difficulty,
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
    checkpoint = Path(config.GAME_MODEL_DIR) / "crystal_caves_final.pth"
    summary = summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        final_eval_payload=eval_payload,
        extra={
            "source_eval_history": source_eval_history,
            "selected_source_episode": best_snapshot.get("episode"),
            "selected_source_eval": best_snapshot.get("source_eval"),
            "checkpoint": str(checkpoint) if save_checkpoints else "in-memory only",
        },
    )
    return summary, best_weights
