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
    exp_config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = bool(
        saved_config.get("invalid_interact_penalty", False)
    )
    exp_config.CRYSTAL_CAVES_INVALID_SHOOT_PENALTY = bool(
        saved_config.get("invalid_shoot_penalty", False)
    )
    exp_config.CRYSTAL_CAVES_NOVELTY_BONUS = bool(saved_config.get("novelty_bonus", False))
    return config


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
