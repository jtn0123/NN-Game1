# ruff: noqa: F401,F403,F405,I001
from .common import *
from .config_helpers import *
from .io_utils import *
from .snapshots import *
from .stats import *
from .vec_envs import *


def prepare_trainer(
    config: Config,
    *,
    episodes: int,
    vec_envs: int,
    transfer_weights: dict[str, dict[str, torch.Tensor]] | None = None,
    strict_transfer: bool = True,
    save_checkpoints: bool = False,
) -> HeadlessTrainer:
    trainer = HeadlessTrainer(config, trainer_args(episodes=episodes, vec_envs=vec_envs))
    if not save_checkpoints:
        disable_checkpoint_writes(trainer)
    if transfer_weights is not None:
        load_weight_snapshot(trainer.agent, transfer_weights, strict=strict_transfer)
    if trainer.agent.epsilon < TUTORIAL_MIN_EPSILON:
        trainer.agent.epsilon = TUTORIAL_MIN_EPSILON
    return trainer


def disable_checkpoint_writes(trainer: HeadlessTrainer) -> None:
    """Avoid large .pth writes in low-disk experiment sessions."""

    def skip_save(filename: str, *args: Any, **kwargs: Any) -> None:
        reason = kwargs.get("save_reason", "checkpoint")
        if not kwargs.get("quiet", False):
            print(f"Skipping checkpoint save ({reason}): {filename}")
        return None

    trainer._save_model = skip_save  # type: ignore[assignment,method-assign]
    trainer._cleanup_old_periodic_saves = lambda *args, **kwargs: None  # type: ignore[method-assign]


def capture_weight_snapshot(agent: Any) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "policy": {
            key: value.detach().cpu().clone()
            for key, value in agent.policy_net.state_dict().items()
        },
        "target": {
            key: value.detach().cpu().clone()
            for key, value in agent.target_net.state_dict().items()
        },
    }


def load_weight_snapshot(
    agent: Any,
    snapshot: dict[str, dict[str, torch.Tensor]],
    *,
    strict: bool = True,
) -> None:
    policy = {key: value.to(agent.device) for key, value in snapshot["policy"].items()}
    target = {key: value.to(agent.device) for key, value in snapshot["target"].items()}
    agent.policy_net.load_state_dict(policy, strict=strict)
    agent.target_net.load_state_dict(target, strict=strict)


class LiveRunMonitor:
    """Write training status while the blocking trainer loop is still running."""

    def __init__(
        self,
        trainer: HeadlessTrainer,
        *,
        run_dir: Path,
        label: str,
        total_episodes: int,
        heartbeat_seconds: float,
    ):
        self.trainer = trainer
        self.run_dir = run_dir
        self.label = label
        self.total_episodes = total_episodes
        self.heartbeat_seconds = heartbeat_seconds
        self.started = time.time()
        self.live_path = run_dir / "live_metrics.json"
        self.history_path = run_dir / "live_metrics.jsonl"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_written_episode = -1

    def start(self) -> None:
        if self.heartbeat_seconds <= 0:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_snapshot("starting", force=True)
        self._thread = threading.Thread(target=self._loop, name=f"{self.label}-live", daemon=True)
        self._thread.start()

    def stop(self, status: str = "complete") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.write_snapshot(status, force=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.write_snapshot("running")

    def write_snapshot(self, status: str, *, force: bool = False) -> None:
        try:
            snapshot = live_snapshot(
                self.trainer,
                label=self.label,
                status=status,
                started=self.started,
                total_episodes=self.total_episodes,
            )
            write_json(self.live_path, snapshot)
            episode = int(snapshot["episode"])
            if force or episode != self._last_written_episode:
                with self.history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
                self._last_written_episode = episode
                print(live_status_line(snapshot), flush=True)
        except Exception as exc:
            print(f"[live {self.label}] monitor failed: {exc}", flush=True)


def live_snapshot(
    trainer: HeadlessTrainer,
    *,
    label: str,
    status: str,
    started: float,
    total_episodes: int,
) -> dict[str, Any]:
    elapsed = max(0.001, time.time() - started)
    episode = int(trainer.current_episode)
    scores = [float(score) for score in trainer.scores]
    eval_history = [
        result.to_dict() for result in (trainer.evaluator.eval_history if trainer.evaluator else [])
    ]
    latest_eval = eval_history[-1] if eval_history else None
    avg_progress = mean_tail(trainer.progresses)
    best_progress = max_or_zero(trainer.progresses)
    win_rate = float(np.mean(trainer.wins[-100:])) if trainer.wins else 0.0
    source_stats = trainer_source_stats(trainer)
    snapshot = {
        "label": label,
        "status": status,
        "episode": episode,
        "total_episodes": total_episodes,
        "episode_pct": episode / total_episodes if total_episodes > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "episodes_per_minute": episode / (elapsed / 60.0),
        "total_steps": int(trainer.total_steps),
        "steps_per_second": trainer.total_steps / elapsed,
        "epsilon": float(trainer.agent.epsilon),
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
        "avg_contact_action_loss_100": float(trainer.agent.get_average_contact_action_loss(100)),
        "avg_contact_action_accuracy_100": float(
            trainer.agent.get_average_contact_action_accuracy(100)
        ),
        "avg_policy_anchor_loss_100": float(trainer.agent.get_average_policy_anchor_loss(100)),
        "avg_policy_anchor_accuracy_100": float(
            trainer.agent.get_average_policy_anchor_accuracy(100)
        ),
        "correction_action_enabled": bool(
            getattr(trainer.config, "CRYSTAL_CAVES_CORRECTION_ACTION_LOSS", False)
        ),
        "correction_action_transitions": int(
            trainer.agent.get_correction_action_transition_count()
        ),
        "correction_action_samples_100": int(trainer.agent.get_correction_action_metric_count(100)),
        "contact_action_head_enabled": bool(
            getattr(trainer.config, "CRYSTAL_CAVES_CONTACT_ACTION_HEAD", False)
        ),
        "contact_action_weight": float(
            getattr(trainer.config, "CRYSTAL_CAVES_CONTACT_ACTION_WEIGHT", 0.0)
        ),
        "contact_action_transitions": int(trainer.agent.get_contact_action_transition_count()),
        "contact_action_samples_100": int(trainer.agent.get_contact_action_metric_count(100)),
        "policy_anchor_enabled": bool(
            getattr(trainer.config, "CRYSTAL_CAVES_POLICY_ANCHOR_LOSS", False)
        ),
        "policy_anchor_weight": float(
            getattr(trainer.config, "CRYSTAL_CAVES_POLICY_ANCHOR_WEIGHT", 0.0)
        ),
        "policy_anchor_min_target_distance_norm": float(
            getattr(
                trainer.config,
                "CRYSTAL_CAVES_POLICY_ANCHOR_MIN_TARGET_DISTANCE_NORM",
                0.0,
            )
        ),
        "policy_anchor_samples_100": int(trainer.agent.get_policy_anchor_metric_count(100)),
        "avg_q_100": mean_tail(trainer.q_values),
        "avg_score_100": mean_tail(scores),
        "last_score": scores[-1] if scores else 0.0,
        "win_rate_100": win_rate,
        "avg_progress_100": avg_progress,
        "best_progress": best_progress,
        "end_reason_counts_100": counter_tail(trainer.end_reasons),
        "mean_phi_parts_100": mean_dicts(
            [
                {key: float(value or 0.0) for key, value in part.items()}
                for part in trainer.progress_parts[-100:]
            ]
        ),
        "eval_count": len(eval_history),
        "latest_eval": latest_eval,
        "source_stats": source_stats,
        "reverse_start_stats": trainer_reverse_start_stats(trainer),
        "archive_stats": trainer_archive_stats(trainer),
    }
    snapshot.update(contact_interleave_metric_aliases(source_stats))
    return snapshot


def live_status_line(snapshot: dict[str, Any]) -> str:
    latest_eval = snapshot.get("latest_eval") or {}
    eval_bits = ""
    if latest_eval:
        eval_bits = (
            " | held-out "
            f"w {100 * latest_eval.get('win_rate', 0):.0f}% "
            f"c {100 * latest_eval.get('mean_crystal_frac', 0):.0f}% "
            f"d {100 * latest_eval.get('mean_depth_frac', 0):.0f}%"
        )
    aux_bits = ""
    if snapshot.get("avg_route_aux_loss_100", 0) > 0:
        aux_bits = (
            f" | aux {snapshot['avg_route_aux_loss_100']:.3f}/"
            f"{100 * snapshot.get('avg_route_aux_accuracy_100', 0):.0f}%"
        )
    demo_bits = ""
    if snapshot.get("avg_demo_action_loss_100", 0) > 0:
        demo_bits = (
            f" | demo {snapshot['avg_demo_action_loss_100']:.3f}/"
            f"{100 * snapshot.get('avg_demo_action_accuracy_100', 0):.0f}%"
        )
    if snapshot.get("avg_demo_conservative_loss_100", 0) > 0:
        demo_bits += f"/cql {snapshot['avg_demo_conservative_loss_100']:.3f}"
    if snapshot.get("avg_close_zone_demo_action_loss_100", 0) > 0:
        demo_bits += (
            f" | cz {snapshot['avg_close_zone_demo_action_loss_100']:.3f}/"
            f"{100 * snapshot.get('avg_close_zone_demo_action_accuracy_100', 0):.0f}%"
        )
    if (
        snapshot.get("correction_action_enabled", False)
        or snapshot.get("correction_action_transitions", 0) > 0
        or snapshot.get("correction_action_samples_100", 0) > 0
    ):
        demo_bits += (
            f" | corr {snapshot['avg_correction_action_loss_100']:.3f}/"
            f"{100 * snapshot.get('avg_correction_action_accuracy_100', 0):.0f}%"
            f" n={snapshot.get('correction_action_samples_100', 0)}"
        )
    if (
        snapshot.get("contact_action_head_enabled", False)
        or snapshot.get("contact_action_transitions", 0) > 0
        or snapshot.get("contact_action_samples_100", 0) > 0
    ):
        demo_bits += (
            f" | head {snapshot['avg_contact_action_loss_100']:.3f}/"
            f"{100 * snapshot.get('avg_contact_action_accuracy_100', 0):.0f}%"
            f" n={snapshot.get('contact_action_samples_100', 0)}"
        )
    if (
        snapshot.get("policy_anchor_enabled", False)
        or snapshot.get("policy_anchor_samples_100", 0) > 0
    ):
        demo_bits += (
            f" | anchor {snapshot['avg_policy_anchor_loss_100']:.3f}/"
            f"{100 * snapshot.get('avg_policy_anchor_accuracy_100', 0):.0f}%"
            f" n={snapshot.get('policy_anchor_samples_100', 0)}"
        )
    return (
        f"[live {snapshot['label']}] ep {snapshot['episode']}/{snapshot['total_episodes']} "
        f"({100 * snapshot['episode_pct']:.0f}%) | "
        f"score100 {snapshot['avg_score_100']:.0f} | "
        f"prog100 {snapshot['avg_progress_100']:.3f} best {snapshot['best_progress']:.3f} | "
        f"win100 {100 * snapshot['win_rate_100']:.0f}% | "
        f"loss {snapshot['avg_loss_100']:.4f} q {snapshot['avg_q_100']:.2f} | "
        f"{snapshot['steps_per_second']:.0f} steps/s"
        f"{aux_bits}"
        f"{demo_bits}"
        f"{eval_bits}"
    )


def run_training(
    trainer: HeadlessTrainer,
    *,
    run_dir: Path,
    label: str,
    total_episodes: int,
    heartbeat_seconds: float,
    target_episodes: int | None = None,
) -> float:
    started = time.time()
    original_max_episodes = trainer.config.MAX_EPISODES
    if target_episodes is not None:
        trainer.config.MAX_EPISODES = target_episodes
    monitor = LiveRunMonitor(
        trainer,
        run_dir=run_dir,
        label=label,
        total_episodes=total_episodes,
        heartbeat_seconds=heartbeat_seconds,
    )
    monitor.start()
    try:
        trainer.train()
    except KeyboardInterrupt:
        monitor.stop("interrupted")
        raise
    else:
        monitor.stop("complete")
        return time.time() - started
    finally:
        trainer.config.MAX_EPISODES = original_max_episodes


def run_training_with_bridge_snapshots(
    trainer: HeadlessTrainer,
    config: Config,
    *,
    run_dir: Path,
    label: str,
    total_episodes: int,
    heartbeat_seconds: float,
    bridge_eval_every: int,
    eval_k: int,
    bridge_eval_max_steps: int | None,
) -> tuple[
    float,
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, dict[str, torch.Tensor]] | None,
]:
    if bridge_eval_every <= 0:
        train_seconds = run_training(
            trainer,
            run_dir=run_dir,
            label=label,
            total_episodes=total_episodes,
            heartbeat_seconds=heartbeat_seconds,
        )
        return train_seconds, [], None, None

    history: list[dict[str, Any]] = []
    best_snapshot: dict[str, Any] | None = None
    best_weights: dict[str, dict[str, torch.Tensor]] | None = None
    train_seconds = 0.0
    history_path = run_dir / "bridge_eval_history.jsonl"
    start_episode = int(trainer.current_episode)
    milestones = list(range(start_episode + bridge_eval_every, total_episodes, bridge_eval_every))
    if not milestones or milestones[-1] != total_episodes:
        milestones.append(total_episodes)

    for target_episode in milestones:
        if trainer.current_episode >= target_episode:
            continue
        train_seconds += run_training(
            trainer,
            run_dir=run_dir,
            label=label,
            total_episodes=total_episodes,
            heartbeat_seconds=heartbeat_seconds,
            target_episodes=target_episode,
        )
        snapshot = bridge_eval_snapshot(
            trainer,
            config,
            eval_k=eval_k,
            max_steps=bridge_eval_max_steps,
        )
        history.append(snapshot)
        append_jsonl(history_path, snapshot)
        print(bridge_snapshot_line(snapshot))
        if best_snapshot is None or bridge_snapshot_score(snapshot) > bridge_snapshot_score(
            best_snapshot
        ):
            best_snapshot = snapshot
            best_weights = capture_weight_snapshot(trainer.agent)
        if trainer.current_episode < target_episode:
            break

    return train_seconds, history, best_snapshot, best_weights


def run_training_with_source_snapshots(
    trainer: HeadlessTrainer,
    config: Config,
    *,
    run_dir: Path,
    label: str,
    total_episodes: int,
    heartbeat_seconds: float,
    source_eval_every: int,
    eval_games: int,
) -> tuple[
    float,
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, dict[str, torch.Tensor]] | None,
]:
    history: list[dict[str, Any]] = []
    best_snapshot: dict[str, Any] | None = None
    best_weights: dict[str, dict[str, torch.Tensor]] | None = None
    history_path = run_dir / "source_eval_history.jsonl"
    train_seconds = 0.0
    start_episode = int(trainer.current_episode)
    step = source_eval_every if source_eval_every > 0 else total_episodes
    milestones = list(range(start_episode + step, total_episodes, step))
    if not milestones or milestones[-1] != total_episodes:
        milestones.append(total_episodes)

    for target_episode in milestones:
        if trainer.current_episode >= target_episode:
            continue
        train_seconds += run_training(
            trainer,
            run_dir=run_dir,
            label=label,
            total_episodes=total_episodes,
            heartbeat_seconds=heartbeat_seconds,
            target_episodes=target_episode,
        )
        snapshot = source_eval_snapshot(
            trainer,
            config,
            run_dir=run_dir,
            label=label,
            games=eval_games,
        )
        history.append(snapshot)
        append_jsonl(history_path, snapshot)
        print(source_snapshot_line(snapshot), flush=True)
        if best_snapshot is None or source_snapshot_score(snapshot) > source_snapshot_score(
            best_snapshot
        ):
            best_snapshot = snapshot
            best_weights = capture_weight_snapshot(trainer.agent)
        if trainer.current_episode < target_episode:
            break

    return train_seconds, history, best_snapshot, best_weights
