"""Crystal Caves staged curriculum runner."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Optional

from config import Config
from src.ai.evaluator import EvalResults
from src.app.headless import HeadlessTrainer
from src.app.training_runtime import write_eval_best_baseline


@dataclass(frozen=True)
class CrystalCurriculumStage:
    """One Crystal Caves curriculum rung."""

    stage_id: str
    name: str
    difficulty: str
    families: str
    default_episodes: int
    min_epsilon: float
    gate: str


@dataclass(frozen=True)
class StageGateResult:
    """Promotion decision for a curriculum stage."""

    ready: bool
    status: str
    detail: str


DEFAULT_CRYSTAL_CURRICULUM: tuple[CrystalCurriculumStage, ...] = (
    CrystalCurriculumStage(
        stage_id="tutorial_platform",
        name="Tutorial platform floor",
        difficulty="tutorial",
        families="platform_network",
        default_episodes=300,
        min_epsilon=0.35,
        gate="learn crystal -> exit on the simplest lock-free caves",
    ),
    CrystalCurriculumStage(
        stage_id="easy_platform",
        name="Easy platform networks",
        difficulty="easy",
        families="platform_network",
        default_episodes=750,
        min_epsilon=0.25,
        gate="held-out wins and stable switch/crystal collection",
    ),
    CrystalCurriculumStage(
        stage_id="easy_mixed",
        name="Easy mixed families",
        difficulty="easy",
        families="",
        default_episodes=900,
        min_epsilon=0.22,
        gate="generalize easy play across all generator families",
    ),
    CrystalCurriculumStage(
        stage_id="normal_platform",
        name="Normal platform networks",
        difficulty="normal",
        families="platform_network",
        default_episodes=900,
        min_epsilon=0.18,
        gate="carry the easy policy into full-objective platform caves",
    ),
    CrystalCurriculumStage(
        stage_id="normal_mixed",
        name="Normal mixed families",
        difficulty="normal",
        families="",
        default_episodes=1200,
        min_epsilon=0.15,
        gate="full Crystal Caves generalization across normal levels",
    ),
)

STAGE_GATE_EXTENSION_FRACTION = 0.50
STAGE_GATE_MAX_MULTIPLIER = 2.0


def planned_stage_episodes(
    stages: tuple[CrystalCurriculumStage, ...],
    total_budget: Optional[int],
    per_stage_override: Optional[int],
) -> list[int]:
    """Return the episode budget for each stage.

    ``--episodes`` is treated as the total curriculum budget. Without it, each
    stage uses its tuned default. ``--curriculum-stage-episodes`` is an explicit
    override for experiments where every stage should have the same length.
    """
    if per_stage_override is not None and per_stage_override > 0:
        return [per_stage_override for _ in stages]

    if total_budget is None or total_budget <= 0:
        return [stage.default_episodes for stage in stages]

    defaults = [stage.default_episodes for stage in stages]
    default_total = sum(defaults)
    raw = [max(1, round(total_budget * episodes / default_total)) for episodes in defaults]
    delta = total_budget - sum(raw)
    raw[-1] = max(1, raw[-1] + delta)
    return raw


# Each stage anneals exploration from its min_epsilon floor down to roughly this
# fraction of it over the stage budget, instead of sitting near the floor the whole
# stage (the global EPSILON_DECAY=0.9995 only moves epsilon ~0.35->0.30 over 300
# episodes). Gives the policy a real explore->exploit schedule within each stage.
STAGE_EPSILON_END_FRACTION = 0.3


def stage_epsilon_decay(start_epsilon: float, budget: int, end_epsilon: float) -> float:
    """Per-episode multiplicative decay that anneals start_epsilon to end_epsilon
    over ``budget`` episodes. Returns 1.0 (no decay) for degenerate inputs."""
    if budget <= 0 or start_epsilon <= 0 or end_epsilon <= 0 or end_epsilon >= start_epsilon:
        return 1.0
    return float((end_epsilon / start_epsilon) ** (1.0 / budget))


def run_crystal_curriculum(
    config: Config,
    args: argparse.Namespace,
    *,
    existing_dashboard: Optional[Any] = None,
) -> None:
    """Run the staged Crystal Caves curriculum, warm-starting each stage."""
    if config.GAME_NAME != "crystal_caves":
        raise ValueError("--crystal-curriculum only supports --game crystal_caves")

    config.CRYSTAL_CAVES_PROCEDURAL = True
    config.USE_CNN_STATE = True
    config.EARLY_STOP_ON_PLATEAU = True
    config.EVAL_EVERY = min(config.EVAL_EVERY, 150)
    full_gate_eval_episodes = config.EVAL_EPISODES
    config.EVAL_EPISODES = min(config.EVAL_EPISODES, 12)

    # decay_epsilon() gates on (episode - per-stage offset), so a non-zero
    # EPSILON_WARMUP froze epsilon for the first EPSILON_WARMUP episodes of EVERY
    # stage (the offset resets each stage). Disable the warmup so each stage anneals
    # exploration from episode 0; the per-stage EPSILON_DECAY set below drives the
    # actual explore->exploit schedule.
    config.EPSILON_WARMUP = 0

    # The periodic held-out eval, eval-best checkpointing, plateau detection, and
    # EARLY_STOP_ON_PLATEAU live ONLY in HeadlessTrainer.train_vectorized(); the
    # single-env train() path runs none of them. The whole curriculum (held-out
    # gate, eval-best warm-start, early-stop rollback) depends on that machinery,
    # so force vectorized training even when the user did not pass --vec-envs.
    # Respect a larger user-provided value.
    curriculum_vec_envs = max(int(getattr(args, "vec_envs", 1) or 1), 8)

    stages = DEFAULT_CRYSTAL_CURRICULUM
    budgets = planned_stage_episodes(
        stages,
        total_budget=getattr(args, "episodes", None),
        per_stage_override=getattr(args, "curriculum_stage_episodes", None),
    )

    print("\n" + "=" * 70)
    print("🎓 CRYSTAL CAVES CURRICULUM")
    print("=" * 70)
    for index, (stage, budget) in enumerate(zip(stages, budgets), start=1):
        families = stage.families or "all"
        print(
            f"   {index}/{len(stages)} {stage.name}: {stage.difficulty}, {families}, +{budget} ep"
        )
    print(f"   vectorized envs per stage: {curriculum_vec_envs}")
    print("=" * 70 + "\n")

    base_model_dir = config.MODEL_DIR
    run_model_dir = os.path.join(
        base_model_dir,
        "crystal_caves_curriculum",
        time.strftime("run_%Y%m%d_%H%M%S"),
    )
    dashboard = existing_dashboard
    model_path = getattr(args, "model", None)

    for index, (stage, budget) in enumerate(zip(stages, budgets), start=1):
        config.MODEL_DIR = os.path.join(
            run_model_dir,
            f"stage{index:02d}_{stage.stage_id}",
        )
        config.CRYSTAL_CAVES_DIFFICULTY = stage.difficulty
        config.CRYSTAL_CAVES_FAMILIES = stage.families
        # On full-objective (normal) stages, perturbing an already-peaked policy with
        # the plateau exploration boost tends to drive it into collapse — prefer
        # early-stop + eval-best rollback there. The win-regression guard covers the
        # earlier stages.
        config.DISABLE_EXPLORATION_BOOST = stage.difficulty == "normal"

        stage_args = copy.copy(args)
        stage_args.headless = True
        stage_args.game = "crystal_caves"
        stage_args.random_caves = True
        stage_args.cnn = True
        stage_args.early_stop = True
        stage_args.cave_difficulty = stage.difficulty
        stage_args.cave_families = stage.families or None
        stage_args.model = model_path
        stage_args.vec_envs = curriculum_vec_envs
        # The trainer's config.MAX_EPISODES is set after load, because loaded
        # checkpoints carry their own episode number.
        stage_args.episodes = None

        next_stage = stages[index].name if index < len(stages) else "complete"
        trainer = HeadlessTrainer(config, stage_args, existing_dashboard=dashboard)
        dashboard = trainer.web_dashboard or dashboard
        stage_start = trainer.current_episode
        trainer.epsilon_episode_offset = stage_start
        stage_target = stage_start + budget
        config.MAX_EPISODES = stage_target
        # Anneal exploration from this stage's floor toward EPSILON_END over its
        # budget so the policy can consolidate as epsilon falls, instead of training
        # the whole stage at the (high) floor under the global slow decay.
        stage_eps_end = max(config.EPSILON_END, stage.min_epsilon * STAGE_EPSILON_END_FRACTION)
        config.EPSILON_DECAY = stage_epsilon_decay(stage.min_epsilon, budget, stage_eps_end)
        if dashboard:
            dashboard.publisher.state.target_episodes = stage_target
            dashboard.publisher.state.training_start_time = trainer.training_start_time
        if trainer.agent.epsilon < stage.min_epsilon:
            old_epsilon = trainer.agent.epsilon
            trainer.agent.epsilon = stage.min_epsilon
            if dashboard:
                dashboard.log(
                    f"🎲 Stage exploration floor: ε {old_epsilon:.3f} → "
                    f"{trainer.agent.epsilon:.3f}",
                    "info",
                )

        _publish_stage(
            dashboard,
            stage=stage,
            index=index,
            total=len(stages),
            start_episode=stage_start,
            target_episode=stage_target,
            status="running",
            gate_result=StageGateResult(False, "checking", "waiting for held-out eval"),
            next_stage_name=next_stage,
            checkpoint_mode="warm-start" if model_path else "fresh",
        )
        if dashboard:
            families = stage.families or "all families"
            dashboard.log(
                f"🎓 Stage {index}/{len(stages)}: {stage.name} "
                f"({stage.difficulty}, {families})",
                "info",
            )

        gate_result = StageGateResult(False, "checking", "waiting for held-out eval")
        max_target = stage_start + max(budget, int(round(budget * STAGE_GATE_MAX_MULTIPLIER)))
        while True:
            trainer.train()
            eval_results = _run_stage_gate_eval(
                trainer,
                dashboard=dashboard,
                eval_episodes=full_gate_eval_episodes,
            )
            gate_result = evaluate_stage_gate(stage, eval_results, dashboard=dashboard)
            _publish_stage(
                dashboard,
                stage=stage,
                index=index,
                total=len(stages),
                start_episode=stage_start,
                target_episode=config.MAX_EPISODES,
                status=("ready" if gate_result.ready else "gate-hold"),
                gate_result=gate_result,
                next_stage_name=next_stage,
                checkpoint_mode="eval-best rollback" if model_path else "fresh",
            )
            if gate_result.ready or index == len(stages):
                break

            if trainer.current_episode >= max_target:
                if dashboard:
                    dashboard.log(
                        f"🛑 Stage gate blocked {stage.name}: {gate_result.detail}",
                        "warning",
                    )
                _publish_stage(
                    dashboard,
                    stage=stage,
                    index=index,
                    total=len(stages),
                    start_episode=stage_start,
                    target_episode=config.MAX_EPISODES,
                    status="blocked",
                    gate_result=gate_result,
                    next_stage_name=next_stage,
                    checkpoint_mode="eval-best rollback" if model_path else "fresh",
                )
                return

            extension = max(1, int(round(budget * STAGE_GATE_EXTENSION_FRACTION)))
            config.MAX_EPISODES = min(max_target, trainer.current_episode + extension)
            if dashboard:
                dashboard.publisher.state.target_episodes = config.MAX_EPISODES
                dashboard.log(
                    f"🚦 Holding {stage.name}: {gate_result.detail}. "
                    f"Extending to episode {config.MAX_EPISODES}.",
                    "warning",
                )
            trainer.running = True

        # Roll the live policy back to this stage's eval-best so the in-memory agent
        # (and the snapshot below) reflect the kept-best, not a post-peak policy.
        trainer._restore_eval_best()
        model_path = _snapshot_stage_eval_best(config, stage, index) or model_path
        _publish_stage(
            dashboard,
            stage=stage,
            index=index,
            total=len(stages),
            start_episode=stage_start,
            target_episode=config.MAX_EPISODES,
            status="complete",
            gate_result=gate_result,
            next_stage_name=next_stage,
            checkpoint_mode="eval-best rollback" if model_path else "fresh",
        )

        if not trainer.running and getattr(config, "EARLY_STOP_ON_PLATEAU", False):
            # Early-stop is stage-local; continue the curriculum from the eval-best
            # checkpoint rather than ending the whole staged session.
            print(f"✓ Stage {index} complete; moving to {next_stage}.")

    if dashboard:
        dashboard.log("🎓 Crystal Caves curriculum complete", "success")


def _snapshot_stage_eval_best(
    config: Config,
    stage: CrystalCurriculumStage,
    stage_index: int,
) -> Optional[str]:
    eval_best = os.path.join(config.GAME_MODEL_DIR, f"{config.GAME_NAME}_eval_best.pth")
    if not os.path.exists(eval_best):
        best = os.path.join(config.GAME_MODEL_DIR, f"{config.GAME_NAME}_best.pth")
        return best if os.path.exists(best) else None

    snapshot_name = f"{config.GAME_NAME}_stage{stage_index:02d}_{stage.stage_id}_eval_best.pth"
    snapshot_path = os.path.join(config.GAME_MODEL_DIR, snapshot_name)
    shutil.copy2(eval_best, snapshot_path)
    return snapshot_path


def evaluate_stage_gate(
    stage: CrystalCurriculumStage,
    eval_results: Optional[EvalResults],
    *,
    dashboard: Optional[Any] = None,
) -> StageGateResult:
    """Return whether a stage has enough held-out evidence to promote."""
    if eval_results is None:
        return StageGateResult(False, "waiting", "no held-out eval yet")

    timeout_share = _reason_share(eval_results, "timeout")
    terminal_fail_share = timeout_share + _reason_share(eval_results, "stalled")
    crystal = eval_results.mean_crystal_frac
    switch = eval_results.mean_switch_rate
    wins = eval_results.win_rate
    recent_training_crystals = 0.0
    if dashboard is not None:
        recent_training_crystals = float(
            getattr(dashboard.publisher.state, "cc_recent_crystal_frac", 0.0) or 0.0
        )

    if stage.stage_id == "tutorial_platform":
        checks = [
            (crystal >= 0.80, f"eval crystals {crystal*100:.0f}% >= 80%"),
            (
                max(wins, recent_training_crystals) >= 0.20,
                f"wins {wins*100:.0f}% or train crystals {recent_training_crystals*100:.0f}% >= 20%",
            ),
            (terminal_fail_share <= 0.80, f"timeout/stall {terminal_fail_share*100:.0f}% <= 80%"),
        ]
    elif stage.stage_id == "easy_platform":
        checks = [
            (crystal >= 0.65, f"eval crystals {crystal*100:.0f}% >= 65%"),
            (switch >= 0.35, f"eval switch {switch*100:.0f}% >= 35%"),
            (wins >= 0.05, f"eval wins {wins*100:.0f}% >= 5%"),
            (terminal_fail_share <= 0.75, f"timeout/stall {terminal_fail_share*100:.0f}% <= 75%"),
        ]
    elif stage.stage_id == "easy_mixed":
        checks = [
            (crystal >= 0.55, f"eval crystals {crystal*100:.0f}% >= 55%"),
            (wins >= 0.05, f"eval wins {wins*100:.0f}% >= 5%"),
            (terminal_fail_share <= 0.80, f"timeout/stall {terminal_fail_share*100:.0f}% <= 80%"),
        ]
    else:
        checks = [
            (crystal >= 0.50, f"eval crystals {crystal*100:.0f}% >= 50%"),
            (wins >= 0.03, f"eval wins {wins*100:.0f}% >= 3%"),
            (terminal_fail_share <= 0.85, f"timeout/stall {terminal_fail_share*100:.0f}% <= 85%"),
        ]

    failed = [detail for ok, detail in checks if not ok]
    if failed:
        return StageGateResult(False, "not ready", "; ".join(failed))
    return StageGateResult(True, "ready", "held-out gate passed")


def _reason_share(eval_results: EvalResults, reason: str) -> float:
    total = max(1, sum(eval_results.end_reason_counts.values()))
    return float(eval_results.end_reason_counts.get(reason, 0) / total)


def _run_stage_gate_eval(
    trainer: HeadlessTrainer,
    *,
    dashboard: Optional[Any],
    eval_episodes: int,
) -> Optional[EvalResults]:
    if trainer.evaluator is None:
        return None

    # Keep-best is win-rate-aware: save the eval-best on the selection score (which
    # win_rate dominates), not raw mean score, so the gate does not preserve a
    # high-score/low-win policy over a winning one.
    previous_best_selection = trainer.evaluator.best_eval_selection
    # The gate eval uses a different (larger) sample than the in-loop evals and must
    # not perturb the plateau/early-stop counter the vectorized trainer reads. Snapshot
    # the evaluator's plateau state, run the gate eval, then restore it after using the
    # results for the eval-best save decision below.
    saved_best = trainer.evaluator.best_eval_score
    saved_best_selection = trainer.evaluator.best_eval_selection
    saved_best_win_rate = trainer.evaluator.best_eval_win_rate
    saved_since_improvement = trainer.evaluator.evals_since_improvement
    saved_history_len = len(trainer.evaluator.eval_history)
    results = trainer.evaluator.evaluate(
        num_episodes=eval_episodes,
        max_steps=trainer.config.EVAL_MAX_STEPS,
        episode_num=trainer.current_episode,
    )
    trainer.evaluator.best_eval_score = saved_best
    trainer.evaluator.best_eval_selection = saved_best_selection
    trainer.evaluator.best_eval_win_rate = saved_best_win_rate
    trainer.evaluator.evals_since_improvement = saved_since_improvement
    del trainer.evaluator.eval_history[saved_history_len:]
    trainer.evaluator.log_results(results)

    if results.selection_score > previous_best_selection:
        eval_best_filename = f"{trainer.config.GAME_NAME}_eval_best.pth"
        trainer._save_model(
            eval_best_filename,
            save_reason="eval_best",
            quiet=True,
            save_replay_buffer=False,
        )
        write_eval_best_baseline(
            trainer.config.GAME_MODEL_DIR,
            trainer.config.GAME_NAME,
            episode=trainer.current_episode,
            mean_score=results.mean_score,
            checkpoint=eval_best_filename,
        )

    if dashboard:
        dashboard.publisher.record_eval(
            episode=trainer.current_episode,
            mean_score=results.mean_score,
            std_score=results.std_score,
            median_score=results.median_score,
            win_rate=results.win_rate,
            num_games=results.num_games,
            crystal_frac=results.mean_crystal_frac,
            switch_rate=results.mean_switch_rate,
            depth_frac=results.mean_depth_frac,
            end_reason_counts=results.end_reason_counts,
        )
        dashboard.log(
            f"🚦 Gate eval: {results.mean_score:.0f} avg, "
            f"{results.win_rate*100:.0f}% wins, "
            f"{results.mean_crystal_frac*100:.0f}% crystals",
            "info",
        )
    return results


def _publish_stage(
    dashboard: Optional[Any],
    *,
    stage: CrystalCurriculumStage,
    index: int,
    total: int,
    start_episode: int,
    target_episode: int,
    status: str,
    next_stage_name: str,
    gate_result: StageGateResult,
    checkpoint_mode: str,
) -> None:
    if dashboard is None:
        return
    dashboard.publisher.record_curriculum_stage(
        active=True,
        stage_index=index,
        stage_total=total,
        stage_id=stage.stage_id,
        stage_name=stage.name,
        difficulty=stage.difficulty,
        families=stage.families or "all",
        start_episode=start_episode,
        target_episode=target_episode,
        status=status,
        gate=stage.gate,
        next_stage_name=next_stage_name,
        gate_ready=gate_result.ready,
        gate_status=gate_result.status,
        gate_detail=gate_result.detail,
        checkpoint_mode=checkpoint_mode,
    )
