"""Crystal Caves staged curriculum runner."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional

from config import Config
from src.app.headless import HeadlessTrainer


@dataclass(frozen=True)
class CrystalCurriculumStage:
    """One Crystal Caves curriculum rung."""

    stage_id: str
    name: str
    difficulty: str
    families: str
    default_episodes: int
    gate: str


DEFAULT_CRYSTAL_CURRICULUM: tuple[CrystalCurriculumStage, ...] = (
    CrystalCurriculumStage(
        stage_id="tutorial_platform",
        name="Tutorial platform floor",
        difficulty="tutorial",
        families="platform_network",
        default_episodes=300,
        gate="learn crystal -> exit on the simplest lock-free caves",
    ),
    CrystalCurriculumStage(
        stage_id="easy_platform",
        name="Easy platform networks",
        difficulty="easy",
        families="platform_network",
        default_episodes=750,
        gate="held-out wins and stable switch/crystal collection",
    ),
    CrystalCurriculumStage(
        stage_id="easy_mixed",
        name="Easy mixed families",
        difficulty="easy",
        families="",
        default_episodes=900,
        gate="generalize easy play across all generator families",
    ),
    CrystalCurriculumStage(
        stage_id="normal_platform",
        name="Normal platform networks",
        difficulty="normal",
        families="platform_network",
        default_episodes=900,
        gate="carry the easy policy into full-objective platform caves",
    ),
    CrystalCurriculumStage(
        stage_id="normal_mixed",
        name="Normal mixed families",
        difficulty="normal",
        families="",
        default_episodes=1200,
        gate="full Crystal Caves generalization across normal levels",
    ),
)


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
    print("=" * 70 + "\n")

    dashboard = existing_dashboard
    base_model_dir = config.MODEL_DIR
    model_path = getattr(args, "model", None) or _default_warm_start_checkpoint(
        base_model_dir,
        config.GAME_NAME,
    )

    for index, (stage, budget) in enumerate(zip(stages, budgets), start=1):
        config.MODEL_DIR = os.path.join(
            base_model_dir,
            "crystal_caves_curriculum",
            f"stage{index:02d}_{stage.stage_id}",
        )
        config.CRYSTAL_CAVES_DIFFICULTY = stage.difficulty
        config.CRYSTAL_CAVES_FAMILIES = stage.families

        stage_args = copy.copy(args)
        stage_args.headless = True
        stage_args.game = "crystal_caves"
        stage_args.random_caves = True
        stage_args.cnn = True
        stage_args.early_stop = True
        stage_args.cave_difficulty = stage.difficulty
        stage_args.cave_families = stage.families or None
        stage_args.model = model_path
        # The trainer's config.MAX_EPISODES is set after load, because loaded
        # checkpoints carry their own episode number.
        stage_args.episodes = None

        next_stage = stages[index].name if index < len(stages) else "complete"
        trainer = HeadlessTrainer(config, stage_args, existing_dashboard=dashboard)
        dashboard = trainer.web_dashboard or dashboard
        stage_start = trainer.current_episode
        stage_target = stage_start + budget
        config.MAX_EPISODES = stage_target

        _publish_stage(
            dashboard,
            stage=stage,
            index=index,
            total=len(stages),
            start_episode=stage_start,
            target_episode=stage_target,
            status="running",
            next_stage_name=next_stage,
        )
        if dashboard:
            families = stage.families or "all families"
            dashboard.log(
                f"🎓 Stage {index}/{len(stages)}: {stage.name} "
                f"({stage.difficulty}, {families})",
                "info",
            )

        trainer.train()

        model_path = _snapshot_stage_eval_best(config, stage, index) or model_path
        _publish_stage(
            dashboard,
            stage=stage,
            index=index,
            total=len(stages),
            start_episode=stage_start,
            target_episode=stage_target,
            status="complete",
            next_stage_name=next_stage,
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


def _default_warm_start_checkpoint(base_model_dir: str, game_name: str) -> Optional[str]:
    """Return the best existing non-curriculum checkpoint, if present."""
    game_dir = os.path.join(base_model_dir, game_name)
    for filename in (f"{game_name}_eval_best.pth", f"{game_name}_best.pth"):
        path = os.path.join(game_dir, filename)
        if os.path.exists(path):
            return path
    return None


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
    )
