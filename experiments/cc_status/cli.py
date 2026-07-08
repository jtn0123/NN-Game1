# ruff: noqa: F401,F403,F405,I001
from .common import *
from .artifacts import validate_status_session_artifacts
from .cli_args import add_status_session_arguments
from .contact_head import parse_contact_head_action_thresholds
from .contact_label_audit import *
from .cli_label_modes import run_label_dataset_mode
from .correction_calibration import *
from .config_helpers import *
from .corrections import *
from .cli_helpers import (
    append_interrupted_run_from_live_metrics,
    close_zone_route_demo_variants,
    tutorial_demo_bc_kwargs,
)
from .io_utils import *
from .recipes import expand_recipe_argv, format_recipe_list
from .reports import *
from .scorecard import *
from .runs_baseline import *
from .runs_demo import *
from .runs_mixed import *
from .runs_route import *
from .runs_transfer import *

_append_interrupted_run_from_live_metrics = append_interrupted_run_from_live_metrics


def _configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def _handle_list_recipes_command(argv: list[str]) -> bool:
    if len(argv) > 1 and argv[1] == "list-recipes":
        print(format_recipe_list())
        return True
    return False


def _expand_run_recipe_command(argv: list[str]) -> list[str]:
    if len(argv) > 1 and argv[1] == "run-recipe":
        try:
            return list(expand_recipe_argv(argv))
        except (KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc
    return argv


def _parse_status_session_args() -> (
    tuple[argparse.ArgumentParser, argparse.Namespace, tuple[str, ...]]
):
    parser = argparse.ArgumentParser()
    add_status_session_arguments(parser)
    opts = parser.parse_args()
    route_demo_variants = parse_route_demo_variants(opts.route_demo_variants)
    return parser, opts, route_demo_variants


def _new_status_session_payload(opts: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    label = opts.label or opts.mode
    run_id = timestamp_id(label)
    out_dir = Path(opts.out_dir or ".Codex/artifacts/cc_sessions") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run_id": run_id,
        "mode": opts.mode,
        "seed": opts.seed,
        "created_at": datetime.now().isoformat(),
        "out_dir": str(out_dir),
        "runs": [],
    }
    return out_dir, payload


def _requires_live_metrics(opts: argparse.Namespace, payload: dict[str, Any]) -> bool:
    return opts.heartbeat_seconds > 0 and any(
        _run_requires_live_metrics(run) for run in payload["runs"] if isinstance(run, dict)
    )


def _run_requires_live_metrics(run: dict[str, Any]) -> bool:
    if any(
        key in run
        for key in (
            "contact_action_head_calibration",
            "contact_label_audit",
            "contact_label_filter",
        )
    ):
        return False
    return float(run.get("train_seconds", 0.0) or 0.0) > 0


def _write_status_session_outputs(
    out_dir: Path,
    opts: argparse.Namespace,
    payload: dict[str, Any],
) -> None:
    write_json(out_dir / "summary.json", payload)
    write_markdown_report(out_dir / "report.md", payload)
    if not opts.no_artifact_validation:
        validation = validate_status_session_artifacts(
            out_dir,
            require_live_metrics=_requires_live_metrics(opts, payload),
        )
        write_json(out_dir / "artifact_validation.json", validation.to_dict())
        if validation.ok:
            print(f"Artifact validation: ok ({out_dir / 'artifact_validation.json'})")
        else:
            print(f"Artifact validation failed ({out_dir / 'artifact_validation.json'})")
            for issue in validation.errors:
                print(f"- {issue.path}: {issue.message}", file=sys.stderr)
            raise SystemExit(1)
    print(f"\nWrote structured session artifacts to {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Report:  {out_dir / 'report.md'}")


def _run_checkpoint_correction_mode(
    parser: argparse.ArgumentParser,
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
) -> bool:
    if opts.mode == "eval-checkpoint":
        if not opts.checkpoint:
            parser.error("eval-checkpoint requires --checkpoint")
        payload["runs"].append(
            run_eval_checkpoint(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                objective=opts.objective,
                label="eval_checkpoint",
            )
        )
        return True
    if opts.mode == "eval-final-contact-option":
        if not opts.checkpoint:
            parser.error("eval-final-contact-option requires --checkpoint")
        payload["runs"].append(
            run_eval_final_contact_option(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                seed=opts.seed,
                eval_games=opts.eval_games,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                close_zone_distance_tiles=opts.final_contact_distance,
                final_contact_commit_steps=opts.final_contact_commit_steps,
                cancel_option_outside_close_zone=opts.final_contact_cancel_outside,
                gate_policy_advantage=opts.final_contact_policy_advantage_gate,
                min_option_advantage=opts.final_contact_min_option_advantage,
                label=opts.label or "eval_final_contact_option",
            )
        )
        return True
    if opts.mode == "collect-corrections":
        if not opts.checkpoint:
            parser.error("collect-corrections requires --checkpoint")
        payload["runs"].append(
            run_collect_corrections(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                seed=opts.seed,
                correction_games=opts.correction_games,
                correction_max_steps=opts.correction_max_steps,
                correction_max_examples=opts.correction_max_examples,
                correction_sample_every=opts.correction_sample_every,
                correction_max_examples_per_game=opts.correction_max_examples_per_game,
                correction_stale_steps=opts.correction_stale_steps,
                correction_loop_tile_visits=opts.correction_loop_tile_visits,
                correction_keep_agreements=opts.correction_keep_agreements,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                correction_label_mode=(
                    "advantage_gate" if opts.final_contact_policy_advantage_gate else "standard"
                ),
                final_contact_distance=opts.final_contact_distance,
                final_contact_commit_steps=opts.final_contact_commit_steps,
                final_contact_min_option_advantage=opts.final_contact_min_option_advantage,
                label=opts.label or "collect_corrections",
            )
        )
        return True
    if opts.mode == "collect-contact-head-corrections":
        if not opts.checkpoint:
            parser.error("collect-contact-head-corrections requires --checkpoint")
        if not opts.correction_dataset:
            parser.error("collect-contact-head-corrections requires --correction-dataset")
        payload["runs"].append(
            run_collect_contact_head_corrections(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                correction_dataset_path=Path(opts.correction_dataset),
                seed=opts.seed,
                correction_games=opts.correction_games,
                correction_max_steps=opts.correction_max_steps,
                correction_max_examples=opts.correction_max_examples,
                correction_sample_every=opts.correction_sample_every,
                correction_max_examples_per_game=opts.correction_max_examples_per_game,
                correction_stale_steps=opts.correction_stale_steps,
                correction_loop_tile_visits=opts.correction_loop_tile_visits,
                correction_keep_agreements=opts.correction_keep_agreements,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                correction_label_mode=(
                    "advantage_gate" if opts.final_contact_policy_advantage_gate else "standard"
                ),
                final_contact_distance=opts.final_contact_distance,
                final_contact_commit_steps=opts.final_contact_commit_steps,
                final_contact_min_option_advantage=opts.final_contact_min_option_advantage,
                contact_action_batch_size=opts.contact_action_batch_size,
                contact_action_distance_tiles=opts.contact_action_distance,
                contact_head_offline_steps=opts.contact_head_offline_steps,
                contact_head_learning_rate=opts.contact_head_learning_rate,
                contact_head_confidence=opts.contact_head_confidence,
                contact_head_jump_confidence=opts.contact_head_jump_confidence,
                contact_head_action_thresholds=parse_contact_head_action_thresholds(
                    opts.contact_head_action_thresholds
                ),
                contact_head_balance_classes=opts.contact_head_balance_classes,
                label=opts.label or "collect_contact_head_corrections",
            )
        )
        return True
    if opts.mode == "correction-finetune":
        if not opts.checkpoint:
            parser.error("correction-finetune requires --checkpoint")
        if not opts.correction_dataset:
            parser.error("correction-finetune requires --correction-dataset")
        payload["runs"].append(
            run_correction_finetune(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                correction_dataset_path=Path(opts.correction_dataset),
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                correction_action_weight=opts.correction_action_weight,
                correction_action_margin=opts.correction_action_margin,
                correction_action_batch_size=opts.correction_action_batch_size,
                policy_anchor_weight=opts.policy_anchor_weight,
                policy_anchor_temperature=opts.policy_anchor_temperature,
                policy_anchor_min_distance_tiles=opts.policy_anchor_min_distance_tiles,
                label=opts.label or "correction_finetune",
            )
        )
        return True
    if opts.mode == "contact-head-finetune":
        if not opts.checkpoint:
            parser.error("contact-head-finetune requires --checkpoint")
        if not opts.correction_dataset:
            parser.error("contact-head-finetune requires --correction-dataset")
        payload["runs"].append(
            run_contact_head_finetune(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                correction_dataset_path=Path(opts.correction_dataset),
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                contact_action_weight=opts.contact_action_weight,
                contact_action_batch_size=opts.contact_action_batch_size,
                contact_action_distance_tiles=opts.contact_action_distance,
                label=opts.label or "contact_head_finetune",
            )
        )
        return True
    if opts.mode == "contact-head-offline":
        if not opts.checkpoint:
            parser.error("contact-head-offline requires --checkpoint")
        if not opts.correction_dataset:
            parser.error("contact-head-offline requires --correction-dataset")
        payload["runs"].append(
            run_contact_head_offline(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                correction_dataset_path=Path(opts.correction_dataset),
                seed=opts.seed,
                eval_games=opts.eval_games,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                contact_action_batch_size=opts.contact_action_batch_size,
                contact_action_distance_tiles=opts.contact_action_distance,
                contact_head_offline_steps=opts.contact_head_offline_steps,
                contact_head_learning_rate=opts.contact_head_learning_rate,
                contact_head_confidence=opts.contact_head_confidence,
                contact_head_jump_confidence=opts.contact_head_jump_confidence,
                contact_head_action_thresholds=parse_contact_head_action_thresholds(
                    opts.contact_head_action_thresholds
                ),
                contact_head_balance_classes=opts.contact_head_balance_classes,
                label=opts.label or "contact_head_offline",
            )
        )
        return True
    if opts.mode == "contact-head-calibrate":
        if not opts.checkpoint:
            parser.error("contact-head-calibrate requires --checkpoint")
        correction_dataset_paths = parse_correction_dataset_paths(opts.correction_datasets)
        if not correction_dataset_paths:
            parser.error("contact-head-calibrate requires --correction-datasets")
        payload["runs"].append(
            run_contact_head_calibration(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                correction_dataset_paths=correction_dataset_paths,
                seed=opts.seed,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                contact_action_batch_size=opts.contact_action_batch_size,
                contact_action_distance_tiles=opts.contact_action_distance,
                contact_head_offline_steps=opts.contact_head_offline_steps,
                contact_head_learning_rate=opts.contact_head_learning_rate,
                contact_head_balance_classes=opts.contact_head_balance_classes,
                calibration_frac=opts.contact_head_calibration_frac,
                calibration_seed=opts.contact_head_calibration_seed,
                min_calibration_accuracy=opts.contact_head_min_calibration_accuracy,
                min_class_examples=opts.contact_head_min_class_examples,
                label=opts.label or "contact_head_calibration",
            )
        )
        return True
    return False


def _run_baseline_diagnostic_mode(
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
) -> bool:
    if opts.mode == "baseline":
        payload["runs"].append(
            run_baseline(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
            )
        )
        return True
    if opts.mode == "diagnose-baseline":
        payload["runs"].append(
            run_diagnostic_baseline(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                cave_pool_size=opts.cave_pool_size,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
            )
        )
        return True
    if opts.mode == "anti-loop":
        payload["runs"].append(
            run_diagnostic_baseline(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                cave_pool_size=opts.cave_pool_size,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                label="anti_loop",
                anti_loop_reward=True,
            )
        )
        return True
    if opts.mode == "invalid-interact":
        payload["runs"].append(
            run_diagnostic_baseline(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                cave_pool_size=opts.cave_pool_size,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                label="invalid_interact",
                invalid_interact_penalty=True,
            )
        )
        return True
    if opts.mode == "novelty-bonus":
        payload["runs"].append(
            run_diagnostic_baseline(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                cave_pool_size=opts.cave_pool_size,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                label="novelty_bonus",
                novelty_bonus=True,
            )
        )
        return True
    return False


def _run_mixed_reset_mode(
    parser: argparse.ArgumentParser,
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
) -> bool:
    if opts.mode == "interleaved":
        payload["runs"].append(
            run_interleaved(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                drill_ratio=opts.interleave_drill_ratio,
                drill_envs_override=opts.interleave_drill_envs,
                save_checkpoints=opts.save_checkpoints,
            )
        )
        return True
    if opts.mode == "bridge-interleaved":
        payload["runs"].append(
            run_bridge_interleaved(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                cave_pool_size=opts.cave_pool_size,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                bridge_ratio=opts.interleave_bridge_ratio,
                bridge_envs_override=opts.interleave_bridge_envs,
                save_checkpoints=opts.save_checkpoints,
                selected_eval_games=opts.selected_eval_games,
                first_crystal_goal=opts.interleave_first_crystal_goal,
            )
        )
        return True
    if opts.mode == "contact-interleaved":
        if not opts.checkpoint:
            parser.error("contact-interleaved requires --checkpoint")
        payload["runs"].append(
            run_contact_interleaved(
                out_dir,
                checkpoint_path=Path(opts.checkpoint),
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                cave_pool_size=opts.cave_pool_size,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                contact_ratio=opts.interleave_contact_ratio,
                contact_envs_override=opts.interleave_contact_envs,
                contact_pool_size=opts.contact_pool_size,
                contact_eval_pool_size=opts.contact_eval_pool_size,
                save_checkpoints=opts.save_checkpoints,
                save_selected_checkpoint=opts.save_selected_checkpoint,
                selected_eval_games=opts.selected_eval_games,
                label=opts.label or "contact_interleaved",
            )
        )
        return True
    if opts.mode == "reverse-start":
        payload["runs"].append(
            run_reverse_start(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                reverse_ratio=opts.reverse_start_ratio,
                reverse_envs_override=opts.reverse_start_envs,
                save_checkpoints=opts.save_checkpoints,
            )
        )
        return True
    if opts.mode == "archive-start":
        payload["runs"].append(
            run_archive_start(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                archive_ratio=opts.archive_start_ratio,
                archive_envs_override=opts.archive_start_envs,
                archive_replay_prob=opts.archive_replay_prob,
                archive_max_size=opts.archive_max_size,
                archive_min_steps=opts.archive_min_steps,
                save_checkpoints=opts.save_checkpoints,
            )
        )
        return True
    return False


def _run_skill_transfer_mode(
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
) -> bool:
    if opts.mode == "drill":
        drill_summary, _ = run_drill_pretrain(
            out_dir,
            episodes=opts.drill_episodes,
            seed=opts.seed,
            eval_k=opts.eval_k,
            train_eval_games=opts.train_eval_games,
            eval_every=opts.eval_every,
            log_every=opts.log_every,
            report_seconds=opts.report_seconds,
            heartbeat_seconds=opts.heartbeat_seconds,
            vec_envs=opts.vec_envs,
            save_checkpoints=opts.save_checkpoints,
            drill_eval_max_steps=opts.drill_eval_max_steps,
        )
        payload["runs"].append(drill_summary)
        return True
    if opts.mode == "bridge":
        bridge_summary, _ = run_bridge_pretrain(
            out_dir,
            episodes=opts.bridge_episodes,
            seed=opts.seed,
            eval_k=opts.eval_k,
            train_eval_games=opts.train_eval_games,
            eval_every=opts.eval_every,
            log_every=opts.log_every,
            report_seconds=opts.report_seconds,
            heartbeat_seconds=opts.heartbeat_seconds,
            vec_envs=opts.vec_envs,
            save_checkpoints=opts.save_checkpoints,
            bridge_eval_max_steps=opts.bridge_eval_max_steps,
            bridge_eval_every=opts.bridge_eval_every,
        )
        payload["runs"].append(bridge_summary)
        return True
    if opts.mode == "transfer":
        payload["runs"].extend(
            run_transfer(
                out_dir,
                episodes=opts.episodes,
                drill_episodes=opts.drill_episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                eval_k=opts.eval_k,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                drill_eval_max_steps=opts.drill_eval_max_steps,
            )
        )
        return True
    if opts.mode == "bridge-transfer":
        payload["runs"].extend(
            run_bridge_transfer(
                out_dir,
                episodes=opts.episodes,
                bridge_episodes=opts.bridge_episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                eval_k=opts.eval_k,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                bridge_eval_max_steps=opts.bridge_eval_max_steps,
                bridge_eval_every=opts.bridge_eval_every,
            )
        )
        return True
    if opts.mode == "bridge-demo-replay":
        payload["runs"].extend(
            run_bridge_demo_replay(
                out_dir,
                episodes=opts.episodes,
                bridge_episodes=opts.bridge_episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                eval_k=opts.eval_k,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                bridge_eval_max_steps=opts.bridge_eval_max_steps,
                bridge_eval_every=opts.bridge_eval_every,
                demo_repeat=opts.demo_repeat,
            )
        )
        return True
    return False


def _run_route_mode(
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
    route_demo_variants: tuple[str, ...],
) -> bool:
    if opts.mode == "first-crystal-route":
        payload["runs"].extend(
            run_first_crystal_route_curriculum(
                out_dir,
                episodes=opts.episodes,
                route_floor_episodes=opts.route_floor_episodes,
                route_scaffold_difficulty=opts.route_scaffold_difficulty,
                cave_pool_size=opts.cave_pool_size,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                selected_eval_games=opts.selected_eval_games,
            )
        )
        return True
    if opts.mode == "first-crystal-direct":
        payload["runs"].append(
            run_first_crystal_direct(
                out_dir,
                episodes=opts.episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                save_selected_checkpoint=opts.save_selected_checkpoint,
                cave_pool_size=opts.cave_pool_size,
                selected_eval_games=opts.selected_eval_games,
                route_aux_weight=opts.route_aux_weight,
                route_aux_deadband=opts.route_aux_deadband,
            )
        )
        return True
    if opts.mode == "route-demo-bc":
        payload["runs"].extend(
            run_route_demo_bc(
                out_dir,
                episodes=opts.episodes,
                route_floor_episodes=opts.route_floor_episodes,
                route_scaffold_difficulty=opts.route_scaffold_difficulty,
                cave_pool_size=opts.cave_pool_size,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                route_demo_levels=opts.route_demo_levels,
                route_demo_max_steps=opts.route_demo_max_steps,
                route_demo_variants=route_demo_variants,
                bc_epochs=opts.bc_epochs,
                bc_batch_size=opts.bc_batch_size,
                demo_repeat=opts.demo_repeat,
            )
        )
        return True
    if opts.mode == "first-crystal-transfer":
        payload["runs"].extend(
            run_first_crystal_transfer(
                out_dir,
                episodes=opts.episodes,
                route_episodes=opts.route_episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
            )
        )
        return True
    return False


def _run_tutorial_demo_mode(
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
    route_demo_variants: tuple[str, ...],
) -> bool:
    if opts.mode == "tutorial-demo-bc":
        payload["runs"].append(
            run_tutorial_demo_bc(
                out_dir,
                **tutorial_demo_bc_kwargs(opts, route_demo_variants),
                invalid_shoot_penalty=opts.invalid_shoot_penalty,
            )
        )
        return True
    if opts.mode == "tutorial-demo-dqfd":
        payload["runs"].append(
            run_tutorial_demo_bc(
                out_dir,
                **tutorial_demo_bc_kwargs(opts, route_demo_variants),
                demo_action_weight=opts.demo_action_weight,
                demo_action_margin=opts.demo_action_margin,
                demo_action_batch_size=opts.demo_action_batch_size,
                demo_conservative_weight=opts.demo_conservative_weight,
                demo_conservative_temperature=opts.demo_conservative_temperature,
                label="tutorial_demo_dqfd",
            )
        )
        return True
    if opts.mode == "tutorial-demo-conservative":
        payload["runs"].append(
            run_tutorial_demo_bc(
                out_dir,
                **tutorial_demo_bc_kwargs(opts, route_demo_variants),
                demo_action_weight=opts.demo_action_weight,
                demo_action_margin=opts.demo_action_margin,
                demo_action_batch_size=opts.demo_action_batch_size,
                demo_conservative_weight=opts.demo_conservative_weight,
                demo_conservative_temperature=opts.demo_conservative_temperature,
                label="tutorial_demo_conservative",
            )
        )
        return True
    if opts.mode == "tutorial-demo-conservative-close-zone":
        payload["runs"].append(
            run_tutorial_demo_bc(
                out_dir,
                **tutorial_demo_bc_kwargs(
                    opts,
                    close_zone_route_demo_variants(opts, route_demo_variants),
                ),
                demo_action_weight=opts.demo_action_weight or 0.03,
                demo_action_margin=opts.demo_action_margin,
                demo_action_batch_size=opts.demo_action_batch_size,
                demo_conservative_weight=opts.demo_conservative_weight or 0.02,
                demo_conservative_temperature=opts.demo_conservative_temperature,
                close_zone_extra_demo_action_weight=opts.close_zone_extra_action_weight,
                close_zone_extra_label_source=opts.close_zone_extra_label_source,
                close_zone_demo_distance=opts.close_zone_demo_distance,
                oracle_close_zone_stride=opts.oracle_close_zone_stride,
                oracle_close_zone_max_per_trajectory=(opts.oracle_close_zone_max_per_trajectory),
                label="tutorial_demo_conservative_close_zone",
            )
        )
        return True
    if opts.mode == "tutorial-demo-oracle-close-zone":
        payload["runs"].append(
            run_tutorial_demo_bc(
                out_dir,
                **tutorial_demo_bc_kwargs(
                    opts,
                    close_zone_route_demo_variants(opts, route_demo_variants),
                ),
                demo_action_weight=opts.demo_action_weight or 0.03,
                demo_action_margin=opts.demo_action_margin,
                demo_action_batch_size=opts.demo_action_batch_size,
                demo_conservative_weight=opts.demo_conservative_weight or 0.02,
                demo_conservative_temperature=opts.demo_conservative_temperature,
                close_zone_extra_demo_action_weight=opts.close_zone_extra_action_weight,
                close_zone_extra_label_source="oracle",
                close_zone_demo_distance=opts.close_zone_demo_distance,
                oracle_close_zone_stride=opts.oracle_close_zone_stride,
                oracle_close_zone_max_per_trajectory=(opts.oracle_close_zone_max_per_trajectory),
                label="tutorial_demo_oracle_close_zone",
            )
        )
        return True
    if opts.mode == "tutorial-demo-close-zone":
        payload["runs"].append(
            run_tutorial_demo_bc(
                out_dir,
                **tutorial_demo_bc_kwargs(opts, route_demo_variants),
                demo_action_weight=opts.close_zone_demo_action_weight,
                demo_action_margin=opts.demo_action_margin,
                demo_action_batch_size=opts.demo_action_batch_size,
                demo_conservative_weight=opts.demo_conservative_weight,
                demo_conservative_temperature=opts.demo_conservative_temperature,
                close_zone_demo_action=True,
                close_zone_demo_distance=opts.close_zone_demo_distance,
                label="tutorial_demo_close_zone",
            )
        )
        return True
    if opts.mode == "tutorial-demo-bridge-finetune":
        payload["runs"].extend(
            run_tutorial_demo_bridge_finetune(
                out_dir,
                episodes=opts.episodes,
                bridge_finetune_episodes=opts.bridge_finetune_episodes,
                seed=opts.seed,
                eval_games=opts.eval_games,
                trace_games=opts.trace_eval_games,
                trace_max_steps=opts.trace_max_steps,
                trace_sample_every=opts.trace_sample_every,
                trace_tail_steps=opts.trace_tail_steps,
                train_eval_games=opts.train_eval_games,
                eval_every=opts.eval_every,
                log_every=opts.log_every,
                report_seconds=opts.report_seconds,
                heartbeat_seconds=opts.heartbeat_seconds,
                vec_envs=opts.vec_envs,
                save_checkpoints=opts.save_checkpoints,
                cave_pool_size=opts.cave_pool_size,
                selected_eval_games=opts.selected_eval_games,
                route_demo_levels=opts.route_demo_levels,
                route_demo_max_steps=opts.route_demo_max_steps,
                route_demo_variants=route_demo_variants,
                bc_epochs=opts.bc_epochs,
                bc_batch_size=opts.bc_batch_size,
                demo_repeat=opts.demo_repeat,
                bridge_ratio=opts.interleave_bridge_ratio,
                bridge_envs_override=opts.interleave_bridge_envs,
            )
        )
        return True
    return False


def _run_baseline_and_transfer_mode(
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
) -> bool:
    if opts.mode == "baseline-and-transfer":
        baseline = run_baseline(
            out_dir,
            episodes=opts.episodes,
            seed=opts.seed,
            eval_games=opts.eval_games,
            train_eval_games=opts.train_eval_games,
            eval_every=opts.eval_every,
            log_every=opts.log_every,
            report_seconds=opts.report_seconds,
            heartbeat_seconds=opts.heartbeat_seconds,
            vec_envs=opts.vec_envs,
            save_checkpoints=opts.save_checkpoints,
        )
        payload["runs"].append(baseline)
        transfer_runs = run_transfer(
            out_dir,
            episodes=opts.episodes,
            drill_episodes=opts.drill_episodes,
            seed=opts.seed,
            eval_games=opts.eval_games,
            eval_k=opts.eval_k,
            train_eval_games=opts.train_eval_games,
            eval_every=opts.eval_every,
            log_every=opts.log_every,
            report_seconds=opts.report_seconds,
            heartbeat_seconds=opts.heartbeat_seconds,
            vec_envs=opts.vec_envs,
            save_checkpoints=opts.save_checkpoints,
            drill_eval_max_steps=opts.drill_eval_max_steps,
        )
        payload["runs"].extend(transfer_runs)
        payload["comparison"] = compare_runs(baseline, transfer_runs[-1])
        return True
    return False


def main() -> None:
    _configure_line_buffering()
    if _handle_list_recipes_command(sys.argv):
        return
    sys.argv = _expand_run_recipe_command(sys.argv)

    parser, opts, route_demo_variants = _parse_status_session_args()
    out_dir, payload = _new_status_session_payload(opts)

    try:
        if run_label_dataset_mode(parser, opts, out_dir, payload):
            pass
        elif _run_checkpoint_correction_mode(parser, opts, out_dir, payload):
            pass
        elif _run_baseline_diagnostic_mode(opts, out_dir, payload):
            pass
        elif _run_mixed_reset_mode(parser, opts, out_dir, payload):
            pass
        elif _run_skill_transfer_mode(opts, out_dir, payload):
            pass
        elif _run_route_mode(opts, out_dir, payload, route_demo_variants):
            pass
        elif _run_tutorial_demo_mode(opts, out_dir, payload, route_demo_variants):
            pass
        elif _run_baseline_and_transfer_mode(opts, out_dir, payload):
            pass
        else:
            raise AssertionError(f"unhandled status-session mode: {opts.mode}")
    except KeyboardInterrupt:
        wrote_partial = append_interrupted_run_from_live_metrics(out_dir, payload)
        if wrote_partial:
            print("\nInterrupted; writing partial status-session artifacts.", file=sys.stderr)
            _write_status_session_outputs(out_dir, opts, payload)
        raise SystemExit(130) from None
    else:
        _write_status_session_outputs(out_dir, opts, payload)
