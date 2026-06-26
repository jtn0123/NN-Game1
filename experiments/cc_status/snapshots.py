# ruff: noqa: F401,F403,F405,I001
from .common import *
from .evals import *
from .scorecard import route_contact_source_snapshot_score


def bridge_eval_snapshot(
    trainer: HeadlessTrainer,
    config: Config,
    *,
    eval_k: int,
    max_steps: int | None,
) -> dict[str, Any]:
    rows = level_set_eval(
        trainer.agent,
        config,
        specs=BRIDGE_CAVES,
        k=eval_k,
        max_steps=max_steps,
    )
    return {
        "episode": int(trainer.current_episode),
        "bridge_eval": rows,
        "rollup": level_eval_rollup(rows),
    }


def bridge_snapshot_line(snapshot: dict[str, Any]) -> str:
    rollup = snapshot["rollup"]
    return (
        f"[bridge eval] ep {snapshot['episode']} | "
        f"w {100 * rollup['mean_win_rate']:.0f}% | "
        f"any {100 * rollup['mean_any_crystal_rate']:.0f}% | "
        f"all {100 * rollup['mean_all_crystals_rate']:.0f}% | "
        f"prog {rollup['mean_progress']:.3f} | "
        f"solved {rollup['solved_levels']}/{rollup['levels']}"
    )


def bridge_snapshot_score(snapshot: dict[str, Any]) -> tuple[float, float, float, float, int]:
    rollup = snapshot.get("rollup") or {}
    return (
        float(rollup.get("mean_win_rate", 0.0) or 0.0),
        float(rollup.get("mean_all_crystals_rate", 0.0) or 0.0),
        float(rollup.get("mean_progress", 0.0) or 0.0),
        float(rollup.get("mean_any_crystal_rate", 0.0) or 0.0),
        int(snapshot.get("episode", 0) or 0),
    )


def selected_bridge_snapshot(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    return max(history, key=bridge_snapshot_score)


def source_eval_snapshot(
    trainer: HeadlessTrainer,
    config: Config,
    *,
    run_dir: Path,
    label: str,
    games: int,
) -> dict[str, Any]:
    eval_payload = final_eval(
        config,
        trainer.agent,
        out_dir=run_dir,
        label=f"{label}_ep{trainer.current_episode}",
        episode=trainer.current_episode,
        games=games,
    )
    return {
        "episode": int(trainer.current_episode),
        "source_eval": eval_payload,
    }


def source_snapshot_score(snapshot: dict[str, Any]) -> tuple[float, float, float, int]:
    return route_contact_source_snapshot_score(snapshot)


def source_snapshot_line(snapshot: dict[str, Any]) -> str:
    eval_payload = snapshot["source_eval"]
    return (
        f"[source eval] ep {snapshot['episode']} | "
        f"w {100 * eval_payload.get('win_rate', 0):.0f}% | "
        f"c {100 * eval_payload.get('mean_crystal_frac', 0):.0f}% | "
        f"d {100 * eval_payload.get('mean_depth_frac', 0):.0f}% | "
        f"score {eval_payload.get('mean_score', 0):.0f}"
    )
