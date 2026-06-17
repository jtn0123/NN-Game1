"""Shared runtime helpers for interactive and headless training."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from inspect import signature
from typing import Any, Callable, Dict, List, Optional

import numpy as np

NN_ANALYSIS_INTERVAL = 100


@dataclass
class NNSnapshot:
    layer_info: List[Dict[str, Any]]
    activations: Dict[str, List[float]]
    q_values: List[float]
    weights: List[List[List[float]]]
    action_labels: List[str]
    input_state: List[float]
    analysis_activations: Dict[str, List[float]]
    analysis_weights: List[List[List[float]]]


def resolve_model_path(
    explicit_path: Optional[str],
    state_size: int,
    action_size: int,
    config: Any,
    inspect_model: Callable[..., Optional[Dict[str, Any]]],
) -> Optional[str]:
    """Resolve an explicit or latest compatible game checkpoint path."""
    model_root = str(config.MODEL_DIR)
    model_dir = str(config.GAME_MODEL_DIR)
    trusted_dirs = [model_root, model_dir]

    if explicit_path and os.path.exists(explicit_path):
        info = inspect_model(
            explicit_path,
            trusted_dirs=trusted_dirs,
            allow_unsafe_fallback=True,
        )
        if info and info.get("state_size") == state_size and info.get("action_size") == action_size:
            return explicit_path

        print(f"⚠️  Specified model incompatible: {os.path.basename(explicit_path)}")
        print(f"   Expected: state_size={state_size}, action_size={action_size}")
        if info:
            print(
                f"   Model has: state_size={info.get('state_size')}, action_size={info.get('action_size')}"
            )
        return None

    if not os.path.exists(model_dir):
        return None

    model_files = [
        (
            os.path.join(model_dir, filename),
            os.path.getmtime(os.path.join(model_dir, filename)),
        )
        for filename in os.listdir(model_dir)
        if filename.endswith(".pth")
    ]
    if not model_files:
        return None

    model_files.sort(key=lambda item: item[1], reverse=True)
    for model_path, _ in model_files:
        info = inspect_model(
            model_path,
            trusted_dirs=trusted_dirs,
            allow_unsafe_fallback=True,
        )
        if info and info.get("state_size") == state_size and info.get("action_size") == action_size:
            print(f"📂 Auto-loading most recent compatible save: {os.path.basename(model_path)}")
            return model_path

        print(f"⚠️  Skipping incompatible save: {os.path.basename(model_path)}")

    print("⚠️  No compatible saved model found for this game")
    return None


def request_save_and_stop(
    *,
    game_name: str,
    save_model: Callable[[str, str], bool],
    set_running: Callable[[bool], None],
    dashboard: Optional[Any] = None,
) -> None:
    """Shared save-and-quit behavior for training runtimes."""
    if dashboard:
        dashboard.log("💾 Saving model before shutdown...", "warning")

    save_success = save_model(f"{game_name}_final.pth", "shutdown")

    if save_success:
        if dashboard:
            dashboard.log("✅ Model saved. Shutting down...", "success")
        print("\n👋 Save & Quit requested. Model saved. Exiting...")
    else:
        if dashboard:
            dashboard.log("⚠️ Save may have failed. Shutting down...", "warning")
        print("\n⚠️ Save & Quit requested. Save may have failed. Exiting...")

    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.5)
    set_running(False)


def is_new_best_score(score: int, best_score: int) -> bool:
    """Return whether an episode score should be treated as a new best."""
    return score > best_score


def should_emit_episode_metrics(
    episode: int,
    is_new_best: bool,
    *,
    warmup_episodes: int = 10,
    interval: int = 5,
) -> bool:
    """Return whether an episode should be emitted to the live dashboard."""
    if episode <= warmup_episodes:
        return True
    if is_new_best:
        return True
    return interval > 0 and episode % interval == 0


def build_nn_snapshot(
    agent: Any,
    game: Any,
    state: np.ndarray,
    *,
    include_analysis: bool = True,
) -> NNSnapshot:
    """Build sampled and full neural-network dashboard payloads once."""
    agent.policy_net.capture_activations = True
    try:
        layer_info = agent.policy_net.get_layer_info()
        q_values = agent.get_q_values(state)
        raw_activations = agent.policy_net.get_activations()
        raw_weights = agent.policy_net.get_weights()
    finally:
        agent.policy_net.capture_activations = False

    max_neurons = 15
    activations: Dict[str, List[float]] = {}
    analysis_activations: Dict[str, List[float]] = {}
    for key, activation in raw_activations.items():
        current = activation[0] if len(activation.shape) > 1 else activation
        if include_analysis:
            analysis_activations[key] = current.tolist()
        activations[key] = current[: min(max_neurons, len(current))].tolist()

    sampled_weights: List[List[List[float]]] = []
    analysis_weights: List[List[List[float]]] = []
    for weight in raw_weights:
        if weight is None:
            continue
        sampled = weight[: min(max_neurons, weight.shape[0]), : min(max_neurons, weight.shape[1])]
        sampled_weights.append(sampled.tolist())
        if include_analysis:
            analysis_weights.append(weight.tolist())

    action_labels = ["LEFT", "STAY", "RIGHT"]
    if hasattr(game, "get_action_labels"):
        action_labels = game.get_action_labels()

    return NNSnapshot(
        layer_info=layer_info,
        activations=activations,
        q_values=q_values.tolist(),
        weights=sampled_weights,
        action_labels=action_labels,
        input_state=state.tolist(),
        analysis_activations=analysis_activations,
        analysis_weights=analysis_weights,
    )


def should_include_nn_analysis(step: int, interval: int = NN_ANALYSIS_INTERVAL) -> bool:
    """Return whether a live NN update should include full inspection payloads."""
    if interval <= 0:
        return True
    return step <= 0 or step % interval == 0


@lru_cache(maxsize=32)
def _snapshot_builder_accepts_include_analysis(
    snapshot_builder: Callable[..., NNSnapshot],
) -> bool:
    return "include_analysis" in signature(snapshot_builder).parameters


def build_runtime_nn_snapshot(
    agent: Any,
    game: Any,
    state: np.ndarray,
    *,
    step: int,
    snapshot_builder: Callable[..., NNSnapshot] = build_nn_snapshot,
    analysis_interval: int = NN_ANALYSIS_INTERVAL,
) -> NNSnapshot:
    """Build a dashboard NN snapshot, including full analysis data only periodically."""
    include_analysis = should_include_nn_analysis(step, analysis_interval)
    try:
        accepts_include_analysis = _snapshot_builder_accepts_include_analysis(snapshot_builder)
    except (TypeError, ValueError):
        accepts_include_analysis = False

    if accepts_include_analysis:
        return snapshot_builder(agent, game, state, include_analysis=include_analysis)
    return snapshot_builder(agent, game, state)


def emit_nn_snapshot_to_dashboard(
    dashboard: Any,
    snapshot: NNSnapshot,
    *,
    selected_action: int,
    step: int,
) -> None:
    """Emit a prepared neural-network snapshot using the dashboard contract."""
    dashboard.emit_nn_visualization(
        layer_info=snapshot.layer_info,
        activations=snapshot.activations,
        q_values=snapshot.q_values,
        selected_action=selected_action,
        weights=snapshot.weights,
        step=step,
        action_labels=snapshot.action_labels,
        input_state=snapshot.input_state,
        analysis_activations=snapshot.analysis_activations or None,
        analysis_weights=snapshot.analysis_weights or None,
    )
