"""Neural-network visualization publishing helpers."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.web.metrics_types import LayerAnalysisData, NeuronInspectionData


class MetricsPublisherNNMixin:
    def update_nn_visualization(
        self: Any,
        layer_info: List[Dict[str, Any]],
        activations: Dict[str, List[float]],
        q_values: List[float],
        selected_action: int,
        weights: List[List[List[float]]],
        step: int,
        action_labels: Optional[List[str]] = None,
    ) -> None:
        """Update neural network visualization data."""

        current_time = time.time()
        if not self.should_update_nn_visualization(current_time):
            return

        self._last_nn_update_time = current_time
        self._nn_data.layer_info = layer_info
        self._nn_data.activations = activations
        self._nn_data.q_values = q_values
        self._nn_data.selected_action = selected_action
        self._nn_data.weights = weights
        self._nn_data.step = step
        if action_labels:
            self._nn_data.action_labels = action_labels

        nn_dict = self._nn_data.to_dict(include_weights=False)
        with self._callback_lock:
            callbacks = self._on_nn_update_callbacks.copy()
        for callback in callbacks:
            callback(nn_dict)

    def should_update_nn_visualization(self: Any, current_time: Optional[float] = None) -> bool:
        """Return whether a neural-network visualization update should run."""

        check_time = time.time() if current_time is None else current_time
        return check_time - self._last_nn_update_time >= self._nn_update_interval

    def get_nn_visualization(self: Any, include_weights: bool = False) -> Dict[str, Any]:
        """Get current neural-network visualization data."""

        return self._nn_data.to_dict(include_weights=include_weights)

    def on_nn_update(self: Any, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for neural-network visualization updates."""

        with self._callback_lock:
            self._on_nn_update_callbacks.append(callback)

    def update_neuron_inspection(
        self: Any,
        layer_idx: int,
        neuron_idx: int,
        layer_name: str,
        current_activation: float,
        activation_history: Optional[List[float]] = None,
        incoming_weights: Optional[List[float]] = None,
        outgoing_weights: Optional[List[float]] = None,
        q_contributions: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update neuron inspection data."""

        key = (layer_idx, neuron_idx)
        if key not in self._neuron_inspection_data:
            self._neuron_inspection_data[key] = NeuronInspectionData(
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
                layer_name=layer_name,
            )

        data = self._neuron_inspection_data[key]
        data.current_activation = current_activation

        if activation_history is not None:
            data.activation_history = list(activation_history)[-500:]

        if incoming_weights is not None:
            incoming_weight_array = np.asarray(incoming_weights)
            if incoming_weight_array.size > 0:
                data.incoming_weights = incoming_weight_array.tolist()
                data.incoming_weight_stats = {
                    "mean": float(np.mean(incoming_weight_array)),
                    "std": float(np.std(incoming_weight_array)),
                    "min": float(np.min(incoming_weight_array)),
                    "max": float(np.max(incoming_weight_array)),
                }

        if outgoing_weights is not None:
            outgoing_weight_array = np.asarray(outgoing_weights)
            if outgoing_weight_array.size > 0:
                data.outgoing_weights = outgoing_weight_array.tolist()
                data.outgoing_weight_stats = {
                    "mean": float(np.mean(outgoing_weight_array)),
                    "std": float(np.std(outgoing_weight_array)),
                    "min": float(np.min(outgoing_weight_array)),
                    "max": float(np.max(outgoing_weight_array)),
                }

        if q_contributions:
            data.q_value_contributions = q_contributions

    def get_neuron_details(self: Any, layer_idx: int, neuron_idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific neuron."""

        key = (layer_idx, neuron_idx)
        if key not in self._neuron_inspection_data:
            return {"error": "Neuron not found"}
        return self._neuron_inspection_data[key].to_dict()

    def update_layer_analysis(
        self: Any,
        layer_idx: int,
        layer_name: str,
        neuron_count: int,
        activations: np.ndarray,
        weights: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None,
    ) -> None:
        """Update layer analysis statistics."""

        if layer_idx not in self._layer_analysis_data:
            self._layer_analysis_data[layer_idx] = LayerAnalysisData(
                layer_idx=layer_idx,
                layer_name=layer_name,
                neuron_count=neuron_count,
            )

        data = self._layer_analysis_data[layer_idx]
        activations = np.asarray(activations)
        if activations.size > 0:
            data.avg_activation = float(np.mean(activations))
            data.activation_std = float(np.std(activations))
            data.activation_min = float(np.min(activations))
            data.activation_max = float(np.max(activations))
            data.dead_neuron_count = int(np.sum(np.abs(activations) < 0.01))
            data.saturated_neuron_count = int(np.sum(np.abs(activations) > 0.95))
            hist, _ = np.histogram(activations, bins=20)
            data.activation_histogram = hist.tolist()
        else:
            data.avg_activation = 0.0
            data.activation_std = 0.0
            data.activation_min = 0.0
            data.activation_max = 0.0
            data.dead_neuron_count = 0
            data.saturated_neuron_count = 0
            data.activation_histogram = [0] * 20

        if weights is not None:
            flat_weights = np.asarray(weights).flatten()
            if flat_weights.size > 0:
                data.weight_mean = float(np.mean(flat_weights))
                data.weight_std = float(np.std(flat_weights))
                data.weight_min = float(np.min(flat_weights))
                data.weight_max = float(np.max(flat_weights))
                hist, _ = np.histogram(flat_weights, bins=20)
                data.weight_histogram = hist.tolist()
            else:
                data.weight_mean = 0.0
                data.weight_std = 0.0
                data.weight_min = 0.0
                data.weight_max = 0.0
                data.weight_histogram = [0] * 20

        if gradients is not None:
            flat_grads = np.asarray(gradients).flatten()
            if flat_grads.size > 0:
                data.gradient_mean = float(np.mean(np.abs(flat_grads)))
                data.gradient_std = float(np.std(flat_grads))
                data.gradient_max_magnitude = float(np.max(np.abs(flat_grads)))
            else:
                data.gradient_mean = 0.0
                data.gradient_std = 0.0
                data.gradient_max_magnitude = 0.0

    def get_layer_analysis(self: Any, layer_idx: int) -> Dict[str, Any]:
        """Get analysis data for a specific layer."""

        if layer_idx not in self._layer_analysis_data:
            return {"error": "Layer not found"}
        return self._layer_analysis_data[layer_idx].to_dict()

    def get_all_layer_analysis(self: Any) -> List[Dict[str, Any]]:
        """Get analysis for all layers."""

        return [
            data.to_dict()
            for data in sorted(self._layer_analysis_data.values(), key=lambda x: x.layer_idx)
        ]

    def on_neuron_select(self: Any, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for neuron selection."""

        with self._callback_lock:
            self._on_neuron_select_callbacks.append(callback)

    def on_layer_analysis(self: Any, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for layer analysis updates."""

        with self._callback_lock:
            self._on_layer_analysis_callbacks.append(callback)
