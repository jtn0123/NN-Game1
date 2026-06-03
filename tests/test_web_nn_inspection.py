"""Neural-network inspection tests for the web dashboard."""

import json
import os

import numpy as np
import pytest
import torch

try:
    from src.web.server import FLASK_AVAILABLE

    WEB_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not WEB_AVAILABLE, reason="Flask/SocketIO not installed")


class TestWebDashboardNNInspection:

    @pytest.fixture
    def web_dashboard(self):
        """Create a WebDashboard instance for testing."""
        try:
            from src.web.server import WebDashboard
            from config import Config

            config = Config()
            config.GAME_NAME = "breakout"

            dashboard = WebDashboard(port=5099, config=config)
            yield dashboard

            # Cleanup
            try:
                dashboard.stop()
            except Exception:
                pass
        except ImportError:
            pytest.skip("WebDashboard not available")

    def test_api_layers_endpoint(self, web_dashboard):
        """GET /api/layers should return all layer analysis data."""
        web_dashboard.publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="input",
            neuron_count=2,
            activations=np.array([0.1, 0.2], dtype=np.float32),
        )

        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get(
                "/api/layers",
                headers={"X-Dashboard-Token": web_dashboard.access_token},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            assert isinstance(data, list)
            assert data[0]["layer_idx"] == 0
            assert data[0]["layer_name"] == "input"

    def test_emit_nn_visualization_updates_phase2_inspection(self, web_dashboard):
        """Live NN snapshots should populate layer and neuron inspection endpoints."""
        layer_info = [
            {"name": "Input", "neurons": 2, "type": "input"},
            {"name": "Hidden 1", "neurons": 2, "type": "hidden"},
            {"name": "Output", "neurons": 2, "type": "output"},
        ]

        web_dashboard.emit_nn_visualization(
            layer_info=layer_info,
            activations={"layer_0": [0.5, 0.6], "layer_1": [0.7, 0.8]},
            q_values=[0.7, 0.8],
            selected_action=1,
            weights=[
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            step=10,
            action_labels=["LEFT", "RIGHT"],
            input_state=[0.9, 0.1],
        )

        hidden = web_dashboard.publisher.get_layer_analysis(1)
        assert hidden["layer_name"] == "Hidden 1"
        assert hidden["avg_activation"] == pytest.approx(0.55)

        hidden_neuron = web_dashboard.publisher.get_neuron_details(1, 0)
        assert hidden_neuron["current_activation"] == pytest.approx(0.5)
        assert hidden_neuron["incoming_weights"] == [0.1, 0.2]
        assert hidden_neuron["outgoing_weights"] == [1.0, 3.0]

        output_neuron = web_dashboard.publisher.get_neuron_details(2, 0)
        assert output_neuron["q_value_contributions"] == {"LEFT": 0.7}

    def test_emit_nn_visualization_maps_dueling_stream_weights(self, web_dashboard):
        """Dueling stream output weights should map to Phase 2 inspection data."""
        layer_info = [
            {"name": "Input", "neurons": 2, "type": "input"},
            {"name": "Shared 1", "neurons": 2, "type": "hidden"},
            {"name": "Value", "neurons": 2, "type": "value_stream"},
            {"name": "Advantage", "neurons": 2, "type": "advantage_stream"},
            {"name": "Output (Q)", "neurons": 2, "type": "output"},
        ]

        web_dashboard.emit_nn_visualization(
            layer_info=layer_info,
            activations={
                "layer_0": [0.5, 0.6],
                "value_hidden": [0.7, 0.8],
                "advantage_hidden": [0.9, 1.0],
                "layer_output": [123.0],
            },
            q_values=[1.1, 1.2],
            selected_action=1,
            weights=[
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0]],
                [[11.0, 12.0], [13.0, 14.0]],
            ],
            step=20,
            action_labels=["LEFT", "RIGHT"],
            input_state=[0.1, 0.2],
        )

        value_neuron = web_dashboard.publisher.get_neuron_details(2, 0)
        advantage_neuron = web_dashboard.publisher.get_neuron_details(3, 0)
        output_neuron = web_dashboard.publisher.get_neuron_details(4, 1)

        assert value_neuron["incoming_weights"] == [1.0, 2.0]
        assert value_neuron["outgoing_weights"] == [9.0]
        assert advantage_neuron["incoming_weights"] == [5.0, 6.0]
        assert advantage_neuron["outgoing_weights"] == [11.0, 13.0]
        assert output_neuron["incoming_weights"] == [9.0, 10.0, 13.0, 14.0]
        assert output_neuron["q_value_contributions"] == {"RIGHT": 1.2}

    def test_emit_nn_visualization_skips_phase2_work_when_throttled(
        self, web_dashboard, monkeypatch
    ):
        """Phase 2 inspection should not run when NN visualization is throttled."""
        calls = []
        original_sync = web_dashboard._sync_phase2_inspection

        def counting_sync(*args, **kwargs):
            calls.append(1)
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(web_dashboard, "_sync_phase2_inspection", counting_sync)
        web_dashboard.publisher._nn_update_interval = 10.0

        payload = {
            "layer_info": [
                {"name": "Input", "neurons": 1, "type": "input"},
                {"name": "Output", "neurons": 1, "type": "output"},
            ],
            "activations": {"layer_0": [0.5]},
            "q_values": [0.5],
            "selected_action": 0,
            "weights": [[[0.1]]],
            "action_labels": ["STAY"],
            "input_state": [0.25],
        }

        web_dashboard.emit_nn_visualization(step=1, **payload)
        web_dashboard.emit_nn_visualization(step=2, **payload)

        assert len(calls) == 1
