"""Route and page tests for the web dashboard."""

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


class TestWebDashboardRoutes:

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

    def test_initialization(self, web_dashboard):
        """WebDashboard should initialize with correct config."""
        assert web_dashboard.port == 5099
        assert web_dashboard.host == "127.0.0.1"
        assert web_dashboard.access_token
        assert web_dashboard.config.GAME_NAME == "breakout"
        assert web_dashboard.publisher is not None

    def test_dashboard_url_uses_openable_localhost_for_wildcard_bind(self):
        """A wildcard bind address should not be printed as the browser URL."""
        from src.web.server import WebDashboard
        from config import Config

        dashboard = WebDashboard(port=5108, host="0.0.0.0", config=Config())

        assert dashboard.dashboard_url().startswith("http://127.0.0.1:5108/?token=")

    def test_emit_metrics(self, web_dashboard):
        """WebDashboard.emit_metrics should update publisher."""
        web_dashboard.emit_metrics(episode=10, score=100, epsilon=0.5, loss=0.01)

        assert web_dashboard.publisher.state.episode == 10
        assert web_dashboard.publisher.state.score == 100

    def test_log_method(self, web_dashboard):
        """WebDashboard.log should add to console log."""
        web_dashboard.log("Test message", level="info")

        assert len(web_dashboard.publisher.console_logs) >= 1
        messages = [m.message for m in web_dashboard.publisher.console_logs]
        assert "Test message" in messages

    def test_full_training_cycle(self, web_dashboard):
        """Simulate a full training cycle with metrics updates."""
        for episode in range(1, 11):
            web_dashboard.emit_metrics(
                episode=episode,
                score=episode * 10,
                epsilon=1.0 - (episode * 0.09),
                loss=1.0 / episode,
            )

        state = web_dashboard.publisher.state
        assert state.episode == 10
        assert state.score == 100
        assert state.best_score == 100

    def test_api_status_endpoint(self, web_dashboard):
        """GET /api/status should return current training state."""
        web_dashboard.emit_metrics(episode=5, score=50, epsilon=0.8, loss=0.05)

        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get(
                "/api/status",
                headers={"X-Dashboard-Token": web_dashboard.access_token},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            # Status endpoint returns nested state
            assert "state" in data or "episode" in data
            if "state" in data:
                assert data["state"]["episode"] == 5
            else:
                assert data["episode"] == 5

    def test_api_models_endpoint(self, web_dashboard):
        """GET /api/models should return available models."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get(
                "/api/models",
                headers={"X-Dashboard-Token": web_dashboard.access_token},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
        assert "models" in data
        assert "current_game" in data

    @pytest.mark.parametrize(
        "path",
        [
            "/api/status",
            "/api/config",
            "/api/games",
            "/api/models",
            "/api/save-status",
            "/api/game-stats",
            "/api/layers",
        ],
    )
    def test_read_api_routes_require_dashboard_token(self, web_dashboard, path):
        """Read APIs should not expose dashboard data to anonymous callers."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get(path)

        assert response.status_code == 401
        assert json.loads(response.data) == {"error": "Unauthorized"}

    def test_dashboard_page_requires_token_before_serving_frontend(self, web_dashboard):
        """Dashboard HTML should not bootstrap anonymous clients with the token."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get("/")

        html = response.data.decode("utf-8")

        assert response.status_code == 401
        assert web_dashboard.access_token not in html

    def test_dashboard_page_serves_tokenized_frontend_contract(self, web_dashboard):
        """Authorized dashboard HTML should expose the token and load core JS before app JS."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get(f"/?token={web_dashboard.access_token}")
            core_response = client.get("/static/dashboard_core.js")
            app_response = client.get("/static/app.js")

        html = response.data.decode("utf-8")

        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert f'<meta name="dashboard-token" content="{web_dashboard.access_token}">' in html
        assert '<meta name="referrer" content="no-referrer">' in html
        assert "Training Dashboard" in html
        assert html.index("dashboard_core.js") < html.index("app.js")
        assert core_response.status_code == 200
        assert app_response.status_code == 200

    def test_launcher_page_serves_tokenized_frontend_contract(self):
        """Launcher mode should also serve an authenticated Socket.IO page."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        dashboard = WebDashboard(port=5103, config=config, launcher_mode=True)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            unauthorized = client.get("/")
            response = client.get(f"/?token={dashboard.access_token}")

        html = response.data.decode("utf-8")
        unauthorized_html = unauthorized.data.decode("utf-8")

        assert unauthorized.status_code == 401
        assert dashboard.access_token not in unauthorized_html
        assert response.status_code == 200
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert f'<meta name="dashboard-token" content="{dashboard.access_token}">' in html
        assert '<meta name="referrer" content="no-referrer">' in html
        assert "auth: { token: DASHBOARD_TOKEN }" in html

    def test_api_models_uses_opaque_ids(self, tmp_path):
        """Model list should not expose absolute local filesystem paths."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)
        model_path = os.path.join(config.GAME_MODEL_DIR, "demo.pth")
        torch.save({"steps": 1, "epsilon": 0.5}, model_path)

        dashboard = WebDashboard(port=5101, config=config)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            response = client.get(
                "/api/models",
                headers={"X-Dashboard-Token": dashboard.access_token},
            )

        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["models"][0]["id"] == "breakout:demo.pth"
        assert "path" not in data["models"][0]

    def test_delete_model_requires_dashboard_token(self, tmp_path):
        """Mutating model routes should reject requests without the session token."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)
        model_path = os.path.join(config.GAME_MODEL_DIR, "demo.pth")
        torch.save({"steps": 1}, model_path)

        dashboard = WebDashboard(port=5102, config=config)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            unauthorized = client.delete("/api/models/breakout:demo.pth")
            authorized = client.delete(
                "/api/models/breakout:demo.pth",
                headers={"X-Dashboard-Token": dashboard.access_token},
            )

        assert unauthorized.status_code == 401
        assert authorized.status_code == 200
        assert not os.path.exists(model_path)

    def test_delete_model_hides_internal_exception_details(self, tmp_path):
        """Unexpected delete failures should not expose server internals to clients."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)

        dashboard = WebDashboard(port=5107, config=config)
        dashboard.model_service.delete = lambda _model_id: (_ for _ in ()).throw(
            RuntimeError(f"secret path: {tmp_path}")
        )
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            response = client.delete(
                "/api/models/breakout:demo.pth",
                headers={"X-Dashboard-Token": dashboard.access_token},
            )

        data = json.loads(response.data)
        assert response.status_code == 500
        assert data == {"error": "Failed to delete model"}
        assert str(tmp_path) not in response.data.decode("utf-8")

    @pytest.mark.parametrize(
        "model_id",
        [
            "breakout:../outside.pth",
            "breakout:%2e%2e/outside.pth",
            "breakout:subdir/demo.pth",
            "legacy:missing.pth",
        ],
    )
    def test_delete_model_rejects_encoded_and_malformed_ids(self, tmp_path, model_id):
        """HTTP delete route should reject traversal-shaped ids with a valid token."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)
        outside = tmp_path / "outside.pth"
        outside.write_bytes(b"keep me")

        dashboard = WebDashboard(port=5106, config=config)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            response = client.delete(
                f"/api/models/{model_id}",
                headers={"X-Dashboard-Token": dashboard.access_token},
            )

        assert response.status_code in {403, 404}
        assert outside.exists()

    def test_api_config_endpoint(self, web_dashboard):
        """GET /api/config should return configuration."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get(
                "/api/config",
                headers={"X-Dashboard-Token": web_dashboard.access_token},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            # Config returns training hyperparameters
            assert "batch_size" in data or "learning_rate" in data
            assert "device" in data

    def test_control_callback_errors_use_generic_ack(self, web_dashboard, tmp_path):
        """Socket control acks should not leak callback exception details."""
        web_dashboard.on_save_as_callback = lambda _filename: (_ for _ in ()).throw(
            RuntimeError(f"secret path: {tmp_path}")
        )

        ack = web_dashboard._handle_save_as_control({"filename": "demo.pth"})

        assert ack == {"success": False, "action": "save_as", "error": "Save failed"}

    def test_logging_during_training(self, web_dashboard):
        """Logging should work during training simulation."""
        web_dashboard.log("Training started", level="info")
        web_dashboard.emit_metrics(episode=1, score=10, epsilon=0.95, loss=0.1)
        web_dashboard.log("Episode 1 complete", level="success")

        logs = web_dashboard.publisher.console_logs
        messages = [log.message for log in logs]

        assert "Training started" in messages
        assert "Episode 1 complete" in messages
