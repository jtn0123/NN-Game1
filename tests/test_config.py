"""
Tests for Config validation.

These tests verify that invalid configurations are caught early
rather than causing cryptic runtime errors during training.
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class TestConfigValidation:
    """Test Config.__post_init__ validation."""

    def test_valid_config_passes(self):
        """Default config should validate without errors."""
        cfg = Config()
        # Should not raise - validation happens in __post_init__
        assert cfg is not None

    def test_invalid_learning_rate_zero(self):
        """LEARNING_RATE=0 should fail validation."""
        cfg = Config()
        cfg.LEARNING_RATE = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_learning_rate_negative(self):
        """Negative LEARNING_RATE should fail validation."""
        cfg = Config()
        cfg.LEARNING_RATE = -0.001
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_validation_runs_under_optimized_python(self):
        """Config validation should not disappear when Python runs with -O."""
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                "-c",
                "from config import Config; Config(LEARNING_RATE=-1)",
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0
        assert "ValueError" in result.stderr

    def test_invalid_gamma_zero(self):
        """GAMMA=0 should fail validation (must be > 0)."""
        cfg = Config()
        cfg.GAMMA = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_gamma_exceeds_one(self):
        """GAMMA > 1 should fail validation."""
        cfg = Config()
        cfg.GAMMA = 1.5
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_valid_gamma_exactly_one(self):
        """GAMMA=1.0 should be valid (no discounting)."""
        cfg = Config()
        cfg.GAMMA = 1.0
        cfg.__post_init__()  # Should not raise

    def test_invalid_batch_size_zero(self):
        """BATCH_SIZE=0 should fail validation."""
        cfg = Config()
        cfg.BATCH_SIZE = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_batch_exceeds_memory(self):
        """BATCH_SIZE > MEMORY_SIZE should fail validation."""
        cfg = Config()
        cfg.BATCH_SIZE = 1000
        cfg.MEMORY_SIZE = 100
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_epsilon_inverted(self):
        """EPSILON_START < EPSILON_END should fail validation."""
        cfg = Config()
        cfg.EPSILON_START = 0.01
        cfg.EPSILON_END = 0.5
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_valid_epsilon_equal(self):
        """EPSILON_START == EPSILON_END should be valid (no decay)."""
        cfg = Config()
        cfg.EPSILON_START = 0.1
        cfg.EPSILON_END = 0.1
        cfg.__post_init__()  # Should not raise

    def test_invalid_learn_every_zero(self):
        """LEARN_EVERY=0 should fail validation."""
        cfg = Config()
        cfg.LEARN_EVERY = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_gradient_steps_zero(self):
        """GRADIENT_STEPS=0 should fail validation."""
        cfg = Config()
        cfg.GRADIENT_STEPS = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_max_steps_zero(self):
        """MAX_STEPS_PER_EPISODE=0 should fail validation."""
        cfg = Config()
        cfg.MAX_STEPS_PER_EPISODE = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_screen_width_zero(self):
        """SCREEN_WIDTH=0 should fail validation."""
        cfg = Config()
        cfg.SCREEN_WIDTH = 0
        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_hidden_layers_empty(self):
        """Empty HIDDEN_LAYERS should fail validation."""
        cfg = Config()
        cfg.HIDDEN_LAYERS = []
        with pytest.raises(ValueError):
            cfg.__post_init__()

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("MEMORY_SIZE", 0),
            ("TARGET_UPDATE", 0),
            ("GRAD_CLIP", 0),
            ("MAX_EPISODES", -1),
            ("SAVE_EVERY", 0),
            ("RENDER_EVERY", -1),
            ("LOG_EVERY", 0),
            ("PLOT_HISTORY_LENGTH", 0),
            ("EVAL_EVERY", -1),
            ("EVAL_EPISODES", 0),
            ("EVAL_MAX_STEPS", 0),
            ("EVAL_PLATEAU_THRESHOLD", 0),
            ("EVAL_PLATEAU_EPSILON_BOOST", -0.1),
            ("EVAL_PLATEAU_EPSILON_BOOST", 1.1),
            ("EVAL_PLATEAU_BOOST_EPISODES", 0),
            ("PER_ALPHA", -0.1),
            ("PER_BETA_START", 0),
            ("PER_BETA_FRAMES", 0),
            ("N_STEP_SIZE", 0),
        ],
    )
    def test_invalid_runtime_numeric_settings(self, field_name, bad_value):
        """Runtime settings that feed loops/buffers should reject unsafe values."""
        cfg = Config()
        setattr(cfg, field_name, bad_value)

        with pytest.raises(ValueError):
            cfg.__post_init__()

    def test_invalid_hidden_layer_size(self):
        """Network hidden layers should have positive integer sizes."""
        cfg = Config()
        cfg.HIDDEN_LAYERS = [128, 0]

        with pytest.raises(ValueError):
            cfg.__post_init__()


class TestConfigDevice:
    """Test Config device selection."""

    def test_force_cpu_returns_cpu(self):
        """FORCE_CPU=True should return CPU device."""
        cfg = Config()
        cfg.FORCE_CPU = True
        assert cfg.DEVICE.type == "cpu"

    def test_device_is_valid(self):
        """Default DEVICE should be a valid torch device."""
        cfg = Config()

        # Should be one of: cpu, cuda, mps
        assert cfg.DEVICE.type in ["cpu", "cuda", "mps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
