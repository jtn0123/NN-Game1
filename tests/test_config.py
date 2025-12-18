"""
Tests for Config validation.

These tests verify that invalid configurations are caught early
rather than causing cryptic runtime errors during training.
"""

import pytest
import sys
import os

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
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_learning_rate_negative(self):
        """Negative LEARNING_RATE should fail validation."""
        cfg = Config()
        cfg.LEARNING_RATE = -0.001
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_gamma_zero(self):
        """GAMMA=0 should fail validation (must be > 0)."""
        cfg = Config()
        cfg.GAMMA = 0
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_gamma_exceeds_one(self):
        """GAMMA > 1 should fail validation."""
        cfg = Config()
        cfg.GAMMA = 1.5
        with pytest.raises(AssertionError):
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
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_batch_exceeds_memory(self):
        """BATCH_SIZE > MEMORY_SIZE should fail validation."""
        cfg = Config()
        cfg.BATCH_SIZE = 1000
        cfg.MEMORY_SIZE = 100
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_epsilon_inverted(self):
        """EPSILON_START < EPSILON_END should fail validation."""
        cfg = Config()
        cfg.EPSILON_START = 0.01
        cfg.EPSILON_END = 0.5
        with pytest.raises(AssertionError):
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
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_gradient_steps_zero(self):
        """GRADIENT_STEPS=0 should fail validation."""
        cfg = Config()
        cfg.GRADIENT_STEPS = 0
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_screen_width_zero(self):
        """SCREEN_WIDTH=0 should fail validation."""
        cfg = Config()
        cfg.SCREEN_WIDTH = 0
        with pytest.raises(AssertionError):
            cfg.__post_init__()

    def test_invalid_hidden_layers_empty(self):
        """Empty HIDDEN_LAYERS should fail validation."""
        cfg = Config()
        cfg.HIDDEN_LAYERS = []
        with pytest.raises(AssertionError):
            cfg.__post_init__()


class TestConfigDevice:
    """Test Config device selection."""

    def test_force_cpu_returns_cpu(self):
        """FORCE_CPU=True should return CPU device."""
        cfg = Config()
        cfg.FORCE_CPU = True
        assert cfg.DEVICE.type == 'cpu'

    def test_device_is_valid(self):
        """Default DEVICE should be a valid torch device."""
        cfg = Config()
        import torch
        # Should be one of: cpu, cuda, mps
        assert cfg.DEVICE.type in ['cpu', 'cuda', 'mps']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
