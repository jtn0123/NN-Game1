"""
Pytest configuration for the test suite.

This file is automatically loaded by pytest and applies configuration
to all tests in the tests/ directory.
"""

import os
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed all RNGs before every test for reproducible, non-flaky runs.

    RL tests lean on random sampling (epsilon-greedy, replay sampling, weight
    init); seeding torch/numpy/random per test removes order- and run-dependent
    flakiness and gives a stable baseline for determinism assertions.
    """
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        import torch

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except ImportError:
        pass
    yield


def pytest_configure(config):
    """Configure pytest with custom markers and warning filters."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

    # Filter expected warnings at the pytest level (works during import time)
    config.addinivalue_line("filterwarnings", "ignore:MAX_EPISODES is 0:UserWarning")

    # Suppress pygame's pkg_resources deprecation warning (external dependency)
    config.addinivalue_line("filterwarnings", "ignore:pkg_resources is deprecated:UserWarning")
