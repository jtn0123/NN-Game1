"""
Pytest configuration for the test suite.

This file is automatically loaded by pytest and applies configuration
to all tests in the tests/ directory.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers and warning filters."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

    # Filter expected warnings at the pytest level (works during import time)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:MAX_EPISODES is 0:UserWarning"
    )

    # Suppress pygame's pkg_resources deprecation warning (external dependency)
    config.addinivalue_line(
        "filterwarnings",
        "ignore:pkg_resources is deprecated:UserWarning"
    )
