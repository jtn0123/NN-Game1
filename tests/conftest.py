"""
Pytest configuration for the test suite.

This file is automatically loaded by pytest and applies configuration
to all tests in the tests/ directory.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers if needed
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Filter expected warnings that would otherwise clutter test output
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    pass


# Configure warning filters
pytest_plugins = []


@pytest.fixture(autouse=True)
def suppress_config_warning():
    """Suppress the MAX_EPISODES=0 warning during tests.

    This warning is useful for production to remind users that training
    will run indefinitely, but it's expected behavior in tests where
    we use the default Config().
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="MAX_EPISODES is 0",
            category=UserWarning
        )
        yield
