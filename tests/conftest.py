"""
Pytest configuration for the test suite.

This file is automatically loaded by pytest and applies configuration
to all tests in the tests/ directory.
"""

import importlib.util


def _missing_package(name: str) -> bool:
    return importlib.util.find_spec(name) is None


collect_ignore = []

if _missing_package("torch"):
    collect_ignore.extend(
        [
            "test_agent.py",
            "test_asteroids.py",
            "test_config.py",
            "test_evaluator.py",
            "test_game.py",
            "test_integration.py",
            "test_network.py",
            "test_pong.py",
            "test_replay_buffer.py",
            "test_rewards.py",
            "test_snake.py",
            "test_space_invaders.py",
            "test_trainer.py",
            "test_visualizer.py",
        ]
    )

if _missing_package("flask") or _missing_package("flask_socketio") or _missing_package("torch"):
    collect_ignore.extend(
        [
            "test_phase1_improvements.py",
            "test_phase2_neuron_inspection.py",
        ]
    )


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
