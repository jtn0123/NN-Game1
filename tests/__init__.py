"""
Tests for Neural Network Game AI
================================

Run all tests:
    pytest tests/

Run with coverage:
    pytest tests/ --cov=src --cov-report=html
"""

# Suppress pygame's pkg_resources deprecation warning (pygame issue #4557)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

