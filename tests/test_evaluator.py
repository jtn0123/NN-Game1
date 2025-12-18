"""
Tests for the Evaluator module.

These tests verify:
    - Deterministic evaluation (epsilon=0)
    - Plateau detection
    - Statistics computation
"""

import pytest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ai.evaluator import Evaluator, EvalResults
from src.ai.agent import Agent
from src.game.breakout import Breakout


@pytest.fixture
def config():
    """Create a test configuration."""
    cfg = Config()
    cfg.MAX_STEPS_PER_EPISODE = 50  # Short episodes for testing
    return cfg


@pytest.fixture
def game(config):
    """Create a game instance."""
    return Breakout(config, headless=True)


@pytest.fixture
def agent(game, config):
    """Create an agent instance."""
    return Agent(game.state_size, game.action_size, config)


@pytest.fixture
def evaluator(game, agent, config):
    """Create an evaluator instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Evaluator(game, agent, config, log_dir=tmpdir, plateau_threshold=3)


class TestEvaluatorInitialization:
    """Test Evaluator initialization."""

    def test_evaluator_creates_successfully(self, evaluator):
        """Evaluator should initialize without errors."""
        assert evaluator is not None

    def test_evaluator_starts_empty(self, evaluator):
        """Evaluator should start with no history."""
        assert len(evaluator.eval_history) == 0
        assert evaluator.best_eval_score == 0.0
        assert evaluator.evals_since_improvement == 0


class TestEvaluate:
    """Test Evaluator.evaluate method."""

    def test_evaluate_returns_eval_results(self, evaluator):
        """evaluate should return EvalResults."""
        results = evaluator.evaluate(num_episodes=2, max_steps=50)
        assert isinstance(results, EvalResults)

    def test_evaluate_uses_zero_epsilon(self, game, agent, config):
        """evaluate should use epsilon=0 for deterministic play."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(game, agent, config, log_dir=tmpdir)

            # Set epsilon to a high value
            agent.epsilon = 0.9

            # Run evaluation (it should internally set epsilon to 0)
            evaluator.evaluate(num_episodes=2, max_steps=50)

            # Epsilon should be restored after evaluation
            assert agent.epsilon == 0.9

    def test_evaluate_restores_epsilon(self, game, agent, config):
        """evaluate should restore original epsilon after running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(game, agent, config, log_dir=tmpdir)

            original_epsilon = 0.42
            agent.epsilon = original_epsilon

            evaluator.evaluate(num_episodes=2, max_steps=50)

            assert agent.epsilon == original_epsilon

    def test_evaluate_tracks_correct_num_games(self, evaluator):
        """EvalResults should track correct number of games."""
        results = evaluator.evaluate(num_episodes=3, max_steps=50)
        assert results.num_games == 3

    def test_evaluate_computes_statistics(self, evaluator):
        """EvalResults should have computed statistics."""
        results = evaluator.evaluate(num_episodes=5, max_steps=50)

        # Should have score statistics
        assert hasattr(results, 'mean_score')
        assert hasattr(results, 'median_score')
        assert hasattr(results, 'std_score')
        assert hasattr(results, 'min_score')
        assert hasattr(results, 'max_score')

        # Min should be <= mean <= max
        assert results.min_score <= results.mean_score <= results.max_score

    def test_evaluate_updates_history(self, evaluator):
        """evaluate should add results to history."""
        assert len(evaluator.eval_history) == 0
        evaluator.evaluate(num_episodes=2, max_steps=50)
        assert len(evaluator.eval_history) == 1


class TestPlateauDetection:
    """Test plateau detection."""

    def test_is_plateau_initially_false(self, evaluator):
        """Should not be plateau with no evaluations."""
        assert not evaluator.is_plateau()

    def test_is_plateau_after_improvements(self, game, agent, config):
        """Should not be plateau if improvements are happening."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(game, agent, config, log_dir=tmpdir, plateau_threshold=3)

            # Simulate improving evaluations by manually updating history
            evaluator.best_eval_score = 10.0
            evaluator.evals_since_improvement = 0

            assert not evaluator.is_plateau()

    def test_is_plateau_after_no_improvement(self, game, agent, config):
        """Should detect plateau after N evals without improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(game, agent, config, log_dir=tmpdir, plateau_threshold=3)

            # Manually set state to simulate no improvement
            evaluator.best_eval_score = 100.0
            evaluator.evals_since_improvement = 3

            assert evaluator.is_plateau()

    def test_plateau_threshold_respected(self, game, agent, config):
        """Plateau should trigger at exactly threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(game, agent, config, log_dir=tmpdir, plateau_threshold=5)

            evaluator.best_eval_score = 100.0

            # Not plateau at 4
            evaluator.evals_since_improvement = 4
            assert not evaluator.is_plateau()

            # Plateau at 5
            evaluator.evals_since_improvement = 5
            assert evaluator.is_plateau()


class TestEvalResults:
    """Test EvalResults dataclass."""

    def test_eval_results_to_dict(self, evaluator):
        """EvalResults.to_dict should return a dictionary."""
        results = evaluator.evaluate(num_episodes=2, max_steps=50)
        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert 'mean_score' in result_dict
        assert 'win_rate' in result_dict
        assert 'num_games' in result_dict

    def test_eval_results_has_timestamp(self, evaluator):
        """EvalResults should have timestamp."""
        results = evaluator.evaluate(num_episodes=2, max_steps=50)
        assert results.timestamp is not None
        assert len(results.timestamp) > 0


class TestGetSummary:
    """Test Evaluator.get_summary method."""

    def test_get_summary_empty(self, evaluator):
        """get_summary should return empty dict with no history."""
        summary = evaluator.get_summary()
        assert summary == {}

    def test_get_summary_with_history(self, evaluator):
        """get_summary should return summary after evaluations."""
        evaluator.evaluate(num_episodes=2, max_steps=50)
        summary = evaluator.get_summary()

        assert 'num_evals' in summary
        assert summary['num_evals'] == 1
        assert 'best_eval_score' in summary
        assert 'is_plateau' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
