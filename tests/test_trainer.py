"""
Tests for the Trainer module.

These tests verify:
    - TrainingMetrics tracking and statistics
    - Episode running and statistics collection
    - Trainer initialization
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ai.trainer import Trainer, TrainingMetrics, EpisodeStats
from src.ai.agent import Agent
from src.game.breakout import Breakout


@pytest.fixture
def config():
    """Create a test configuration."""
    cfg = Config()
    cfg.MAX_STEPS_PER_EPISODE = 100  # Short episodes for testing
    cfg.BATCH_SIZE = 8
    cfg.MEMORY_SIZE = 100
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
def trainer(game, agent, config):
    """Create a trainer instance."""
    return Trainer(game, agent, config)


class TestTrainingMetrics:
    """Test TrainingMetrics class."""

    def test_initialization(self):
        """Metrics should initialize with empty lists."""
        metrics = TrainingMetrics(history_length=100)
        assert len(metrics.scores) == 0
        assert len(metrics.rewards) == 0
        assert len(metrics.wins) == 0

    def test_add_episode_stats(self):
        """Adding stats should update all lists."""
        metrics = TrainingMetrics()
        stats = EpisodeStats(
            episode=0,
            score=100,
            steps=50,
            total_reward=25.5,
            epsilon=0.5,
            avg_loss=0.01,
            duration=1.5,
            bricks_broken=10,
            won=False
        )
        metrics.add(stats)
        assert len(metrics.scores) == 1
        assert metrics.scores[0] == 100
        assert metrics.rewards[0] == 25.5
        assert metrics.wins[0] is False

    def test_get_recent_average_empty(self):
        """get_recent_average should return None for empty history."""
        metrics = TrainingMetrics()
        result = metrics.get_recent_average('scores', n=100)
        assert result is None

    def test_get_recent_average_single_value(self):
        """get_recent_average should work with single value."""
        metrics = TrainingMetrics()
        stats = EpisodeStats(
            episode=0, score=100, steps=50, total_reward=25.0,
            epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=10, won=False
        )
        metrics.add(stats)
        result = metrics.get_recent_average('scores', n=100)
        assert result == 100.0

    def test_get_recent_average_multiple_values(self):
        """get_recent_average should compute correct average."""
        metrics = TrainingMetrics()
        for i in range(5):
            stats = EpisodeStats(
                episode=i, score=i * 10, steps=50, total_reward=float(i),
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=i, won=False
            )
            metrics.add(stats)
        # Scores: 0, 10, 20, 30, 40 -> avg = 20
        result = metrics.get_recent_average('scores', n=100)
        assert result == 20.0

    def test_get_recent_average_last_n(self):
        """get_recent_average should only use last n values."""
        metrics = TrainingMetrics()
        for i in range(10):
            stats = EpisodeStats(
                episode=i, score=i * 10, steps=50, total_reward=float(i),
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=i, won=False
            )
            metrics.add(stats)
        # Last 3 scores: 70, 80, 90 -> avg = 80
        result = metrics.get_recent_average('scores', n=3)
        assert result == 80.0

    def test_get_best_score_empty(self):
        """get_best_score should return 0 for empty history."""
        metrics = TrainingMetrics()
        assert metrics.get_best_score() == 0

    def test_get_best_score(self):
        """get_best_score should return highest score."""
        metrics = TrainingMetrics()
        for score in [10, 50, 30, 20]:
            stats = EpisodeStats(
                episode=0, score=score, steps=50, total_reward=0.0,
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=0, won=False
            )
            metrics.add(stats)
        assert metrics.get_best_score() == 50

    def test_get_win_rate_empty(self):
        """get_win_rate should return 0 for empty history."""
        metrics = TrainingMetrics()
        assert metrics.get_win_rate() == 0.0

    def test_get_win_rate(self):
        """get_win_rate should compute correct win rate."""
        metrics = TrainingMetrics()
        for won in [True, False, True, True, False]:
            stats = EpisodeStats(
                episode=0, score=100, steps=50, total_reward=0.0,
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=0, won=won
            )
            metrics.add(stats)
        # 3 wins out of 5 = 60%
        assert metrics.get_win_rate() == 0.6

    def test_get_win_rate_last_n(self):
        """get_win_rate should only use last n episodes."""
        metrics = TrainingMetrics()
        # First 5: all losses
        for _ in range(5):
            stats = EpisodeStats(
                episode=0, score=100, steps=50, total_reward=0.0,
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=0, won=False
            )
            metrics.add(stats)
        # Last 5: all wins
        for _ in range(5):
            stats = EpisodeStats(
                episode=0, score=100, steps=50, total_reward=0.0,
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=0, won=True
            )
            metrics.add(stats)
        # Last 5 should be 100% wins
        assert metrics.get_win_rate(n=5) == 1.0

    def test_history_trimming(self):
        """Metrics should trim to history_length."""
        metrics = TrainingMetrics(history_length=5)
        for i in range(10):
            stats = EpisodeStats(
                episode=i, score=i, steps=50, total_reward=0.0,
                epsilon=0.5, avg_loss=0.01, duration=1.0, bricks_broken=0, won=False
            )
            metrics.add(stats)
        assert len(metrics.scores) == 5
        # Should keep the last 5: 5,6,7,8,9
        assert metrics.scores[0] == 5


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_trainer_creates_successfully(self, trainer):
        """Trainer should initialize without errors."""
        assert trainer is not None

    def test_trainer_has_game(self, trainer, game):
        """Trainer should have reference to game."""
        assert trainer.game is game

    def test_trainer_has_agent(self, trainer, agent):
        """Trainer should have reference to agent."""
        assert trainer.agent is agent

    def test_trainer_has_metrics(self, trainer):
        """Trainer should have TrainingMetrics instance."""
        assert isinstance(trainer.metrics, TrainingMetrics)

    def test_trainer_starts_at_episode_zero(self, trainer):
        """Trainer should start at episode 0."""
        assert trainer.current_episode == 0
        assert trainer.total_steps == 0


class TestRunEpisode:
    """Test Trainer.run_episode method."""

    def test_run_episode_returns_stats(self, trainer):
        """run_episode should return EpisodeStats."""
        stats = trainer.run_episode(render=False)
        assert isinstance(stats, EpisodeStats)

    def test_run_episode_stats_have_correct_fields(self, trainer):
        """EpisodeStats should have all required fields."""
        stats = trainer.run_episode(render=False)
        assert hasattr(stats, 'episode')
        assert hasattr(stats, 'score')
        assert hasattr(stats, 'steps')
        assert hasattr(stats, 'total_reward')
        assert hasattr(stats, 'epsilon')
        assert hasattr(stats, 'avg_loss')
        assert hasattr(stats, 'duration')
        assert hasattr(stats, 'bricks_broken')
        assert hasattr(stats, 'won')

    def test_run_episode_increments_steps(self, trainer):
        """run_episode should increment total_steps."""
        initial_steps = trainer.total_steps
        trainer.run_episode(render=False)
        assert trainer.total_steps > initial_steps

    def test_run_episode_steps_positive(self, trainer):
        """Episode should have positive step count."""
        stats = trainer.run_episode(render=False)
        assert stats.steps > 0

    def test_run_episode_duration_positive(self, trainer):
        """Episode duration should be positive."""
        stats = trainer.run_episode(render=False)
        assert stats.duration > 0


class TestTrainSavesCheckpoints:
    """Test that train() saves checkpoints correctly."""

    def test_train_saves_best_model(self, game, agent, config):
        """Training should save best model when score improves."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            config.MODEL_DIR = tmpdir
            config.LOG_EVERY = 100  # Suppress logging
            config.SAVE_EVERY = 1000  # Don't periodic save during test

            trainer = Trainer(game, agent, config)

            # Run just 3 episodes to see if best model is saved
            trainer.train(num_episodes=3)

            # Check that best model was saved
            best_model_path = os.path.join(tmpdir, f'{config.GAME_NAME}_best.pth')
            assert os.path.exists(best_model_path), "Best model should be saved during training"

    def test_train_saves_final_model(self, game, agent, config):
        """Training should save final model at end."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            config.MODEL_DIR = tmpdir
            config.LOG_EVERY = 100
            config.SAVE_EVERY = 1000

            trainer = Trainer(game, agent, config)
            trainer.train(num_episodes=2)

            # Check that final model was saved
            final_model_path = os.path.join(tmpdir, f'{config.GAME_NAME}_final.pth')
            assert os.path.exists(final_model_path), "Final model should be saved at training end"


class TestEvaluateResetsEpsilon:
    """Test that evaluate() properly handles epsilon."""

    def test_evaluate_resets_epsilon(self, trainer):
        """Evaluate should temporarily set epsilon to 0 then restore."""
        # Set epsilon to a known value
        trainer.agent.epsilon = 0.5

        # Run evaluation
        results = trainer.evaluate(num_episodes=2, render=False)

        # Epsilon should be restored after evaluation
        assert trainer.agent.epsilon == 0.5

    def test_evaluate_uses_no_exploration(self, trainer):
        """During evaluation, epsilon should be 0 (no exploration)."""
        trainer.agent.epsilon = 0.5

        # We can't directly check epsilon during evaluation, but we can
        # verify the results are reasonable (evaluation ran without error)
        results = trainer.evaluate(num_episodes=2, render=False)

        assert 'mean_score' in results
        assert 'max_score' in results
        assert 'min_score' in results
        assert 'win_rate' in results
        assert results['mean_score'] >= 0

    def test_evaluate_returns_correct_statistics(self, trainer):
        """Evaluate should return correct statistics structure."""
        results = trainer.evaluate(num_episodes=3, render=False)

        assert isinstance(results['mean_score'], float)
        assert isinstance(results['max_score'], (int, float))
        assert isinstance(results['min_score'], (int, float))
        assert 0.0 <= results['win_rate'] <= 1.0

        # Max should be >= mean >= min
        assert results['max_score'] >= results['mean_score']
        assert results['mean_score'] >= results['min_score']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
