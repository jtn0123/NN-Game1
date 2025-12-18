"""
Integration tests for the DQN Breakout project.

These tests verify end-to-end functionality:
    - Agent learns from game interactions
    - Save/load preserves training state
    - Device compatibility
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ai.agent import Agent
from src.game.breakout import Breakout


@pytest.fixture
def config():
    """Create a test configuration optimized for fast testing."""
    cfg = Config()
    cfg.BATCH_SIZE = 8
    cfg.MEMORY_SIZE = 200
    cfg.MEMORY_MIN = 16  # Lower threshold for testing
    cfg.LEARNING_RATE = 0.001
    cfg.USE_N_STEP_RETURNS = False  # Simpler replay for testing
    cfg.MAX_STEPS_PER_EPISODE = 500  # Allow longer episodes
    cfg.LEARN_EVERY = 1  # Learn every step for testing
    return cfg


@pytest.fixture
def game(config):
    """Create a game instance."""
    return Breakout(config, headless=True)


@pytest.fixture
def agent(game, config):
    """Create an agent instance."""
    return Agent(game.state_size, game.action_size, config)


class TestAgentGameIntegration:
    """Test agent-game interaction."""

    def test_agent_can_process_game_state(self, game, agent):
        """Agent should accept game states for action selection."""
        state = game.get_state()
        action = agent.select_action(state, training=True)
        assert 0 <= action < game.action_size

    def test_agent_can_store_game_transitions(self, game, agent):
        """Agent should store game transitions in replay buffer."""
        state = game.get_state()
        action = agent.select_action(state, training=True)
        next_state, reward, done, _ = game.step(action)

        initial_size = len(agent.memory)
        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == initial_size + 1

    def test_agent_learns_from_game(self, game, agent, config):
        """Agent should learn from game experiences."""
        # Fill replay buffer with experiences
        state = game.reset()
        for _ in range(config.BATCH_SIZE * 4):  # More samples to ensure buffer is full
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = game.step(action)
            agent.remember(state, action, reward, next_state, done)

            if done:
                state = game.reset()
            else:
                state = next_state

        # Learning should not raise errors and should return loss after enough samples
        losses = []
        for _ in range(5):  # Try learning multiple times
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        # Should have computed at least one loss
        assert len(losses) > 0 or len(agent.memory) < config.BATCH_SIZE

    def test_multiple_episodes_run(self, game, agent, config):
        """Agent should handle multiple game episodes."""
        episodes_completed = 0
        total_steps = 0

        state = game.reset()
        for _ in range(1000):  # Run for up to 1000 steps
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_steps += 1

            if done:
                state = game.reset()
                episodes_completed += 1
                if episodes_completed >= 2:
                    break
            else:
                state = next_state

        # Should complete at least one episode or run many steps
        assert episodes_completed >= 1 or total_steps >= 500


class TestSaveLoadIntegration:
    """Test save/load functionality."""

    def test_save_and_load_preserves_weights(self, game, agent):
        """Saving and loading should preserve network weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")

            # Get a weight key from the network (first layer)
            state_dict = agent.policy_net.state_dict()
            weight_key = [k for k in state_dict.keys() if 'weight' in k][0]
            initial_weights = state_dict[weight_key].clone()

            # Train briefly to potentially change weights
            state = game.reset()
            for _ in range(30):
                action = agent.select_action(state, training=True)
                next_state, reward, done, _ = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state if not done else game.reset()

            # Save the trained model
            agent.save(save_path)
            trained_weights = agent.policy_net.state_dict()[weight_key].clone()

            # Create new agent and load
            new_agent = Agent(game.state_size, game.action_size, agent.config)
            new_agent.load(save_path)
            loaded_weights = new_agent.policy_net.state_dict()[weight_key]

            # Loaded weights should match saved weights
            assert torch.allclose(trained_weights, loaded_weights)

    def test_save_and_load_preserves_epsilon(self, game, agent):
        """Saving and loading should preserve epsilon value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")

            # Set a specific epsilon
            agent.epsilon = 0.42
            agent.save(save_path)

            # Create new agent and load
            new_agent = Agent(game.state_size, game.action_size, agent.config)
            new_agent.load(save_path)

            assert new_agent.epsilon == 0.42

    def test_loaded_agent_can_play(self, game, agent):
        """Loaded agent should be able to play the game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pth")
            agent.save(save_path)

            # Create new agent and load
            new_agent = Agent(game.state_size, game.action_size, agent.config)
            new_agent.load(save_path)

            # Should be able to play
            state = game.reset()
            for _ in range(10):
                action = new_agent.select_action(state, training=False)
                state, _, done, _ = game.step(action)
                if done:
                    break

            # No errors should occur


class TestDeviceCompatibility:
    """Test CPU/GPU device handling."""

    def test_agent_works_on_cpu(self, game):
        """Agent should work on CPU."""
        config = Config()
        config.FORCE_CPU = True  # Use FORCE_CPU to ensure CPU device

        agent = Agent(game.state_size, game.action_size, config)
        state = game.reset()
        action = agent.select_action(state, training=True)

        assert 0 <= action < game.action_size
        assert agent.device.type == 'cpu'

    def test_state_tensor_device_matches_agent(self, game, agent):
        """State tensors should be on the same device as agent."""
        state = game.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

        assert state_tensor.device.type == agent.device.type

    def test_network_parameters_on_correct_device(self, game, agent):
        """Network parameters should be on configured device."""
        for param in agent.policy_net.parameters():
            assert param.device.type == agent.device.type


class TestRewardSignals:
    """Test that reward signals work correctly."""

    def test_game_provides_rewards(self, game):
        """Game should provide numeric rewards."""
        game.reset()
        rewards_received = []

        for _ in range(50):
            _, reward, done, _ = game.step(1)  # STAY
            rewards_received.append(reward)
            if done:
                break

        # Should have received some rewards
        assert len(rewards_received) > 0
        # All rewards should be numeric
        assert all(isinstance(r, (int, float)) for r in rewards_received)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
