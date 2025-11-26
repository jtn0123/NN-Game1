"""
Tests for the DQN Agent.

These tests verify:
    - Agent initialization
    - Action selection (epsilon-greedy)
    - Experience storage
    - Learning step
    - Save/Load functionality
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
from src.ai.network import DQN
from src.ai.replay_buffer import ReplayBuffer


@pytest.fixture
def config():
    """Create test configuration."""
    cfg = Config()
    cfg.MEMORY_SIZE = 1000
    cfg.MEMORY_MIN = 100
    cfg.BATCH_SIZE = 32
    return cfg


@pytest.fixture
def agent(config):
    """Create agent instance."""
    return Agent(
        state_size=config.STATE_SIZE,
        action_size=config.ACTION_SIZE,
        config=config
    )


class TestAgentInitialization:
    """Test agent initialization."""
    
    def test_agent_creates_successfully(self, config):
        """Agent should initialize without errors."""
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        assert agent is not None
    
    def test_networks_initialized(self, agent):
        """Both policy and target networks should be initialized."""
        assert agent.policy_net is not None
        assert agent.target_net is not None
    
    def test_networks_same_architecture(self, agent):
        """Policy and target networks should have same architecture."""
        policy_params = sum(p.numel() for p in agent.policy_net.parameters())
        target_params = sum(p.numel() for p in agent.target_net.parameters())
        assert policy_params == target_params
    
    def test_epsilon_starts_at_max(self, agent, config):
        """Epsilon should start at EPSILON_START."""
        assert agent.epsilon == config.EPSILON_START
    
    def test_memory_initialized(self, agent):
        """Replay buffer should be initialized."""
        assert agent.memory is not None
        assert len(agent.memory) == 0


class TestActionSelection:
    """Test action selection."""
    
    def test_select_action_returns_valid_action(self, agent, config):
        """Selected action should be in valid range."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action < config.ACTION_SIZE
    
    def test_greedy_action_selection(self, agent, config):
        """With epsilon=0, should always select best action."""
        agent.epsilon = 0
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        # Get multiple actions - should all be the same (deterministic)
        actions = [agent.select_action(state, training=False) for _ in range(10)]
        assert len(set(actions)) == 1
    
    def test_exploration_with_high_epsilon(self, agent, config):
        """With epsilon=1, should be random."""
        agent.epsilon = 1.0
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        # Get multiple actions - should have variety
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        assert len(set(actions)) > 1


class TestExperienceStorage:
    """Test experience storage in replay buffer."""
    
    def test_remember_adds_experience(self, agent, config):
        """Remember should add experience to buffer."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        action = 1
        reward = 1.0
        next_state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        done = False
        
        initial_size = len(agent.memory)
        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == initial_size + 1
    
    def test_memory_capacity(self, agent, config):
        """Buffer should not exceed capacity."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        for _ in range(config.MEMORY_SIZE + 100):
            agent.remember(state, 0, 0, state, False)
        
        assert len(agent.memory) == config.MEMORY_SIZE


class TestLearning:
    """Test learning functionality."""
    
    def test_no_learning_without_enough_samples(self, agent, config):
        """Should not learn without minimum samples."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        # Add fewer than minimum required
        for _ in range(config.MEMORY_MIN - 10):
            agent.remember(state, 0, 1.0, state, False)
        
        loss = agent.learn()
        assert loss is None
    
    def test_learning_with_enough_samples(self, agent, config):
        """Should learn with enough samples."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        # Add enough experiences
        for _ in range(config.MEMORY_MIN + 50):
            agent.remember(state, np.random.randint(0, 3), 1.0, state, False)
        
        loss = agent.learn()
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_epsilon_decay(self, agent, config):
        """Epsilon should decay correctly."""
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        expected = initial_epsilon * config.EPSILON_DECAY
        assert agent.epsilon == max(config.EPSILON_END, expected)


class TestEpsilonDecayStrategies:
    """Test epsilon decay with different strategies.
    
    Tests for bug fix: When EPSILON_DECAY is 1.0 with linear/cosine strategies,
    the calculation of decay_episodes would divide by log(1.0)=0, causing a crash.
    """
    
    def test_linear_decay_with_epsilon_decay_one(self, config):
        """Linear decay should not crash when EPSILON_DECAY is 1.0."""
        config.EXPLORATION_STRATEGY = 'linear'
        config.EPSILON_DECAY = 1.0  # This would cause division by zero
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        initial_epsilon = agent.epsilon
        # Should not raise an exception
        agent.decay_epsilon()
        # With decay=1.0, epsilon should remain unchanged (no decay)
        assert agent.epsilon == initial_epsilon
    
    def test_cosine_decay_with_epsilon_decay_one(self, config):
        """Cosine decay should not crash when EPSILON_DECAY is 1.0."""
        config.EXPLORATION_STRATEGY = 'cosine'
        config.EPSILON_DECAY = 1.0  # This would cause division by zero
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        initial_epsilon = agent.epsilon
        # Should not raise an exception
        agent.decay_epsilon()
        # With decay=1.0, epsilon should remain unchanged (no decay)
        assert agent.epsilon == initial_epsilon
    
    def test_linear_decay_normal_operation(self, config):
        """Linear decay should work correctly with normal EPSILON_DECAY."""
        config.EXPLORATION_STRATEGY = 'linear'
        config.EPSILON_DECAY = 0.995
        config.EPSILON_WARMUP = 0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        # Decay multiple times
        for _ in range(100):
            agent.decay_epsilon()
        
        # Epsilon should have decreased
        assert agent.epsilon < config.EPSILON_START
        assert agent.epsilon >= config.EPSILON_END
    
    def test_cosine_decay_normal_operation(self, config):
        """Cosine decay should work correctly with normal EPSILON_DECAY."""
        config.EXPLORATION_STRATEGY = 'cosine'
        config.EPSILON_DECAY = 0.995
        config.EPSILON_WARMUP = 0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        # Decay multiple times
        for _ in range(100):
            agent.decay_epsilon()
        
        # Epsilon should have decreased
        assert agent.epsilon < config.EPSILON_START
        assert agent.epsilon >= config.EPSILON_END
    
    def test_calculate_decay_episodes_returns_positive(self, config):
        """Helper method should return positive decay episodes with normal config."""
        config.EPSILON_DECAY = 0.995
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        decay_episodes = agent._calculate_decay_episodes()
        assert decay_episodes > 0
    
    def test_calculate_decay_episodes_returns_zero_for_decay_one(self, config):
        """Helper method should return 0 when EPSILON_DECAY is 1.0."""
        config.EPSILON_DECAY = 1.0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        decay_episodes = agent._calculate_decay_episodes()
        assert decay_episodes == 0


class TestQValues:
    """Test Q-value computation."""
    
    def test_get_q_values_shape(self, agent, config):
        """Q-values should have correct shape."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        q_values = agent.get_q_values(state)
        assert q_values.shape == (config.ACTION_SIZE,)
    
    def test_q_values_dtype(self, agent, config):
        """Q-values should be float."""
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        q_values = agent.get_q_values(state)
        assert q_values.dtype == np.float32 or q_values.dtype == np.float64


class TestSaveLoad:
    """Test save/load functionality."""
    
    def test_save_creates_file(self, agent):
        """Save should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.pth")
            agent.save(filepath)
            assert os.path.exists(filepath)
    
    def test_load_restores_state(self, agent, config):
        """Load should restore agent state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.pth")
            
            # Modify agent state
            agent.epsilon = 0.5
            agent.steps = 1000
            agent.save(filepath)
            
            # Create new agent and load
            new_agent = Agent(
                state_size=config.STATE_SIZE,
                action_size=config.ACTION_SIZE,
                config=config
            )
            new_agent.load(filepath)
            
            assert new_agent.epsilon == 0.5
            assert new_agent.steps == 1000


class TestTargetNetwork:
    """Test target network updates."""
    
    def test_update_target_network(self, agent):
        """Target network should sync with policy network."""
        # Modify policy network
        with torch.no_grad():
            for param in agent.policy_net.parameters():
                param.add_(1.0)
        
        # Update target
        agent.update_target_network()
        
        # Check they're equal
        for p_param, t_param in zip(
            agent.policy_net.parameters(),
            agent.target_net.parameters()
        ):
            assert torch.allclose(p_param, t_param)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

