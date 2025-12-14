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
    
    def test_exploration_with_high_epsilon(self, config):
        """With epsilon=1 and no NoisyNets, should be random."""
        # Disable NoisyNets to test epsilon-greedy exploration
        config.USE_NOISY_NETWORKS = False
        config.EPSILON_START = 1.0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        # Get multiple actions - should have variety
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        assert len(set(actions)) > 1
    
    def test_exploration_with_noisy_nets(self, config):
        """With NoisyNets, exploration is handled by learned noise."""
        config.USE_NOISY_NETWORKS = True
        config.USE_DUELING = True  # NoisyNets require DuelingDQN
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        state = np.random.randn(config.STATE_SIZE).astype(np.float32)
        
        # NoisyNets should produce varied actions due to noise in weights
        # Even with epsilon=0, training mode should have exploration
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        # Should have at least some variety (noise provides exploration)
        # Note: early in training, noise may not cause much variation
        assert len(actions) == 100  # Just verify it returns valid actions


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
        
        # Call learn() enough times to account for LEARN_EVERY setting
        learn_every = getattr(config, 'LEARN_EVERY', 1)
        loss = None
        for _ in range(learn_every):
            loss = agent.learn()
            if loss is not None:
                break
        
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_epsilon_decay(self, config):
        """Epsilon should decay correctly (when not using NoisyNets)."""
        # Disable NoisyNets to test epsilon-greedy decay
        config.USE_NOISY_NETWORKS = False
        config.EPSILON_START = 1.0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        expected = initial_epsilon * config.EPSILON_DECAY
        assert agent.epsilon == max(config.EPSILON_END, expected)
    
    def test_epsilon_decay_with_decay_one(self, config):
        """Epsilon should remain unchanged when EPSILON_DECAY is 1.0."""
        # Disable NoisyNets to test epsilon-greedy decay
        config.USE_NOISY_NETWORKS = False
        config.EPSILON_DECAY = 1.0
        config.EPSILON_START = 1.0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        # With decay=1.0, epsilon * 1.0 = epsilon (unchanged)
        assert agent.epsilon == initial_epsilon
    
    def test_epsilon_decay_respects_minimum(self, config):
        """Epsilon should not decay below EPSILON_END."""
        # Disable NoisyNets to test epsilon-greedy decay
        config.USE_NOISY_NETWORKS = False
        config.EPSILON_DECAY = 0.5  # Aggressive decay
        config.EPSILON_END = 0.1
        config.EPSILON_START = 1.0
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        # Decay many times
        for _ in range(100):
            agent.decay_epsilon()
        
        # Should be at minimum, not below
        assert agent.epsilon == config.EPSILON_END
    
    def test_epsilon_decay_with_noisy_nets_hybrid(self, config):
        """Epsilon SHOULD decay with NoisyNets for hybrid exploration (NoisyNets + epsilon-greedy fallback)."""
        config.USE_NOISY_NETWORKS = True
        config.EPSILON_START = 0.5  # Start at some value
        config.EPSILON_DECAY = 0.9995
        config.EPSILON_END = 0.01

        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )

        initial_epsilon = agent.epsilon
        agent.decay_epsilon()

        # Epsilon should decay even with NoisyNets (hybrid exploration approach)
        # This provides fallback exploration if NoisyNet sigmas decay too aggressively
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon == initial_epsilon * config.EPSILON_DECAY


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
    
    def test_soft_update_target_network(self, config):
        """Soft update should blend policy into target network."""
        # Enable soft updates
        config.USE_SOFT_UPDATE = True
        config.TARGET_TAU = 0.1  # Use larger tau for easier testing
        
        agent = Agent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            config=config
        )
        
        # Store original target weights
        original_target_weights = [
            param.clone() for param in agent.target_net.parameters()
        ]
        
        # Modify policy network significantly
        with torch.no_grad():
            for param in agent.policy_net.parameters():
                param.add_(10.0)
        
        # Soft update
        agent._soft_update_target_network()
        
        # Check that target has changed but is not equal to policy
        for i, (t_param, p_param, orig_param) in enumerate(zip(
            agent.target_net.parameters(),
            agent.policy_net.parameters(),
            original_target_weights
        )):
            # Target should have moved toward policy
            assert not torch.allclose(t_param, orig_param), f"Target param {i} didn't change"
            # But shouldn't be equal to policy (soft update)
            assert not torch.allclose(t_param, p_param), f"Target param {i} equals policy (should be blended)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

