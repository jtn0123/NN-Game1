"""Tests for optional DQN agent experiment hooks."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from experiments.cc_status.policy_anchor import (
    FrozenPolicyAnchorProvider,
    install_policy_anchor_provider,
)
from src.ai.action_margin_loss import sample_action_margin_loss
from src.ai.agent import Agent
from src.ai.extension_contracts import AuxiliaryLossContribution, AuxiliaryMetric


@pytest.fixture
def config():
    cfg = Config()
    cfg.MEMORY_SIZE = 1000
    cfg.MEMORY_MIN = 100
    cfg.BATCH_SIZE = 32
    return cfg


class _ExternalAuxProvider:
    def auxiliary_loss_contributions(self, states):
        loss = states.sum() * 0.0 + torch.tensor(0.5, device=states.device)
        return (
            AuxiliaryLossContribution(
                name="external_aux",
                loss=loss,
                weight=0.25,
            ),
        )


def test_action_margin_loss_helper_samples_finite_supervised_loss():
    torch.manual_seed(0)
    policy_net = torch.nn.Linear(3, 5)
    states = torch.randn(6, 3)
    actions = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long)

    result = sample_action_margin_loss(
        policy_net,
        states,
        actions,
        batch_size=4,
        margin=0.6,
        conservative_temperature=0.7,
    )

    assert result is not None
    assert torch.isfinite(result.loss)
    assert result.conservative_loss is not None
    assert torch.isfinite(result.conservative_loss)
    assert 0.0 <= result.accuracy <= 1.0


class TestRouteAuxiliaryLoss:
    """Test Crystal Caves route-direction auxiliary supervision."""

    def _spatial_config(self):
        cfg = Config()
        cfg.USE_CNN_STATE = True
        cfg.STATE_LAYOUT = {"window": (11, 19), "gmap": (6, 11), "meta": 20}
        cfg.CRYSTAL_CAVES_ROUTE_AUX_LOSS = True
        cfg.CRYSTAL_CAVES_ROUTE_AUX_WEIGHT = 0.05
        cfg.USE_PRIORITIZED_REPLAY = False
        cfg.USE_N_STEP_RETURNS = False
        return cfg

    def test_route_auxiliary_targets_come_from_metadata_direction(self):
        cfg = self._spatial_config()
        size = 11 * 19 + 6 * 11 + 20
        agent = Agent(state_size=size, action_size=10, config=cfg)
        states = torch.zeros((4, size), dtype=torch.float32, device=agent.device)
        meta_start = 11 * 19 + 6 * 11
        states[:, meta_start + 17] = 0.25
        states[:, meta_start + 18] = 0.25
        states[0, meta_start + 15] = 0.75
        states[0, meta_start + 16] = 0.50
        states[1, meta_start + 15] = 0.25
        states[1, meta_start + 16] = 0.25
        states[2, meta_start + 15] = 0.50
        states[2, meta_start + 16] = 0.75
        states[3, meta_start + 15] = 0.50
        states[3, meta_start + 16] = 0.50

        targets = agent._route_auxiliary_targets(states)

        assert targets is not None
        labels, mask = targets
        assert mask.tolist() == [True, True, True, True]
        assert labels.tolist() == [5, 0, 7, 4]

    def test_route_auxiliary_loss_is_finite_and_tracks_accuracy(self):
        cfg = self._spatial_config()
        size = 11 * 19 + 6 * 11 + 20
        agent = Agent(state_size=size, action_size=10, config=cfg)
        states = torch.zeros((8, size), dtype=torch.float32, device=agent.device)
        meta_start = 11 * 19 + 6 * 11
        states[:, meta_start + 15] = 0.75
        states[:, meta_start + 16] = 0.50
        states[:, meta_start + 17] = 0.25
        states[:, meta_start + 18] = 0.25

        result = agent._route_auxiliary_loss(states)

        assert result is not None
        loss, accuracy = result
        assert torch.isfinite(loss)
        assert 0.0 <= accuracy <= 1.0


class TestDemoActionSupervision:
    """Test optional demonstration action-margin supervision."""

    def test_demo_action_dataset_validation_and_margin_loss(self, config):
        config.CRYSTAL_CAVES_DEMO_ACTION_LOSS = True
        config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT = 0.05
        config.CRYSTAL_CAVES_DEMO_ACTION_MARGIN = 0.8
        config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE = 8
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False

        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(16, config.STATE_SIZE).astype(np.float32)
        actions = np.random.randint(0, config.ACTION_SIZE, size=16).astype(np.int64)

        summary = agent.set_demo_action_dataset(states, actions)
        result = agent._demo_action_supervised_loss()

        assert summary == {"demo_action_transitions": 16}
        assert result is not None
        loss, accuracy, conservative_loss = result
        assert torch.isfinite(loss)
        assert conservative_loss is None
        assert 0.0 <= accuracy <= 1.0

    def test_demo_action_conservative_loss_is_optional(self, config):
        config.CRYSTAL_CAVES_DEMO_ACTION_LOSS = True
        config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT = 0.0
        config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT = 0.02
        config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE = 0.7
        config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE = 8
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False

        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(16, config.STATE_SIZE).astype(np.float32)
        actions = np.random.randint(0, config.ACTION_SIZE, size=16).astype(np.int64)

        agent.set_demo_action_dataset(states, actions)
        result = agent._demo_action_supervised_loss()

        assert result is not None
        loss, accuracy, conservative_loss = result
        assert torch.isfinite(loss)
        assert conservative_loss is not None
        assert torch.isfinite(conservative_loss)
        assert 0.0 <= accuracy <= 1.0

    def test_auxiliary_loss_contributions_have_stable_names_and_metrics(self, config):
        config.CRYSTAL_CAVES_DEMO_ACTION_LOSS = True
        config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT = 0.03
        config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT = 0.02
        config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE = 1.0
        config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS = True
        config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT = 0.04
        config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE = 8
        config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE = 8
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False

        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(16, config.STATE_SIZE).astype(np.float32)
        actions = np.random.randint(0, config.ACTION_SIZE, size=16).astype(np.int64)
        agent.set_demo_action_dataset(states, actions)
        agent.set_close_zone_demo_action_dataset(states, actions)

        contributions = agent._auxiliary_loss_contributions(
            torch.as_tensor(states[:8], dtype=torch.float32, device=agent.device)
        )

        assert [contribution.name for contribution in contributions] == [
            "demo_action",
            "demo_conservative",
            "close_zone_demo_action",
        ]
        assert [contribution.weight for contribution in contributions] == [0.03, 0.02, 0.04]
        metric_histories = {
            metric.history for contribution in contributions for metric in contribution.metrics
        }
        assert metric_histories == {
            "demo_action_losses",
            "demo_action_accuracies",
            "demo_conservative_losses",
            "close_zone_demo_action_losses",
            "close_zone_demo_action_accuracies",
        }
        assert all(torch.isfinite(contribution.weighted_loss()) for contribution in contributions)

    def test_auxiliary_metric_append_rejects_unknown_history(self, config):
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False
        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        contribution = AuxiliaryLossContribution(
            name="bad_metric",
            loss=torch.tensor(1.0),
            weight=1.0,
            metrics=(AuxiliaryMetric("missing_metric_history", 1.0),),
        )

        with agent._losses_lock:
            with pytest.raises(AttributeError, match="missing_metric_history"):
                agent._append_auxiliary_metrics_locked((contribution,))

    def test_close_zone_demo_action_loss_is_separate(self, config):
        config.CRYSTAL_CAVES_DEMO_ACTION_LOSS = False
        config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS = True
        config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT = 0.03
        config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE = 8
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False

        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(16, config.STATE_SIZE).astype(np.float32)
        actions = np.random.randint(0, config.ACTION_SIZE, size=16).astype(np.int64)

        summary = agent.set_close_zone_demo_action_dataset(states, actions)
        result = agent._close_zone_demo_action_supervised_loss()

        assert summary == {"close_zone_demo_action_transitions": 16}
        assert agent._demo_action_supervised_loss() is None
        assert result is not None
        loss, accuracy = result
        assert torch.isfinite(loss)
        assert 0.0 <= accuracy <= 1.0

    def test_correction_action_dataset_validation_and_margin_loss(self, config):
        config.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS = True
        config.CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT = 0.02
        config.CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN = 0.6
        config.CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE = 8
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False

        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(16, config.STATE_SIZE).astype(np.float32)
        actions = np.random.randint(0, config.ACTION_SIZE, size=16).astype(np.int64)

        summary = agent.set_correction_action_dataset(states, actions)
        result = agent._correction_action_supervised_loss()

        assert summary == {"correction_action_transitions": 16}
        assert result is not None
        loss, accuracy = result
        assert torch.isfinite(loss)
        assert 0.0 <= accuracy <= 1.0

    def test_correction_action_contributes_named_auxiliary_loss(self, config):
        config.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS = True
        config.CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT = 0.02
        config.CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN = 0.6
        config.CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE = 8
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False

        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(16, config.STATE_SIZE).astype(np.float32)
        actions = np.random.randint(0, config.ACTION_SIZE, size=16).astype(np.int64)
        agent.set_correction_action_dataset(states, actions)

        contributions = agent._auxiliary_loss_contributions(
            torch.as_tensor(states[:8], dtype=torch.float32, device=agent.device)
        )

        assert [contribution.name for contribution in contributions] == ["correction_action"]
        assert [contribution.weight for contribution in contributions] == [0.02]
        metric_histories = {
            metric.history for contribution in contributions for metric in contribution.metrics
        }
        assert metric_histories == {
            "correction_action_losses",
            "correction_action_accuracies",
        }
        assert all(torch.isfinite(contribution.weighted_loss()) for contribution in contributions)

    def test_contact_action_head_contributes_named_auxiliary_loss(self):
        cfg = Config()
        cfg.USE_CNN_STATE = True
        cfg.STATE_LAYOUT = {"window": (11, 19), "gmap": (6, 11), "meta": 20}
        cfg.CRYSTAL_CAVES_CONTACT_ACTION_HEAD = True
        cfg.CRYSTAL_CAVES_CONTACT_ACTION_WEIGHT = 0.02
        cfg.CRYSTAL_CAVES_CONTACT_ACTION_BATCH_SIZE = 8
        cfg.USE_PRIORITIZED_REPLAY = False
        cfg.USE_N_STEP_RETURNS = False
        size = 11 * 19 + 6 * 11 + 20
        agent = Agent(state_size=size, action_size=10, config=cfg)
        states = np.random.randn(16, size).astype(np.float32)
        actions = np.random.randint(0, 10, size=16).astype(np.int64)

        summary = agent.set_contact_action_dataset(states, actions)
        contributions = agent._auxiliary_loss_contributions(
            torch.as_tensor(states[:8], dtype=torch.float32, device=agent.device)
        )

        assert summary == {"contact_action_transitions": 16}
        assert [contribution.name for contribution in contributions] == ["contact_action"]
        assert [contribution.weight for contribution in contributions] == [0.02]
        metric_histories = {
            metric.history for contribution in contributions for metric in contribution.metrics
        }
        assert metric_histories == {
            "contact_action_losses",
            "contact_action_accuracies",
        }
        assert all(torch.isfinite(contribution.weighted_loss()) for contribution in contributions)
        with agent._losses_lock:
            agent._append_auxiliary_metrics_locked(contributions)
        assert agent.get_contact_action_transition_count() == 16
        assert agent.get_contact_action_metric_count(100) == 1
        assert agent.get_average_contact_action_loss(100) >= 0.0
        assert 0.0 <= agent.get_average_contact_action_accuracy(100) <= 1.0

    def test_external_auxiliary_provider_can_be_registered(self, config):
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False
        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = torch.zeros((4, config.STATE_SIZE), dtype=torch.float32, device=agent.device)

        agent.register_auxiliary_loss_provider(_ExternalAuxProvider())
        contributions = agent._auxiliary_loss_contributions(states)

        assert [contribution.name for contribution in contributions] == ["external_aux"]
        assert [contribution.weight for contribution in contributions] == [0.25]
        assert all(torch.isfinite(contribution.weighted_loss()) for contribution in contributions)

    def test_policy_anchor_provider_contributes_metrics(self, config):
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False
        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = torch.randn((4, config.STATE_SIZE), dtype=torch.float32, device=agent.device)
        teacher_net = agent.policy_net.__class__(
            config.STATE_SIZE,
            config.ACTION_SIZE,
            config,
        ).to(agent.device)
        teacher_net.load_state_dict(agent.policy_net.state_dict())
        teacher_net.eval()
        for parameter in teacher_net.parameters():
            parameter.requires_grad_(False)

        agent.register_auxiliary_loss_provider(
            FrozenPolicyAnchorProvider(
                policy_net=agent.policy_net,
                teacher_net=teacher_net,
                weight=0.02,
                temperature=1.0,
            )
        )

        contributions = agent._auxiliary_loss_contributions(states)

        assert [contribution.name for contribution in contributions] == ["policy_anchor"]
        assert [contribution.weight for contribution in contributions] == [0.02]
        metric_histories = {
            metric.history for contribution in contributions for metric in contribution.metrics
        }
        assert metric_histories == {"policy_anchor_losses", "policy_anchor_accuracies"}
        assert all(torch.isfinite(contribution.weighted_loss()) for contribution in contributions)

        with agent._losses_lock:
            agent._append_auxiliary_metrics_locked(contributions)
        assert agent.get_policy_anchor_metric_count(100) == 1
        assert agent.get_average_policy_anchor_loss(100) >= 0.0
        assert 0.0 <= agent.get_average_policy_anchor_accuracy(100) <= 1.0

    def test_policy_anchor_provider_can_mask_close_zone_states(self):
        policy_net = torch.nn.Linear(4, 3)
        teacher_net = torch.nn.Linear(4, 3)
        teacher_net.load_state_dict(policy_net.state_dict())
        provider = FrozenPolicyAnchorProvider(
            policy_net=policy_net,
            teacher_net=teacher_net,
            weight=0.02,
            temperature=1.0,
            min_target_distance_norm=0.1,
            target_distance_index=2,
        )
        states = torch.tensor(
            [
                [0.0, 0.0, 0.05, 0.0],
                [0.0, 0.0, 0.20, 0.0],
            ],
            dtype=torch.float32,
        )

        contributions = provider.auxiliary_loss_contributions(states)

        assert [contribution.name for contribution in contributions] == ["policy_anchor"]
        assert torch.isfinite(contributions[0].weighted_loss())

        close_only = states[:1]
        assert provider.auxiliary_loss_contributions(close_only) == ()

    def test_install_policy_anchor_summary_includes_distance_mask(self, config):
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False
        config.STATE_LAYOUT = {"window": (1, 2), "gmap": (1, 1), "meta": 20}
        agent = Agent(state_size=23, action_size=config.ACTION_SIZE, config=config)

        summary = install_policy_anchor_provider(
            agent,
            weight=0.02,
            temperature=1.0,
            min_target_distance_norm=0.07,
        )

        assert summary["enabled"] is True
        assert summary["min_target_distance_norm"] == 0.07
        assert summary["target_distance_index"] == 20

    def test_register_auxiliary_provider_requires_provider_method(self, config):
        config.USE_PRIORITIZED_REPLAY = False
        config.USE_N_STEP_RETURNS = False
        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)

        with pytest.raises(TypeError, match="auxiliary_loss_contributions"):
            agent.register_auxiliary_loss_provider(object())

    def test_demo_action_dataset_rejects_wrong_state_size(self, config):
        agent = Agent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, config=config)
        states = np.random.randn(2, config.STATE_SIZE + 1).astype(np.float32)
        actions = np.zeros(2, dtype=np.int64)

        with pytest.raises(ValueError, match="demo state size mismatch"):
            agent.set_demo_action_dataset(states, actions)
