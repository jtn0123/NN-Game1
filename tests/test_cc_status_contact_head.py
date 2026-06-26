"""Focused tests for Crystal Caves contact-head experiment helpers."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.cc_status.contact_head as contact_head  # noqa: E402
from config import Config  # noqa: E402
from src.ai.agent import Agent  # noqa: E402


class _LowConfidencePolicy:
    training = True

    def __init__(self):
        self.train_called = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True
        self.train_called = True

    def contact_action_logits(self, state):
        del state
        return torch.zeros((1, 10), dtype=torch.float32)


class _JumpConfidencePolicy(_LowConfidencePolicy):
    def contact_action_logits(self, state):
        del state
        logits = torch.zeros((1, 10), dtype=torch.float32)
        logits[0, 5] = 3.6
        return logits


class _LeftJumpConfidencePolicy(_LowConfidencePolicy):
    def contact_action_logits(self, state):
        del state
        logits = torch.zeros((1, 10), dtype=torch.float32)
        logits[0, 4] = 3.6
        return logits


class _PolicyFallbackAgent:
    device = torch.device("cpu")

    def __init__(self):
        self.policy_net = _LowConfidencePolicy()

    def select_action(self, state, training=False):
        del state, training
        return 3


def _spatial_contact_agent() -> Agent:
    cfg = Config()
    cfg.FORCE_CPU = True
    cfg.USE_CNN_STATE = True
    cfg.STATE_LAYOUT = {"window": (11, 19), "gmap": (6, 11), "meta": 20}
    cfg.CRYSTAL_CAVES_CONTACT_ACTION_HEAD = True
    cfg.USE_PRIORITIZED_REPLAY = False
    cfg.USE_N_STEP_RETURNS = False
    size = 11 * 19 + 6 * 11 + 20
    return Agent(state_size=size, action_size=10, config=cfg)


def test_contact_head_confidence_gate_falls_back_to_policy(monkeypatch):
    monkeypatch.setattr(
        contact_head,
        "objective_snapshot",
        lambda game: {"target_distance_tiles": 1.0},
    )
    agent = _PolicyFallbackAgent()
    stats = contact_head.new_contact_action_head_stats()

    action, decision = contact_head.contact_action_head_action(
        agent,
        np.zeros(295, dtype=np.float32),
        game=object(),
        action_labels=[str(index) for index in range(10)],
        close_zone_distance_tiles=3.0,
        min_confidence=0.75,
    )
    contact_head.record_contact_action_head_decision(
        stats,
        decision,
        action,
        [str(index) for index in range(10)],
    )
    payload = contact_head.contact_action_head_stats_payload(stats)

    assert action == 3
    assert decision["source"] == "policy"
    assert decision["head_confidence_rejected"] is True
    assert payload["head_actions"] == 0
    assert payload["policy_actions"] == 1
    assert payload["confidence_rejected_actions"] == 1


def test_contact_head_jump_confidence_gate_is_stricter(monkeypatch):
    monkeypatch.setattr(
        contact_head,
        "objective_snapshot",
        lambda game: {"target_distance_tiles": 1.0},
    )
    agent = _PolicyFallbackAgent()
    agent.policy_net = _JumpConfidencePolicy()
    stats = contact_head.new_contact_action_head_stats()
    action_labels = [
        "IDLE",
        "LEFT",
        "RIGHT",
        "JUMP",
        "LEFT_JUMP",
        "RIGHT_JUMP",
        "SHOOT",
        "LEFT_SHOOT",
        "RIGHT_SHOOT",
        "INTERACT",
    ]

    action, decision = contact_head.contact_action_head_action(
        agent,
        np.zeros(295, dtype=np.float32),
        game=object(),
        action_labels=action_labels,
        close_zone_distance_tiles=3.0,
        min_confidence=0.75,
        jump_min_confidence=0.85,
    )
    contact_head.record_contact_action_head_decision(
        stats,
        decision,
        action,
        action_labels,
    )
    payload = contact_head.contact_action_head_stats_payload(stats)

    assert action == 3
    assert decision["source"] == "policy"
    assert decision["head_action"] == 5
    assert decision["head_action_label"] == "RIGHT_JUMP"
    assert decision["required_confidence"] == 0.85
    assert decision["base_min_confidence"] == 0.75
    assert decision["jump_min_confidence"] == 0.85
    assert payload["head_actions"] == 0
    assert payload["policy_actions"] == 1
    assert payload["confidence_rejected_actions"] == 1
    assert payload["rejected_head_action_counts"] == {"RIGHT_JUMP": 1}


def test_contact_head_action_threshold_rejects_only_named_action(monkeypatch):
    monkeypatch.setattr(
        contact_head,
        "objective_snapshot",
        lambda game: {"target_distance_tiles": 1.0},
    )
    agent = _PolicyFallbackAgent()
    agent.policy_net = _LeftJumpConfidencePolicy()
    action_labels = [
        "IDLE",
        "LEFT",
        "RIGHT",
        "JUMP",
        "LEFT_JUMP",
        "RIGHT_JUMP",
        "SHOOT",
        "LEFT_SHOOT",
        "RIGHT_SHOOT",
        "INTERACT",
    ]

    action, decision = contact_head.contact_action_head_action(
        agent,
        np.zeros(295, dtype=np.float32),
        game=object(),
        action_labels=action_labels,
        close_zone_distance_tiles=3.0,
        min_confidence=0.75,
        action_min_confidences={"LEFT_JUMP": 0.85},
    )

    assert action == 3
    assert decision["source"] == "policy"
    assert decision["head_action"] == 4
    assert decision["head_action_label"] == "LEFT_JUMP"
    assert decision["required_confidence"] == 0.85
    assert decision["action_min_confidences"] == {"LEFT_JUMP": 0.85}


def test_contact_head_action_threshold_parser_normalizes_labels():
    thresholds = contact_head.parse_contact_head_action_thresholds("left-jump:0.90, RIGHT_JUMP:0.8")

    assert thresholds == {"LEFT_JUMP": 0.9, "RIGHT_JUMP": 0.8}


def test_contact_head_action_threshold_validation_rejects_unknown_label():
    try:
        contact_head.validate_contact_head_action_thresholds(
            {"LEFTJUMP": 0.9},
            ["LEFT_JUMP"],
        )
    except ValueError as exc:
        assert "LEFTJUMP" in str(exc)
    else:
        raise AssertionError("expected unknown action label to be rejected")


def test_offline_contact_head_training_keeps_route_params_frozen():
    agent = _spatial_contact_agent()
    rng = np.random.default_rng(0)
    states = rng.normal(size=(18, agent.state_size)).astype(np.float32)
    actions = np.array([1, 3, 5] * 6, dtype=np.int64)
    before = {
        name: param.detach().clone()
        for name, param in agent.policy_net.named_parameters()
        if not name.startswith("contact_action_head.")
    }

    result = contact_head.train_contact_action_head_offline(
        agent,
        states,
        actions,
        steps=5,
        batch_size=9,
        learning_rate=0.01,
        balance_classes=True,
        action_labels=[str(index) for index in range(10)],
    )

    assert result["mode"] == "offline_head_only"
    assert result["balance_classes"] is True
    assert result["route_max_abs_delta"] == 0.0
    assert result["head_max_abs_delta"] > 0.0
    assert result["dataset_eval"]["label_counts"] == {"1": 6, "3": 6, "5": 6}
    for name, param in agent.policy_net.named_parameters():
        if not name.startswith("contact_action_head."):
            assert torch.equal(param.detach(), before[name])
