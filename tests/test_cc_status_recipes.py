"""Tests for named Crystal Caves status-session recipes."""

import pytest

from experiments.cc_status.recipes import (
    expand_recipe_argv,
    format_recipe_list,
    get_recipe,
    recommended_recipes,
)


def test_b3s_recipe_is_the_promoted_comparison_baseline():
    recipe = get_recipe("b3s_conservative_demo_q")

    assert recipe.baseline is True
    assert recipe.recommended is False
    assert recipe.mode == "tutorial-demo-conservative"
    assert recipe.option_value("episodes") == 300
    assert recipe.option_value("selected_eval_games") == 30
    assert recipe.option_value("cave_pool_size") == 512
    assert recipe.option_value("route_demo_variants") == ("direct", "recovery")
    assert recipe.option_value("demo_action_weight") == 0.03
    assert recipe.option_value("demo_conservative_weight") == 0.02
    assert "10/30" in recipe.promotion_rule
    assert "60-game" in recipe.promotion_rule


def test_b3s_recipe_builds_expected_status_session_command():
    recipe = get_recipe("b3s_conservative_demo_q")
    args = recipe.cli_args()

    assert args[0] == "tutorial-demo-conservative"
    assert "--route-demo-variants" in args
    assert args[args.index("--route-demo-variants") + 1] == "direct,recovery"
    assert "--save-selected-checkpoint" in args
    assert "--label" in args
    assert (
        args[args.index("--label") + 1]
        == "tutorial_demo_conservative_recovery_pool512_select30_300"
    )

    command = recipe.python_command("/python")
    assert command[:3] == ("/python", "-u", "experiments/cc_status_session.py")
    assert command[3:] == args


def test_smoke_recipe_is_not_promotable():
    recipe = get_recipe("b3s_conservative_smoke")

    assert recipe.baseline is False
    assert recipe.recommended is False
    assert recipe.option_value("episodes") == 1
    assert recipe.option_value("selected_eval_games") == 1
    assert "Smoke only" in recipe.promotion_rule


def test_correction_collect_recipe_requires_checkpoint_override():
    recipe = get_recipe("b3s_correction_collect")

    assert recipe.mode == "collect-corrections"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("correction_games") == 30
    assert recipe.option_value("correction_max_examples") == 1024

    with pytest.raises(ValueError, match="--checkpoint"):
        expand_recipe_argv(("runner", "run-recipe", "b3s_correction_collect"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b3s_correction_collect",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "collect-corrections"
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_correction_finetune_recipe_requires_checkpoint_and_dataset_overrides():
    recipe = get_recipe("b3s_correction_finetune")

    assert recipe.mode == "correction-finetune"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.option_value("episodes") == 300
    assert recipe.option_value("correction_action_weight") == 0.02
    assert recipe.option_value("correction_action_margin") == 0.6

    with pytest.raises(ValueError, match="--checkpoint, --correction-dataset"):
        expand_recipe_argv(("runner", "run-recipe", "b3s_correction_finetune"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b3s_correction_finetune",
            "--checkpoint=selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "correction-finetune"
    assert "--save-selected-checkpoint" in expanded
    assert expanded[-3:] == (
        "--checkpoint=selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_anchored_contact_correction_recipe_requires_checkpoint_and_dataset_overrides():
    recipe = get_recipe("b5_anchored_contact_correction")

    assert recipe.mode == "correction-finetune"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.option_value("correction_action_weight") == 0.005
    assert recipe.option_value("policy_anchor_weight") == 0.02
    assert recipe.option_value("policy_anchor_temperature") == 1.0
    assert "anchoring" in recipe.description

    with pytest.raises(ValueError, match="--checkpoint, --correction-dataset"):
        expand_recipe_argv(("runner", "run-recipe", "b5_anchored_contact_correction"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b5_anchored_contact_correction",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "correction-finetune"
    assert "--policy-anchor-weight" in expanded
    assert "--policy-anchor-temperature" in expanded
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_contact_interleaved_recipe_requires_checkpoint_override():
    recipe = get_recipe("b6_contact_interleaved")

    assert recipe.mode == "contact-interleaved"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("episodes") == 300
    assert recipe.option_value("vec_envs") == 8
    assert recipe.option_value("interleave_contact_ratio") == 0.25
    assert recipe.option_value("selected_eval_games") == 30
    assert "normal RL rewards only" in recipe.description

    with pytest.raises(ValueError, match="--checkpoint"):
        expand_recipe_argv(("runner", "run-recipe", "b6_contact_interleaved"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b6_contact_interleaved",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "contact-interleaved"
    assert "--interleave-contact-ratio" in expanded
    assert "--save-selected-checkpoint" in expanded
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_generated_contact_pool_recipe_is_archived_after_regression():
    recipe = get_recipe("b7_contact_pool_interleaved")

    assert recipe.mode == "contact-interleaved"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("interleave_contact_ratio") == 0.125
    assert recipe.option_value("contact_pool_size") == 128
    assert recipe.option_value("contact_eval_pool_size") == 32
    assert "route/contact scorecard" in recipe.promotion_rule

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b7_contact_pool_interleaved",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "contact-interleaved"
    assert "--contact-pool-size" in expanded
    assert "--contact-eval-pool-size" in expanded
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_history_state_recipe_is_archived_after_comparison():
    recipe = get_recipe("b8_history_state_conservative")

    assert recipe.mode == "tutorial-demo-conservative"
    assert recipe.recommended is False
    assert recipe.required_overrides == ()
    assert recipe.option_value("history_state") is True
    assert recipe.option_value("history_steps") == 4
    assert recipe.option_value("selected_eval_games") == 30
    assert "fresh architecture probe" in recipe.promotion_rule

    expanded = expand_recipe_argv(("runner", "run-recipe", "b8_history_state_smoke"))

    assert expanded[1] == "tutorial-demo-conservative"
    assert "--history-state" in expanded
    assert expanded[expanded.index("--history-steps") + 1] == "4"


def test_c51_distributional_recipe_is_archived_after_comparison():
    recipe = get_recipe("b9_c51_distributional")

    assert recipe.mode == "tutorial-demo-conservative"
    assert recipe.recommended is False
    assert recipe.required_overrides == ()
    assert recipe.option_value("distributional_dqn") is True
    assert recipe.option_value("c51_atoms") == 51
    assert recipe.option_value("c51_v_min") == -20.0
    assert recipe.option_value("c51_v_max") == 120.0
    assert "route/contact scorecard" in recipe.promotion_rule

    expanded = expand_recipe_argv(("runner", "run-recipe", "b9_c51_distributional_smoke"))

    assert expanded[1] == "tutorial-demo-conservative"
    assert "--distributional-dqn" in expanded
    assert expanded[expanded.index("--c51-atoms") + 1] == "51"


def test_generated_contact_pool_smoke_recipe_requires_checkpoint_override():
    recipe = get_recipe("b7_contact_pool_smoke")

    assert recipe.mode == "contact-interleaved"
    assert recipe.option_value("episodes") == 20
    assert recipe.option_value("contact_pool_size") == 16
    assert recipe.option_value("contact_eval_pool_size") == 4
    assert recipe.required_overrides == ("checkpoint",)

    with pytest.raises(ValueError, match="--checkpoint"):
        expand_recipe_argv(("runner", "run-recipe", "b7_contact_pool_smoke"))


def test_final_contact_option_recipe_requires_checkpoint_override():
    recipe = get_recipe("b3s_final_contact_option_eval")

    assert recipe.mode == "eval-final-contact-option"
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("eval_games") == 30
    assert recipe.option_value("final_contact_distance") == 3.0
    assert recipe.option_value("final_contact_commit_steps") == 8
    assert "depth guardrail" in recipe.promotion_rule

    with pytest.raises(ValueError, match="--checkpoint"):
        expand_recipe_argv(("runner", "run-recipe", "b3s_final_contact_option_eval"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b3s_final_contact_option_eval",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "eval-final-contact-option"
    assert "--final-contact-commit-steps" in expanded
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_final_contact_advantage_gate_recipe_is_archived_after_validation():
    recipe = get_recipe("b10_final_contact_advantage_gate_eval")

    assert recipe.mode == "eval-final-contact-option"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("final_contact_distance") == 3.0
    assert recipe.option_value("final_contact_commit_steps") == 8
    assert recipe.option_value("final_contact_cancel_outside") is True
    assert recipe.option_value("final_contact_policy_advantage_gate") is True
    assert recipe.option_value("final_contact_min_option_advantage") == 250.0
    assert "depth" in recipe.promotion_rule

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b10_final_contact_advantage_gate_smoke",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "eval-final-contact-option"
    assert "--final-contact-policy-advantage-gate" in expanded
    assert expanded[expanded.index("--final-contact-min-option-advantage") + 1] == "250.0"
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_advantage_gate_correction_collect_recipe_requires_checkpoint_override():
    recipe = get_recipe("b11_advantage_gate_correction_collect")

    assert recipe.mode == "collect-corrections"
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("final_contact_policy_advantage_gate") is True
    assert recipe.option_value("final_contact_min_option_advantage") == 250.0
    assert recipe.option_value("correction_max_examples") == 256
    assert "Dataset only" in recipe.promotion_rule

    with pytest.raises(ValueError, match="--checkpoint"):
        expand_recipe_argv(("runner", "run-recipe", "b11_advantage_gate_correction_collect"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b11_advantage_gate_correction_collect",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "collect-corrections"
    assert "--final-contact-policy-advantage-gate" in expanded
    assert expanded[expanded.index("--final-contact-min-option-advantage") + 1] == "250.0"
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_advantage_gate_correction_finetune_recipe_requires_dataset_override():
    recipe = get_recipe("b11_advantage_gate_correction_finetune")

    assert recipe.mode == "correction-finetune"
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.option_value("correction_action_weight") == 0.001
    assert recipe.option_value("policy_anchor_weight") == 0.05
    assert "route-preservation" in recipe.promotion_rule

    with pytest.raises(ValueError, match="--checkpoint, --correction-dataset"):
        expand_recipe_argv(("runner", "run-recipe", "b11_advantage_gate_correction_finetune"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b11_advantage_gate_correction_finetune",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "correction-finetune"
    assert expanded[expanded.index("--correction-action-weight") + 1] == "0.001"
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_route_masked_correction_finetune_recipe_requires_dataset_override():
    recipe = get_recipe("b13_route_masked_correction_finetune")

    assert recipe.mode == "correction-finetune"
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.option_value("correction_action_weight") == 0.001
    assert recipe.option_value("policy_anchor_weight") == 0.10
    assert recipe.option_value("policy_anchor_min_distance_tiles") == 3.0
    assert "Route-preserving" in recipe.promotion_rule

    with pytest.raises(ValueError, match="--checkpoint, --correction-dataset"):
        expand_recipe_argv(("runner", "run-recipe", "b13_route_masked_correction_finetune"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b13_route_masked_correction_finetune",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "correction-finetune"
    assert expanded[expanded.index("--policy-anchor-min-distance-tiles") + 1] == "3.0"
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_contact_head_finetune_recipe_requires_dataset_override():
    recipe = get_recipe("b14_contact_head_finetune")

    assert recipe.mode == "contact-head-finetune"
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.option_value("contact_action_weight") == 0.02
    assert recipe.option_value("contact_action_batch_size") == 32
    assert recipe.option_value("contact_action_distance") == 3.0
    assert "close-zone action head" in recipe.description

    with pytest.raises(ValueError, match="--checkpoint, --correction-dataset"):
        expand_recipe_argv(("runner", "run-recipe", "b14_contact_head_finetune"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b14_contact_head_finetune",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "contact-head-finetune"
    assert expanded[expanded.index("--contact-action-weight") + 1] == "0.02"
    assert expanded[expanded.index("--contact-action-distance") + 1] == "3.0"
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_contact_head_offline_recipe_requires_dataset_override():
    recipe = get_recipe("b15_contact_head_offline")

    assert recipe.mode == "contact-head-offline"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.label == "b15_contact_head_offline_balanced_conf075_500_eval30"
    assert recipe.option_value("eval_games") == 30
    assert recipe.option_value("contact_action_batch_size") == 32
    assert recipe.option_value("contact_head_offline_steps") == 500
    assert recipe.option_value("contact_head_learning_rate") == 0.001
    assert recipe.option_value("contact_head_confidence") == 0.75
    assert recipe.option_value("contact_head_balance_classes") is True
    assert "confidence" in recipe.description

    with pytest.raises(ValueError, match="--checkpoint, --correction-dataset"):
        expand_recipe_argv(("runner", "run-recipe", "b15_contact_head_offline"))

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b15_contact_head_offline",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "contact-head-offline"
    assert expanded[expanded.index("--contact-head-offline-steps") + 1] == "500"
    assert expanded[expanded.index("--contact-head-confidence") + 1] == "0.75"
    assert "--contact-head-balance-classes" in expanded
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_contact_head_jump_gated_recipe_targets_seed_one_failure_case():
    recipe = get_recipe("b16_contact_head_jump_gated")

    assert recipe.mode == "contact-head-offline"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.label == "b16_contact_head_jump_conf085_seed1_eval30"
    assert recipe.option_value("seed") == 1
    assert recipe.option_value("eval_games") == 30
    assert recipe.option_value("contact_head_confidence") == 0.75
    assert recipe.option_value("contact_head_jump_confidence") == 0.85
    assert recipe.option_value("contact_head_balance_classes") is True
    assert "jump" in recipe.description

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b16_contact_head_jump_gated",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "correction_examples.npz",
        )
    )

    assert expanded[1] == "contact-head-offline"
    assert expanded[expanded.index("--seed") + 1] == "1"
    assert expanded[expanded.index("--contact-head-confidence") + 1] == "0.75"
    assert expanded[expanded.index("--contact-head-jump-confidence") + 1] == "0.85"
    assert "--contact-head-balance-classes" in expanded
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "correction_examples.npz",
    )


def test_hard_seed_advantage_gate_collect_recipe_builds_dataset_command():
    recipe = get_recipe("b17_advantage_gate_correction_collect_seed1")

    assert recipe.mode == "collect-corrections"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.label == "b17_advantage_gate_correction_collect_seed1"
    assert recipe.option_value("seed") == 1
    assert recipe.option_value("correction_games") == 60
    assert recipe.option_value("correction_max_examples") == 512
    assert recipe.option_value("correction_max_examples_per_game") == 12
    assert recipe.option_value("final_contact_policy_advantage_gate") is True
    assert "hard-seed" in recipe.description

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b17_advantage_gate_correction_collect_seed1",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "collect-corrections"
    assert expanded[expanded.index("--seed") + 1] == "1"
    assert expanded[expanded.index("--correction-max-examples") + 1] == "512"
    assert "--final-contact-policy-advantage-gate" in expanded
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_contact_head_policy_collect_recipe_builds_b25_dataset_command():
    recipe = get_recipe("b25_contact_head_policy_collect")

    assert recipe.mode == "collect-contact-head-corrections"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint", "correction_dataset")
    assert recipe.label == "b25_b24_policy_visited_contact_collect_seed1"
    assert recipe.option_value("seed") == 1
    assert recipe.option_value("correction_games") == 60
    assert recipe.option_value("contact_head_confidence") == 0.75
    assert recipe.option_value("contact_head_action_thresholds") == "LEFT_JUMP:0.90"
    assert recipe.option_value("final_contact_policy_advantage_gate") is True
    assert "visited" in recipe.description

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b25_contact_head_policy_collect",
            "--checkpoint",
            "selected.pth",
            "--correction-dataset",
            "stable_labels.npz",
        )
    )

    assert expanded[1] == "collect-contact-head-corrections"
    assert expanded[expanded.index("--contact-head-action-thresholds") + 1] == ("LEFT_JUMP:0.90")
    assert "--final-contact-policy-advantage-gate" in expanded
    assert expanded[-4:] == (
        "--checkpoint",
        "selected.pth",
        "--correction-dataset",
        "stable_labels.npz",
    )


def test_combined_contact_head_calibration_recipe_uses_b11_and_b17_datasets():
    recipe = get_recipe("b18_contact_head_combined_calibration")

    assert recipe.mode == "contact-head-calibrate"
    assert recipe.recommended is False
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.label == "b18_contact_head_combined_calibration_b11_b17"
    assert recipe.option_value("contact_head_calibration_frac") == 0.25
    assert recipe.option_value("contact_head_min_calibration_accuracy") == 0.70
    assert recipe.option_value("contact_head_min_class_examples") == 10
    assert recipe.option_value("heartbeat_seconds") == 0
    datasets = recipe.option_value("correction_datasets")
    assert isinstance(datasets, tuple)
    assert len(datasets) == 2
    assert "b11_advantage_gate_correction_collect" in str(datasets[0])
    assert "b17_advantage_gate_correction_collect_seed1" in str(datasets[1])

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b18_contact_head_combined_calibration",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "contact-head-calibrate"
    assert "--correction-datasets" in expanded
    assert (
        "b11_advantage_gate_correction_collect"
        in expanded[expanded.index("--correction-datasets") + 1]
    )
    assert (
        "b17_advantage_gate_correction_collect_seed1"
        in expanded[expanded.index("--correction-datasets") + 1]
    )
    assert expanded[expanded.index("--contact-head-calibration-frac") + 1] == "0.25"
    assert expanded[expanded.index("--heartbeat-seconds") + 1] == "0"
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_contact_only_correction_collect_recipe_disables_stale_and_loop_triggers():
    recipe = get_recipe("b4_contact_only_correction_collect")

    assert recipe.mode == "collect-corrections"
    assert recipe.required_overrides == ("checkpoint",)
    assert recipe.option_value("correction_stale_steps") == 999999
    assert recipe.option_value("correction_loop_tile_visits") == 999999
    assert recipe.option_value("correction_max_examples") == 512
    assert "close-zone" in recipe.description

    expanded = expand_recipe_argv(
        (
            "runner",
            "run-recipe",
            "b4_contact_only_correction_collect",
            "--checkpoint",
            "selected.pth",
        )
    )

    assert expanded[1] == "collect-corrections"
    assert "--correction-stale-steps" in expanded
    assert "999999" in expanded
    assert expanded[-2:] == ("--checkpoint", "selected.pth")


def test_recommended_recipes_are_explicit():
    recommended = recommended_recipes()

    assert [recipe.key for recipe in recommended] == []


def test_recipe_list_is_user_facing_and_marks_baseline():
    text = format_recipe_list()

    assert "b3s_conservative_demo_q [baseline]" in text
    assert "b10_final_contact_advantage_gate_eval" in text
    assert "b10_final_contact_advantage_gate_eval [recommended]" not in text
    assert "b11_advantage_gate_correction_collect" in text
    assert "b11_advantage_gate_correction_finetune" in text
    assert "b13_route_masked_correction_finetune" in text
    assert "b14_contact_head_finetune" in text
    assert "b15_contact_head_offline" in text
    assert "b15_contact_head_offline [recommended]" not in text
    assert "b16_contact_head_jump_gated" in text
    assert "b16_contact_head_jump_gated [recommended]" not in text
    assert "b17_advantage_gate_correction_collect_seed1" in text
    assert "b17_advantage_gate_correction_collect_seed1 [recommended]" not in text
    assert "b18_contact_head_combined_calibration" in text
    assert "b18_contact_head_combined_calibration [recommended]" not in text
    assert "b9_c51_distributional" in text
    assert "b9_c51_distributional [recommended]" not in text
    assert "b8_history_state_conservative" in text
    assert "b7_contact_pool_interleaved" in text
    assert "b6_contact_interleaved" in text
    assert "b4_contact_only_correction_collect" in text
    assert "b5_anchored_contact_correction" in text
    assert "b3s_final_contact_option_eval" in text
    assert "requires override: --checkpoint" in text
    assert "requires override: --checkpoint, --correction-dataset" in text
    assert "run-recipe <key>" in text


def test_expand_recipe_argv_keeps_user_overrides_last():
    expanded = expand_recipe_argv(
        (
            "experiments/cc_status_session.py",
            "run-recipe",
            "b3s_conservative_smoke",
            "--episodes",
            "2",
            "--label",
            "override_label",
        )
    )

    assert expanded[0] == "experiments/cc_status_session.py"
    assert expanded[1] == "tutorial-demo-conservative"
    assert expanded[-4:] == ("--episodes", "2", "--label", "override_label")


def test_expand_recipe_argv_requires_known_recipe():
    with pytest.raises(KeyError, match="Unknown status-session recipe"):
        expand_recipe_argv(("runner", "run-recipe", "missing"))

    with pytest.raises(ValueError, match="requires a recipe key"):
        expand_recipe_argv(("runner", "run-recipe"))
