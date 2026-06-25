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
    assert recipe.recommended is True
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
    assert recipe.recommended is True
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
    assert recipe.recommended is True
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


def test_recommended_recipes_are_explicit():
    recommended = recommended_recipes()

    assert [recipe.key for recipe in recommended] == [
        "b3s_conservative_demo_q",
        "b3s_correction_collect",
        "b3s_correction_finetune",
    ]


def test_recipe_list_is_user_facing_and_marks_baseline():
    text = format_recipe_list()

    assert "b3s_conservative_demo_q [baseline recommended]" in text
    assert "b3s_correction_collect [recommended]" in text
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
