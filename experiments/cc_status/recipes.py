"""Named status-session recipes for comparable Crystal Caves NN runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

RecipeValue = str | int | float | bool | tuple[str, ...] | None


@dataclass(frozen=True)
class RecipeOption:
    """One CLI option for a status-session recipe."""

    name: str
    value: RecipeValue

    def cli_tokens(self) -> tuple[str, ...]:
        if self.value is None:
            return ()

        flag = f"--{self.name.replace('_', '-')}"
        if isinstance(self.value, bool):
            return (flag,) if self.value else ()
        if isinstance(self.value, tuple):
            return flag, ",".join(str(item) for item in self.value)
        return flag, str(self.value)


@dataclass(frozen=True)
class StatusSessionRecipe:
    """A reproducible status-session command and its promotion context."""

    key: str
    mode: str
    label: str
    description: str
    options: tuple[RecipeOption, ...]
    promotion_rule: str
    tags: tuple[str, ...] = ()
    baseline: bool = False
    recommended: bool = False
    required_overrides: tuple[str, ...] = ()
    docs: str = ""

    def option_value(self, name: str) -> RecipeValue:
        for option in self.options:
            if option.name == name:
                return option.value
        raise KeyError(f"Recipe '{self.key}' has no option '{name}'")

    def cli_args(self) -> tuple[str, ...]:
        args: list[str] = [self.mode]
        for option in self.options:
            args.extend(option.cli_tokens())
        return tuple(args)

    def python_command(
        self,
        python_bin: str = "/Users/justin/.pyenv/versions/3.12.11/bin/python",
    ) -> tuple[str, ...]:
        return (
            python_bin,
            "-u",
            "experiments/cc_status_session.py",
            *self.cli_args(),
        )


B3S_FULL_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 300),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 16),
    RecipeOption("selected_eval_games", 30),
    RecipeOption("train_eval_games", 8),
    RecipeOption("eval_every", 50),
    RecipeOption("trace_eval_games", 4),
    RecipeOption("trace_max_steps", 3000),
    RecipeOption("trace_sample_every", 25),
    RecipeOption("trace_tail_steps", 120),
    RecipeOption("vec_envs", 8),
    RecipeOption("cave_pool_size", 512),
    RecipeOption("route_demo_levels", 128),
    RecipeOption("route_demo_max_steps", 800),
    RecipeOption("route_demo_variants", ("direct", "recovery")),
    RecipeOption("demo_selection_mode", "all"),
    RecipeOption("bc_epochs", 6),
    RecipeOption("bc_batch_size", 128),
    RecipeOption("demo_repeat", 4),
    RecipeOption("demo_action_weight", 0.03),
    RecipeOption("demo_action_margin", 0.8),
    RecipeOption("demo_conservative_weight", 0.02),
    RecipeOption("demo_conservative_temperature", 1.0),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "tutorial_demo_conservative_recovery_pool512_select30_300"),
)

B3S_SMOKE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 1),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 1),
    RecipeOption("selected_eval_games", 1),
    RecipeOption("train_eval_games", 1),
    RecipeOption("eval_every", 1),
    RecipeOption("trace_eval_games", 1),
    RecipeOption("trace_max_steps", 80),
    RecipeOption("trace_sample_every", 20),
    RecipeOption("trace_tail_steps", 40),
    RecipeOption("vec_envs", 1),
    RecipeOption("cave_pool_size", 64),
    RecipeOption("route_demo_levels", 16),
    RecipeOption("route_demo_max_steps", 800),
    RecipeOption("route_demo_variants", ("direct", "recovery")),
    RecipeOption("demo_selection_mode", "all"),
    RecipeOption("bc_epochs", 1),
    RecipeOption("bc_batch_size", 64),
    RecipeOption("demo_repeat", 1),
    RecipeOption("demo_action_weight", 0.03),
    RecipeOption("demo_action_margin", 0.8),
    RecipeOption("demo_conservative_weight", 0.02),
    RecipeOption("demo_conservative_temperature", 1.0),
    RecipeOption("heartbeat_seconds", 10),
    RecipeOption("log_every", 1),
    RecipeOption("report_seconds", 1),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "b3s_conservative_smoke"),
)

B3S_CORRECTION_COLLECT_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("correction_games", 30),
    RecipeOption("correction_max_steps", 1200),
    RecipeOption("correction_max_examples", 1024),
    RecipeOption("correction_sample_every", 4),
    RecipeOption("correction_max_examples_per_game", 64),
    RecipeOption("correction_stale_steps", 90),
    RecipeOption("correction_loop_tile_visits", 8),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b3s_correction_collect_disagreement"),
)

B3S_CORRECTION_FINETUNE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 300),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 16),
    RecipeOption("selected_eval_games", 30),
    RecipeOption("train_eval_games", 8),
    RecipeOption("eval_every", 50),
    RecipeOption("trace_eval_games", 4),
    RecipeOption("trace_max_steps", 3000),
    RecipeOption("trace_sample_every", 25),
    RecipeOption("trace_tail_steps", 120),
    RecipeOption("vec_envs", 8),
    RecipeOption("correction_action_weight", 0.02),
    RecipeOption("correction_action_margin", 0.6),
    RecipeOption("correction_action_batch_size", 64),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "b3s_correction_finetune_300"),
)

RECIPES: dict[str, StatusSessionRecipe] = {
    "b3s_conservative_demo_q": StatusSessionRecipe(
        key="b3s_conservative_demo_q",
        mode="tutorial-demo-conservative",
        label="tutorial_demo_conservative_recovery_pool512_select30_300",
        description=(
            "Current promoted Crystal Caves route baseline: direct+recovery "
            "successful demos, online demo action margin, and conservative demo-Q."
        ),
        options=B3S_FULL_OPTIONS,
        promotion_rule=(
            "Future route methods should beat B3s's 10/30 selected first-crystal "
            "wins and confirm on expanded 60-game validation, or tie wins while "
            "materially improving close-zone distance, jump/contact, and loop metrics."
        ),
        tags=("crystal-caves", "route", "baseline", "conservative-demo-q"),
        baseline=True,
        recommended=True,
        docs="docs/cc_nn_experiment_tracker/findings_2026_06_24_part2.md#2026-06-24-b3s-conservative-demo-q-regularizer",
    ),
    "b3s_conservative_smoke": StatusSessionRecipe(
        key="b3s_conservative_smoke",
        mode="tutorial-demo-conservative",
        label="b3s_conservative_smoke",
        description="Fast mechanics check for the promoted B3s conservative demo-Q path.",
        options=B3S_SMOKE_OPTIONS,
        promotion_rule="Smoke only; never use this one-episode run for promotion decisions.",
        tags=("crystal-caves", "smoke", "conservative-demo-q"),
        docs="docs/cc_nn_experiment_tracker/findings_2026_06_24_part2.md#2026-06-24-b3s-conservative-demo-q-regularizer",
    ),
    "b3s_correction_collect": StatusSessionRecipe(
        key="b3s_correction_collect",
        mode="collect-corrections",
        label="b3s_correction_collect_disagreement",
        description=(
            "Collect disagreement-only policy-visited correction labels from the "
            "current B3s selected checkpoint."
        ),
        options=B3S_CORRECTION_COLLECT_OPTIONS,
        promotion_rule=(
            "Dataset artifact only; inspect disagreement rate, trigger mix, and "
            "label action mix before using it for correction-finetune."
        ),
        tags=("crystal-caves", "correction", "dataset", "b3s"),
        recommended=True,
        required_overrides=("checkpoint",),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#correction-dataset-collection",
    ),
    "b3s_correction_finetune": StatusSessionRecipe(
        key="b3s_correction_finetune",
        mode="correction-finetune",
        label="b3s_correction_finetune_300",
        description=(
            "Fine-tune the B3s selected checkpoint with a low-weight "
            "policy-visited correction action loss."
        ),
        options=B3S_CORRECTION_FINETUNE_OPTIONS,
        promotion_rule=(
            "Compare selected eval against B3s's 10/30 and validate against the "
            "19/60 expanded baseline before promotion."
        ),
        tags=("crystal-caves", "correction", "finetune", "b3s"),
        recommended=True,
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#correction-fine-tune",
    ),
}


def get_recipe(key: str) -> StatusSessionRecipe:
    try:
        return RECIPES[key]
    except KeyError as exc:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"Unknown status-session recipe '{key}'. Known recipes: {known}") from exc


def recommended_recipes() -> tuple[StatusSessionRecipe, ...]:
    return tuple(recipe for recipe in RECIPES.values() if recipe.recommended)


def recipe_rows(recipes: Iterable[StatusSessionRecipe] = RECIPES.values()) -> tuple[str, ...]:
    rows = []
    for recipe in sorted(recipes, key=lambda item: item.key):
        flags = []
        if recipe.baseline:
            flags.append("baseline")
        if recipe.recommended:
            flags.append("recommended")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        requires = _format_required_overrides(recipe)
        rows.append(f"{recipe.key}{flag_text}: {recipe.description}{requires}")
    return tuple(rows)


def format_recipe_list() -> str:
    return "\n".join(
        (
            "Available Crystal Caves status-session recipes:",
            *recipe_rows(),
            "",
            "Run one with: python experiments/cc_status_session.py run-recipe <key> [overrides]",
        )
    )


def expand_recipe_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Expand ``run-recipe KEY`` into the underlying status-session CLI argv."""
    if len(argv) < 3:
        known = ", ".join(sorted(RECIPES))
        raise ValueError(f"run-recipe requires a recipe key. Known recipes: {known}")
    recipe = get_recipe(argv[2])
    _validate_required_overrides(recipe, argv[3:])
    return (argv[0], *recipe.cli_args(), *argv[3:])


def _format_required_overrides(recipe: StatusSessionRecipe) -> str:
    if not recipe.required_overrides:
        return ""
    flags = ", ".join(_flag_for_name(name) for name in recipe.required_overrides)
    return f" (requires override: {flags})"


def _flag_for_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _override_flags(argv: Sequence[str]) -> set[str]:
    flags: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        flags.add(flag)
    return flags


def _validate_required_overrides(
    recipe: StatusSessionRecipe, overrides: Sequence[str]
) -> None:
    if not recipe.required_overrides:
        return
    present = _override_flags(overrides)
    missing = [
        _flag_for_name(name)
        for name in recipe.required_overrides
        if _flag_for_name(name) not in present
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"run-recipe {recipe.key} requires explicit override(s): {missing_text}"
        )
