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

B8_HISTORY_STATE_OPTIONS: tuple[RecipeOption, ...] = (
    *B3S_FULL_OPTIONS[:-1],
    RecipeOption("history_state", True),
    RecipeOption("history_steps", 4),
    RecipeOption("label", "b8_history_state_conservative_pool512_select30_300"),
)

B8_HISTORY_STATE_SMOKE_OPTIONS: tuple[RecipeOption, ...] = (
    *B3S_SMOKE_OPTIONS[:-1],
    RecipeOption("history_state", True),
    RecipeOption("history_steps", 4),
    RecipeOption("label", "b8_history_state_smoke"),
)

B9_C51_DISTRIBUTIONAL_OPTIONS: tuple[RecipeOption, ...] = (
    *B3S_FULL_OPTIONS[:-1],
    RecipeOption("distributional_dqn", True),
    RecipeOption("c51_atoms", 51),
    RecipeOption("c51_v_min", -20.0),
    RecipeOption("c51_v_max", 120.0),
    RecipeOption("label", "b9_c51_distributional_pool512_select30_300"),
)

B9_C51_DISTRIBUTIONAL_SMOKE_OPTIONS: tuple[RecipeOption, ...] = (
    *B3S_SMOKE_OPTIONS[:-1],
    RecipeOption("distributional_dqn", True),
    RecipeOption("c51_atoms", 51),
    RecipeOption("c51_v_min", -20.0),
    RecipeOption("c51_v_max", 120.0),
    RecipeOption("label", "b9_c51_distributional_smoke"),
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

B3S_CONTACT_ONLY_CORRECTION_COLLECT_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("correction_games", 30),
    RecipeOption("correction_max_steps", 1200),
    RecipeOption("correction_max_examples", 512),
    RecipeOption("correction_sample_every", 4),
    RecipeOption("correction_max_examples_per_game", 64),
    RecipeOption("correction_stale_steps", 999999),
    RecipeOption("correction_loop_tile_visits", 999999),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b4_contact_only_correction_collect"),
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

B5_ANCHORED_CONTACT_CORRECTION_OPTIONS: tuple[RecipeOption, ...] = (
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
    RecipeOption("correction_action_weight", 0.005),
    RecipeOption("correction_action_margin", 0.6),
    RecipeOption("correction_action_batch_size", 64),
    RecipeOption("policy_anchor_weight", 0.02),
    RecipeOption("policy_anchor_temperature", 1.0),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "b5_anchored_contact_correction_w005_a002_300"),
)

B6_CONTACT_INTERLEAVED_OPTIONS: tuple[RecipeOption, ...] = (
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
    RecipeOption("interleave_contact_ratio", 0.25),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "b6_contact_interleaved_25pct_300"),
)

B7_CONTACT_POOL_INTERLEAVED_OPTIONS: tuple[RecipeOption, ...] = (
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
    RecipeOption("interleave_contact_ratio", 0.125),
    RecipeOption("contact_pool_size", 128),
    RecipeOption("contact_eval_pool_size", 32),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "b7_contact_pool_interleaved_12pct_300"),
)

B7_CONTACT_POOL_SMOKE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 20),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 4),
    RecipeOption("selected_eval_games", 4),
    RecipeOption("train_eval_games", 2),
    RecipeOption("eval_every", 10),
    RecipeOption("trace_eval_games", 1),
    RecipeOption("trace_max_steps", 120),
    RecipeOption("trace_sample_every", 30),
    RecipeOption("trace_tail_steps", 60),
    RecipeOption("vec_envs", 4),
    RecipeOption("cave_pool_size", 64),
    RecipeOption("interleave_contact_ratio", 0.25),
    RecipeOption("contact_pool_size", 16),
    RecipeOption("contact_eval_pool_size", 4),
    RecipeOption("heartbeat_seconds", 0),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("save_selected_checkpoint", True),
    RecipeOption("label", "b7_contact_pool_smoke_20"),
)

B3S_FINAL_CONTACT_OPTION_SMOKE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 4),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b3s_final_contact_option_smoke"),
)

B3S_FINAL_CONTACT_OPTION_EVAL_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 30),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b3s_final_contact_option_eval30"),
)

B10_FINAL_CONTACT_ADVANTAGE_GATE_SMOKE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 4),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("final_contact_cancel_outside", True),
    RecipeOption("final_contact_policy_advantage_gate", True),
    RecipeOption("final_contact_min_option_advantage", 250.0),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b10_final_contact_advantage_gate_smoke"),
)

B10_FINAL_CONTACT_ADVANTAGE_GATE_EVAL_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 30),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("final_contact_cancel_outside", True),
    RecipeOption("final_contact_policy_advantage_gate", True),
    RecipeOption("final_contact_min_option_advantage", 250.0),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b10_final_contact_advantage_gate_eval30"),
)

B11_ADVANTAGE_GATE_CORRECTION_COLLECT_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("correction_games", 60),
    RecipeOption("correction_max_steps", 1200),
    RecipeOption("correction_max_examples", 256),
    RecipeOption("correction_sample_every", 1),
    RecipeOption("correction_max_examples_per_game", 8),
    RecipeOption("correction_stale_steps", 999999),
    RecipeOption("correction_loop_tile_visits", 999999),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("final_contact_policy_advantage_gate", True),
    RecipeOption("final_contact_min_option_advantage", 250.0),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b11_advantage_gate_correction_collect"),
)

B17_ADVANTAGE_GATE_CORRECTION_COLLECT_SEED1_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 1),
    RecipeOption("correction_games", 60),
    RecipeOption("correction_max_steps", 1200),
    RecipeOption("correction_max_examples", 512),
    RecipeOption("correction_sample_every", 1),
    RecipeOption("correction_max_examples_per_game", 12),
    RecipeOption("correction_stale_steps", 999999),
    RecipeOption("correction_loop_tile_visits", 999999),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("final_contact_policy_advantage_gate", True),
    RecipeOption("final_contact_min_option_advantage", 250.0),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b17_advantage_gate_correction_collect_seed1"),
)

B11_ADVANTAGE_GATE_DATASET_PATH = ".Codex/artifacts/cc_sessions/20260625_125000_b11_advantage_gate_correction_collect/20260625_123917_b11_advantage_gate_correction_collect/b11_advantage_gate_correction_collect/corrections/b11_advantage_gate_correction_collect_heldout/correction_examples.npz"
B17_ADVANTAGE_GATE_DATASET_PATH = ".Codex/artifacts/cc_sessions/20260625_165300_b17_advantage_gate_correction_collect_seed1/b17_advantage_gate_correction_collect_seed1/corrections/b17_advantage_gate_correction_collect_seed1_heldout/correction_examples.npz"

B18_CONTACT_HEAD_COMBINED_CALIBRATION_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption(
        "correction_datasets",
        (B11_ADVANTAGE_GATE_DATASET_PATH, B17_ADVANTAGE_GATE_DATASET_PATH),
    ),
    RecipeOption("contact_action_batch_size", 32),
    RecipeOption("contact_action_distance", 3.0),
    RecipeOption("contact_head_offline_steps", 500),
    RecipeOption("contact_head_learning_rate", 0.001),
    RecipeOption("contact_head_balance_classes", True),
    RecipeOption("contact_head_calibration_frac", 0.25),
    RecipeOption("contact_head_calibration_seed", 18),
    RecipeOption("contact_head_min_calibration_accuracy", 0.70),
    RecipeOption("contact_head_min_class_examples", 10),
    RecipeOption("heartbeat_seconds", 0),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b18_contact_head_combined_calibration_b11_b17"),
)

B11_ADVANTAGE_GATE_CORRECTION_FINETUNE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 150),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 16),
    RecipeOption("train_eval_games", 8),
    RecipeOption("eval_every", 50),
    RecipeOption("vec_envs", 8),
    RecipeOption("correction_action_weight", 0.001),
    RecipeOption("correction_action_margin", 0.6),
    RecipeOption("correction_action_batch_size", 32),
    RecipeOption("policy_anchor_weight", 0.05),
    RecipeOption("policy_anchor_temperature", 1.5),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b11_advantage_gate_correction_finetune_w001_a005_150"),
)

B13_ROUTE_MASKED_CORRECTION_FINETUNE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 150),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 16),
    RecipeOption("train_eval_games", 8),
    RecipeOption("eval_every", 50),
    RecipeOption("vec_envs", 8),
    RecipeOption("correction_action_weight", 0.001),
    RecipeOption("correction_action_margin", 0.6),
    RecipeOption("correction_action_batch_size", 32),
    RecipeOption("policy_anchor_weight", 0.10),
    RecipeOption("policy_anchor_temperature", 1.5),
    RecipeOption("policy_anchor_min_distance_tiles", 3.0),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b13_route_masked_correction_finetune_w001_a010_150"),
)

B14_CONTACT_HEAD_FINETUNE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("episodes", 150),
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 16),
    RecipeOption("train_eval_games", 8),
    RecipeOption("eval_every", 50),
    RecipeOption("vec_envs", 8),
    RecipeOption("contact_action_weight", 0.02),
    RecipeOption("contact_action_batch_size", 32),
    RecipeOption("contact_action_distance", 3.0),
    RecipeOption("heartbeat_seconds", 30),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b14_contact_head_finetune_w002_150"),
)

B15_CONTACT_HEAD_OFFLINE_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 0),
    RecipeOption("eval_games", 30),
    RecipeOption("contact_action_batch_size", 32),
    RecipeOption("contact_action_distance", 3.0),
    RecipeOption("contact_head_offline_steps", 500),
    RecipeOption("contact_head_learning_rate", 0.001),
    RecipeOption("contact_head_confidence", 0.75),
    RecipeOption("contact_head_balance_classes", True),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b15_contact_head_offline_balanced_conf075_500_eval30"),
)

B16_CONTACT_HEAD_JUMP_GATED_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 1),
    RecipeOption("eval_games", 30),
    RecipeOption("contact_action_batch_size", 32),
    RecipeOption("contact_action_distance", 3.0),
    RecipeOption("contact_head_offline_steps", 500),
    RecipeOption("contact_head_learning_rate", 0.001),
    RecipeOption("contact_head_confidence", 0.75),
    RecipeOption("contact_head_jump_confidence", 0.85),
    RecipeOption("contact_head_balance_classes", True),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b16_contact_head_jump_conf085_seed1_eval30"),
)

B24_CONTACT_HEAD_LEFT_JUMP_GATED_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 1),
    RecipeOption("eval_games", 30),
    RecipeOption("contact_action_batch_size", 32),
    RecipeOption("contact_action_distance", 3.0),
    RecipeOption("contact_head_offline_steps", 500),
    RecipeOption("contact_head_learning_rate", 0.001),
    RecipeOption("contact_head_confidence", 0.75),
    RecipeOption("contact_head_action_thresholds", "LEFT_JUMP:0.90"),
    RecipeOption("contact_head_balance_classes", True),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b24_stable_contact_head_left_jump_conf090_seed1_eval30"),
)

B25_CONTACT_HEAD_CORRECTION_COLLECT_OPTIONS: tuple[RecipeOption, ...] = (
    RecipeOption("seed", 1),
    RecipeOption("correction_games", 60),
    RecipeOption("correction_max_steps", 1200),
    RecipeOption("correction_max_examples", 512),
    RecipeOption("correction_sample_every", 1),
    RecipeOption("correction_max_examples_per_game", 12),
    RecipeOption("correction_stale_steps", 999999),
    RecipeOption("correction_loop_tile_visits", 999999),
    RecipeOption("final_contact_distance", 3.0),
    RecipeOption("final_contact_commit_steps", 8),
    RecipeOption("final_contact_policy_advantage_gate", True),
    RecipeOption("final_contact_min_option_advantage", 250.0),
    RecipeOption("contact_action_batch_size", 32),
    RecipeOption("contact_action_distance", 3.0),
    RecipeOption("contact_head_offline_steps", 500),
    RecipeOption("contact_head_learning_rate", 0.001),
    RecipeOption("contact_head_confidence", 0.75),
    RecipeOption("contact_head_action_thresholds", "LEFT_JUMP:0.90"),
    RecipeOption("contact_head_balance_classes", True),
    RecipeOption("log_every", 5),
    RecipeOption("report_seconds", 20),
    RecipeOption("label", "b25_b24_policy_visited_contact_collect_seed1"),
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
        recommended=False,
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
        required_overrides=("checkpoint",),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#correction-dataset-collection",
    ),
    "b4_contact_only_correction_collect": StatusSessionRecipe(
        key="b4_contact_only_correction_collect",
        mode="collect-corrections",
        label="b4_contact_only_correction_collect",
        description=(
            "Collect close-zone-only policy-visited correction labels from the "
            "B3s selected checkpoint by disabling stale and loop triggers."
        ),
        options=B3S_CONTACT_ONLY_CORRECTION_COLLECT_OPTIONS,
        promotion_rule=(
            "Dataset artifact only; proceed to fine-tune only if close-zone "
            "disagreement examples are numerous and action mix is usable."
        ),
        tags=("crystal-caves", "correction", "dataset", "close-zone", "b3s", "b4"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#latest-tested-method-and-next-direction",
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
            "Archived comparison path; B4 showed unanchored correction fine-tunes "
            "regress route depth."
        ),
        tags=("crystal-caves", "correction", "finetune", "b3s"),
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#correction-fine-tune",
    ),
    "b5_anchored_contact_correction": StatusSessionRecipe(
        key="b5_anchored_contact_correction",
        mode="correction-finetune",
        label="b5_anchored_contact_correction_w005_a002_300",
        description=(
            "Fine-tune B3s on the B4 contact-only dataset while anchoring the "
            "current policy to the restored B3s teacher on replay states."
        ),
        options=B5_ANCHORED_CONTACT_CORRECTION_OPTIONS,
        promotion_rule=(
            "Beat B3s's 10/30 selected wins without losing the depth guardrail; "
            "early-stop if held-out depth stays shallow at ep100/150."
        ),
        tags=("crystal-caves", "correction", "policy-anchor", "b3s", "b5"),
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#latest-tested-method-and-next-direction",
    ),
    "b6_contact_interleaved": StatusSessionRecipe(
        key="b6_contact_interleaved",
        mode="contact-interleaved",
        label="b6_contact_interleaved_25pct_300",
        description=(
            "Restore the B3s selected route policy, then train normal tutorial lanes "
            "interleaved with tiny contact-skill lanes using normal RL rewards only."
        ),
        options=B6_CONTACT_INTERLEAVED_OPTIONS,
        promotion_rule=(
            "Promote only if selected 30-game held-out wins beat B3s 10/30 or tie it "
            "while improving contact-lane exit rate, full-lane first-crystal rate, and "
            "near-miss distance without a depth regression."
        ),
        tags=("crystal-caves", "contact", "interleaved", "b3s", "b6"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#latest-tested-method-and-next-direction",
    ),
    "b7_contact_pool_interleaved": StatusSessionRecipe(
        key="b7_contact_pool_interleaved",
        mode="contact-interleaved",
        label="b7_contact_pool_interleaved_12pct_300",
        description=(
            "Restore B3s, then train mostly normal tutorial lanes with one generated "
            "contact-pool lane and a held-out contact-pool eval."
        ),
        options=B7_CONTACT_POOL_INTERLEAVED_OPTIONS,
        promotion_rule=(
            "Use the route/contact scorecard as the first gate: stop early if the "
            "candidate trails B3s's 1.821 score or loses route depth before improving "
            "contact-pool eval. Promote only through the normal selected/validation gate."
        ),
        tags=("crystal-caves", "contact-pool", "interleaved", "b3s", "b7"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r3-game-faithful-contact-curriculum-variant",
    ),
    "b7_contact_pool_smoke": StatusSessionRecipe(
        key="b7_contact_pool_smoke",
        mode="contact-interleaved",
        label="b7_contact_pool_smoke_20",
        description=(
            "Fast smoke for generated contact-pool interleaving and held-out contact-pool eval."
        ),
        options=B7_CONTACT_POOL_SMOKE_OPTIONS,
        promotion_rule="Smoke only; proves generated contact pool artifacts and metrics work.",
        tags=("crystal-caves", "contact-pool", "interleaved", "smoke", "b7"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r3-game-faithful-contact-curriculum-variant",
    ),
    "b8_history_state_conservative": StatusSessionRecipe(
        key="b8_history_state_conservative",
        mode="tutorial-demo-conservative",
        label="b8_history_state_conservative_pool512_select30_300",
        description=(
            "Fresh B3s-style conservative demo-Q run with opt-in 4-step action/approach "
            "history appended to Crystal Caves metadata."
        ),
        options=B8_HISTORY_STATE_OPTIONS,
        promotion_rule=(
            "Compare against B3s using the route/contact scorecard first; history changes "
            "state size, so this is a fresh architecture probe, not a B3s checkpoint fine-tune."
        ),
        tags=("crystal-caves", "history-state", "architecture", "b8"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r4-routepolicy-architecture-probe",
    ),
    "b8_history_state_smoke": StatusSessionRecipe(
        key="b8_history_state_smoke",
        mode="tutorial-demo-conservative",
        label="b8_history_state_smoke",
        description="Fast smoke for the opt-in Crystal Caves history-state architecture.",
        options=B8_HISTORY_STATE_SMOKE_OPTIONS,
        promotion_rule="Smoke only; proves the larger state size trains, saves, and reports.",
        tags=("crystal-caves", "history-state", "architecture", "smoke", "b8"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r4-routepolicy-architecture-probe",
    ),
    "b9_c51_distributional": StatusSessionRecipe(
        key="b9_c51_distributional",
        mode="tutorial-demo-conservative",
        label="b9_c51_distributional_pool512_select30_300",
        description=(
            "Fresh B3s-style conservative demo-Q run with opt-in C51 distributional "
            "DQN value head/loss."
        ),
        options=B9_C51_DISTRIBUTIONAL_OPTIONS,
        promotion_rule=(
            "Compare against B3s and B8 using the route/contact scorecard. Promote "
            "only if selected wins beat B3s or tie while improving route/contact "
            "score, depth, or close-zone behavior; confirm with expanded validation."
        ),
        tags=("crystal-caves", "distributional-dqn", "c51", "algorithm", "b9"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r5-algorithm-upgrade-distributional-or-ppo-style-probe",
    ),
    "b9_c51_distributional_smoke": StatusSessionRecipe(
        key="b9_c51_distributional_smoke",
        mode="tutorial-demo-conservative",
        label="b9_c51_distributional_smoke",
        description="Fast smoke for the opt-in C51 distributional DQN path.",
        options=B9_C51_DISTRIBUTIONAL_SMOKE_OPTIONS,
        promotion_rule="Smoke only; proves the C51 head/loss trains, saves, and reports.",
        tags=("crystal-caves", "distributional-dqn", "c51", "algorithm", "smoke", "b9"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r5-algorithm-upgrade-distributional-or-ppo-style-probe",
    ),
    "b3s_final_contact_option_smoke": StatusSessionRecipe(
        key="b3s_final_contact_option_smoke",
        mode="eval-final-contact-option",
        label="b3s_final_contact_option_smoke",
        description=("Fast artifact-validation smoke for the B3s eval-only final-contact option."),
        options=B3S_FINAL_CONTACT_OPTION_SMOKE_OPTIONS,
        promotion_rule="Smoke only; use the 30-game recipe and 60-game validation for decisions.",
        tags=("crystal-caves", "final-contact", "option", "smoke", "b3s"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#latest-tested-method-and-next-direction",
    ),
    "b3s_final_contact_option_eval": StatusSessionRecipe(
        key="b3s_final_contact_option_eval",
        mode="eval-final-contact-option",
        label="b3s_final_contact_option_eval30",
        description=(
            "Evaluate the B3s selected checkpoint with a close-zone final-contact "
            "option layered over the NN route policy."
        ),
        options=B3S_FINAL_CONTACT_OPTION_EVAL_OPTIONS,
        promotion_rule=(
            "Beat B3s's 10/30 selected wins, then rerun with --eval-games 60 and "
            "pass the expanded-validation depth guardrail before promotion."
        ),
        tags=("crystal-caves", "final-contact", "option", "b3s"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXTENSION_ARCHITECTURE.md#latest-tested-method-and-next-direction",
    ),
    "b10_final_contact_advantage_gate_smoke": StatusSessionRecipe(
        key="b10_final_contact_advantage_gate_smoke",
        mode="eval-final-contact-option",
        label="b10_final_contact_advantage_gate_smoke",
        description=(
            "Fast smoke for policy-advantage-gated final-contact control on top of "
            "the B3s selected checkpoint."
        ),
        options=B10_FINAL_CONTACT_ADVANTAGE_GATE_SMOKE_OPTIONS,
        promotion_rule=(
            "Smoke only; proves the policy-vs-option gate records acceptance and "
            "rejection metrics."
        ),
        tags=("crystal-caves", "final-contact", "option", "advantage-gate", "smoke", "b10"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#next-recommendation",
    ),
    "b10_final_contact_advantage_gate_eval": StatusSessionRecipe(
        key="b10_final_contact_advantage_gate_eval",
        mode="eval-final-contact-option",
        label="b10_final_contact_advantage_gate_eval30",
        description=(
            "Evaluate the B3s selected checkpoint with a final-contact option that "
            "only overrides the NN route policy when simulated option advantage is clear."
        ),
        options=B10_FINAL_CONTACT_ADVANTAGE_GATE_EVAL_OPTIONS,
        promotion_rule=(
            "Promote only if it keeps the selected win lift, confirms on 60-game "
            "validation, and either raw depth or outcome-conditioned non-success "
            "route depth clears the B3s guardrail."
        ),
        tags=("crystal-caves", "final-contact", "option", "advantage-gate", "b10"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#next-recommendation",
    ),
    "b11_advantage_gate_correction_collect": StatusSessionRecipe(
        key="b11_advantage_gate_correction_collect",
        mode="collect-corrections",
        label="b11_advantage_gate_correction_collect",
        description=(
            "Collect only B10 policy-advantage-gate accepted close-zone disagreement "
            "labels from the B3s selected checkpoint."
        ),
        options=B11_ADVANTAGE_GATE_CORRECTION_COLLECT_OPTIONS,
        promotion_rule=(
            "Dataset only; inspect kept count, disagreement rate, action mix, and gate "
            "rejection rate before using it for fine-tuning."
        ),
        tags=("crystal-caves", "correction", "dataset", "advantage-gate", "b11"),
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r6-action-selection-control-probe-b10-advantage-gated-final-contact",
    ),
    "b17_advantage_gate_correction_collect_seed1": StatusSessionRecipe(
        key="b17_advantage_gate_correction_collect_seed1",
        mode="collect-corrections",
        label="b17_advantage_gate_correction_collect_seed1",
        description=(
            "Collect a larger hard-seed B10 policy-advantage-gate accepted close-zone "
            "label set from the B3s selected checkpoint for contact-head data coverage."
        ),
        options=B17_ADVANTAGE_GATE_CORRECTION_COLLECT_SEED1_OPTIONS,
        promotion_rule=(
            "Dataset only; compare kept count, disagreement rate, action mix, and gate "
            "rejection rate against B11 before building a combined/calibrated contact "
            "head run. Do not promote from dataset collection alone."
        ),
        tags=("crystal-caves", "correction", "dataset", "advantage-gate", "hard-seed", "b17"),
        recommended=False,
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r11-b16-jump-gated-offline-contact-head",
    ),
    "b18_contact_head_combined_calibration": StatusSessionRecipe(
        key="b18_contact_head_combined_calibration",
        mode="contact-head-calibrate",
        label="b18_contact_head_combined_calibration_b11_b17",
        description=(
            "Combine B11 seed-0 and B17 seed-1 B10-gated labels, hold out a "
            "stratified calibration split, and fit only the frozen-route contact head."
        ),
        options=B18_CONTACT_HEAD_COMBINED_CALIBRATION_OPTIONS,
        promotion_rule=(
            "Dataset/calibration only. Continue to a selected held-out contact-head "
            "eval only if calibration passes overall accuracy, class coverage, and "
            "route-weight immobility checks."
        ),
        tags=("crystal-caves", "contact-head", "calibration", "dataset", "b18"),
        recommended=False,
        required_overrides=("checkpoint",),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r13-b18-combined-contact-head-calibration",
    ),
    "b11_advantage_gate_correction_finetune": StatusSessionRecipe(
        key="b11_advantage_gate_correction_finetune",
        mode="correction-finetune",
        label="b11_advantage_gate_correction_finetune_w001_a005_150",
        description=(
            "Low-weight correction fine-tune from B10 gate-accepted labels with a "
            "small frozen-policy anchor to preserve B3s route behavior."
        ),
        options=B11_ADVANTAGE_GATE_CORRECTION_FINETUNE_OPTIONS,
        promotion_rule=(
            "Treat as a route-preservation probe. Continue only if held-out depth stays "
            "near B3s while crystal/contact improves; otherwise archive like B4/B5."
        ),
        tags=("crystal-caves", "correction", "finetune", "advantage-gate", "b11"),
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#r6-action-selection-control-probe-b10-advantage-gated-final-contact",
    ),
    "b13_route_masked_correction_finetune": StatusSessionRecipe(
        key="b13_route_masked_correction_finetune",
        mode="correction-finetune",
        label="b13_route_masked_correction_finetune_w001_a010_150",
        description=(
            "Fine-tune from B10 gate-accepted labels while anchoring only states "
            "outside the close-zone route mask."
        ),
        options=B13_ROUTE_MASKED_CORRECTION_FINETUNE_OPTIONS,
        promotion_rule=(
            "Route-preserving B10 internalization probe. Continue only if correction "
            "accuracy rises while non-success depth and route/contact score stay near "
            "B3s; compare learned policy against B3s and B10."
        ),
        tags=("crystal-caves", "correction", "finetune", "advantage-gate", "b13"),
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#next-recommendation",
    ),
    "b14_contact_head_finetune": StatusSessionRecipe(
        key="b14_contact_head_finetune",
        mode="contact-head-finetune",
        label="b14_contact_head_finetune_w002_150",
        description=(
            "Train a detached close-zone action head from B10 gate-accepted labels and "
            "use it only inside the target-distance mask during held-out eval."
        ),
        options=B14_CONTACT_HEAD_FINETUNE_OPTIONS,
        promotion_rule=(
            "Mechanism probe after B13. Continue only if contact-head accuracy rises "
            "and selector eval beats B13 on wins or route/contact score without losing "
            "B3s-level non-success depth."
        ),
        tags=("crystal-caves", "contact-head", "finetune", "advantage-gate", "b14"),
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#next-recommendation",
    ),
    "b15_contact_head_offline": StatusSessionRecipe(
        key="b15_contact_head_offline",
        mode="contact-head-offline",
        label="b15_contact_head_offline_balanced_conf075_500_eval30",
        description=(
            "Freeze B3s and train only the detached close-zone action head offline "
            "from B10 gate-accepted labels with balanced classes and confidence gating."
        ),
        options=B15_CONTACT_HEAD_OFFLINE_OPTIONS,
        promotion_rule=(
            "Next pure-NN adapter probe after B14. Promote only if confidence-gated "
            "selector eval improves B13/B3s route-contact behavior without broad "
            "JUMP-heavy overrides; high supervised accuracy alone is insufficient."
        ),
        tags=("crystal-caves", "contact-head", "offline", "balanced", "b15"),
        recommended=False,
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#next-recommendation",
    ),
    "b16_contact_head_jump_gated": StatusSessionRecipe(
        key="b16_contact_head_jump_gated",
        mode="contact-head-offline",
        label="b16_contact_head_jump_conf085_seed1_eval30",
        description=(
            "B15 robustness probe: freeze B3s, train the balanced offline contact "
            "head, but require higher confidence before jump-variant head actions "
            "override the base policy on the seed-1 failure case."
        ),
        options=B16_CONTACT_HEAD_JUMP_GATED_OPTIONS,
        promotion_rule=(
            "Continue only if seed-1 beats the same-seed B3s/B15 controls on wins "
            "or route-contact score without losing non-success route depth; then "
            "rerun seed 0 to confirm the stricter gate does not erase B15's selected "
            "validation lift."
        ),
        tags=(
            "crystal-caves",
            "contact-head",
            "offline",
            "balanced",
            "jump-gate",
            "b16",
        ),
        recommended=False,
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#next-recommendation",
    ),
    "b24_contact_head_left_jump_gated": StatusSessionRecipe(
        key="b24_contact_head_left_jump_gated",
        mode="contact-head-offline",
        label="b24_stable_contact_head_left_jump_conf090_seed1_eval30",
        description=(
            "B21/B23 follow-up: keep the stable-label frozen-route contact head, "
            "but require extra confidence only for LEFT_JUMP overrides on the "
            "seed-1 route-depth failure case."
        ),
        options=B24_CONTACT_HEAD_LEFT_JUMP_GATED_OPTIONS,
        promotion_rule=(
            "Continue only if seed-1 selected eval preserves B21's 10/30 wins or "
            "improves route/contact score while recovering non-success route depth. "
            "If it passes, run seed-1 val60 and a seed-0 guardrail."
        ),
        tags=(
            "crystal-caves",
            "contact-head",
            "offline",
            "balanced",
            "per-action-gate",
            "b24",
        ),
        recommended=False,
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#strategic-reanalysis-after-b23",
    ),
    "b25_contact_head_policy_collect": StatusSessionRecipe(
        key="b25_contact_head_policy_collect",
        mode="collect-contact-head-corrections",
        label="b25_b24_policy_visited_contact_collect_seed1",
        description=(
            "Collect B10 advantage-gated labels from states visited by the B24-style "
            "frozen-route contact-head adapter, so the next stable-label filter sees "
            "the learned selector's own failure distribution."
        ),
        options=B25_CONTACT_HEAD_CORRECTION_COLLECT_OPTIONS,
        promotion_rule=(
            "Dataset only. Continue only if kept labels add useful class coverage and "
            "the following B20-style filter/audit passes stability checks before any "
            "new contact-head eval."
        ),
        tags=(
            "crystal-caves",
            "correction",
            "dataset",
            "contact-head",
            "policy-visited",
            "b25",
        ),
        recommended=False,
        required_overrides=("checkpoint", "correction_dataset"),
        docs="CC_NN_EXPERIMENT_TRACKER.md#strategic-reanalysis-after-b24",
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


def _validate_required_overrides(recipe: StatusSessionRecipe, overrides: Sequence[str]) -> None:
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
        raise ValueError(f"run-recipe {recipe.key} requires explicit override(s): {missing_text}")
