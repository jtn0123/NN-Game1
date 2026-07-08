# ruff: noqa: F401,F403,F405,I001
from .common import *

STATUS_SESSION_MODES = (
    "baseline",
    "diagnose-baseline",
    "anti-loop",
    "novelty-bonus",
    "invalid-interact",
    "drill",
    "bridge",
    "transfer",
    "bridge-transfer",
    "bridge-demo-replay",
    "first-crystal-route",
    "first-crystal-direct",
    "tutorial-demo-bc",
    "tutorial-demo-dqfd",
    "tutorial-demo-conservative",
    "tutorial-demo-close-zone",
    "tutorial-demo-conservative-close-zone",
    "tutorial-demo-oracle-close-zone",
    "tutorial-demo-bridge-finetune",
    "route-demo-bc",
    "first-crystal-transfer",
    "baseline-and-transfer",
    "interleaved",
    "bridge-interleaved",
    "contact-interleaved",
    "reverse-start",
    "archive-start",
    "eval-checkpoint",
    "eval-final-contact-option",
    "collect-corrections",
    "collect-contact-head-corrections",
    "correction-finetune",
    "contact-head-finetune",
    "contact-head-offline",
    "contact-head-calibrate",
    "contact-label-audit",
    "contact-label-filter",
)


def add_status_session_arguments(parser: argparse.ArgumentParser) -> None:
    _add_mode_argument(parser)
    _add_training_schedule_arguments(parser)
    _add_state_architecture_arguments(parser)
    _add_reward_shaping_arguments(parser)
    _add_route_demo_arguments(parser)
    _add_demo_supervision_arguments(parser)
    _add_eval_logging_trace_arguments(parser)
    _add_interleave_bridge_arguments(parser)
    _add_reverse_archive_arguments(parser)
    _add_runtime_arguments(parser)
    _add_checkpoint_correction_arguments(parser)
    _add_artifact_persistence_arguments(parser)


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "mode",
        choices=STATUS_SESSION_MODES,
    )


def _add_training_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--drill-episodes", type=int, default=600)
    parser.add_argument("--bridge-episodes", type=int, default=400)
    parser.add_argument(
        "--bridge-finetune-episodes",
        type=int,
        default=100,
        help="For tutorial-demo-bridge-finetune, short bridge interleave fine-tune episodes after B3g route training.",
    )
    parser.add_argument(
        "--route-episodes",
        type=int,
        default=150,
        help="For first-crystal-transfer, first-crystal route pretraining episodes.",
    )
    parser.add_argument(
        "--route-floor-episodes",
        type=int,
        default=75,
        help="For first-crystal-route/route-demo-bc, route-floor pretraining episodes before normal tutorial route training.",
    )
    parser.add_argument(
        "--route-scaffold-difficulty",
        default="route_floor",
        choices=["route_floor", "route_catch", "route_offset"],
        help="For first-crystal-route/route-demo-bc, training-only scaffold difficulty.",
    )
    parser.add_argument(
        "--cave-pool-size",
        type=int,
        default=None,
        help="For procedural tutorial modes, override training cave pool size.",
    )


def _add_state_architecture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--history-state",
        action="store_true",
        help=(
            "Append compact short action/approach history to Crystal Caves metadata. "
            "Changes state size, so old checkpoints cannot be restored directly."
        ),
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=4,
        help="Number of recent action/approach entries to append when --history-state is enabled.",
    )
    parser.add_argument(
        "--geo-compass",
        action="store_true",
        help=(
            "Append the RUN-13 geodesic corridor compass (4 route-direction "
            "scalars) to the Crystal Caves state. Changes state size, so old "
            "checkpoints cannot be restored directly."
        ),
    )
    parser.add_argument(
        "--geo-compass-hazard-aware",
        action="store_true",
        help=(
            "Route the geo-compass around static hazards (RUN-18 variant). "
            "Requires --geo-compass; same 4 dims, no extra state change."
        ),
    )
    parser.add_argument(
        "--distributional-dqn",
        action="store_true",
        help=(
            "Use a C51-style distributional value head/loss while keeping expected "
            "Q-values for action selection. Changes checkpoint shape."
        ),
    )
    parser.add_argument(
        "--c51-atoms",
        type=int,
        default=51,
        help="Number of categorical support atoms for --distributional-dqn.",
    )
    parser.add_argument(
        "--c51-v-min",
        type=float,
        default=-20.0,
        help="Minimum categorical value support for --distributional-dqn.",
    )
    parser.add_argument(
        "--c51-v-max",
        type=float,
        default=120.0,
        help="Maximum categorical value support for --distributional-dqn.",
    )


def _add_reward_shaping_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geodesic-potential",
        action="store_true",
        help=(
            "Fold the telescoping BFS-geodesic closeness term into the PBRS potential "
            "and disable the additive per-step approach reward (PR #35 lever)."
        ),
    )
    parser.add_argument(
        "--geodesic-potential-weight",
        type=float,
        default=0.3,
        help="Weight of the geodesic closeness term when --geodesic-potential is enabled.",
    )
    parser.add_argument(
        "--show-locked-exit",
        action="store_true",
        help=(
            "Show the still-locked exit in the coarse global objective map at a "
            "distinct value (PR #35 lever)."
        ),
    )
    parser.add_argument(
        "--reverse-curriculum-p",
        type=float,
        default=0.0,
        help=(
            "Fraction of TRAINING resets that start mid-solution (PR #36 lever). "
            "0 disables the reverse curriculum; eval episodes are never affected."
        ),
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=None,
        help=(
            "Override the learn-time negative reward clamp (config REWARD_CLIP, "
            "default 5.0). 0 disables clamping so terminal penalties (death -12, "
            "timeout -8, stall -6) reach the learner at full magnitude."
        ),
    )


def _add_route_demo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--route-demo-levels",
        type=int,
        default=32,
        help="For route-demo-bc/tutorial-demo-bc/tutorial demo modes, number of levels to script for demonstrations.",
    )
    parser.add_argument(
        "--route-demo-max-steps",
        type=int,
        default=800,
        help="For route-demo-bc/tutorial-demo-bc/tutorial demo modes, max scripted controller steps per demonstration attempt.",
    )
    parser.add_argument(
        "--route-demo-variants",
        default="direct",
        help=(
            "Comma-separated route demo controller variants to try per level: "
            "direct,recovery,sweep,beam."
        ),
    )
    parser.add_argument(
        "--demo-selection-mode",
        default="all",
        choices=sorted(DEMO_SELECTION_MODES),
        help=(
            "For tutorial demo modes, choose all successful scripted trajectories or "
            "the B3r filtered-weighted demo selector."
        ),
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=6,
        help="For route-demo-bc/tutorial-demo-bc/tutorial demo modes, behavior-cloning epochs over successful scripted transitions.",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=128,
        help="For route-demo-bc/tutorial-demo-bc/tutorial demo modes, supervised action batch size.",
    )


def _add_demo_supervision_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--demo-action-weight",
        type=float,
        default=0.0,
        help="For tutorial-demo-dqfd/tutorial-demo-conservative/tutorial-demo-close-zone, online DQfD-style demo action loss weight.",
    )
    parser.add_argument(
        "--demo-action-margin",
        type=float,
        default=0.8,
        help="For tutorial-demo-dqfd/tutorial-demo-conservative/tutorial-demo-close-zone, Q margin required for demo actions.",
    )
    parser.add_argument(
        "--demo-action-batch-size",
        type=int,
        default=64,
        help="For tutorial-demo-dqfd/tutorial-demo-conservative/tutorial-demo-close-zone, demo states sampled per DQN update.",
    )
    parser.add_argument(
        "--demo-conservative-weight",
        type=float,
        default=0.0,
        help="For tutorial-demo-conservative modes, CQL-style weight on non-demo action Q-values.",
    )
    parser.add_argument(
        "--demo-conservative-temperature",
        type=float,
        default=1.0,
        help="For tutorial-demo-conservative modes, logsumexp temperature for conservative Q loss.",
    )
    parser.add_argument(
        "--close-zone-demo-distance",
        type=float,
        default=CLOSE_ZONE_DISTANCE_TILES,
        help="For tutorial-demo-close-zone, demo states within this many tiles of the objective are used for online action supervision.",
    )
    parser.add_argument(
        "--close-zone-demo-action-weight",
        type=float,
        default=0.12,
        help="For tutorial-demo-close-zone, supervised action-margin weight applied to close-zone demo states.",
    )
    parser.add_argument(
        "--close-zone-extra-action-weight",
        type=float,
        default=0.03,
        help="For tutorial-demo-conservative-close-zone, extra low-weight action-margin loss on close-zone demo states.",
    )
    parser.add_argument(
        "--close-zone-extra-label-source",
        default="scripted",
        choices=sorted(DEMO_CLOSE_ZONE_EXTRA_LABEL_SOURCES),
        help="For close-zone extra supervision, choose scripted or oracle-relabeled close-zone actions.",
    )
    parser.add_argument(
        "--oracle-close-zone-stride",
        type=int,
        default=4,
        help="For oracle close-zone labels, label every Nth close-zone state.",
    )
    parser.add_argument(
        "--oracle-close-zone-max-per-trajectory",
        type=int,
        default=8,
        help="For oracle close-zone labels, cap labels per successful trajectory; 0 means no cap.",
    )
    parser.add_argument(
        "--invalid-shoot-penalty",
        action="store_true",
        help="For tutorial-demo-bc, penalize shoot actions with no plausible firing-lane target.",
    )


def _add_eval_logging_trace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--eval-k", type=int, default=16)
    parser.add_argument(
        "--train-eval-games",
        type=int,
        default=8,
        help="Small held-out eval sample used during training checkpoints.",
    )
    parser.add_argument(
        "--selected-eval-games",
        type=int,
        default=0,
        help="For selected-checkpoint modes, evaluate the restored best checkpoint on this many held-out games after training.",
    )
    parser.add_argument(
        "--route-aux-weight",
        type=float,
        default=0.0,
        help="For first-crystal-direct, enable objective-direction auxiliary loss with this weight.",
    )
    parser.add_argument(
        "--route-aux-deadband",
        type=float,
        default=0.01,
        help="For first-crystal-direct route auxiliary labels, treat smaller normalized dx/dy as centered.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Run quick held-out eval every N training episodes; 0 disables it.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Flush trainer console progress every N completed episodes.",
    )
    parser.add_argument(
        "--report-seconds",
        type=float,
        default=20.0,
        help="Flush trainer console progress at least this often when episodes complete.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Write live_metrics.json/jsonl and print a monitor heartbeat every N seconds; 0 disables it.",
    )
    parser.add_argument(
        "--trace-eval-games",
        type=int,
        default=4,
        help="For diagnostic/selected-checkpoint modes, number of held-out games to trace after training.",
    )
    parser.add_argument(
        "--trace-max-steps",
        type=int,
        default=3000,
        help="For diagnostic/selected-checkpoint modes, max steps per traced held-out game.",
    )
    parser.add_argument(
        "--trace-sample-every",
        type=int,
        default=25,
        help="For diagnostic/selected-checkpoint modes, record a detailed Q/objective sample every N steps.",
    )
    parser.add_argument(
        "--trace-tail-steps",
        type=int,
        default=120,
        help="For diagnostic/selected-checkpoint modes, retain this many final compact steps per traced game.",
    )


def _add_interleave_bridge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interleave-drill-ratio",
        type=float,
        default=0.25,
        help="For interleaved mode, fraction of vector env lanes assigned to drills.",
    )
    parser.add_argument(
        "--interleave-drill-envs",
        type=int,
        default=None,
        help="For interleaved mode, exact drill lane count. Overrides --interleave-drill-ratio.",
    )
    parser.add_argument(
        "--drill-eval-max-steps",
        type=int,
        default=None,
        help="Optional per-drill greedy eval step cap; default uses config.EVAL_MAX_STEPS.",
    )
    parser.add_argument(
        "--bridge-eval-max-steps",
        type=int,
        default=None,
        help="Optional per-bridge greedy eval step cap; default uses config.EVAL_MAX_STEPS.",
    )
    parser.add_argument(
        "--bridge-eval-every",
        type=int,
        default=50,
        help="In bridge mode, run deterministic per-bridge eval every N episodes; 0 disables.",
    )
    parser.add_argument(
        "--interleave-bridge-ratio",
        type=float,
        default=0.125,
        help="For bridge-interleaved mode, fraction of vector env lanes assigned to bridges.",
    )
    parser.add_argument(
        "--interleave-bridge-envs",
        type=int,
        default=None,
        help="For bridge-interleaved mode, exact bridge lane count. Overrides --interleave-bridge-ratio.",
    )
    parser.add_argument(
        "--interleave-first-crystal-goal",
        action="store_true",
        help="For bridge-interleaved mode, use the first-crystal route objective on full tutorial lanes while keeping bridge lanes full-skill.",
    )
    parser.add_argument(
        "--interleave-contact-ratio",
        type=float,
        default=0.25,
        help="For contact-interleaved mode, fraction of vector env lanes assigned to contact levels.",
    )
    parser.add_argument(
        "--interleave-contact-envs",
        type=int,
        default=None,
        help="For contact-interleaved mode, exact contact lane count. Overrides --interleave-contact-ratio.",
    )
    parser.add_argument(
        "--contact-pool-size",
        type=int,
        default=0,
        help=(
            "For contact-interleaved mode, use a deterministic generated contact pool "
            "of this size instead of the five fixed contact levels."
        ),
    )
    parser.add_argument(
        "--contact-eval-pool-size",
        type=int,
        default=0,
        help=(
            "For contact-interleaved mode, run an extra held-out generated contact-pool "
            "eval of this size; final held-out tutorial eval is unchanged."
        ),
    )


def _add_reverse_archive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reverse-start-ratio",
        type=float,
        default=0.25,
        help="For reverse-start mode, fraction of vector env lanes reset near objectives.",
    )
    parser.add_argument(
        "--reverse-start-envs",
        type=int,
        default=None,
        help="For reverse-start mode, exact reverse-start lane count. Overrides --reverse-start-ratio.",
    )
    parser.add_argument(
        "--archive-start-ratio",
        type=float,
        default=0.25,
        help="For archive-start mode, fraction of vector env lanes reset from archived states.",
    )
    parser.add_argument(
        "--archive-start-envs",
        type=int,
        default=None,
        help="For archive-start mode, exact archive lane count. Overrides --archive-start-ratio.",
    )
    parser.add_argument(
        "--archive-replay-prob",
        type=float,
        default=0.7,
        help="For archive-start mode, probability an archive lane reset replays an archived state.",
    )
    parser.add_argument(
        "--archive-max-size",
        type=int,
        default=64,
        help="For archive-start mode, maximum number of archived state snapshots.",
    )
    parser.add_argument(
        "--archive-min-steps",
        type=int,
        default=30,
        help="For archive-start mode, minimum steps before a full-lane state can be archived.",
    )
    parser.add_argument(
        "--demo-repeat",
        type=int,
        default=4,
        help="For bridge-demo-replay/route-demo-bc/tutorial-demo-bc/tutorial-demo-dqfd/tutorial-demo-close-zone, repeat collected demonstration trajectories N times when seeding replay.",
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vec-envs", type=int, default=8)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out-dir", default=None)


def _add_checkpoint_correction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "For eval-checkpoint/eval-final-contact-option/contact-interleaved/"
            "collect-corrections/correction-finetune, "
            "path to a selected-weight checkpoint."
        ),
    )
    parser.add_argument(
        "--final-contact-distance",
        type=float,
        default=CLOSE_ZONE_DISTANCE_TILES,
        help=(
            "For eval-final-contact-option, switch from the NN policy to the local "
            "final-contact option inside this many target tiles."
        ),
    )
    parser.add_argument(
        "--final-contact-commit-steps",
        type=int,
        default=ROUTE_BEAM_COMMIT_STEPS,
        help=(
            "For eval-final-contact-option, number of oracle-planned close-zone "
            "actions to commit before replanning."
        ),
    )
    parser.add_argument(
        "--final-contact-cancel-outside",
        action="store_true",
        help=(
            "For eval-final-contact-option, cancel queued close-zone actions once "
            "the target leaves the close-zone distance."
        ),
    )
    parser.add_argument(
        "--final-contact-policy-advantage-gate",
        action="store_true",
        help=(
            "For eval-final-contact-option, only let the local option take over when "
            "its simulated macro beats the NN policy rollout by the configured margin."
        ),
    )
    parser.add_argument(
        "--final-contact-min-option-advantage",
        type=float,
        default=0.0,
        help=(
            "For --final-contact-policy-advantage-gate, minimum simulated score "
            "advantage required before the option overrides the NN policy."
        ),
    )
    parser.add_argument(
        "--correction-dataset",
        default=None,
        help=(
            "For correction-finetune/contact-head modes, path to a policy-visited "
            "correction .npz dataset."
        ),
    )
    parser.add_argument(
        "--correction-datasets",
        default=None,
        help=(
            "For contact-head-calibrate, comma-separated policy-visited correction "
            ".npz datasets to combine."
        ),
    )
    parser.add_argument(
        "--correction-games",
        type=int,
        default=8,
        help="For collect-corrections, held-out games to roll out for policy-visited labels.",
    )
    parser.add_argument(
        "--correction-max-steps",
        type=int,
        default=1200,
        help="For collect-corrections, max greedy policy steps per held-out game.",
    )
    parser.add_argument(
        "--correction-max-examples",
        type=int,
        default=512,
        help="For collect-corrections, cap saved correction states across all games.",
    )
    parser.add_argument(
        "--correction-sample-every",
        type=int,
        default=4,
        help="For collect-corrections, sample stale/loop correction candidates every N steps.",
    )
    parser.add_argument(
        "--correction-max-examples-per-game",
        type=int,
        default=64,
        help="For collect-corrections, cap saved correction states per held-out game.",
    )
    parser.add_argument(
        "--correction-stale-steps",
        type=int,
        default=90,
        help="For collect-corrections, label states after this many steps without progress.",
    )
    parser.add_argument(
        "--correction-loop-tile-visits",
        type=int,
        default=8,
        help="For collect-corrections, label states revisiting one tile at least this many times.",
    )
    parser.add_argument(
        "--correction-keep-agreements",
        action="store_true",
        help=(
            "For collect-corrections, keep states even when the policy already matches the "
            "correction label. Useful for smoke/debug dataset checks."
        ),
    )
    parser.add_argument(
        "--correction-action-weight",
        type=float,
        default=0.02,
        help="For correction-finetune, supervised correction action-margin loss weight.",
    )
    parser.add_argument(
        "--correction-action-margin",
        type=float,
        default=0.6,
        help="For correction-finetune, Q margin required for correction-label actions.",
    )
    parser.add_argument(
        "--correction-action-batch-size",
        type=int,
        default=64,
        help="For correction-finetune, correction states sampled per DQN update.",
    )
    parser.add_argument(
        "--policy-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "For correction-finetune, KL distillation weight that anchors the "
            "current policy to the restored checkpoint on replay states."
        ),
    )
    parser.add_argument(
        "--policy-anchor-temperature",
        type=float,
        default=1.0,
        help="For correction-finetune, softmax temperature for policy-anchor KL loss.",
    )
    parser.add_argument(
        "--policy-anchor-min-distance-tiles",
        type=float,
        default=0.0,
        help=(
            "For correction-finetune, anchor only replay states whose active target is "
            "at least this many tiles away. Use 0 for the old global anchor."
        ),
    )
    parser.add_argument(
        "--contact-action-weight",
        type=float,
        default=0.02,
        help="For contact-head-finetune, detached close-zone action-head loss weight.",
    )
    parser.add_argument(
        "--contact-action-batch-size",
        type=int,
        default=64,
        help="For contact-head-finetune, contact-label states sampled per DQN update.",
    )
    parser.add_argument(
        "--contact-action-distance",
        type=float,
        default=CLOSE_ZONE_DISTANCE_TILES,
        help="For contact-head-finetune, target-distance tiles where the learned head is active.",
    )
    parser.add_argument(
        "--contact-head-offline-steps",
        type=int,
        default=500,
        help="For contact-head-offline, supervised head-only optimization steps.",
    )
    parser.add_argument(
        "--contact-head-learning-rate",
        type=float,
        default=0.001,
        help="For contact-head-offline, Adam learning rate for only the detached head.",
    )
    parser.add_argument(
        "--contact-head-confidence",
        type=float,
        default=0.75,
        help="For contact-head-offline, minimum softmax confidence before overriding policy.",
    )
    parser.add_argument(
        "--contact-head-jump-confidence",
        type=float,
        default=0.0,
        help=(
            "For contact-head-offline, higher softmax confidence required before "
            "jump-variant head actions override policy; 0 uses the base threshold."
        ),
    )
    parser.add_argument(
        "--contact-head-action-thresholds",
        default="",
        help=(
            "For contact-head-offline, comma-separated ACTION:confidence gates "
            "that override only specific head actions, e.g. LEFT_JUMP:0.90."
        ),
    )
    parser.add_argument(
        "--contact-head-balance-classes",
        action="store_true",
        help="For contact-head-offline, sample balanced action classes during head training.",
    )
    parser.add_argument(
        "--contact-head-calibration-frac",
        type=float,
        default=0.25,
        help="For contact-head-calibrate, fraction of each action class held out for calibration.",
    )
    parser.add_argument(
        "--contact-head-calibration-seed",
        type=int,
        default=18,
        help="For contact-head-calibrate, RNG seed for the stratified calibration split.",
    )
    parser.add_argument(
        "--contact-head-min-calibration-accuracy",
        type=float,
        default=0.70,
        help="For contact-head-calibrate, minimum held-out label accuracy before the data audit passes.",
    )
    parser.add_argument(
        "--contact-head-min-class-examples",
        type=int,
        default=10,
        help="For contact-head-calibrate, minimum train examples required for each observed action class.",
    )
    parser.add_argument(
        "--contact-label-state-decimals",
        type=int,
        default=3,
        help="For contact-label-audit, decimals used when grouping duplicate state vectors.",
    )
    parser.add_argument(
        "--contact-label-adjacent-step-window",
        type=int,
        default=2,
        help="For contact-label-audit, adjacent sampled steps considered for label-flip checks.",
    )
    parser.add_argument(
        "--contact-label-top-groups",
        type=int,
        default=12,
        help="For contact-label-audit, max example groups to include per audit section.",
    )
    parser.add_argument(
        "--contact-label-filter-majority-threshold",
        type=float,
        default=0.67,
        help=(
            "For contact-label-filter, minimum semantic-bucket majority share "
            "required before keeping majority-label rows."
        ),
    )


def _add_artifact_persistence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Allow large .pth checkpoint writes. Off by default for low-disk sessions.",
    )
    parser.add_argument(
        "--save-selected-checkpoint",
        action="store_true",
        help=(
            "Save only the selected policy/target weights and metadata for later "
            "re-eval, without replay memory or full trainer state."
        ),
    )
    parser.add_argument(
        "--no-artifact-validation",
        action="store_true",
        help="Skip post-run validation of summary/report/live/selected-checkpoint artifacts.",
    )
