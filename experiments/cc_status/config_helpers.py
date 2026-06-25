# ruff: noqa: F401,F403,F405,I001
from .common import *


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def timestamp_id(label: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)
    return f"{stamp}_{clean}"


def base_config(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_every: int,
    train_eval_games: int,
    log_every: int,
    report_seconds: float,
) -> Config:
    config = Config()
    install_crystal_caves_experiment_defaults(config)
    config.GAME_NAME = "crystal_caves"
    config.MODEL_DIR = str(out_dir / "models")
    config.LOG_DIR = str(out_dir / "logs")
    config.MAX_EPISODES = episodes
    config.CRYSTAL_CAVES_SEED = seed
    config.USE_CNN_STATE = True
    config.EVAL_EVERY = eval_every
    config.EVAL_EPISODES = train_eval_games
    config.LOG_EVERY = log_every
    config.REPORT_INTERVAL_SECONDS = report_seconds
    config.EPSILON_WARMUP = 0
    config.EPSILON_DECAY = stage_epsilon_decay(
        TUTORIAL_MIN_EPSILON,
        episodes,
        max(config.EPSILON_END, TUTORIAL_MIN_EPSILON * TUTORIAL_EPSILON_END_FRACTION),
    )
    config.EARLY_STOP_ON_PLATEAU = False
    return config


def full_tutorial_config(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_every: int,
    train_eval_games: int,
    log_every: int,
    report_seconds: float,
) -> Config:
    config = base_config(
        out_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    config.CRYSTAL_CAVES_PROCEDURAL = True
    config.CRYSTAL_CAVES_DRILLS = False
    cc_experiment_config(config).CRYSTAL_CAVES_BRIDGES = False
    config.CRYSTAL_CAVES_DIFFICULTY = "tutorial"
    config.CRYSTAL_CAVES_FAMILIES = "platform_network"
    return config


def first_crystal_config(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_every: int,
    train_eval_games: int,
    log_every: int,
    report_seconds: float,
    difficulty: str = "tutorial",
) -> Config:
    config = full_tutorial_config(
        out_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    cc_experiment_config(config).CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL = True
    config.CRYSTAL_CAVES_DIFFICULTY = difficulty
    return config


def apply_cave_pool_override(config: Config, cave_pool_size: int | None) -> None:
    if cave_pool_size is not None:
        if cave_pool_size <= 0:
            raise ValueError("cave_pool_size must be positive")
        config.CRYSTAL_CAVES_POOL_SIZE = cave_pool_size


def apply_route_aux_override(
    config: Config,
    *,
    route_aux_weight: float,
    route_aux_deadband: float,
) -> None:
    if route_aux_weight < 0:
        raise ValueError("route_aux_weight must be non-negative")
    if route_aux_deadband < 0:
        raise ValueError("route_aux_deadband must be non-negative")
    if route_aux_weight > 0:
        exp_config = cc_experiment_config(config)
        exp_config.CRYSTAL_CAVES_ROUTE_AUX_LOSS = True
        exp_config.CRYSTAL_CAVES_ROUTE_AUX_WEIGHT = route_aux_weight
        exp_config.CRYSTAL_CAVES_ROUTE_AUX_DEADBAND = route_aux_deadband


def apply_demo_action_override(
    config: Config,
    *,
    demo_action_weight: float,
    demo_action_margin: float,
    demo_action_batch_size: int,
    demo_conservative_weight: float = 0.0,
    demo_conservative_temperature: float = 1.0,
) -> None:
    if demo_action_weight < 0:
        raise ValueError("demo_action_weight must be non-negative")
    if demo_action_margin < 0:
        raise ValueError("demo_action_margin must be non-negative")
    if demo_action_batch_size <= 0:
        raise ValueError("demo_action_batch_size must be positive")
    if demo_conservative_weight < 0:
        raise ValueError("demo_conservative_weight must be non-negative")
    if demo_conservative_temperature <= 0:
        raise ValueError("demo_conservative_temperature must be positive")
    if demo_action_weight > 0 or demo_conservative_weight > 0:
        exp_config = cc_experiment_config(config)
        exp_config.CRYSTAL_CAVES_DEMO_ACTION_LOSS = True
        exp_config.CRYSTAL_CAVES_DEMO_ACTION_WEIGHT = demo_action_weight
        exp_config.CRYSTAL_CAVES_DEMO_ACTION_MARGIN = demo_action_margin
        exp_config.CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE = demo_action_batch_size
        exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT = demo_conservative_weight
        exp_config.CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE = demo_conservative_temperature


def apply_close_zone_demo_action_override(
    config: Config,
    *,
    close_zone_demo_action_weight: float,
    close_zone_demo_action_batch_size: int,
) -> None:
    if close_zone_demo_action_weight < 0:
        raise ValueError("close_zone_demo_action_weight must be non-negative")
    if close_zone_demo_action_batch_size <= 0:
        raise ValueError("close_zone_demo_action_batch_size must be positive")
    if close_zone_demo_action_weight > 0:
        exp_config = cc_experiment_config(config)
        exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS = True
        exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT = close_zone_demo_action_weight
        exp_config.CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE = (
            close_zone_demo_action_batch_size
        )


def apply_correction_action_override(
    config: Config,
    *,
    correction_action_weight: float,
    correction_action_margin: float,
    correction_action_batch_size: int,
) -> None:
    if correction_action_weight < 0:
        raise ValueError("correction_action_weight must be non-negative")
    if correction_action_margin < 0:
        raise ValueError("correction_action_margin must be non-negative")
    if correction_action_batch_size <= 0:
        raise ValueError("correction_action_batch_size must be positive")
    if correction_action_weight > 0:
        exp_config = cc_experiment_config(config)
        exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_LOSS = True
        exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT = correction_action_weight
        exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN = correction_action_margin
        exp_config.CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE = correction_action_batch_size


def parse_route_demo_variants(raw: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not variants:
        raise ValueError("at least one route demo controller variant is required")
    unknown = set(variants) - ROUTE_DEMO_VARIANTS
    if unknown:
        raise ValueError(f"unknown route demo controller variants: {sorted(unknown)}")
    return variants


def parse_demo_selection_mode(raw: str) -> str:
    mode = raw.strip()
    if mode not in DEMO_SELECTION_MODES:
        raise ValueError(f"unknown demo selection mode: {mode!r}")
    return mode


def demo_action_arrays(
    trajectories: list[list[tuple[np.ndarray, int, float, np.ndarray, bool]]],
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(state, action) for trajectory in trajectories for state, action, *_ in trajectory]
    if not pairs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    states = np.stack([state for state, _ in pairs]).astype(np.float32, copy=False)
    actions = np.array([action for _, action in pairs], dtype=np.int64)
    return states, actions


def drill_config(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_every: int,
    train_eval_games: int,
    log_every: int,
    report_seconds: float,
) -> Config:
    config = base_config(
        out_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    config.CRYSTAL_CAVES_PROCEDURAL = False
    config.CRYSTAL_CAVES_DRILLS = True
    cc_experiment_config(config).CRYSTAL_CAVES_BRIDGES = False
    config.EVAL_EVERY = 0
    config.MAX_EPISODES = episodes
    return config


def bridge_config(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    eval_every: int,
    train_eval_games: int,
    log_every: int,
    report_seconds: float,
) -> Config:
    config = base_config(
        out_dir,
        episodes=episodes,
        seed=seed,
        eval_every=eval_every,
        train_eval_games=train_eval_games,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    config.CRYSTAL_CAVES_PROCEDURAL = False
    config.CRYSTAL_CAVES_DRILLS = False
    cc_experiment_config(config).CRYSTAL_CAVES_BRIDGES = True
    config.EVAL_EVERY = 0
    config.MAX_EPISODES = episodes
    return config


def trainer_args(*, episodes: int, vec_envs: int) -> argparse.Namespace:
    args = create_parser().parse_args(
        [
            "--headless",
            "--game",
            "crystal_caves",
            "--cnn",
            "--vec-envs",
            str(vec_envs),
            "--episodes",
            str(episodes),
        ]
    )
    args.vec_envs = vec_envs
    args.episodes = episodes
    args.model = None
    return args
