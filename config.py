"""
Configuration file for Neural Network Game AI
==============================================

All hyperparameters, game settings, and visualization options are centralized here.
Modify these values to experiment with different training configurations.

Usage:
    from config import Config
    cfg = Config()
    print(cfg.LEARNING_RATE)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class GameSettings:
    """Typed view of game and screen settings."""

    game_name: str
    screen_width: int
    screen_height: int
    fps: int
    lives: int


@dataclass(frozen=True)
class NetworkSettings:
    """Typed view of neural-network architecture settings."""

    action_size: int
    hidden_layers: Tuple[int, ...]
    activation: str
    use_dueling: bool
    use_noisy_networks: bool
    use_distributional_dqn: bool
    c51_num_atoms: int
    c51_v_min: float
    c51_v_max: float


@dataclass(frozen=True)
class TrainingSettings:
    """Typed view of core learning hyperparameters."""

    learning_rate: float
    gamma: float
    batch_size: int
    memory_size: int
    target_update: int
    learn_every: int
    gradient_steps: int


@dataclass(frozen=True)
class RuntimeSettings:
    """Typed view of runtime paths and execution limits."""

    model_dir: str
    game_model_dir: str
    log_dir: str
    device: torch.device
    max_episodes: int
    max_steps_per_episode: int


@dataclass
class Config:
    """
    Central configuration for the entire project.

    Sections:
    1. Game Settings - Breakout game parameters
    2. Neural Network - Architecture configuration
    3. Training - Learning hyperparameters
    4. Exploration - Epsilon-greedy settings
    5. Visualization - Display options
    6. System - Hardware and paths
    """

    # =========================================================================
    # GAME SELECTION
    # =========================================================================

    # Current game to play/train
    # Options: registered game IDs from src.game, e.g. 'breakout', 'crystal_caves'
    GAME_NAME: str = "breakout"

    # Crystal Caves: when True, the caves are procedurally generated instead of
    # the authored layouts. CRYSTAL_CAVES_SEED makes a procedural run reproducible.
    # CRYSTAL_CAVES_FAMILIES restricts which level families are used (comma-
    # separated, e.g. "platform_network,snake_bands"); empty = all families. This
    # is the knob for curriculum training (start easy, add families over stages).
    CRYSTAL_CAVES_PROCEDURAL: bool = False
    # Drill mode: use the hand-authored single-skill teaching levels instead of the
    # authored/procedural caves (for skill diagnostics and motor-skill pre-training).
    CRYSTAL_CAVES_DRILLS: bool = False
    # Curated mode: use the small hand-crafted Crystal-Caves-style level set instead of
    # the procedural generator. Every level is certified winnable by the physics-faithful
    # reachability oracle (experiments/cc_status/level_reach.py), so they sidestep the
    # generator's fairness problems. See src/game/crystal_caves_handcrafted_levels.py.
    CRYSTAL_CAVES_IMPORTED: bool = False
    # Human-play demo recording: when set (a directory path), --human mode saves every
    # finished episode as a replayable action-sequence JSON (see src/app/demo_recorder.py).
    RECORD_DEMOS_DIR: Optional[str] = None
    CRYSTAL_CAVES_SEED: int = 0
    CRYSTAL_CAVES_FAMILIES: str = ""
    # AI-1 rich state: a wider perception window (19x11 ~ the 1991 view) + a coarse
    # global objective map. When False, the legacy 11x9 window (119-feature state)
    # that reached ~8% wins. Toggleable so the two can be compared head-to-head.
    CRYSTAL_CAVES_RICH_STATE: bool = True
    # Geodesic next-step "corridor compass": append a few metadata scalars that point down
    # the actual traversable route toward the active objective (read from the cached BFS
    # distance field), instead of relying only on the euclidean target compass which points
    # straight at the objective THROUGH walls. RUN-11 showed the wall is long-range route-
    # to-exit navigation (FAR probe 0.12) precisely because the only directional signal is
    # that wall-blind euclidean compass. This is an OBSERVATION (computable identically at
    # eval), NOT a reward — so it does not re-trigger the disconfirmed geodesic-PBRS lever.
    # Off by default; an experiment lever. Adds GEO_COMPASS_FEATURES scalars to the state.
    CRYSTAL_CAVES_GEO_COMPASS: bool = False
    # Make the corridor compass route AROUND static hazards instead of straight through
    # them. The plain compass BFS treats hazard tiles as ordinary floor, so it steers the
    # agent into spikes — and the RUN-17 trusted death-trace shows hazards are the single
    # largest death source (~0.28 of episodes) and likely drive much of the stall mass
    # (the agent freezes when the only suggested step is into a hazard). This re-weights a
    # SEPARATE compass-only field so hazard tiles cost HAZARD_COST extra steps but stay
    # passable (the objective never becomes unreachable); the PBRS shaping field is left
    # untouched. Same 4 compass dims, so no network-shape change. Requires GEO_COMPASS.
    CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE: bool = False
    # Extra route cost (in tiles) charged for stepping onto a hazard tile in the hazard-
    # aware compass field. Higher = wider detours around hazards; must stay finite so a
    # hazard-only corridor remains routable rather than infinitely avoided.
    CRYSTAL_CAVES_GEO_COMPASS_HAZARD_COST: float = 8.0
    # Enemy-motion perception (RUN-22 survival lever): append per-enemy metadata for the
    # nearest few enemies VISIBLE INSIDE the perception window — [present, dx, dy, vx,
    # is_flyer] each. A single-frame tile window cannot distinguish an enemy moving toward
    # vs away from the player (the killed-dominant failure mode of RUN-20/21), and a human
    # player watching the screen DOES see motion — so this is fair perception, not an
    # oracle: enemies outside the window contribute nothing. Off by default; A/B lever.
    CRYSTAL_CAVES_ENEMY_MOTION: bool = False
    # Use a convolutional Q-network that reads the perception window as a 2D grid
    # (the right architecture for the spatial rich state). Requires the game to set
    # config.STATE_LAYOUT. Off by default; the MLP path keeps the live NN visualizer.
    USE_CNN_STATE: bool = False
    # Spatial layout of the state vector (window/gmap/meta dims), published by the
    # game at runtime so a CNN can reshape the flat state. None for non-spatial games.
    STATE_LAYOUT: Optional[dict] = None
    # SpatialDQN: global-average-pool the conv feature map instead of flattening it.
    # Flatten preserves absolute tile position (memorizes layouts); GAP is translation-
    # invariant and is the standard ProcGen fix for the train-solves/test-fails gap.
    # Off by default (keeps current behavior); a generalization experiment lever.
    CRYSTAL_CAVES_CNN_GLOBAL_POOL: bool = False
    # Objective/threat budget for generated caves: "easy" is a learnable
    # curriculum floor (2-3 crystals, no hazards/enemies); "normal" is the full
    # game (10-14 crystals + hazards + enemies).
    CRYSTAL_CAVES_DIFFICULTY: str = "normal"
    # Procedural diversity. Previously procedural mode generated only 4 caves and
    # reset() always reloaded CAVES[level_index] (which only advances on a win), so
    # the agent trained+evaluated on essentially ONE fixed level — it memorised it
    # instead of generalising. Now training samples a random cave from a pool of
    # POOL_SIZE distinct caves each episode, while evaluation uses a fixed, disjoint
    # HELD-OUT set (one cave per eval game → reproducible, and unseen → a true
    # generalisation measure). 0 restores the legacy single-set behaviour.
    CRYSTAL_CAVES_POOL_SIZE: int = 64
    # Fold a telescoping geodesic "closeness to the current objective" term into the
    # potential-based shaping (PBRS), replacing the farmable additive per-step approach
    # reward. Because it is potential-based (terminal Phi=0, telescopes to 0 over a full
    # episode) it is policy-invariant — it only changes how fast the policy is found,
    # not which policy is optimal. Off by default: this is an experiment lever to A/B on
    # the chain-progress surrogate, not a silent default change.
    CRYSTAL_CAVES_GEODESIC_POTENTIAL: bool = False
    # Weight of the geodesic closeness term in the potential, relative to completion
    # progress (whose component weights sum to 1.0). Small enough that collecting an
    # objective always nets a positive shaped step despite the target-switch dip.
    CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT: float = 0.3
    # Apply the geodesic route-shaping ONLY after the exit unlocks (i.e. only for
    # leg 2: route-to-exit), leaving leg 1 (find+collect) on its normal, working
    # approach reward. Targets the collect->exit conversion wall without the
    # learnability hit dense geodesic caused when it also shaped pre-collection.
    CRYSTAL_CAVES_GEODESIC_AFTER_UNLOCK: bool = False
    # Show the still-locked exit in the coarse global objective map (at a distinct,
    # lower value than the unlocked exit) so the agent can learn the route to it before
    # the last crystal is collected, instead of the exit appearing only at unlock.
    CRYSTAL_CAVES_SHOW_LOCKED_EXIT: bool = False
    # No-progress stall window (steps) for Crystal Caves; 0 keeps the game's built-in
    # default (720). RUN-26 fidelity lever: DATA-1 showed the stall clock owns a large,
    # learning-invariant share of endings, so arms may widen it (e.g. 1440) explicitly.
    CRYSTAL_CAVES_STALL_WINDOW_STEPS: int = 0
    # Episode step cap override for Crystal Caves; 0 keeps the game's built-in
    # default (3000). The 1991 original has NO level timer — the cap is a
    # training-harness artifact, and the level-validity audit (PR #39) shows
    # perfect tours already use 0.55-0.92 of it. Opt-in fidelity lever.
    CRYSTAL_CAVES_MAX_STEPS_OVERRIDE: int = 0
    # Reverse curriculum: begin a fraction of TRAINING episodes from a valid
    # mid-solution state (a subset of crystals pre-collected, gates opened) so the
    # agent gets dense reps of finishing the collect->...->exit chain rather than only
    # ever seeing it from scratch. Solvability-preserving; never applied during eval.
    # Off by default — an experiment lever to A/B on the chain-progress surrogate.
    CRYSTAL_CAVES_REVERSE_CURRICULUM: bool = False
    # Fraction of training resets that use a mid-solution start (in [0, 1]). A trainer
    # may anneal this toward 0 via CrystalCaves.set_reverse_curriculum_p().
    CRYSTAL_CAVES_REVERSE_CURRICULUM_P: float = 0.5
    # Linear anneal of the reverse-curriculum probability down to 0.0 over the first
    # N training episodes (then held at 0). A fixed p hurt held-out performance because
    # half of training never saw the full-from-spawn task; annealing p -> 0 lets the
    # policy finish on full-length episodes. 0 = no annealing (constant p, the legacy
    # behaviour). Only takes effect when CRYSTAL_CAVES_REVERSE_CURRICULUM is on.
    CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES: int = 0
    # Reverse-curriculum follow-up: also RELOCATE the player toward the remaining
    # objectives on a mid-solution start, shortening the navigation horizon. The
    # destination is verified with the jump-aware reachability oracle so every
    # relocated start can still reach all remaining objectives + the exit (falls back
    # to the spawn if no verified tile is found). Requires REVERSE_CURRICULUM on.
    # Separate flag so it can be A/B'd independently of the base reverse curriculum.
    CRYSTAL_CAVES_REVERSE_CURRICULUM_RELOCATE: bool = False
    # Reverse-EXIT curriculum (leg-2 practice): on a fraction of TRAINING resets, begin
    # the episode already in the post-collection state — all crystals collected, every
    # gate open, exit unlocked — with the player dropped on a safe standing tile near the
    # open exit (jump-aware oracle-verified reachable). This gives dense, isolated reps of
    # the documented WALL: the collect->exit conversion / route-to-exit skill the leg-2
    # probe measured at ~0.5 held-out. Distinct from the base reverse curriculum (which
    # keeps SOME crystals); this clears ALL of them to drill only the final exit hop.
    # Solvability-preserving (a subset/empty objective set with all doors open is always
    # reachable from a spawn that could clear the full level); never applied during eval.
    CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM: bool = False
    # Fraction of training resets that use the reverse-exit (near-open-exit) start, [0, 1].
    CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P: float = 0.5
    # FAR variant: instead of hugging the exit (which only drills the trivial final hop the
    # agent already aces, ~0.73 held-out), drop the player on a random reachable tile a real
    # distance from the exit. RUN-11 showed the actual wall is long-range route-to-exit
    # navigation (FAR probe ~0.12), so this drills the genuinely-missing skill. Requires
    # CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM on; a separate flag so NEAR vs FAR can be A/B'd.
    CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_FAR: bool = False
    # NGU-style episodic novelty bonus: a small per-step intrinsic reward for reaching
    # a (tile_x, tile_y, crystals_remaining, switches_used) cell not yet seen THIS
    # episode, decaying as 1/sqrt(visits). Attacks the "stops reaching new cells ->
    # times out" failure. Off by default; an experiment lever (flows through n-step).
    # Reward calibration (RUN-23): death/hit penalties were hardcoded (-12 / -3) and
    # tuned for the ~10-14-crystal generated levels. The hand-crafted set holds 30-34
    # crystals (+150+ potential), so death's RELATIVE cost collapsed — dying costs less
    # than three crystals, making reckless play rational. Knobs so the scale can be A/B'd.
    CRYSTAL_CAVES_DEATH_PENALTY: float = -12.0
    CRYSTAL_CAVES_HIT_PENALTY: float = -3.0
    CRYSTAL_CAVES_NGU_BONUS: bool = False
    CRYSTAL_CAVES_NGU_BETA: float = 0.02
    # --- Demonstration learning (DQfD-lite + demo-prefix starts, RUN-26) ---
    # Directory of verified winning demo JSONs (human recorder or planner output).
    # When set, the training harness builds a never-overwritten demo buffer and every
    # gradient step adds a demo minibatch loss: n-step TD + the DQfD large-margin term.
    DEMO_DIR: Optional[str] = None
    DEMO_BATCH_FRACTION: float = 0.125  # demo minibatch size as a fraction of BATCH_SIZE
    DEMO_MARGIN: float = 0.8  # margin by which the demonstrated action must beat others
    DEMO_MARGIN_WEIGHT: float = 1.0
    DEMO_TD_WEIGHT: float = 1.0
    DEMO_PRETRAIN_STEPS: int = 0  # demo-only gradient steps before env interaction
    # Opening-focused imitation (phase 2): keep only the first N steps of each demo
    # route in the demo store, so the margin loss becomes a pure prior on the route
    # OPENING — the segment the backward ladder's win-banking never reaches (0 = all).
    DEMO_OPENING_ONLY_STEPS: int = 0
    # Linear decay of the margin weight to zero over this many GLOBAL episodes
    # (0 = constant). RUN-62 showed constant weight is an early accelerant but a
    # deep-regime anchor; decay keeps the first phase and removes the second.
    DEMO_MARGIN_DECAY_EPISODES: int = 0
    # Re-ignition (RUN-63 insight): from this GLOBAL episode on, the margin scale
    # floors at DEMO_MARGIN_REIGNITE_SCALE — for when the backward ladder's
    # frontier reaches the demo-covered opening, where episode starts and demo
    # states finally overlap and imitation aligns instead of anchoring (0 = off).
    DEMO_MARGIN_REIGNITE_EPISODE: int = 0
    DEMO_MARGIN_REIGNITE_SCALE: float = 0.5
    # Backward curriculum: probability a TRAINING episode starts mid-route by replaying
    # a random 10-85% prefix of a winning demo (imported set only; eval unaffected).
    CRYSTAL_CAVES_DEMO_RESET_P: float = 0.0
    # Backward demo curriculum (Salimans & Chen / Go-Explore phase 2): instead of
    # random 10-85% prefix cuts, start DEMO_BACKWARD_START_OFFSET steps before the
    # demo's win and retreat the start point only as the agent banks wins at each
    # rung. Requires DEMO_DIR + CRYSTAL_CAVES_DEMO_RESET_P > 0.
    CRYSTAL_CAVES_DEMO_BACKWARD: bool = False
    # Ladder pace overrides (0 = game-class defaults: retreat 40 steps / 3 wins).
    CRYSTAL_CAVES_DEMO_BACKWARD_RETREAT: int = 0
    CRYSTAL_CAVES_DEMO_BACKWARD_WINS: int = 0
    # Probability a TRAINING episode resamples its level uniformly among DEMOED
    # levels (0 = uniform over all levels). Level sampling dominates backward-
    # ladder throughput; bias concentrates rung attempts. Eval unaffected.
    CRYSTAL_CAVES_DEMO_LEVEL_BIAS: float = 0.0
    # Windowed backward starts: sample the start offset uniformly from
    # [frontier - WINDOW, frontier] so deep rungs keep a learning signal; only
    # exact-frontier attempts bank rung credit. 0 = frontier-only starts.
    CRYSTAL_CAVES_DEMO_BACKWARD_WINDOW: int = 0
    # Deep-rung easing threshold (steps-from-win): past it a rung costs 1 win
    # and retreats half-steps. 0 = off.
    CRYSTAL_CAVES_DEMO_BACKWARD_DEEP: int = 0
    # Restore full health when a demo-prefix start hands control to the agent
    # (training only). Corrects the pessimistic HP-1 bias of tank-and-grab
    # harvester routes.
    CRYSTAL_CAVES_DEMO_HEAL_ON_HANDOFF: bool = False
    # Win-at-K training tier (RUN-25): during TRAINING the exit opens once K crystals
    # are held (0 = off, real all-crystals rule). A curriculum on the task definition —
    # the agent practices the full collect->route->exit chain thousands of times before
    # a 30-crystal clear is within reach, instead of never reaching the endgame at all.
    # Eval always keeps the real win rule, so reported win rates stay canonical.
    CRYSTAL_CAVES_WIN_AT_K: int = 0
    # Ramp for the win-at-K tier (per-game-instance episodes; 0 = static K). K climbs
    # linearly from CRYSTAL_CAVES_WIN_AT_K to the level's full crystal count across
    # this many episodes, merging the training tier into the real all-crystals rule —
    # avoids overfitting a "grab K then leave" policy that eval would punish.
    CRYSTAL_CAVES_WIN_AT_K_RAMP_EPISODES: int = 0
    # Hold K at the floor for this many per-instance episodes before the ramp starts
    # (0 = ramp immediately). The agent needs a consolidation phase of dense wins
    # before the bar moves, or the ramp outruns it and wins never begin.
    CRYSTAL_CAVES_WIN_AT_K_RAMP_DELAY: int = 0
    # Truncation-aware bootstrapping (Pardo et al. 2018, "Time Limits in RL").
    # When an episode ends only because it hit a time/no-progress cutoff ("timeout"
    # or "stalled") rather than a real environment terminal ("won"/"killed"), the
    # stored transition is marked NOT done so the TD target still bootstraps the
    # value of the final state. The episode is still reset normally. This stops the
    # agent from learning that "the clock running out" is a real terminal worth
    # value 0, which otherwise drags Q-values down. Off by default; an A/B lever.
    # Only affects the vectorized training path (the one Crystal Caves uses).
    CRYSTAL_CAVES_TRUNCATION_BOOTSTRAP: bool = False
    # Infinite-levels training (ProcGen's dominant generalization lever): when on,
    # each TRAINING reset generates a fresh procedural cave (seed offset 1_000_000+,
    # disjoint from the fixed pool at offset 0 and the held-out eval block at offset
    # 500000) instead of sampling the fixed pool. Eval is unaffected. Off by default.
    CRYSTAL_CAVES_REGENERATE_EACH_EPISODE: bool = False
    # Drop level-identity / absolute-position features from the observation that let
    # the agent MEMORISE rather than generalise: zeroes the level_index slot (a pure
    # train-set ID) and the absolute player_x/player_y slots, keeping the egocentric
    # window + target compass. Shape-preserving (zeroed, not removed). Off by default.
    CRYSTAL_CAVES_DROP_LEAK_FEATURES: bool = False

    # =========================================================================
    # SCREEN SETTINGS
    # =========================================================================

    # Screen dimensions (shared across all games)
    SCREEN_WIDTH: int = 800
    SCREEN_HEIGHT: int = 600

    # =========================================================================
    # BREAKOUT SETTINGS
    # =========================================================================

    # Breakout-specific
    PADDLE_WIDTH: int = 100
    PADDLE_HEIGHT: int = 15
    PADDLE_SPEED: int = 8

    BALL_RADIUS: int = 8
    BALL_SPEED: int = 6

    BRICK_ROWS: int = 5
    BRICK_COLS: int = 10
    BRICK_WIDTH: int = 70
    BRICK_HEIGHT: int = 25
    BRICK_PADDING: int = 5
    BRICK_OFFSET_TOP: int = 60
    BRICK_OFFSET_LEFT: int = 35

    # Game mechanics
    LIVES: int = 3
    FPS: int = 60

    # =========================================================================
    # SPACE INVADERS SETTINGS
    # =========================================================================

    # Grid of aliens (classic: 5 rows x 11 columns = 55 aliens)
    SI_ALIEN_ROWS: int = 5
    SI_ALIEN_COLS: int = 11
    SI_ALIEN_WIDTH: int = 36
    SI_ALIEN_HEIGHT: int = 26
    SI_ALIEN_PADDING: int = 12
    SI_ALIEN_OFFSET_TOP: int = 100  # Start aliens higher for more play space
    SI_ALIEN_OFFSET_LEFT: int = 70

    # Alien movement - tuned for AI learning with slower initial speed
    SI_ALIEN_SPEED_X: float = 0.8  # Slower initial speed (was 2.0)
    SI_ALIEN_SPEED_Y: int = 10  # Smaller drops (was 20) - gives AI more time
    SI_ALIEN_SPEED_MULTIPLIER: float = 1.03  # Gradual speed increase

    # Player ship
    SI_SHIP_WIDTH: int = 50
    SI_SHIP_HEIGHT: int = 30
    SI_SHIP_SPEED: int = 7  # Slightly faster for responsive control
    SI_SHIP_Y_OFFSET: int = 80  # More space from bottom for base visual

    # Bullets
    SI_BULLET_WIDTH: int = 4
    SI_BULLET_HEIGHT: int = 15
    SI_BULLET_SPEED: int = 12  # Faster bullets
    SI_MAX_PLAYER_BULLETS: int = 2  # Limited to 2 like original
    SI_ALIEN_SHOOT_CHANCE: float = 0.001  # Reduced from 0.002 - less spam
    SI_ALIEN_BULLET_SPEED: int = 4  # Slower alien bullets for fairness

    # UFO bonus
    SI_UFO_CHANCE: float = 0.0008  # Slightly rarer
    SI_UFO_SPEED: int = 3
    SI_UFO_POINTS: int = 100  # 50-300 random in original, we use fixed

    # Shields/bunkers (classic Space Invaders defense)
    # Set to False to disable bunkers for simpler gameplay/training
    SI_SHIELDS_ENABLED: bool = False
    SI_SHIELD_COUNT: int = 4
    SI_SHIELD_WIDTH: int = 50
    SI_SHIELD_HEIGHT: int = 35

    # Space Invaders rewards - tuned for AI training
    SI_REWARD_ALIEN_HIT: float = 1.0  # Per alien killed
    SI_REWARD_UFO_HIT: float = 5.0  # UFO bonus
    SI_REWARD_PLAYER_DEATH: float = -2.5  # Death penalty (3 lives, so moderate penalty)
    SI_REWARD_LEVEL_CLEAR: float = 0.0  # Removed - score-based rewards only

    # Anti-passive rewards - CRITICAL for learning aggressive play
    # Without these, model learns to stand still (death penalty >> step penalty)
    SI_REWARD_STEP: float = -0.01  # 10x stronger time penalty (was -0.001)
    SI_REWARD_SHOOT: float = 0.02  # Small reward for shooting (encourages aggression)
    SI_REWARD_STAY: float = -0.02  # Penalty for doing nothing (action=1)

    # Win condition: Number of levels/waves to complete to win (0 = endless mode, no wins)
    SI_WIN_LEVELS: int = 10  # Complete 10 levels to win

    # Space Invaders colors (CRT phosphor aesthetic)
    SI_COLOR_BACKGROUND: Tuple[int, int, int] = (0, 5, 0)  # Near black with green tint
    SI_COLOR_SHIP: Tuple[int, int, int] = (0, 255, 100)  # Bright green
    SI_COLOR_BULLET: Tuple[int, int, int] = (0, 255, 200)  # Cyan
    SI_COLOR_ALIEN_1: Tuple[int, int, int] = (255, 60, 100)  # Pink/magenta (top rows)
    SI_COLOR_ALIEN_2: Tuple[int, int, int] = (100, 255, 100)  # Green (middle)
    SI_COLOR_ALIEN_3: Tuple[int, int, int] = (100, 200, 255)  # Cyan (bottom)
    SI_COLOR_UFO: Tuple[int, int, int] = (255, 50, 50)  # Red
    SI_COLOR_SHIELD: Tuple[int, int, int] = (0, 220, 80)  # Bright green bunkers

    # Curriculum learning (optional - disabled by default)
    # When enabled, starts with easier game settings and gradually increases difficulty
    SI_CURRICULUM_ENABLED: bool = False

    # Curriculum stages: each stage defines game parameters and episode count
    # Format: {'alien_rows': int, 'alien_shoot_chance': float, 'episodes': int or None}
    # None for episodes means "continue indefinitely at this difficulty"
    SI_CURRICULUM_STAGES: List[dict] = field(
        default_factory=lambda: [
            {"alien_rows": 2, "alien_shoot_chance": 0.0005, "episodes": 500},
            {"alien_rows": 3, "alien_shoot_chance": 0.0008, "episodes": 500},
            {"alien_rows": 4, "alien_shoot_chance": 0.001, "episodes": 500},
            {
                "alien_rows": 5,
                "alien_shoot_chance": 0.001,
                "episodes": None,
            },  # Full game
        ]
    )

    # =========================================================================
    # NEURAL NETWORK ARCHITECTURE
    # =========================================================================

    # Input size is calculated based on game state:
    # - Ball position (x, y) = 2
    # - Ball velocity (dx, dy) = 2
    # - Paddle position (x) = 1
    # - Brick states = BRICK_ROWS * BRICK_COLS

    @property
    def STATE_SIZE(self) -> int:
        """Calculate input layer size based on game state representation.

        NOTE: This is a legacy property used only for standalone tests.
        In production, use game.state_size instead, which is calculated
        dynamically by each game class.
        """
        if self.GAME_NAME == "breakout":
            ball_info = 4  # x, y, dx, dy
            paddle_info = 1  # x position
            tracking_info = 3  # relative_x, predicted_landing, distance_to_target
            brick_info = self.BRICK_ROWS * self.BRICK_COLS  # binary brick states
            return ball_info + paddle_info + tracking_info + brick_info
        elif self.GAME_NAME == "space_invaders":
            # Space Invaders has dynamic state size based on aliens/bullets
            # This is an approximation - use game.state_size in production
            max_player_bullets = 3
            num_aliens = 55  # 5 rows * 11 cols
            return 1 + max_player_bullets * 2 + num_aliens + 5 + 7
        elif self.GAME_NAME == "crystal_caves":
            # Local 11x9 tile window plus 20 metadata features.
            return 11 * 9 + 20
        else:
            # Default fallback
            return 128

    # Action space
    ACTION_SIZE: int = 3  # LEFT, STAY, RIGHT

    # Hidden layer architecture
    # More neurons = more capacity but slower training
    # [512, 512, 256, 128] - 4 layers, slightly more capacity than original
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [512, 512, 256, 128])

    # Activation function: 'relu', 'leaky_relu', 'tanh'
    ACTIVATION: str = "relu"

    # Use Dueling DQN architecture (separates value and advantage streams)
    # This helps the network learn which states are valuable independent of actions
    USE_DUELING: bool = True

    # Distributional DQN / C51 probe. Disabled by default because it changes the
    # output head and checkpoint shape. When enabled, networks still return
    # expected Q-values from forward(), while the agent trains the categorical
    # value distribution through explicit distributional methods.
    USE_DISTRIBUTIONAL_DQN: bool = False
    C51_NUM_ATOMS: int = 51
    C51_V_MIN: float = -20.0
    C51_V_MAX: float = 120.0

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================

    # Learning rate - How big of steps to take during optimization
    # Too high: unstable training, loss explodes
    # Too low: very slow learning
    # Typical range: 0.0001 to 0.001
    LEARNING_RATE: float = 0.0001

    # L2 weight decay (Adam). 0 = off (default). A small value (e.g. 1e-4) is a
    # standard regularizer against memorization in procedurally generated tasks;
    # exposed as an experiment lever for the Crystal Caves generalization work.
    WEIGHT_DECAY: float = 0.0

    # Discount factor (gamma) - How much to value future rewards
    # 0.99 = far-sighted, considers distant future
    # 0.90 = more short-sighted, prefers immediate rewards
    # 0.99 = more far-sighted for Space Invaders long-term planning
    GAMMA: float = 0.99

    # Batch size - Number of experiences to sample per training step
    # Larger = more stable gradients but slower per step
    # M4 CPU optimal: 128 (balances throughput and stability)
    BATCH_SIZE: int = 128

    # Replay buffer capacity
    # Larger = more diverse experiences but more memory
    # 500k experiences ≈ 40MB - retains ~5x more history
    MEMORY_SIZE: int = 500_000

    # Minimum experiences before training starts
    MEMORY_MIN: int = 1000

    # Target network update frequency (in steps) - used for hard updates
    # How often to sync target network with policy network
    TARGET_UPDATE: int = 1000

    # Soft target update coefficient (TAU)
    # If > 0, uses soft updates instead of hard updates
    # target = TAU * policy + (1 - TAU) * target
    # Typical values: 0.001 to 0.01
    # 0.005 provides faster learning propagation while maintaining stability
    TARGET_TAU: float = 0.005

    # Use soft updates instead of hard updates
    USE_SOFT_UPDATE: bool = True

    # Gradient clipping to prevent exploding gradients
    GRAD_CLIP: float = 1.0

    # =========================================================================
    # PERFORMANCE OPTIMIZATION
    # =========================================================================
    #
    # M4 MacBook Benchmark Results (headless training):
    #   CPU B=128, LE=8, GS=2:  ~5,000 steps/sec, 663 grad/sec (balanced)
    #   CPU B=128, LE=16, GS=4: ~2,900 steps/sec, 719 grad/sec (max learning)
    #   MPS B=256, LE=4:        ~640 steps/sec (GPU overhead dominates)
    #
    # CONCLUSION: Use CPU for small models - MPS transfer overhead is too high.
    # =========================================================================

    # Learn every N steps (1 = every step, higher = faster but less frequent learning)
    # Lower value = more frequent updates for faster learning
    LEARN_EVERY: int = 4

    # Number of gradient updates per learning call
    # Compensates for LEARN_EVERY > 1 to maintain learning throughput
    # Rule of thumb: GRADIENT_STEPS = LEARN_EVERY / 2 (for similar grad/sec)
    GRADIENT_STEPS: int = 2

    # Use torch.compile() for potential speedup (PyTorch 2.0+)
    # Note: Minimal benefit on CPU for small models, can cause overhead
    USE_TORCH_COMPILE: bool = False  # Disabled - minimal benefit for this model size

    # Compile mode: 'default', 'reduce-overhead', 'max-autotune'
    TORCH_COMPILE_MODE: str = "reduce-overhead"

    # Use mixed precision (float16) for faster computation on GPU/MPS
    # Only beneficial on GPU - CPU uses float32 regardless
    USE_MIXED_PRECISION: bool = False  # Disabled - using CPU by default

    # Force CPU device (faster than MPS for small models on M4)
    # Set via --cpu flag or environment variable
    FORCE_CPU: bool = False

    # =========================================================================
    # EXPLORATION SETTINGS (Epsilon-Greedy)
    # =========================================================================

    # Starting exploration rate (1.0 = 100% random)
    # With NoisyNets: small epsilon as fallback exploration
    EPSILON_START: float = 0.1

    # Minimum exploration rate
    EPSILON_END: float = 0.02

    # Decay rate per episode (higher = slower decay)
    EPSILON_DECAY: float = 0.9995

    # Exploration decay strategy: 'exponential', 'linear', 'cosine'
    EXPLORATION_STRATEGY: str = "exponential"

    # Warmup episodes before epsilon starts decaying
    # Allows buffer to fill with diverse experiences
    EPSILON_WARMUP: int = 200

    # =========================================================================
    # PRIORITIZED EXPERIENCE REPLAY
    # =========================================================================

    # Enable prioritized replay (samples important experiences more often)
    # Improves learning efficiency by 30-40% at cost of ~10% speed overhead
    # Enabled for Space Invaders improvements
    USE_PRIORITIZED_REPLAY: bool = True

    # Priority exponent (0 = uniform sampling, 1 = full prioritization)
    PER_ALPHA: float = 0.6

    # Importance sampling start (anneals to 1.0 over PER_BETA_FRAMES)
    PER_BETA_START: float = 0.4

    # Number of frames over which to anneal beta from start to 1.0
    # Slower annealing for more stable learning
    PER_BETA_FRAMES: int = 200000

    # =========================================================================
    # LEARNING RATE SCHEDULING
    # =========================================================================

    # Enable learning rate scheduler (reduces LR over time for fine-tuning)
    # Disabled by default - enable for long training runs (5000+ episodes)
    USE_LR_SCHEDULER: bool = False

    # Scheduler type: 'cosine' (smooth decay) or 'step' (periodic drops)
    LR_SCHEDULER_TYPE: str = "cosine"

    # For step scheduler: decay LR every N episodes
    LR_SCHEDULER_STEP: int = 500

    # For step scheduler: multiply LR by this factor
    LR_SCHEDULER_GAMMA: float = 0.5

    # Minimum learning rate (floor for schedulers)
    LR_MIN: float = 1e-5

    # Explicit, horizon-matched LR decay driven by the trainer (cosine from
    # LEARNING_RATE down to LR_MIN over the run's episodes). Unlike USE_LR_SCHEDULER
    # (whose T_max is fixed at 2000), this completes over THIS run's episode count,
    # so a 600-episode run actually freezes the policy near its peak by the end —
    # the fix for late-training Q-drift / win-rate volatility. Off by default.
    LR_DECAY: bool = False

    # =========================================================================
    # N-STEP RETURNS
    # =========================================================================

    # Enable N-step returns for faster reward propagation
    # Trades off bias vs variance in value estimation
    USE_N_STEP_RETURNS: bool = True

    # Number of steps to look ahead for n-step returns.
    # Raised from 3 to 6: Crystal Caves wins require a long switch->crystals->exit
    # sequence (40+ steps), and 3-step returns can't propagate the terminal reward
    # back far enough — the agent's Q-values stayed pinned near 0. 6 carries the
    # signal twice as far without the variance of a very long horizon.
    N_STEP_SIZE: int = 6

    # =========================================================================
    # NOISY NETWORKS
    # =========================================================================

    # Enable NoisyNet (learnable parameter noise for exploration)
    # NoisyNets + hybrid epsilon-greedy (Exp 2 showed best stability)
    USE_NOISY_NETWORKS: bool = True

    # Standard deviation for noise initialization
    # Higher values = more initial exploration (0.5 standard, 0.7 for more exploration)
    NOISY_STD_INIT: float = 0.5

    # =========================================================================
    # REWARD SHAPING
    # =========================================================================

    # Rewards for different events (tuned for stable learning)
    REWARD_BRICK_HIT: float = 2.0  # Breaking a brick - primary positive signal
    REWARD_GAME_OVER: float = -5.0  # Losing a life - moderate negative to avoid risk aversion
    REWARD_WIN: float = 100.0  # Clearing all bricks - strong completion incentive
    REWARD_PADDLE_HIT: float = 0.2  # Ball hitting paddle - encourages survival
    REWARD_STEP: float = 0.0  # Per-step reward (can set negative for urgency)

    # Dense reward shaping for ball tracking
    REWARD_TRACKING_GOOD: float = 0.01  # Reward for moving toward predicted ball landing
    REWARD_TRACKING_BAD: float = -0.01  # Penalty for moving away from predicted landing

    # Reward clipping to prevent extreme gradients during training
    # Set to 0 to disable clipping
    # Note: Only clips negative rewards to preserve win bonus signal
    REWARD_CLIP: float = 5.0

    # =========================================================================
    # VISUALIZATION SETTINGS
    # =========================================================================

    # Colors (RGB tuples)
    COLOR_BACKGROUND: Tuple[int, int, int] = (15, 15, 35)
    COLOR_PADDLE: Tuple[int, int, int] = (52, 152, 219)
    COLOR_BALL: Tuple[int, int, int] = (241, 196, 15)
    COLOR_BRICK_COLORS: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (231, 76, 60),  # Red
            (230, 126, 34),  # Orange
            (241, 196, 15),  # Yellow
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
        ]
    )
    COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)

    # Neural network visualizer
    VIS_NEURON_RADIUS: int = 8
    VIS_LAYER_SPACING: int = 150
    VIS_NEURON_SPACING: int = 20
    VIS_MAX_NEURONS_DISPLAY: int = 20  # Limit for very large layers
    VIS_FAST_MODE: bool = False  # Skip glow/highlight effects for performance

    # Activation coloring
    VIS_COLOR_INACTIVE: Tuple[int, int, int] = (50, 50, 50)
    VIS_COLOR_ACTIVE: Tuple[int, int, int] = (0, 255, 128)
    VIS_COLOR_WEIGHT_POS: Tuple[int, int, int] = (100, 200, 100)
    VIS_COLOR_WEIGHT_NEG: Tuple[int, int, int] = (200, 100, 100)

    # Dashboard
    PLOT_HISTORY_LENGTH: int = 100  # Number of episodes to show in plots

    # Training HUD (Heads-Up Display)
    HUD_ENABLED: bool = True  # Show on-screen training stats
    HUD_OPACITY: float = 0.8  # Opacity of HUD elements (0.0 to 1.0)

    # =========================================================================
    # TRAINING CONTROL
    # =========================================================================

    # Total episodes to train (0 = unlimited, train until manually stopped)
    MAX_EPISODES: int = 0

    # Maximum steps per episode (prevents infinite games)
    MAX_STEPS_PER_EPISODE: int = 10000

    # Save model every N episodes
    SAVE_EVERY: int = 100

    # =========================================================================
    # EVALUATION SETTINGS (Deterministic Performance Tracking)
    # =========================================================================

    # Run deterministic evaluation (ε=0) every N episodes
    # This measures TRUE performance, separate from noisy training metrics
    # Set to 0 to disable periodic evaluation
    EVAL_EVERY: int = 500

    # Number of games per evaluation run
    EVAL_EPISODES: int = 30

    # Max steps per eval game (prevents stuck games)
    EVAL_MAX_STEPS: int = 5000

    # Plateau detection: warn if no improvement after N evals
    EVAL_PLATEAU_THRESHOLD: int = 5

    # Early-stop: end a training run once eval has plateaued for this many evals,
    # instead of training the live policy past its peak into collapse (the best
    # checkpoint already holds the peak). Opt-in via --early-stop. Off by default.
    EARLY_STOP_ON_PLATEAU: bool = False
    EARLY_STOP_PATIENCE: int = 4

    # Auto-exploration boost: when plateau detected, reset epsilon to explore new strategies
    # This helps escape local optima by forcing the agent to try new behaviors
    EVAL_PLATEAU_EPSILON_BOOST: float = 0.15  # Reset epsilon to this value when plateau detected
    EVAL_PLATEAU_BOOST_EPISODES: int = 1000  # Keep boosted epsilon for this many episodes
    # Hard-disable the plateau exploration boost (e.g. on full-objective Crystal
    # Caves stages where perturbing a peaked policy tends to drive it into collapse).
    DISABLE_EXPLORATION_BOOST: bool = False
    # Skip the boost when the held-out win rate has fallen below this fraction of the
    # best win rate seen — that is regression, not a plateau, and should early-stop
    # rather than inject more randomness into an already-collapsing policy.
    EVAL_BOOST_WIN_REGRESSION_FRAC: float = 0.7

    # Held-out "keep-best" selection score. Crystal Caves uses continuous progress
    # signals rather than raw win rate so eval-best / plateau can see improvements
    # before wins appear. Other games keep the older win/score fallback.
    EVAL_SELECTION_W_WIN: float = 1.0
    EVAL_SELECTION_W_CRYSTAL: float = 0.5
    EVAL_SELECTION_W_DEPTH: float = 0.3
    EVAL_SELECTION_W_TARGET_DISTANCE: float = 0.2
    EVAL_SELECTION_W_EXIT_UNLOCKED: float = 1.0
    EVAL_SELECTION_W_SCORE: float = 0.0001

    # Render every N episodes during training (0 = never)
    RENDER_EVERY: int = 1

    # Print stats every N episodes
    LOG_EVERY: int = 10

    # Report interval for headless mode (seconds between progress reports)
    REPORT_INTERVAL_SECONDS: float = 5.0

    # =========================================================================
    # SYSTEM SETTINGS
    # =========================================================================

    # Device selection
    @property
    def DEVICE(self) -> torch.device:
        """Auto-detect CUDA/MPS/CPU, or force CPU if configured."""
        # Force CPU mode (faster for small models on M4)
        if self.FORCE_CPU:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Paths
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"

    # Logging settings
    # Level options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True

    @property
    def GAME_MODEL_DIR(self) -> str:
        """Get game-specific model directory (e.g., 'models/breakout/')."""
        import os

        return os.path.join(self.MODEL_DIR, self.GAME_NAME)

    @property
    def game(self) -> GameSettings:
        """Return grouped game settings for new code paths."""
        return GameSettings(
            game_name=self.GAME_NAME,
            screen_width=self.SCREEN_WIDTH,
            screen_height=self.SCREEN_HEIGHT,
            fps=self.FPS,
            lives=self.LIVES,
        )

    @property
    def network(self) -> NetworkSettings:
        """Return grouped neural-network settings for new code paths."""
        return NetworkSettings(
            action_size=self.ACTION_SIZE,
            hidden_layers=tuple(self.HIDDEN_LAYERS),
            activation=self.ACTIVATION,
            use_dueling=self.USE_DUELING,
            use_noisy_networks=self.USE_NOISY_NETWORKS,
            use_distributional_dqn=self.USE_DISTRIBUTIONAL_DQN,
            c51_num_atoms=self.C51_NUM_ATOMS,
            c51_v_min=self.C51_V_MIN,
            c51_v_max=self.C51_V_MAX,
        )

    @property
    def training(self) -> TrainingSettings:
        """Return grouped training settings for new code paths."""
        return TrainingSettings(
            learning_rate=self.LEARNING_RATE,
            gamma=self.GAMMA,
            batch_size=self.BATCH_SIZE,
            memory_size=self.MEMORY_SIZE,
            target_update=self.TARGET_UPDATE,
            learn_every=self.LEARN_EVERY,
            gradient_steps=self.GRADIENT_STEPS,
        )

    @property
    def runtime(self) -> RuntimeSettings:
        """Return grouped runtime settings for new code paths."""
        return RuntimeSettings(
            model_dir=self.MODEL_DIR,
            game_model_dir=self.GAME_MODEL_DIR,
            log_dir=self.LOG_DIR,
            device=self.DEVICE,
            max_episodes=self.MAX_EPISODES,
            max_steps_per_episode=self.MAX_STEPS_PER_EPISODE,
        )

    # Random seed for reproducibility (None for random)
    SEED: Optional[int] = None

    def __post_init__(self):
        """Validation and derived calculations."""
        self._require(self.LEARNING_RATE > 0, "Learning rate must be positive")
        self._require(self.WEIGHT_DECAY >= 0, "Weight decay must be non-negative")
        self._require(0 < self.GAMMA <= 1, "Gamma must be in (0, 1]")
        self._require(self.MEMORY_SIZE > 0, "Memory size must be positive")
        self._require(self.BATCH_SIZE > 0, "Batch size must be positive")
        self._require(
            self.BATCH_SIZE <= self.MEMORY_SIZE,
            f"Batch size ({self.BATCH_SIZE}) cannot exceed memory size ({self.MEMORY_SIZE})",
        )
        self._require(self.TARGET_UPDATE > 0, "TARGET_UPDATE must be positive")
        self._require(self.GRAD_CLIP > 0, "GRAD_CLIP must be positive")
        self._require(self.EPSILON_START >= self.EPSILON_END, "Epsilon start must be >= end")
        self._require(self.LEARN_EVERY >= 1, "LEARN_EVERY must be >= 1")
        self._require(self.GRADIENT_STEPS >= 1, "GRADIENT_STEPS must be >= 1")
        self._require(self.C51_NUM_ATOMS >= 2, "C51_NUM_ATOMS must be at least 2")
        self._require(self.C51_V_MIN < self.C51_V_MAX, "C51_V_MIN must be less than C51_V_MAX")
        self._require(self.MAX_EPISODES >= 0, "MAX_EPISODES must be non-negative")
        self._require(self.MAX_STEPS_PER_EPISODE > 0, "MAX_STEPS_PER_EPISODE must be positive")
        self._require(self.SAVE_EVERY > 0, "SAVE_EVERY must be positive")
        self._require(self.RENDER_EVERY >= 0, "RENDER_EVERY must be non-negative")
        self._require(self.LOG_EVERY > 0, "LOG_EVERY must be positive")
        self._require(self.PLOT_HISTORY_LENGTH > 0, "PLOT_HISTORY_LENGTH must be positive")
        self._require(self.EVAL_EVERY >= 0, "EVAL_EVERY must be non-negative")
        self._require(self.EVAL_EPISODES > 0, "EVAL_EPISODES must be positive")
        self._require(self.EVAL_MAX_STEPS > 0, "EVAL_MAX_STEPS must be positive")
        self._require(self.EVAL_PLATEAU_THRESHOLD > 0, "EVAL_PLATEAU_THRESHOLD must be positive")
        self._require(
            0 <= self.EVAL_PLATEAU_EPSILON_BOOST <= 1,
            "EVAL_PLATEAU_EPSILON_BOOST must be between 0 and 1",
        )
        self._require(
            self.EVAL_PLATEAU_BOOST_EPISODES > 0,
            "EVAL_PLATEAU_BOOST_EPISODES must be positive",
        )
        self._require(self.PER_ALPHA >= 0, "PER_ALPHA must be non-negative")
        self._require(0 < self.PER_BETA_START <= 1, "PER_BETA_START must be in (0, 1]")
        self._require(self.PER_BETA_FRAMES > 0, "PER_BETA_FRAMES must be positive")
        self._require(self.N_STEP_SIZE > 0, "N_STEP_SIZE must be positive")
        self._require(
            math.isfinite(self.CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT)
            and self.CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT >= 0,
            "CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT must be finite and non-negative",
        )
        self._require(
            0.0 <= self.CRYSTAL_CAVES_REVERSE_CURRICULUM_P <= 1.0,
            "CRYSTAL_CAVES_REVERSE_CURRICULUM_P must be in [0, 1]",
        )
        self._require(
            self.CRYSTAL_CAVES_STALL_WINDOW_STEPS >= 0,
            "CRYSTAL_CAVES_STALL_WINDOW_STEPS must be non-negative (0 = game default)",
        )
        self._require(
            self.CRYSTAL_CAVES_MAX_STEPS_OVERRIDE >= 0,
            "CRYSTAL_CAVES_MAX_STEPS_OVERRIDE must be non-negative (0 = game default)",
        )
        self._require(
            self.CRYSTAL_CAVES_WIN_AT_K_RAMP_EPISODES >= 0,
            "CRYSTAL_CAVES_WIN_AT_K_RAMP_EPISODES must be non-negative (0 = static K)",
        )
        self._require(
            self.CRYSTAL_CAVES_WIN_AT_K_RAMP_DELAY >= 0,
            "CRYSTAL_CAVES_WIN_AT_K_RAMP_DELAY must be non-negative (0 = no hold)",
        )
        self._require(
            self.CRYSTAL_CAVES_DEMO_BACKWARD_RETREAT >= 0,
            "CRYSTAL_CAVES_DEMO_BACKWARD_RETREAT must be non-negative (0 = default)",
        )
        self._require(
            self.CRYSTAL_CAVES_DEMO_BACKWARD_WINS >= 0,
            "CRYSTAL_CAVES_DEMO_BACKWARD_WINS must be non-negative (0 = default)",
        )
        self._require(
            0.0 <= self.CRYSTAL_CAVES_DEMO_LEVEL_BIAS <= 1.0,
            "CRYSTAL_CAVES_DEMO_LEVEL_BIAS must be in [0, 1]",
        )
        self._require(
            self.CRYSTAL_CAVES_DEMO_BACKWARD_WINDOW >= 0,
            "CRYSTAL_CAVES_DEMO_BACKWARD_WINDOW must be non-negative (0 = frontier only)",
        )
        self._require(
            self.CRYSTAL_CAVES_DEMO_BACKWARD_DEEP >= 0,
            "CRYSTAL_CAVES_DEMO_BACKWARD_DEEP must be non-negative (0 = off)",
        )
        self._require(
            self.DEMO_OPENING_ONLY_STEPS >= 0,
            "DEMO_OPENING_ONLY_STEPS must be non-negative (0 = keep full routes)",
        )
        self._require(
            self.DEMO_MARGIN_DECAY_EPISODES >= 0,
            "DEMO_MARGIN_DECAY_EPISODES must be non-negative (0 = constant weight)",
        )
        self._require(
            self.DEMO_MARGIN_REIGNITE_EPISODE >= 0,
            "DEMO_MARGIN_REIGNITE_EPISODE must be non-negative (0 = off)",
        )
        self._require(
            0.0 <= self.DEMO_MARGIN_REIGNITE_SCALE <= 1.0,
            "DEMO_MARGIN_REIGNITE_SCALE must be in [0, 1]",
        )
        self._require(
            0.0 <= self.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P <= 1.0,
            "CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P must be in [0, 1]",
        )
        self._require(
            isinstance(self.CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES, int)
            and self.CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES >= 0,
            "CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES must be a non-negative integer",
        )
        self._require(
            math.isfinite(self.CRYSTAL_CAVES_NGU_BETA) and self.CRYSTAL_CAVES_NGU_BETA >= 0,
            "CRYSTAL_CAVES_NGU_BETA must be finite and non-negative",
        )
        self._require(self.SCREEN_WIDTH > 0, "Screen width must be positive")
        self._require(self.SCREEN_HEIGHT > 0, "Screen height must be positive")
        self._require(self.BALL_SPEED > 0, "Ball speed must be positive")
        self._require(len(self.HIDDEN_LAYERS) > 0, "Must have at least one hidden layer")
        self._require(
            all(isinstance(size, int) and size > 0 for size in self.HIDDEN_LAYERS),
            "Hidden layer sizes must be positive integers",
        )
        # Warn about unlimited training
        if self.MAX_EPISODES == 0:
            import warnings

            warnings.warn(
                "MAX_EPISODES is 0 - training will run indefinitely until manually stopped",
                UserWarning,
            )

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)


# Global config instance for easy importing
config = Config()


if __name__ == "__main__":
    # Print configuration summary
    cfg = Config()
    print("=" * 60)
    print("Neural Network Game AI - Configuration Summary")
    print("=" * 60)
    print(f"\n📺 Game: {cfg.SCREEN_WIDTH}x{cfg.SCREEN_HEIGHT}")
    print(f"🧱 Bricks: {cfg.BRICK_ROWS}x{cfg.BRICK_COLS} = {cfg.BRICK_ROWS * cfg.BRICK_COLS}")
    print("\n🧠 Neural Network:")
    print(f"   Input size: {cfg.STATE_SIZE}")
    print(f"   Hidden layers: {cfg.HIDDEN_LAYERS}")
    print(f"   Output size: {cfg.ACTION_SIZE}")
    print("\n📊 Training:")
    print(f"   Learning rate: {cfg.LEARNING_RATE}")
    print(f"   Batch size: {cfg.BATCH_SIZE}")
    print(f"   Gamma: {cfg.GAMMA}")
    print("\n🎲 Exploration:")
    print(f"   Epsilon: {cfg.EPSILON_START} → {cfg.EPSILON_END}")
    print(f"   Decay: {cfg.EPSILON_DECAY}")
    print(f"\n💻 Device: {cfg.DEVICE}")
    print("=" * 60)
