# ruff: noqa: F401,F403,F405,I001
from .common import *
from .config_helpers import *
from .stats import *


class InterleavedCrystalCavesVec:
    """Vectorized Crystal Caves env that mixes full tutorial caves and drill caves.

    The trainer sees a normal vectorized environment. Internally, a fixed number
    of lanes reset into full procedural tutorial caves and the rest reset into
    hand-authored drills. This gives every replay batch both real objective
    context and clean motor-skill reps.
    """

    def __init__(
        self,
        *,
        full_config: Config,
        skill_config: Config,
        full_envs: int,
        skill_envs: int,
        skill_source: str,
        headless: bool = True,
    ):
        if full_envs <= 0:
            raise ValueError("interleaved training requires at least one full env")
        if skill_envs <= 0:
            raise ValueError("interleaved training requires at least one skill env")
        if not skill_source:
            raise ValueError("skill_source must not be empty")

        self.full_envs = full_envs
        self.skill_envs = skill_envs
        self.skill_source = skill_source
        self.num_envs = full_envs + skill_envs
        self.config = full_config
        self.headless = headless
        self.sources = ["full"] * full_envs + [skill_source] * skill_envs
        self.envs = [
            CrystalCaves(full_config if source == "full" else skill_config, headless=headless)
            for source in self.sources
        ]
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        if any(env.state_size != self.state_size for env in self.envs):
            raise ValueError("interleaved env sources must expose the same state_size")
        if any(env.action_size != self.action_size for env in self.envs):
            raise ValueError("interleaved env sources must expose the same action_size")

        self._states = np.empty((self.num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(self.num_envs, dtype=np.float32)
        self._dones = np.empty(self.num_envs, dtype=np.bool_)
        self._pending_resets = np.zeros(self.num_envs, dtype=np.bool_)
        self._last_infos: list[dict[str, Any]] = []
        self._source_history: dict[str, dict[str, deque[Any]]] = {
            source: {
                "scores": deque(maxlen=100),
                "wins": deque(maxlen=100),
                "progresses": deque(maxlen=100),
                "crystal_fracs": deque(maxlen=100),
                "exit_unlocked": deque(maxlen=100),
                "end_reasons": deque(maxlen=100),
            }
            for source in sorted(set(self.sources))
        }
        self._source_episodes: Counter[str] = Counter()

    def reset(self) -> np.ndarray:
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self._step_into_buffers(actions)
        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        for i, done in enumerate(self._dones):
            if done:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        return states_to_return, rewards_to_return, dones_to_return, self._last_infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        for i in range(self.num_envs):
            if self._pending_resets[i]:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        self._step_into_buffers(actions)
        return self._states, self._rewards, self._dones, self._last_infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seeds: list[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def source_stats(self) -> dict[str, Any]:
        return {
            source: source_stat_snapshot(history, self._source_episodes[source])
            for source, history in self._source_history.items()
        }

    def _step_into_buffers(self, actions: np.ndarray) -> None:
        infos: list[dict[str, Any]] = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            source = self.sources[i]
            next_state, reward, done, info = env.step(int(action))
            info = dict(info)
            info["training_source"] = source
            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)
            if done:
                self._record_source_episode(source, info)
                env.reset()
                self._pending_resets[i] = True
        self._last_infos = infos

    def _record_source_episode(self, source: str, info: dict[str, Any]) -> None:
        history = self._source_history[source]
        history["scores"].append(float(info.get("score", 0) or 0))
        history["wins"].append(bool(info.get("won", False)))
        history["progresses"].append(float(info.get("progress", 0.0) or 0.0))
        progress_parts = info.get("progress_parts") or {}
        history["crystal_fracs"].append(float(progress_parts.get("crystal_frac", 0.0) or 0.0))
        history["exit_unlocked"].append(bool(info.get("exit_unlocked", False)))
        reason = str(info.get("end_reason", "") or "")
        if not reason or reason == "running":
            reason = "won" if info.get("won", False) else "ended"
        history["end_reasons"].append(reason)
        self._source_episodes[source] += 1


def reverse_start_modes(reverse_envs: int) -> list[str]:
    if reverse_envs <= 0:
        raise ValueError("reverse_envs must be positive")
    modes = ["reverse_objective", "reverse_exit"]
    return [modes[i % len(modes)] for i in range(reverse_envs)]


def _safe_reverse_start_tile(
    game: CrystalCaves,
    *,
    col: int,
    row: int,
    target_tile: tuple[int, int],
) -> bool:
    if not (1 <= col < game.level_cols - 1 and 1 <= row < game.level_rows - 1):
        return False
    tile = (col, row)
    if tile == target_tile:
        return False
    if (
        tile in game.crystals
        or tile in game.switches
        or tile == game.exit_pos
        or tile in game.hazards
        or tile in game.doors
    ):
        return False
    if game._solid_at(col, row):
        return False
    if not game._solid_at(col, row + 1) and game.grid[row + 1][col] != game.ELEVATOR:
        return False
    rect = game._player_rect(col * game.TILE_SIZE + 5, row * game.TILE_SIZE + 1)
    return not game._rect_collides_solid(rect)


def place_player_near_tile(
    game: CrystalCaves,
    target_tile: tuple[int, int],
    *,
    max_radius: int = 5,
) -> bool:
    target_col, target_row = target_tile
    candidates: list[tuple[int, int, int]] = []
    for radius in range(1, max_radius + 1):
        for col in range(target_col - radius, target_col + radius + 1):
            for row in range(target_row - 2, target_row + 3):
                distance = abs(col - target_col) + abs(row - target_row)
                if distance <= radius:
                    candidates.append((distance, col, row))
    for _, col, row in sorted(candidates):
        if not _safe_reverse_start_tile(game, col=col, row=row, target_tile=target_tile):
            continue
        # Require the target (e.g. the open exit) to be jump-aware reachable FROM this
        # start, so we never drop the agent in an un-jumpable pocket on the wrong side
        # of a gap — which would otherwise read as a false "can't reach" / false ceiling.
        if target_tile not in game._oracle_reachable((col, row)):
            continue
        game.player_x = col * game.TILE_SIZE + 5
        game.player_y = row * game.TILE_SIZE + 1
        game.vx = 0.0
        game.vy = 0.0
        game.facing = 1 if target_col >= col else -1
        game.grounded = game._is_on_surface()
        game.coyote_timer = 6 if game.grounded else 0
        game._max_depth_row = max(game._max_depth_row, game._player_tile()[1])
        game._progress = game._progress_potential()[0]
        game._target_best_distances = {}
        game.steps_since_progress = 0
        return True
    return False


def place_player_random_reachable(
    game: CrystalCaves,
    target_tile: tuple[int, int],
    *,
    min_distance: int = 4,
) -> bool:
    """Drop the player on a RANDOM safe standing tile from which ``target_tile`` is
    jump-aware oracle-reachable, preferring tiles at least ``min_distance`` (Manhattan)
    away. Unlike ``place_player_near_tile`` (which hugs the target), this samples the
    full level, so a reverse-exit start measures genuine long-range navigation to the
    open exit rather than the trivial final hop next to it."""
    target_col, target_row = target_tile
    candidates: list[tuple[int, int, int]] = []
    for row in range(1, game.level_rows - 1):
        for col in range(1, game.level_cols - 1):
            if not _safe_reverse_start_tile(game, col=col, row=row, target_tile=target_tile):
                continue
            dist = abs(col - target_col) + abs(row - target_row)
            candidates.append((col, row, dist))
    if not candidates:
        return False
    far = [c for c in candidates if c[2] >= min_distance]
    pool = far if far else candidates
    for idx in np.random.permutation(len(pool)):
        col, row, _ = pool[int(idx)]
        if target_tile not in game._oracle_reachable((col, row)):
            continue
        game.player_x = col * game.TILE_SIZE + 5
        game.player_y = row * game.TILE_SIZE + 1
        game.vx = 0.0
        game.vy = 0.0
        game.facing = 1 if target_col >= col else -1
        game.grounded = game._is_on_surface()
        game.coyote_timer = 6 if game.grounded else 0
        game._max_depth_row = max(game._max_depth_row, game._player_tile()[1])
        game._progress = game._progress_potential()[0]
        game._target_best_distances = {}
        game.steps_since_progress = 0
        return True
    return False


def apply_reverse_start(game: CrystalCaves, mode: str) -> bool:
    far = False
    reverse_exit_snapshot = None
    if mode in ("reverse_exit", "reverse_exit_far"):
        reverse_exit_snapshot = (
            set(game.crystals),
            set(game.used_switches),
            set(game.open_colors),
            bool(game.exit_unlocked),
        )
        game.crystals.clear()
        game.used_switches = set(game.switches)
        game.open_colors = set(game.switch_color.values())
        game.exit_unlocked = True
        target_tile = game.exit_pos
        far = mode == "reverse_exit_far"
    elif mode == "reverse_objective":
        target, _ = game._current_target()
        if target is None:
            return False
        _, col, row = target
        target_tile = (col, row)
    else:
        raise ValueError(f"unknown reverse start mode: {mode}")

    if far:
        applied = place_player_random_reachable(game, target_tile)
    else:
        applied = place_player_near_tile(game, target_tile)
    if not applied and reverse_exit_snapshot is not None:
        # Placement failed AFTER the world was mutated: restore the pre-curriculum
        # state so the caller does not train on a half-applied reverse-exit episode.
        crystals, used_switches, open_colors, exit_unlocked = reverse_exit_snapshot
        game.crystals.clear()
        game.crystals.update(crystals)
        game.used_switches = used_switches
        game.open_colors = open_colors
        game.exit_unlocked = exit_unlocked
    game._progress = game._progress_potential()[0]
    return applied


class ReverseStartCrystalCavesVec:
    """Vectorized full-cave env with some lanes reset near late objectives."""

    def __init__(
        self,
        *,
        full_config: Config,
        full_envs: int,
        reverse_envs: int,
        headless: bool = True,
    ):
        if full_envs <= 0:
            raise ValueError("reverse-start training requires at least one full env")
        if reverse_envs <= 0:
            raise ValueError("reverse-start training requires at least one reverse env")

        self.full_envs = full_envs
        self.reverse_envs = reverse_envs
        self.num_envs = full_envs + reverse_envs
        self.config = full_config
        self.headless = headless
        self.sources = ["full"] * full_envs + reverse_start_modes(reverse_envs)
        self.envs = [CrystalCaves(full_config, headless=headless) for _ in self.sources]
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        if any(env.state_size != self.state_size for env in self.envs):
            raise ValueError("reverse-start envs must expose the same state_size")
        if any(env.action_size != self.action_size for env in self.envs):
            raise ValueError("reverse-start envs must expose the same action_size")

        self._states = np.empty((self.num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(self.num_envs, dtype=np.float32)
        self._dones = np.empty(self.num_envs, dtype=np.bool_)
        self._pending_resets = np.zeros(self.num_envs, dtype=np.bool_)
        self._last_infos: list[dict[str, Any]] = []
        self._source_history: dict[str, dict[str, deque[Any]]] = {
            source: {
                "scores": deque(maxlen=100),
                "wins": deque(maxlen=100),
                "progresses": deque(maxlen=100),
                "crystal_fracs": deque(maxlen=100),
                "exit_unlocked": deque(maxlen=100),
                "end_reasons": deque(maxlen=100),
            }
            for source in sorted(set(self.sources))
        }
        self._source_episodes: Counter[str] = Counter()
        self._reverse_attempts: Counter[str] = Counter()
        self._reverse_applied: Counter[str] = Counter()

    def reset(self) -> np.ndarray:
        for i in range(self.num_envs):
            self._reset_env(i)
        return self._states.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self._step_into_buffers(actions)
        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        for i, done in enumerate(self._dones):
            if done:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        return states_to_return, rewards_to_return, dones_to_return, self._last_infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        for i in range(self.num_envs):
            if self._pending_resets[i]:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        self._step_into_buffers(actions)
        return self._states, self._rewards, self._dones, self._last_infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seeds: list[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def source_stats(self) -> dict[str, Any]:
        return {
            source: source_stat_snapshot(history, self._source_episodes[source])
            for source, history in self._source_history.items()
        }

    def reverse_start_stats(self) -> dict[str, Any]:
        return {
            source: {
                "attempts": int(self._reverse_attempts[source]),
                "applied": int(self._reverse_applied[source]),
                "apply_rate": (
                    self._reverse_applied[source] / self._reverse_attempts[source]
                    if self._reverse_attempts[source]
                    else 0.0
                ),
            }
            for source in sorted(set(self.sources))
            if source.startswith("reverse_")
        }

    def _reset_env(self, i: int) -> None:
        env = self.envs[i]
        source = self.sources[i]
        self._states[i] = env.reset()
        if source.startswith("reverse_"):
            self._reverse_attempts[source] += 1
            if apply_reverse_start(env, source):
                self._reverse_applied[source] += 1
            self._states[i] = env.get_state()

    def _step_into_buffers(self, actions: np.ndarray) -> None:
        infos: list[dict[str, Any]] = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            source = self.sources[i]
            next_state, reward, done, info = env.step(int(action))
            info = dict(info)
            info["training_source"] = source
            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)
            if done:
                self._record_source_episode(source, info)
                self._reset_env(i)
                self._pending_resets[i] = True
        self._last_infos = infos

    def _record_source_episode(self, source: str, info: dict[str, Any]) -> None:
        history = self._source_history[source]
        history["scores"].append(float(info.get("score", 0) or 0))
        history["wins"].append(bool(info.get("won", False)))
        history["progresses"].append(float(info.get("progress", 0.0) or 0.0))
        progress_parts = info.get("progress_parts") or {}
        history["crystal_fracs"].append(float(progress_parts.get("crystal_frac", 0.0) or 0.0))
        history["exit_unlocked"].append(bool(info.get("exit_unlocked", False)))
        reason = str(info.get("end_reason", "") or "")
        if not reason or reason == "running":
            reason = "won" if info.get("won", False) else "ended"
        history["end_reasons"].append(reason)
        self._source_episodes[source] += 1


class ArchiveStartCrystalCavesVec:
    """Vectorized full-cave env with some lanes replaying archived mid-run states."""

    def __init__(
        self,
        *,
        full_config: Config,
        full_envs: int,
        archive_envs: int,
        replay_prob: float,
        max_size: int,
        min_steps: int,
        headless: bool = True,
    ):
        if full_envs <= 0:
            raise ValueError("archive-start training requires at least one full env")
        if archive_envs <= 0:
            raise ValueError("archive-start training requires at least one archive env")
        if not 0.0 <= replay_prob <= 1.0:
            raise ValueError("archive replay probability must be between 0 and 1")
        if max_size <= 0:
            raise ValueError("archive max size must be positive")
        if min_steps < 0:
            raise ValueError("archive min steps must be non-negative")

        self.full_envs = full_envs
        self.archive_envs = archive_envs
        self.archive_replay_prob = replay_prob
        self.archive_max_size = max_size
        self.archive_min_steps = min_steps
        self.num_envs = full_envs + archive_envs
        self.config = full_config
        self.headless = headless
        self.sources = ["full"] * full_envs + ["archive"] * archive_envs
        self.envs = [CrystalCaves(full_config, headless=headless) for _ in self.sources]
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        if any(env.state_size != self.state_size for env in self.envs):
            raise ValueError("archive-start envs must expose the same state_size")
        if any(env.action_size != self.action_size for env in self.envs):
            raise ValueError("archive-start envs must expose the same action_size")

        self._rng = random.Random(int(full_config.CRYSTAL_CAVES_SEED) + 7919)
        self._states = np.empty((self.num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(self.num_envs, dtype=np.float32)
        self._dones = np.empty(self.num_envs, dtype=np.bool_)
        self._pending_resets = np.zeros(self.num_envs, dtype=np.bool_)
        self._last_infos: list[dict[str, Any]] = []
        self._source_history: dict[str, dict[str, deque[Any]]] = {
            source: {
                "scores": deque(maxlen=100),
                "wins": deque(maxlen=100),
                "progresses": deque(maxlen=100),
                "crystal_fracs": deque(maxlen=100),
                "exit_unlocked": deque(maxlen=100),
                "end_reasons": deque(maxlen=100),
            }
            for source in sorted(set(self.sources))
        }
        self._source_episodes: Counter[str] = Counter()
        self._archive: list[dict[str, Any]] = []
        self._archive_current_keys: set[tuple[str, int, int, int, int]] = set()
        self._archive_seen_keys: set[tuple[str, int, int, int, int]] = set()
        self._archive_stores = 0
        self._archive_evictions = 0
        self._archive_replay_attempts = 0
        self._archive_replays = 0
        self._archive_store_failures = 0

    def reset(self) -> np.ndarray:
        for i in range(self.num_envs):
            self._reset_env(i)
        return self._states.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        self._step_into_buffers(actions)
        states_to_return = self._states.copy()
        rewards_to_return = self._rewards.copy()
        dones_to_return = self._dones.copy()

        for i, done in enumerate(self._dones):
            if done:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        return states_to_return, rewards_to_return, dones_to_return, self._last_infos

    def step_no_copy(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        for i in range(self.num_envs):
            if self._pending_resets[i]:
                self._states[i] = self.envs[i].get_state()
                self._pending_resets[i] = False

        self._step_into_buffers(actions)
        return self._states, self._rewards, self._dones, self._last_infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def seed(self, seeds: list[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def source_stats(self) -> dict[str, Any]:
        return {
            source: source_stat_snapshot(history, self._source_episodes[source])
            for source, history in self._source_history.items()
        }

    def archive_stats(self) -> dict[str, Any]:
        return {
            "size": len(self._archive),
            "max_size": self.archive_max_size,
            "seen_milestones": len(self._archive_seen_keys),
            "stores": self._archive_stores,
            "evictions": self._archive_evictions,
            "replay_attempts": self._archive_replay_attempts,
            "replays": self._archive_replays,
            "replay_rate": (
                self._archive_replays / self._archive_replay_attempts
                if self._archive_replay_attempts
                else 0.0
            ),
            "store_failures": self._archive_store_failures,
        }

    def _reset_env(self, i: int) -> None:
        source = self.sources[i]
        if source == "archive":
            self._archive_replay_attempts += 1
            if self._archive and self._rng.random() < self.archive_replay_prob:
                entry = self._rng.choice(self._archive)
                self.envs[i] = copy.deepcopy(entry["game"])
                self._archive_replays += 1
                self._states[i] = self.envs[i].get_state()
                return
        self._states[i] = self.envs[i].reset()

    def _step_into_buffers(self, actions: np.ndarray) -> None:
        infos: list[dict[str, Any]] = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            source = self.sources[i]
            next_state, reward, done, info = env.step(int(action))
            info = dict(info)
            info["training_source"] = source
            self._states[i] = next_state
            self._rewards[i] = reward
            self._dones[i] = done
            infos.append(info)
            if source == "full" and not done:
                self._maybe_archive(env)
            if done:
                self._record_source_episode(source, info)
                self._reset_env(i)
                self._pending_resets[i] = True
        self._last_infos = infos

    def _maybe_archive(self, game: CrystalCaves) -> None:
        if game.steps < self.archive_min_steps or game.game_over or game.won:
            return
        key = archive_milestone_key(game)
        if key in self._archive_seen_keys:
            return
        try:
            snapshot = copy.deepcopy(game)
        except Exception:
            self._archive_store_failures += 1
            return
        if len(self._archive) >= self.archive_max_size:
            evicted = self._archive.pop(0)
            self._archive_current_keys.discard(evicted["key"])
            self._archive_evictions += 1
        self._archive.append(
            {
                "key": key,
                "game": snapshot,
                "steps": int(game.steps),
                "progress": float(game._progress),
            }
        )
        self._archive_current_keys.add(key)
        self._archive_seen_keys.add(key)
        self._archive_stores += 1

    def _record_source_episode(self, source: str, info: dict[str, Any]) -> None:
        history = self._source_history[source]
        history["scores"].append(float(info.get("score", 0) or 0))
        history["wins"].append(bool(info.get("won", False)))
        history["progresses"].append(float(info.get("progress", 0.0) or 0.0))
        progress_parts = info.get("progress_parts") or {}
        history["crystal_fracs"].append(float(progress_parts.get("crystal_frac", 0.0) or 0.0))
        history["exit_unlocked"].append(bool(info.get("exit_unlocked", False)))
        reason = str(info.get("end_reason", "") or "")
        if not reason or reason == "running":
            reason = "won" if info.get("won", False) else "ended"
        history["end_reasons"].append(reason)
        self._source_episodes[source] += 1


def source_stat_snapshot(history: dict[str, deque[Any]], total_episodes: int) -> dict[str, Any]:
    scores = [float(value) for value in history["scores"]]
    wins = [bool(value) for value in history["wins"]]
    progresses = [float(value) for value in history["progresses"]]
    crystal_fracs = [float(value) for value in history.get("crystal_fracs", ())]
    exits = [bool(value) for value in history.get("exit_unlocked", ())]
    return {
        "episodes": total_episodes,
        "window_episodes": len(scores),
        "avg_score_100": mean_tail(scores),
        "win_rate_100": float(np.mean(wins)) if wins else 0.0,
        "crystal_rate_100": (
            float(np.mean([value > 0 for value in crystal_fracs])) if crystal_fracs else 0.0
        ),
        "avg_crystal_frac_100": mean_tail(crystal_fracs),
        "exit_rate_100": float(np.mean(exits)) if exits else 0.0,
        "avg_progress_100": mean_tail(progresses),
        "best_progress_100": max_or_zero(progresses),
        "end_reason_counts_100": dict(Counter(str(value) for value in history["end_reasons"])),
    }


def interleave_counts(
    *,
    vec_envs: int,
    skill_ratio: float,
    skill_envs: int | None,
) -> tuple[int, int]:
    if vec_envs < 2:
        raise ValueError("interleaved training requires --vec-envs >= 2")
    if skill_envs is None:
        if skill_ratio <= 0:
            raise ValueError("interleaved training needs a positive skill ratio")
        skill_envs = max(1, round(vec_envs * skill_ratio))
    skill_envs = max(1, min(int(skill_envs), vec_envs - 1))
    return vec_envs - skill_envs, skill_envs


def reverse_start_counts(
    *,
    vec_envs: int,
    reverse_ratio: float,
    reverse_envs: int | None,
) -> tuple[int, int]:
    return interleave_counts(
        vec_envs=vec_envs,
        skill_ratio=reverse_ratio,
        skill_envs=reverse_envs,
    )


def archive_start_counts(
    *,
    vec_envs: int,
    archive_ratio: float,
    archive_envs: int | None,
) -> tuple[int, int]:
    return interleave_counts(
        vec_envs=vec_envs,
        skill_ratio=archive_ratio,
        skill_envs=archive_envs,
    )


def archive_milestone_key(game: CrystalCaves) -> tuple[str, int, int, int, int]:
    """Compact key for a useful full-cave state reached by normal exploration."""

    target, _ = game._current_target()
    target_kind = str(target[0]) if target else "none"
    player_col, player_row = game._player_tile()
    region_cols = 6
    region_rows = 4
    region_col = min(
        region_cols - 1,
        max(0, int(player_col * region_cols / max(1, game.level_cols))),
    )
    region_row = min(
        region_rows - 1,
        max(0, int(player_row * region_rows / max(1, game.level_rows))),
    )
    crystals_collected = max(0, int(game.initial_crystals - len(game.crystals)))
    depth_bucket = min(4, max(0, int(game._max_depth_row * 5 / max(1, game.level_rows))))
    return (target_kind, region_col, region_row, crystals_collected, depth_bucket)


def make_interleaved_drill_config(full_config: Config) -> Config:
    drill_config = Config()
    for key, value in vars(full_config).items():
        setattr(drill_config, key, value)
    drill_config.CRYSTAL_CAVES_PROCEDURAL = False
    drill_config.CRYSTAL_CAVES_DRILLS = True
    exp_config = cc_experiment_config(drill_config)
    exp_config.CRYSTAL_CAVES_BRIDGES = False
    exp_config.CRYSTAL_CAVES_CONTACT_LEVELS = False
    return drill_config


def make_interleaved_bridge_config(full_config: Config) -> Config:
    bridge_config = Config()
    for key, value in vars(full_config).items():
        setattr(bridge_config, key, value)
    bridge_config.CRYSTAL_CAVES_PROCEDURAL = False
    bridge_config.CRYSTAL_CAVES_DRILLS = False
    exp_config = cc_experiment_config(bridge_config)
    exp_config.CRYSTAL_CAVES_BRIDGES = True
    exp_config.CRYSTAL_CAVES_CONTACT_LEVELS = False
    return bridge_config


def make_interleaved_contact_config(full_config: Config) -> Config:
    contact_config = Config()
    for key, value in vars(full_config).items():
        setattr(contact_config, key, value)
    contact_config.CRYSTAL_CAVES_PROCEDURAL = False
    contact_config.CRYSTAL_CAVES_DRILLS = False
    exp_config = cc_experiment_config(contact_config)
    exp_config.CRYSTAL_CAVES_BRIDGES = False
    exp_config.CRYSTAL_CAVES_CONTACT_LEVELS = True
    return contact_config


def install_interleaved_vec_env(
    trainer: HeadlessTrainer,
    *,
    run_dir: Path,
    full_envs: int,
    skill_envs: int,
    skill_source: str,
    skill_config: Config,
) -> None:
    old_vec = trainer.vec_env
    skill_config.MODEL_DIR = str(run_dir / f"{skill_source}_source" / "models")
    skill_config.LOG_DIR = str(run_dir / f"{skill_source}_source" / "logs")
    mixed = InterleavedCrystalCavesVec(
        full_config=trainer.config,
        skill_config=skill_config,
        full_envs=full_envs,
        skill_envs=skill_envs,
        skill_source=skill_source,
        headless=True,
    )
    trainer.vec_env = cast(Any, mixed)
    trainer.game = mixed.envs[0]
    trainer.num_envs = mixed.num_envs
    if old_vec is not None:
        old_vec.close()


def install_reverse_start_vec_env(
    trainer: HeadlessTrainer,
    *,
    full_envs: int,
    reverse_envs: int,
) -> None:
    old_vec = trainer.vec_env
    mixed = ReverseStartCrystalCavesVec(
        full_config=trainer.config,
        full_envs=full_envs,
        reverse_envs=reverse_envs,
        headless=True,
    )
    trainer.vec_env = cast(Any, mixed)
    trainer.game = mixed.envs[0]
    trainer.num_envs = mixed.num_envs
    if old_vec is not None:
        old_vec.close()


def install_archive_start_vec_env(
    trainer: HeadlessTrainer,
    *,
    full_envs: int,
    archive_envs: int,
    replay_prob: float,
    max_size: int,
    min_steps: int,
) -> None:
    old_vec = trainer.vec_env
    mixed = ArchiveStartCrystalCavesVec(
        full_config=trainer.config,
        full_envs=full_envs,
        archive_envs=archive_envs,
        replay_prob=replay_prob,
        max_size=max_size,
        min_steps=min_steps,
        headless=True,
    )
    trainer.vec_env = cast(Any, mixed)
    trainer.game = mixed.envs[0]
    trainer.num_envs = mixed.num_envs
    if old_vec is not None:
        old_vec.close()
