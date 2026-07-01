"""
Tests for the Crystal Caves-style game implementation.

These tests focus on the reusable base: NN-compatible state, core platformer
mechanics, pickups, switches, hazards, shooting, rendering, and vectorized
training support.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from scripts.render_crystal_caves_gallery import render_gallery
from src.app.headless import HeadlessTrainer, reverse_curriculum_p_for_episode
from src.game import get_game_info, list_games
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_art import SPRITES
from src.game.crystal_caves_entities import CaveSpec, Elevator, Enemy
from src.game.crystal_caves_vec import VecCrystalCaves


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def game(config):
    """Create a headless Crystal Caves game."""
    return CrystalCaves(config, headless=True)


def place_on_floor(game: CrystalCaves, col: int | None = None) -> tuple[int, int]:
    """Place the player on a roomy floor tile: a solid tile with several tiles of
    headroom above it and open space on both sides, so jumping and walking are
    unobstructed (denser caves have many low-ceiling nooks)."""
    for row in range(3, game.level_rows):
        for candidate_col in range(2, game.level_cols - 2):
            if col is not None and candidate_col != col:
                continue
            if game.grid[row][candidate_col] != CrystalCaves.SOLID:
                continue
            # Three tiles of headroom for a full jump.
            if any(game._solid_at(candidate_col, row - k) for k in (1, 2, 3)):
                continue
            # Open at body height on both sides so horizontal moves are free.
            if game._solid_at(candidate_col - 1, row - 1) or game._solid_at(
                candidate_col + 1, row - 1
            ):
                continue
            game.player_x = candidate_col * game.TILE_SIZE + 5
            game.player_y = row * game.TILE_SIZE - game.PLAYER_HEIGHT
            game.vx = 0
            game.vy = 0
            return candidate_col, row
    raise AssertionError("No valid floor tile found")


def clear_lane(game: CrystalCaves, col: int, row: int, length: int = 6) -> None:
    """Open a clear horizontal lane to the right at body height so shooting
    tests have an unobstructed line of fire in the denser caves."""
    for dc in range(1, length):
        c = col + dc
        if 0 < c < game.level_cols - 1:
            game.grid[row - 1][c] = CrystalCaves.EMPTY


class TestCrystalCavesInitialization:
    """Test game initialization and registry integration."""

    def test_game_creates_successfully(self, config):
        game = CrystalCaves(config, headless=True)
        assert game is not None

    def test_state_shape(self, game):
        state = game.get_state()
        expected_size = (
            game.WINDOW_COLS * game.WINDOW_ROWS
            + game.GLOBAL_MAP_COLS * game.GLOBAL_MAP_ROWS
            + game.METADATA_SIZE
        )
        assert state.shape == (expected_size,)
        assert game.state_size == expected_size

    def test_action_size_and_labels(self, game):
        assert game.action_size == 10
        assert game.get_action_labels()[0] == "IDLE"
        assert game.get_action_labels()[-1] == "INTERACT"

    def test_registry_includes_game(self):
        assert "crystal_caves" in list_games()
        info = get_game_info("crystal_caves")
        assert info is not None
        assert info["name"] == "Crystal Caves"

    def test_first_cave_has_authored_room_beats(self, game):
        """The first cave should be an authored intro, not a random tile field."""
        player_col, _ = game._player_tile()
        switch_col = next(iter(game.switches))[0]
        exit_col = game.exit_pos[0]

        assert player_col < 6
        assert switch_col > player_col
        assert exit_col > switch_col
        assert len(game.crystals) >= 8
        assert len(game.ammo_pickups) >= 3
        assert len(game.doors) >= 1
        assert len(game.hazards) >= 3


class TestCrystalCavesState:
    """Test the neural-network state representation."""

    def test_state_is_float32(self, game):
        assert game.get_state().dtype == np.float32

    def test_state_values_are_normalized(self, game):
        state = game.get_state()
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

    def test_center_tile_marks_player(self, game):
        state = game.get_state()
        center_idx = (game.WINDOW_COLS * game.WINDOW_ROWS) // 2
        assert state[center_idx] == 1.0

    def test_target_metadata_points_to_active_objective(self, game):
        # Levers are thrown first (they gate crystals); throw them so the crystal-
        # collection phase is active, then verify the compass points at a crystal.
        game.used_switches = set(game.switches)
        game.open_colors = set(game.door_color.values())
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE - game.TILE_SIZE
        game.player_y = crystal[1] * game.TILE_SIZE

        state = game.get_state()
        metadata_start = (
            game.WINDOW_COLS * game.WINDOW_ROWS + game.GLOBAL_MAP_COLS * game.GLOBAL_MAP_ROWS
        )
        target_dx = state[metadata_start + 15]
        target_distance = state[metadata_start + 17]
        target_kind = state[metadata_start + 18]

        assert target_dx > 0.5
        assert 0.0 <= target_distance <= 1.0
        assert target_kind == pytest.approx(0.25)

    def test_target_points_to_switch_before_crystals(self, game):
        """With levers unthrown and crystals remaining, the compass targets the
        switch first — it gates a crystal, so it must be thrown to finish."""
        assert game.switches and game.crystals
        target, _ = game._current_target()
        assert target is not None and target[0] == "switch"

    def test_history_state_is_opt_in(self, game):
        """Default Crystal Caves state shape must stay checkpoint-compatible."""
        assert game.METADATA_SIZE == game.BASE_METADATA_SIZE
        assert game.config.STATE_LAYOUT["meta"] == game.BASE_METADATA_SIZE
        assert game.get_state().shape == (game.state_size,)

    def test_history_state_appends_action_metadata(self, config):
        config.CRYSTAL_CAVES_HISTORY_STATE = True
        config.CRYSTAL_CAVES_HISTORY_STEPS = 4
        game = CrystalCaves(config, headless=True)

        expected_history = game.HISTORY_FEATURES_PER_STEP * 4
        assert game.METADATA_SIZE == game.BASE_METADATA_SIZE + expected_history
        assert game.config.STATE_LAYOUT["meta"] == game.METADATA_SIZE
        assert game.get_state().shape == (game.state_size,)

        empty_history_tail = game.get_state()[-game.HISTORY_FEATURES_PER_STEP :]
        assert empty_history_tail.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

        place_on_floor(game)
        state, *_ = game.step(CrystalCaves.RIGHT_JUMP)
        newest_action = state[-game.HISTORY_FEATURES_PER_STEP :]

        assert newest_action[0] == 0.0  # idle
        assert newest_action[2] == 1.0  # right
        assert newest_action[3] == 1.0  # jump
        assert 0.0 <= newest_action[6] <= 1.0

        reset_state = game.reset()
        reset_tail = reset_state[-game.HISTORY_FEATURES_PER_STEP :]
        assert reset_tail.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])


class TestCrystalCavesMovement:
    """Test platforming actions and physics."""

    def test_move_right_increases_x(self, game):
        place_on_floor(game)
        initial_x = game.player_x
        game.step(CrystalCaves.RIGHT)
        assert game.player_x > initial_x

    def test_moving_toward_same_target_gets_small_progress_reward(self, game):
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE - game.TILE_SIZE
        game.player_y = crystal[1] * game.TILE_SIZE
        game.vx = 0
        game.vy = 0

        _, reward, _, info = game.step(CrystalCaves.RIGHT)

        assert reward > -0.01
        assert info["steps_since_progress"] == 0

    def test_approach_reward_gradient_clears_living_penalty(self, game):
        """Closing distance at full speed must be a clearly-positive net gradient.

        Guards the fix for the greedy stall: the old per-frame approach reward netted
        only ~+0.0005/step against the -0.01 living penalty (sub-noise), so the policy
        idled until the stall timer fired. The gradient must comfortably clear the
        living penalty so approaching the objective is unambiguously worthwhile.
        """
        living_penalty = 0.01
        full_speed_tiles = game.MOVE_SPEED / game.TILE_SIZE
        per_step = min(
            game.APPROACH_REWARD_CLIP_POS,
            full_speed_tiles * game.APPROACH_REWARD_SCALE,
        )
        assert per_step - living_penalty > 0.005

    def test_approach_reward_penalizes_moving_away(self, game):
        """Receding from the objective stays penalized (negative clip preserved)."""
        full_speed_tiles = game.MOVE_SPEED / game.TILE_SIZE
        receding = max(
            game.APPROACH_REWARD_CLIP_NEG,
            -full_speed_tiles * game.APPROACH_REWARD_SCALE,
        )
        assert receding < 0

    def test_progress_pbrs_potential_is_zero_at_terminal(self, game):
        game._progress_initial = 0.2
        game.game_over = True

        assert game._progress_pbrs_potential(raw_progress=0.9) == 0.0
        assert game._progress_pbrs_potential(raw_progress=0.9, terminal=True) == 0.0

    def test_progress_pbrs_discounted_sum_telescopes(self, config):
        config.GAMMA = 0.9
        game = CrystalCaves(config, headless=True)
        game._progress_initial = 0.2
        gamma = config.GAMMA
        phi_start = game._progress_pbrs_potential(raw_progress=0.2, terminal=False)
        phi_mid = game._progress_pbrs_potential(raw_progress=0.45, terminal=False)
        phi_late = game._progress_pbrs_potential(raw_progress=0.75, terminal=False)

        shaping_rewards = [
            gamma * phi_mid - phi_start,
            gamma * phi_late - phi_mid,
            gamma * 0.0 - phi_late,
        ]
        discounted_total = sum(
            (gamma**step) * reward for step, reward in enumerate(shaping_rewards)
        )

        assert phi_start == 0.0
        assert phi_mid > phi_start
        assert phi_late > phi_mid
        assert discounted_total == pytest.approx(0.0, abs=1e-6)

    @staticmethod
    def _place_at_tile(game: CrystalCaves, tile: tuple[int, int]) -> None:
        """Center the player on a tile so ``_player_tile()`` returns it exactly."""
        col, row = tile
        game.player_x = col * game.TILE_SIZE + game.TILE_SIZE / 2 - game.PLAYER_WIDTH / 2
        game.player_y = row * game.TILE_SIZE + game.TILE_SIZE / 2 - game.PLAYER_HEIGHT / 2

    def test_target_closeness_increases_as_player_approaches(self, game: CrystalCaves) -> None:
        # Geodesic closeness is a route distance over the BFS field, so derive the
        # near/far tiles from the field itself to guarantee they are connected.
        field = game._geodesic_distance_field()
        assert field
        near_tile = min(field, key=lambda t: field[t])  # an objective tile (dist 0)
        far_tile = max(field, key=lambda t: field[t])
        assert field[far_tile] > field[near_tile]

        self._place_at_tile(game, far_tile)
        far = game._target_closeness()
        self._place_at_tile(game, near_tile)
        near = game._target_closeness()

        assert 0.0 <= far < near <= 1.0

    def test_geodesic_potential_telescopes_to_zero(self, config: Config) -> None:
        config.GAMMA = 0.9
        config.CRYSTAL_CAVES_GEODESIC_POTENTIAL = True
        config.CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT = 0.3
        game = CrystalCaves(config, headless=True)
        gamma = config.GAMMA

        # Reset captured an initial base+closeness, so Phi(start) must be 0.
        phi_start = game._progress_phi
        assert phi_start == pytest.approx(0.0, abs=1e-6)

        # Approach the objective over connected field tiles (distinct distances,
        # decreasing) without collecting it, accumulating PBRS shaping like step()
        # does, then end the episode (terminal Phi=0). PBRS telescopes the discounted
        # shaping to gamma^N*Phi(terminal) - Phi(start) = 0 for any such path.
        field = game._geodesic_distance_field()
        by_distance = sorted({d: t for t, d in field.items()}.items(), reverse=True)
        path = [tile for _, tile in by_distance[:3]]
        assert len(path) >= 2

        shaping_rewards = []
        prev_phi = phi_start
        closeness_seen = []
        for tile in path:
            self._place_at_tile(game, tile)
            closeness_seen.append(game._target_closeness())
            next_phi = game._progress_pbrs_potential()
            shaping_rewards.append(gamma * next_phi - prev_phi)
            prev_phi = next_phi
        shaping_rewards.append(gamma * 0.0 - prev_phi)  # terminal Phi = 0

        # Sanity: approaching really did raise closeness (the shaping is non-trivial).
        assert closeness_seen[-1] > closeness_seen[0]
        discounted_total = sum(
            (gamma**step) * reward for step, reward in enumerate(shaping_rewards)
        )
        assert discounted_total == pytest.approx(0.0, abs=1e-6)

    def test_geodesic_potential_zero_at_terminal(self, config: Config) -> None:
        config.CRYSTAL_CAVES_GEODESIC_POTENTIAL = True
        game = CrystalCaves(config, headless=True)
        game.game_over = True
        assert game._progress_pbrs_potential(raw_progress=0.9) == 0.0

    def test_geodesic_flag_disables_additive_approach_reward(self, config: Config) -> None:
        config.CRYSTAL_CAVES_GEODESIC_POTENTIAL = True
        game = CrystalCaves(config, headless=True)
        game.switches.clear()
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE - game.TILE_SIZE * 4
        game.player_y = crystal[1] * game.TILE_SIZE
        target, distance = game._current_target()

        game.steps_since_progress = 50
        game.player_x += game.TILE_SIZE
        target_after, closer = game._current_target()
        # Precondition: the move was a genuine approach to the same objective, so the
        # assertions below exercise the intended (approach) code path.
        assert target_after == target
        assert closer < distance
        reward = game._target_progress_reward(target, distance)

        # No additive approach reward when the geodesic potential supplies it...
        assert reward == 0.0
        # ...but the stall timer is still reset on real approach.
        assert game.steps_since_progress == 0

    def test_locked_exit_hidden_in_global_map_by_default(self, config: Config) -> None:
        config.CRYSTAL_CAVES_SHOW_LOCKED_EXIT = False
        game = CrystalCaves(config, headless=True)
        assert not game.exit_unlocked
        gc, gr = game.GLOBAL_MAP_COLS, game.GLOBAL_MAP_ROWS
        start = game.WINDOW_ROWS * game.WINDOW_COLS
        cw = max(1.0, game.level_cols / gc)
        ch = max(1.0, game.level_rows / gr)
        ex = min(gc - 1, int(game.exit_pos[0] / cw))
        ey = min(gr - 1, int(game.exit_pos[1] / ch))
        # Guard against a co-located objective masking the exit cell, mirroring the
        # enabled-path test, so this verifies the isolated exit cell stays empty.
        occupied = set()
        for c, r in game.crystals | (game.switches - game.used_switches):
            occupied.add((min(gc - 1, int(c / cw)), min(gr - 1, int(r / ch))))
        if (ex, ey) in occupied:
            pytest.skip("exit shares a coarse cell with an objective in this cave")
        state = game.get_state()
        gmap = state[start : start + gc * gr]
        # With the flag off the locked exit must be fully hidden: neither the locked
        # marker (0.2) nor the unlocked marker (0.9) — the cell stays empty.
        assert gmap[ey * gc + ex] == pytest.approx(0.0)

    def test_locked_exit_visible_when_flag_enabled(self, config: Config) -> None:
        config.CRYSTAL_CAVES_SHOW_LOCKED_EXIT = True
        game = CrystalCaves(config, headless=True)
        # Move the exit to an empty cell so no crystal/switch overrides its marker.
        game.crystals = {c for c in game.crystals}
        gc, gr = game.GLOBAL_MAP_COLS, game.GLOBAL_MAP_ROWS
        start = game.WINDOW_ROWS * game.WINDOW_COLS
        cw = max(1.0, game.level_cols / gc)
        ch = max(1.0, game.level_rows / gr)
        ex = min(gc - 1, int(game.exit_pos[0] / cw))
        ey = min(gr - 1, int(game.exit_pos[1] / ch))
        # Skip if an objective shares the exit's coarse cell (it would dominate).
        occupied = set()
        for c, r in game.crystals | (game.switches - game.used_switches):
            occupied.add((min(gc - 1, int(c / cw)), min(gr - 1, int(r / ch))))
        if (ex, ey) in occupied:
            pytest.skip("exit shares a coarse cell with an objective in this cave")
        state = game.get_state()
        gmap = state[start : start + gc * gr]
        assert gmap[ey * gc + ex] == pytest.approx(0.2)

    def test_new_closest_approach_to_crystal_gets_nonfarmable_bonus(self, game):
        game.switches.clear()
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE - game.TILE_SIZE * 4
        game.player_y = crystal[1] * game.TILE_SIZE

        target, distance = game._current_target()
        assert target is not None and target[0] == "crystal"
        assert game._target_best_approach_reward(target, distance) == 0.0

        game.player_x += game.TILE_SIZE
        _, closer_distance = game._current_target()
        reward = game._target_best_approach_reward(target, closer_distance)
        assert reward > 0.0

        game.player_x -= game.TILE_SIZE
        _, farther_distance = game._current_target()
        assert game._target_best_approach_reward(target, farther_distance) == 0.0

    def test_anti_loop_penalty_fires_when_stuck_without_approach(self, config):
        config.CRYSTAL_CAVES_ANTI_LOOP_REWARD = True
        game = CrystalCaves(config, headless=True)
        game.enemies.clear()
        game.hazards.clear()
        place_on_floor(game)

        rewards = [game.step(CrystalCaves.IDLE)[1] for _ in range(65)]
        info = game._info()

        assert min(rewards) < -0.03
        assert info["anti_loop_penalty_total"] < 0.0

    def test_novelty_bonus_rewards_new_region_once(self, config):
        config.CRYSTAL_CAVES_NOVELTY_BONUS = True
        game = CrystalCaves(config, headless=True)

        try:
            start_cell = game._novelty_cell()
            assert game._novelty_region_reward() == 0.0

            for row in range(1, game.level_rows - 1):
                for col in range(1, game.level_cols - 1):
                    game.player_x = col * game.TILE_SIZE + 5
                    game.player_y = row * game.TILE_SIZE + 1
                    if game._novelty_cell() != start_cell:
                        reward = game._novelty_region_reward()
                        assert reward == game.NOVELTY_REGION_BONUS
                        assert game._novelty_region_reward() == 0.0
                        info = game._info()
                        assert info["novelty_bonus_total"] == game.NOVELTY_REGION_BONUS
                        return
            raise AssertionError("No different novelty cell found")
        finally:
            game.close()

    def test_novelty_bonus_is_capped_below_crystal_reward(self, game):
        assert game.NOVELTY_REGION_CAP < 5.0

    def test_move_left_decreases_x(self, game):
        place_on_floor(game)
        initial_x = game.player_x
        game.step(CrystalCaves.LEFT)
        assert game.player_x < initial_x

    def test_jump_sets_upward_velocity(self, game):
        place_on_floor(game)
        game.step(CrystalCaves.JUMP)
        assert game.vy < 0

    def test_solid_wall_blocks_player(self, game):
        game.player_x = game.TILE_SIZE + 1
        game.player_y = (game.level_rows - 2) * game.TILE_SIZE
        game.vx = -20
        game._move_player()
        assert game.player_x >= game.TILE_SIZE


class TestCrystalCavesObjectives:
    """Test crystals, switches, doors, and exits."""

    def test_collecting_crystal_increases_score(self, game):
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE + 4
        game.player_y = crystal[1] * game.TILE_SIZE + 2
        initial_count = len(game.crystals)
        _, reward, done, info = game.step(CrystalCaves.IDLE)
        assert len(game.crystals) == initial_count - 1
        assert not done
        assert reward > 0
        assert info["score"] >= 100

    def test_first_crystal_goal_ends_episode_on_collection(self, config):
        config.CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL = True
        game = CrystalCaves(config, headless=True)
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE + 4
        game.player_y = crystal[1] * game.TILE_SIZE + 2
        initial_count = len(game.crystals)

        _, reward, done, info = game.step(CrystalCaves.IDLE)

        assert len(game.crystals) == initial_count - 1
        assert done
        assert info["won"]
        assert info["end_reason"] == "first_crystal_goal"
        assert reward >= game.FIRST_CRYSTAL_GOAL_BONUS

    def test_all_crystals_unlock_exit(self, game):
        game.crystals.clear()
        game.exit_unlocked = False
        _, reward, _, info = game.step(CrystalCaves.IDLE)
        assert game.exit_unlocked
        assert info["exit_unlocked"]
        assert reward >= game.ALL_CRYSTALS_COLLECTED_BONUS - 1.0

    def test_exit_unlock_emits_visual_feedback(self, config):
        game = CrystalCaves(config, headless=False)
        game.crystals.clear()
        game.exit_unlocked = False

        game.step(CrystalCaves.IDLE)

        assert game.exit_unlocked
        assert any(event.text == "EXIT OPEN" for event in game.visual_events)

    def test_headless_pickups_do_not_store_visual_events(self, game):
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE + 4
        game.player_y = crystal[1] * game.TILE_SIZE + 2

        game.step(CrystalCaves.IDLE)

        assert game.visual_events == []

    def test_reaching_unlocked_exit_wins(self, game):
        game.crystals.clear()
        game.exit_unlocked = True
        game.player_x = game.exit_pos[0] * game.TILE_SIZE + 5
        game.player_y = game.exit_pos[1] * game.TILE_SIZE + 1
        _, reward, done, info = game.step(CrystalCaves.IDLE)
        assert done
        assert info["won"]
        assert reward >= 20

    def test_switch_opens_doors(self, game):
        switch = next(iter(game.switches))
        game.player_x = switch[0] * game.TILE_SIZE + 4
        game.player_y = switch[1] * game.TILE_SIZE + 2
        _, reward, _, info = game.step(CrystalCaves.INTERACT)
        assert game.doors_open
        assert info["doors_open"]
        assert reward > 0

    def test_invalid_interact_has_no_penalty_by_default(self, game):
        game.switches.clear()

        reward = game._try_interact()
        info = game._info()

        assert reward == 0.0
        assert info["invalid_interact_count"] == 0
        assert info["invalid_interact_penalty_total"] == 0.0

    def test_invalid_interact_penalty_tracks_useless_interactions(self, config):
        config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = True
        game = CrystalCaves(config, headless=True)
        game.switches.clear()

        reward = game._try_interact()
        info = game._info()

        assert reward == game.INVALID_INTERACT_PENALTY
        assert info["invalid_interact_count"] == 1
        assert info["invalid_interact_penalty_total"] == game.INVALID_INTERACT_PENALTY

    def test_invalid_interact_penalty_keeps_valid_switch_reward(self, config):
        config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = True
        game = CrystalCaves(config, headless=True)
        switch = next(iter(game.switches))
        game.player_x = switch[0] * game.TILE_SIZE + 4
        game.player_y = switch[1] * game.TILE_SIZE + 2

        reward = game._try_interact()
        info = game._info()

        assert reward == game.SWITCH_THROW_BONUS
        assert info["doors_open"]
        assert info["invalid_interact_count"] == 0

    def test_colour_keyed_lever_opens_only_its_own_door(self):
        """A red lever opens only red doors; the blue door stays shut until the
        blue lever is thrown (authentic colour-keyed locks)."""
        from src.game.crystal_caves_gen import DOOR2, generate_cave

        spec = None
        for seed in range(60):
            cand = generate_cave(seed, "blue_rock", "platform_network", difficulty="normal")
            if DOOR2 in "".join(cand.layout):
                spec = cand
                break
        assert spec is not None, "expected a two-lock level"
        cfg = Config()
        cfg.CRYSTAL_CAVES_PROCEDURAL = True
        g = CrystalCaves(cfg, headless=True)
        g._randomize_levels = False  # pin to the specific two-lock cave under test
        g.CAVES = (spec,) + g.CAVES[1:]
        g.level_index = 0
        g.level = spec
        g.reset()
        red_sw = next(s for s, c in g.switch_color.items() if c == "red")
        red_door = next(d for d, c in g.door_color.items() if c == "red")
        blue_door = next(d for d, c in g.door_color.items() if c == "blue")
        g.player_x = red_sw[0] * g.TILE_SIZE + 4
        g.player_y = red_sw[1] * g.TILE_SIZE + 2
        g.step(CrystalCaves.INTERACT)
        assert g._door_open(red_door)
        assert not g._door_open(blue_door)  # blue lock untouched
        assert not g.doors_open  # not all doors open yet

    def test_elevator_carries_player_up_and_down(self, game):
        """A player standing on an elevator platform is carried as it oscillates
        between the top and bottom of its shaft (no input needed)."""
        ts = game.TILE_SIZE
        col = 5
        # carve an open vertical shaft so the player can ride without obstruction
        for row in range(4, 12):
            game.grid[row][col] = game.EMPTY
        game.elevators = [Elevator(col=col, top=4, bottom=10, pos=10.0, direction=-1)]
        game._refresh_elevator_rects()
        game.player_x = col * ts + 4
        game.player_y = 10 * ts - game.PLAYER_HEIGHT  # standing on the platform
        feet = []
        for _ in range(260):
            game.step(CrystalCaves.IDLE)
            feet.append(game.player_y + game.PLAYER_HEIGHT)
        # the player must have travelled most of the shaft and reversed direction
        assert max(feet) - min(feet) > ts * 3
        assert min(feet) <= 5 * ts  # rode up near the top
        assert max(feet) >= 10 * ts - ts  # and back down near the bottom

    @pytest.mark.parametrize("climb_tile", [CrystalCaves.LADDER, CrystalCaves.ELEVATOR])
    def test_ladder_and_elevator_rails_are_climbable(self, config, climb_tile):
        layout = tuple(
            row.replace("H", climb_tile)
            for row in (
                "########",
                "#..H..*#",
                "#..H...#",
                "#..H..E#",
                "#..H...#",
                "#P.H...#",
                "#..H...#",
                "########",
            )
        )
        spec = CaveSpec("climb test", layout, (0, 0, 0), (255, 255, 255))
        game = CrystalCaves(config, headless=True)
        game.level = spec
        game._load_level(spec)
        game.player_x = 3 * game.TILE_SIZE + 5
        game.player_y = 5 * game.TILE_SIZE + 1
        start_y = game.player_y

        for _ in range(8):
            game.step(CrystalCaves.JUMP)
        climbed_y = game.player_y

        for _ in range(8):
            game.step(CrystalCaves.IDLE)

        assert climbed_y < start_y - 12
        assert game.player_y > climbed_y + 8


class TestCrystalCavesCombatAndDanger:
    """Test enemies, bullets, ammo, hazards, and damage."""

    def test_shooting_enemy_kills_it(self, game):
        floor_col, floor_row = place_on_floor(game)
        clear_lane(game, floor_col, floor_row)
        game.facing = 1
        game.ammo = 5
        enemy = Enemy(game.player_x + 90, game.player_y + 4, 0.0)
        game.enemies = [enemy]

        game.step(CrystalCaves.SHOOT)
        for _ in range(16):
            game.step(CrystalCaves.IDLE)

        assert not enemy.alive
        assert game.score >= 200

    def test_invalid_shoot_penalty_counts_empty_lane_shot(self, config):
        config.CRYSTAL_CAVES_INVALID_SHOOT_PENALTY = True
        game = CrystalCaves(config, headless=True)
        floor_col, floor_row = place_on_floor(game)
        clear_lane(game, floor_col, floor_row)
        game.enemies = []
        game.air_tanks = set()
        game.facing = 1
        game.ammo = 5

        _, _, _, info = game.step(CrystalCaves.SHOOT)

        assert info["invalid_shoot_count"] == 1
        assert info["invalid_shoot_penalty_total"] == pytest.approx(
            CrystalCaves.INVALID_SHOOT_PENALTY
        )

    def test_invalid_shoot_penalty_allows_enemy_lane_shot(self, config):
        config.CRYSTAL_CAVES_INVALID_SHOOT_PENALTY = True
        game = CrystalCaves(config, headless=True)
        floor_col, floor_row = place_on_floor(game)
        clear_lane(game, floor_col, floor_row)
        game.facing = 1
        game.ammo = 5
        game.enemies = [Enemy(game.player_x + 90, game.player_y + 4, 0.0)]

        _, _, _, info = game.step(CrystalCaves.SHOOT)

        assert info["invalid_shoot_count"] == 0
        assert info["invalid_shoot_penalty_total"] == 0.0

    def test_hazard_damages_player(self, game):
        hazard = next(iter(game.hazards))
        game.player_x = hazard[0] * game.TILE_SIZE + 4
        game.player_y = hazard[1] * game.TILE_SIZE + 2
        _, reward, _, info = game.step(CrystalCaves.IDLE)
        assert info["health"] == game.MAX_HEALTH - 1
        assert reward < 0

    def test_ammo_pickup_increases_ammo(self, game):
        ammo = next(iter(game.ammo_pickups))
        game.ammo = 0
        game.player_x = ammo[0] * game.TILE_SIZE + 4
        game.player_y = ammo[1] * game.TILE_SIZE + 2
        _, reward, _, info = game.step(CrystalCaves.IDLE)
        assert info["ammo"] >= 5
        assert reward > 0

    def test_shooting_air_tank_penalizes_and_removes_tank(self, game):
        floor_col, floor_row = place_on_floor(game)
        clear_lane(game, floor_col, floor_row)
        tank = (floor_col + 3, floor_row - 1)
        game.air_tanks = {tank}
        game.facing = 1
        game.ammo = 5

        total_reward = 0.0
        _, reward, _, _ = game.step(CrystalCaves.SHOOT)
        total_reward += reward
        for _ in range(12):
            _, reward, _, _ = game.step(CrystalCaves.IDLE)
            total_reward += reward
            if tank not in game.air_tanks:
                break

        assert tank not in game.air_tanks
        assert total_reward < -1.0


class TestCrystalCavesPowerupsAndTreasure:
    """Power-ups (super shot, gravity flip, freeze) and treasure pickups."""

    @staticmethod
    def _put_pickup_under_player(game):
        """Return the player's current tile (an open, touchable tile)."""
        return game._player_tile()

    def test_power_shot_sets_super_timer(self, game):
        tile = self._put_pickup_under_player(game)
        game.powerups = {tile: game.POWER_SHOT}
        game.super_timer = 0
        reward = game._collect_pickups()
        assert game.super_timer == game.MAX_POWER_TIMER
        assert reward > 0
        assert tile not in game.powerups

    def test_gravity_power_flips_gravity(self, game):
        tile = self._put_pickup_under_player(game)
        game.powerups = {tile: game.GRAVITY_POWER}
        before = game.gravity_dir
        game._collect_pickups()
        assert game.gravity_dir == -before
        assert game.gravity_timer > 0

    def test_gravity_field_restores_after_timer(self, game):
        game.gravity_dir = -1
        game.gravity_timer = 1
        game.step(CrystalCaves.IDLE)  # ticks the gravity timer to 0
        assert game.gravity_dir == 1  # field restored to normal

    def test_freeze_power_sets_timer_and_holds_enemies(self, game):
        from src.game.crystal_caves_entities import Enemy

        tile = self._put_pickup_under_player(game)
        game.powerups = {tile: game.FREEZE_POWER}
        game._collect_pickups()
        assert game.freeze_timer > 0
        # a frozen enemy does not move
        enemy = Enemy(game.player_x + 120, game.player_y, 1.2)
        game.enemies = [enemy]
        x0 = enemy.x
        game._update_enemies()
        assert enemy.x == x0  # frozen in place

    def test_treasure_pickup_rewards_and_scores(self, game):
        tile = self._put_pickup_under_player(game)
        game.treasures = {tile}
        score0 = game.score
        reward = game._collect_pickups()
        assert reward > 0
        assert game.score == score0 + 300
        assert tile not in game.treasures

    def test_damage_triggers_screen_shake(self, game):
        """Taking damage kicks the camera (render-only juice) without affecting
        training state."""
        game.invuln_timer = 0
        game._damage_player()
        assert game.shake_timer == game.SHAKE_FRAMES
        shaken = game._camera()
        game.shake_timer = 0
        assert shaken != game._camera()  # the shake actually moved the camera


class TestCrystalCavesRenderAndVectorized:
    """Test render and vectorized training surfaces."""

    def test_render_does_not_crash(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=False)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render(surface)

    def test_agent_view_overlay_renders(self, config):
        """The educational agent-view overlay (perception window + goal compass)
        renders without error when enabled."""
        pygame.init()
        game = CrystalCaves(config, headless=False)
        game.reset()
        game.show_agent_overlay = True
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render(surface)  # must not raise

    def test_render_outputs_high_contrast_pixel_art(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=False)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render(surface)

        pixels = pygame.surfarray.array3d(surface)
        non_black_pixels = np.count_nonzero(np.any(pixels > 0, axis=2))
        bright_green_pixels = np.count_nonzero((pixels[:, :, 1] > 220) & (pixels[:, :, 0] < 140))
        bright_blue_pixels = np.count_nonzero((pixels[:, :, 2] > 180) & (pixels[:, :, 0] < 150))
        warm_prop_pixels = np.count_nonzero(
            (pixels[:, :, 0] > 150) & (pixels[:, :, 1] > 80) & (pixels[:, :, 2] < 90)
        )

        assert non_black_pixels > config.SCREEN_WIDTH * config.SCREEN_HEIGHT * 0.08
        assert bright_green_pixels > 500
        assert bright_blue_pixels > 500
        assert warm_prop_pixels > 150

    def test_render_contains_pixel_art_hud_and_player_palette(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=False)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render(surface)

        pixels = pygame.surfarray.array3d(surface)
        hud = pixels[:, config.SCREEN_HEIGHT - game.HUD_HEIGHT :, :]
        playfield = pixels[:, : config.SCREEN_HEIGHT - game.HUD_HEIGHT, :]

        hud_green = np.count_nonzero(
            (hud[:, :, 1] > 220) & (hud[:, :, 0] < 120) & (hud[:, :, 2] < 140)
        )
        hud_yellow = np.count_nonzero(
            (hud[:, :, 0] > 220) & (hud[:, :, 1] > 180) & (hud[:, :, 2] < 120)
        )
        player_pink = np.count_nonzero(
            (playfield[:, :, 0] > 220) & (playfield[:, :, 1] < 120) & (playfield[:, :, 2] > 120)
        )
        player_yellow = np.count_nonzero(
            (playfield[:, :, 0] > 220) & (playfield[:, :, 1] > 180) & (playfield[:, :, 2] < 120)
        )

        assert hud_green > 300
        assert hud_yellow > 80
        assert player_pink > 20
        assert player_yellow > 40

    def test_b_grade_sprite_catalog_has_signature_visuals(self):
        assert {
            "mylo_shoot",
            "mylo_hurt",
            "slug_enemy",
            "bat_enemy",
            "eye_turret",
            "clear_block",
            "walking_rock",
            "eye_flyer",
        }.issubset(SPRITES)

    def test_title_screen_renders_dos_front_end(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=False)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render_title_screen(surface)

        pixels = pygame.surfarray.array3d(surface)
        cyan_title_pixels = np.count_nonzero((pixels[:, :, 2] > 180) & (pixels[:, :, 1] > 160))
        yellow_instruction_pixels = np.count_nonzero(
            (pixels[:, :, 0] > 220) & (pixels[:, :, 1] > 170) & (pixels[:, :, 2] < 120)
        )
        bordered_panel_pixels = np.count_nonzero(np.all(pixels > 180, axis=2))

        assert cyan_title_pixels > 1200
        assert yellow_instruction_pixels > 450
        assert bordered_panel_pixels > 250

    def test_title_screen_renders_from_headless_game(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=True)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

        game.render_title_screen(surface)

        pixels = pygame.surfarray.array3d(surface)
        non_black_pixels = np.count_nonzero(np.any(pixels > 0, axis=2))
        cyan_title_pixels = np.count_nonzero((pixels[:, :, 2] > 180) & (pixels[:, :, 1] > 160))

        assert non_black_pixels > config.SCREEN_WIDTH * config.SCREEN_HEIGHT * 0.12
        assert cyan_title_pixels > 1200

    def test_first_cave_uses_authored_visual_dressing(self, game):
        pieces = CrystalCaves.CAVE_DRESSING[0]
        kinds = {piece.kind for piece in pieces}

        assert len(pieces) >= 12
        assert {
            "beacon",
            "mine_sign",
            "generator",
            "terminal",
            "warning_post",
            "vacuum",
            "zapper",
            "elevator_frame",
            "clear_blocks",
            "room_label",
            "eye_turret",
            "bat_perch",
        }.issubset(kinds)

    def test_collecting_crystal_emits_visual_feedback(self, config):
        game = CrystalCaves(config, headless=False)
        crystal = next(iter(game.crystals))
        game.player_x = crystal[0] * game.TILE_SIZE + 4
        game.player_y = crystal[1] * game.TILE_SIZE + 2

        game.step(CrystalCaves.IDLE)

        assert any(event.kind == "sparkle" and event.text == "+100" for event in game.visual_events)

    def test_visual_events_expire_deterministically(self, config):
        game = CrystalCaves(config, headless=False)
        game._add_visual_event("spark", 20, 20, ttl=2, text="POP")

        assert len(game.visual_events) == 1
        game._update_visual_events()
        assert len(game.visual_events) == 1
        game._update_visual_events()
        assert game.visual_events == []

    def test_hazard_corridor_keeps_readable_danger_language(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=False)
        hazard = min(game.hazards, key=lambda tile: (tile[1], tile[0]))
        game.player_x = max(0, hazard[0] * game.TILE_SIZE - 120)
        game.player_y = max(0, (hazard[1] - 2) * game.TILE_SIZE)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

        game.render(surface)

        pixels = pygame.surfarray.array3d(surface)
        playfield = pixels[:, : config.SCREEN_HEIGHT - game.HUD_HEIGHT, :]
        red_warning_pixels = np.count_nonzero(
            (playfield[:, :, 0] > 180) & (playfield[:, :, 1] < 110) & (playfield[:, :, 2] < 130)
        )
        yellow_alert_pixels = np.count_nonzero(
            (playfield[:, :, 0] > 220) & (playfield[:, :, 1] > 170) & (playfield[:, :, 2] < 130)
        )
        white_spike_pixels = np.count_nonzero(np.all(playfield > 190, axis=2))

        assert red_warning_pixels > 900
        assert yellow_alert_pixels > 900
        assert white_spike_pixels > 1200

    def test_support_rails_are_not_dominant_in_authored_cave(self, game):
        first_cave_rails = [
            (col, row)
            for row in range(game.level_rows)
            for col in range(game.level_cols)
            if game._should_draw_support_rail(col, row)
        ]

        game.level_index = 1
        game.reset()
        later_cave_rails = [
            (col, row)
            for row in range(game.level_rows)
            for col in range(game.level_cols)
            if game._should_draw_support_rail(col, row)
        ]

        assert first_cave_rails == []
        assert len(later_cave_rails) < 16

    def test_hud_is_clean_period_footer(self, config):
        """Play HUD is a label-less period footer (CCV-04/05): a compact bar with
        no compartment dividers, green score numerals, and red heart pips."""
        pygame.init()
        game = CrystalCaves(config, headless=False)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render(surface)

        pixels = pygame.surfarray.array3d(surface)
        hud_y = config.SCREEN_HEIGHT - game.HUD_HEIGHT

        # Compact period footer height.
        assert game.HUD_HEIGHT == 38

        # No bright vertical compartment dividers in the HUD body.
        separator_columns = [118, 230, 374, 594, 690]
        visible_separators = 0
        for column in separator_columns:
            column_pixels = pixels[column, hud_y + 6 : hud_y + game.HUD_HEIGHT, :]
            bright = np.count_nonzero(np.all(column_pixels > 150, axis=1))
            if bright > 20:
                visible_separators += 1
        assert visible_separators == 0

        # Green score numerals and red heart pips are present.
        hud = pixels[:, hud_y:, :]
        hud_green = np.count_nonzero(
            (hud[:, :, 1] > 200) & (hud[:, :, 0] < 120) & (hud[:, :, 2] < 140)
        )
        hud_red = np.count_nonzero(
            (hud[:, :, 0] > 200) & (hud[:, :, 1] < 120) & (hud[:, :, 2] < 120)
        )
        assert hud_green > 300
        assert hud_red > 40

    def test_background_fill_kills_the_black_void(self, config):
        """Every episode renders a dense themed back-wall: the play area must not
        be dominated by pure-black void (CCV-08/01, plan metric < 0.10)."""
        pygame.init()
        for episode in range(3):
            game = CrystalCaves(config, headless=False)
            game.level_index = episode
            game.reset()
            surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            game.render(surface)

            pixels = pygame.surfarray.array3d(surface)
            play = pixels[:, : config.SCREEN_HEIGHT - game.HUD_HEIGHT, :]
            total = play.shape[0] * play.shape[1]
            pure_black = np.count_nonzero(np.all(play == 0, axis=2))
            ratio = pure_black / total
            # Residual black is only outlines/sprites; the themed fill covers the
            # rest. Generous ceiling guards against regressing to a void backdrop.
            assert ratio < 0.15, f"episode {episode} black-void ratio {ratio:.3f}"

    def test_episodes_have_distinct_dominant_hues(self, config):
        """Each episode keeps a distinct palette register (CCV-02): blue / rust /
        gray-tech, measured by the mean color of the back-wall fill."""
        pygame.init()
        means = []
        for episode in range(3):
            game = CrystalCaves(config, headless=False)
            game.level_index = episode
            game.reset()
            surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            game.render(surface)
            pixels = pygame.surfarray.array3d(surface).astype(np.float64)
            play = pixels[:, : config.SCREEN_HEIGHT - game.HUD_HEIGHT, :]
            means.append(play.reshape(-1, 3).mean(axis=0))

        blue, rust, tech = means
        # Episode 0 reads blue (blue channel leads); episode 1 reads warm/rust
        # (red channel leads its own blue).
        assert blue[2] > blue[0]
        assert rust[0] > rust[2]
        # The three registers are measurably different from one another.
        assert np.linalg.norm(blue - rust) > 12
        assert np.linalg.norm(blue - tech) > 12

    def test_platform_trim_does_not_dominate_first_cave(self, config):
        pygame.init()
        game = CrystalCaves(config, headless=False)
        surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        game.render(surface)

        pixels = pygame.surfarray.array3d(surface)
        playfield = pixels[:, : config.SCREEN_HEIGHT - game.HUD_HEIGHT, :]
        bright_green_pixels = np.count_nonzero(
            (playfield[:, :, 1] > 220) & (playfield[:, :, 0] < 130) & (playfield[:, :, 2] < 170)
        )
        blue_metal_pixels = np.count_nonzero(
            (playfield[:, :, 2] > 100) & (playfield[:, :, 0] < 130) & (playfield[:, :, 1] > 60)
        )

        assert bright_green_pixels < blue_metal_pixels
        assert bright_green_pixels < config.SCREEN_WIDTH * config.SCREEN_HEIGHT * 0.025

    def test_gallery_renderer_writes_review_screenshots(self, tmp_path):
        paths = render_gallery(tmp_path / "crystal_gallery")

        assert [path.name for path in paths] == [
            "01_title.png",
            "02_start.png",
            "03_crystal_pocket.png",
            "04_pickup_sparkle.png",
            "05_switch_room.png",
            "06_hazard_corridor.png",
            "07_air_tank.png",
            "08_enemy_room.png",
            "09_shooting_frame.png",
            "10_exit_open.png",
            "11_episode2_amber.png",
            "12_episode3_moon.png",
        ]
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_vec_env_shapes(self, config):
        size = CrystalCaves(config, headless=True).state_size
        vec = VecCrystalCaves(3, config, headless=True)
        states = vec.reset()
        assert states.shape == (3, size)

        actions = np.array([CrystalCaves.IDLE, CrystalCaves.RIGHT, CrystalCaves.JUMP])
        next_states, rewards, dones, infos = vec.step(actions)
        assert next_states.shape == (3, size)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3


class TestReverseCurriculum:
    """Mid-solution reverse-curriculum starts (solvability-preserving, training-only)."""

    def test_off_by_default_keeps_full_start(self, config: Config) -> None:
        assert config.CRYSTAL_CAVES_REVERSE_CURRICULUM is False
        game = CrystalCaves(config, headless=True)
        game.reset()
        assert len(game.crystals) == game.initial_crystals

    def test_pre_collects_subset_and_opens_gates(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 1.0
        game = CrystalCaves(config, headless=True)
        game.reset()

        # A strict, non-empty subset of crystals remains...
        assert 0 < len(game.crystals) < game.initial_crystals
        # ...all gates are opened so the kept objectives stay reachable...
        assert game.used_switches == set(game.switches)
        assert game.open_colors == set(game.door_color.values())
        # ...and the exit is locked iff crystals remain (here: still locked).
        assert game.exit_unlocked == (len(game.crystals) == 0)
        # State stays well-formed and normalized.
        state = game.get_state()
        assert np.all(np.isfinite(state))
        assert np.all(state >= 0.0) and np.all(state <= 1.0)

    def test_skipped_in_eval_mode(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 1.0
        game = CrystalCaves(config, headless=True)
        game._eval_mode = True  # eval must always measure the full task
        game.reset()
        assert len(game.crystals) == game.initial_crystals

    def test_p_zero_keeps_full_start(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 0.0
        game = CrystalCaves(config, headless=True)
        game.reset()
        assert len(game.crystals) == game.initial_crystals

    def test_set_reverse_curriculum_p_clamps(self, config: Config) -> None:
        game = CrystalCaves(config, headless=True)
        game.set_reverse_curriculum_p(2.0)
        assert game._reverse_curriculum_p == 1.0
        game.set_reverse_curriculum_p(-1.0)
        assert game._reverse_curriculum_p == 0.0


class TestReverseCurriculumAnnealSchedule:
    """The pure linear-anneal schedule for the reverse-curriculum probability."""

    def test_starts_at_start_p(self) -> None:
        assert reverse_curriculum_p_for_episode(0.5, 0, 100) == pytest.approx(0.5)

    def test_zero_at_and_after_anneal_episodes(self) -> None:
        assert reverse_curriculum_p_for_episode(0.5, 100, 100) == pytest.approx(0.0)
        assert reverse_curriculum_p_for_episode(0.5, 250, 100) == pytest.approx(0.0)

    def test_monotonically_decreasing_in_between(self) -> None:
        values = [reverse_curriculum_p_for_episode(0.8, ep, 100) for ep in range(0, 101, 10)]
        assert all(later <= earlier for earlier, later in zip(values, values[1:]))
        # Strictly decreasing while annealing (no flat plateau before the end).
        assert all(later < earlier for earlier, later in zip(values[:-1], values[1:]))

    def test_midpoint_is_half_start_p(self) -> None:
        assert reverse_curriculum_p_for_episode(0.6, 50, 100) == pytest.approx(0.3)

    def test_anneal_episodes_zero_keeps_constant(self) -> None:
        for ep in (0, 5, 1000):
            assert reverse_curriculum_p_for_episode(0.5, ep, 0) == pytest.approx(0.5)

    def test_config_validates_non_negative_anneal_episodes(self) -> None:
        cfg = Config()
        cfg.CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES = -1
        with pytest.raises(ValueError, match="ANNEAL_EPISODES"):
            cfg.__post_init__()


class TestReverseCurriculumTrainerHook:
    """The trainer's per-episode hook anneals every env's reverse-curriculum p."""

    def _make_trainer(self, config: Config, vec_envs: int) -> "HeadlessTrainer":
        config.GAME_NAME = "crystal_caves"
        config.MAX_EPISODES = 1
        config.EVAL_EVERY = 0  # skip the evaluator (no extra game instance / I/O)
        config.USE_NOISY_NETWORKS = False
        args = SimpleNamespace(
            lr=None,
            episodes=None,
            learn_every=None,
            gradient_steps=None,
            batch_size=None,
            torch_compile=False,
            turbo=False,
            vec_envs=vec_envs,
            model=None,
            web=False,
            port=0,
            host="127.0.0.1",
        )
        return HeadlessTrainer(config, args)

    def test_hook_anneals_each_vec_env_over_episodes(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 0.5
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES = 10
        trainer = self._make_trainer(config, vec_envs=4)
        assert trainer.vec_env is not None

        trainer.current_episode = 0
        trainer._apply_reverse_curriculum_schedule()
        assert all(env._reverse_curriculum_p == pytest.approx(0.5) for env in trainer.vec_env.envs)

        trainer.current_episode = 5
        trainer._apply_reverse_curriculum_schedule()
        assert all(env._reverse_curriculum_p == pytest.approx(0.25) for env in trainer.vec_env.envs)

        trainer.current_episode = 10
        trainer._apply_reverse_curriculum_schedule()
        assert all(env._reverse_curriculum_p == pytest.approx(0.0) for env in trainer.vec_env.envs)

    def test_hook_is_noop_when_anneal_disabled(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 0.5
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_ANNEAL_EPISODES = 0
        trainer = self._make_trainer(config, vec_envs=2)
        assert trainer.vec_env is not None

        for env in trainer.vec_env.envs:
            env.set_reverse_curriculum_p(0.5)
        trainer.current_episode = 100
        trainer._apply_reverse_curriculum_schedule()
        # p left constant (no annealing) when ANNEAL_EPISODES == 0.
        assert all(env._reverse_curriculum_p == pytest.approx(0.5) for env in trainer.vec_env.envs)

    def test_hook_is_noop_for_non_crystal_game(self) -> None:
        cfg = Config()
        cfg.GAME_NAME = "breakout"
        cfg.MAX_EPISODES = 1
        cfg.EVAL_EVERY = 0
        args = SimpleNamespace(
            lr=None,
            episodes=None,
            learn_every=None,
            gradient_steps=None,
            batch_size=None,
            torch_compile=False,
            turbo=False,
            vec_envs=1,
            model=None,
            web=False,
            port=0,
            host="127.0.0.1",
        )
        trainer = HeadlessTrainer(cfg, args)
        # No crystal envs -> empty list, hook does nothing and must not raise.
        assert trainer._crystal_caves_envs() == []
        trainer._apply_reverse_curriculum_schedule()


class TestReverseCurriculumRelocation:
    """Oracle-verified player relocation toward the remaining objectives (#4 follow-up)."""

    def _vanilla_spawn_tile(self) -> tuple[int, int]:
        game = CrystalCaves(Config(), headless=True)
        game._randomize_levels = False
        game.level_index = 0
        game.reset()
        return game._player_tile()

    def test_relocation_off_keeps_spawn(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 1.0
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_RELOCATE = False
        game = CrystalCaves(config, headless=True)
        game._randomize_levels = False
        game.level_index = 0
        game.reset()
        assert game._player_tile() == self._vanilla_spawn_tile()

    def test_relocation_keeps_objectives_oracle_reachable(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 1.0
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_RELOCATE = True
        game = CrystalCaves(config, headless=True)
        game._randomize_levels = False
        game.level_index = 0
        game.reset()

        # Safety invariant: from the relocated start, every remaining crystal AND the
        # exit are still reachable under the jump-aware oracle (start stays solvable).
        targets = set(game.crystals) | {game.exit_pos}
        assert targets <= game._oracle_reachable(game._player_tile())

    def test_relocation_does_not_move_farther_from_objectives(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_P = 1.0
        config.CRYSTAL_CAVES_REVERSE_CURRICULUM_RELOCATE = True
        game = CrystalCaves(config, headless=True)
        game._randomize_levels = False
        game.level_index = 0
        game.reset()

        targets = set(game.crystals) | {game.exit_pos}

        def nearest_sq(tile: tuple[int, int]) -> int:
            return min((tile[0] - oc) ** 2 + (tile[1] - orow) ** 2 for oc, orow in targets)

        spawn = self._vanilla_spawn_tile()
        # The relocated start is at least as close to an objective as the spawn.
        assert nearest_sq(game._player_tile()) <= nearest_sq(spawn)


class TestReverseExitCurriculum:
    """Reverse-EXIT curriculum: post-collection start near the open exit (leg-2 drill)."""

    def test_off_keeps_crystals(self, config: Config) -> None:
        assert config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM is False
        game = CrystalCaves(config, headless=True)
        game.reset()
        assert len(game.crystals) > 0
        assert game.exit_unlocked is False

    def test_on_clears_crystals_and_unlocks_exit(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P = 1.0
        game = CrystalCaves(config, headless=True)
        game.reset()
        assert len(game.crystals) == 0
        assert game.exit_unlocked is True
        # Post-collection world state: every gate is open.
        assert game.used_switches == set(game.switches)

    def test_start_can_reach_exit_oracle(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P = 1.0
        game = CrystalCaves(config, headless=True)
        for _ in range(8):
            game.reset()
            # Safety invariant: the open exit is jump-aware reachable from the start.
            assert game.exit_pos in game._oracle_reachable(game._player_tile())

    def test_eval_mode_disables_curriculum(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P = 1.0
        game = CrystalCaves(config, headless=True)
        game._eval_mode = True
        game.reset()
        # Eval starts are always full-from-scratch (curriculum is training-only).
        assert len(game.crystals) > 0
        assert game.exit_unlocked is False

    def test_p_zero_is_noop(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P = 0.0
        game = CrystalCaves(config, headless=True)
        game.reset()
        assert len(game.crystals) > 0

    def test_far_variant_starts_distant_and_reachable(self, config: Config) -> None:
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM = True
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P = 1.0
        config.CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_FAR = True
        game = CrystalCaves(config, headless=True)
        for _ in range(8):
            game.reset()
            assert len(game.crystals) == 0 and game.exit_unlocked is True
            col, row = game._player_tile()
            # FAR drill: a real distance from the exit, but still oracle-reachable.
            exit_col, exit_row = game.exit_pos
            assert (
                abs(col - exit_col) + abs(row - exit_row)
                >= game.REVERSE_EXIT_CURRICULUM_FAR_MIN_DIST
            )
            assert game.exit_pos in game._oracle_reachable((col, row))


class TestGeoCompass:
    """Geodesic next-step corridor compass (RUN-11 navigation fix)."""

    def test_off_by_default_no_size_change(self, config: Config) -> None:
        assert config.CRYSTAL_CAVES_GEO_COMPASS is False
        base = CrystalCaves(config, headless=True).state_size
        cfg2 = Config()
        cfg2.CRYSTAL_CAVES_GEO_COMPASS = True
        assert (
            CrystalCaves(cfg2, headless=True).state_size == base + CrystalCaves.GEO_COMPASS_FEATURES
        )

    def test_compass_points_down_the_route(self, config: Config) -> None:
        config.CRYSTAL_CAVES_GEO_COMPASS = True
        game = CrystalCaves(config, headless=True)
        for _ in range(6):
            game.reset()
            field = game._geodesic_distance_field()
            pcol, prow = game._player_tile()
            here = field.get((pcol, prow))
            if here is None or here == 0:
                continue
            step_dx, step_dy, reachable, dist_norm = game._geodesic_next_step_compass()
            assert reachable == 1.0
            assert 0.0 <= dist_norm <= 1.0
            # The suggested step must strictly reduce the route distance to the objective.
            nd = field.get((pcol + int(step_dx), prow + int(step_dy)))
            assert nd is not None and nd < here

    def test_unreachable_returns_zero_reachable(self, config: Config) -> None:
        config.CRYSTAL_CAVES_GEO_COMPASS = True
        game = CrystalCaves(config, headless=True)
        game.reset()
        # An empty field (no objective tiles in it) -> reachable flag is 0.
        game._geodesic_field = {}
        game._geodesic_field_key = (game._active_target_tiles(), frozenset(game.open_colors))
        step_dx, step_dy, reachable, _ = game._geodesic_next_step_compass()
        assert (step_dx, step_dy, reachable) == (0.0, 0.0, 0.0)


class TestHazardAwareCompass:
    """RUN-17 survival lever: the compass routes AROUND static hazards, not through them."""

    @staticmethod
    def _stub_corridor(game, *, hazard_aware: bool):
        """Two parallel 5-wide corridors (rows 0 and 1) joined at every column. Objective at
        (4,0); a hazard sits at (2,0) on the direct top-row route, with the bottom row a free
        detour. Player at (1,0), one tile west of the hazard."""
        free = {(c, r) for c in range(5) for r in range(2)}
        game._solid_at = lambda c, r: (c, r) not in free
        game._active_target_tiles = lambda: frozenset({(4, 0)})
        game._player_tile = lambda: (1, 0)
        game.hazards = {(2, 0)}
        game.open_colors = set()
        game.level_cols, game.level_rows = 5, 2
        game._geo_compass_hazard_aware = hazard_aware
        game._hazard_field = None
        game._hazard_field_key = None
        game._geodesic_field = None
        game._geodesic_field_key = None

    def test_plain_compass_steps_into_hazard(self, config):
        # Control: the hazard-blind compass takes the shortest route, straight onto (2,0).
        game = CrystalCaves(config, headless=True)
        self._stub_corridor(game, hazard_aware=False)
        step_dx, step_dy, reachable, _ = game._geodesic_next_step_compass()
        assert reachable == 1.0
        assert (step_dx, step_dy) == (1.0, 0.0)  # east, onto the hazard at (2,0)

    def test_hazard_aware_compass_detours_around_hazard(self, config):
        # Arm B: the hazard-aware compass detours down to the free bottom row instead.
        config.CRYSTAL_CAVES_GEO_COMPASS = True
        config.CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE = True
        game = CrystalCaves(config, headless=True)
        self._stub_corridor(game, hazard_aware=True)
        step_dx, step_dy, reachable, _ = game._geodesic_next_step_compass()
        assert reachable == 1.0
        # Leaves the hazard top row (dy=+1) and lands on a non-hazard detour tile, rather
        # than stepping east onto (2,0) like the plain compass does.
        assert step_dy == 1.0
        landing = (1 + int(step_dx), 0 + int(step_dy))
        assert landing not in game.hazards
        assert (step_dx, step_dy) != (1.0, 0.0)

    def test_hazard_cost_raises_route_distance(self, config):
        # The hazard-weighted field must cost >= the plain field everywhere, and strictly
        # more at the player tile where the cheap route crosses the hazard.
        config.CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE = True
        game = CrystalCaves(config, headless=True)
        self._stub_corridor(game, hazard_aware=True)
        plain = game._geodesic_distance_field()
        weighted = game._hazard_aware_distance_field()
        for tile, d in plain.items():
            assert weighted[tile] >= d - 1e-9
        assert weighted[(1, 0)] > plain[(1, 0)]

    def test_no_hazards_matches_plain_route(self, config):
        # Degenerate safety: with no hazards the hazard-aware field equals the plain field,
        # so arm B reduces exactly to the proven compass.
        config.CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE = True
        game = CrystalCaves(config, headless=True)
        self._stub_corridor(game, hazard_aware=True)
        game.hazards = set()
        game._hazard_field = None
        plain = game._geodesic_distance_field()
        weighted = game._hazard_aware_distance_field()
        assert {k: float(v) for k, v in plain.items()} == {k: float(v) for k, v in weighted.items()}

    def test_off_by_default_no_size_change(self, config):
        # The lever adds no state dims (same 4 compass scalars) and is off by default.
        assert config.CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE is False
        base = CrystalCaves(config, headless=True)
        base_compass = Config()
        base_compass.CRYSTAL_CAVES_GEO_COMPASS = True
        haz = Config()
        haz.CRYSTAL_CAVES_GEO_COMPASS = True
        haz.CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE = True
        assert (
            CrystalCaves(haz, headless=True).state_size
            == CrystalCaves(base_compass, headless=True).state_size
        )
        assert base.state_size <= CrystalCaves(haz, headless=True).state_size


class TestRouteAux:
    """Op 2: learnable geodesic route via an auxiliary head + trailing label slots."""

    def test_off_by_default(self, config: Config) -> None:
        assert getattr(config, "CRYSTAL_CAVES_ROUTE_AUX_GEODESIC", False) is False
        game = CrystalCaves(config, headless=True)
        assert game._route_label_dims == 0
        assert game.config.STATE_LAYOUT["route_label"] == 0

    def test_adds_trailing_label_slots(self, config: Config) -> None:
        base = CrystalCaves(config, headless=True).state_size
        cfg2 = Config()
        cfg2.CRYSTAL_CAVES_ROUTE_AUX_GEODESIC = True
        game = CrystalCaves(cfg2, headless=True)
        assert game.state_size == base + CrystalCaves.GEO_COMPASS_FEATURES
        assert game.config.STATE_LAYOUT["route_label"] == CrystalCaves.GEO_COMPASS_FEATURES
        # the trailing slots ARE the geodesic compass (label-only; meta size is unchanged)
        game.reset()
        state = game.get_state()
        trailing = state[-CrystalCaves.GEO_COMPASS_FEATURES :]
        assert np.allclose(trailing, np.array(game._geodesic_next_step_compass(), dtype=np.float32))

    def test_policy_is_route_blind(self, config: Config) -> None:
        """The network must slice the label off — scrambling it cannot change Q-values."""
        import torch

        from src.ai.network import DuelingDQN

        config.CRYSTAL_CAVES_ROUTE_AUX_GEODESIC = True
        config.CRYSTAL_CAVES_ROUTE_AUX_LOSS = True
        game = CrystalCaves(config, headless=True)
        game.reset()
        net = DuelingDQN(state_size=game.state_size, action_size=game.action_size, config=config)
        assert net.core_in == game.state_size - CrystalCaves.GEO_COMPASS_FEATURES
        assert hasattr(net, "route_aux_head")
        t = torch.tensor(game.get_state().astype(np.float32)).unsqueeze(0)
        q = net(t).detach().clone()
        scrambled = t.clone()
        scrambled[:, -CrystalCaves.GEO_COMPASS_FEATURES :] = torch.randn(
            CrystalCaves.GEO_COMPASS_FEATURES
        )
        assert torch.allclose(q, net(scrambled).detach(), atol=1e-6)


class TestNGUBonus:
    """NGU-style episodic novelty bonus (#5)."""

    def test_off_returns_zero(self, config: Config) -> None:
        assert config.CRYSTAL_CAVES_NGU_BONUS is False
        game = CrystalCaves(config, headless=True)
        game.reset()
        assert game._ngu_bonus() == 0.0

    def test_decays_with_revisits(self, config: Config) -> None:
        config.CRYSTAL_CAVES_NGU_BONUS = True
        config.CRYSTAL_CAVES_NGU_BETA = 0.1
        game = CrystalCaves(config, headless=True)
        game.reset()

        first = game._ngu_bonus()
        second = game._ngu_bonus()
        third = game._ngu_bonus()
        assert first == pytest.approx(0.1)
        assert second == pytest.approx(0.1 / (2**0.5))
        assert third == pytest.approx(0.1 / (3**0.5))
        assert first > second > third

    def test_progress_change_is_novel_again(self, config: Config) -> None:
        config.CRYSTAL_CAVES_NGU_BONUS = True
        config.CRYSTAL_CAVES_NGU_BETA = 0.1
        game = CrystalCaves(config, headless=True)
        game.reset()

        assert game._ngu_bonus() == pytest.approx(0.1)  # first visit at this state
        # Simulate collecting a crystal: the (pos x progress) key changes -> novel.
        game.crystals.pop()
        assert game._ngu_bonus() == pytest.approx(0.1)

    def test_reset_clears_visit_counts(self, config: Config) -> None:
        config.CRYSTAL_CAVES_NGU_BONUS = True
        config.CRYSTAL_CAVES_NGU_BETA = 0.1
        game = CrystalCaves(config, headless=True)
        game.reset()
        game._ngu_bonus()
        game.reset()
        # After a fresh episode the same state is novel again.
        assert game._ngu_bonus() == pytest.approx(0.1)

    def test_step_reward_includes_ngu_bonus(self) -> None:
        """The bonus must actually flow through the public step() reward, not just
        _ngu_bonus(). Same level + action + RNG, so the only delta is the bonus."""

        def first_step_reward(ngu_on: bool) -> float:
            cfg = Config()
            cfg.CRYSTAL_CAVES_NGU_BONUS = ngu_on
            cfg.CRYSTAL_CAVES_NGU_BETA = 0.1
            game = CrystalCaves(cfg, headless=True)
            game._randomize_levels = False
            game.level_index = 0
            np.random.seed(0)
            game.reset()
            np.random.seed(0)  # match any per-step RNG across both runs
            _, reward, _, _ = game.step(CrystalCaves.IDLE)
            return reward

        reward_off = first_step_reward(False)
        reward_on = first_step_reward(True)
        # First visit of the post-step (tile x progress) cell -> beta / sqrt(1).
        assert reward_on == pytest.approx(reward_off + 0.1)

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
    def test_invalid_beta_rejected(self, value: float) -> None:
        cfg = Config()
        cfg.CRYSTAL_CAVES_NGU_BETA = value
        with pytest.raises(ValueError, match="CRYSTAL_CAVES_NGU_BETA"):
            cfg.__post_init__()


class TestDeathSourceAttribution:
    """Audit B6: hazard vs enemy death attribution must be independent (no hazard bias)."""

    def _attribute(self, game: CrystalCaves) -> str:
        game.invuln_timer = 0
        game.health = 3
        game._last_damage_source = "none"
        game._check_player_danger()
        return game._last_damage_source

    def test_independent_hazard_enemy_and_both_bucket(self, game: CrystalCaves) -> None:
        import pygame

        game.reset()
        player_rect = game._player_rect()
        pcol, prow = game._player_tile()

        class _FakeEnemy:
            alive = True

            def __init__(self, rect: pygame.Rect) -> None:
                self.rect = rect

        # hazard only
        game.hazards = {(pcol, prow)}
        game.enemies = []
        assert self._attribute(game) == "hazard"

        # enemy only
        game.hazards = set()
        game.enemies = [_FakeEnemy(pygame.Rect(player_rect))]
        assert self._attribute(game) == "enemy"

        # BOTH overlap on the same frame — pre-fix this was forced to 'hazard'.
        game.hazards = {(pcol, prow)}
        game.enemies = [_FakeEnemy(pygame.Rect(player_rect))]
        assert self._attribute(game) == "both"


class TestTerminalPrecedence:
    """Audit B7 / latent-2 / B9: first terminal event in a frame wins; correct level label."""

    def test_b7_terminal_win_not_overwritten_by_same_frame_damage(self, game: CrystalCaves) -> None:
        game.reset()
        game.game_over = True  # e.g. a FIRST_CRYSTAL_GOAL win fired in _collect_pickups
        game.won = True
        game._end_reason = "first_crystal_goal"
        pcol, prow = game._player_tile()
        game.hazards = {(pcol, prow)}  # a fatal hazard overlaps the same frame
        game.health = 1
        game.invuln_timer = 0
        game._check_player_danger()
        assert game.won is True
        assert game._end_reason == "first_crystal_goal"  # pre-fix: overwritten to "killed"

    def test_latent2_death_not_overwritten_by_same_frame_exit(self, game: CrystalCaves) -> None:
        game.reset()
        game.game_over = True  # a fatal hit fired earlier this frame
        game.won = False
        game._end_reason = "killed"
        game.exit_unlocked = True
        ex, ey = game.exit_pos
        game.player_x = float(ex * game.TILE_SIZE)
        game.player_y = float(ey * game.TILE_SIZE)
        game._check_exit()
        assert game.won is False
        assert game._end_reason == "killed"  # pre-fix: overwritten to "won"

    def test_b9_won_episode_reports_played_level_not_next(self, game: CrystalCaves) -> None:
        game.reset()
        played = game.level_index
        game.exit_unlocked = True
        ex, ey = game.exit_pos
        game.player_x = float(ex * game.TILE_SIZE)
        game.player_y = float(ey * game.TILE_SIZE)
        game._check_exit()
        assert game.won is True
        assert game.level_index != played  # internal cursor advanced
        assert game._info()["level"] == played  # but the report shows the PLAYED level


class TestStallTimerNetProgress:
    """Audit B8: the stall timer resets only on NET progress, not instantaneous oscillation."""

    def test_oscillation_does_not_reset_stall_timer(self, game: CrystalCaves, monkeypatch) -> None:
        game.reset()
        game.game_over = False
        target = ("crystal", 5, 5)
        game._stall_best = {target: 90.0}

        def drive(current_distance: float, prev_distance: float) -> None:
            monkeypatch.setattr(game, "_current_target", lambda: (target, current_distance))
            game._target_progress_reward(target, prev_distance)

        # Moves CLOSER than the previous step (instantaneous progress) but NOT past the
        # closest-ever 90 -> pre-fix this reset the timer every wiggle; now it must not.
        game.steps_since_progress = 50
        drive(current_distance=95.0, prev_distance=99.0)
        assert game.steps_since_progress == 50

        # A genuine new closest-ever approach DOES reset the timer.
        drive(current_distance=80.0, prev_distance=95.0)
        assert game.steps_since_progress == 0


class TestReInteractNotFarmable:
    """Audit R2-A: re-interacting an already-thrown switch must not pay positive reward."""

    def test_reinteract_used_switch_is_noop(self, game: CrystalCaves) -> None:
        game.reset()
        pcol, prow = game._player_tile()
        game.switches = {(pcol, prow)}
        game.used_switches = {(pcol, prow)}  # already thrown
        game.config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = False
        assert game._try_interact() == 0.0  # pre-fix: +0.05 (farmable)

    def test_reinteract_penalized_when_flag_on(self, game: CrystalCaves) -> None:
        game.reset()
        pcol, prow = game._player_tile()
        game.switches = {(pcol, prow)}
        game.used_switches = {(pcol, prow)}
        game.config.CRYSTAL_CAVES_INVALID_INTERACT_PENALTY = True
        assert game._try_interact() < 0.0
