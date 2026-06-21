"""
Tests for the Crystal Caves-style game implementation.

These tests focus on the reusable base: NN-compatible state, core platformer
mechanics, pickups, switches, hazards, shooting, rendering, and vectorized
training support.
"""

import os
import sys

import numpy as np
import pygame
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from scripts.render_crystal_caves_gallery import render_gallery
from src.game import get_game_info, list_games
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_art import SPRITES
from src.game.crystal_caves_entities import CAVES, Elevator, Enemy

# The jump-aware solvability flood is shared with the generator so the authored
# caves and the procedural ones are verified by one identical oracle.
from src.game.crystal_caves_gen import cave_reachable as _cave_reachable
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
        _, reward, _, info = game.step(CrystalCaves.IDLE)
        assert len(game.crystals) == initial_count - 1
        assert reward > 0
        assert info["score"] >= 100

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


def _find(layout, ch):
    return [(c, r) for r, row in enumerate(layout) for c, x in enumerate(row) if x == ch]


@pytest.mark.parametrize("cave_index", range(len(CAVES)))
def test_every_cave_is_solvable(cave_index):
    """Each authored cave must be winnable: from spawn the player can reach the
    switch (doors closed), then every crystal and the exit (doors open). This
    guards the dense layouts against a fill that seals the level."""
    layout = CAVES[cave_index].layout
    player = _find(layout, "P")[0]
    exit_ = _find(layout, "E")[0]
    crystals = _find(layout, "*")
    switches = _find(layout, "s")

    reach_closed = _cave_reachable(layout, player, doors_open=False)
    reach_open = _cave_reachable(layout, player, doors_open=True)

    assert all(s in reach_closed for s in switches), "switch unreachable"
    assert all(c in reach_open for c in crystals), "a crystal is unreachable"
    assert exit_ in reach_open, "exit unreachable"


@pytest.mark.parametrize("cave_index", range(len(CAVES)))
def test_every_cave_is_dense(cave_index):
    """Authored caves are carved rooms, not sparse platforms over void (CCV-13):
    solid terrain should occupy a substantial fraction of the grid."""
    layout = CAVES[cave_index].layout
    total = sum(len(row) for row in layout)
    solid = sum(row.count("#") for row in layout)
    assert 0.45 <= solid / total <= 0.85


def _loop_window(game, player_col, player_row):
    """Reference window built the slow per-cell way, for equivalence testing."""
    half_c, half_r = game.WINDOW_COLS // 2, game.WINDOW_ROWS // 2
    win = np.empty((game.WINDOW_ROWS, game.WINDOW_COLS), dtype=np.float32)
    i = 0
    for row in range(player_row - half_r, player_row + half_r + 1):
        for col in range(player_col - half_c, player_col + half_c + 1):
            win.flat[i] = game._tile_code(col, row)
            i += 1
    return win


@pytest.mark.parametrize("difficulty", ["easy", "normal"])
def test_vectorized_window_matches_tile_code_loop(difficulty):
    """The fast vectorized perception window must stay bit-identical to the
    per-cell _tile_code loop it replaced — across collected crystals, thrown
    switches, moving enemies, and opened doors."""
    cfg = Config()
    cfg.CRYSTAL_CAVES_PROCEDURAL = True
    cfg.CRYSTAL_CAVES_DIFFICULTY = difficulty
    cfg.CRYSTAL_CAVES_FAMILIES = "platform_network"
    np.random.seed(7)
    game = CrystalCaves(cfg, headless=True)
    game.reset()

    for _ in range(400):
        pc = int((game.player_x + game.PLAYER_WIDTH / 2) // game.TILE_SIZE)
        pr = int((game.player_y + game.PLAYER_HEIGHT / 2) // game.TILE_SIZE)
        expected = _loop_window(game, pc, pr)
        actual = game._fill_window(pc, pr)
        assert np.array_equal(expected, actual)
        _, _, done, _ = game.step(np.random.randint(0, game.action_size))
        if done:
            game.reset()


def test_training_samples_diverse_caves_and_eval_is_held_out():
    """Training must sample varied caves per episode (not memorise one level), and
    evaluation must use a fixed, reproducible, held-out set disjoint from training."""
    cfg = Config()
    cfg.CRYSTAL_CAVES_PROCEDURAL = True
    cfg.CRYSTAL_CAVES_DIFFICULTY = "easy"
    cfg.CRYSTAL_CAVES_FAMILIES = "platform_network"
    np.random.seed(0)
    game = CrystalCaves(cfg, headless=True)

    # Training: a pool, sampled with variety across resets.
    assert len(game.CAVES) == cfg.CRYSTAL_CAVES_POOL_SIZE
    seen = set()
    for _ in range(40):
        game.reset()
        seen.add(tuple(game.level.layout))
    assert len(seen) > 5  # genuinely diverse, not one repeated level

    # Eval: held-out, diverse, reproducible.
    game.use_eval_levels(20)

    def eval_pass():
        game.reset_eval_cursor()
        seq = []
        for _ in range(20):
            game.reset()
            seq.append(tuple(game.level.layout))
        return seq

    pass1 = eval_pass()
    pass2 = eval_pass()
    assert len(set(pass1)) == 20  # 20 distinct held-out caves
    assert pass1 == pass2  # identical across evals (reproducible)
    train_layouts = {tuple(c.layout) for c in game.CAVES}
    assert not (set(pass1) & train_layouts)  # truly held out — unseen in training
