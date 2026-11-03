"""Native power feedback follows authoritative timers, including expiry/reset."""

from src.unity_bridge.session import CaveSession


def test_powered_shot_snapshot_reports_pickup_duration_and_paused_time():
    session = CaveSession()
    game = session.game
    game.player_x, game.player_y = 6 * 32 + 4, 21 * 32 + 1
    picked = session.handle({"op": "step", "actions": [0]})
    assert picked["super_timer"] == game.MAX_POWER_TIMER == 420
    assert session.handle({"op": "snapshot"})["super_timer"] == 420
    game.enemies, game.thorns, game.stalactites = [], [], []
    game.hazards.clear()
    for _ in range(420):
        expired = session.handle({"op": "step", "actions": [0]})
    assert expired["super_timer"] == 0 and expired["health"] == 3


def test_power_feedback_clears_after_reset_and_mine_entry():
    session = CaveSession()
    session.game.super_timer, session.game.freeze_timer = 40, 60
    assert session.snapshot()["super_timer"] == 40
    session.reset(0)
    assert session.snapshot()["super_timer"] == session.snapshot()["freeze_timer"] == 0
    mine = session.handle({"op": "mine", "cleared": []})
    assert mine["super_timer"] == mine["freeze_timer"] == 0
