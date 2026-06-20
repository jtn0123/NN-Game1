"""
Tests for the procedural Crystal Caves audio engine.

These tests must be green in CI with no real audio device. They run under
``SDL_VIDEODRIVER=dummy`` (set below as a safety net), which is exactly how the
training/gallery/CI environments invoke the game. In that mode the engine is
expected to disable itself and treat every public method as a safe no-op.
"""

import os

# Ensure the headless audio path even if pytest is invoked without the env var.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.game.crystal_caves_audio import (  # noqa: E402
    SFX_NAMES,
    CrystalCavesAudio,
)

EXPECTED_SFX = {
    "pickup",
    "gem",
    "shoot",
    "jump",
    "land",
    "switch",
    "door",
    "damage",
    "gravity",
    "win",
    "lose",
}


def test_disabled_engine_is_inert() -> None:
    """An explicitly disabled engine reports disabled and never raises."""
    audio = CrystalCavesAudio(enabled=False)
    assert audio.enabled is False
    # Must not raise.
    audio.play("pickup")
    audio.play("gem")
    audio.start_music()
    audio.stop_music()


def test_dummy_driver_forces_disabled() -> None:
    """Under the dummy video driver, enabled=True still ends up disabled.

    Every play/music call must be a safe no-op.
    """
    assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
    audio = CrystalCavesAudio(enabled=True)
    assert audio.enabled is False

    for name in SFX_NAMES:
        audio.play(name)  # safe no-op
    audio.start_music()
    audio.stop_music()


def test_unknown_sound_never_raises() -> None:
    """Playing an unknown sound name is a silent no-op, never an error."""
    for enabled in (True, False):
        audio = CrystalCavesAudio(enabled=enabled)
        audio.play("nonexistent_sound")
        audio.play("")


def test_sfx_name_set_is_complete() -> None:
    """The published SFX name set matches the expected canonical set.

    Guards against accidental renames or dropped/added effects.
    """
    assert set(SFX_NAMES) == EXPECTED_SFX
    # Names should be unique and ordered consistently.
    assert len(SFX_NAMES) == len(set(SFX_NAMES))


def test_play_does_not_raise_for_every_name() -> None:
    """Calling play() for each canonical name is always safe."""
    audio = CrystalCavesAudio(enabled=True)
    for name in SFX_NAMES:
        audio.play(name)
