"""
Procedural chiptune audio for the Crystal Caves game.

This module synthesizes all of its sound effects at runtime with numpy --
there are no external audio assets. The voicing intentionally evokes the
1991 DOS-platformer aesthetic: short square/triangle/sine blips with snappy
envelopes and small arpeggios.

Headless safety is a first-class concern. The module is imported and used in
CI, the screenshot gallery, and training runs where there is no audio device
(``SDL_VIDEODRIVER=dummy``). In every one of those contexts the class degrades
to a fully inert no-op object: it never opens a mixer and ``play`` can never
raise. See :class:`CrystalCavesAudio` for the disable conditions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

# Sample rate used for both the mixer and the synthesized buffers.
_SAMPLE_RATE = 22050

# Canonical, ordered set of sound-effect names. Game code wires events to
# these; the test suite pins the set to guard against accidental renames.
SFX_NAMES: Tuple[str, ...] = (
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
)


class CrystalCavesAudio:
    """Self-contained procedural sound engine for Crystal Caves.

    The engine pre-generates a small library of chiptune sound effects and
    plays them on demand. It is safe to construct in any environment: if audio
    is unavailable it silently disables itself and every public method becomes
    a no-op.

    Args:
        enabled: Master switch. When ``False`` the engine stays disabled.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the engine, disabling itself if audio is unavailable.

        ``self.enabled`` ends up ``False`` (and no mixer is opened) when any of
        the following hold:

        * ``enabled`` is ``False``;
        * ``SDL_VIDEODRIVER`` is ``"dummy"`` (our headless/CI/training marker);
        * ``pygame.mixer.init(...)`` raises for any reason.
        """
        self.enabled: bool = False
        self._sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._channels: int = 1
        self._music_channel: Optional["pygame.mixer.Channel"] = None
        self._music_sound: Optional["pygame.mixer.Sound"] = None

        if not enabled:
            return
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=_SAMPLE_RATE, size=-16, channels=1, buffer=512)
            init = pygame.mixer.get_init()
            if not init:
                return
            self._channels = init[2]
            self._build_library()
        except Exception:
            # Any failure -> stay completely inert. Never let audio break the
            # game, the gallery, or a training run.
            self.enabled = False
            self._sounds = {}
            return

        self.enabled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def play(self, name: str) -> None:
        """Play a named sound effect.

        No-op if the engine is disabled or ``name`` is unknown. Fully guarded:
        this method can never raise.

        Args:
            name: One of :data:`SFX_NAMES`.
        """
        if not self.enabled:
            return
        sound = self._sounds.get(name)
        if sound is None:
            return
        try:
            channel = pygame.mixer.find_channel(True)
            if channel is not None:
                channel.play(sound)
            else:
                sound.play()
        except Exception:
            # Swallow everything: a missed sound must never crash the game.
            pass

    def start_music(self) -> None:
        """Start the looping background chiptune phrase.

        Quiet and tasteful. No-op when disabled; fully exception-safe.
        """
        if not self.enabled:
            return
        try:
            if self._music_sound is None:
                self._music_sound = self._make_sound(self._music_phrase())
            channel = self._music_channel
            if channel is None or not channel.get_busy():
                self._music_channel = self._music_sound.play(loops=-1)
                if self._music_channel is not None:
                    self._music_channel.set_volume(0.18)
        except Exception:
            pass

    def stop_music(self) -> None:
        """Stop the looping background music. No-op when disabled; safe."""
        if not self.enabled:
            return
        try:
            if self._music_channel is not None:
                self._music_channel.stop()
                self._music_channel = None
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sound-effect synthesis
    # ------------------------------------------------------------------
    def _build_library(self) -> None:
        """Synthesize every entry in :data:`SFX_NAMES` into the sound cache."""
        builders = {
            "pickup": self._sfx_pickup,
            "gem": self._sfx_gem,
            "shoot": self._sfx_shoot,
            "jump": self._sfx_jump,
            "land": self._sfx_land,
            "switch": self._sfx_switch,
            "door": self._sfx_door,
            "damage": self._sfx_damage,
            "gravity": self._sfx_gravity,
            "win": self._sfx_win,
            "lose": self._sfx_lose,
        }
        for name in SFX_NAMES:
            self._sounds[name] = self._make_sound(builders[name]())

    # -- individual effects --------------------------------------------
    def _sfx_pickup(self) -> np.ndarray:
        """Bright rising blip for collecting an item."""
        a = self._square(660.0, 0.06, volume=0.35)
        b = self._square(990.0, 0.06, volume=0.35)
        return self._envelope(np.concatenate([a, b]), attack=0.005, release=0.04)

    def _sfx_gem(self) -> np.ndarray:
        """Sparkly two-note shimmer for a gem/crystal."""
        a = self._sine(880.0, 0.07, volume=0.32)
        b = self._sine(1318.0, 0.10, volume=0.32)
        return self._envelope(np.concatenate([a, b]), attack=0.004, release=0.06)

    def _sfx_shoot(self) -> np.ndarray:
        """Short laser zap: fast downward frequency sweep."""
        wave = self._sweep(1200.0, 280.0, 0.12, volume=0.30, wave="square")
        return self._envelope(wave, attack=0.002, release=0.07)

    def _sfx_jump(self) -> np.ndarray:
        """Quick rising chirp."""
        wave = self._sweep(380.0, 760.0, 0.10, volume=0.32, wave="triangle")
        return self._envelope(wave, attack=0.003, release=0.05)

    def _sfx_land(self) -> np.ndarray:
        """Low thud for landing on the ground."""
        wave = self._sweep(220.0, 120.0, 0.10, volume=0.36, wave="triangle")
        return self._envelope(wave, attack=0.001, release=0.06)

    def _sfx_switch(self) -> np.ndarray:
        """Mechanical two-tone clunk for toggling a switch."""
        a = self._square(330.0, 0.05, volume=0.34)
        b = self._square(247.0, 0.07, volume=0.34)
        return self._envelope(np.concatenate([a, b]), attack=0.001, release=0.04)

    def _sfx_door(self) -> np.ndarray:
        """Heavier clunk/whoosh for a door opening."""
        clunk = self._square(180.0, 0.06, volume=0.36)
        whoosh = self._sweep(160.0, 300.0, 0.16, volume=0.28, wave="triangle")
        return self._envelope(np.concatenate([clunk, whoosh]), attack=0.002, release=0.10)

    def _sfx_damage(self) -> np.ndarray:
        """Harsh descending buzz for taking damage."""
        wave = self._sweep(440.0, 110.0, 0.22, volume=0.34, wave="square")
        # Add a little dissonant detune for grit.
        detune = self._sweep(466.0, 116.0, 0.22, volume=0.18, wave="square")
        return self._envelope(wave + detune, attack=0.001, release=0.12)

    def _sfx_gravity(self) -> np.ndarray:
        """Slow pitch sweep/whoosh for a gravity flip."""
        wave = self._sweep(160.0, 520.0, 0.40, volume=0.28, wave="sine")
        return self._envelope(wave, attack=0.02, release=0.18)

    def _sfx_win(self) -> np.ndarray:
        """Short ascending arpeggio fanfare for clearing the exit."""
        notes = [523.0, 659.0, 784.0, 1046.0]
        return self._arpeggio(notes, note_dur=0.09, volume=0.33, wave="square")

    def _sfx_lose(self) -> np.ndarray:
        """Descending sad arpeggio for death/failure."""
        notes = [523.0, 415.0, 330.0, 262.0]
        return self._arpeggio(notes, note_dur=0.12, volume=0.32, wave="triangle")

    # ------------------------------------------------------------------
    # Background music
    # ------------------------------------------------------------------
    def _music_phrase(self) -> np.ndarray:
        """Build a short, simple looping chiptune phrase (cave-y minor feel)."""
        # A minor pentatonic-ish wander, quiet square-wave melody.
        melody = [
            (440.0, 0.20),
            (523.0, 0.20),
            (659.0, 0.20),
            (523.0, 0.20),
            (587.0, 0.20),
            (440.0, 0.20),
            (330.0, 0.40),
            (0.0, 0.20),
        ]
        chunks: List[np.ndarray] = []
        for freq, dur in melody:
            if freq <= 0.0:
                chunks.append(self._silence(dur))
            else:
                note = self._square(freq, dur, volume=0.22)
                chunks.append(self._envelope(note, attack=0.01, release=0.05))
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # Low-level waveform helpers
    # ------------------------------------------------------------------
    def _times(self, duration: float) -> np.ndarray:
        """Return a time axis array for ``duration`` seconds."""
        count = max(1, int(_SAMPLE_RATE * duration))
        return np.linspace(0.0, duration, count, endpoint=False)

    def _sine(self, freq: float, duration: float, volume: float) -> np.ndarray:
        """Generate a sine wave."""
        t = self._times(duration)
        return volume * np.sin(2.0 * np.pi * freq * t)

    def _square(self, freq: float, duration: float, volume: float) -> np.ndarray:
        """Generate a square wave (classic chiptune timbre)."""
        t = self._times(duration)
        return volume * np.sign(np.sin(2.0 * np.pi * freq * t))

    def _triangle(self, freq: float, duration: float, volume: float) -> np.ndarray:
        """Generate a triangle wave."""
        t = self._times(duration)
        phase = (freq * t) % 1.0
        tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
        return volume * tri

    def _sweep(
        self,
        start_freq: float,
        end_freq: float,
        duration: float,
        volume: float,
        wave: str = "sine",
    ) -> np.ndarray:
        """Generate a wave whose frequency glides from start to end."""
        t = self._times(duration)
        if duration <= 0.0:
            return np.zeros(0, dtype=np.float64)
        freqs = np.linspace(start_freq, end_freq, t.shape[0])
        # Integrate instantaneous frequency to get continuous phase.
        phase = 2.0 * np.pi * np.cumsum(freqs) / _SAMPLE_RATE
        if wave == "square":
            raw = np.sign(np.sin(phase))
        elif wave == "triangle":
            frac = (phase / (2.0 * np.pi)) % 1.0
            raw = 2.0 * np.abs(2.0 * frac - 1.0) - 1.0
        else:
            raw = np.sin(phase)
        return volume * raw

    def _arpeggio(
        self,
        freqs: List[float],
        note_dur: float,
        volume: float,
        wave: str = "square",
    ) -> np.ndarray:
        """Concatenate enveloped notes into an arpeggio."""
        chunks: List[np.ndarray] = []
        for freq in freqs:
            if wave == "triangle":
                note = self._triangle(freq, note_dur, volume)
            elif wave == "sine":
                note = self._sine(freq, note_dur, volume)
            else:
                note = self._square(freq, note_dur, volume)
            chunks.append(self._envelope(note, attack=0.004, release=0.03))
        return np.concatenate(chunks)

    def _silence(self, duration: float) -> np.ndarray:
        """Return a block of silence."""
        return np.zeros(self._times(duration).shape[0], dtype=np.float64)

    def _envelope(self, wave: np.ndarray, attack: float, release: float) -> np.ndarray:
        """Apply a linear attack/release envelope to avoid clicks."""
        n = wave.shape[0]
        if n == 0:
            return wave
        env = np.ones(n, dtype=np.float64)
        attack_n = min(n, int(_SAMPLE_RATE * attack))
        release_n = min(n, int(_SAMPLE_RATE * release))
        if attack_n > 0:
            env[:attack_n] = np.linspace(0.0, 1.0, attack_n)
        if release_n > 0:
            env[-release_n:] = np.minimum(env[-release_n:], np.linspace(1.0, 0.0, release_n))
        return wave * env

    def _make_sound(self, wave: np.ndarray) -> "pygame.mixer.Sound":
        """Convert a float waveform in [-1, 1] into a pygame ``Sound``.

        The output buffer shape matches the mixer's channel count (mono vs
        stereo) reported by ``pygame.mixer.get_init()``.
        """
        clipped = np.clip(wave, -1.0, 1.0)
        samples = (clipped * 32767.0).astype(np.int16)
        if self._channels >= 2:
            samples = np.column_stack([samples] * self._channels)
        samples = np.ascontiguousarray(samples)
        return pygame.mixer.Sound(buffer=samples.tobytes())
