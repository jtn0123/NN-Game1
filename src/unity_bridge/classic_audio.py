"""Render Crystal Caves' original PC-speaker tone programs for the Unity player.

The shareware SND data is kept as source material, not sampled from a narrated
video. Frequencies, pauses, vibrato gating and priorities belong to each event.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
# DOS PIT divisor 0x2147 from the original sound-service implementation.
TONE_RATE = 1193181 / 0x2147
SOURCE = Path(__file__).resolve().parents[2] / "unity/ArtSource/ClassicSounds"
EVENT_INDEX = {
    "jump": 0,
    "enter": 2,
    "win": 3,
    "lose": 4,
    "enemy_defeat": 5,
    "enemy_hurt": 6,
    "gem": 7,
    "ammo": 8,
    "treasure": 10,
    "pickup": 11,
    "shoot": 13,
    "power_shoot": 14,
    "freeze": 17,
    "switch": 19,
    "door": 20,
    "gravity": 21,
    "stalactite": 24,
    "empty": 29,
    "thorn": 30,
    "damage": 30,
}


@dataclass(frozen=True)
class SpeakerSound:
    frequencies: tuple[int, ...]
    priority: int
    vibrate: int


def read_programs(source: Path = SOURCE) -> list[SpeakerSound]:
    programs = []
    for index in range(1, 4):
        raw = (source / f"CC1-{index}.SND").read_bytes()
        if len(raw) != 12 * 610:
            raise ValueError("Crystal Caves sound files must contain twelve 610-byte records")
        for offset in range(0, len(raw), 610):
            fields = struct.unpack_from("<300h5H", raw, offset)
            tones = fields[:300]
            if -1 not in tones:
                raise ValueError("unterminated Crystal Caves tone program")
            end = tones.index(-1)
            if any(frequency < 0 for frequency in tones[:end]) or fields[302] == 0:
                raise ValueError("invalid Crystal Caves tone program")
            programs.append(SpeakerSound(tuple(tones[:end]), fields[300], fields[302]))
    return programs


def render(program: SpeakerSound) -> np.ndarray:
    count = round(len(program.frequencies) * SAMPLE_RATE / TONE_RATE)
    output = np.zeros(count, dtype=np.float64)
    for index, frequency in enumerate(program.frequencies):
        start = round(index * SAMPLE_RATE / TONE_RATE)
        stop = round((index + 1) * SAMPLE_RATE / TONE_RATE)
        if not frequency or index % program.vibrate:
            continue
        # Square-wave PC-speaker pulses, including the original rapid gating.
        pulses = np.floor(np.arange(stop - start) * (2 * frequency / SAMPLE_RATE)) % 2
        output[start:stop] = np.where(pulses == 0, 0.23, -0.23)
    return output


def sound_bank() -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    programs = read_programs()
    clips = {name: render(programs[index]) for name, index in EVENT_INDEX.items()}
    metadata = {
        name: {"source_index": index, "priority": programs[index].priority}
        for name, index in EVENT_INDEX.items()
    }
    # Ordinary landing has no invented bass thud in the classic soundscape.
    clips["land"] = np.zeros(round(SAMPLE_RATE * 0.01), dtype=np.float64)
    metadata["land"] = {"source_index": -1, "priority": 0}
    return clips, metadata
