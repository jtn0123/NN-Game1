"""Golden traces that pin the C# simulation to the Python reference, frame by frame.

Each trace is a list of protocol operations (exactly what Unity sends) and, for
every operation, a digest of the resulting snapshot. Full snapshots are stored
at regular checkpoints so a divergence can be explained field by field.

Canonical encoding (mirrored by ``unity/SimTests/Canonical.cs``):

* ``null``, ``true``, ``false``
* numbers: an integral value with magnitude below 2**53 is written as a decimal
  integer, so ``5`` and ``5.0`` agree; any other finite float is ``d`` followed
  by the 16 lower-case hex digits of its IEEE-754 bit pattern.
* strings are quoted verbatim and may not contain quotes, backslashes or
  control characters
* lists are ``[a,b]``; objects are ``{"k":v}`` with keys in ordinal order

The digest is the first 16 hex digits of the SHA-256 of that UTF-8 text.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import struct
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .session import CaveSession

TRACE_VERSION = 1
CHECKPOINT_EVERY = 60
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "unity/SimTests/Traces"

# Presentation-irrelevant or AI/training-only fields. The C# player has no
# policy, recorder or training budget, and effect ids only key Unity's pools.
EXCLUDED_FIELDS = frozenset(
    {
        "action",
        "action_labels",
        "ai_available",
        "demos_saved",
        "last_reward",
        "max_steps",
        "movement_profile",
        "policy_name",
        "q_values",
        "recording",
        "stall_limit",
        "stall_steps",
        "state_size",
        "total_reward",
        "training_limits",
    }
)


def canonical_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The comparable part of a protocol snapshot."""
    result = {key: value for key, value in snapshot.items() if key not in EXCLUDED_FIELDS}
    # Effects are keyed by id in Unity; their list order follows Python set
    # iteration for same-frame pickups, so compare them as a sorted multiset.
    effects = [
        {key: value for key, value in effect.items() if key != "id"}
        for effect in snapshot["effects"]
    ]
    result["effects"] = sorted(effects, key=encode)
    return result


def encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite number in snapshot: {value!r}")
        if number == int(number) and abs(number) < 2**53:
            return str(int(number))
        return "d" + struct.pack(">d", number).hex()
    if isinstance(value, str):
        if any(ch in value for ch in '"\\') or any(ord(ch) < 32 for ch in value):
            raise ValueError(f"string cannot be encoded canonically: {value!r}")
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(encode(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-16-be"))
        return "{" + ",".join(f'"{key}":{encode(value[key])}' for key in keys) + "}"
    raise TypeError(f"unsupported snapshot value: {type(value).__name__}")


def digest(snapshot: dict[str, Any]) -> str:
    text = encode(canonical_snapshot(snapshot))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def control(move: int, jump: bool, shoot: bool, interact: bool) -> dict[str, Any]:
    return {"move": move, "jump": jump, "shoot": shoot, "interact": interact}


def human_step(controls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {"op": "human_step", "controls": list(controls)}


def record(name: str, level: int, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay operations through a fresh session and fingerprint every result."""
    session = CaveSession(level=level)
    initial = session.snapshot()
    digests = []
    checkpoints: dict[str, Any] = {"-1": canonical_snapshot(initial)}
    for index, operation in enumerate(operations):
        snapshot = session.handle(json.loads(json.dumps(operation)))
        digests.append(digest(snapshot))
        if index % CHECKPOINT_EVERY == 0 or index == len(operations) - 1:
            checkpoints[str(index)] = canonical_snapshot(snapshot)
    return {
        "version": TRACE_VERSION,
        "name": name,
        "level": level,
        "initial_digest": digest(initial),
        "operations": compress(operations),
        "digests": digests,
        "checkpoints": checkpoints,
    }


def compress(operations: list[dict[str, Any]]) -> list[Any]:
    """Run-length encode single-frame human steps; keep other operations verbatim."""
    result: list[Any] = []
    for operation in operations:
        controls = operation.get("controls")
        if operation["op"] == "human_step" and controls is not None and len(controls) == 1:
            c = controls[0]
            key = [c["move"], int(c["jump"]), int(c["shoot"]), int(c["interact"])]
            if result and isinstance(result[-1], list) and result[-1][1:] == key:
                result[-1][0] += 1
            else:
                result.append([1, *key])
        else:
            result.append(operation)
    return result


def expand(operations: list[Any]) -> Iterator[dict[str, Any]]:
    for operation in operations:
        if isinstance(operation, list):
            count, move, jump, shoot, interact = operation
            for _ in range(count):
                yield human_step([control(move, bool(jump), bool(shoot), bool(interact))])
        else:
            yield operation


class Explorer:
    """Seeded human-like button mashing that keeps moving and pressing things."""

    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed)
        self.left = 0
        self.move = 1
        self.jump_left = 0

    def next(self) -> dict[str, Any]:
        r = self.random
        if self.left <= 0:
            self.move = r.choices((-1, 0, 1), (4, 1, 4))[0]
            self.left = r.randint(6, 90)
        self.left -= 1
        if self.jump_left <= 0 and r.random() < 0.04:
            self.jump_left = r.randint(1, 30)
        jump = self.jump_left > 0
        self.jump_left -= 1
        return control(self.move, jump, r.random() < 0.05, r.random() < 0.03)


def explore_cave(level: int, seed: int, frames: int, batched: bool = False) -> list[dict[str, Any]]:
    """Explore one cave, re-entering it after a death or win like the Unity menus."""
    explorer = Explorer(seed * 1009 + level)
    session = CaveSession(level=level)
    operations: list[dict[str, Any]] = []
    used = 0
    while used < frames:
        if session.game.game_over:
            operation: dict[str, Any] = {"op": "reset", "level": level}
            operations.append(operation)
            session.handle(operation)
            continue
        size = explorer.random.randint(1, 8) if batched else 1
        first = explorer.next()
        operation = human_step([first] * size)
        operations.append(operation)
        session.handle(json.loads(json.dumps(operation)))
        used += size
    return operations


def explore_mine(seed: int, frames: int) -> list[dict[str, Any]]:
    """Walk the main mine, enter caves through their doors and come back."""
    explorer = Explorer(seed * 7919)
    session = CaveSession(level=0)
    cleared = [1, 4]
    operations: list[dict[str, Any]] = [{"op": "mine", "cleared": cleared}]
    session.handle(operations[0])
    used = 0
    cave_frames = 0
    while used < frames:
        snapshot = session.snapshot()
        operation: dict[str, Any]
        if snapshot["realm"] == "mine" and snapshot["portal_level"] >= 0:
            operation = {"op": "reset", "level": snapshot["portal_level"]}
            cave_frames = 0
        elif snapshot["realm"] == "cave" and (snapshot["done"] or cave_frames > 240):
            operation = {"op": "mine", "cleared": cleared}
        else:
            operation = human_step([explorer.next()])
            used += 1
            cave_frames += snapshot["realm"] == "cave"
        operations.append(operation)
        session.handle(json.loads(json.dumps(operation)))
    return operations


def captured_routes() -> list[tuple[str, int, list[dict[str, Any]]]]:
    """Capture every human route the route regression tests replay."""
    import pytest

    from .human_controls import HumanControlsMixin

    recordings: dict[int, tuple[int, list[dict[str, Any]], Any]] = {}
    original = HumanControlsMixin.step_human

    def capture(self: Any, controls: Any) -> Any:
        if id(self) not in recordings or recordings[id(self)][2] is not self:
            recordings[id(self)] = (self.level_index, [], self)
        recordings[id(self)][1].append(
            human_step([control(controls.move, controls.jump, controls.shoot, controls.interact)])
        )
        return original(self, controls)

    tests = Path(__file__).resolve().parents[2] / "tests"
    files = sorted(str(path) for path in tests.glob("test_classic_*_route.py"))
    HumanControlsMixin.step_human = capture  # type: ignore[method-assign]
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:randomly", *files])
    finally:
        HumanControlsMixin.step_human = original  # type: ignore[method-assign]
    if code != 0:
        raise RuntimeError("route tests failed; refusing to record their inputs")
    # Keep the longest distinct route per cave plus every other long one.
    seen: set[tuple[int, str]] = set()
    routes: list[tuple[str, int, list[dict[str, Any]]]] = []
    for level, operations, _game in sorted(
        recordings.values(), key=lambda item: (item[0], -len(item[1]))
    ):
        key = (level, json.dumps(operations))
        if len(operations) < 300 or key in seen:
            continue
        seen.add(key)
        per_level = sum(route[1] == level for route in routes)
        if per_level < 2:
            routes.append((f"route_cave{level:02d}_{per_level}", level, operations))
    return routes


def coverage(trace_files: Iterable[Path]) -> dict[str, int]:
    """Count mechanics that the checkpoint-free digests cannot show by themselves."""
    counts: dict[str, int] = {}

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    for path in trace_files:
        trace = load(path)
        session = CaveSession(level=trace["level"])
        previous = session.snapshot()
        for operation in expand(trace["operations"]):
            snapshot = session.handle(operation)
            bump(f"op:{operation['op']}")
            for sound in snapshot["sounds"]:
                bump(f"sound:{sound}")
            for effect in snapshot["effects"]:
                if effect["ttl"] == effect["max_ttl"]:
                    bump(f"effect:{effect['kind']}")
            if snapshot["done"] and not previous["done"]:
                bump(f"end:{snapshot['end_reason'] or 'none'}")
            player = snapshot["player"]
            if snapshot["score"] < previous["score"] and snapshot["realm"] == previous["realm"]:
                bump("state:score_penalty")
            if player["gravity_dir"] < 0:
                bump("state:reversed_gravity")
            if player["climbing"]:
                bump("state:climbing")
            if snapshot["freeze_timer"] > 0:
                bump("state:frozen")
            if snapshot["super_timer"] > 0:
                bump("state:powered")
            if snapshot["realm"] == "mine":
                bump("state:mine")
            ids = {entity["id"]: entity for entity in snapshot["entities"]}
            if any(key.startswith("egg_") for key in ids):
                bump("state:bat_egg")
            if any(
                e["sprite"].startswith("green_thorn_") and e["sprite"] != "green_thorn_0"
                for e in ids.values()
            ):
                bump("state:thorn_grown")
            if any(e.get("hit") for e in ids.values()):
                bump("state:enemy_hit")
            if any(e["sprite"] == "secret_cache_empty" for e in ids.values()):
                bump("state:secret_revealed")
            if any(e["sprite"].endswith("_on") for e in ids.values()):
                bump("state:switch_used")
            if any(
                e.get("asleep") is False and e["sprite"] == "walking_rock" for e in ids.values()
            ):
                bump("state:rock_awake")
            for lift in (e for e in ids.values() if e["id"].startswith("lift_")):
                rect_top = lift["y"]
                if (
                    abs(player["y"] + 30 - rect_top) < 0.5
                    and abs(player["x"] + 11 - (lift["x"] + 16)) < 20
                ):
                    bump("state:riding_lift")
            previous = snapshot
    return dict(sorted(counts.items()))


# Mechanics every trace set must exercise, so parity cannot pass vacuously.
REQUIRED_COVERAGE = (
    "end:killed",
    "end:won",
    "op:human_step",
    "op:mine",
    "op:reset",
    "sound:ammo",
    "sound:empty",
    "sound:enemy_defeat",
    "sound:enemy_hurt",
    "sound:enter",
    "sound:power_shoot",
    "sound:shoot",
    "sound:stalactite",
    "sound:switch",
    "sound:thorn",
    "sound:treasure",
    "sound:win",
    "state:bat_egg",
    "state:climbing",
    "state:enemy_hit",
    "state:frozen",
    "state:mine",
    "state:powered",
    "state:reversed_gravity",
    "state:score_penalty",
    "state:secret_revealed",
    "state:thorn_grown",
    "state:riding_lift",
    "state:rock_awake",
    "state:switch_used",
)


# (mechanic, cave, explorer seed): freeze pickup at frame 1013, reversed
# gravity at 690 and a shot air machine at 210, all from normal spawn.
TARGETED_SEEDS = (("freeze", 7, 18), ("gravity", 4, 13), ("air_shot", 4, 11))


def write(trace: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{trace['name']}.json.gz"
    text = json.dumps(trace, separators=(",", ":"), sort_keys=True)
    # mtime=0 keeps the archive byte-identical across regenerations.
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return path


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        trace: dict[str, Any] = json.load(handle)
    return trace


def build_all(output: Path, frames: int, include_routes: bool) -> list[Path]:
    paths = []
    for old in output.glob("*.json.gz"):
        old.unlink()
    jobs: list[tuple[str, int, list[dict[str, Any]]]] = []
    for level in range(16):
        jobs.append((f"explore_cave{level:02d}_a", level, explore_cave(level, 1, frames)))
        jobs.append((f"explore_cave{level:02d}_b", level, explore_cave(level, 2, frames)))
    for level in (0, 5, 9, 13):
        jobs.append((f"batched_cave{level:02d}", level, explore_cave(level, 3, frames, True)))
    # Pinned seeds found by search; each reaches a mechanic the broad runs miss.
    for name, level, seed in TARGETED_SEEDS:
        jobs.append((f"targeted_{name}", level, explore_cave(level, seed, frames)))
    jobs.append(("explore_mine_a", 0, explore_mine(4, frames * 2)))
    if include_routes:
        jobs.extend(captured_routes())
    for name, level, operations in jobs:
        paths.append(write(record(name, level, operations), output))
        print(f"{name}: {len(operations)} operations", flush=True)
    return paths


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=2400)
    parser.add_argument("--no-routes", action="store_true")
    parser.add_argument("--coverage", action="store_true", help="only report coverage")
    args = parser.parse_args(argv)
    if not args.coverage:
        build_all(args.output, args.frames, not args.no_routes)
    counts = coverage(sorted(args.output.glob("*.json.gz")))
    for key, count in counts.items():
        print(f"  {key}: {count}")
    missing = [key for key in REQUIRED_COVERAGE if key not in counts]
    if missing:
        raise SystemExit(f"traces miss required mechanics: {', '.join(missing)}")


if __name__ == "__main__":
    main()
