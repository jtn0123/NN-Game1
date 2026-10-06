"""Build and launch Crystal Caves with its local Python simulation."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "unity"
PLAYER = PROJECT / "Builds/Crystal Caves.app/Contents/MacOS/Crystal Caves"


def unity_editor(explicit: str | None) -> Path:
    if explicit:
        result = Path(explicit).expanduser()
    else:
        editors = sorted(
            Path("/Applications/Unity/Hub/Editor").glob("*/Unity.app/Contents/MacOS/Unity")
        )
        if not editors:
            raise RuntimeError("Unity Editor was not found. Supply --unity /path/to/Unity.")
        result = editors[-1]
    if not result.is_file():
        raise RuntimeError(f"Unity Editor does not exist: {result}")
    return result


def build(editor: Path, artifact_dir: Path) -> None:
    subprocess.run([sys.executable, "-m", "src.unity_bridge.export_assets"], cwd=ROOT, check=True)
    log = artifact_dir / "build.log"
    print(f"Building the Unity player. Log: {log}", flush=True)
    result = subprocess.run(
        [
            str(editor),
            "-batchmode",
            "-quit",
            "-projectPath",
            str(PROJECT),
            "-executeMethod",
            "CrystalCaves.Pilot.Editor.CaveBuild.Build",
            "-logFile",
            str(log),
        ],
        cwd=ROOT,
    )
    if result.returncode or not PLAYER.is_file():
        raise RuntimeError(f"Unity build failed. Read {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="rebuild the player before starting")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--unity", help="path to the Unity Editor executable")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cnn-state", action="store_true")
    parser.add_argument("--legacy-state", action="store_true")
    parser.add_argument("--record-demos", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 <= args.level < 16:
        parser.error("port must be 1-65535 and level must be 0-15")
    artifact_dir = ROOT / ".Codex/artifacts/unity-pilot"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.build or args.build_only or not PLAYER.is_file():
        build(unity_editor(args.unity), artifact_dir)
    if args.build_only:
        return
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise RuntimeError(
                f"Port {args.port} is already in use. Stop the existing bridge or choose --port."
            ) from error
    command = [
        sys.executable,
        "-m",
        "src.unity_bridge",
        "--port",
        str(args.port),
        "--level",
        str(args.level),
    ]
    if args.model:
        command += ["--model", str(args.model.resolve())]
    if args.cnn_state:
        command += ["--cnn-state"]
    if args.legacy_state:
        command += ["--legacy-state"]
    if args.record_demos:
        command += ["--record-demos", str(args.record_demos.resolve())]
    environment = {**os.environ, "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    bridge = subprocess.Popen(command, cwd=ROOT, env=environment)
    player = None
    try:
        deadline = time.monotonic() + 30
        while True:
            if bridge.poll() is not None:
                raise RuntimeError("The bridge could not start. See the message above.")
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("The bridge did not become ready within 30 seconds.")
                time.sleep(0.1)
        player = subprocess.Popen(
            [
                str(PLAYER),
                "--bridge-port",
                str(args.port),
                "-logFile",
                str(artifact_dir / "player.log"),
            ],
            cwd=ROOT,
        )
        print(
            "Crystal Caves started. Close its window or press Ctrl-C here to stop both processes.",
            flush=True,
        )
        while player.poll() is None:
            if bridge.poll() is not None:
                raise RuntimeError("The bridge stopped while the player was running.")
            time.sleep(0.2)
    finally:
        for process in (player, bridge):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Crystal Caves stopped.")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
