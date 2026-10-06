"""Run isolated native presentation checks against the built macOS player."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "unity/Builds/Crystal Caves.app"


def players(executable: str) -> list[int]:
    listing = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    result = []
    for line in listing.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[1].startswith(executable + " "):
            result.append(int(parts[0]))
    return result


def activate(executable: str, evidence: Path) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        running = players(executable)
        if running:
            program = (
                "import AppKit; if let app = NSRunningApplication(processIdentifier: "
                + str(running[0])
                + ") { print(app.activate(options: [])) }"
            )
            result = subprocess.run(
                ["/usr/bin/swift", "-e", program], check=True, capture_output=True, text=True
            )
            (evidence / "activation.log").write_text(result.stdout + result.stderr)
            break
        time.sleep(0.1)
    desktop = subprocess.run(
        [
            "/usr/bin/swift",
            "-e",
            'import AppKit; print(NSWorkspace.shared.frontmostApplication?.localizedName ?? "none")',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (evidence / "desktop.log").write_text(desktop.stdout + desktop.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("menus", "display", "render", "vsync"))
    args = parser.parse_args()
    if not APP.is_dir():
        parser.error("Build the player first with python scripts/unity_pilot.py --build")
    output = ROOT / ".Codex/artifacts/unity-pilot" / ("settings-" + args.case)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crystal-caves-settings-") as directory:
        temporary = Path(directory)
        staged = temporary / "Crystal Caves.app"
        evidence = temporary / "evidence"
        evidence.mkdir()
        shutil.copytree(APP, staged)
        shutil.copy2(ROOT / "docs/unity-review/review-state.json", evidence / "state.json")
        executable = str((staged / "Contents/MacOS/Crystal Caves").resolve())
        try:
            subprocess.run(
                [
                    "open",
                    "-n",
                    str(staged),
                    "--args",
                    "--settings-smoke",
                    str(evidence),
                    "--settings-case",
                    args.case,
                    "--settings-state",
                    str(evidence / "state.json"),
                    "-force-gfx-direct",
                    "-screen-width",
                    "1280",
                    "-screen-height",
                    "800",
                    "-screen-fullscreen",
                    "0",
                    "-logFile",
                    str(evidence / "player.log"),
                ],
                check=True,
            )
            if args.case == "display":
                activate(executable, evidence)
            report = evidence / "settings-report.json"
            deadline = time.monotonic() + 65
            while not report.exists() and time.monotonic() < deadline:
                time.sleep(0.25)
            if not report.exists():
                raise RuntimeError("Native check timed out; evidence: " + str(output))
            result = json.loads(report.read_text())
            print(json.dumps(result, indent=2))
            if not result["success"]:
                raise RuntimeError(result["error"])
            if result.get("skipped"):
                raise RuntimeError("Native acceptance incomplete: " + "; ".join(result["skipped"]))
        finally:
            shutil.copytree(evidence, output, dirs_exist_ok=True)
            for pid in players(executable):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    print("Evidence: " + str(output))


if __name__ == "__main__":
    main()
