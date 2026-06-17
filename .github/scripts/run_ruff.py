#!/usr/bin/env python3
"""Run ruff, installing it into the active environment if needed."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def ensure_ruff() -> None:
    """Install ruff when the current Python environment is missing it."""
    if importlib.util.find_spec("ruff") is not None:
        return
    print("ruff is not installed; installing it into the active Python environment")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "ruff>=0.4.0"],
        check=True,
    )


def main(argv: list[str]) -> int:
    ensure_ruff()
    return subprocess.run([sys.executable, "-m", "ruff", *argv]).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
