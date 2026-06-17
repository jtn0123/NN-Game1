"""Bootstrap a local NN-Game1 development environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, required: bool = True) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    result = subprocess.run(command, cwd=ROOT, check=False)
    if required and result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-node", action="store_true", help="Skip npm dependency setup")
    parser.add_argument(
        "--skip-hooks",
        action="store_true",
        help="Skip installing pre-commit hooks",
    )
    args = parser.parse_args()

    python = sys.executable
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", "requirements.txt", "-c", "constraints.txt"])

    if not args.skip_node and (ROOT / "package-lock.json").exists():
        if shutil.which("npm"):
            run(["npm", "ci"])
        else:
            print("npm is not installed; skipped Node dependency setup")

    if not args.skip_hooks:
        if shutil.which("pre-commit"):
            run(["pre-commit", "install"], required=False)
        else:
            print("pre-commit is not available; install dependencies first or rerun bootstrap")

    print("Development environment bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
