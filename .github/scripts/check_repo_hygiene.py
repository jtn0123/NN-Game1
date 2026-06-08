"""Fail when scratch or generated files are tracked as source."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DISALLOWED_TRACKED_PATTERNS = (
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"\.py[co]$"),
    re.compile(r"\.backup$"),
    re.compile(r"(^|/)fix_claude\.sh$"),
)


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_files = result.stdout.splitlines()
    violations = [
        path
        for path in tracked_files
        if Path(path).exists()
        if any(pattern.search(path) for pattern in DISALLOWED_TRACKED_PATTERNS)
    ]

    if not violations:
        print("Repository hygiene check passed")
        return 0

    print("Repository hygiene check failed; remove or untrack these scratch files:")
    for path in violations:
        print(f"  - {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
