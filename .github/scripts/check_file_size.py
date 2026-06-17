#!/usr/bin/env python3
"""Fail when source files grow past the configured line-count budget."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SOURCE_EXTENSIONS = {".py", ".js", ".css"}
DEFAULT_ROOTS = ("src", "main.py", "config.py")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(path) for path in result.stdout.splitlines()]


def is_in_scope(path: Path, roots: tuple[str, ...]) -> bool:
    path_text = path.as_posix()
    for root in roots:
        if path_text == root or path_text.startswith(f"{root.rstrip('/')}/"):
            return True
    return False


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument("--root", action="append", dest="roots")
    args = parser.parse_args()

    roots = tuple(args.roots or DEFAULT_ROOTS)
    violations: list[tuple[int, Path]] = []
    for path in tracked_files():
        if path.suffix not in SOURCE_EXTENSIONS or not is_in_scope(path, roots):
            continue
        if not path.exists():
            continue
        line_count = count_lines(path)
        if line_count > args.max_lines:
            violations.append((line_count, path))

    if not violations:
        print(f"Source file size check passed: all files are <= {args.max_lines} lines")
        return 0

    print(f"Source file size check failed: files over {args.max_lines} lines")
    for line_count, path in sorted(violations, reverse=True):
        print(f"  {line_count:5d} {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
