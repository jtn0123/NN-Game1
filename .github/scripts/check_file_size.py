#!/usr/bin/env python3
"""Fail when source files grow past the configured line-count budget."""

from __future__ import annotations

import argparse
import fnmatch
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


def parse_budget(raw_budget: str) -> tuple[str, int]:
    if "=" not in raw_budget:
        raise argparse.ArgumentTypeError("budgets must use PATTERN=MAX_LINES")
    pattern, max_lines = raw_budget.rsplit("=", 1)
    if not pattern:
        raise argparse.ArgumentTypeError("budget pattern cannot be empty")
    try:
        parsed_max = int(max_lines)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget max lines must be an integer") from exc
    if parsed_max <= 0:
        raise argparse.ArgumentTypeError("budget max lines must be positive")
    return pattern, parsed_max


def line_budget(path: Path, default_max: int, budgets: list[tuple[str, int]]) -> int:
    path_text = path.as_posix()
    matched = [max_lines for pattern, max_lines in budgets if fnmatch.fnmatch(path_text, pattern)]
    return min(matched) if matched else default_max


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        type=parse_budget,
        help="Per-pattern line budget as PATTERN=MAX_LINES, e.g. 'src/game/*.py=950'",
    )
    args = parser.parse_args()

    roots = tuple(args.roots or DEFAULT_ROOTS)
    violations: list[tuple[int, int, Path]] = []
    for path in tracked_files():
        if path.suffix not in SOURCE_EXTENSIONS or not is_in_scope(path, roots):
            continue
        if not path.exists():
            continue
        max_lines = line_budget(path, args.max_lines, args.budget)
        line_count = count_lines(path)
        if line_count > max_lines:
            violations.append((line_count, max_lines, path))

    if not violations:
        print("Source file size check passed")
        return 0

    print("Source file size check failed: files over configured budgets")
    for line_count, max_lines, path in sorted(violations, reverse=True):
        print(f"  {line_count:5d}/{max_lines:<5d} {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
