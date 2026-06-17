#!/usr/bin/env python3
"""Validate that dependency entrypoints stay aligned for local setup."""

from __future__ import annotations

from pathlib import Path

REQUIRED_REQUIREMENTS_LINE = "-e .[web,test,dev]"
REQUIRED_PYPROJECT_MARKERS = [
    "[project]",
    "[project.optional-dependencies]",
    "web = [",
    "test = [",
    "dev = [",
]


def main() -> int:
    failures: list[str] = []
    requirements = Path("requirements.txt")
    pyproject = Path("pyproject.toml")
    package_json = Path("package.json")
    package_lock = Path("package-lock.json")

    if REQUIRED_REQUIREMENTS_LINE not in requirements.read_text(encoding="utf-8"):
        failures.append(f"{requirements} must include {REQUIRED_REQUIREMENTS_LINE!r}")

    pyproject_text = pyproject.read_text(encoding="utf-8")
    for marker in REQUIRED_PYPROJECT_MARKERS:
        if marker not in pyproject_text:
            failures.append(f"{pyproject} is missing dependency marker {marker!r}")

    if not package_json.exists():
        failures.append("package.json is missing")
    if not package_lock.exists():
        failures.append("package-lock.json is missing")

    if failures:
        print("Dependency file check failed")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("Dependency file check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
