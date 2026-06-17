#!/usr/bin/env python3
"""Run pip-audit, installing the audit tool into the active environment if needed."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

DEFAULT_AUDIT_ARGS = [
    "-r",
    "requirements.txt",
    "--ignore-vuln",
    "CVE-2025-3000",
]


def ensure_pip_audit() -> None:
    """Install pip-audit when the current Python environment is missing it."""
    if importlib.util.find_spec("pip_audit") is not None:
        return
    print("pip-audit is not installed; installing it into the active Python environment")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pip-audit>=2.7.0"],
        check=True,
    )


def main(argv: list[str]) -> int:
    ensure_pip_audit()
    audit_args = argv or DEFAULT_AUDIT_ARGS
    return subprocess.run([sys.executable, "-m", "pip_audit", *audit_args]).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
