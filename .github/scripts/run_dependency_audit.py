#!/usr/bin/env python3
"""Run pip-audit, installing the audit tool into the active environment if needed."""

from __future__ import annotations

import datetime
import importlib.util
import subprocess
import sys

DEFAULT_AUDIT_ARGS = [
    "-r",
    "requirements.txt",
    "--ignore-vuln",
    "CVE-2025-3000",
]

# Surface the ignored CVE so it cannot be carried silently forever. After this date
# the audit prints a (non-fatal) CI warning to recheck for a patched torch release
# and drop the ignore above. Bump the date when re-reviewed.
CVE_IGNORE_REVIEW = "2026-09-01"


def warn_if_review_overdue() -> None:
    """Emit a CI warning if the CVE-2025-3000 ignore is past its review date."""
    if datetime.date.today().isoformat() > CVE_IGNORE_REVIEW:
        print(
            f"::warning::CVE-2025-3000 ignore is past its review date "
            f"({CVE_IGNORE_REVIEW}); recheck for a patched torch release and drop it.",
            file=sys.stderr,
        )


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
    warn_if_review_overdue()
    ensure_pip_audit()
    audit_args = argv or DEFAULT_AUDIT_ARGS
    return subprocess.run([sys.executable, "-m", "pip_audit", *audit_args]).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
