#!/usr/bin/env python3
"""Fail when dependency-audit vulnerability ignores need review."""

from __future__ import annotations

import datetime as dt

AUDIT_WAIVERS = {
    "CVE-2025-3000": {
        "package": "torch",
        "review_by": dt.date(2026, 9, 1),
        "reason": "No patched torch release is available on PyPI yet.",
    },
}


def main() -> int:
    today = dt.date.today()
    expired = [
        (vuln_id, waiver)
        for vuln_id, waiver in AUDIT_WAIVERS.items()
        if waiver["review_by"] < today
    ]
    if not expired:
        print("Dependency audit waiver check passed")
        return 0

    print("Dependency audit waiver check failed: review expired")
    for vuln_id, waiver in expired:
        print(
            f"  {vuln_id} ({waiver['package']}) review_by={waiver['review_by']}: "
            f"{waiver['reason']}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
