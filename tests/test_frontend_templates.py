"""Static frontend checks for dashboard templates and generated markup."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INLINE_HANDLER_RE = re.compile(
    r"\bon(?:click|change|input|submit|keyup|keydown|mouseover|mouseout)\s*=",
    re.IGNORECASE,
)


def test_frontend_markup_does_not_use_inline_event_handlers():
    """Templates and generated markup should bind behavior from JavaScript."""
    paths = [
        ROOT / "src" / "web" / "templates" / "dashboard.html",
        ROOT / "src" / "web" / "templates" / "launcher.html",
        ROOT / "src" / "web" / "static" / "app.js",
    ]

    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in INLINE_HANDLER_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line_no}:{match.group(0)}")

    assert offenders == []
