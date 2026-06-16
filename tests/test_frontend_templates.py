"""Static frontend checks for dashboard templates and generated markup."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INLINE_HANDLER_RE = re.compile(
    r"\bon(?:click|change|input|submit|keyup|keydown|mouseover|mouseout)\s*=",
    re.IGNORECASE,
)
DATA_ACTION_RE = re.compile(r'data-action="([^"]+)"')
ACTION_REGISTRY_RE = re.compile(
    r"const\s+DASHBOARD_ACTIONS\s*=\s*Object\.freeze\(\{(?P<body>.*?)\n\}\);",
    re.DOTALL,
)
ACTION_KEY_RE = re.compile(r"'([^']+)'\s*:")
STYLE_BLOCK_RE = re.compile(r"<style\b", re.IGNORECASE)


def test_frontend_markup_does_not_use_inline_event_handlers():
    """Templates and generated markup should bind behavior from JavaScript."""
    paths = [
        ROOT / "src" / "web" / "templates" / "dashboard.html",
        ROOT / "src" / "web" / "templates" / "launcher.html",
        ROOT / "src" / "web" / "static" / "app.js",
        ROOT / "src" / "web" / "static" / "launcher.js",
    ]

    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in INLINE_HANDLER_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line_no}:{match.group(0)}")

    assert offenders == []


def test_dashboard_data_actions_have_registered_handlers():
    """Every dashboard data-action should be handled by the central dispatcher."""
    app_js = (ROOT / "src" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    registry_match = ACTION_REGISTRY_RE.search(app_js)
    assert registry_match, "DASHBOARD_ACTIONS registry not found"

    registered = set(ACTION_KEY_RE.findall(registry_match.group("body")))
    actions = set()
    for relative_path in [
        "src/web/templates/dashboard.html",
        "src/web/static/app.js",
    ]:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        actions.update(DATA_ACTION_RE.findall(text))

    assert actions - registered == set()


def test_launcher_template_uses_static_stylesheet():
    """Launcher styling should live in a cacheable static stylesheet."""
    template = (ROOT / "src" / "web" / "templates" / "launcher.html").read_text(encoding="utf-8")

    assert not STYLE_BLOCK_RE.search(template)
    assert "launcher.css" in template
