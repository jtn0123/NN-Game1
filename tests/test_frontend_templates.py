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


def test_crystal_caves_panel_markup_and_binding_present():
    """The Crystal Caves panel markup and its JS binding must stay in sync.

    Each element the JS writes to needs a matching id in the template, otherwise
    the panel silently renders blank.
    """
    dashboard = (ROOT / "src" / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")
    app_js = (ROOT / "src" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "src" / "web" / "static" / "styles_layout.css").read_text(encoding="utf-8")

    assert 'id="crystal-caves-panel"' in dashboard
    assert "updateCrystalCaves" in app_js

    element_ids = [
        "cc-progress-fill",
        "cc-progress-text",
        "cc-progress-best",
        "cc-crystal-fill",
        "cc-crystals-text",
        "cc-switch",
        "cc-depth",
        "cc-difficulty",
        "cc-outcome",
        "cc-outcomes",
    ]
    for element_id in element_ids:
        assert f'id="{element_id}"' in dashboard, f"missing template id: {element_id}"
        assert element_id in app_js, f"app.js never targets id: {element_id}"

    # The panel reuses the gauge styling and ships its own colour classes.
    assert ".cc-fill-crystal" in styles
    assert ".cc-best-marker" in styles


def test_curriculum_panel_markup_and_binding_present():
    """The Crystal Caves curriculum panel must stay wired to dashboard state."""
    dashboard = (ROOT / "src" / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")
    app_js = (ROOT / "src" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "src" / "web" / "static" / "styles_layout.css").read_text(encoding="utf-8")

    assert 'id="curriculum-panel"' in dashboard
    assert "updateCurriculum" in app_js

    for element_id in [
        "curriculum-stage-count",
        "curriculum-stage-name",
        "curriculum-stage-meta",
        "curriculum-status",
        "curriculum-episode-text",
        "curriculum-stage-fill",
        "curriculum-gate",
        "curriculum-next",
    ]:
        assert f'id="{element_id}"' in dashboard, f"missing template id: {element_id}"
        assert element_id in app_js, f"app.js never targets id: {element_id}"

    assert ".curriculum-panel" in styles
    assert ".curriculum-fill" in styles


def test_held_out_eval_panel_markup_and_binding_present():
    """The held-out Evaluation panel markup must stay in sync with its JS binding."""
    dashboard = (ROOT / "src" / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")
    app_js = (ROOT / "src" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "src" / "web" / "static" / "styles_layout.css").read_text(encoding="utf-8")

    assert 'id="eval-panel"' in dashboard
    assert "updateEval" in app_js

    for element_id in [
        "eval-mean",
        "eval-std",
        "eval-median",
        "eval-winrate",
        "eval-best",
        "eval-games",
        "eval-verdict",
        "eval-verdict-label",
        "eval-verdict-detail",
        "eval-last-ep",
        "eval-spark-line",
        "eval-spark-dot",
    ]:
        assert f'id="{element_id}"' in dashboard, f"missing template id: {element_id}"
        assert element_id in app_js, f"app.js never targets id: {element_id}"

    assert ".eval-panel" in styles
    assert ".eval-verdict" in styles
    assert "eval_is_baseline" in app_js


def test_headless_dashboard_hides_dead_visual_controls():
    """Headless runs should not show no-op preview/speed controls."""
    app_js = (ROOT / "src" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "src" / "web" / "static" / "styles_layout.css").read_text(encoding="utf-8")

    assert "is-headless-mode" in app_js
    assert ".is-headless-mode .preview-card" in styles
    assert ".is-headless-mode .speed-control" in styles


def test_footer_tooltip_is_game_neutral():
    dashboard = (ROOT / "src" / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")

    assert "learns to play Breakout" not in dashboard
    assert "learns to play the selected game" in dashboard


def test_nn_panel_has_desktop_readability_controls_and_crystal_caves_labels():
    dashboard = (ROOT / "src" / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")
    app_js = (ROOT / "src" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    nn_js = (ROOT / "src" / "web" / "static" / "dashboard_nn.js").read_text(encoding="utf-8")

    assert 'id="nn-connections-toggle"' in dashboard
    assert "toggle-nn-connections" in app_js
    assert "ensureNNConnectionsToggle" in nn_js
    assert "showConnections = false" in nn_js
    assert "actionGlyph" in nn_js
    assert "LEFT_JUMP" in nn_js
    assert "RIGHT_SHOOT" in nn_js
    assert "actionIcons[i] || '?'" not in nn_js


def test_model_browser_has_focusable_labeled_destructive_controls():
    core_js = (ROOT / "src" / "web" / "static" / "dashboard_core.js").read_text(encoding="utf-8")
    controls_js = (ROOT / "src" / "web" / "static" / "dashboard_controls.js").read_text(
        encoding="utf-8"
    )

    assert "<span>Delete</span>" in core_js
    assert "handleLoadModalKeydown" in controls_js
    assert "loadModalPreviouslyFocused" in controls_js
    assert "focusLoadModal" in controls_js
