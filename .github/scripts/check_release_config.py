#!/usr/bin/env python3
"""Validate the semantic-release and PR-title release contract."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ALLOWED_TAGS = [
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "style",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
]
EXPECTED_MINOR_TAGS = ["feat"]
EXPECTED_PATCH_TAGS = ["fix", "perf"]


def require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"unexpected {name}: {actual!r} != {expected!r}")


def semantic_release_config() -> dict:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data["tool"]["semantic_release"]


def validate_semantic_release_config(config: dict) -> None:
    require_equal(
        "semantic-release version targets",
        config.get("version_toml"),
        [
            "pyproject.toml:project.version",
            "pyproject.toml:tool.semantic_release.version",
        ],
    )
    require_equal("semantic-release branch", config.get("branch"), "main")
    require_equal("semantic-release tag_format", config.get("tag_format"), "v{version}")
    require_equal("semantic-release upload_to_pypi", config.get("upload_to_pypi"), False)
    require_equal("semantic-release assets", config.get("assets"), ["VERSION"])
    if "NEW_VERSION" not in str(config.get("build_command", "")):
        raise ValueError("build_command must write NEW_VERSION to VERSION")

    parser_options = config["commit_parser_options"]
    require_equal("allowed_tags", parser_options.get("allowed_tags"), EXPECTED_ALLOWED_TAGS)
    require_equal("minor_tags", parser_options.get("minor_tags"), EXPECTED_MINOR_TAGS)
    require_equal("patch_tags", parser_options.get("patch_tags"), EXPECTED_PATCH_TAGS)


def pr_title_lint_types(workflow_text: str) -> list[str]:
    match = re.search(r"(?ms)^\s+types:\s*\|\n(?P<body>(?:\s{12}\w+\n)+)", workflow_text)
    if not match:
        raise ValueError("could not find semantic-pull-request types block")
    return [line.strip() for line in match.group("body").splitlines() if line.strip()]


def validate_pr_title_lint(workflow_text: str) -> None:
    require_equal("PR title lint types", pr_title_lint_types(workflow_text), EXPECTED_ALLOWED_TAGS)
    for required in (
        "Allow Dependabot-generated bump titles",
        "action-semantic-pull-request@v6.1.1",
        "subjectPattern:",
        "requireScope: false",
    ):
        if required not in workflow_text:
            raise ValueError(f"PR title lint missing {required}")


def validate_release_workflow(workflow_text: str) -> None:
    required_snippets = [
        "python .github/scripts/check_release_config.py",
        "semantic-release version",
        "semantic-release publish",
        "Warn loudly if no release was cut",
        "Build Python package artifacts",
        "Attach package artifacts to GitHub release",
        "Strip build-only files from GitHub release",
    ]
    for snippet in required_snippets:
        if snippet not in workflow_text:
            raise ValueError(f"release workflow missing {snippet!r}")


def validate_ci_workflow(workflow_text: str) -> None:
    required_snippets = [
        "python .github/scripts/check_release_config.py",
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "actions/setup-node@v6",
    ]
    for snippet in required_snippets:
        if snippet not in workflow_text:
            raise ValueError(f"CI workflow missing {snippet!r}")


def main() -> int:
    validate_semantic_release_config(semantic_release_config())
    validate_pr_title_lint((ROOT / ".github/workflows/pr-title-lint.yml").read_text())
    validate_release_workflow((ROOT / ".github/workflows/release.yml").read_text())
    validate_ci_workflow((ROOT / ".github/workflows/ci.yml").read_text())
    version = (ROOT / "VERSION").read_text().strip()
    if not version:
        raise ValueError("VERSION must not be empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
