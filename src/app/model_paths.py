"""Shared checkpoint path and model-id helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

CHECKPOINT_SUFFIX = ".pth"
LEGACY_MODEL_SOURCE = "legacy"


@dataclass(frozen=True)
class ModelDirectory:
    """Allowed model directory plus browser-facing source label."""

    path: str
    source: str


def model_search_dirs(config: Any) -> list[tuple[str, str]]:
    """Return allowed model directories as `(directory, source)` pairs."""
    return [
        (config.GAME_MODEL_DIR, config.GAME_NAME),
        (config.MODEL_DIR, LEGACY_MODEL_SOURCE),
    ]


def model_id(source: str, filename: str) -> str:
    """Create a browser-safe model identifier without exposing local paths."""
    return f"{source}:{filename}"


def parse_model_id(model_ref: str) -> Optional[Tuple[str, str]]:
    """Parse an opaque browser model id into `(source, filename)`."""
    if not model_ref or ":" not in model_ref:
        return None
    source, filename = model_ref.split(":", 1)
    if not source or not is_safe_checkpoint_filename(filename):
        return None
    return source, filename


def is_safe_checkpoint_filename(filename: str) -> bool:
    """Return whether a checkpoint filename is a plain local `.pth` basename."""
    return (
        bool(filename)
        and filename.endswith(CHECKPOINT_SUFFIX)
        and os.path.basename(filename) == filename
        and "/" not in filename
        and "\\" not in filename
    )


def normalize_checkpoint_filename(filename: str, default: str = "custom_save") -> str:
    """Return a safe checkpoint filename with a `.pth` suffix."""
    base = os.path.basename(filename.strip())
    if base.endswith(CHECKPOINT_SUFFIX):
        base = base[: -len(CHECKPOINT_SUFFIX)]
    base = "".join(c for c in base if c.isalnum() or c in "_-").strip()
    if not base:
        base = default
    return f"{base}{CHECKPOINT_SUFFIX}"


def is_allowed_checkpoint_path(candidate: str, model_dir: str) -> bool:
    """Return whether `candidate` is a `.pth` file inside `model_dir`."""
    real_dir = os.path.realpath(model_dir)
    try:
        return os.path.commonpath([real_dir, candidate]) == real_dir and candidate.endswith(
            CHECKPOINT_SUFFIX
        )
    except ValueError:
        return False


def allowed_model_roots(model_dirs: Iterable[ModelDirectory]) -> list[str]:
    """Return real paths for allowed model roots."""
    return [os.path.realpath(entry.path) for entry in model_dirs]
