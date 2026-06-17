"""Shared checkpoint discovery for app and web model services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from src.app.model_paths import (
    ModelDirectory,
    allowed_model_roots,
    is_allowed_checkpoint_path,
    model_id,
    parse_model_id,
)


@dataclass(frozen=True)
class CheckpointCandidate:
    """A discovered checkpoint file with stable sorting metadata."""

    path: str
    directory: str
    filename: str
    source: str
    modified: float


def iter_checkpoint_candidates(model_dirs: Iterable[Tuple[str, str]]) -> List[CheckpointCandidate]:
    """Return unique `.pth` files sorted newest-first across model directories."""
    candidates: List[CheckpointCandidate] = []
    seen_paths: set[str] = set()

    for model_dir, source in model_dirs:
        if not os.path.exists(model_dir):
            continue

        for filename in os.listdir(model_dir):
            if not filename.endswith(".pth"):
                continue
            path = os.path.realpath(os.path.join(model_dir, filename))
            if path in seen_paths:
                continue
            seen_paths.add(path)
            candidates.append(
                CheckpointCandidate(
                    path=path,
                    directory=os.path.realpath(model_dir),
                    filename=filename,
                    source=source,
                    modified=os.path.getmtime(path),
                )
            )

    candidates.sort(key=lambda candidate: candidate.modified, reverse=True)
    return candidates


class CheckpointRepository:
    """Shared checkpoint catalog, id resolution, and deletion policy."""

    def __init__(self, model_dirs: Iterable[Tuple[str, str]]):
        self._model_dirs = [
            ModelDirectory(path=model_dir, source=source) for model_dir, source in model_dirs
        ]

    @property
    def model_dirs(self) -> List[Tuple[str, str]]:
        """Return allowed model directories as `(directory, source)` pairs."""
        return [(entry.path, entry.source) for entry in self._model_dirs]

    @staticmethod
    def model_id(source: str, filename: str) -> str:
        """Create a browser-safe model identifier without exposing local paths."""
        return model_id(source, filename)

    def candidates(self) -> List[CheckpointCandidate]:
        """Return discovered checkpoints sorted newest-first."""
        return iter_checkpoint_candidates(self.model_dirs)

    def resolve(self, model_ref: str) -> str | None:
        """Resolve an opaque model id, or legacy absolute path, to an allowed file."""
        if not model_ref:
            return None

        if os.path.isabs(model_ref):
            return self._resolve_absolute_path(model_ref)

        parsed_model_id = parse_model_id(model_ref)
        if parsed_model_id is None:
            return None
        source, filename = parsed_model_id

        for entry in self._model_dirs:
            if source != entry.source:
                continue
            candidate = os.path.realpath(os.path.join(entry.path, filename))
            if is_allowed_checkpoint_path(candidate, entry.path):
                return candidate
        return None

    def delete(self, model_ref: str) -> tuple[bool, str | None, str | None]:
        """Delete a model by id. Returns `(success, filename, error)`."""
        full_path = self.resolve(model_ref)
        if full_path is None:
            return False, None, "Invalid model id"

        if not os.path.exists(full_path):
            return False, None, "Model not found"

        if self._contains_symlink(full_path):
            return False, None, "Cannot delete files with symbolic links in path"

        if not full_path.endswith(".pth"):
            return False, None, "Invalid file type"

        filename = os.path.basename(full_path)
        os.remove(full_path)
        return True, filename, None

    def _resolve_absolute_path(self, model_ref: str) -> str | None:
        candidate = os.path.realpath(model_ref)
        for entry in self._model_dirs:
            if is_allowed_checkpoint_path(candidate, entry.path):
                return candidate
        return None

    def _contains_symlink(self, full_path: str) -> bool:
        allowed_roots = allowed_model_roots(self._model_dirs)
        current_path = full_path
        while current_path and current_path not in allowed_roots:
            if os.path.islink(current_path):
                return True
            current_path = os.path.dirname(current_path)
        return False
