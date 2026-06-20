"""Model listing and deletion helpers for the web dashboard."""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.app.checkpoint_catalog import iter_checkpoint_candidates
from src.app.model_paths import (
    ModelDirectory,
    allowed_model_roots,
    is_allowed_checkpoint_path,
    model_id,
    parse_model_id,
)
from src.utils.checkpoint_loader import load_checkpoint


class ModelService:
    """Encapsulates model filesystem operations behind opaque model ids."""

    def __init__(self, model_dirs: Iterable[Tuple[str, str]]):
        self._model_dirs = [
            ModelDirectory(path=model_dir, source=source) for model_dir, source in model_dirs
        ]

    @property
    def model_dirs(self) -> List[Tuple[str, str]]:
        return [(entry.path, entry.source) for entry in self._model_dirs]

    @staticmethod
    def model_id(source: str, filename: str) -> str:
        return model_id(source, filename)

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []

        for candidate in iter_checkpoint_candidates(self.model_dirs):
            model_info: Dict[str, Any] = {
                "name": candidate.filename,
                "id": self.model_id(candidate.source, candidate.filename),
                "source": candidate.source,
                "size": os.path.getsize(candidate.path),
                "modified": candidate.modified,
                "modified_str": datetime.fromtimestamp(candidate.modified).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "is_loadable": True,
                "has_metadata": False,
                "metadata": None,
            }

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    checkpoint = load_checkpoint(
                        candidate.path,
                        map_location="cpu",
                        trusted_dirs=[candidate.directory],
                        allow_unsafe_fallback=True,
                    )
                model_info["steps"] = checkpoint.get("steps", None)
                model_info["epsilon"] = checkpoint.get("epsilon", None)
                if "metadata" in checkpoint:
                    model_info["has_metadata"] = True
                    model_info["metadata"] = checkpoint["metadata"]
            except Exception as exc:
                model_info["is_loadable"] = False
                model_info["load_error"] = type(exc).__name__

            models.append(model_info)

        models.sort(key=lambda item: item["modified"], reverse=True)
        return models

    def resolve(self, model_ref: str) -> Optional[str]:
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

    def delete(self, model_ref: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Delete a model by id. Returns (success, filename, error)."""
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

    def _resolve_absolute_path(self, model_ref: str) -> Optional[str]:
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
