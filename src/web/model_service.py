"""Model listing and deletion helpers for the web dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.checkpoint_loader import load_checkpoint


@dataclass(frozen=True)
class ModelDirectory:
    path: str
    source: str


class ModelService:
    """Encapsulates model filesystem operations behind opaque model ids."""

    def __init__(self, model_dirs: Iterable[Tuple[str, str]]):
        self._model_dirs = [
            ModelDirectory(path=model_dir, source=source)
            for model_dir, source in model_dirs
        ]

    @property
    def model_dirs(self) -> List[Tuple[str, str]]:
        return [(entry.path, entry.source) for entry in self._model_dirs]

    @staticmethod
    def model_id(source: str, filename: str) -> str:
        return f"{source}:{filename}"

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        seen_paths = set()

        for entry in self._model_dirs:
            if not os.path.exists(entry.path):
                continue

            for filename in os.listdir(entry.path):
                if not filename.endswith(".pth"):
                    continue

                path = os.path.join(entry.path, filename)
                if path in seen_paths:
                    continue
                seen_paths.add(path)

                model_info: Dict[str, Any] = {
                    "name": filename,
                    "id": self.model_id(entry.source, filename),
                    "source": entry.source,
                    "size": os.path.getsize(path),
                    "modified": os.path.getmtime(path),
                    "modified_str": datetime.fromtimestamp(
                        os.path.getmtime(path)
                    ).strftime("%Y-%m-%d %H:%M"),
                    "has_metadata": False,
                    "metadata": None,
                }

                try:
                    checkpoint = load_checkpoint(
                        path,
                        map_location="cpu",
                        trusted_dirs=[entry.path],
                        allow_unsafe_fallback=False,
                    )
                    model_info["steps"] = checkpoint.get("steps", None)
                    model_info["epsilon"] = checkpoint.get("epsilon", None)
                    if "metadata" in checkpoint:
                        model_info["has_metadata"] = True
                        model_info["metadata"] = checkpoint["metadata"]
                except Exception:
                    pass

                models.append(model_info)

        models.sort(key=lambda item: item["modified"], reverse=True)
        return models

    def resolve(self, model_ref: str) -> Optional[str]:
        """Resolve an opaque model id, or legacy absolute path, to an allowed file."""
        if not model_ref:
            return None

        if os.path.isabs(model_ref):
            return self._resolve_absolute_path(model_ref)

        if ":" not in model_ref:
            return None

        source, filename = model_ref.split(":", 1)
        if (
            not filename
            or os.path.basename(filename) != filename
            or "/" in filename
            or "\\" in filename
        ):
            return None

        for entry in self._model_dirs:
            if source != entry.source:
                continue
            candidate = os.path.realpath(os.path.join(entry.path, filename))
            if self._is_allowed_file(candidate, entry.path):
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
            if self._is_allowed_file(candidate, entry.path):
                return candidate
        return None

    @staticmethod
    def _is_allowed_file(candidate: str, model_dir: str) -> bool:
        real_dir = os.path.realpath(model_dir)
        try:
            return os.path.commonpath(
                [real_dir, candidate]
            ) == real_dir and candidate.endswith(".pth")
        except ValueError:
            return False

    def _contains_symlink(self, full_path: str) -> bool:
        allowed_roots = [os.path.realpath(entry.path) for entry in self._model_dirs]
        current_path = full_path
        while current_path and current_path not in allowed_roots:
            if os.path.islink(current_path):
                return True
            current_path = os.path.dirname(current_path)
        return False
