"""Model listing and deletion helpers for the web dashboard."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.app.checkpoint_catalog import CheckpointRepository
from src.utils.checkpoint_loader import load_checkpoint_with_status


class ModelService:
    """Encapsulates model filesystem operations behind opaque model ids."""

    def __init__(self, model_dirs: Iterable[Tuple[str, str]]):
        self._repository = CheckpointRepository(model_dirs)

    @property
    def model_dirs(self) -> List[Tuple[str, str]]:
        return self._repository.model_dirs

    @staticmethod
    def model_id(source: str, filename: str) -> str:
        return CheckpointRepository.model_id(source, filename)

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []

        for candidate in self._repository.candidates():
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
                "requires_unsafe_load": False,
            }

            try:
                load_result = load_checkpoint_with_status(
                    candidate.path,
                    map_location="cpu",
                    trusted_dirs=[candidate.directory],
                    allow_unsafe_fallback=True,
                )
                checkpoint = load_result.checkpoint
                model_info["requires_unsafe_load"] = load_result.used_unsafe_fallback
                if load_result.used_unsafe_fallback:
                    model_info["security_warning"] = (
                        "Legacy checkpoint requires compatibility fallback; re-save it "
                        "after loading to migrate to the restricted format."
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
        return self._repository.resolve(model_ref)

    def delete(self, model_ref: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Delete a model by id. Returns (success, filename, error)."""
        return self._repository.delete(model_ref)
