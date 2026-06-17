"""Shared checkpoint discovery for app and web model services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple


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
