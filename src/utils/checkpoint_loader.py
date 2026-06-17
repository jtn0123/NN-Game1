"""
Checkpoint loading helpers.

Prefer PyTorch's restricted unpickler and only fall back to unrestricted loading
for checkpoints from explicitly trusted local model directories.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Checkpoint payload plus whether unrestricted compatibility loading was used."""

    checkpoint: Any
    used_unsafe_fallback: bool = False


def _is_under_trusted_dir(filepath: str, trusted_dirs: Optional[Iterable[str]]) -> bool:
    if not trusted_dirs:
        return False

    real_file = os.path.realpath(filepath)
    for directory in trusted_dirs:
        if not directory:
            continue
        real_dir = os.path.realpath(directory)
        try:
            if os.path.commonpath([real_file, real_dir]) == real_dir:
                return True
        except ValueError:
            continue
    return False


def load_checkpoint(
    filepath: str,
    map_location: Any = "cpu",
    *,
    trusted_dirs: Optional[Iterable[str]] = None,
    allow_unsafe_fallback: bool = False,
) -> Any:
    """
    Load a checkpoint using `weights_only=True` first.

    Some legacy checkpoints may contain NumPy replay-buffer arrays or other
    pickled metadata that PyTorch's restricted loader rejects. Those are only
    loaded with `weights_only=False` when the caller opts in and the file lives
    under an explicitly trusted directory.
    """
    return load_checkpoint_with_status(
        filepath,
        map_location=map_location,
        trusted_dirs=trusted_dirs,
        allow_unsafe_fallback=allow_unsafe_fallback,
    ).checkpoint


def load_checkpoint_with_status(
    filepath: str,
    map_location: Any = "cpu",
    *,
    trusted_dirs: Optional[Iterable[str]] = None,
    allow_unsafe_fallback: bool = False,
) -> CheckpointLoadResult:
    """
    Load a checkpoint and report whether legacy unrestricted loading was needed.

    This is useful for model browsers and migration tooling that need to warn
    users about old checkpoint formats without changing the normal load API.
    """
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        return CheckpointLoadResult(checkpoint=checkpoint)
    except Exception as safe_error:
        if allow_unsafe_fallback and _is_under_trusted_dir(filepath, trusted_dirs):
            checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
            warnings.warn(
                (
                    f"Falling back to unrestricted checkpoint load for trusted local file: "
                    f"{filepath}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return CheckpointLoadResult(checkpoint=checkpoint, used_unsafe_fallback=True)

        raise RuntimeError(
            "Checkpoint could not be loaded with PyTorch's restricted loader. "
            "Only trusted local model directories may use the compatibility fallback."
        ) from safe_error
