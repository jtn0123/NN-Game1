"""
Checkpoint loading helpers.

Prefer PyTorch's restricted unpickler and only fall back to unrestricted loading
for checkpoints from explicitly trusted local model directories.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Iterable, Optional

import torch


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
    try:
        return torch.load(filepath, map_location=map_location, weights_only=True)
    except Exception as safe_error:
        if allow_unsafe_fallback and _is_under_trusted_dir(filepath, trusted_dirs):
            warnings.warn(
                (
                    f"Falling back to unrestricted checkpoint load for trusted local file: "
                    f"{filepath}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.load(filepath, map_location=map_location, weights_only=False)

        raise RuntimeError(
            "Checkpoint could not be loaded with PyTorch's restricted loader. "
            "Only trusted local model directories may use the compatibility fallback."
        ) from safe_error
