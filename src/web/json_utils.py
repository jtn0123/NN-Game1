"""JSON conversion helpers for web payloads."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_json_safe(obj: Any) -> Any:
    """Convert NumPy containers and scalars into JSON-compatible values."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    return obj
