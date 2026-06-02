"""Tests for safer checkpoint loading helpers."""

import os
import tempfile

import pytest
import torch

from src.utils.checkpoint_loader import load_checkpoint


def test_load_checkpoint_uses_restricted_loader_by_default(tmp_path):
    """Normal checkpoints should load through weights_only=True."""
    path = tmp_path / "checkpoint.pth"
    torch.save({"steps": 10}, path)

    checkpoint = load_checkpoint(str(path), map_location="cpu")

    assert checkpoint["steps"] == 10


def test_load_checkpoint_rejects_untrusted_unsafe_fallback(monkeypatch, tmp_path):
    """Untrusted paths should not fall back to weights_only=False."""
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"placeholder")
    calls = []

    def fake_load(filepath, map_location=None, weights_only=True):
        calls.append(weights_only)
        raise ValueError("restricted load failed")

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="restricted loader"):
        load_checkpoint(str(path), map_location="cpu")

    assert calls == [True]


def test_load_checkpoint_allows_trusted_compatibility_fallback(monkeypatch):
    """Trusted local model dirs may use the legacy loader as a compatibility fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pth")
        open(path, "wb").close()
        calls = []

        def fake_load(filepath, map_location=None, weights_only=True):
            calls.append(weights_only)
            if weights_only:
                raise ValueError("restricted load failed")
            return {"loaded": True}

        monkeypatch.setattr(torch, "load", fake_load)

        with pytest.warns(RuntimeWarning, match="unrestricted checkpoint load"):
            checkpoint = load_checkpoint(
                path,
                map_location="cpu",
                trusted_dirs=[tmpdir],
                allow_unsafe_fallback=True,
            )

        assert checkpoint == {"loaded": True}
        assert calls == [True, False]
