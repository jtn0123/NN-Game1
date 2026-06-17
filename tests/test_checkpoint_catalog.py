from __future__ import annotations

import os

from src.app.checkpoint_catalog import iter_checkpoint_candidates


def test_checkpoint_catalog_sorts_newest_first_and_deduplicates_real_paths(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    old_model = model_dir / "old.pth"
    new_model = model_dir / "new.pth"
    old_model.write_bytes(b"old")
    new_model.write_bytes(b"new")
    os.utime(old_model, (100, 100))
    os.utime(new_model, (200, 200))

    candidates = iter_checkpoint_candidates(
        [(str(model_dir), "game"), (str(model_dir), "duplicate")]
    )

    assert [candidate.filename for candidate in candidates] == ["new.pth", "old.pth"]
    assert [candidate.source for candidate in candidates] == ["game", "game"]
    assert all(candidate.directory == os.path.realpath(model_dir) for candidate in candidates)
