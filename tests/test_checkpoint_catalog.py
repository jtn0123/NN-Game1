from __future__ import annotations

import os

from src.app.checkpoint_catalog import CheckpointRepository, iter_checkpoint_candidates


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


def test_checkpoint_repository_resolves_ids_and_deletes_allowed_models(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "best.pth"
    model_path.write_bytes(b"checkpoint")
    repository = CheckpointRepository([(str(model_dir), "game")])

    assert repository.model_id("game", "best.pth") == "game:best.pth"
    assert repository.resolve("game:best.pth") == os.path.realpath(model_path)

    success, filename, error = repository.delete("game:best.pth")

    assert success is True
    assert filename == "best.pth"
    assert error is None
    assert not model_path.exists()


def test_checkpoint_repository_rejects_unknown_and_unsafe_refs(tmp_path):
    model_dir = tmp_path / "models"
    outside_dir = tmp_path / "outside"
    model_dir.mkdir()
    outside_dir.mkdir()
    (model_dir / "safe.pth").write_bytes(b"checkpoint")
    outside = outside_dir / "safe.pth"
    outside.write_bytes(b"checkpoint")
    repository = CheckpointRepository([(str(model_dir), "game")])

    assert repository.resolve("game:../safe.pth") is None
    assert repository.resolve("other:safe.pth") is None
    assert repository.resolve(str(outside)) is None
    assert repository.delete("missing:safe.pth") == (False, None, "Invalid model id")
