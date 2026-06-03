import os

import pytest
import torch

from src.web.model_service import ModelService


def test_model_service_lists_opaque_ids_without_paths(tmp_path):
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    torch.save({"steps": 7, "epsilon": 0.25}, model_dir / "best.pth")

    service = ModelService([(str(model_dir), "breakout")])
    models = service.list_models()

    assert models[0]["id"] == "breakout:best.pth"
    assert models[0]["steps"] == 7
    assert models[0]["is_loadable"] is True
    assert "path" not in models[0]


def test_model_service_marks_unreadable_checkpoints(tmp_path):
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    (model_dir / "broken.pth").write_bytes(b"not a checkpoint")

    service = ModelService([(str(model_dir), "breakout")])
    models = service.list_models()

    assert models[0]["id"] == "breakout:broken.pth"
    assert models[0]["is_loadable"] is False
    assert models[0]["has_metadata"] is False
    assert "load_error" in models[0]
    assert "path" not in models[0]


def test_model_service_resolves_ids_and_rejects_traversal(tmp_path):
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pth"
    model_path.write_bytes(b"checkpoint")

    service = ModelService([(str(model_dir), "breakout")])

    assert service.resolve("breakout:best.pth") == os.path.realpath(model_path)
    assert service.resolve("breakout:../best.pth") is None
    assert service.resolve("legacy:best.pth") is None


@pytest.mark.parametrize(
    "model_ref",
    [
        "",
        "breakout:",
        "breakout:.",
        "breakout:..",
        "breakout:best.pth/extra",
        "breakout:%2e%2e/best.pth",
        "breakout:subdir/best.pth",
        "breakout:\\best.pth",
        "unknown:best.pth",
    ],
)
def test_model_service_rejects_malformed_model_ids(tmp_path, model_ref):
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    (model_dir / "best.pth").write_bytes(b"checkpoint")

    service = ModelService([(str(model_dir), "breakout")])

    assert service.resolve(model_ref) is None


def test_model_service_resolves_legacy_absolute_paths_only_inside_allowed_dirs(
    tmp_path,
):
    model_dir = tmp_path / "models" / "breakout"
    outside_dir = tmp_path / "outside"
    model_dir.mkdir(parents=True)
    outside_dir.mkdir()
    allowed_model = model_dir / "best.pth"
    outside_model = outside_dir / "best.pth"
    allowed_model.write_bytes(b"checkpoint")
    outside_model.write_bytes(b"checkpoint")

    service = ModelService([(str(model_dir), "breakout")])

    assert service.resolve(str(allowed_model)) == os.path.realpath(allowed_model)
    assert service.resolve(str(outside_model)) is None


def test_model_service_delete_rejects_symlink_path(tmp_path):
    model_dir = tmp_path / "models" / "breakout"
    target_dir = tmp_path / "target"
    model_dir.mkdir(parents=True)
    target_dir.mkdir()
    (target_dir / "unsafe.pth").write_bytes(b"checkpoint")
    os.symlink(target_dir, model_dir / "linked")

    service = ModelService([(str(model_dir), "breakout")])

    success, filename, error = service.delete("breakout:linked")

    assert not success
    assert filename is None
    assert error == "Invalid model id"


def test_model_service_deletes_allowed_model(tmp_path):
    model_dir = tmp_path / "models" / "breakout"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pth"
    model_path.write_bytes(b"checkpoint")

    service = ModelService([(str(model_dir), "breakout")])
    success, filename, error = service.delete("breakout:best.pth")

    assert success
    assert filename == "best.pth"
    assert error is None
    assert not model_path.exists()
