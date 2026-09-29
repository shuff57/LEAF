import argparse
import json

import pytest
from PIL import Image

from app.backends.base import BackendUnavailable
from app.backends.plantnet300k import PlantNet300K

torch = pytest.importorskip("torch")
resnet18 = pytest.importorskip("torchvision.models").resnet18

SPECIES = {"30": "Species c", "100": "Species a", "200": "Species b"}


def _models_dir(tmp_path, monkeypatch, checkpoint):
    """Random-weight 3-class ResNet18 plus species files, under the default names."""
    d = tmp_path / "plantnet300k"
    d.mkdir()
    torch.save(checkpoint, d / "resnet18_weights_best_acc.tar")
    (d / "plantnet300K_species_id_2_name.json").write_text(json.dumps(SPECIES))
    (d / "class_idx_to_species_id.json").write_text(json.dumps({"0": "200", "1": "30", "2": "100"}))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.delenv("PLANTNET_ALLOW_PICKLE", raising=False)


def test_load_and_predict_with_random_weights(tmp_path, monkeypatch):
    # Same layout as the official checkpoint: {"epoch", "model", "optimizer"}.
    _models_dir(tmp_path, monkeypatch, {"epoch": 1, "model": resnet18(num_classes=3).state_dict()})
    b = PlantNet300K()
    assert b.is_available() == (True, "")
    b.load()
    assert b.names == ["Species b", "Species c", "Species a"]  # from class_idx_to_species_id.json

    preds = b.predict(Image.new("RGB", (300, 200), (40, 140, 60)), top_k=5)
    assert len(preds) == 3  # top_k is capped at the class count
    assert {p.name for p in preds} == set(SPECIES.values())
    scores = [p.score for p in preds]
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) == pytest.approx(1.0, abs=1e-4)


def test_pickled_objects_need_explicit_opt_in(tmp_path, monkeypatch):
    checkpoint = {"model": resnet18(num_classes=3).state_dict(), "args": argparse.Namespace(lr=0.1)}
    _models_dir(tmp_path, monkeypatch, checkpoint)
    with pytest.raises(BackendUnavailable, match="PLANTNET_ALLOW_PICKLE"):
        PlantNet300K().load()

    monkeypatch.setenv("PLANTNET_ALLOW_PICKLE", "1")
    b = PlantNet300K()
    b.load()
    assert b.loaded
