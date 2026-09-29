import json

import pytest
from PIL import Image

from app.backends.inat21 import INat21

timm = pytest.importorskip("timm")
save_file = pytest.importorskip("safetensors.torch").save_file

# iNat21 also covers animals and fungi; config.json describes each species as "common name, group".
NAMES = ["Canis lupus", "Quercus robur", "Apis mellifera", "Rosa canina"]
GROUPS = {
    "Canis lupus": "Gray Wolf, Mammal",
    "Quercus robur": "English oak, Plant",
    "Apis mellifera": "Western Honey Bee, Insect",
    "Rosa canina": "dog rose, Plant",
}


def test_inat21_loads_local_folder_and_answers_with_plants_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("DEVICE", "cpu")
    folder = tmp_path / "inat21"
    assert INat21().is_available() == (False, f"Missing {folder / 'config.json'}")

    # Random-weight stand-in with the layout of the timm hub repo.
    folder.mkdir()
    config = {
        "architecture": "resnet18",
        "num_classes": len(NAMES),
        "label_names": NAMES,
        "label_descriptions": GROUPS,
        "pretrained_cfg": {"input_size": [3, 64, 64], "crop_pct": 1.0},
    }
    (folder / "config.json").write_text(json.dumps(config))
    save_file(
        timm.create_model("resnet18", num_classes=len(NAMES)).state_dict(),
        str(folder / "model.safetensors"),
    )

    b = INat21()
    b.load()
    assert b.names == ["Quercus robur", "Rosa canina"]
    preds = b.predict(Image.new("RGB", (120, 90), (40, 140, 60)), top_k=5)
    assert sorted(p.name for p in preds) == ["Quercus robur", "Rosa canina"]
    assert sum(p.score for p in preds) == pytest.approx(1.0, abs=1e-4)
