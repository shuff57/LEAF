import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("ENABLE_MOCK", "1")
    monkeypatch.setenv("DEVICE", "cpu")
    # Offer every model; test_only_bioclip_by_default covers the default.
    monkeypatch.setenv("BACKENDS", "plantnet300k,bioclip,bioclip2,inat21,inat21-convnext")
    from fastapi.testclient import TestClient

    from app import backends
    from app.main import app

    backends.reset()
    return TestClient(app)


@pytest.fixture()
def jpeg_bytes():
    def make(size=(320, 240), color=(40, 140, 60)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, "JPEG")
        return buf.getvalue()

    return make
