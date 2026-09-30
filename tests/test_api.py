def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_ready"] >= 1  # the mock


def test_backends_lists_mock_and_reports_missing_weights(client):
    body = client.get("/api/backends").json()
    by_id = {b["id"]: b for b in body["backends"]}
    assert by_id["mock"]["available"] is True
    assert by_id["plantnet300k"]["available"] is False
    assert by_id["plantnet300k"]["reason"]  # tells the user what is missing
    assert body["default"] == "mock"


def test_bioclip_entries_name_their_checkpoint(client, monkeypatch):
    monkeypatch.setenv("BIOCLIP_MODEL", "hf-hub:example/some-bioclip")
    labels = {b["id"]: b["label"] for b in client.get("/api/backends").json()["backends"]}
    assert labels["bioclip"] == "BioCLIP (some-bioclip)"
    assert labels["bioclip2"] == "BioCLIP (bioclip-2)"


def test_identify_happy_path(client, jpeg_bytes):
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "mock", "top_k": "3"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "mock"
    assert body["cold_start"] is True
    assert len(body["predictions"]) == 3
    scores = [p["score"] for p in body["predictions"]]
    assert scores == sorted(scores, reverse=True)

    # second call reuses the loaded model
    r2 = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "mock"},
    )
    assert r2.json()["cold_start"] is False
    assert len(r2.json()["predictions"]) == 5


def test_identify_defaults_to_first_ready_backend(client, jpeg_bytes):
    r = client.post("/api/identify", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["backend"] == "mock"


def test_top_k_is_clamped(client, jpeg_bytes):
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "mock", "top_k": "999"},
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) <= 20


def test_rejects_non_image(client):
    r = client.post(
        "/api/identify",
        files={"file": ("notes.txt", b"definitely not an image", "text/plain")},
        data={"backend": "mock"},
    )
    assert r.status_code == 415
    assert "not an image" in r.json()["detail"]


def test_rejects_oversized_upload(client, jpeg_bytes, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes((800, 800)), "image/jpeg")},
        data={"backend": "mock"},
    )
    assert r.status_code == 413


def test_unknown_backend(client, jpeg_bytes):
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "nope"},
    )
    assert r.status_code == 404


def test_unavailable_backend_explains_why(client, jpeg_bytes):
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "plantnet300k"},
    )
    assert r.status_code == 503
    assert r.json()["detail"]  # a human-readable reason (missing weights, or PyTorch not installed)


def test_no_models_ready(tmp_path, monkeypatch, jpeg_bytes):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("ENABLE_MOCK", raising=False)
    from fastapi.testclient import TestClient

    from app import backends
    from app.main import app

    backends.reset()
    r = TestClient(app).post("/api/identify", files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 503
    assert "README" in r.json()["detail"]


def test_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Identify a plant" in r.text


def test_only_bioclip_by_default(client, monkeypatch, jpeg_bytes):
    monkeypatch.delenv("BACKENDS")  # the client fixture offers every model
    ids = [b["id"] for b in client.get("/api/backends").json()["backends"]]
    assert ids == ["bioclip", "mock"]
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "inat21"},
    )
    assert r.status_code == 404  # not offered

    monkeypatch.setenv("BACKENDS", "bioclip,bioclip3")
    typo = client.get("/api/backends").json()["backends"][-1]
    assert typo["id"] == "bioclip3"
    assert typo["available"] is False


def test_load_before_the_first_photo(client):
    first = client.post("/api/load", params={"backend": "mock"}).json()
    assert first["backend"] == "mock"
    assert first["load_ms"] is not None
    assert client.post("/api/load", params={"backend": "mock"}).json()["load_ms"] is None
    assert client.post("/api/load", params={"backend": "nope"}).status_code == 404
