import json

import pytest

from app import common_names


def test_identify_adds_common_names(client, jpeg_bytes, tmp_path):
    # The client fixture points MODELS_DIR at tmp_path. Keys are scientific names without the
    # author citation, so the mock's "Taraxacum officinale F.H.Wigg." finds its entry.
    (tmp_path / "common_names.json").write_text(json.dumps({"Taraxacum officinale": "Dandelion"}))
    r = client.post(
        "/api/identify",
        files={"file": ("p.jpg", jpeg_bytes(), "image/jpeg")},
        data={"backend": "mock", "top_k": "10"},
    )
    common = {p["name"]: p["common_name"] for p in r.json()["predictions"]}
    assert common["Taraxacum officinale F.H.Wigg."] == "Dandelion"
    assert common["Quercus robur L."] is None


def test_pick_matches_the_name_not_the_top_hit():
    # Shapes taken from real iNaturalist /v1/taxa answers.
    dandelion = [
        {
            "name": "Taraxacum",
            "rank": "section",
            "matched_term": "Taraxacum officinale",
            "preferred_common_name": "common dandelions",
        },
        {
            "name": "Taraxacum officinale",
            "rank": "species",
            "matched_term": "Taraxacum officinale",
            "preferred_common_name": "common dandelion",
        },
    ]
    assert common_names.pick(dandelion, "Taraxacum officinale") == "Common dandelion"
    synonym = [
        {
            "name": "Pelargonium × hybridum",
            "matched_term": "Pelargonium × hortorum",
            "preferred_common_name": "garden geranium",
        }
    ]
    assert common_names.pick(synonym, "Pelargonium × hortorum") == "Garden geranium"
    assert common_names.pick(synonym, "Rosa canina") is None


def test_builder_takes_inat21_names_without_the_network(tmp_path, monkeypatch):
    # iNat21's config.json describes every species, plants or not, as "common name, group".
    descriptions = {
        "Taraxacum officinale": "common dandelion, Plant",
        "Canis lupus": "Gray Wolf, Mammal",
    }
    (tmp_path / "inat21").mkdir()
    (tmp_path / "inat21" / "config.json").write_text(json.dumps({"label_descriptions": descriptions}))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(common_names, "lookup", lambda name: pytest.fail(f"looked up {name}"))
    common_names.main()
    saved = json.loads((tmp_path / "common_names.json").read_text(encoding="utf-8"))
    assert saved == {"Taraxacum officinale": "Common dandelion"}
