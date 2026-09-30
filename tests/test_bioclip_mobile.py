import numpy as np
import pytest
from PIL import Image

from app.backends.bioclip import BIOCLIP_25, BioClipMobile, _table_file

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("torchvision")

NAMES = ["Red plant", "Green plant", "Blue plant"]


def _stand_in(path):
    """The Mobile model's interface, [1, 3, 224, 224] RGB in 0..1 -> unit-length embedding, with
    the mean colour as the embedding."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["image"], ["mean"], axes=[2, 3], keepdims=0),
            helper.make_node("ReduceL2", ["mean"], ["norm"], axes=[1], keepdims=1),
            helper.make_node("Div", ["mean", "norm"], ["embedding"]),
        ],
        "stand_in",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def test_mobile_scores_photos_against_bioclip_25_label_table(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("DEVICE", "cpu")
    (tmp_path / "bioclip").mkdir()
    (tmp_path / "bioclip" / "labels.txt").write_text("\n".join(NAMES) + "\n")
    (tmp_path / "bioclip-mobile").mkdir()
    _stand_in(str(tmp_path / "bioclip-mobile" / "flora_student_fp16.onnx"))

    ok, reason = BioClipMobile().is_available()
    assert not ok
    assert "BioCLIP 2.5" in reason  # the label table comes from the full model

    table = _table_file(BIOCLIP_25, NAMES)
    table.parent.mkdir(parents=True)
    np.save(table, np.eye(3, dtype=np.float32))  # one axis per colour
    b = BioClipMobile()
    assert b.is_available() == (True, "")
    b.load()
    preds = b.predict(Image.new("RGB", (300, 200), (40, 140, 60)), top_k=5)
    assert [p.name for p in preds] == ["Green plant", "Blue plant", "Red plant"]
    assert sum(p.score for p in preds) == pytest.approx(1.0, abs=1e-5)
