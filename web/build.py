"""Build the browser's files into models/browser/ (gitignored, like every model file).

    .venv/bin/python web/build.py [folder of .jpg photos to check the export with]

BioCLIP 1, 2 and 2.5: the image encoder is exported to ONNX in fp16, with CLIP's normalisation and
the final L2 normalisation inside the graph, so the page feeds it RGB in 0..1 and gets a unit-length
embedding back. BioCLIP 2.5 Mobile already works that way. The page scores that embedding against
a label table (the text embeddings of the species list, stored as fp16), so the browser never runs
a text encoder.

Files over 24 MiB are cut into parts, <file>.0, <file>.1, ..., which the page downloads and joins:
Cloudflare serves files up to 25 MiB, and Chrome keeps no single cache entry as big as BioCLIP 2
(610 MB). models.json lists the models, the files in parts with their full sizes, and a version, a
hash of every file: the page names its cache after it, so a rebuild replaces what browsers kept.

Needs the weights (BioCLIP 1, 2 and 2.5 in models/hf, the Mobile model in models/bioclip-mobile), the
label tables in models/bioclip/cache (the server writes each the first time its model runs; the
Mobile model uses BioCLIP 2.5's), and onnx, onnxscript and onnxruntime in this environment.
ONNX Runtime Web, the library that runs the models in the page, comes from npm at a pinned version
and hash.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "browser"
PART = 24 * 2**20
# Shown under each model in the page's picker. Memory is the page's peak in Chrome on the laptop, GPU
# buffers included, as a phone would count it (see the README).
NOTES = {
    "bioclip-mobile": "Made for phones: the quickest, but it misses more plants.",
    "bioclip1": "The first BioCLIP (2023), quicker than BioCLIP 2.",
    "bioclip2": "Named all 16 test photos. Needs about 3 GB of memory, which can be too much for a phone.",
    "bioclip25": "The newest and largest. Needs about 5 GB of memory, so it is for computers.",
}
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODELS_DIR", str(ROOT / "models"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the downloaded weights; never fetch new ones

ORT_TARBALL = "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.30.0.tgz"
ORT_SHA512 = "q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=="
ORT_FILES = [  # the WebGPU build (it also runs on the CPU) and the smaller CPU-only build
    "ort.webgpu.min.mjs",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
    "ort.wasm.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
]


def fetch_ort() -> None:
    data = urllib.request.urlopen(ORT_TARBALL, timeout=600).read()
    if base64.b64encode(hashlib.sha512(data).digest()).decode() != ORT_SHA512:
        raise SystemExit(f"{ORT_TARBALL} does not match its pinned hash")
    (OUT / "ort").mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for name in ORT_FILES:
            (OUT / "ort" / name).write_bytes(tar.extractfile(f"package/dist/{name}").read())
    print("ONNX Runtime Web 1.30.0 ->", OUT / "ort")


def export_encoder(model_id: str, name: str, photos: list[Path]) -> float:
    """Write OUT/<name>.onnx and return the model's logit scale."""
    from collections import Counter

    import numpy as np
    import onnx
    import onnxruntime as ort
    import open_clip
    import torch
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from PIL import Image
    from torchvision import transforms

    class Encoder(torch.nn.Module):
        """RGB in 0..1 -> unit-length image embedding."""

        def __init__(self, visual, mean, std):
            super().__init__()
            self.visual = visual
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, image):
            f = self.visual((image - self.mean) / self.std)
            return f / f.norm(dim=-1, keepdim=True)

    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    cfg = open_clip.get_model_preprocess_cfg(model)
    encoder = Encoder(model.visual, cfg["mean"], cfg["std"]).eval()
    fp32, out = OUT / f"{name}.fp32.onnx", OUT / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(  # weights in a side file: BioCLIP 2.5 is 2.5 GB in fp32, one ONNX file stops at 2 GB
            encoder, (torch.rand(1, 3, 224, 224),), str(fp32), input_names=["image"],
            output_names=["embedding"], dynamo=True, external_data=True,
        )
    graph = convert_float_to_float16(str(fp32), keep_io_types=True)  # from a path it infers shapes past 2 GB
    for f in OUT.glob(f"{name}.fp32.onnx*"):
        f.unlink()
    # The weights go in a file of their own, <name>.onnx.data, which the page hands ONNX Runtime Web
    # as external data. Inside the model file they cost the page far more memory (see the README).
    onnx.save_model(graph, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=f"{name}.onnx.data", size_threshold=1024)
    ops = Counter(n.op_type for n in graph.graph.node)
    print(f"{name}: opset {graph.opset_import[0].version}, {sum(ops.values())} nodes {dict(ops.most_common(12))}")

    # The fp16 file against PyTorch in fp32, with the server's preprocessing minus the
    # normalisation, which is now inside the graph.
    tf = transforms.Compose([t for t in preprocess.transforms if not isinstance(t, transforms.Normalize)])
    inputs = [tf(Image.open(p).convert("RGB")) for p in photos] or [torch.rand(3, 224, 224) for _ in range(4)]
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    cosines = []
    for x in inputs:
        with torch.no_grad():
            want = encoder(x[None]).numpy()[0]
        got = session.run(None, {"image": x[None].numpy()})[0][0]
        cosines.append(float(np.dot(want, got) / np.linalg.norm(got)))
    print(f"{name}: {(out.stat().st_size + (OUT / f'{name}.onnx.data').stat().st_size) / 1e6:.0f} MB, "
          f"fp16 vs PyTorch on {len(inputs)} inputs: "
          f"lowest cosine {min(cosines):.5f}")
    if min(cosines) < 0.999:
        raise SystemExit(f"{name}: the fp16 export drifted from PyTorch")
    return model.logit_scale.detach().exp().item()


def write_table(model_id: str, file: str) -> int:
    """Write the label table of `model_id` as fp16 to OUT/<file>; return its width."""
    import numpy as np

    from app.backends.bioclip import _load_labels, _table_file

    src = _table_file(model_id, _load_labels())
    if not src.is_file():
        raise SystemExit(f"No label table for {model_id} and this species list ({src}). "
                         "Identify one photo with that model on the server to build it.")
    table = np.load(src)
    table.astype("<f2").tofile(OUT / file)
    return int(table.shape[1])


def split_big_files() -> dict[str, int]:
    """Cut every file over PART into <file>.0, <file>.1, ...; return {url the page asks for: size}."""
    split = {}
    for f in sorted(p for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > PART):
        data = f.read_bytes()
        for i in range(0, len(data), PART):
            f.with_name(f"{f.name}.{i // PART}").write_bytes(data[i : i + PART])
        f.unlink()
        split[f"models/{f.relative_to(OUT).as_posix()}"] = len(data)
    return split


def main() -> None:
    from app.backends.bioclip import BIOCLIP_25, BioClipMobile, _load_labels
    from app.common_names import load as common_names
    from app.config import get_settings
    from app.labels import scientific_name

    photos = sorted(Path(sys.argv[1]).glob("*.jpg")) if len(sys.argv) > 1 else []
    shutil.rmtree(OUT, ignore_errors=True)  # a clean build: no files or parts left from an earlier one
    OUT.mkdir(parents=True)
    fetch_ort()

    shutil.copyfile(get_settings().models_dir / BioClipMobile.weights, OUT / "bioclip-mobile.onnx")
    models = [{
        "id": "bioclip-mobile", "label": "BioCLIP 2.5 Mobile", "file": "bioclip-mobile.onnx",
        "table": "table-bioclip25.f16", "dim": write_table(BIOCLIP_25, "table-bioclip25.f16"),
        "scale": BioClipMobile.logit_scale,
        "crop": 224 / 255,  # its training: shorter side to 255, centre 224
        "note": NOTES["bioclip-mobile"],
    }]
    for name, label, model_id, table in [
        ("bioclip1", "BioCLIP 1", "hf-hub:imageomics/bioclip", "table-bioclip1.f16"),
        ("bioclip2", "BioCLIP 2", "hf-hub:imageomics/bioclip-2", "table-bioclip2.f16"),
        ("bioclip25", "BioCLIP 2.5", BIOCLIP_25, "table-bioclip25.f16"),  # the Mobile model's table
    ]:
        scale = export_encoder(model_id, name, photos)
        models.append({
            "id": name, "label": label, "file": f"{name}.onnx", "data": f"{name}.onnx.data", "table": table,
            "dim": write_table(model_id, table), "scale": scale,
            "crop": 1.0,  # CLIP: shorter side to 224, centre 224
            "note": NOTES[name],
        })
    for m in models:
        m["bytes"] = sum((OUT / m[k]).stat().st_size for k in ("file", "data") if k in m)

    names = _load_labels()
    common = common_names()
    (OUT / "labels.json").write_text(json.dumps(
        {"names": names, "common": [common.get(scientific_name(n)) for n in names]}, ensure_ascii=False
    ), encoding="utf-8")
    split = split_big_files()
    version = hashlib.sha256()
    for f in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "models.json"):
        version.update(f.relative_to(OUT).as_posix().encode())
        version.update(f.read_bytes())
    (OUT / "models.json").write_text(json.dumps(
        {"version": version.hexdigest()[:12], "split": split, "models": models}, indent=1
    ), encoding="utf-8")
    for m in models:
        print(f"{m['id']:15s} {m['bytes'] / 1e6:6.0f} MB  table {m['dim']}-d  scale {m['scale']:.2f}")


if __name__ == "__main__":
    main()
