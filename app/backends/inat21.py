"""iNaturalist 2021 classifiers from timm, answering with plants only.

Each model reads a copy of its Hugging Face repo (config.json and model.safetensors) from a
folder under MODELS_DIR; README.md lists the downloads.
"""
from __future__ import annotations

import importlib.util

from PIL import Image

from ..config import get_settings
from ..device import pick_device
from ..labels import read_json
from .base import Backend, BackendUnavailable, Prediction


class INat21(Backend):
    id = "inat21"
    label = "iNat21 EVA-02 Large"
    description = "4,271 plant species from iNaturalist 2021, worldwide, North American natives included."
    folder = "inat21"  # under MODELS_DIR

    def is_available(self) -> tuple[bool, str]:
        if not importlib.util.find_spec("timm"):
            return False, "timm is not installed"
        folder = get_settings().models_dir / self.folder
        for name in ("config.json", "model.safetensors"):
            if not (folder / name).is_file():
                return False, f"Missing {folder / name}"
        return True, ""

    def load(self) -> None:
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config

        ok, reason = self.is_available()
        if not ok:
            raise BackendUnavailable(reason)
        s = get_settings()
        folder = s.models_dir / self.folder
        # The model knows all 10,000 iNat21 species (insects, birds, fungi...). config.json
        # describes each one as "common name, group"; keep the plants so every answer is one.
        cfg = read_json(folder / "config.json")
        names, groups = cfg["label_names"], cfg["label_descriptions"]
        keep = [i for i, n in enumerate(names) if groups[n].endswith(", Plant")]
        self.names = [names[i] for i in keep]

        self.device = pick_device(s.device)
        model = timm.create_model(f"local-dir:{folder}", pretrained=True)
        self.model = model.eval().to(self.device)
        # The preprocessing the model was trained with: input size, crop, normalisation.
        self._tf = create_transform(**resolve_model_data_config(model), is_training=False)
        self._keep = torch.tensor(keep, device=self.device)
        self._torch = torch
        self.loaded = True

    def _predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        torch = self._torch
        x = self._tf(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            # Softmax over the plants only, so scores add up to 1 across the names it can return.
            probs = self.model(x)[0].float()[self._keep].softmax(dim=0)
            values, indices = torch.topk(probs, min(top_k, probs.numel()))
        return [
            Prediction(self.names[i], float(v)) for v, i in zip(values.tolist(), indices.tolist())
        ]


class INat21ConvNeXt(INat21):
    id = "inat21-convnext"
    label = "iNat21 ConvNeXt Large"
    description = "The same 4,271 iNaturalist plants from a lighter network: faster, less memory."
    folder = "inat21-convnext"
