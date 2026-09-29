"""ResNet trained on Pl@ntNet-300K (1,081 species).

Weights and the species files come from https://github.com/plantnet/PlantNet-300K
(see README.md for where to put them).
"""
from __future__ import annotations

import importlib.util
import logging

from PIL import Image

from ..config import get_settings
from ..device import pick_device
from ..labels import build_class_names, extract_state_dict, read_json
from .base import Backend, BackendUnavailable, Prediction

log = logging.getLogger(__name__)


def _load_checkpoint(path, allow_pickle: bool):
    import pickle

    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        if not allow_pickle:
            raise BackendUnavailable(
                f"{path.name} contains non-tensor objects, so PyTorch refused to load it in safe "
                "mode. If the file came from the official PlantNet-300K release and you trust it, "
                "set PLANTNET_ALLOW_PICKLE=1."
            ) from None
        log.warning("Loading %s with weights_only=False (PLANTNET_ALLOW_PICKLE=1)", path)
        return torch.load(path, map_location="cpu", weights_only=False)


class PlantNet300K(Backend):
    id = "plantnet300k"
    label = "PlantNet-300K ResNet"
    description = "1,081 species. Small and fast; strongest on European garden and wild plants."

    def is_available(self) -> tuple[bool, str]:
        s = get_settings()
        if not (importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")):
            return False, "PyTorch and torchvision are not installed"
        for path, what in ((s.plantnet_weights, "weights"), (s.plantnet_species_json, "species file")):
            if not path.is_file():
                return False, f"Missing {what}: {path}"
        return True, ""

    def load(self) -> None:
        import torch
        from torch import nn
        from torchvision import models, transforms

        s = get_settings()
        ok, reason = self.is_available()
        if not ok:
            raise BackendUnavailable(reason)

        state = extract_state_dict(_load_checkpoint(s.plantnet_weights, s.plantnet_allow_pickle))
        n_classes = int(state["fc.weight"].shape[0])

        species = read_json(s.plantnet_species_json)
        class_idx = (
            read_json(s.plantnet_class_idx_json) if s.plantnet_class_idx_json.is_file() else None
        )
        self.names = build_class_names(species, class_idx, n_classes)
        if class_idx is None:
            log.warning(
                "No class_idx_to_species_id.json found; assuming class order = sorted species ids"
            )

        if not hasattr(models, s.plantnet_arch):
            raise BackendUnavailable(f"Unknown PLANTNET_ARCH: {s.plantnet_arch}")
        model = getattr(models, s.plantnet_arch)(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model.load_state_dict(state)

        self.device = pick_device(s.device)
        self.model = model.eval().to(self.device)
        self._tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._torch = torch
        self.loaded = True

    def _predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        torch = self._torch
        x = self._tf(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=1)[0]
            values, indices = torch.topk(probs, min(top_k, probs.numel()))
        return [
            Prediction(self.names[i], float(v)) for v, i in zip(values.tolist(), indices.tolist())
        ]
