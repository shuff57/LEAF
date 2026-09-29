"""BioCLIP: zero-shot classification against a list of species names.

The image encoder runs per request. Text embeddings for the label list are
computed once and cached on disk, so restarts are quick.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging

from PIL import Image

from ..config import get_settings
from ..device import pick_device
from ..labels import read_json, read_label_lines, scientific_name
from .base import Backend, BackendUnavailable, Prediction

log = logging.getLogger(__name__)

PROMPT = "a photo of {}."


def _load_labels() -> list[str]:
    """BIOCLIP_LABELS if present, otherwise the PlantNet-300K species without author citations."""
    s = get_settings()
    if s.bioclip_labels.is_file():
        return read_label_lines(s.bioclip_labels)
    if s.plantnet_species_json.is_file():
        return sorted({scientific_name(n) for n in read_json(s.plantnet_species_json).values()})
    raise BackendUnavailable(
        f"No label list. Create {s.bioclip_labels} (one species per line) "
        "or provide the PlantNet-300K species file."
    )


class BioClip(Backend):
    id = "bioclip"
    description = "Zero-shot over your own species list. Broader coverage, heavier to run."
    hub_id = ""  # empty: use BIOCLIP_MODEL

    def __init__(self) -> None:
        super().__init__()
        self.model_id = self.hub_id or get_settings().bioclip_model
        # Name the checkpoint, so the picker and each result say which BioCLIP answered.
        self.label = f"BioCLIP ({self.model_id.rsplit('/', 1)[-1]})"

    def is_available(self) -> tuple[bool, str]:
        s = get_settings()
        if not importlib.util.find_spec("open_clip"):
            return False, "open_clip_torch is not installed"
        if not (s.bioclip_labels.is_file() or s.plantnet_species_json.is_file()):
            return False, f"No label list at {s.bioclip_labels}"
        return True, ""

    def load(self) -> None:
        import numpy as np
        import open_clip
        import torch

        s = get_settings()
        ok, reason = self.is_available()
        if not ok:
            raise BackendUnavailable(reason)

        self.device = pick_device(s.device)
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_id)
        tokenizer = open_clip.get_tokenizer(self.model_id)
        model = model.eval().to(self.device)

        self.names = _load_labels()
        key = hashlib.sha1(
            (self.model_id + "\n" + "\n".join(self.names)).encode("utf-8")
        ).hexdigest()[:16]
        cache = s.models_dir / "bioclip" / "cache" / f"text-{key}.npy"

        if cache.is_file():
            text = torch.from_numpy(np.load(cache))
        else:
            log.info("Encoding %d labels (one-time, cached to %s)", len(self.names), cache)
            chunks = []
            with torch.inference_mode():
                for i in range(0, len(self.names), 256):
                    batch = [PROMPT.format(n) for n in self.names[i : i + 256]]
                    feats = model.encode_text(tokenizer(batch).to(self.device))
                    chunks.append((feats / feats.norm(dim=-1, keepdim=True)).float().cpu())
            text = torch.cat(chunks)
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, text.numpy())
            except OSError as exc:  # read-only volume is fine, just slower next start
                log.warning("Could not write text-embedding cache: %s", exc)

        self.model = model
        self.preprocess = preprocess
        self.text = text.to(self.device)
        self._torch = torch
        self.loaded = True

    def _predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        torch = self._torch
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_image(x).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp().float() * feats @ self.text.T
            probs = logits.softmax(dim=-1)[0]
            values, indices = torch.topk(probs, min(top_k, probs.numel()))
        return [
            Prediction(self.names[i], float(v)) for v, i in zip(values.tolist(), indices.tolist())
        ]


class BioClip2(BioClip):
    id = "bioclip2"
    hub_id = "hf-hub:imageomics/bioclip-2"
    description = "The previous BioCLIP, same species list: half the memory and about twice as fast."
