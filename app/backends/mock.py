"""Fake backend so you can exercise the API and UI without any model files.

Enable with ENABLE_MOCK=1. Results are deterministic per image but meaningless.
"""
from __future__ import annotations

import hashlib
import time

from PIL import Image

from ..config import get_settings
from .base import Backend, Prediction

_NAMES = [
    "Quercus robur L.",
    "Eschscholzia californica Cham.",
    "Taraxacum officinale F.H.Wigg.",
    "Lavandula angustifolia Mill.",
    "Acer palmatum Thunb.",
    "Rosa canina L.",
    "Trifolium repens L.",
    "Bellis perennis L.",
    "Hedera helix L.",
    "Ranunculus acris L.",
]


class Mock(Backend):
    id = "mock"
    label = "Mock (no model)"
    description = "Deterministic placeholder results for testing. Not real identification."
    hidden_when_unavailable = True

    def is_available(self) -> tuple[bool, str]:
        return (True, "") if get_settings().enable_mock else (False, "ENABLE_MOCK is not set")

    def load(self) -> None:
        time.sleep(0.05)
        self.loaded = True

    def _predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        digest = hashlib.sha256(image.tobytes()[:65536]).digest()
        order = sorted(range(len(_NAMES)), key=lambda i: digest[i])
        weights = [0.55, 0.2, 0.1, 0.06, 0.04, 0.02, 0.01, 0.01, 0.005, 0.005]
        return [Prediction(_NAMES[i], w) for i, w in zip(order, weights)][:top_k]
