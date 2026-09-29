from __future__ import annotations

import threading
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Prediction:
    name: str
    score: float  # softmax probability, 0..1


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run (missing weights, missing library, ...)."""


class Backend:
    """One identification model. Subclasses fill in the class attributes and methods."""

    id: str = ""
    label: str = ""
    description: str = ""
    # Hide from the UI entirely when unavailable (used by the mock backend).
    hidden_when_unavailable: bool = False

    def __init__(self) -> None:
        self.loaded = False
        self.device = "cpu"
        self._infer_lock = threading.Lock()

    def is_available(self) -> tuple[bool, str]:
        """Cheap check that does not load the model. Returns (ok, reason_if_not)."""
        raise NotImplementedError

    def load(self) -> None:
        """Load weights onto self.device. Called once, under a lock."""
        raise NotImplementedError

    def _predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        raise NotImplementedError

    def predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        # One inference at a time: keeps memory predictable on an iGPU or small CPU.
        with self._infer_lock:
            return self._predict(image, top_k)
