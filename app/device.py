"""Pick a torch device and describe the runtime. torch is imported lazily."""
from __future__ import annotations

from typing import Any


def pick_device(preference: str = "auto") -> str:
    """Return "cuda" or "cpu". ROCm builds of PyTorch report AMD GPUs as "cuda"."""
    import torch

    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference in ("cuda", "gpu"):
        raise RuntimeError(
            "DEVICE=cuda was requested but PyTorch sees no GPU. "
            "For AMD, check that /dev/kfd and /dev/dri are passed into the container."
        )
    return "cpu"


def describe_runtime(preference: str = "auto") -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None, "device": "cpu", "device_label": "cpu (PyTorch not installed)"}

    hip = getattr(torch.version, "hip", None)
    info: dict[str, Any] = {"torch": torch.__version__, "hip": hip, "gpu_visible": False}
    if preference != "cpu" and torch.cuda.is_available():
        kind = "ROCm" if hip else "CUDA"
        name = torch.cuda.get_device_name(0)
        info.update(gpu_visible=True, device="cuda", device_label=f"{kind} GPU ({name})")
    else:
        info.update(device="cpu", device_label="cpu")
    return info
