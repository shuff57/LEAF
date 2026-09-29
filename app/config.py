"""Runtime settings, read from environment variables.

Read fresh on every call to get_settings() so tests (and `docker compose`
overrides) can change them without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    device: str  # "auto", "cpu" or "cuda" (ROCm builds of PyTorch also use "cuda")
    max_upload_bytes: int
    enable_mock: bool
    default_backend: str | None

    # PlantNet-300K ResNet
    plantnet_weights: Path
    plantnet_species_json: Path
    plantnet_class_idx_json: Path
    plantnet_arch: str
    plantnet_allow_pickle: bool

    # BioCLIP (zero-shot against a label list)
    bioclip_model: str
    bioclip_labels: Path

    @classmethod
    def from_env(cls) -> "Settings":
        models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
        pn_dir = models_dir / "plantnet300k"
        return cls(
            models_dir=models_dir,
            device=os.environ.get("DEVICE", "auto").strip().lower(),
            max_upload_bytes=int(float(os.environ.get("MAX_UPLOAD_MB", "15")) * 1024 * 1024),
            enable_mock=_flag("ENABLE_MOCK"),
            default_backend=os.environ.get("DEFAULT_BACKEND") or None,
            plantnet_weights=Path(
                os.environ.get("PLANTNET_WEIGHTS", pn_dir / "resnet18_weights_best_acc.tar")
            ),
            plantnet_species_json=Path(
                os.environ.get(
                    "PLANTNET_SPECIES_JSON", pn_dir / "plantnet300K_species_id_2_name.json"
                )
            ),
            plantnet_class_idx_json=Path(
                os.environ.get("PLANTNET_CLASS_IDX_JSON", pn_dir / "class_idx_to_species_id.json")
            ),
            plantnet_arch=os.environ.get("PLANTNET_ARCH", "resnet18"),
            plantnet_allow_pickle=_flag("PLANTNET_ALLOW_PICKLE"),
            bioclip_model=os.environ.get("BIOCLIP_MODEL", "hf-hub:imageomics/bioclip-2"),
            bioclip_labels=Path(
                os.environ.get("BIOCLIP_LABELS", models_dir / "bioclip" / "labels.txt")
            ),
        )


def get_settings() -> Settings:
    return Settings.from_env()
