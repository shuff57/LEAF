"""Label helpers. Pure Python so they can be tested without PyTorch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_class_names(
    species_names: dict[str, str],
    class_idx_to_species_id: dict[str, Any] | None,
    n_classes: int,
) -> list[str]:
    """Return the species name for each class index 0..n_classes-1.

    With the official class_idx_to_species_id.json the mapping is explicit.
    Without it we fall back to torchvision's ImageFolder convention: class i is
    the i-th species id when the ids are sorted *as strings*. Sorting them
    numerically instead silently mislabels every class, so don't.
    """
    if class_idx_to_species_id is not None:
        try:
            return [
                species_names[str(class_idx_to_species_id[str(i)])] for i in range(n_classes)
            ]
        except KeyError as missing:
            raise ValueError(
                f"class-index mapping does not cover the model: missing key {missing}"
            ) from None

    ids = sorted(species_names.keys())
    if len(ids) != n_classes:
        raise ValueError(
            f"the species file lists {len(ids)} species but the model has {n_classes} classes; "
            "provide class_idx_to_species_id.json"
        )
    return [species_names[i] for i in ids]


def extract_state_dict(raw: Any) -> dict[str, Any]:
    """Accept a bare state dict or a training checkpoint that wraps one."""
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is not a dict; expected a state dict or a training checkpoint")
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(raw.get(key), dict):
            raw = raw[key]
            break
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    if "fc.weight" not in state:
        raise ValueError(
            "checkpoint has no 'fc.weight' entry; this loader only supports ResNet-style models"
        )
    return state


def read_label_lines(path: Path) -> list[str]:
    """One label per line. Blank lines and lines starting with # are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    labels = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not labels:
        raise ValueError(f"{path} contains no labels")
    return labels
