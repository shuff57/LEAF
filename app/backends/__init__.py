"""Backend registry. Models load lazily on first use, once, under a lock."""
from __future__ import annotations

import threading
import time

from .base import Backend, BackendUnavailable
from .bioclip import BioClip
from .mock import Mock
from .plantnet300k import PlantNet300K

_CLASSES: list[type[Backend]] = [PlantNet300K, BioClip, Mock]
_instances: dict[str, Backend] = {}
_locks: dict[str, threading.Lock] = {c.id: threading.Lock() for c in _CLASSES}
_registry_lock = threading.Lock()


def _instance(backend_id: str) -> Backend:
    with _registry_lock:
        if backend_id not in _instances:
            cls = next((c for c in _CLASSES if c.id == backend_id), None)
            if cls is None:
                raise KeyError(backend_id)
            _instances[backend_id] = cls()
        return _instances[backend_id]


def reset() -> None:
    """Forget loaded models (used by tests)."""
    with _registry_lock:
        _instances.clear()


def list_backends() -> list[dict]:
    out = []
    for cls in _CLASSES:
        b = _instance(cls.id)
        ok, reason = b.is_available()
        if not ok and cls.hidden_when_unavailable:
            continue
        out.append(
            {
                "id": cls.id,
                "label": cls.label,
                "description": cls.description,
                "available": ok,
                "reason": reason,
                "loaded": b.loaded,
            }
        )
    return out


def get_ready(backend_id: str) -> tuple[Backend, float | None]:
    """Return a loaded backend and the load time in ms (None if it was already loaded)."""
    b = _instance(backend_id)  # KeyError for unknown ids
    with _locks[backend_id]:
        if b.loaded:
            return b, None
        ok, reason = b.is_available()
        if not ok:
            raise BackendUnavailable(reason)
        t0 = time.perf_counter()
        b.load()
        return b, (time.perf_counter() - t0) * 1000


__all__ = ["Backend", "BackendUnavailable", "get_ready", "list_backends", "reset"]
