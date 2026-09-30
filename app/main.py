from __future__ import annotations

import io
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import backends, common_names
from .backends import Backend, BackendUnavailable
from .config import get_settings
from .device import describe_runtime
from .labels import scientific_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("plant-id")

app = FastAPI(title="plant-id", docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def isolate(request: Request, call_next):
    """Cross-origin isolation: without it the page's on-device models run on one CPU thread.
    And no-cache, so a browser checks for a newer page or models.json instead of mixing an old
    model list with rebuilt model files; the page keeps the model files in its own cache anyway."""
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers.setdefault("Cache-Control", "no-cache")
    return response

MAX_SIDE = 1536  # photos are downscaled before inference; the models resize anyway


@app.get("/api/health")
def health() -> dict:
    listing = backends.list_backends()
    return {
        "status": "ok",
        "runtime": describe_runtime(get_settings().device),
        "models_ready": sum(1 for b in listing if b["available"]),
        "models_total": len(listing),
    }


@app.get("/api/backends")
def list_backends() -> dict:
    listing = backends.list_backends()
    default = get_settings().default_backend
    if default not in {b["id"] for b in listing if b["available"]}:
        default = next((b["id"] for b in listing if b["available"]), None)
    return {"backends": listing, "default": default}


def _decode(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=415,
            detail="That file is not an image the server can read. Use a JPEG, PNG or WebP photo.",
        ) from None
    img = ImageOps.exif_transpose(img).convert("RGB")  # phones store rotation in EXIF
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def _ready(backend: str | None) -> tuple[Backend, float | None]:
    """Load `backend` (default: the first ready model), or raise the HTTP error that says why not."""
    if not backend:
        backend = next((b["id"] for b in backends.list_backends() if b["available"]), None)
        if backend is None:
            raise HTTPException(
                status_code=503,
                detail="No models are ready. Check the model files described in the README.",
            )
    try:
        return backends.get_ready(backend)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {backend}") from None
    except BackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception:
        log.exception("Failed to load %s", backend)
        raise HTTPException(
            status_code=500, detail=f"{backend} failed to load. See the server log."
        ) from None


@app.post("/api/load")
def load_model(backend: str | None = None) -> dict:
    """Load a model before the first photo arrives; the page calls this when it opens."""
    model, load_ms = _ready(backend)
    return {
        "backend": model.id,
        "label": model.label,
        "device": model.device,
        "load_ms": None if load_ms is None else round(load_ms, 1),
    }


@app.post("/api/identify")
def identify(
    file: UploadFile = File(...),
    backend: str | None = Form(None),
    top_k: int = Form(5),
) -> dict:
    settings = get_settings()

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Photo is larger than {settings.max_upload_bytes // (1024 * 1024)} MB. "
            "Use a smaller photo.",
        )
    image = _decode(data)
    top_k = max(1, min(int(top_k), 20))

    model, load_ms = _ready(backend)

    try:
        t0 = time.perf_counter()
        preds = model.predict(image, top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception:
        log.exception("Inference failed for %s", backend)
        raise HTTPException(
            status_code=500, detail="Identification failed. See the server log."
        ) from None

    common = common_names.load()
    return {
        "backend": model.id,
        "label": model.label,
        "device": model.device,
        "latency_ms": round(latency_ms, 1),
        "load_ms": None if load_ms is None else round(load_ms, 1),
        "cold_start": load_ms is not None,
        "image": {"width": image.width, "height": image.height},
        "predictions": [
            {
                "name": p.name,
                "common_name": common.get(scientific_name(p.name)),
                "score": round(p.score, 6),
            }
            for p in preds
        ],
    }


# The on-device models and ONNX Runtime Web, which web/build.py writes; read when the service starts.
if (get_settings().models_dir / "browser").is_dir():
    app.mount("/models", StaticFiles(directory=get_settings().models_dir / "browser"), name="models")

# Static UI last, so /api/* routes win.
app.mount(
    "/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static"
)
