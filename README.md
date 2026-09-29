# plant-id

A small self-hosted plant identification service: a FastAPI backend with swappable models, plus a test web page served from the same container.

| Model | What it is | Good for |
|---|---|---|
| `plantnet300k` | ResNet trained on Pl@ntNet-300K (1,081 species) | Fast, small; European garden and wild plants |
| `bioclip` | BioCLIP 2.5, zero-shot against a species list you provide | Broader coverage; heavier |
| `mock` | Placeholder results, no model | Testing the API and UI |

## Quick start (no model files needed)

```bash
docker compose --profile cpu build
ENABLE_MOCK=1 docker compose --profile cpu up
```

Open http://localhost:8000, drop in a photo, choose **Identify**. The mock model returns fixed placeholder names; it only proves the plumbing works.

Without Docker: `pip install -r requirements.txt` (plus a PyTorch build), then `MODELS_DIR=./models HF_HOME=./models/hf ENABLE_MOCK=1 uvicorn app.main:app --port 8000`. `HF_HOME` keeps the BioCLIP download in `models/hf`, as in the container.

## Add the real models

Everything model-related lives in `./models`, mounted at `/models`.

### PlantNet-300K ResNet

Put these in `models/plantnet300k/` (these names are the service's defaults):

| File | Download with `curl -L -o <file> '<url>'` |
|---|---|
| `resnet18_weights_best_acc.tar` (94 MB) | `https://seafile.plantnet.org/d/01ab6658dad6447c95ae/files/?p=/resnet18_weights_best_acc.tar&dl=1` |
| `plantnet300K_species_id_2_name.json` | `https://seafile.plantnet.org/d/bed81bc15e8944969cf6/files/?p=/plantnet300K_species_id_2_name.json&dl=1` |
| `class_idx_to_species_id.json` | `https://seafile.plantnet.org/d/bed81bc15e8944969cf6/files/?p=/class_idx_to_species_id.json&dl=1` |

These are the "pre-trained models" and "metadata files" shares linked from the [PlantNet-300K README](https://github.com/plantnet/PlantNet-300K). Share links can be re-created, so check the files against the SHA-256 sums of the verified copies:

```
72140ea713acfe712914a7ee29494aad7b8cb1839091902750ddcee40a9ea150  resnet18_weights_best_acc.tar
9c8185db8ebb75958ffa77c8c1b1012b8e3dd493fc64c7cfb33f2049df084d23  plantnet300K_species_id_2_name.json
12139cf778d9c960c53c484be4d37f100928e52e27d60d39f4897862a58fe2ec  class_idx_to_species_id.json
```

For other file names set `PLANTNET_WEIGHTS`, `PLANTNET_SPECIES_JSON` and `PLANTNET_CLASS_IDX_JSON`. For another ResNet depth from the same share set `PLANTNET_ARCH` (for example `resnet50`).

The official checkpoint loads in PyTorch's safe mode (`weights_only=True`), so it does not need `PLANTNET_ALLOW_PICKLE`. If another checkpoint fails with a message about "safe mode", it contains non-tensor objects; only if you trust the file, set `PLANTNET_ALLOW_PICKLE=1`.

Without `class_idx_to_species_id.json` the service assumes class order equals species ids sorted **as strings** (torchvision's ImageFolder convention, which the official file matches exactly). Sorting numerically would mislabel every class.

The 1,081 species were picked for the dataset, not for being common: dandelion, daisy, ivy and sunflower are not among them, so this model can never return them. Use BioCLIP with your own label list for those.

### BioCLIP

- The default is BioCLIP 2.5 Huge, `hf-hub:imageomics/bioclip-2.5-vith14` (ViT-H/14, 3.9 GB download, about 7.7 GiB of RAM). `BIOCLIP_MODEL=hf-hub:imageomics/bioclip-2` selects BioCLIP 2 (ViT-L/14, 1.7 GB, about 3.6 GiB, 4x faster on CPU). The page names the checkpoint in use.
- The model downloads from Hugging Face on first use into `models/hf` (needs internet once; afterwards it runs offline).
- BioCLIP only returns species that are on its list, `models/bioclip/labels.txt`, one scientific name per line (`labels.example.txt` shows the format). Without that file it uses the PlantNet-300K species with author citations removed and duplicates merged (1,019 names). To extend that list, write it out and append your own species:

  ```bash
  python -c "import json; from app.labels import scientific_name; print('\n'.join(sorted({scientific_name(n) for n in json.load(open('models/plantnet300k/plantnet300K_species_id_2_name.json')).values()})))" > models/bioclip/labels.txt
  ```

- Label embeddings are computed once per model and label list and cached in `models/bioclip/cache`. The first start with a new list is slow on CPU (1,019 names: 3.5 minutes for BioCLIP 2.5, 40 seconds for BioCLIP 2 on a 12-core Ryzen AI 9 HX 370); later starts take about 10 seconds.

## Run

```bash
docker compose --profile cpu  up --build -d   # CPU
docker compose --profile rocm up --build -d   # AMD GPU (Radeon 890M etc.)
```

The page shows which device is in use and which models are ready. A model that isn't ready says what's missing.

### AMD GPU notes (untested)

`Dockerfile.rocm` is built from ROCm/PyTorch conventions; I had no AMD hardware to try it on. If the page says `cpu` when you expected the GPU:

```bash
docker compose --profile rocm exec plant-id-rocm \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

- `False`: check that `/dev/kfd` and `/dev/dri` exist on the host and that your user is in the `video` and `render` groups.
- RDNA 3.5 GPUs like the 890M need a ROCm release that supports them. Reports from other projects say ROCm 6.4.3 does not and 6.4.4 or newer does, and 7.2 supports them natively. Make sure the `rocm/pytorch` tag you pull is new enough; older stacks needed `HSA_OVERRIDE_GFX_VERSION=11.0.0` (commented out in `docker-compose.yml`).
- Set `DEVICE=cpu` to force the CPU, or `DEVICE=cuda` to fail loudly if no GPU is visible.

## Put it on the network

There is no login. Anyone who can reach the port can upload images and use your CPU/GPU, so put it behind something. With Caddy:

```
plants.example.com {
    basic_auth {
        steven <bcrypt hash from `caddy hash-password`>
    }
    reverse_proxy plant-id:8000
}
```

Or publish it through a Cloudflare tunnel and protect it with Cloudflare Access. The page uses relative URLs, so it also works under a sub-path if your proxy strips the prefix.

## API

Interactive docs at `/api/docs`.

```bash
curl -F file=@fern.jpg -F backend=plantnet300k -F top_k=5 http://localhost:8000/api/identify
```

```json
{
  "backend": "plantnet300k",
  "label": "PlantNet-300K ResNet",
  "device": "cuda",
  "latency_ms": 41.2,
  "load_ms": null,
  "cold_start": false,
  "image": {"width": 1536, "height": 1152},
  "predictions": [{"name": "Pteridium aquilinum (L.) Kuhn", "score": 0.81}]
}
```

- `GET /api/health`: status, device, how many models are ready
- `GET /api/backends`: each model, whether it's ready, and why not if it isn't
- `POST /api/identify`: `file` (required), `backend` (defaults to the first ready model), `top_k` (1 to 20, default 5)

The first request to each model loads it and is slow; `cold_start` and `load_ms` tell you when that happened. Photos over `MAX_UPLOAD_MB` (default 15) are rejected. Scores are softmax probabilities within each model's own label set, not calibrated odds of being right.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/models` | Root for weights, labels, caches |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` (ROCm also uses `cuda`) |
| `DEFAULT_BACKEND` | first ready | Pre-selected model in the UI |
| `MAX_UPLOAD_MB` | `15` | Upload size limit |
| `ENABLE_MOCK` | off | Show the placeholder model |
| `PLANTNET_*` | see above | Weights, species files, architecture, pickle opt-in |
| `BIOCLIP_MODEL`, `BIOCLIP_LABELS` | see above | Hub id and label list |

## Layout

```
app/main.py              API routes and static UI
app/backends/            one file per model; registry loads them lazily
app/labels.py            label mapping and checkpoint parsing (pure Python)
app/static/index.html    the test page, no external requests
tests/                   pytest suite (runs with the mock backend, no PyTorch needed)
```

To add a model, subclass `Backend` in `app/backends/`, implement `is_available`, `load` and `_predict`, and add it to `_CLASSES` in `app/backends/__init__.py`.

## What has and hasn't been tested

Tested: the API (upload handling, error cases, model registry, clamping, cold-start reporting), label and checkpoint parsing, a live server smoke test, and the web page's interaction flow in a simulated DOM. Run the suite with `pip install -r requirements-dev.txt && pytest`.

**Not tested:** the PlantNet-300K and BioCLIP backends against real weights, the ROCm image, and the page in a real browser. My environment had no PyTorch, model downloads or GPU. Expect to fix small things on first run (most likely weight filenames or the ROCm base image tag); the error messages are written to point at the cause.

Identification from a photo is a lead, not a verdict. Don't eat or handle a plant based on this alone.
