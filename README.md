# plant-id

A small self-hosted plant identification service: a FastAPI backend with swappable models, plus a test web page served from the same container.

| Model | What it is | Good for |
|---|---|---|
| `plantnet300k` | ResNet trained on Pl@ntNet-300K (1,081 species) | Fast, small; European garden and wild plants |
| `bioclip` | BioCLIP 2.5, zero-shot against a species list you provide | Anything on your list; heaviest |
| `bioclip2` | BioCLIP 2, same species list | The same at half the memory |
| `inat21` | EVA-02 Large fine-tuned on iNaturalist 2021 (4,271 plant species) | Worldwide plants, North American natives included |
| `inat21-convnext` | ConvNeXt Large fine-tuned on iNaturalist 2021 (same species) | The same, lighter and faster |
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

The page offers two BioCLIP checkpoints that share one species list:

- `bioclip`: BioCLIP 2.5 Huge by default, `hf-hub:imageomics/bioclip-2.5-vith14` (ViT-H/14, 3.9 GB download, about 7.7 GiB of RAM). `BIOCLIP_MODEL` swaps in another open_clip checkpoint, for example `hf-hub:imageomics/bioclip` (the first BioCLIP: ViT-B/16, 1.5 GiB, about 0.2 s per photo on CPU).
- `bioclip2`: BioCLIP 2, `hf-hub:imageomics/bioclip-2` (ViT-L/14, 1.7 GB download, about 3.6 GiB, 2-4x faster than 2.5 on CPU).

Both download from Hugging Face on first use into `models/hf` (needs internet once; afterwards they run offline). The page names the checkpoint in use.

BioCLIP only returns species that are on its list, `models/bioclip/labels.txt`, one scientific name per line (`labels.example.txt` shows the format). Without that file it uses the 1,019 PlantNet-300K species (author citations removed, duplicates merged). None of the six California and garden plants in the comparison below is on that list, so no BioCLIP could name them. Adding the 4,271 iNaturalist 2021 plant species from `models/inat21/config.json` (next section) gives 5,035 names, and BioCLIP 2 and 2.5 then named all six. Run this from this directory to write that list (PlantNet names only if the iNat21 config is absent), then append your own species at the end:

```bash
python - > models/bioclip/labels.txt <<'EOF'
import json, os
from app.labels import scientific_name
names = {scientific_name(n) for n in json.load(open("models/plantnet300k/plantnet300K_species_id_2_name.json")).values()}
if os.path.exists("models/inat21/config.json"):
    cfg = json.load(open("models/inat21/config.json"))
    names |= {n for n in cfg["label_names"] if cfg["label_descriptions"][n].endswith(", Plant")}
print("\n".join(sorted(names)))
EOF
```

Label embeddings are computed once per model and list and cached in `models/bioclip/cache`. That first start is slow on CPU: for 5,035 names about 6 minutes for BioCLIP 2 and 19 minutes for BioCLIP 2.5 on a 12-core Ryzen AI 9 HX 370 (1,019 names: 40 seconds and 3.5 minutes). Later starts take about 10 seconds. Copy `models/bioclip/cache` along with the weights when you deploy.

### iNaturalist 2021 classifiers

Two `timm` models fine-tuned on the iNaturalist 2021 competition set. They know 10,000 species of every kind; the service keeps the 4,271 plants, so every answer is a plant. Many North American natives that PlantNet-300K lacks are among them (California poppy, poison oak, coast live oak).

| Picker entry | Folder | Hugging Face repo | Download |
|---|---|---|---|
| `inat21`, EVA-02 Large | `models/inat21/` | `timm/eva02_large_patch14_clip_336.merged2b_ft_inat21` | 1.26 GB |
| `inat21-convnext`, ConvNeXt Large | `models/inat21-convnext/` | `timm/convnext_large_mlp.laion2b_ft_augreg_inat21` | 0.86 GB |

Each folder holds the repo's `config.json` (it carries the species names) and `model.safetensors`:

```bash
H=https://huggingface.co/timm
mkdir -p models/inat21 models/inat21-convnext
curl -L -o models/inat21/config.json                $H/eva02_large_patch14_clip_336.merged2b_ft_inat21/resolve/main/config.json
curl -L -o models/inat21/model.safetensors          $H/eva02_large_patch14_clip_336.merged2b_ft_inat21/resolve/main/model.safetensors
curl -L -o models/inat21-convnext/config.json       $H/convnext_large_mlp.laion2b_ft_augreg_inat21/resolve/main/config.json
curl -L -o models/inat21-convnext/model.safetensors $H/convnext_large_mlp.laion2b_ft_augreg_inat21/resolve/main/model.safetensors
```

Their license is CC BY-NC 4.0: non-commercial use only.

### How the models compare

Sixteen photos from Wikipedia and Wikimedia Commons: ten species that are in PlantNet-300K, and six California or garden plants that are not (California poppy, poison oak, coast live oak, dandelion, English ivy, sunflower). CPU only, on a 12-core Ryzen AI 9 HX 370, one run per model while other programs were busy, so read the times as rough: they varied up to 2x between runs.

| Model | Names it can return | Top-1, 10 PlantNet species | Top-1, 6 California/garden | Time per photo | RAM |
|---|---|---|---|---|---|
| `plantnet300k` | 1,081 | 10 | 0 (none listed) | 0.04 s | 0.5 GiB |
| `bioclip` (2.5), 5,035-name list | 5,035 | 10 | 6 | 1.9 s | 7.7 GiB |
| `bioclip2`, 5,035-name list | 5,035 | 10 | 6 | 1.2 s | 3.6 GiB |
| `inat21` | 4,271 | 8 | 5 | 1.8 s | 2.7 GiB |
| `inat21-convnext` | 4,271 | 8 | 5 | 1.2 s | 2.0 GiB |

Time per photo is the warm median of 20 runs; RAM is the peak of a process with that one model loaded. The iNat21 models' two PlantNet "misses" are the ZZ plant, which they don't list, and *Anemone nemorosa*, which they name correctly by its newer name *Anemonoides nemorosa*. Both put dandelion second, behind *Taraxacum erythrospermum*. A PlantCLEF 2024 classifier (DINOv2 ViT-B/14, 7,806 European species, `vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights`) was tested too and left out: 5 of 16 top-1.

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
