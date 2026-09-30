# L.E.A.F.

Lightweight Engine for Assessing Flora: a small self-hosted plant identification service. A FastAPI backend with swappable models serves a web page from the same container, and the page can also run models in the browser, on the device that took the photo. The browser-only version is live at https://leaf.lefthanddev.com.

| Model | What it is | Good for |
|---|---|---|
| `plantnet300k` | ResNet trained on Pl@ntNet-300K (1,081 species) | Fast, small; European garden and wild plants |
| `bioclip` | BioCLIP 2.5, zero-shot against a species list you provide | Anything on your list; heaviest |
| `bioclip2` | BioCLIP 2, same species list | The same answers at half the memory and time; the best all-rounder in the comparison below |
| `bioclip1` | The first BioCLIP (2023), same species list | A fifth of BioCLIP 2.5's memory, about 0.13 s a photo on a CPU |
| `bioclip-mobile` | BioCLIP 2.5 Mobile, a 24 MB model trained to copy BioCLIP 2.5, same species list | Phones and browsers; less accurate |
| `inat21` | EVA-02 Large fine-tuned on iNaturalist 2021 (4,271 plant species) | Worldwide plants, North American natives included |
| `inat21-convnext` | ConvNeXt Large fine-tuned on iNaturalist 2021 (same species) | The same, lighter and faster |
| `mock` | Placeholder results, no model | Testing the API and UI |

The service offers only `bioclip` (BioCLIP 2.5) unless `BACKENDS` names others, for example `BACKENDS=bioclip,inat21-convnext`; `ENABLE_MOCK=1` adds `mock`.

## Quick start (no model files needed)

```bash
ENABLE_MOCK=1 docker compose --profile cpu up --build
ENABLE_MOCK=1 podman-compose --profile cpu up --build   # the same with Podman
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

Four BioCLIP models share one species list:

- `bioclip`: BioCLIP 2.5 Huge by default, `hf-hub:imageomics/bioclip-2.5-vith14`: a ViT-H/14 image encoder and a 24-layer text encoder, 986 million parameters. open_clip downloads only its `open_clip_model.safetensors`, 3.94 GB (3.7 GiB in `models/hf`), and the label cache for 5,035 names adds 21 MB. It takes about 7.7 GiB of RAM on the CPU, or 3.9 GiB of GTT on the Radeon 890M, and loads in about 10 seconds once its label cache exists. `BIOCLIP_MODEL` swaps in another open_clip checkpoint.
- `bioclip2`: BioCLIP 2, `hf-hub:imageomics/bioclip-2` (ViT-L/14, 1.7 GB download, about 3.6 GiB of RAM, about twice as fast as 2.5).
- `bioclip1`: the first BioCLIP, `hf-hub:imageomics/bioclip` (ViT-B/16, 0.6 GB download, about 1.5 GiB of RAM).
- `bioclip-mobile`: [BioCLIP 2.5 Mobile](https://huggingface.co/crazedcodernate/bioclip-2.5-mobile-fastvit) by Nate Hamilton (MIT license), an 11.6-million-parameter FastViT trained to reproduce BioCLIP 2.5's image embeddings on the 4,271 iNaturalist 2021 plants. Its answers land in BioCLIP 2.5's embedding space, so it is scored against BioCLIP 2.5's label table and has no text encoder of its own; that table has to exist first (identify one photo with `bioclip`). It runs on onnxruntime, about 0.6 GiB of RAM. Its author reports it gives BioCLIP 2.5's top answer 72% of the time on 2,000 held-out photos, and warns against using it for edibility or toxicity. The service expects its fp16 file at `models/bioclip-mobile/flora_student_fp16.onnx`:

  ```bash
  mkdir -p models/bioclip-mobile
  curl -L -o models/bioclip-mobile/flora_student_fp16.onnx \
    https://huggingface.co/crazedcodernate/bioclip-2.5-mobile-fastvit/resolve/29b474ea2a5d72b4646f036ead9441e0a22a5c62/flora_student_fp16.onnx
  sha256sum models/bioclip-mobile/flora_student_fp16.onnx   # b152ee0b3fe8f7b6e01f27a580fa74fbec53c0519e4dccaebdb9e289d140c579
  ```

The three open_clip models download from Hugging Face on first use into `models/hf` (needs internet once; afterwards they run offline). When the page offers more than one model, it names the model each answer came from.

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

Label embeddings are computed once per model and list and cached in `models/bioclip/cache`. That first start is slow on CPU: for 5,035 names about 2 minutes for BioCLIP 1, 6 for BioCLIP 2 and 19 for BioCLIP 2.5 on a 12-core Ryzen AI 9 HX 370 (1,019 names: 40 seconds for BioCLIP 2, 3.5 minutes for 2.5). Later starts take about 10 seconds. Copy `models/bioclip/cache` along with the weights when you deploy.

### iNaturalist 2021 classifiers

Two `timm` models fine-tuned on the iNaturalist 2021 competition set. They know 10,000 species of every kind; the service keeps the 4,271 plants, so every answer is a plant. Many North American natives that PlantNet-300K lacks are among them (California poppy, poison oak, coast live oak).

| Menu entry | Folder | Hugging Face repo | Download |
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

### Common names

Each match shows an English common name under the scientific name when `models/common_names.json` has one. Build the file once, from this directory, with internet access; it needs only Python 3:

```bash
MODELS_DIR=./models python3 -m app.common_names
```

It covers the PlantNet-300K species, the BioCLIP list and the iNat21 plants. The 4,271 iNat21 plants take their names from `models/inat21/config.json`; the rest are iNaturalist's preferred English names, looked up one a second because iNaturalist asks API clients to stay under 60 requests a minute. For the 5,035-name list that took 11 minutes and named 4,814 species; the others show only the scientific name. Without the iNat21 config every name is looked up, about 85 minutes. Re-run it after adding species to `labels.txt`: it looks up only names that are not in the file yet, and the running service reads the new file without a restart. The service itself never contacts iNaturalist, and the file lives in `models/`, so it travels with the weights.

### How the models compare

Sixteen photos from Wikipedia and Wikimedia Commons: ten species that are in PlantNet-300K, and six California or garden plants that are not (California poppy, poison oak, coast live oak, dandelion, English ivy, sunflower). Both BioCLIP entries used the 5,035-name list described above.

| Model | Names it can return | Top-1, 10 PlantNet species | Top-1, 6 California/garden | CPU, 12 cores | Radeon 890M | RAM (CPU) |
|---|---|---|---|---|---|---|
| `plantnet300k` | 1,081 | 10 | 0 (none listed) | 22 ms | 9 ms | 0.5 GiB |
| `bioclip` (2.5) | 5,035 | 10 | 6 | 0.96 s | 1.12 s | 7.7 GiB |
| `bioclip2` | 5,035 | 10 | 6 | 0.55 s | 0.48 s | 3.6 GiB |
| `inat21` | 4,271 | 8 | 5 | 1.12 s | 3.7 s | 2.7 GiB |
| `inat21-convnext` | 4,271 | 8 | 5 | 0.63 s | 0.89 s | 2.0 GiB |

BioCLIP 2 is the best all-rounder here: every photo right, like BioCLIP 2.5, at half the memory and about half the time. Sixteen clear photos make a smoke test, not an evaluation.

Times are the server's own `latency_ms` (the model only, not the upload): the warm median of 20 requests through the API of the CPU and ROCm containers under Podman on a Ryzen AI 9 HX 370 laptop. The models run in fp32, which this iGPU does slowly (see AMD GPU notes), so the GPU only helps PlantNet-300K and BioCLIP 2; EVA-02 on the GPU ranged from 1.5 to 9.6 s. Every model named the same plants on the GPU as on the CPU, under Docker as under Podman; one GPU run under Docker was noisier (BioCLIP 2.5 took 1.2 to 4.3 s per photo), so the table uses the Podman runs. RAM is the peak of a CPU process with that one model loaded; with all five loaded on the CPU, as after a compare run, the server held 7.9 GiB and peaked at 9.9 GiB while loading them.

The iNat21 models' two PlantNet "misses" are the ZZ plant, which they don't list, and *Anemone nemorosa*, which they name correctly by its newer name *Anemonoides nemorosa*. Both put dandelion second, behind *Taraxacum erythrospermum*. A PlantCLEF 2024 classifier (DINOv2 ViT-B/14, 7,806 European species, `vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights`) was tested too and left out: 5 of 16 top-1.

The four BioCLIPs on the same 16 photos, through the service's backends on the laptop's CPU (warm median of 20, in a later, quieter run than the table above):

| Model | Top-1, of 16 | Same top-1 as BioCLIP 2.5 | Time per photo | RAM |
|---|---|---|---|---|
| `bioclip` (2.5) | 16 | 16 | 0.77 s | 7.7 GiB |
| `bioclip2` | 16 | 16 | 0.42 s | 3.6 GiB |
| `bioclip1` | 15 | 15 | 0.13 s | 1.5 GiB |
| `bioclip-mobile` | 13 | 13 | 0.05 s | 0.6 GiB |

BioCLIP 1 called the coast live oak a European aspen (the oak came third). The Mobile model named the thistle's genus but not its species (*Cirsium neomexicanum*), and missed the pomegranate still life and the ZZ plant, which is not among the iNaturalist 2021 plants it was trained on.

## Run

```bash
docker compose --profile cpu  up --build -d   # CPU
docker compose --profile rocm up --build -d   # AMD GPU (Radeon 890M etc.)
```

Docker and Podman run the same file, tested with Docker Engine 29.8 (Compose 5.5) and Podman 5.8 (podman-compose 1.6). With Podman, use `podman-compose` in place of `docker compose`. `podman compose`, with a space, hands off to Docker's compose plugin when that is installed, and then fails with `failed to connect to the docker API at unix:///run/user/1000/podman/podman.sock` because Podman's API socket is not running; `PODMAN_COMPOSE_PROVIDER=podman-compose podman compose ...` makes it use podman-compose instead. The ROCm image is large: a 10 GB download that takes 30 GB on disk under Podman and 40 GB under Docker, which keeps the download as well.

When the page opens it loads the model, so the first photo doesn't wait for it, and a bar under the title fills while it loads. The server can't report how far a load has got, so the fill is paced by time, about 70% after 7 seconds, and completes when the model is ready (BioCLIP 2.5 usually takes 7 to 10 seconds on the CPU); a slower load creeps toward 90% and waits there. A photo shows a scan while it is being identified. With one model offered the page shows no model menu and no model names; with several it shows a menu above the photo, a compare option and a heading per model, and a model that isn't ready is greyed out in the menu with what's missing. Identify sits beside Choose photo under the photo, in equal columns (three, with Take photo, on a touch screen). The page doesn't show the device; `GET /api/health` reports it. To offer the model that did best in the comparison above instead, set `BACKENDS=bioclip2`.

### AMD GPU notes

Tested on a Radeon 890M (gfx1150, Ryzen AI 9 HX 370) under Fedora 44, with rootless Podman and with Docker Engine. `Dockerfile.rocm` builds on `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (ROCm 7.2.4, PyTorch 2.10), which supports gfx1150 natively: `HSA_OVERRIDE_GFX_VERSION` is not needed, so leave it unset. To check that the container sees the GPU:

```bash
docker compose --profile rocm exec plant-id-rocm \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

It should print `True 7.2.53211`, and `curl localhost:8000/api/health` should report `ROCm GPU (AMD Radeon 890M Graphics)`.

- `Memory critical error by agent node-0 ... Reason: Memory in use.` followed by a core dump: SELinux stopped the container from mapping `/dev/kfd`. The compose file sets `label=disable` for the ROCm service; `--privileged` also works. The host-wide alternative, `sudo setsebool -P container_use_devices true`, was not tested here. This came up only under Podman: Docker Engine leaves its SELinux support off unless you enable it, and its containers ran unconfined (`spc_t`).
- `Unable to find group render`: the image has no `render` group, so the compose file adds only `video`. On the Fedora 44 test host `/dev/kfd` and `/dev/dri/renderD128` are open to every user anyway; on other hosts check that they exist and that the container can open them.
- Set `DEVICE=cpu` to force the CPU, or `DEVICE=cuda` to fail loudly if no GPU is visible.
- An iGPU keeps the weights in GTT, system memory it borrows: about 8 GiB with all five models loaded, plus 2.8 GB of ordinary RAM for the container.
- This iGPU is slow at fp32, the precision the models run in: a 4096×4096 matrix multiply reached 0.1 TFLOPS in fp32 and 1.6 TFLOPS in fp16. Under fp16 autocast, BioCLIP 2's image encoder took 111 ms instead of 480 and EVA-02 336 ms instead of 1,469, with outputs matching the CPU's (cosine similarity 1.00000). The service does not use fp16 yet. `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` gave mixed results in one run (BioCLIP 2 faster in fp16, EVA-02 slower in both precisions and three times slower in fp32), so it is not set.

## Models on this device

When `models/browser` holds a build (below), the menu also offers models **On this device**: BioCLIP 2.5 Mobile, BioCLIP 2 and BioCLIP 2.5, which run in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.30. Photos never leave the device, and the browser keeps each model after its first download; the bar under the title shows the download as it goes. They use the service's species list and common names, and the service serves their files at `/models/`. A model runs on the GPU through WebGPU when the browser offers it, and otherwise on the CPU through WebAssembly; if the GPU fails to start it, the page moves it to the CPU by itself. One on-device model is kept loaded at a time, and compare mode runs it next to the server's models. Choosing another model during a download stops that download after the part in progress (the parts kept so far stay for next time), and the model loaded before stays usable until the new one starts.

Under the menu, a table shows which devices each model fits best, its download and the memory it takes while it starts (the memory table below, rounded), with the chosen model marked. At :8000 its first row covers the server's models, which fit any device that can reach the server. `build.py` holds the table's text, in `FIT`.

The same page without the server is L.E.A.F. at https://leaf.lefthanddev.com, on Cloudflare. It starts loading a model as soon as it opens, picked for the device: BioCLIP 2.5 Mobile on a phone or tablet (a coarse pointer), with Data Saver on, on a mobile data connection (`navigator.connection.type` is `cellular`), on a slow one (its `effectiveType` is `2g` or `3g`), or when the browser reports under 8 GB of memory (`navigator.deviceMemory`, which Chrome 153 gave as 32 on the laptop); BioCLIP 2 otherwise. The status line says why it picked the small one. Safari and Firefox report neither memory nor connection, so on an iPhone only the touch screen counts, and Chrome reports the connection type only on Android and ChromeOS. BioCLIP 2.5 is never picked for you. At :8000 the server's default model comes first, since it costs the device nothing.

```bash
pip install onnx onnxscript     # for build.py only; onnxruntime is in requirements.txt
.venv/bin/python web/build.py   # writes models/browser/: fp16 ONNX models, label tables, ONNX Runtime Web
python3 web/pages.py            # writes models/pages/: the page and those files, ready for Cloudflare
wrangler deploy --name leaf --assets models/pages --compatibility-date 2026-09-30 --domain leaf.lefthanddev.com
```

`build.py` exports the BioCLIP 2 and 2.5 image encoders to fp16 ONNX with the colour normalisation inside the graph and the weights in a file of their own (`<model>.onnx.data`); on four photos their embeddings matched PyTorch's with a cosine similarity of 0.99998 or better. It copies the Mobile model, writes each model's label table as fp16, and cuts every file over 24 MiB into parts. It reads the label tables from `models/bioclip/cache`, so run each model on the service once first; it says which one is missing. A full build took 6 minutes on the laptop, and the files fill 1.9 GB. The service looks for `models/browser` when it starts, so restart it after the first build.

Chrome 154 on the Ryzen AI 9 HX 370 laptop, the same 16 photos, run on the earlier test page with the same models and preprocessing:

| Model | Download: model + label table | Top-1, of 16 | Same top-1 as the service | CPU, WebAssembly, 4 threads | GPU, WebGPU, Radeon 890M |
|---|---|---|---|---|---|
| BioCLIP 2.5 Mobile | 24 + 10 MB | 12 | 15 | 0.15 s | 0.05 s |
| BioCLIP 2 | 610 + 8 MB | 16 | 16 | 2.1 s | 0.15 s |
| BioCLIP 2.5 | 1,267 + 10 MB | not run | not run | 4 s | 0.8 s |

- Times are medians per photo, including preparing the photo and scoring the 5,035 names. BioCLIP 2.5 came later: its times run from Identify to the answer on three photos (one on the CPU), and it named the two plant photos it saw as the service did. ONNX Runtime Web adds 14 MB for the CPU path or 27 MB for the GPU path.
- From the browser's cache a model was ready in 0.4 s (Mobile) and 1.5 s (BioCLIP 2) on the GPU, and 0.6 and 6.0 s on the CPU. Those downloads came from this machine. From leaf.lefthanddev.com over this laptop's connection, a first visit was ready in 45 s with BioCLIP 2 on the GPU (610 MB downloaded in 39 s) and in 4.5 s with Mobile on the CPU (24 MB in 1.3 s). At 50 Mbps, 610 MB takes about 100 s.
- In fp16 on the GPU, BioCLIP 2 answered faster in the browser (0.15 s) than the service does on the same laptop (0.42 s on its CPU, 0.48 s on its GPU in fp32).
- The one disagreement with the service is the coast live oak, a near-tie for the smaller models, and the page shrinks photos with the browser's canvas rather than PIL. In the browser the Mobile model called it an olive tree. Scores differ too: on the naked-man orchid, Mobile gave *Orchis italica* 66% on the service, 85% in the browser on the CPU and 93% on the GPU.
- WebGPU on Linux with AMD graphics is off by default. Chrome offered the Radeon only when launched with `--enable-unsafe-webgpu --use-angle=vulkan --enable-features=Vulkan` (and `--disable-vulkan-surface` headless); otherwise the page uses the CPU. Windows, macOS, ChromeOS, most Android 12+ phones and Safari 26 have it on by default ([WebGPU implementation status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)), and so did Chrome on an iPhone with iOS 26.6.
- When WebGPU fails to start a model, ONNX Runtime Web's WebGPU build throws a bare number, and trying the CPU in that same build then hung the page at 100% of a core or crashed it (tested by making `requestDevice` fail). So the page moves to the separate CPU-only build and stays on the CPU for the rest of the visit.
- The page needs `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on every response, or WebAssembly runs on one thread. The service sends them, with `Cache-Control: no-cache` so a browser checks for a newer `models.json` instead of mixing an old model list with rebuilt files, and `pages.py` writes a `_headers` file that makes Cloudflare send them.
- The site is a Worker with static assets only, which is what Cloudflare Pages has become: wrangler 4.135 turns `wrangler pages project create` into a Workers deploy. The custom domain overrides the `*.lefthanddev.com` wildcard for `leaf` only. Uploading the full 2 GB took 4 minutes from the laptop, and a later deploy uploads only the files that changed; a few requests fail with Cloudflare's code -1 and go through on wrangler's retries.
- Right after a deploy, Cloudflare answered one BioCLIP 2 part with a 500 twice, then served it. The page tries a request that meets a network error or a 5xx twice more, a second apart.
- Cloudflare serves files up to 25 MiB, so the weights of BioCLIP 2 and 2.5 and the 27 MB WebGPU runtime come in 24 MiB parts, which the page downloads and joins. The browser keeps each part as its own entry: Chrome refused to keep 610 MB as one (500 MB went in), and an interrupted download resumes from the last whole part.
- `models.json` carries a version, a hash of every file in the build. The page names its cache after it, so after a build with other models or species a returning visitor downloads them again and the page deletes the old copies.
- Cloudflare Web Analytics is on for lefthanddev.com, so Cloudflare adds its beacon (`static.cloudflareinsights.com`) to the page. It reports page loads, not photos. To keep the page free of third-party requests, turn it off for this hostname in the Web Analytics settings.
- With tracking prevention on, Safari deletes a site's stored data after 7 days without an interaction, so occasional iPhone users would download the model again.

Memory at its peak while a model starts, in the same Chrome: the page's process, and on the GPU path the GPU buffers, which on a phone come out of the same memory. The second row is the earlier build, with the weights inside the model file:

| Model | GPU path: page + GPU buffers | CPU path: page |
|---|---|---|
| BioCLIP 2.5 Mobile | 0.6 + 0.1 GB | 0.5 GB |
| BioCLIP 2, weights inside the model file | 3.1 + 1.3 GB | 3.7 GB |
| BioCLIP 2 | 1.4 + 1.3 GB | 2.7 GB |
| BioCLIP 2.5 | 2.3 + 2.7 GB | 5.1 GB |

### Why BioCLIP 2 broke on an iPhone

An iPhone with iOS 26.6.2 and Chrome 154 for iOS, which runs on Apple's WebKit like every iPhone browser, loaded Mobile and then BioCLIP 2 on the GPU. Cloudflare's request log shows it downloaded all 25 parts of BioCLIP 2, then reloaded the page again and again for five minutes without asking for another model file, with both the GPU and the CPU runtime. That is what iOS does when it closes a page for using too much memory: WebKit reloads it once and then shows "A problem repeatedly occurred" ([WebKit bug 279637](https://bugs.webkit.org/show_bug.cgi?id=279637)). That build needed 4.4 GB to start BioCLIP 2 on the GPU and 3.7 GB on the CPU (the table above).

With the weights in a file of their own, which ONNX Runtime Web takes as external data, the same start needs 2.7 GB either way. Whether that fits depends on the phone: iOS doesn't publish how much memory one page may use, so the answer is to try it. BioCLIP 2.5 needs about 5 GB, so it is left to computers. If a model still closes the page as it starts, the page remembers it for that tab, and after the reload it says so and starts nothing, instead of starting the model again.

## Put it on the network

There is no login. Anyone who can reach the port can upload images and use your CPU/GPU, so put it behind something. The compose file publishes port 8000 on every network interface; on Fedora Workstation, whose default firewall zone opens ports 1025 to 65535, other machines on your network can reach it. If only a proxy on the same machine should, change the port line to `"127.0.0.1:8000:8000"`. With Caddy:

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
curl -F file=@poppy.jpg -F top_k=1 http://localhost:8000/api/identify
```

```json
{
  "backend": "bioclip",
  "label": "BioCLIP (bioclip-2.5-vith14)",
  "device": "cpu",
  "latency_ms": 838.8,
  "load_ms": null,
  "cold_start": false,
  "image": {"width": 1280, "height": 960},
  "predictions": [{"name": "Eschscholzia californica", "common_name": "Californian poppy", "score": 0.999803}]
}
```

- `GET /api/health`: status, device, how many models are ready
- `GET /api/backends`: each model `BACKENDS` offers, whether it's ready, and why not if it isn't
- `POST /api/load?backend=<id>`: load a model before the first photo (the page calls it when it opens); `load_ms` is `null` if it was already loaded
- `POST /api/identify`: `file` (required), `backend` (defaults to the first ready model), `top_k` (1 to 20, default 5)
- `GET /models/...`: the on-device models and ONNX Runtime Web from `models/browser`, when a build is there

The first request to each model loads it and is slow; `cold_start` and `load_ms` tell you when that happened. Photos over `MAX_UPLOAD_MB` (default 15) are rejected. Scores are softmax probabilities within each model's own label set, not calibrated odds of being right. `common_name` comes from `models/common_names.json` and is `null` when there isn't one.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/models` | Root for weights, labels, caches |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` (ROCm also uses `cuda`) |
| `DEFAULT_BACKEND` | first ready | Pre-selected model in the UI |
| `BACKENDS` | `bioclip` | Models the page and API offer, comma-separated: `plantnet300k`, `bioclip`, `bioclip2`, `bioclip1`, `bioclip-mobile`, `inat21`, `inat21-convnext`. An unknown id shows up as not ready |
| `MAX_UPLOAD_MB` | `15` | Upload size limit |
| `ENABLE_MOCK` | off | Show the placeholder model |
| `PLANTNET_*` | see above | Weights, species files, architecture, pickle opt-in |
| `BIOCLIP_MODEL`, `BIOCLIP_LABELS` | see above | Hub id and label list |

`docker-compose.yml` takes `BACKENDS`, `DEVICE`, `DEFAULT_BACKEND` and `ENABLE_MOCK` from your shell or from a `.env` file beside it, for example `BACKENDS=bioclip,bioclip2 docker compose --profile rocm up -d`. Add any other variable under `environment:` in that file.

## Layout

```
app/main.py              API routes and static UI
app/backends/            one file per model; registry loads them lazily
app/labels.py            label mapping and checkpoint parsing (pure Python)
app/common_names.py      common-name table and the script that builds it
app/static/index.html    the page: the server's models and the on-device ones, no external requests
tests/                   pytest suite; the backend tests skip without PyTorch
web/                     build.py (the on-device models) and pages.py (the static site)
```

To add a model, subclass `Backend` in `app/backends/`, implement `is_available`, `load` and `_predict`, and add it to `_CLASSES` in `app/backends/__init__.py`.

## What has and hasn't been tested

Tested on a Ryzen AI 9 HX 370 laptop with Fedora 44:

- Every model against its real weights (the comparison above), and the BioCLIP label cache across restarts.
- Both images under Podman (`podman-compose`) and under Docker Engine (`docker compose`): the CPU image, and the ROCm image on the Radeon 890M, where all five models ran on the GPU and named the same plants as on the CPU. Docker runs the container as root, so anything it writes into `models/` (a new BioCLIP label cache, a first download) will belong to root; in these runs everything was already there and it wrote nothing.
- The page in Chromium, driven with Playwright: one model and compare mode with all five models, light and dark mode at 1280 and 390 pixels wide, and a 1.7 MB phone photo, which the page shrank to 0.3 MB before upload and kept upright. No console errors, and no requests to anything but the server itself.
- The loading bar and photo scan with BioCLIP 2.5 alone. In the CPU container the bar filled from empty to 73% during an 8.5-second load, then completed and faded without moving anything below it; a failed load (simulated with an error reply) leaves it full and red, with the message under it. The scan covers only the photo for landscape and portrait shots and stops sweeping when the browser asks for reduced motion; the bar still fills then, since it shows progress. With two models offered, the picker, compare mode and model headings still appear.
- `pytest`: the API, label and checkpoint parsing, the PlantNet-300K and iNat21 load and predict code on small random-weight models, and the BioCLIP 2.5 Mobile scoring on a stand-in ONNX model (the model tests skip without PyTorch or onnxruntime). Run it with `pip install -r requirements-dev.txt && pytest`.
- The browser test page that came before, in Chrome 154: all three models on the CPU and on the GPU, first download and reload from the browser's cache (the 16-photo table above).
- The on-device models in the page, in Chrome for Testing 153 driven with Playwright, at :8000 next to the server's models and as the static site served locally: BioCLIP 2 and 2.5 on the GPU and on the CPU, Mobile on the GPU and, after a simulated GPU failure, on the CPU, compare mode with an on-device model, a reload that took every model file from the cache, an emulated Pixel 7, a server that failed each of two model files once with a 500, and a leftover "starting" mark, after which the page explained itself and started nothing. The memory figures come from these runs: peak resident memory of the page's process, and the GPU memory the amdgpu driver reports for Chrome's GPU process. No console errors apart from those test 500s.
- The model menu and the automatic pick, the same way. The menu and buttons at 1280, 412, 375 and 360 pixels wide: equal columns in one row, every label on one line, nothing wider than the screen. The pick with 4 GB of memory reported, Data Saver on, a `cellular` connection (those three set by a test script), DevTools' 3G throttling and an emulated Pixel 7, each of which chose Mobile and said why, and a desktop, which chose BioCLIP 2. On a line throttled to 40 Mbps, Mobile chosen during BioCLIP 2's download, which stopped that download after the part in progress, then BioCLIP 2 chosen again, which took its three kept parts from the cache, fetched the other 22 and released Mobile; a server model chosen during a download, which stopped it; compare mode with the three server models and Mobile on the device. Then the live site, through workers.dev: Mobile on an emulated Pixel 7, which identified the orchid, and BioCLIP 2 starting on a desktop.
- The live site, through the Worker's workers.dev address because the network in use at the time blocked lefthanddev.com: an emulated Pixel 7 (Mobile on the CPU) and a desktop (BioCLIP 2 on the GPU), each followed by a reload from the cache. That run met the 500s above.

**Not tested yet:** taking a photo with "Take photo" (the button shows on an emulated touch screen), drag-and-drop or paste, Docker with its SELinux support turned on, and the on-device models on a real phone since the fix, in Safari or Firefox, or on Android, where the automatic pick would read real memory and connection reports rather than ones set by a script or DevTools. A page really closed for lack of memory wasn't reproduced; only the mark it leaves behind was, and whether iOS keeps that mark across its reload is untested.

## License

The code is MIT licensed (`LICENSE`). The model weights are not part of this repository and keep their own licenses: the BioCLIPs, BioCLIP 2.5 Mobile included, are MIT, and the iNaturalist 2021 classifiers are CC BY-NC 4.0, non-commercial only.

Identification from a photo is a lead, not a verdict. Don't eat or handle a plant based on this alone.
