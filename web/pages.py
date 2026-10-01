"""Put L.E.A.F. into models/pages/ for Cloudflare, then publish it:

    python3 web/pages.py
    wrangler deploy --name leaf --assets models/pages --compatibility-date 2026-09-30 --domain leaf.lefthanddev.com

The site is the service's pages (app/static/: the home page and the app page at app.html) with the
on-device models from web/build.py, and no server: api/backends offers no server models, so the app
page uses the on-device ones. It goes up
as a Worker with static assets only, which is what Cloudflare Pages has become. _headers makes
Cloudflare send the two headers the service sends, without which WebAssembly runs on one thread.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "pages"
HEADERS = "/*\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    # Hard links: the 2 GB of models are not copied again.
    shutil.copytree(ROOT / "models" / "browser", OUT / "models", copy_function=os.link)
    for page in (ROOT / "app" / "static").iterdir():  # index.html, app.html, site.css
        if page.is_file():
            shutil.copyfile(page, OUT / page.name)
    (OUT / "api").mkdir()
    (OUT / "api" / "backends").write_text('{"backends": [], "default": null}\n', encoding="utf-8")
    (OUT / "_headers").write_text(HEADERS, encoding="utf-8")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"{OUT}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB, "
          f"largest {max(p.stat().st_size for p in files) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
