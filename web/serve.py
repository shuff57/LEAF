"""Serve the browser test on http://127.0.0.1:8002 (python3 web/serve.py [port]).

The page comes from web/, the models and ONNX Runtime Web from models/browser/ (web/build.py
writes them). Every response carries the two headers that make the page cross-origin isolated,
which ONNX Runtime Web needs to run on more than one CPU thread.
"""
from __future__ import annotations

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
    }

    def translate_path(self, path: str) -> str:
        if path.startswith("/models/"):
            self.directory, path = str(ROOT / "models" / "browser"), path[len("/models") :]
        elif path.startswith("/ort/"):
            self.directory = str(ROOT / "models" / "browser")
        else:
            self.directory = str(ROOT / "web")
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    print(f"http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
