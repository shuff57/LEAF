"""English common names for the species the models can return.

The page shows them under the scientific names. They come from iNaturalist, gathered once into
MODELS_DIR/common_names.json, so the service never needs the network for them. Build or extend
that file with:

    MODELS_DIR=./models python3 -m app.common_names

It covers the PlantNet-300K species, the BioCLIP label list and the iNat21 plants. The iNat21
models' config.json already carries iNaturalist's names for their species; the others are looked
up one a second, because iNaturalist asks API clients to stay under 60 requests a minute.
Re-running looks up only species that are not in the file yet.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from .config import get_settings
from .labels import read_json, read_label_lines, scientific_name

log = logging.getLogger(__name__)

API = "https://api.inaturalist.org/v1/taxa"
PLANTAE = 47126  # iNaturalist's taxon id for the plant kingdom


@lru_cache(maxsize=1)
def _read(path: Path, mtime: float) -> dict[str, str | None]:
    return read_json(path)


def load() -> dict[str, str | None]:
    """Scientific name -> common name or None. Re-read when the file changes; {} without it."""
    path = get_settings().models_dir / "common_names.json"
    try:
        return _read(path, path.stat().st_mtime)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:  # a broken file must not break identification
        log.warning("Ignoring %s: %s", path, exc)
        return {}


def _capitalized(name: str) -> str:
    return name[:1].upper() + name[1:]


def pick(results: list[dict], name: str) -> str | None:
    """Common name of the taxon called `name`, else of the accepted taxon `name` is a synonym of.

    Search results can list a broader group first (the section Taraxacum for "Taraxacum
    officinale"), so match on the name instead of taking the top hit.
    """
    for key in ("name", "matched_term"):
        for r in results:
            if (r.get(key) or "").lower() == name.lower():
                common = r.get("preferred_common_name")
                return _capitalized(common) if common else None
    return None


def lookup(name: str) -> str | None:
    query = urllib.parse.urlencode({"q": name, "taxon_id": PLANTAE, "locale": "en", "per_page": 10})
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": "plant-id (self-hosted plant identifier)"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return pick(json.load(response)["results"], name)


def _save(path: Path, table: dict[str, str | None]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)  # atomic, so the running service never reads half a file


def main() -> None:
    s = get_settings()
    path = s.models_dir / "common_names.json"
    table: dict[str, str | None] = read_json(path) if path.is_file() else {}
    wanted: set[str] = set()
    if s.plantnet_species_json.is_file():
        wanted |= {scientific_name(n) for n in read_json(s.plantnet_species_json).values()}
    if s.bioclip_labels.is_file():
        wanted |= {scientific_name(n) for n in read_label_lines(s.bioclip_labels)}
    inat21 = s.models_dir / "inat21" / "config.json"
    if inat21.is_file():
        # It describes each of its species as "common name, group": the plants need no lookup.
        for name, description in read_json(inat21)["label_descriptions"].items():
            common, _, group = description.rpartition(", ")
            if group == "Plant":
                wanted.add(name)
                if common and common.lower() != name.lower():
                    table.setdefault(name, _capitalized(common))
    todo = sorted(wanted - table.keys())
    print(f"{len(wanted)} species, {len(todo)} to look up: about {len(todo) / 60:.0f} min", flush=True)
    for i, name in enumerate(todo):
        if i % 50 == 0:
            _save(path, table)
            print(f"{i}/{len(todo)} looked up", flush=True)
        started = time.monotonic()
        try:
            table[name] = lookup(name)
        except OSError as exc:  # not saved, so the next run tries again
            print(f"skipped {name}: {exc}", flush=True)
        time.sleep(max(0.0, started + 1 - time.monotonic()))
    _save(path, table)
    named = sum(1 for n in wanted if table.get(n))
    print(f"Done: {named} of {len(wanted)} species have a common name. Saved to {path}")


if __name__ == "__main__":
    main()
