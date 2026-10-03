"""Record raw llama-server responses for the replay tests.

Usage: ``python record.py BASE_URL`` against a running llama-server started
with the command in README.md. Sends exactly the request bodies the
``llama-cpp`` adapter sends — one content part per POST, built by the adapter's
own ``_content_part`` — for the two probe pictures in this directory and the
four probe queries, and writes each response body verbatim to
``embeddings/<key>.json`` with ``<key> = part_key(part)``. ``GET /v1/models`` is
written verbatim to ``models.json``, and ``manifest.json`` maps each picture and
query to its key.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PICTURES = ("orbit_kick", "tunnel_temp")
QUERIES = (
    "orbit kick near BPM 7",
    "horizontal orbit distortion after fill",
    "tunnel air temperature drift",
    "RF cavity trip strip chart",
)


def part_key(part: dict) -> str:
    """The sha256 that names a content part's recorded response."""
    canonical = json.dumps(part, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main(base: str) -> None:
    import requests

    from osprey.models.providers.llama_cpp import LLAMA_CPP_DEFAULT_MODEL, _content_part

    base = base.rstrip("/")
    (HERE / "embeddings").mkdir(exist_ok=True)
    models = requests.get(f"{base}/v1/models", timeout=30)
    models.raise_for_status()
    (HERE / "models.json").write_bytes(models.content.rstrip(b"\n") + b"\n")

    manifest: dict = {"model": LLAMA_CPP_DEFAULT_MODEL, "pictures": {}, "queries": {}}
    items = [
        ("pictures", name, ((HERE / f"{name}.png").read_bytes(), "image/png")) for name in PICTURES
    ]
    items += [("queries", query, query) for query in QUERIES]
    for group, label, item in items:
        part = _content_part(item)
        key = part_key(part)
        body = {"model": LLAMA_CPP_DEFAULT_MODEL, "input": [{"content": [part]}]}
        response = requests.post(f"{base}/v1/embeddings", json=body, timeout=600)
        response.raise_for_status()
        (HERE / "embeddings" / f"{key}.json").write_bytes(response.content.rstrip(b"\n") + b"\n")
        manifest[group][label] = key
        print(f"{group:8s} {label!r:45s} -> {key}")
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
