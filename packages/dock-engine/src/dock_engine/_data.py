"""Package data: the case corpus and the rules digest from the manifest.

A wheel carries both as package data (`dock_engine/cases/`,
`dock_engine/manifest.json`). An editable install, or `src/` placed on the path
unbuilt (the live-CA venue's `.pth`), has no package data; there the files are
read from the checkout, `packages/dock-engine/cases/` and
`packages/dock-engine/dist/manifest.json`.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

# packages/dock-engine/src/dock_engine/_data.py -> packages/dock-engine
_CHECKOUT = Path(__file__).resolve().parents[2]


def _resource(name: str) -> Path | None:
    # importlib.resources rather than Path(__file__) alone, so an installed
    # wheel answers from its own package data.
    resource = files("dock_engine") / name
    if resource.is_dir() or resource.is_file():
        return Path(str(resource))
    return None


def cases_dir() -> Path:
    """The directory of case JSON, one subdirectory per case kind."""
    return _resource("cases") or _CHECKOUT / "cases"


def rules_digest() -> str:
    """The 64-hex `rules_digest` recorded in the engine's manifest."""
    manifest = _resource("manifest.json") or _CHECKOUT / "dist" / "manifest.json"
    digest = json.loads(manifest.read_text(encoding="utf-8"))["rules_digest"]
    if not isinstance(digest, str):
        raise TypeError(f"{manifest}: rules_digest is not a string")
    return digest
