"""Every byte copy from CANOPY's screen runtime is the file the pin names.

``cases-pin.json`` records, for the vendored dock engine and each copied case, the
runtime commit and path it came from and its git blob. A copy edited, missing or extra
fails here; the fix is a re-copy and re-pin, never an edit. The vendored bundle's sha256
in ``vendor_manifest.json`` must name the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from osprey.interfaces.web_terminal.dock_layout import LAYOUT_REASONS

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((Path(__file__).resolve().parent / "cases-pin.json").read_text(encoding="utf-8"))
FILES = PIN["files"]
MANIFEST = ROOT / "src" / "osprey" / "interfaces" / "vendor_manifest.json"
BUNDLE = ROOT / "src" / "osprey" / "interfaces" / "web_terminal" / "static" / "vendor"


def _blob(path: Path) -> str:
    """The git blob id of *path*'s bytes, as ``git hash-object`` computes it."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.parametrize("entry", FILES, ids=[Path(e["copy"]).name for e in FILES])
def test_each_copy_hashes_to_its_pinned_blob(entry: dict[str, str]) -> None:
    assert _blob(ROOT / entry["copy"]) == entry["blob"], entry["copy"]


def test_the_copied_cases_are_exactly_the_pinned_set() -> None:
    on_disk = {
        str(path.relative_to(ROOT))
        for path in (Path(__file__).resolve().parent / "cases").rglob("*.json")
    }
    pinned = {e["copy"] for e in FILES if e["copy"].startswith("tests/")}
    assert on_disk == pinned


def test_every_copy_comes_from_one_runtime_commit() -> None:
    assert len({e["commit"] for e in FILES}) == 1
    assert re.fullmatch(r"[0-9a-f]{40}", FILES[0]["commit"])


@pytest.mark.parametrize("filename", ["dock-engine.mjs", "dock-engine.d.mts"])
def test_the_vendor_manifest_names_the_same_bytes(filename: str) -> None:
    (asset,) = [a for a in json.loads(MANIFEST.read_text())["assets"] if a["filename"] == filename]
    assert hashlib.sha256((BUNDLE / filename).read_bytes()).hexdigest() == asset["sha256"]
    assert asset["source"]["commit"] == FILES[0]["commit"]
    assert "url" not in asset, "the bundle is committed, never fetched"


def test_the_bundle_refuses_with_the_servers_reasons() -> None:
    """The browser's engine and the server's rules speak one reason vocabulary, in order."""
    source = (BUNDLE / "dock-engine.mjs").read_text(encoding="utf-8")
    block = re.search(r"var LAYOUT_REASONS = \{(.*?)\};", source, re.S)
    assert block is not None, "LAYOUT_REASONS object not found in the vendored engine"
    assert tuple(re.findall(r'^\s*\w+:\s*"([^"]+)",?\s*$', block.group(1), re.M)) == LAYOUT_REASONS
