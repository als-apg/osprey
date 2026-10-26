"""One reason vocabulary and one rules digest across the JS bundle and the Python rules.

``LAYOUT_REASONS`` equals the values of the bundle's ``LAYOUT_REASONS`` object, in order,
and the rules digest recomputed here by the C-MANIFEST byte recipe equals both
:func:`rules_digest` and ``dist/manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from dock_engine import LAYOUT_REASONS, cases_dir, rules_digest

PKG = Path(__file__).resolve().parents[2] / "packages" / "dock-engine"
BUNDLE = PKG / "dist" / "dock-engine.mjs"
MANIFEST = PKG / "dist" / "manifest.json"
LAYOUT_TS = PKG / "ts" / "layout.ts"


def _bundle_reasons() -> tuple[str, ...]:
    source = BUNDLE.read_text(encoding="utf-8")
    block = re.search(r"var LAYOUT_REASONS = \{(.*?)\};", source, re.S)
    assert block is not None, "LAYOUT_REASONS object not found in dist/dock-engine.mjs"
    entries = re.findall(r'^\s*\w+:\s*"([^"]+)",?\s*$', block.group(1), re.M)
    lines = [line for line in block.group(1).splitlines() if line.strip()]
    assert len(entries) == len(lines), "an entry of the JS LAYOUT_REASONS was not parsed"
    return tuple(entries)


def _recomputed_digest() -> str:
    root = cases_dir()
    digest = hashlib.sha256()
    reasons = json.dumps(list(LAYOUT_REASONS), separators=(",", ":"))
    digest.update(f"{reasons}\n".encode())
    paths = sorted(
        f"{kind}/{path.name}"
        for kind in ("normalize", "validate")
        for path in (root / kind).glob("*.json")
    )
    assert paths, "no normalize/validate cases found"
    for rel in paths:
        digest.update(f"{rel}\n".encode())
        digest.update((root / rel).read_bytes())
    digest.update(LAYOUT_TS.read_bytes())
    return digest.hexdigest()


def test_python_reasons_equal_bundle_reasons() -> None:
    """The Python tuple is Object.values of the bundle's LAYOUT_REASONS, in order."""
    assert _bundle_reasons() == LAYOUT_REASONS


def test_reasons_are_distinct() -> None:
    """No reason string appears twice in the vocabulary."""
    assert len(set(LAYOUT_REASONS)) == len(LAYOUT_REASONS)


def test_rules_digest_recomputes() -> None:
    """The C-MANIFEST recipe over reasons, cases and layout.ts gives rules_digest()."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recomputed = _recomputed_digest()
    assert re.fullmatch(r"[0-9a-f]{64}", recomputed)
    assert recomputed == rules_digest()
    assert rules_digest() == manifest["rules_digest"]
