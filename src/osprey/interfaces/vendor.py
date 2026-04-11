"""Vendor asset management — download and verify third-party JS/CSS/fonts.

All vendor assets (JS libraries, CSS, fonts) are declared in
vendor_manifest.json and fetched at build time via ``osprey vendor fetch``.
This eliminates both git-tracked binaries and runtime CDN dependencies.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "vendor_manifest.json"


def _load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_one(url: str, dest: Path, expected_sha256: str) -> None:
    """Download a single file and verify its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "osprey-vendor/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
    actual = _sha256(dest)
    if actual != expected_sha256:
        dest.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {dest.name}: "
            f"expected {expected_sha256[:12]}…, got {actual[:12]}…"
        )


def fetch_all(base_dir: Path | None = None, quiet: bool = False) -> list[str]:
    """Download all vendor assets declared in the manifest.

    Idempotent: skips files that already exist with a matching checksum.
    Returns list of files that were actually downloaded (not skipped).
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    manifest = _load_manifest()
    downloaded: list[str] = []

    for asset in manifest.get("assets", []):
        for target in asset["targets"]:
            dest = base_dir / target / asset["filename"]
            if dest.exists() and _sha256(dest) == asset["sha256"]:
                continue
            if not quiet:
                print(f"  {asset['filename']} → {target}")
            _fetch_one(asset["url"], dest, asset["sha256"])
            downloaded.append(str(dest))

    for bundle in manifest.get("font_bundles", []):
        base_url = bundle["base_url"]
        target_dir = base_dir / bundle["target"]
        for filename, sha256 in bundle["files"].items():
            dest = target_dir / filename
            if dest.exists() and _sha256(dest) == sha256:
                continue
            if not quiet:
                print(f"  {filename} → {bundle['target']}")
            _fetch_one(base_url + filename, dest, sha256)
            downloaded.append(str(dest))

    return downloaded


def verify_all(base_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Verify all vendor assets exist and have correct checksums.

    Returns (ok_files, problems) where problems is a list of human-readable
    error strings.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent

    manifest = _load_manifest()
    ok: list[str] = []
    problems: list[str] = []

    for asset in manifest.get("assets", []):
        for target in asset["targets"]:
            dest = base_dir / target / asset["filename"]
            if not dest.exists():
                problems.append(f"MISSING: {target}{asset['filename']}")
            elif _sha256(dest) != asset["sha256"]:
                problems.append(f"CHECKSUM MISMATCH: {target}{asset['filename']}")
            else:
                ok.append(str(dest))

    for bundle in manifest.get("font_bundles", []):
        target = bundle["target"]
        target_dir = base_dir / target
        for filename, sha256 in bundle["files"].items():
            dest = target_dir / filename
            if not dest.exists():
                problems.append(f"MISSING: {target}{filename}")
            elif _sha256(dest) != sha256:
                problems.append(f"CHECKSUM MISMATCH: {target}{filename}")
            else:
                ok.append(str(dest))

    return ok, problems
