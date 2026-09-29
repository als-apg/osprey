"""The demo-video manifest: what each committed poster and published MP4 is.

``docs/source/_static/demo/manifest.json`` is committed together with the two
posters. Per theme it names the take (OSPREY version, recording time, video
length, the real session time it covers and the fast-forward factor) and the
MP4 and poster by name and SHA-256. ``release`` names the GitHub release
(``docs-media-vYYYY.M.P``) that carries the MP4s, one release per OSPREY
version, so each docs version plays its own video; it is ``null`` until the
current takes are uploaded, and a new take resets it.

The docs build downloads exactly the release named here and checks each MP4
against its hash; ``upload-check`` checks the local files against it before an
upload.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DEMO_DIR = REPO_ROOT / "docs" / "source" / "_static" / "demo"
MANIFEST_PATH = STATIC_DEMO_DIR / "manifest.json"

RELEASE_PREFIX = "docs-media-v"
# A final or pre-release OSPREY version: YYYY.M.P with an optional aN/bN/rcN.
_VERSION = re.compile(r"\d{4}\.\d{1,2}\.\d{1,3}(?:(?:a|b|rc)\d+)?")

ENTRY_FIELDS = (
    "osprey_version",
    "recorded_at",
    "duration_s",
    "real_session_s",
    "speed",
    "mp4",
    "poster",
)
_FILE_FIELDS = ("name", "sha256")


def sha256(path: Path) -> str:
    """The SHA-256 of the file at *path*, as hex."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def release_name(version: str) -> str:
    """``docs-media-v<version>`` for an OSPREY version ``YYYY.M.P[aN|bN|rcN]``."""
    if not _VERSION.fullmatch(version or ""):
        raise ValueError(f"not a release version (YYYY.M.P[aN|bN|rcN]): {version!r}")
    return f"{RELEASE_PREFIX}{version}"


def entry(
    *,
    osprey_version: str,
    recorded_at: float,
    duration_s: float,
    real_session_s: float,
    speed: float,
    mp4: Path,
    poster: Path,
) -> dict[str, Any]:
    """The manifest entry for one theme's take."""
    return {
        "osprey_version": osprey_version,
        "recorded_at": datetime.fromtimestamp(recorded_at, UTC).isoformat(),
        "duration_s": round(duration_s, 2),
        "real_session_s": round(real_session_s, 1),
        "speed": round(speed, 2),
        "mp4": {"name": Path(mp4).name, "sha256": sha256(mp4)},
        "poster": {"name": Path(poster).name, "sha256": sha256(poster)},
    }


def missing_fields(theme_entry: dict[str, Any]) -> list[str]:
    """The fields an entry lacks, as dotted names; empty when complete."""
    missing = [f for f in ENTRY_FIELDS if theme_entry.get(f) in (None, "")]
    for key in ("mp4", "poster"):
        value = theme_entry.get(key)
        if isinstance(value, dict):
            missing += [f"{key}.{f}" for f in _FILE_FIELDS if not value.get(f)]
    return missing


def load(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """The manifest at *path*; an absent file reads as no takes, not uploaded."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"release": None, "themes": {}}
    data.setdefault("release", None)
    data.setdefault("themes", {})
    return data


def save(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def put_theme(path: Path, theme: str, theme_entry: dict[str, Any]) -> None:
    """Record *theme*'s new take; it is not uploaded yet, so ``release`` is cleared."""
    manifest = load(path)
    manifest["themes"][theme] = theme_entry
    manifest["release"] = None
    save(manifest, path)


def set_release(path: Path, release: str) -> None:
    """Name the release that now carries the takes in the manifest."""
    manifest = load(path)
    manifest["release"] = release
    save(manifest, path)
