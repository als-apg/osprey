"""Publish the demo videos to their GitHub release, and fetch them for the docs build.

Each OSPREY version has its own release, ``docs-media-vYYYY.M.P``, so every
docs version plays the video recorded for it. :func:`upload` creates that
release with both MP4s in one command (a release never exists without its
assets), or, for a retake of the same version, replaces that release's MP4s;
it never touches another version's release. It then names the release in the
manifest, which is committed with the posters.

:func:`fetch` runs in the docs build. It downloads exactly the release the
committed manifest names and checks each MP4 against its SHA-256. A release or
asset that does not exist costs the page its video, never the build: the page
shows the poster. A hash mismatch or any other ``gh`` failure fails a strict
run (a deploy) and is a notice otherwise (a pull-request preview).

Every ``gh`` call goes through ``run`` (``subprocess.run`` by default).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docs.screenshots import video_manifest

Run = Callable[..., subprocess.CompletedProcess]

THEMES = ("dark", "light")
_NOT_FOUND = "release not found"
_NO_ASSET = "no assets match"


def _gh(run: Run, *args: str) -> subprocess.CompletedProcess:
    return run(["gh", *args], capture_output=True, text=True, check=False)


def _videos(manifest: dict[str, Any]) -> list[dict[str, str]]:
    return [manifest["themes"][t]["mp4"] for t in THEMES if t in manifest["themes"]]


def upload(version: str, out_dir: Path, static_dir: Path, run: Run = subprocess.run) -> int:
    """Upload both MP4s in *out_dir* to the release for *version*; name it in the manifest."""
    try:
        release = video_manifest.release_name(version)
    except ValueError as exc:
        print(
            f"error: {exc}; pass VERSION=YYYY.M.P or a pre-release such as 2026.9.0b4",
            file=sys.stderr,
        )
        return 2
    manifest_path = Path(static_dir) / video_manifest.MANIFEST_PATH.name
    videos = [str(Path(out_dir) / v["name"]) for v in _videos(video_manifest.load(manifest_path))]

    view = _gh(run, "release", "view", release)
    if view.returncode == 0:
        done = _gh(run, "release", "upload", release, *videos, "--clobber")
    elif _NOT_FOUND in (view.stderr or "").lower():
        done = _gh(
            run,
            "release",
            "create",
            release,
            *videos,
            "--latest=false",
            "--title",
            f"Docs media v{version}",
            "--notes",
            f"The landing-page demo videos for the OSPREY {version} documentation.",
        )
    else:
        done = view
    if done.returncode != 0:
        print(f"error: gh failed for {release}: {(done.stderr or '').strip()}", file=sys.stderr)
        return 1
    video_manifest.set_release(manifest_path, release)
    print(f"uploaded to {release}; commit {manifest_path} with the posters")
    return 0


def _drop_videos(static_dir: Path, videos: list[dict[str, str]]) -> None:
    for video in videos:
        (Path(static_dir) / video["name"]).unlink(missing_ok=True)


def fetch(static_dir: Path, strict: bool, run: Run = subprocess.run) -> int:
    """Download the manifest's release into *static_dir* and verify each MP4."""
    manifest = video_manifest.load(Path(static_dir) / video_manifest.MANIFEST_PATH.name)
    release = manifest.get("release")
    videos = _videos(manifest)
    if not release or not videos:
        print("::notice::no demo-video release named in the manifest; the page shows the poster.")
        return 0

    def give_up(message: str, fail: bool) -> int:
        _drop_videos(static_dir, videos)
        if fail and strict:
            print(f"::error::{message}")
            return 1
        print(f"::notice::{message}; the page shows the poster.")
        return 0

    view = _gh(run, "release", "view", release)
    if view.returncode != 0:
        detail = (view.stderr or "").strip()
        return give_up(f"gh release view {release}: {detail}", _NOT_FOUND not in detail.lower())
    for video in videos:
        got = _gh(
            run,
            "release",
            "download",
            release,
            "-p",
            video["name"],
            "-D",
            str(static_dir),
            "--clobber",
        )
        if got.returncode != 0:
            detail = (got.stderr or "").strip()
            return give_up(f"{release} {video['name']}: {detail}", _NO_ASSET not in detail.lower())
        if video_manifest.sha256(Path(static_dir) / video["name"]) != video["sha256"]:
            return give_up(
                f"{release} {video['name']} does not match its sha256 in the manifest", True
            )
    print(f"fetched {len(videos)} demo video(s) from {release}")
    return 0
