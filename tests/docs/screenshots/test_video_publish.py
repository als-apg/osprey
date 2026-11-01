"""Unit tests for publishing the demo videos to a per-version GitHub release.

Every ``gh`` call goes through an injected runner, so nothing touches GitHub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from docs.screenshots import video_manifest, video_publish

THEMES = ("dark", "light")


def _published(
    tmp_path: Path,
    release: str | None = None,
    name: str = "landing",
    stem: str = "osprey-demo",
) -> tuple[Path, Path, Path]:
    """One video's takes in both themes: MP4s in ``out``, posters + manifest in ``static``."""
    out, static = tmp_path / "out", tmp_path / "static"
    out.mkdir(exist_ok=True)
    static.mkdir(exist_ok=True)
    manifest = static / "manifest.json"
    for theme in THEMES:
        mp4 = out / f"{stem}-{theme}.mp4"
        mp4.write_bytes(b"video-" + stem.encode() + theme.encode())
        poster = static / f"{stem}-{theme}-poster.jpg"
        poster.write_bytes(b"poster")
        video_manifest.put_theme(
            manifest,
            name,
            theme,
            video_manifest.entry(
                osprey_version="2026.9.0",
                recorded_at=1_790_000_000.0,
                duration_s=58.0,
                real_session_s=300.0,
                speed=16.0,
                mp4=mp4,
                poster=poster,
            ),
        )
    if release:
        video_manifest.set_release(manifest, name, release)
    return out, static, manifest


def _release(manifest: Path, name: str = "landing") -> str | None:
    return video_manifest.video(video_manifest.load(manifest), name)["release"]


class Gh:
    """Scripted ``gh``: answers by subcommand, records every call."""

    def __init__(self, answers: dict[str, tuple[int, str]] | None = None, assets=None) -> None:
        self.answers = answers or {}
        self.assets = assets or {}
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, check=False):
        assert cmd[0] == "gh" and capture_output and text and not check
        self.calls.append(cmd)
        key = " ".join(cmd[1:3])
        code, err = self.answers.get(key, (0, ""))
        if key == "release download" and code == 0:
            dest = Path(cmd[cmd.index("-D") + 1])
            name = cmd[cmd.index("-p") + 1]
            if name not in self.assets:
                return subprocess.CompletedProcess(cmd, 1, "", "no assets match the file pattern")
            (dest / name).write_bytes(self.assets[name])
        return subprocess.CompletedProcess(cmd, code, "", err)


NOT_FOUND = (1, "release not found")


# --- upload -------------------------------------------------------------------


def test_a_new_version_gets_its_release_with_both_videos_in_one_command(tmp_path) -> None:
    out, static, manifest = _published(tmp_path)
    gh = Gh({"release view": NOT_FOUND})

    assert video_publish.upload("2026.9.0", out, static, run=gh) == 0

    view, create = gh.calls
    assert view[:4] == ["gh", "release", "view", "docs-media-v2026.9.0"]
    assert create[:4] == ["gh", "release", "create", "docs-media-v2026.9.0"]
    assert str(out / "osprey-demo-dark.mp4") in create
    assert str(out / "osprey-demo-light.mp4") in create
    assert "--latest=false" in create and "--clobber" not in create
    assert _release(manifest) == "docs-media-v2026.9.0"


def test_a_retake_for_the_same_version_replaces_that_release_s_videos(tmp_path) -> None:
    out, static, manifest = _published(tmp_path)
    gh = Gh()

    assert video_publish.upload("2026.9.0", out, static, run=gh) == 0

    view, upload = gh.calls
    assert upload[:4] == ["gh", "release", "upload", "docs-media-v2026.9.0"]
    assert "--clobber" in upload
    assert _release(manifest) == "docs-media-v2026.9.0"


def test_upload_touches_no_other_release(tmp_path) -> None:
    out, static, _ = _published(tmp_path)
    gh = Gh({"release view": NOT_FOUND})
    video_publish.upload("2026.9.0", out, static, run=gh)
    for call in gh.calls:
        names = [a for a in call if a.startswith("docs-media")]
        assert names == ["docs-media-v2026.9.0"]


@pytest.mark.parametrize("version", ["2026.9", "v2026.9.0", "latest"])
def test_upload_refuses_a_malformed_version_before_calling_gh(tmp_path, version, capsys) -> None:
    out, static, _ = _published(tmp_path)
    gh = Gh()
    assert video_publish.upload(version, out, static, run=gh) == 2
    assert gh.calls == []
    assert "YYYY.M.P" in capsys.readouterr().err


def test_a_failed_create_leaves_the_manifest_not_uploaded(tmp_path, capsys) -> None:
    out, static, manifest = _published(tmp_path)
    gh = Gh({"release view": NOT_FOUND, "release create": (1, "HTTP 401")})
    assert video_publish.upload("2026.9.0", out, static, run=gh) == 1
    assert _release(manifest) is None
    assert "HTTP 401" in capsys.readouterr().err


def test_an_unexpected_view_error_stops_the_upload(tmp_path) -> None:
    out, static, manifest = _published(tmp_path)
    gh = Gh({"release view": (1, "HTTP 502")})
    assert video_publish.upload("2026.9.0", out, static, run=gh) == 1
    assert len(gh.calls) == 1
    assert _release(manifest) is None


def test_pending_lists_the_videos_with_takes_and_no_release(tmp_path) -> None:
    _published(tmp_path, release="docs-media-v2026.9.0")
    _, _, manifest = _published(tmp_path, name="second", stem="osprey-second")
    assert video_publish.pending(video_manifest.load(manifest)) == ["second"]


def test_upload_by_default_publishes_only_the_videos_not_uploaded_yet(tmp_path) -> None:
    _published(tmp_path, release="docs-media-v2026.9.0b4")
    out, static, manifest = _published(tmp_path, name="second", stem="osprey-second")
    gh = Gh({"release view": NOT_FOUND})

    assert video_publish.upload("2026.9.0", out, static, run=gh) == 0

    _, create = gh.calls
    assert str(out / "osprey-second-dark.mp4") in create
    assert str(out / "osprey-second-light.mp4") in create
    assert not [a for a in create if "osprey-demo" in a]
    assert _release(manifest, "second") == "docs-media-v2026.9.0"
    assert _release(manifest, "landing") == "docs-media-v2026.9.0b4"


def test_upload_of_a_named_video_retakes_only_that_video(tmp_path) -> None:
    _published(tmp_path, release="docs-media-v2026.9.0")
    out, static, manifest = _published(tmp_path, name="second", stem="osprey-second")
    gh = Gh()

    assert video_publish.upload("2026.9.0", out, static, names=["landing"], run=gh) == 0

    _, upload = gh.calls
    assert str(out / "osprey-demo-dark.mp4") in upload
    assert not [a for a in upload if "osprey-second" in a]
    assert _release(manifest, "landing") == "docs-media-v2026.9.0"
    assert _release(manifest, "second") is None


def test_upload_with_nothing_pending_calls_no_gh(tmp_path, capsys) -> None:
    out, static, _ = _published(tmp_path, release="docs-media-v2026.9.0")
    gh = Gh()
    assert video_publish.upload("2026.9.0", out, static, run=gh) == 0
    assert gh.calls == []
    assert "nothing to upload" in capsys.readouterr().out


# --- fetch (docs build) -------------------------------------------------------


def _assets(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in out.glob("*.mp4")}


def test_fetch_downloads_the_named_release_and_checks_each_hash(tmp_path) -> None:
    out, static, _ = _published(tmp_path, release="docs-media-v2026.9.0")
    gh = Gh(assets=_assets(out))

    assert video_publish.fetch(static, strict=True, run=gh) == 0

    downloads = [c for c in gh.calls if c[1:3] == ["release", "download"]]
    assert {c[3] for c in downloads} == {"docs-media-v2026.9.0"}
    for theme in THEMES:
        assert (static / f"osprey-demo-{theme}.mp4").read_bytes() == b"video-osprey-demo" + (
            theme.encode()
        )


def test_fetch_downloads_each_video_from_its_own_release(tmp_path) -> None:
    _published(tmp_path, release="docs-media-v2026.9.0b4")
    out, static, _ = _published(
        tmp_path, release="docs-media-v2026.10.0", name="second", stem="osprey-second"
    )
    gh = Gh(assets=_assets(out))

    assert video_publish.fetch(static, strict=True, run=gh) == 0

    downloads = {c[c.index("-p") + 1]: c[3] for c in gh.calls if c[1:3] == ["release", "download"]}
    assert downloads == {
        "osprey-demo-dark.mp4": "docs-media-v2026.9.0b4",
        "osprey-demo-light.mp4": "docs-media-v2026.9.0b4",
        "osprey-second-dark.mp4": "docs-media-v2026.10.0",
        "osprey-second-light.mp4": "docs-media-v2026.10.0",
    }
    assert len(list(static.glob("*.mp4"))) == 4


def test_fetch_of_a_video_not_uploaded_yet_keeps_the_other_videos(tmp_path, capsys) -> None:
    _published(tmp_path, release="docs-media-v2026.9.0")
    out, static, _ = _published(tmp_path, name="second", stem="osprey-second")
    gh = Gh(assets=_assets(out))

    assert video_publish.fetch(static, strict=True, run=gh) == 0

    assert sorted(p.name for p in static.glob("*.mp4")) == [
        "osprey-demo-dark.mp4",
        "osprey-demo-light.mp4",
    ]
    assert "no release named for the second video" in capsys.readouterr().out


def test_fetch_with_no_videos_in_the_manifest_is_a_notice(tmp_path, capsys) -> None:
    gh = Gh()
    assert video_publish.fetch(tmp_path, strict=True, run=gh) == 0
    assert gh.calls == []
    assert "::notice::" in capsys.readouterr().out


def test_fetch_without_a_named_release_keeps_the_posters_only(tmp_path, capsys) -> None:
    _, static, _ = _published(tmp_path)
    gh = Gh()
    assert video_publish.fetch(static, strict=True, run=gh) == 0
    assert gh.calls == []
    assert "::notice::" in capsys.readouterr().out


@pytest.mark.parametrize("strict", [True, False])
def test_a_missing_release_is_a_notice_even_when_strict(tmp_path, capsys, strict) -> None:
    _, static, _ = _published(tmp_path, release="docs-media-v2026.9.0")
    gh = Gh({"release view": NOT_FOUND})
    assert video_publish.fetch(static, strict=strict, run=gh) == 0
    assert "::notice::" in capsys.readouterr().out
    assert not list(static.glob("*.mp4"))


def test_a_missing_asset_is_a_notice_and_leaves_no_video(tmp_path, capsys) -> None:
    out, static, _ = _published(tmp_path, release="docs-media-v2026.9.0")
    assets = _assets(out)
    del assets["osprey-demo-light.mp4"]
    gh = Gh(assets=assets)
    assert video_publish.fetch(static, strict=True, run=gh) == 0
    assert "::notice::" in capsys.readouterr().out
    assert not list(static.glob("*.mp4"))


@pytest.mark.parametrize(("strict", "code"), [(True, 1), (False, 0)])
def test_a_hash_mismatch_drops_the_videos_and_fails_only_when_strict(
    tmp_path, capsys, strict, code
) -> None:
    out, static, _ = _published(tmp_path, release="docs-media-v2026.9.0")
    assets = _assets(out)
    assets["osprey-demo-dark.mp4"] = b"some other take"
    gh = Gh(assets=assets)
    assert video_publish.fetch(static, strict=strict, run=gh) == code
    assert not list(static.glob("*.mp4"))
    assert "sha256" in capsys.readouterr().out


@pytest.mark.parametrize(("strict", "code"), [(True, 1), (False, 0)])
def test_an_unexpected_gh_error_fails_only_when_strict(tmp_path, capsys, strict, code) -> None:
    _, static, _ = _published(tmp_path, release="docs-media-v2026.9.0")
    gh = Gh({"release view": (1, "HTTP 401: Bad credentials")})
    assert video_publish.fetch(static, strict=strict, run=gh) == code
    out = capsys.readouterr().out
    assert "HTTP 401" in out
    assert ("::error::" in out) is strict
