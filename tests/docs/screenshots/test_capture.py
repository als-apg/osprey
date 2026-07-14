"""Unit tests for the doc-screenshot capture runner.

CI-safe: the pure helpers (versioning, manifest I/O, filename shaping) never
touch a browser, and the one real end-to-end capture skips cleanly when the
chromium binary or Playwright is unavailable.
"""

from __future__ import annotations

import json

import pytest
from docs.screenshots import capture, recipes
from docs.screenshots.capture import (
    ScreenshotSkip,
    _output_filename,
    osprey_version,
    stamp_manifest,
)
from docs.screenshots.recipes import DocShot, SubView
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# App factory resolvable by dotted path (importlib in _capture_standalone)
# ---------------------------------------------------------------------------

_TRIVIAL_HTML = (
    "<!doctype html><html><head><title>t</title></head><body>"
    '<div id="target" style="width:100px;height:40px;background:#333"></div>'
    "</body></html>"
)


def make_trivial_app() -> FastAPI:
    """Zero-arg FastAPI factory whose ``/`` serves a fixed-size target element."""
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return _TRIVIAL_HTML

    return app


# ---------------------------------------------------------------------------
# Version + filename helpers
# ---------------------------------------------------------------------------


def test_osprey_version_non_empty() -> None:
    version = osprey_version()
    assert isinstance(version, str)
    assert version


def test_output_filename_theme_suffix_rule() -> None:
    # Single-theme recipes carry no theme suffix (parity with e.g. ariel_search.png).
    assert _output_filename("x", "light", 1) == "x.png"
    # Multi-theme recipes disambiguate with a theme suffix.
    assert _output_filename("x", "light", 2) == "x_light.png"
    assert _output_filename("x", "dark", 2) == "x_dark.png"


def test_subview_single_theme_filename_parity() -> None:
    shot = DocShot(
        name="ariel",
        environment="standalone_interface",
        kind="static",
        app_factory="tests.docs.screenshots.test_capture:make_trivial_app",
        themes=("light",),
        subviews=(
            SubView(anchor="#search", out="ariel_search"),
            SubView(anchor="#browse", out="ariel_browse"),
        ),
    )
    names = [_output_filename(out, "light", len(shot.themes)) for out in shot.output_names()]
    assert names == ["ariel_search.png", "ariel_browse.png"]


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip_and_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(capture, "output_dir", lambda: tmp_path)

    stamp_manifest("alpha", "static")
    stamp_manifest("beta", "agentic")

    manifest = json.loads((tmp_path / recipes.MANIFEST_NAME).read_text())
    assert set(manifest) == {"alpha", "beta"}
    for name, expected_kind in (("alpha", "static"), ("beta", "agentic")):
        entry = manifest[name]
        assert set(entry) == {"osprey_version", "captured_utc", "kind"}
        assert entry["kind"] == expected_kind
        assert entry["osprey_version"]
        assert entry["captured_utc"]

    # A second stamp of the same name overwrites in place (idempotent shape).
    stamp_manifest("alpha", "agentic")
    manifest = json.loads((tmp_path / recipes.MANIFEST_NAME).read_text())
    assert set(manifest) == {"alpha", "beta"}
    assert manifest["alpha"]["kind"] == "agentic"


def test_manifest_recovers_from_malformed_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(capture, "output_dir", lambda: tmp_path)
    (tmp_path / recipes.MANIFEST_NAME).write_text("not json at all")

    stamp_manifest("gamma", "static")

    manifest = json.loads((tmp_path / recipes.MANIFEST_NAME).read_text())
    assert set(manifest) == {"gamma"}


# ---------------------------------------------------------------------------
# tutorial_stack seam
# ---------------------------------------------------------------------------


def test_tutorial_stack_provider_skips(monkeypatch) -> None:
    # When the container stack can't be brought up, capture_tutorial_stack must
    # degrade to a ScreenshotSkip rather than crash. Hermetic: mock the stack
    # builder to skip so this never touches a real container runtime (keeping
    # this file CI-safe regardless of whether podman happens to be running).
    from contextlib import contextmanager

    @contextmanager
    def _skip_stack(*, artifact_port):
        raise ScreenshotSkip("container runtime unavailable")
        yield  # pragma: no cover - unreachable; marks this a generator

    monkeypatch.setattr(capture, "_tutorial_stack", _skip_stack)

    shot = DocShot(
        name="hero",
        environment="tutorial_stack",
        kind="static",
    )
    with pytest.raises(ScreenshotSkip):
        capture.capture_tutorial_stack(lambda: None, shot, agentic=False)


# ---------------------------------------------------------------------------
# Real standalone element-crop capture (skips cleanly without chromium)
# ---------------------------------------------------------------------------


def test_standalone_element_capture_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(capture, "output_dir", lambda: tmp_path)

    shot = DocShot(
        name="trivial",
        environment="standalone_interface",
        kind="static",
        app_factory="tests.docs.screenshots.test_capture:make_trivial_app",
        capture_mode="element",
        element_selector="#target",
        themes=("light",),
    )

    try:
        with capture.chromium_context() as browser:
            paths = capture._capture_standalone(browser, shot)
    except ScreenshotSkip as exc:
        pytest.skip(f"chromium/playwright unavailable: {exc}")

    assert len(paths) == 1
    png_path = paths[0]
    assert png_path == tmp_path / "trivial.png"
    assert png_path.exists()
    data = png_path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 8
