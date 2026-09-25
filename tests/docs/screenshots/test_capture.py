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
from playwright.sync_api import Error as PlaywrightError

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
    # The seam this stands in for takes the artifact port by keyword.
    def _skip_stack(*, artifact_port):  # noqa: ARG001
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
# Playwright driver start (a driver that cannot start is a named skip)
# ---------------------------------------------------------------------------


class _DriverThatWillNotStart:
    """Stands in for ``sync_playwright()`` whose ``start()`` raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.exited = False

    def start(self):
        raise self.error

    def __exit__(self, *exc_info) -> None:
        self.exited = True


@pytest.mark.parametrize(
    "node_options",
    [None, "--require /gone/preload.js"],
    ids=["no-node-options", "node-options"],
)
def test_driver_that_exits_before_its_handshake_is_a_skip(monkeypatch, node_options) -> None:
    error = AttributeError(
        "'PlaywrightContextManager' object has no attribute '_playwright'", name="_playwright"
    )
    stub = _DriverThatWillNotStart(error)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: stub)
    if node_options is None:
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
    else:
        monkeypatch.setenv("NODE_OPTIONS", node_options)

    with pytest.raises(ScreenshotSkip, match="exited before its handshake") as info:
        with capture.chromium_context():
            pytest.fail("no browser should be yielded")

    assert info.value.__cause__ is error
    assert stub.exited is True
    if node_options is None:
        assert "NODE_OPTIONS" not in str(info.value)
    else:
        assert "NODE_OPTIONS='--require /gone/preload.js'" in str(info.value)


def test_driver_that_cannot_be_spawned_is_a_skip(monkeypatch) -> None:
    error = FileNotFoundError(2, "No such file or directory", "/gone/node")
    stub = _DriverThatWillNotStart(error)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: stub)
    monkeypatch.setenv("NODE_OPTIONS", "--require /gone/preload.js")

    with pytest.raises(ScreenshotSkip, match="playwright driver did not start") as info:
        with capture.chromium_context():
            pytest.fail("no browser should be yielded")

    assert "/gone/node" in str(info.value)
    assert "NODE_OPTIONS" not in str(info.value)
    assert info.value.__cause__ is error
    assert stub.exited is True


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("boom", name="_connection"),
        PlaywrightError("It looks like you are using Playwright Sync API inside the asyncio loop."),
        RuntimeError("boom"),
    ],
    ids=["other-attribute", "sync-api-in-loop", "runtime-error"],
)
def test_other_start_failures_are_not_a_skip(monkeypatch, error) -> None:
    stub = _DriverThatWillNotStart(error)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: stub)

    with pytest.raises(type(error)) as info:
        with capture.chromium_context():
            pytest.fail("no browser should be yielded")

    assert info.value is error


def test_a_real_driver_killed_by_its_preload_is_a_skip(monkeypatch, tmp_path) -> None:
    # The preload file is never created, so node exits before the driver handshake.
    monkeypatch.setenv("NODE_OPTIONS", f"--require {tmp_path / 'deleted-preload.js'}")

    with pytest.raises(ScreenshotSkip, match="exited before its handshake"):
        with capture.chromium_context():
            pytest.fail("no browser should be yielded")


# ---------------------------------------------------------------------------
# hermetic_hub recipes go through the contact sheet's capture routine
# ---------------------------------------------------------------------------


def test_hermetic_hub_recipe_captures_each_theme_in_its_mode(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager

    from docs.screenshots import contact_sheet

    hub = object()
    calls: list[tuple] = []

    @contextmanager
    def _stub_hub():
        yield hub

    def _record(_browser, got_hub, theme, mode, dest, *, stage=None, **_kw):
        assert got_hub is hub
        calls.append((theme, mode, stage, dest))

    monkeypatch.setattr(contact_sheet, "hermetic_hub", _stub_hub)
    monkeypatch.setattr(contact_sheet, "capture_hub_view", _record)
    monkeypatch.setattr(capture, "output_dir", lambda: tmp_path)

    (recipe,) = [s for s in recipes.REGISTRY if s.name == "customize_sheet"]
    paths = capture._capture_hermetic_hub(object(), recipe)

    by_theme = {theme: (mode, stage, dest) for theme, mode, stage, dest in calls}
    assert len(calls) == len(recipe.themes) == 2
    assert by_theme["light"] == (
        "expert",
        "customize_sheet",
        tmp_path / "customize_sheet_light.png",
    )
    assert by_theme["dark"] == ("expert", "customize_sheet", tmp_path / "customize_sheet_dark.png")
    assert sorted(paths) == sorted(dest for *_, dest in calls)


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
