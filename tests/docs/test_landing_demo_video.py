"""The docs video blocks: markup that works without JavaScript, and a page that
falls back to the poster when the video cannot play.

The markup is read straight out of the landing page's ``docs/source/index.rst``,
and the player is the shared ``_static/js/doc-video.js``; a block for a page one
level down is written out below. The browser tests load a block
with ``custom.css`` and the player in real Chromium; they skip when the
Playwright browser is not installed, and the playing-video ones when ffmpeg is
missing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests._playwright import chromium_available

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "docs" / "source" / "index.rst"
CSS = REPO / "docs" / "source" / "_static" / "custom.css"
PLAYER = REPO / "docs" / "source" / "_static" / "js" / "doc-video.js"


def _block(page: Path = INDEX, marker: str = 'id="osprey-demo"') -> str:
    """The ``raw:: html`` block of *page* that holds *marker*, dedented."""
    lines = page.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == ".. raw:: html"]
    for start in starts:
        body = []
        for line in lines[start + 1 :]:
            if line.strip() and not line.startswith("   "):
                break
            body.append(line[3:])
        html = "\n".join(body)
        if marker in html:
            return html
    raise AssertionError(f"no {marker} block in {page.name}")


def _video_tag(html: str) -> str:
    return re.search(r"<video\b[^>]*>", html).group(0)


# --- markup -------------------------------------------------------------------


def test_the_markup_names_the_poster_and_the_video_without_javascript() -> None:
    html = _block()
    assert 'poster="_static/demo/osprey-demo-light-poster.jpg"' in _video_tag(html)
    assert '<source src="_static/demo/osprey-demo-light.mp4" type="video/mp4">' in html


def test_the_video_never_autoplays_from_markup() -> None:
    # Playback starts from the script, which honours prefers-reduced-motion.
    assert "autoplay" not in _video_tag(_block())


def test_the_video_is_labelled_and_described_by_the_note() -> None:
    html = _block()
    tag = _video_tag(html)
    assert 'aria-label="' in tag
    assert 'aria-describedby="osprey-demo-note"' in tag
    assert re.search(r'<p class="osprey-demo-note" id="osprey-demo-note">', html)


def test_the_speed_buttons_are_hidden_until_the_script_runs() -> None:
    group = re.search(r'<div class="osprey-demo-speed"[^>]*>', _block()).group(0)
    assert " hidden" in group


def test_the_default_speed_is_real_time() -> None:
    script = PLAYER.read_text(encoding="utf-8")
    assert "setRate(1);" in script
    assert "setRate(1.5);" not in script


# --- in the browser -------------------------------------------------------------


@pytest.fixture
def site(tmp_path: Path):
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from PIL import Image

    demo = tmp_path / "_static" / "demo"
    demo.mkdir(parents=True)
    for theme, shade in (("light", 230), ("dark", 30)):
        Image.new("RGB", (64, 36), (shade, shade, shade)).save(
            demo / f"osprey-demo-{theme}-poster.jpg"
        )
    shutil.copy(CSS, tmp_path / "custom.css")
    (tmp_path / "_static" / "js").mkdir()
    shutil.copy(PLAYER, tmp_path / "_static" / "js" / "doc-video.js")

    def page(theme: str = "light") -> Path:
        index = tmp_path / "index.html"
        index.write_text(
            f'<!doctype html><html data-theme="{theme}"><head>'
            '<link rel="stylesheet" href="custom.css"></head>'
            f"<body>{_block()}</body></html>",
            encoding="utf-8",
        )
        return index

    def add_videos() -> None:
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not on PATH")
        for theme in ("light", "dark"):
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "testsrc=size=64x36:rate=25:duration=2",
                 "-pix_fmt", "yuv420p", "-c:v", "libx264",
                 str(demo / f"osprey-demo-{theme}.mp4")],
                check=True, timeout=120,
            )  # fmt: skip

    page.add_videos = add_videos
    return page


def _open(site_page: Path, **context_args):
    from playwright.sync_api import sync_playwright

    p = sync_playwright().start()
    browser = p.chromium.launch()
    context = browser.new_context(**context_args)
    tab = context.new_page()
    tab.goto(site_page.as_uri())
    tab.wait_for_timeout(1200)
    return p, browser, tab


def _state(tab) -> dict:
    return tab.evaluate(
        """() => {
          const v = document.getElementById('osprey-demo');
          const img = document.querySelector('.osprey-demo-poster');
          const shown = (el) => !!el && el.offsetParent !== null;
          const pressed = [...document.querySelectorAll('.osprey-demo-speed button')]
            .filter((b) => b.getAttribute('aria-pressed') === 'true').map((b) => b.dataset.rate);
          return {
            video: shown(v), poster: shown(img), posterSrc: img ? img.getAttribute('src') : null,
            speed: shown(document.querySelector('.osprey-demo-speed')),
            note: shown(document.getElementById('osprey-demo-note')),
            rate: v.playbackRate, paused: v.paused, pressed,
            controls: v.controls || v.hasAttribute('controls'),
          };
        }"""
    )


def test_without_javascript_the_poster_shows_and_the_speed_buttons_do_not(site) -> None:
    p, browser, tab = _open(site(), java_script_enabled=False)
    try:
        video = tab.locator("#osprey-demo")
        shown, poster = video.is_visible(), video.get_attribute("poster")
        speed = tab.locator(".osprey-demo-speed").is_visible()
        note = tab.locator("#osprey-demo-note").is_visible()
    finally:
        browser.close()
        p.stop()
    assert shown and poster == "_static/demo/osprey-demo-light-poster.jpg"
    assert note and not speed


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_video_that_cannot_load_leaves_the_poster_alone(site, theme) -> None:
    p, browser, tab = _open(site(theme))
    try:
        state = _state(tab)
    finally:
        browser.close()
        p.stop()
    assert state["poster"] and not state["video"]
    assert state["posterSrc"] == f"_static/demo/osprey-demo-{theme}-poster.jpg"
    assert not state["speed"] and not state["note"]
    # A plain still: no dead play bar invites a click.
    assert not state["controls"]


def test_a_playing_video_starts_at_real_time_with_its_controls(site) -> None:
    site.add_videos()
    p, browser, tab = _open(site())
    try:
        state = _state(tab)
    finally:
        browser.close()
        p.stop()
    assert state["video"] and not state["poster"]
    assert state["speed"] and state["note"]
    assert state["rate"] == 1 and state["pressed"] == ["1"]
    assert state["controls"]
    assert not state["paused"]


def test_reduced_motion_does_not_autoplay(site) -> None:
    site.add_videos()
    p, browser, tab = _open(site(), reduced_motion="reduce")
    try:
        state = _state(tab)
    finally:
        browser.close()
        p.stop()
    assert state["video"] and state["paused"]


# --- a page one level down embeds the same player -----------------------------

NESTED_BLOCK = """\
<div class="osprey-demo">
  <video id="osprey-second" preload="metadata" muted loop playsinline controls
         poster="../_static/demo/osprey-second-light-poster.jpg"
         aria-label="Video of a second recording in the web terminal"
         aria-describedby="osprey-second-note">
    <source src="../_static/demo/osprey-second-light.mp4" type="video/mp4">
    <img src="../_static/demo/osprey-second-light-poster.jpg"
         alt="The web terminal.">
  </video>
  <p class="osprey-demo-note" id="osprey-second-note">
    One unedited session on the control-assistant demo, played in real time.
  </p>
  <div class="osprey-demo-speed" role="group" aria-label="Playback speed" hidden>
    <span>Speed</span>
    <button type="button" data-rate="1">1×</button>
  </div>
</div>
<script src="../_static/js/doc-video.js"></script>
"""


def test_a_nested_block_follows_the_dark_theme(tmp_path: Path) -> None:
    if not chromium_available():
        pytest.skip("Playwright Chromium is not installed")
    from PIL import Image

    demo = tmp_path / "_static" / "demo"
    demo.mkdir(parents=True)
    for theme in ("light", "dark"):
        Image.new("RGB", (64, 36)).save(demo / f"osprey-second-{theme}-poster.jpg")
    (tmp_path / "_static" / "js").mkdir()
    shutil.copy(PLAYER, tmp_path / "_static" / "js" / "doc-video.js")
    page = tmp_path / "how-to" / "index.html"
    page.parent.mkdir()
    page.write_text(
        f'<!doctype html><html data-theme="dark"><body>{NESTED_BLOCK}</body></html>',
        encoding="utf-8",
    )
    p, browser, tab = _open(page)
    try:
        poster = tab.evaluate(
            "() => document.querySelector('.osprey-demo-poster')?.getAttribute('src')"
            " || document.getElementById('osprey-second').getAttribute('poster')"
        )
    finally:
        browser.close()
        p.stop()
    assert poster == "../_static/demo/osprey-second-dark-poster.jpg"
