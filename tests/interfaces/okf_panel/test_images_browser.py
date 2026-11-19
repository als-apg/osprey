"""Browser tests: bundle images in the KNOWLEDGE panel's reading pane.

What no unit test can show runs here, in a real browser against a live panel:

  1. the example bundle's figure is displayed (a non-zero natural size), with the
     panel served on its own and with the panel opened in the web terminal the way
     a chat citation opens it, and when the concept is asked for while the panel's
     start-up overview is still loading;
  2. a PNG in each of the three image syntaxes, and an SVG, are displayed;
  3. a concept holding remote references of every kind the reader's sanitiser
     must neutralise makes no request to any of the hosts it names, and shows each
     image's alt text in place of the picture;
  4. a missing, a disallowed, an over-limit and a corrupt image each show their
     alt text and a short reason;
  5. a concept opened by a leading-slash id shows its image;
  6. an SVG carrying a ``script``, a ``style`` element and a remote image runs no
     script and fetches nothing, embedded in a concept and opened directly at its
     asset URL, standalone and through the web terminal.

Every image except the example figure is generated into ``tmp_path`` at test time.
The no-request assertions name only this module's own hosts (``*.invalid``), so they
hold whether the panel's vendor scripts come from a CDN or from the panel itself.

Run:
    uv run pytest tests/interfaces/okf_panel/test_images_browser.py -m browser
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote, urlsplit

import pytest
from PIL import Image
from playwright.sync_api import Browser, BrowserContext, Frame, Page, Route, expect

import osprey
from osprey.interfaces._serving import run_app_server
from osprey.services.facility_knowledge.okf.assets import MAX_IMAGE_BYTES
from tests.interfaces._browser import wait_for_dock_settled, wait_for_page_ready
from tests.interfaces._hub import hub_server
from tests.interfaces._panel_host import open_panel_host
from tests.interfaces._panel_launch import publish_panel_urls
from tests.interfaces.web_terminal._chat_server import PLANS, live_chat_server

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: The shipped example bundle; ``devices/bpm`` embeds its one figure.
EXAMPLE_BUNDLE = Path(osprey.__file__).parent / "templates" / "facilities" / "example" / "knowledge"
EXAMPLE_FIGURE_ALT_START = "Cross-section of a button BPM"

_LOAD_TIMEOUT_MS = 10_000

#: How long a page is left to make any late request before "nothing was fetched"
#: is read. The concept under test also holds a local image that must load first,
#: so the reader has already requested everything it was going to.
_QUIET_S = 0.75

# Remote hosts this module names, one per vector. ``.invalid`` never resolves;
# every request to one is also aborted at the context so none leaves the machine.
REMOTE_HOSTS: dict[str, str] = {
    "inline": "okf-inline.invalid",
    "reference": "okf-reference.invalid",
    "html": "okf-html.invalid",
    "protocol_relative": "okf-protocol-relative.invalid",
    "srcset": "okf-srcset.invalid",
    "style_attr": "okf-style-attr.invalid",
    "style_element": "okf-style-element.invalid",
    "background": "okf-background.invalid",
    "input_image": "okf-input-image.invalid",
    "video_poster": "okf-video-poster.invalid",
    "audio": "okf-audio.invalid",
    "iframe": "okf-iframe.invalid",
    "object": "okf-object.invalid",
    "svg_image": "okf-svg-image.invalid",
    "svg_feimage": "okf-svg-feimage.invalid",
}

#: Hosts the hostile SVG names: its ``<image>``, its ``<style>`` import and its script.
SVG_HOSTS = ("okf-svgfile-image.invalid", "okf-svgfile-style.invalid", "okf-svgfile-script.invalid")
SVG_SCRIPT_MARKER = "okf-svg-script-ran"

HOSTILE_SVG = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="60" height="40" viewBox="0 0 60 40">
  <style>@import url("http://{SVG_HOSTS[1]}/s.css"); rect {{ fill: #c03; }}</style>
  <rect id="body" width="60" height="40"/>
  <image href="http://{SVG_HOSTS[0]}/i.png" xlink:href="http://{SVG_HOSTS[0]}/x.png"
         width="10" height="10"/>
  <script>
    console.log("{SVG_SCRIPT_MARKER}");
    document.documentElement.setAttribute("data-ran", "1");
    try {{ fetch("http://{SVG_SCRIPT_MARKER}.invalid/f"); }} catch (e) {{}}
    try {{ new Image().src = "http://{SVG_HOSTS[2]}/img"; }} catch (e) {{}}
  </script>
</svg>
"""


# ---------------------------------------------------------------------------
# Bundle built by the test (decision 13: nothing but the example figure is committed)
# ---------------------------------------------------------------------------


def _png(width: int = 48, height: int = 32, color: tuple[int, int, int] = (20, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _bmp() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format="BMP")
    return buf.getvalue()


def _concept(root: Path, concept_id: str, title: str, body: str) -> None:
    path = root / f"{concept_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntype: Reference\ntitle: {title}\ndescription: {title} for the image tests.\n"
        f"---\n\n{body}\n",
        encoding="utf-8",
    )


def _remote_corpus_body() -> str:
    h = REMOTE_HOSTS
    return f"""Remote references; none of them may be requested.

![Remote inline](http://{h["inline"]}/a.png)

![Remote reference][remote]

<img src="https://{h["html"]}/c.png" alt="Remote html">

![Remote protocol-relative](//{h["protocol_relative"]}/d.png)

![Data image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==)

<img srcset="http://{h["srcset"]}/e.png 1x, http://{h["srcset"]}/e2.png 2x" alt="Remote srcset">

<div style="background-image: url('http://{h["style_attr"]}/f.png')">styled block</div>

<style>body {{ background: url("http://{h["style_element"]}/g.png"); }}</style>

<table background="http://{h["background"]}/h.png"><tr><td>cell</td></tr></table>

<input type="image" src="http://{h["input_image"]}/i.png" alt="Remote input">

<video poster="http://{h["video_poster"]}/j.png" src="http://{h["video_poster"]}/j.mp4"></video>

<audio src="http://{h["audio"]}/k.mp3" autoplay></audio>

<iframe src="http://{h["iframe"]}/l.html"></iframe>

<object data="http://{h["object"]}/m.svg" type="image/svg+xml"></object>

<svg width="20" height="20"><image href="http://{h["svg_image"]}/n.png" width="20" height="20"/><filter id="f"><feImage href="http://{h["svg_feimage"]}/o.png"/></filter></svg>

![Local sentinel](figures/inline.png)

[remote]: https://{h["reference"]}/b.png
"""


@pytest.fixture
def image_bundle(tmp_path: Path) -> Path:
    """A bundle whose concepts exercise every image outcome the reader has."""
    root = tmp_path / "bundle"
    figures = root / "pages" / "figures"
    figures.mkdir(parents=True)
    (figures / "inline.png").write_bytes(_png(48, 32))
    (figures / "ref.png").write_bytes(_png(40, 30, (200, 100, 20)))
    (figures / "html.png").write_bytes(_png(36, 24, (20, 160, 60)))
    (figures / "with space.png").write_bytes(_png(30, 20, (90, 90, 90)))
    (figures / "hostile.svg").write_text(HOSTILE_SVG, encoding="utf-8")
    (figures / "picture.bmp").write_bytes(_bmp())
    (figures / "huge.png").write_bytes(_png() + b"\0" * (MAX_IMAGE_BYTES + 1))
    (figures / "corrupt.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"not really a png" * 64)
    (root / "shared").mkdir()
    (root / "shared" / "root.png").write_bytes(_png(44, 28, (150, 20, 150)))

    _concept(
        root,
        "pages/syntaxes",
        "Image syntaxes",
        "![PNG inline](figures/inline.png)\n\n"
        "![PNG reference][ref]\n\n"
        '<img src="figures/html.png" alt="PNG html">\n\n'
        "![PNG root-relative](/shared/root.png)\n\n"
        "![PNG percent-encoded](figures/with%20space.png)\n\n"
        "![SVG figure](figures/hostile.svg)\n\n"
        "[ref]: figures/ref.png\n",
    )
    _concept(root, "pages/remote", "Remote references", _remote_corpus_body())
    _concept(
        root,
        "pages/broken",
        "Broken images",
        "![Missing figure](figures/gone.png)\n\n"
        "![Disallowed figure](figures/picture.bmp)\n\n"
        "![Oversized figure](figures/huge.png)\n\n"
        "![Corrupt figure](figures/corrupt.png)\n\n"
        "![Working figure](figures/inline.png)\n",
    )
    _concept(root, "pages/hostile-svg", "Hostile SVG", "![Hostile SVG](figures/hostile.svg)\n")
    (root / "index.md").write_text(
        "# Image test bundle\n\n"
        "- [Image syntaxes](pages/syntaxes.md) - Image syntaxes for the image tests.\n"
        "- [Remote references](pages/remote.md) - Remote references for the image tests.\n"
        "- [Broken images](pages/broken.md) - Broken images for the image tests.\n"
        "- [Hostile SVG](pages/hostile-svg.md) - Hostile SVG for the image tests.\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Servers, engines, probes
# ---------------------------------------------------------------------------


@pytest.fixture
def okf_url(image_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The panel, served on its own over :func:`image_bundle`."""
    from osprey.interfaces.okf_panel.app import create_app

    monkeypatch.chdir(tmp_path)
    with run_app_server(create_app(bundle_path=str(image_bundle))) as url:
        yield url


@pytest.fixture
def hub_okf_url(okf_url: str, tmp_path: Path) -> Iterator[str]:
    """The same panel behind a live web terminal: its address there, ``<hub>/panel/okf``."""
    watch = tmp_path / "hub"
    watch.mkdir()
    with hub_server(
        watch, enabled_panels=("okf",), launch=publish_panel_urls({"okf": okf_url})
    ) as (base_url, _app):
        yield f"{base_url}/panel/okf"


@pytest.fixture(params=["chromium_browser", "webkit_browser"], ids=["chromium", "webkit"])
def engine_browser(request: pytest.FixtureRequest) -> Browser:
    """Each browser engine in turn."""
    return request.getfixturevalue(request.param)


class RequestProbe:
    """Every request a context makes to a ``.invalid`` host, and every console line.

    ``urls``/``hosts`` hold every request the page made, including one the
    browser refuses at a response's policy before it leaves the page (it fails
    with ``csp`` and is listed in ``failed``). ``routed`` holds the ones that
    reached the network, where the context's route aborts them, so none waits
    on a lookup that cannot succeed.
    """

    def __init__(self, context: BrowserContext) -> None:
        self.hosts: list[str] = []
        self.console: list[str] = []
        self.routed: list[str] = []
        self.urls: list[str] = []
        self.failed: list = []
        context.on("requestfailed", lambda r: self.failed.append((r.url, r.failure)))
        context.on("request", self._on_request)
        context.route(lambda url: _is_test_host(url), self._abort)

    def _on_request(self, request) -> None:
        if _is_test_host(request.url):
            self.urls.append(request.url)
            self.hosts.append(urlsplit(request.url).hostname or "")

    def _abort(self, route) -> None:
        self.routed.append(route.request.url)
        route.abort()

    def watch(self, page: Page) -> Page:
        page.on("console", lambda msg: self.console.append(msg.text))
        return page

    def requested(self, hosts) -> list[str]:
        wanted = set(hosts)
        return [h for h in self.hosts if h in wanted]


def _is_test_host(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return host.endswith(".invalid")


def _open_standalone(context: BrowserContext, panel_url: str, concept_id: str) -> Page:
    """Open the panel on its own, on *concept_id*, and wait for that concept's page."""
    page = context.new_page()
    page.goto(f"{panel_url}/#{quote(concept_id, safe='/')}", wait_until="domcontentloaded")
    wait_for_page_ready(page)
    _wait_concept(page, concept_id)
    return page


def _wait_until(page: Page, condition, timeout_s: float = _LOAD_TIMEOUT_MS / 1000) -> None:
    """Let *page*'s event handlers run until *condition* holds; fail past *timeout_s*."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        page.wait_for_timeout(20)


def _wait_concept(target: Page | Frame, concept_id: str) -> None:
    reader = target.locator(f'#reader-content[data-concept-id="{concept_id}"]')
    expect(reader).to_have_count(1, timeout=_LOAD_TIMEOUT_MS)
    expect(reader.locator(".osprey-md-rendered")).to_have_count(1, timeout=_LOAD_TIMEOUT_MS)


#: Resolves with the image whose alt text starts with the argument, once it has
#: loaded and decoded to a non-zero size.
_IMAGE_SHOWN_JS = """
(altStart) => {
  const img = [...document.querySelectorAll('#reader-content img')]
    .find((el) => (el.getAttribute('alt') || '').startsWith(altStart));
  return !!img && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0;
}
"""

_IMAGE_FACTS_JS = """
(altStart) => {
  const img = [...document.querySelectorAll('#reader-content img')]
    .find((el) => (el.getAttribute('alt') || '').startsWith(altStart));
  const box = img.getBoundingClientRect();
  return {
    src: img.getAttribute('src'),
    naturalWidth: img.naturalWidth,
    naturalHeight: img.naturalHeight,
    shownWidth: box.width,
    shownHeight: box.height,
  };
}
"""


def assert_image_displayed(target: Page | Frame, alt_start: str) -> dict:
    """Wait for the reading pane's image *alt_start* to be on screen; return its facts."""
    target.wait_for_function(_IMAGE_SHOWN_JS, arg=alt_start, timeout=_LOAD_TIMEOUT_MS)
    facts = target.evaluate(_IMAGE_FACTS_JS, alt_start)
    assert facts["naturalWidth"] > 0 and facts["naturalHeight"] > 0, facts
    assert facts["shownWidth"] > 0 and facts["shownHeight"] > 0, facts
    assert facts["src"].startswith(("/api/asset?", "/panel/okf/api/asset?")), facts
    return facts


def assert_shown_as_alt_text(target: Page | Frame, alt: str, reason: str) -> None:
    """The image *alt* is replaced by its alt text and *reason*; no picture stands for it."""
    fallback = target.locator("#reader-content .okf-image-fallback", has_text=alt)
    expect(fallback).to_have_count(1, timeout=_LOAD_TIMEOUT_MS)
    expect(fallback).to_be_visible()
    expect(fallback.locator(".okf-image-fallback__reason")).to_contain_text(reason)
    expect(target.locator(f'#reader-content img[alt="{alt}"]')).to_have_count(0)


def _quiet(page: Page) -> None:
    """Leave the page time for any late request before the probe is read."""
    page.wait_for_timeout(_QUIET_S * 1000)


# ---------------------------------------------------------------------------
# 1. The example bundle's figure
# ---------------------------------------------------------------------------


def test_example_figure_is_displayed_standalone(engine_browser, tmp_path, monkeypatch):
    """The shipped figure decodes to a real picture in the panel served on its own."""
    from osprey.interfaces.okf_panel.app import create_app

    monkeypatch.chdir(tmp_path)
    with run_app_server(create_app(bundle_path=str(EXAMPLE_BUNDLE))) as url:
        context = engine_browser.new_context()
        page = _open_standalone(context, url, "devices/bpm")
        facts = assert_image_displayed(page, EXAMPLE_FIGURE_ALT_START)
        assert facts["src"] == (
            "/api/asset?target=figures%2Fbpm-button-pickups.png&from=devices%2Fbpm"
        ), facts
        context.close()


def test_example_figure_is_displayed_from_a_chat_citation(tmp_path, chromium_browser):
    """A citation in a chat answer opens KNOWLEDGE on the concept, figure displayed.

    The panel is reached through the web terminal's panel proxy, so the image
    request goes through it too.
    """
    from osprey.interfaces.okf_panel.app import create_app

    with (
        run_app_server(create_app(str(EXAMPLE_BUNDLE))) as okf_url,
        live_chat_server(
            tmp_path, enabled_panels=frozenset({"artifacts", "okf"}), panel_urls={"okf": okf_url}
        ) as (base_url, _app),
    ):
        PLANS["show me the bpm figure"] = [
            ("text", "The pickups are drawn in [BPM](panel/okf#devices/bpm)."),
            ("result",),
        ]
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        wait_for_page_ready(page)
        textarea = page.locator("#operator-container .op-input-area textarea")
        expect(textarea).to_be_visible(timeout=_LOAD_TIMEOUT_MS)
        wait_for_dock_settled(page)
        textarea.fill("show me the bpm figure")
        textarea.press("Enter")

        citation = page.locator("#operator-container .op-entry.assistant .osprey-md-rendered a")
        expect(citation).to_have_attribute("href", "panel/okf#devices/bpm", timeout=10_000)
        citation.click()

        tile = page.locator('.dock-iframe-overlay iframe[data-panel-id="okf"]')
        expect(tile).to_be_visible(timeout=_LOAD_TIMEOUT_MS)
        frame = tile.element_handle().content_frame()
        assert frame is not None
        _wait_concept(frame, "devices/bpm")
        facts = assert_image_displayed(frame, EXAMPLE_FIGURE_ALT_START)
        assert facts["src"].startswith("/panel/okf/api/asset?"), facts
        page.close()


def test_a_concept_opened_during_start_up_outlives_the_late_overview(
    chromium_browser, tmp_path, monkeypatch
):
    """A concept asked for while the start-up overview loads stays when the overview answers.

    The overview's answer is held until the concept has rendered, the order a
    citation's fragment navigation meets a panel still starting up.
    """
    from osprey.interfaces.okf_panel.app import create_app

    monkeypatch.chdir(tmp_path)
    with run_app_server(create_app(bundle_path=str(EXAMPLE_BUNDLE))) as url:
        context = chromium_browser.new_context()
        held: list[Route] = []
        context.route("**/api/structure", lambda route: held.append(route))
        page = context.new_page()
        page.goto(f"{url}/", wait_until="domcontentloaded")
        _wait_until(page, lambda: len(held) == 1)

        page.evaluate("location.hash = '#devices/bpm'")
        _wait_concept(page, "devices/bpm")
        assert_image_displayed(page, EXAMPLE_FIGURE_ALT_START)

        held[0].continue_()
        wait_for_page_ready(page)
        expect(page.locator('#reader-content[data-concept-id="devices/bpm"]')).to_have_count(1)
        assert page.evaluate("history.state") == {"id": "devices/bpm"}
        assert_image_displayed(page, EXAMPLE_FIGURE_ALT_START)
        context.close()


# ---------------------------------------------------------------------------
# 2. The three syntaxes and an SVG
# ---------------------------------------------------------------------------


def test_png_in_each_syntax_and_an_svg_are_displayed(engine_browser, okf_url):
    context = engine_browser.new_context()
    page = _open_standalone(context, okf_url, "pages/syntaxes")
    sizes = {
        "PNG inline": (48, 32),
        "PNG reference": (40, 30),
        "PNG html": (36, 24),
        "PNG root-relative": (44, 28),
        "PNG percent-encoded": (30, 20),
        "SVG figure": (60, 40),
    }
    for alt, (width, height) in sizes.items():
        facts = assert_image_displayed(page, alt)
        assert (facts["naturalWidth"], facts["naturalHeight"]) == (width, height), (alt, facts)
    expect(page.locator("#reader-content .okf-image-fallback")).to_have_count(0)
    context.close()


# ---------------------------------------------------------------------------
# 3. Remote references reach no other host
# ---------------------------------------------------------------------------


def test_remote_references_make_no_request_and_show_alt_text(engine_browser, okf_url):
    context = engine_browser.new_context()
    probe = RequestProbe(context)
    page = probe.watch(context.new_page())
    page.goto(f"{okf_url}/#pages/remote", wait_until="domcontentloaded")
    wait_for_page_ready(page)
    _wait_concept(page, "pages/remote")
    # The local image loads, so every request the reader makes has been made.
    assert_image_displayed(page, "Local sentinel")
    _quiet(page)

    assert probe.requested(REMOTE_HOSTS.values()) == [], probe.hosts
    assert probe.hosts == [], probe.hosts

    for alt in (
        "Remote inline",
        "Remote reference",
        "Remote html",
        "Remote protocol-relative",
        "Data image",
    ):
        assert_shown_as_alt_text(page, alt, "external image, not loaded")
    # The only picture left on the page is the one from the bundle.
    sources = page.eval_on_selector_all(
        "#reader-content img", "els => els.map((el) => el.getAttribute('src'))"
    )
    assert sources == ["/api/asset?target=figures%2Finline.png&from=pages%2Fremote"], sources
    context.close()


# ---------------------------------------------------------------------------
# 4. An image that cannot be shown degrades visibly
# ---------------------------------------------------------------------------


def test_unshowable_images_show_alt_text_and_a_reason(engine_browser, okf_url):
    context = engine_browser.new_context()
    page = _open_standalone(context, okf_url, "pages/broken")
    assert_shown_as_alt_text(page, "Missing figure", "file not found")
    assert_shown_as_alt_text(page, "Disallowed figure", "image type not allowed")
    assert_shown_as_alt_text(page, "Oversized figure", "larger than 10 MB")
    assert_shown_as_alt_text(page, "Corrupt figure", "cannot be displayed")
    # The rest of the concept renders normally.
    assert_image_displayed(page, "Working figure")
    context.close()


# ---------------------------------------------------------------------------
# 5. A leading-slash id
# ---------------------------------------------------------------------------


def test_a_concept_opened_by_a_leading_slash_id_shows_its_image(chromium_browser, okf_url):
    context = chromium_browser.new_context()
    page = _open_standalone(context, okf_url, "/pages/syntaxes")
    facts = assert_image_displayed(page, "PNG inline")
    assert (facts["naturalWidth"], facts["naturalHeight"]) == (48, 32), facts
    context.close()


# ---------------------------------------------------------------------------
# 6. A hostile SVG runs nothing and fetches nothing
# ---------------------------------------------------------------------------

_SVG_ASSET_QUERY = "/api/asset?target=figures%2Fhostile.svg&from=pages%2Fhostile-svg"


def _assert_no_script_ran(probe: RequestProbe) -> None:
    assert not [line for line in probe.console if SVG_SCRIPT_MARKER in line], probe.console


@pytest.mark.parametrize("path", ["standalone", "web-terminal"])
def test_hostile_svg_embedded_runs_no_script_and_fetches_nothing(
    path, chromium_browser, okf_url, request
):
    panel_url = okf_url if path == "standalone" else request.getfixturevalue("hub_okf_url")
    context = chromium_browser.new_context()
    probe = RequestProbe(context)
    page = probe.watch(context.new_page())
    host = open_panel_host(page, f"{panel_url}/#pages/hostile-svg")
    _wait_concept(host.frame, "pages/hostile-svg")
    facts = assert_image_displayed(host.frame, "Hostile SVG")
    assert (facts["naturalWidth"], facts["naturalHeight"]) == (60, 40), facts
    if path == "web-terminal":
        assert facts["src"].startswith("/panel/okf/api/asset?"), facts
    _quiet(page)
    # Inside an <img> an SVG is a picture: it asks for nothing at all.
    assert probe.hosts == [], (probe.hosts, probe.failed)
    _assert_no_script_ran(probe)
    context.close()


@pytest.mark.parametrize("path", ["standalone", "web-terminal"])
def test_hostile_svg_opened_directly_runs_no_script_and_fetches_nothing(
    path, chromium_browser, okf_url, request
):
    panel_url = okf_url if path == "standalone" else request.getfixturevalue("hub_okf_url")
    context = chromium_browser.new_context()
    probe = RequestProbe(context)
    page = probe.watch(context.new_page())
    response = page.goto(  # page-ready: the document is the bundle's SVG, not an OSPREY page
        f"{panel_url}{_SVG_ASSET_QUERY}", wait_until="load"
    )
    assert response is not None and response.status == 200, response
    assert response.headers["content-type"].startswith("image/svg+xml"), response.headers
    # The figure itself is the document: it rendered, and its script never ran.
    expect(page.locator("rect#body")).to_have_count(1)
    _quiet(page)
    assert page.locator("svg").first.get_attribute("data-ran") is None
    _assert_no_script_ran(probe)
    # As a document the SVG names its remote image and stylesheet, and the
    # browser stops both at the response's policy: each request is refused
    # before it leaves the page, so none reaches the network (the route).
    assert probe.routed == [], probe.routed
    blocked = {url for url, failure in probe.failed if failure == "csp"}
    attempted = [r for r in probe.urls if _is_test_host(r)]
    assert all(url in blocked for url in attempted), (attempted, probe.failed)
    context.close()
