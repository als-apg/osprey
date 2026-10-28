"""Browser proof that an artifact's script runs on an opaque origin (FR 9).

Every agent-written, non-inert document the gallery serves carries
``Content-Security-Policy: sandbox …`` without ``allow-same-origin``
(:mod:`osprey.interfaces.opaque_origin`). The browser then gives that document
an opaque origin: its requests leave with ``Origin: null`` and without the
console's SameSite cookie, so a script an agent wrote into an HTML artifact
cannot turn a machine's write switch on with the operator's session, however
the operator opens the artifact. A header test pins the header; only a browser
proves the browser honours it on every path an operator actually takes — the
gallery preview, "Open in new tab" and Print — and that the documents still
render, follow the theme and print once they are opaque.

**Everything here runs through the hub.** The artifacts app is served in this
process and the web-terminal hub is pointed at it the way a deployment is
(``_launch_panel_server`` publishes its URL), so every page loads from
``/panel/artifacts/…`` on the hub's origin, through the panel proxy, with the
hub's auth gate on. The browser carries the operator's session cookie
(``chromium_browser`` authorizes every context), which is exactly the
credential an artifact script would try to borrow. A first-party asset an
opaque document imports and the C-ASSETS exemption list leaves out fails
here: the opaque document cannot send the cookie the gate wants.

Each attack test ends with the control: the same request from the gallery's
own (first-party) page **does** turn the switch on. Without it, "the switch
stayed off" could mean the request shape was wrong rather than that the
origin was refused.

Run:
    .venv/bin/pytest tests/interfaces/artifacts/test_opaque_origin_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import yaml

from osprey.interfaces._serving import run_app_server
from osprey_connectors import posture_store
from osprey_connectors_tests._harness import write_control_context
from tests.interfaces._hub import LONG_LIVED_SHELL, hub_server
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.web_terminal._posture_chip import reset_process_memos

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import Frame, Page

with suppress(ImportError):
    from playwright.sync_api import expect

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: One bound for every wait: under parallel load a step that takes
#: milliseconds can take seconds, and a state that never arrives fails either way.
TIMEOUT = 15_000

#: Where the hub serves the gallery.
GALLERY = "/panel/artifacts/"

#: The target every attack tries to turn on. The simulator: no tier or
#: acknowledgement stands between the operator and turning it on, so the
#: first-party control below can flip it and a refusal is the origin's alone.
TARGET = "va"

# ---------------------------------------------------------------------------
# Fixtures: the artifacts
# ---------------------------------------------------------------------------

#: The attack an agent could write into any HTML artifact: turn the simulator's
#: writes on with whatever credential the browser attaches. Two shapes — the
#: chip's own gesture as a credentialed JSON ``fetch`` (read the epoch, POST,
#: retry once on ``409``; needs CORS, which the opaque origin never gets) and a
#: ``no-cors`` simple request (reaches the server; the gate must refuse it).
#: Served same-origin, the first shape turns the switch on.
#: The route is assembled at run time so the proxy's root-path rewrite cannot
#: retarget it at the artifacts app. The outcome lands on ``<html
#: data-probe>`` and is posted to whichever window embeds or opened the page.
_ATTACK_SCRIPT = """<script>
(function () {
  var url = location.protocol + '//' + location.host + '/' + 'api/terminal/posture';
  var body = JSON.stringify({ target: '__TARGET__', state: 'on', epoch: null });
  var out = { origin: String(self.origin) };
  try { out.cookie = document.cookie; } catch (e) { out.cookie = 'denied'; }
  function post(epoch) {
    return fetch(url, { method: 'POST', credentials: 'include',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ target: '__TARGET__', state: 'on', epoch: epoch }) });
  }
  var a = fetch(url, { credentials: 'include' })
    .then(function (r) { return r.json(); })
    .then(function (v) { return v.switch_epoch; }, function () { return null; })
    .then(post)
    .then(function (r) {
      return r.status === 409
        ? r.json().then(function (j) { return post(j.detail.epoch); }) : r;
    })
    .then(function (r) { out.cors = r.status; }, function () { out.cors = 'blocked'; });
  var b = fetch(url, { method: 'POST', mode: 'no-cors', credentials: 'include',
                       headers: { 'Content-Type': 'text/plain' }, body: body })
    .then(function (r) { out.simple = r.type; }, function () { out.simple = 'blocked'; });
  Promise.all([a, b]).then(function () {
    document.documentElement.setAttribute('data-probe', JSON.stringify(out));
    var peer = window.opener || window.parent;
    if (peer && peer !== window) { peer.postMessage({ type: 'osprey-test-probe' }, '*'); }
  });
})();
</script>""".replace("__TARGET__", TARGET)

ATTACK_TITLE = "Beam report"
_ATTACK_HTML = (
    "<html><head><meta charset='utf-8'><title>report</title></head><body>"
    "<p id='report'>Beam report</p>" + _ATTACK_SCRIPT + "</body></html>"
)

# A Plotly page the way ``go.Figure.to_html(include_plotlyjs=False)`` writes
# one, with a dark palette baked into the layout: the bridge must re-theme it.
PLOT_TITLE = "Orbit plot"
_PLOT_HTML = """<html>
<head><meta charset="utf-8" /></head>
<body>
<div style="height:500px; width:700px;">
<div id="plot-a" class="plotly-graph-div" style="height:100%; width:100%;"></div>
<script>
Plotly.newPlot("plot-a",
  [{"type":"scatter","x":[0,1,2],"y":[1,3,2],"mode":"lines"}],
  {"paper_bgcolor":"rgba(15, 15, 25, 1)","plot_bgcolor":"rgba(15, 15, 25, 1)",
   "font":{"color":"white"},"width":700,"height":500});
</script>
</div>
</body>
</html>"""

#: A markdown artifact carrying both script-bearing HTML and math. Each script
#: vector marks the document it runs in; the math must still typeset.
MARKDOWN_TITLE = "Shift notes"
_MARKDOWN = """# Shift notes

Emittance follows $\\epsilon = \\sigma^2 / \\beta$ along the ring.

$$E = mc^2$$

<img src="x" onerror="document.documentElement.setAttribute('data-md-pwned', 'onerror')">

<script>document.documentElement.setAttribute('data-md-pwned', 'script');</script>

<a href="javascript:document.documentElement.setAttribute('data-md-pwned','href')">link</a>
"""

NOTEBOOK_TITLE = "Tune scan notebook"
NOTEBOOK_TOKEN = "UNIQUE_CELL_TOKEN"

PDF_TITLE = "Shift summary PDF"


def _notebook_bytes() -> bytes:
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Tune scan\n\nInline math $x^2$."),
        nbformat.v4.new_code_cell(f"print('{NOTEBOOK_TOKEN}')"),
    ]
    return nbformat.writes(nb).encode()


def _pdf_bytes() -> bytes:
    """A minimal one-page PDF with a text line, cross-reference table and all."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        None,  # the content stream, filled in below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = b"BT /F1 18 Tf 20 70 Td (Shift summary) Tj ET"
    objects[3] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


@dataclass(frozen=True)
class Seeded:
    """What one hub launch serves: its address and each artifact's id."""

    base_url: str
    ids: dict[str, str]
    paths: dict[str, str]


def _seed(workspace: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Save the fixture artifacts; return their ids and file routes by title."""
    from osprey.stores.artifact_store import ArtifactStore

    store = ArtifactStore(workspace_root=workspace)
    specs = [
        (ATTACK_TITLE, _ATTACK_HTML.encode(), "report.html", "html", "text/html"),
        (PLOT_TITLE, _PLOT_HTML.encode(), "orbit.html", "plot_html", "text/html"),
        (MARKDOWN_TITLE, _MARKDOWN.encode(), "notes.md", "markdown", "text/markdown"),
        (
            NOTEBOOK_TITLE,
            _notebook_bytes(),
            "scan.ipynb",
            "notebook",
            "application/x-ipynb+json",
        ),
        (PDF_TITLE, _pdf_bytes(), "summary.pdf", "file", "application/pdf"),
    ]
    ids: dict[str, str] = {}
    paths: dict[str, str] = {}
    for title, content, filename, artifact_type, mime in specs:
        entry = store.save_file(
            file_content=content,
            filename=filename,
            artifact_type=artifact_type,
            title=title,
            description="fixture",
            mime_type=mime,
            tool_source="test_fixture",
        )
        ids[title] = entry.id
        paths[title] = f"{GALLERY}files/{entry.id}/{entry.filename}"
    return ids, paths


# ---------------------------------------------------------------------------
# The hub
# ---------------------------------------------------------------------------


def _write_config(path: Path) -> Path:
    """A render with the simulator as a control target and writes armed."""
    gateway = {"address": "gw", "port": 5064, "use_name_server": True}
    standin = {"address": "localhost", "port": 5074, "use_name_server": True}
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "live_standin",
                    "writes": "allowed",
                    "limits_checking": {"enabled": True, "mode": "exclusive"},
                    "connector": {
                        "live_standin": {
                            "probe_channel": "SR:PROBE",
                            "gateways": {"read_only": dict(standin), "write_access": dict(standin)},
                        },
                        "virtual_accelerator": {
                            "probe_channel": "SIM:PROBE",
                            "gateways": {"read_only": dict(gateway)},
                        },
                    },
                },
                "services": {"live_standin": {"port": 5074}},
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def hub_with_gallery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed=_seed,
    offline: bool = True,
    web_theme: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Serve the artifacts app behind a real hub, auth gate on, record owned.

    The artifacts app runs on its own port in this process; the hub publishes
    that address as the ``artifact`` panel, so ``/panel/artifacts/…`` on the
    hub is the proxied gallery — the path a deployment's browser takes. A
    control-context record owned by this process is written before the hub
    starts, so ``POST /api/terminal/posture`` is judgeable and a first-party
    turn-on can land (the control every attack is measured against).

    ``OSPREY_OFFLINE=1`` by default: every vendor library (Plotly, KaTeX and
    its fonts, marked, DOMPurify, highlight.js) is then loaded from the
    artifacts app's own ``/static/`` tree *through the proxy*, so an asset the
    C-ASSETS exemption list leaves out fails here instead of being quietly
    fetched from a CDN. That needs ``osprey vendor fetch`` to have run on the
    host, which the gate host has.

    Args:
        seed: Called with the artifacts workspace before either app starts;
            its return value is yielded beside the hub's address.

    Yields:
        ``(base_url, seeded)`` — the hub's address and ``seed``'s result.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_OFFLINE", "1" if offline else "0")
    if web_theme is None:
        monkeypatch.delenv("OSPREY_WEB_THEME", raising=False)
    else:
        monkeypatch.setenv("OSPREY_WEB_THEME", web_theme)

    artifacts_root = tmp_path / "artifacts_ws"
    artifacts_root.mkdir()
    seeded = seed(artifacts_root)

    watch = tmp_path / "_agent_data"
    watch.mkdir(exist_ok=True)
    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    config = _write_config(tmp_path / "config.yml")
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    reset_process_memos()
    write_control_context(root, target=TARGET, generation=1)

    from osprey.interfaces.artifacts.app import create_app as create_artifacts_app

    artifacts_app = create_artifacts_app(workspace_root=artifacts_root)
    try:
        with (
            run_app_server(artifacts_app) as artifacts_url,
            hub_server(
                watch,
                patch(
                    "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery"
                    ".snapshot_session_ids",
                    side_effect=lambda *_a, **_k: set(),
                ),
                launch=publish_artifact_url(artifacts_url),
                shell_command=LONG_LIVED_SHELL,
                config_path=config,
            ) as (base_url, _hub),
        ):
            yield base_url, seeded
    finally:
        reset_process_memos()


@pytest.fixture
def hub(tmp_path, monkeypatch) -> Iterator[Seeded]:
    with hub_with_gallery(tmp_path, monkeypatch) as (base_url, (ids, paths)):
        yield Seeded(base_url, ids, paths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _switch(page: Page, base_url: str) -> str | None:
    """The simulator's write switch as the operator's own session reads it."""
    resp = page.request.get(f"{base_url}/api/terminal/posture")
    assert resp.ok, resp.status
    view = resp.json()
    for row in view["targets"]:
        if row["target"] == TARGET:
            return row["switch"]
    raise AssertionError(f"no {TARGET!r} row in {json.dumps(view)[:2000]}")


def _first_party_turn_on(page: Page, epoch: str | None) -> dict:
    """POST the attack's own request from the gallery's first-party page."""
    return page.evaluate(
        """async ([target, epoch]) => {
          const r = await fetch('/' + 'api/terminal/posture', {
            method: 'POST', credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target, state: 'on', epoch }),
          });
          let body = null;
          try { body = await r.json(); } catch (e) { body = null; }
          return { status: r.status, body };
        }""",
        [TARGET, epoch],
    )


def _assert_control_flips(page: Page, base_url: str) -> None:
    """The same request from first-party script turns the switch on.

    Sent the way the chip sends it: with the ``switch_epoch`` it read, and
    retried once with the epoch a ``409 stopped_since`` names (an epoch-less
    record is stamped on the first gesture).
    """
    epoch = page.request.get(f"{base_url}/api/terminal/posture").json().get("switch_epoch")
    answer = _first_party_turn_on(page, epoch)
    if answer["status"] == 409:
        answer = _first_party_turn_on(page, answer["body"]["detail"]["epoch"])
    assert answer["status"] == 200, f"first-party turn-on answered {answer}"
    assert _switch(page, base_url) == "on"


def _probe(frame: Frame | Page) -> dict:
    """Wait for the attack script to report, and return what it saw."""
    frame.wait_for_function(
        "() => document.documentElement.hasAttribute('data-probe')", timeout=TIMEOUT
    )
    return json.loads(frame.evaluate("() => document.documentElement.getAttribute('data-probe')"))


def _assert_refused(probe: dict) -> None:
    assert probe["origin"] == "null", probe
    # An opaque document has no cookie jar to read.
    assert probe["cookie"] in ("denied", ""), probe
    # The credentialed JSON request needs CORS the console never grants.
    assert probe["cors"] == "blocked", probe


def open_gallery(page: Page, base_url: str, theme: str = "dark") -> None:
    page.goto(f"{base_url}{GALLERY}?theme={theme}")
    expect(page.locator(".tree-item").first).to_be_visible(timeout=TIMEOUT)


def select_artifact(page: Page, title: str) -> None:
    page.locator(".tree-item", has_text=title).click()
    expect(page.locator("#preview-content .preview-header-title")).to_have_text(
        title, timeout=TIMEOUT
    )


def artifact_frame(page: Page, artifact_id: str) -> Frame:
    """The preview iframe's frame for one artifact, once it has navigated.

    Under parallel load the preview can re-render while the first navigation is still in
    flight; that navigation then aborts (``net::ERR_ABORTED``) and a fresh iframe takes its
    place. A re-render can also land just after the first load, detaching the frame that
    loaded. Only an aborted or detached navigation, an iframe detached before its frame
    could be read, or a frame detached before its first script ran, is retried, against
    the iframe that replaced it; the frame must still land on the artifact's own URL.
    """
    from playwright.sync_api import Error as PlaywrightError

    for attempt in range(3):
        iframe = page.locator("#preview-content .preview-viewport iframe")
        expect(iframe).to_have_count(1, timeout=TIMEOUT)
        handle = iframe.element_handle()
        assert handle is not None
        frame = handle.content_frame()
        if frame is None:
            assert attempt < 2, "the preview iframe never held a frame"
            continue
        try:
            frame.wait_for_url(f"**{artifact_id}**", timeout=TIMEOUT)
            frame.wait_for_load_state("load", timeout=TIMEOUT)
            # The loaded frame must answer a script before it is handed out.
            frame.evaluate("() => 0")
        except PlaywrightError as error:
            # Both name a navigation cut short by the re-render: Chromium aborts it,
            # or Playwright reports the frame it was waiting on detached.
            message = str(error).lower()
            aborted = "err_aborted" in message or "frame was detached" in message
            if not aborted or attempt == 2:
                raise
            continue
        return frame
    raise AssertionError("unreachable")


def _stub_print(page: Page) -> None:
    """Make ``window.print`` mark the document instead of opening a dialog.

    Installed as an init script, so it is in place in every document the
    context loads — the print popup included — before that document's own
    scripts run.
    """
    page.context.add_init_script(
        "window.print = function () {"
        " document.documentElement.setAttribute('data-printed', '1'); };"
    )


def _print_popup(page: Page) -> Page:
    """Press the preview's Print button; return the window it opened."""
    with page.expect_popup(timeout=TIMEOUT) as popup:
        page.locator("#preview-content .print-action-btn").click()
    return popup.value


def _wait_printed(target: Page | Frame) -> None:
    target.wait_for_function(
        "() => document.documentElement.getAttribute('data-printed') === '1'", timeout=TIMEOUT
    )


# ---------------------------------------------------------------------------
# An artifact's script cannot turn a switch on
# ---------------------------------------------------------------------------


def test_artifact_script_in_gallery_cannot_turn_writes_on(hub, chromium_browser):
    """In the gallery preview the script runs, opaque, and the switch stays off."""
    page = chromium_browser.new_page()
    assert _switch(page, hub.base_url) == "off"
    open_gallery(page, hub.base_url)
    select_artifact(page, ATTACK_TITLE)

    iframe = page.locator("#preview-content .preview-viewport iframe")
    sandbox = iframe.get_attribute("sandbox") or ""
    assert "allow-scripts" in sandbox and "allow-same-origin" not in sandbox

    probe = _probe(artifact_frame(page, hub.ids[ATTACK_TITLE]))
    _assert_refused(probe)
    assert _switch(page, hub.base_url) == "off"

    _assert_control_flips(page, hub.base_url)


def test_artifact_script_in_new_tab_cannot_turn_writes_on(hub, chromium_browser):
    """Opened top-level ("Open in new tab"), the header alone makes it opaque."""
    page = chromium_browser.new_page()
    open_gallery(page, hub.base_url)
    select_artifact(page, ATTACK_TITLE)

    with page.expect_popup(timeout=TIMEOUT) as popup_info:
        page.locator("#preview-content a[data-theme-link]").click()
    popup = popup_info.value
    popup.wait_for_load_state("load", timeout=TIMEOUT)
    assert hub.ids[ATTACK_TITLE] in popup.url
    assert GALLERY in popup.url

    _assert_refused(_probe(popup))
    expect(popup.locator("#report")).to_have_text("Beam report")
    assert _switch(page, hub.base_url) == "off"

    _assert_control_flips(page, hub.base_url)


def test_artifact_script_in_print_window_cannot_turn_writes_on(hub, chromium_browser):
    """Print opens the page in its own window; it prints, and stays opaque."""
    page = chromium_browser.new_page()
    _stub_print(page)
    open_gallery(page, hub.base_url)
    select_artifact(page, ATTACK_TITLE)

    popup = _print_popup(page)
    _assert_refused(_probe(popup))
    _wait_printed(popup)
    assert _switch(page, hub.base_url) == "off"

    _assert_control_flips(page, hub.base_url)


def test_direct_navigation_to_artifact_route_is_opaque(hub, chromium_browser):
    """A link straight to the proxied file route (no gallery involved) is
    opaque too: the sandbox comes from the response, not from the frame."""
    page = chromium_browser.new_page()
    response = page.goto(f"{hub.base_url}{hub.paths[ATTACK_TITLE]}")
    assert response is not None and response.ok
    csp = response.headers.get("content-security-policy", "")
    assert csp.startswith("sandbox") and "allow-same-origin" not in csp

    _assert_refused(_probe(page))
    assert _switch(page, hub.base_url) == "off"


# ---------------------------------------------------------------------------
# Markdown: sanitized in the gallery, math still typesets
# ---------------------------------------------------------------------------


def test_markdown_script_html_runs_nothing_in_gallery(hub, chromium_browser):
    """The gallery renders markdown into its own (first-party) document, so
    the sanitizer is the only barrier: no vector may run, math must render."""
    page = chromium_browser.new_page()
    open_gallery(page, hub.base_url)
    select_artifact(page, MARKDOWN_TITLE)

    rendered = page.locator("#preview-content .osprey-md-rendered")
    expect(rendered.locator("h1")).to_have_text("Shift notes", timeout=TIMEOUT)
    expect(rendered.locator(".katex").first).to_be_visible(timeout=TIMEOUT)
    # Let the broken image fire its error event, then look for any trace.
    page.wait_for_timeout(500)
    assert rendered.locator("script").count() == 0
    assert rendered.locator("[onerror]").count() == 0
    hrefs = rendered.locator("a").evaluate_all("els => els.map(a => a.getAttribute('href'))")
    assert not any((h or "").lower().startswith("javascript:") for h in hrefs), hrefs
    for anchor in rendered.locator("a").all():
        with suppress(Exception):
            anchor.click(timeout=2_000)
    assert page.evaluate("() => document.documentElement.getAttribute('data-md-pwned')") is None
    assert _switch(page, hub.base_url) == "off"


def test_markdown_math_renders_opaque_in_new_tab_and_prints(hub, chromium_browser):
    """The rendered markdown page is opaque, loads KaTeX and its fonts through
    the proxy (C-ASSETS), runs none of the artifact's script, and prints."""
    page = chromium_browser.new_page()
    _stub_print(page)
    open_gallery(page, hub.base_url, theme="light")
    select_artifact(page, MARKDOWN_TITLE)

    with page.expect_popup(timeout=TIMEOUT) as popup_info:
        page.locator("#preview-content a[data-theme-link]").click()
    tab = popup_info.value
    tab.wait_for_load_state("load", timeout=TIMEOUT)
    assert "/api/markdown/" in tab.url and GALLERY in tab.url
    assert tab.evaluate("() => String(self.origin)") == "null"
    expect(tab.locator(".katex").first).to_be_visible(timeout=TIMEOUT)
    tab.wait_for_function(
        "() => document.fonts.check('16px KaTeX_Main')"
        " && getComputedStyle(document.querySelector('.katex')).fontFamily.includes('KaTeX')",
        timeout=TIMEOUT,
    )
    tab.wait_for_function(
        "() => document.documentElement.getAttribute('data-theme') === 'light'", timeout=TIMEOUT
    )
    tab.wait_for_timeout(500)
    assert tab.evaluate("() => document.documentElement.getAttribute('data-md-pwned')") is None

    popup = _print_popup(page)
    expect(popup.locator(".katex").first).to_be_visible(timeout=TIMEOUT)
    _wait_printed(popup)
    assert _switch(page, hub.base_url) == "off"


# ---------------------------------------------------------------------------
# Plotly, notebook and PDF still render, follow the theme and print
# ---------------------------------------------------------------------------

_PLOT_PROBE = """() => {
  const css = getComputedStyle(document.documentElement);
  const a = document.getElementById('plot-a');
  const fa = a && a._fullLayout;
  return {
    theme: document.documentElement.getAttribute('data-theme'),
    paperToken: css.getPropertyValue('--chart-paper-bg').trim(),
    textToken: css.getPropertyValue('--chart-axis-text').trim(),
    paper: fa && fa.paper_bgcolor,
    font: fa && fa.font.color,
  };
}"""


def _plot_follows(frame: Frame | Page, theme: str) -> None:
    frame.wait_for_function(
        """(theme) => {
          const a = document.getElementById('plot-a');
          const css = getComputedStyle(document.documentElement);
          return document.documentElement.getAttribute('data-theme') === theme
            && a && a._fullLayout && a.style.visibility === 'visible'
            && a._fullLayout.paper_bgcolor === css.getPropertyValue('--chart-paper-bg').trim();
        }""",
        arg=theme,
        timeout=TIMEOUT,
    )
    probe = frame.evaluate(_PLOT_PROBE)
    assert probe["paperToken"], "tokens.css did not resolve through the proxy"
    assert probe["paper"] == probe["paperToken"]
    assert probe["font"] == probe["textToken"]


def set_gallery_theme(page: Page, theme: str) -> None:
    """Change the gallery's theme through the module instance it runs."""
    page.evaluate(
        """async (theme) => {
          const entry = performance.getEntriesByType('resource')
            .map((e) => e.name).find((n) => n.endsWith('/js/theme-manager.js'));
          const mod = await import(entry);
          mod.setTheme(theme);
        }""",
        theme,
    )


def test_plotly_artifact_renders_follows_theme_and_prints(hub, chromium_browser):
    page = chromium_browser.new_page()
    _stub_print(page)
    open_gallery(page, hub.base_url, theme="dark")
    select_artifact(page, PLOT_TITLE)

    frame = artifact_frame(page, hub.ids[PLOT_TITLE])
    assert frame.evaluate("() => String(self.origin)") == "null"
    _plot_follows(frame, "dark")

    # A live theme change in the gallery reaches the opaque frame.
    set_gallery_theme(page, "retro-light")
    _plot_follows(frame, "retro-light")

    popup = _print_popup(page)
    _plot_follows(popup, "light")
    _wait_printed(popup)


def test_notebook_renders_and_prints(hub, chromium_browser):
    page = chromium_browser.new_page()
    _stub_print(page)
    open_gallery(page, hub.base_url)
    select_artifact(page, NOTEBOOK_TITLE)

    frame = artifact_frame(page, hub.ids[NOTEBOOK_TITLE])
    assert frame.evaluate("() => String(self.origin)") == "null"
    expect(frame.locator("body")).to_contain_text(NOTEBOOK_TOKEN, timeout=TIMEOUT)
    expect(frame.locator("body")).to_contain_text("Tune scan", timeout=TIMEOUT)

    popup = _print_popup(page)
    expect(popup.locator("body")).to_contain_text(NOTEBOOK_TOKEN, timeout=TIMEOUT)
    _wait_printed(popup)


def test_pdf_renders_unsandboxed_and_prints(hub, chromium_browser):
    """A PDF is on the inert list: served without the sandbox (Chrome's viewer
    does not render under one), framed without one, printed directly."""
    page = chromium_browser.new_page()
    _stub_print(page)
    open_gallery(page, hub.base_url)

    pdf_id = hub.ids[PDF_TITLE]
    with page.expect_response(lambda r: f"/files/{pdf_id}/" in r.url, timeout=TIMEOUT) as info:
        select_artifact(page, PDF_TITLE)
    response = info.value
    assert response.status == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "content-security-policy" not in response.headers
    assert response.headers.get("x-content-type-options") == "nosniff"

    iframe = page.locator("#preview-content .preview-viewport iframe")
    expect(iframe).to_have_count(1, timeout=TIMEOUT)
    assert iframe.get_attribute("sandbox") is None

    # Inert: printed on the direct path — a same-origin window on the file's
    # own URL, pinned light. Headless Chromium has no PDF viewer and takes the
    # document as a download, so the evidence is the window and its request.
    with (
        page.context.expect_event(
            "request",
            lambda r: f"/files/{pdf_id}/" in r.url and "theme=light" in r.url,
            timeout=TIMEOUT,
        ) as request_info,
        page.expect_popup(timeout=TIMEOUT),
    ):
        page.locator("#preview-content .print-action-btn").click()
    assert GALLERY in request_info.value.url
