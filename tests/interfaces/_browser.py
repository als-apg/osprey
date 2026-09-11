"""General-purpose "did this page load clean" assertion for Playwright suites.

``assert_page_loads_clean`` is the shared negative-space check that browser
suites reach for after navigating to a page: no uncaught JS exceptions, and no
same-origin script/stylesheet subresource came back 4xx/5xx (the classic
"module path typo'd, page silently half-loads" failure mode). It is
intentionally narrow by default — see the module-breakage rationale below —
with an opt-in ``console.error`` arm and an ``allowlist`` escape hatch for
known-benign noise.

``wait_for_dock_settled`` is the other shared wait: it blocks until the dock has
finished arranging the shell, for any suite that drives or photographs the
web-terminal hub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.sync_api import Page

# Bounded settle wait after `load` fires. Some interfaces (e.g. the web
# terminal's SSE-backed panels) hold long-lived connections, so `networkidle`
# may never fire; this is best-effort settling, not a correctness gate.
_NETWORK_IDLE_TIMEOUT_MS = 5000

_SCRIPT_STYLESHEET_RESOURCE_TYPES = ("script", "stylesheet")
_SCRIPT_STYLESHEET_EXTENSIONS = (".js", ".css")


def _same_origin(url: str, origin: tuple[str, str | None, int | None]) -> bool:
    parsed = urlparse(url)
    return (parsed.scheme, parsed.hostname, parsed.port) == origin


def _is_script_or_stylesheet(response) -> bool:  # noqa: ANN001 - Playwright Response
    request = getattr(response, "request", None)
    resource_type = getattr(request, "resource_type", None) if request is not None else None
    if resource_type in _SCRIPT_STYLESHEET_RESOURCE_TYPES:
        return True
    path = urlparse(response.url).path
    return path.endswith(_SCRIPT_STYLESHEET_EXTENSIONS)


def assert_page_loads_clean(
    page: Page,
    url: str,
    *,
    capture_console: bool = False,
    allowlist: Callable[[str, str], bool] | None = None,
) -> None:
    """Navigate to ``url`` and assert the page loaded without breakage.

    Gates on two signals by default:

    - ``pageerror``: any uncaught JS exception thrown while the page ran.
    - failed same-origin subresource responses: any same-origin response with
      ``status >= 400`` whose request is a script or stylesheet (matched via
      Playwright's ``resource_type`` or a ``.js``/``.css`` URL path suffix).
      This is the reliable, low-noise module-breakage detector — a 404 on a
      ``<script type="module" src="...">`` means the page silently didn't run
      the code it depends on.

    Args:
        page: An open Playwright ``Page`` (not yet navigated to ``url``;
            listeners are attached before navigation so nothing is missed).
        url: The URL to navigate to.
        capture_console: If True, also gate on ``console.error`` messages.
            Off by default: in Chromium, a failed subresource load *also*
            emits a browser-generated console error ("Failed to load
            resource: the server responded with a status of 404 ()") that is
            indistinguishable from a genuine ``console.error()`` call from
            page script. Gating on it by default would fail a page for any
            backend hiccup already caught by the response-arm above, and
            would double-count (or mask, depending on ordering) the very
            ``dom.js`` 404 the guard test relies on. Enable this arm only
            when a suite wants to catch app code that explicitly logs via
            ``console.error`` without also throwing.
        allowlist: Optional predicate ``(kind, detail) -> bool`` called for
            every collected item; return True to suppress it as known-benign.
            ``kind`` is one of ``"pageerror"``, ``"response"``, ``"console"``.
            ``detail`` is the exception message / response URL / console
            text, respectively. Default (None) suppresses nothing.

    Raises:
        AssertionError: If any non-allowlisted item survived, listing each
            surviving item (kind + detail) for diagnosability.
    """
    origin = (urlparse(url).scheme, urlparse(url).hostname, urlparse(url).port)
    collected: list[tuple[str, str]] = []

    def _on_pageerror(error) -> None:  # noqa: ANN001 - Playwright error arg
        collected.append(("pageerror", str(error)))

    def _on_response(response) -> None:  # noqa: ANN001 - Playwright Response
        try:
            if response.status < 400:
                return
            if not _same_origin(response.url, origin):
                return
            if not _is_script_or_stylesheet(response):
                return
        except Exception:  # pragma: no cover - defensive against odd responses
            return
        collected.append(("response", response.url))

    def _on_console(msg) -> None:  # noqa: ANN001 - Playwright ConsoleMessage
        if getattr(msg, "type", None) == "error":
            collected.append(("console", msg.text))

    page.on("pageerror", _on_pageerror)
    page.on("response", _on_response)
    if capture_console:
        page.on("console", _on_console)

    page.goto(url, wait_until="load")
    try:
        page.wait_for_load_state("networkidle", timeout=_NETWORK_IDLE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass  # best-effort settle only — see module docstring

    if allowlist is not None:
        surviving = [item for item in collected if not allowlist(item[0], item[1])]
    else:
        surviving = collected

    if surviving:
        lines = "\n".join(f"  - [{kind}] {detail}" for kind, detail in surviving)
        raise AssertionError(f"Page did not load clean at {url}:\n{lines}")


#: Resolves once the dock has finished arranging the shell: the boot layout is
#: announced final AND the console has held its place for a frame after it. The
#: announcement lands in the same frame as the last re-parent, so the flag alone
#: is the boundary rather than the far side of it.
DOCK_SETTLED_JS = """
async (budgetMs) => {
  const dock = await import('/static/js/dock-workspace.js');
  const place = () => {
    let out = '';
    for (let n = document.querySelector('#operator-container'); n; n = n.parentElement) {
      out += '>' + (n.id || n.className || n.tagName);
    }
    return out;
  };
  const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
  const deadline = Date.now() + budgetMs;
  let previous = null;
  while (Date.now() < deadline) {
    await frame();
    const current = place();
    if (dock.bootLayoutSettled() && current === previous) return true;
    previous = current;
  }
  return false;
}
"""


def wait_for_dock_settled(page: Page, *, budget_ms: int = 10_000) -> None:
    """Block until the dock has finished arranging the shell.

    The terminal card carrying both ``#terminal-container`` and
    ``#operator-container`` is server-rendered into ``#dock-panel-sources`` and
    re-parented twice on the way to its tile: once when the dock adopts the
    subtree, and again when the boot layout is applied over the default
    arrangement. The console mounts independently of either, so its textarea is
    on screen and actionable before the shell is still.

    Typing into it there is lost rather than delayed. Playwright checks
    actionability and then writes, and a re-parent between those two steps takes
    focus off the element, so the text is never inserted: the box reads empty,
    ``submit`` finds no prompt and returns, and the turn never starts. The
    equivalent hazard for a keystroke is the event landing on whatever the
    detached node left behind.

    Both moves are boot-only, so waiting for them here puts every re-parent
    ahead of every interaction rather than between two of them.

    Call this AFTER the page's own mount check, never before. The wait reads the
    dock's state by importing its module, and a caller that has not yet waited
    for anything makes that a cold fetch of a module graph still being loaded --
    which fails outright rather than waiting, and reports a fetch error in place
    of whatever the test was doing.
    """
    assert page.evaluate(DOCK_SETTLED_JS, budget_ms), (
        "the dock never settled: the boot layout was not announced final, or the "
        "console was still being re-parented when the wait ran out"
    )
