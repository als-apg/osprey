"""General-purpose "did this page load clean" assertion for Playwright suites.

``assert_page_loads_clean`` is the shared negative-space check that browser
suites reach for after navigating to a page: no uncaught JS exceptions, and no
same-origin script/stylesheet subresource came back 4xx/5xx (the classic
"module path typo'd, page silently half-loads" failure mode). It is
intentionally narrow by default — see the module-breakage rationale below —
with an opt-in ``console.error`` arm and an ``allowlist`` escape hatch for
known-benign noise.
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
