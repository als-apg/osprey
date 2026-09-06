"""``GET /panel/{id}/terminal-static/{asset}`` — the hub's own static tree, reachable from a panel page.

The notebook panel is JupyterLab, served through the terminal's proxy under
``/panel/jupyter/``. The browser's origin is therefore the hub, but every path
the Lab page emits is relative to JupyterLab's own ``base_url``, which is that
same panel prefix — so a ``/static/js/...`` reference from the Lab page resolves
into the *sidecar's* namespace, where the hub's chip modules do not exist. This
route is the address that lands back on the hub's files from inside a panel's
path, and task 8.2's injected ``<script type="module">`` is its only caller.

Four properties are pinned here, because each one is a way the route could ship
looking correct and be wrong:

* **Containment.** ``asset_path`` is whatever the browser put after the prefix.
  A traversal, an absolute path, a directory and a symlink out of the tree must
  all answer 404 — and the check has to live before the read, not after it.
* **Verbatim bodies.** Unlike its design-system neighbour, this route must NOT
  run ``_rewrite_content``. The chip reaches the hub's API root-absolutely
  through ``withPrefix()``; rewriting ``'/api/...'`` into ``'/panel/jupyter/api/
  ...'`` would aim its reads at the sidecar, which answers none of them.
* **Content types.** A module served as ``text/plain`` is a module the browser
  refuses to execute, and nothing in the JS would report why.
* **Tier.** Exactly the design-system route's, with ``PANEL_TIER_ROUTES``
  untouched.

The suite fetches real files — the chip's actual relative import closure — so a
rename that breaks 8.2's import graph fails here rather than in a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces import web_auth
from osprey.interfaces.web_terminal.routes.proxy import (
    _TERMINAL_STATIC_DIR,
    _TERMINAL_STATIC_SUFFIXES,
    _contained_asset,
    router,
)

#: A panel id in the route's own namespace. The route never reads it — the tree
#: it serves is the hub's, not the panel's — but every URL carries one.
PANEL_ID = "jupyter"

#: The chip's relative import closure, as 3.5's handoff pins it, plus the two
#: modules the popover pulls in. Every one of these has to be fetchable through
#: this route or the Lab bar's module graph breaks at the first import.
CHIP_MODULES = (
    "js/control-target-chip.js",
    "js/control-target-popover.js",
    "js/control-target-facts.js",
    "js/api.js",
    "js/activity-format.js",
    "js/confirm-skip.js",
    "js/posture-confirm.js",
    "js/modal-overlay.js",
)


@pytest.fixture
def client() -> TestClient:
    """The proxy router alone.

    No ``app.state``: this route reads none of it, and an app without the state
    the *catch-all* proxy needs is a useful second assertion — a request that
    fell through to ``proxy_panel`` would fail loudly rather than quietly
    passing through to a backend.
    """
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as started:
        yield started


def _url(asset: str) -> str:
    return f"/panel/{PANEL_ID}/terminal-static/{asset}"


# ---- it serves the real files ---- #


@pytest.mark.parametrize("asset", CHIP_MODULES)
def test_every_chip_module_is_reachable(client: TestClient, asset: str) -> None:
    """The whole import closure answers, byte-for-byte."""
    response = client.get(_url(asset))

    assert response.status_code == 200
    assert response.content == (_TERMINAL_STATIC_DIR / asset).read_bytes()


def test_a_module_is_served_as_javascript(client: TestClient) -> None:
    """A module served as anything else is a module the browser will not run."""
    response = client.get(_url("js/control-target-chip.js"))

    assert response.headers["content-type"].startswith("text/javascript")


def test_the_body_is_not_rewritten(client: TestClient) -> None:
    """The load-bearing difference from the design-system route.

    ``api.js`` carries the literal ``'/api/session'``. The proxy's rewrite turns
    quote-delimited ``/api/`` into ``/panel/<id>/api/``, which on the Lab page
    would send the chip's reads to JupyterLab. The rewrite is right for the
    design system, whose assets *are* the panel's; it is wrong here.
    """
    response = client.get(_url("js/api.js"))

    assert "'/api/session'" in response.text
    assert f"/panel/{PANEL_ID}/api/" not in response.text


def test_the_stylesheet_the_chip_needs_is_reachable(client: TestClient) -> None:
    """``.ctc-anchor`` must be ``position: relative`` or the popover floats free.

    3.5's handoff names ``terminal.css`` as where that rule already lives, so
    the Lab page loads it through this same route.
    """
    response = client.get(_url("css/terminal.css"))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert ".ctc-anchor" in response.text


def test_assets_are_served_uncached(client: TestClient) -> None:
    """Filenames here are unversioned, so a redeploy must reach an open page."""
    response = client.get(_url("js/control-target-chip.js"))

    assert "no-store" in response.headers["cache-control"]


def test_head_answers_without_a_body(client: TestClient) -> None:
    """The Lab page's loader may probe before it imports."""
    response = client.head(_url("js/control-target-chip.js"))

    assert response.status_code == 200
    assert response.content == b""


# ---- containment ---- #


def test_an_unknown_asset_is_not_found(client: TestClient) -> None:
    response = client.get(_url("js/no-such-module.js"))

    assert response.status_code == 404


def test_a_directory_is_not_a_listing(client: TestClient) -> None:
    """``is_file()`` is what stands between this route and an index of the tree."""
    response = client.get(_url("js"))

    assert response.status_code == 404


#: Traversals that name a file which REALLY EXISTS outside the static root.
#: A traversal aimed at a missing file 404s whether or not the route contains
#: anything, and would pass this suite while proving nothing — so each target is
#: asserted present before the refusal is asserted.
ESCAPING_ASSETS = (
    pytest.param("%2e%2e/app.py", "../app.py", id="encoded-traversal"),
    pytest.param(
        "js/%2e%2e/%2e%2e/jupyter_sidecar.py",
        "js/../../jupyter_sidecar.py",
        id="encoded-traversal-mid-path",
    ),
    pytest.param("%2e%2e/%2e%2e/web_auth.py", "../../web_auth.py", id="encoded-traversal-two-up"),
)


@pytest.mark.parametrize(("asset", "relative"), ESCAPING_ASSETS)
def test_a_traversal_to_a_real_file_is_refused(
    client: TestClient, asset: str, relative: str
) -> None:
    """Percent-encoded dots arrive decoded, so the route sees a real ``..``.

    The ASGI ``path`` is already percent-decoded by the time Starlette matches,
    which is exactly why the route cannot leave normalisation to the router.
    """
    escape_target = (_TERMINAL_STATIC_DIR / relative).resolve()
    assert escape_target.is_file(), (
        f"{relative} no longer names a real file outside the static root, so this "
        "case would pass without the route containing anything — repoint it"
    )

    response = client.get(_url(asset))

    assert response.status_code == 404


@pytest.mark.parametrize(
    "asset",
    [
        pytest.param("js/%00", id="bare-nul"),
        pytest.param("js/api.js%00.txt", id="nul-truncation"),
    ],
)
def test_a_nul_byte_in_the_path_is_refused(client: TestClient, asset: str) -> None:
    """A path the OS cannot look at is a 404, not a 500.

    ASGI delivers ``%00`` decoded, and ``Path.resolve()`` raises ``ValueError``
    out of ``lstat`` on it — before any containment check would run. The helper
    answers that the same way it answers every other not-a-file-under-the-root
    path, so both panel-scoped static routes are covered by this one fix; the
    design-system route 500'd on the same input before it.
    """
    response = client.get(_url(asset))

    assert response.status_code == 404


# ---- the suffix allow-list ---- #


def test_the_page_sources_are_not_served(client: TestClient) -> None:
    """This route serves a module graph, not documents.

    ``index.html`` and ``session.html`` are raw Jinja sources sitting in the
    same tree. Serving them here is not an exposure — ``/static/index.html``
    hands the identical bytes to an unauthenticated caller, since ``/static`` is
    an exempt mount — but it is reach this route has no use for.
    """
    for page in ("index.html", "session.html"):
        assert (_TERMINAL_STATIC_DIR / page).is_file(), f"{page} moved; repoint this test"
        assert client.get(_url(page)).status_code == 404


def test_the_allow_list_covers_every_asset_a_module_graph_needs(client: TestClient) -> None:
    """The suffixes 8.2 can actually reach for, pinned as a set.

    Held here rather than imported so a widening of the route's own frozenset
    fails this file instead of agreeing with itself.
    """
    assert _TERMINAL_STATIC_SUFFIXES == {".js", ".mjs", ".css", ".map", ".svg", ".woff2", ".png"}


def test_the_design_system_route_still_serves_html(client: TestClient) -> None:
    """The allow-list is this route's alone.

    The design-system route serves whole pages on purpose — ``theme-lab.html``
    is the one ``osprey theme-lab`` opens — so narrowing the shared helper
    instead of this one route would have broken it.
    """
    response = client.get(f"/panel/{PANEL_ID}/design-system/theme-lab.html")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_an_absolute_path_is_refused(client: TestClient) -> None:
    """``Path("/root") / "/etc/passwd"`` is ``/etc/passwd`` — the root is discarded."""
    assert Path("/etc/passwd").is_file()

    response = client.get(_url("/etc/passwd"))

    assert response.status_code == 404


def test_the_helper_refuses_a_symlink_out_of_the_tree(tmp_path: Path) -> None:
    """``resolve()`` follows the link, so the escape shows up in the comparison.

    Driven through the helper rather than the route because planting a symlink
    in the packaged static tree would be a change to shipped files.
    """
    root = tmp_path / "static"
    (root / "js").mkdir(parents=True)
    secret = tmp_path / "outside" / "secret.js"
    secret.parent.mkdir()
    secret.write_text("// not yours\n")
    (root / "js" / "escape.js").symlink_to(secret)
    (root / "js" / "real.js").write_text("// yours\n")

    assert _contained_asset(root, "js/escape.js") is None
    assert _contained_asset(root, "js/real.js") == (root / "js" / "real.js").resolve()


def test_the_helper_refuses_traversal_and_absolute_paths(tmp_path: Path) -> None:
    """The three shapes the route depends on it refusing, at the unit."""
    root = tmp_path / "static"
    root.mkdir()
    (root / "ok.js").write_text("// yours\n")
    (tmp_path / "secret.js").write_text("// not yours\n")

    assert _contained_asset(root, "../secret.js") is None
    assert _contained_asset(root, str(tmp_path / "secret.js")) is None
    assert _contained_asset(root, "") is None
    assert _contained_asset(root, "ok.js") == (root / "ok.js").resolve()


# ---- the tier ---- #


def test_the_route_sits_at_the_design_system_route_s_tier() -> None:
    """Same tier as its neighbour, and the panel tier is untouched.

    The panel token is the weaker credential handed to in-process companions.
    Serving the terminal's own modules is not something it should unlock, and
    the Lab page importing them is a page the operator is already logged in to.
    """
    terminal_static = "/panel/jupyter/terminal-static/js/control-target-chip.js"
    design_system = "/panel/jupyter/design-system/css/tokens.css"

    assert web_auth.classify("GET", terminal_static, False) == web_auth.classify(
        "GET", design_system, False
    )
    assert web_auth.classify("GET", terminal_static, False) is web_auth.Tier.OPERATOR
    assert not any(path.startswith("/panel/") for _, path in web_auth.PANEL_TIER_ROUTES)


def test_the_route_is_declared_above_the_catch_all() -> None:
    """Starlette matches in declaration order; below ``proxy_panel`` it is dead.

    Asserted on the router's own route list rather than by fetching, because a
    fetch that reached the catch-all would fail for its own reasons and never
    name the cause.
    """
    paths = [getattr(route, "path", "") for route in router.routes]
    static_route = paths.index("/panel/{panel_id}/terminal-static/{asset_path:path}")
    catch_all = paths.index("/panel/{panel_id}/{path:path}")

    assert static_route < catch_all
