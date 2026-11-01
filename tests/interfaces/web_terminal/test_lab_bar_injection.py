"""What the panel proxy injects into the notebook panel's pages, and where.

The end-to-end half of this lives in ``test_proxy_jupyter_integration.py``,
against a real ``jupyter_server`` behind the proxy — but that module spawns a
process, so it is marked slow and runs in some lanes only. Everything here is a
pure function over strings, needs no sidecar, and answers the two questions a
slow lane is the wrong place to ask:

- **which pages get the bar.** A content type is not a page. jupyter-server
  serves a user's own ``.html`` file as ``text/html``, so the file viewer, the
  ``view/`` wrapper and an ``nbconvert`` export all look like the Lab page to a
  gate that only reads headers. The bar in one of those would be a second chip
  in a viewer frame, a second event stream against the browser's connection
  cap, and the hub's stylesheet cascading into a document somebody else wrote.
- **what the tags look like under a per-user mount.** Every integration fixture
  runs with ``compute_url_prefix() == ""``, so the ``/u/<user>`` shape of the
  scope key, its target and the module ``src`` — the whole reason the import
  map exists — is only ever produced here.
"""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock

import pytest

from osprey.interfaces.common_middleware import EXTERNAL_ORIGIN_ENV
from osprey.interfaces.web_terminal.routes.proxy import (
    _PANEL_STATE_MAP,
    _control_target_bar_markup,
    _inject_control_target_bar,
    _lab_page_csp,
)

#: A minimal document with the one thing the injection needs.
PAGE = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>x</title></head><body></body></html>'

#: The panel the bar belongs to, and the module it loads.
JUPYTER = "jupyter"
MODULE = "control-target-lab-bar.js"


def inject(path: str, *, panel: str = JUPYTER, text: str = PAGE, prefix: str = "") -> str:
    return _inject_control_target_bar(text, panel, path, "text/html", prefix)


# ---------------------------------------------------------------------------
# Which pages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "lab",
        "lab/",
        "lab/tree/notebook.ipynb",
        "lab/workspaces/auto-x",
        "tree",
        "tree/subdir",
        "notebooks/notebook.ipynb",
        "doc/tree/notebook.ipynb",
        "consoles/1",
        "edit/script.py",
    ],
)
def test_jupyterlabs_own_pages_carry_the_bar(path: str) -> None:
    """Every page an operator drives the machine from."""
    assert MODULE in inject(path)


@pytest.mark.parametrize(
    "path",
    [
        "files/report.html",
        "files/nested/dir/report.html",
        "view/files/report.html",
        "nbconvert/html/notebook.ipynb",
        "login",
        "logout",
        "static/lab/index.html",
        # Under lab/ but not Lab's application page: the prebuilt-extension
        # tree serves any file type, and the API serves theme assets.
        "lab/extensions/x/static/a.html",
        "lab/api/themes/x/index.html",
        "lab/api/settings",
        "lab/workspaces",
        "doc/extensions/x.html",
    ],
)
def test_a_document_the_sidecar_serves_for_someone_else_does_not(path: str) -> None:
    """The user's own HTML, and the pages that wrap it, are relayed untouched.

    This is the finding the panel-id gate alone did not cover: all of these
    answer ``text/html`` with a ``<head>``, exactly like the Lab page.
    """
    assert inject(path) == PAGE


def test_the_page_list_is_an_allow_list() -> None:
    """A page nobody has thought of yet is out, not in."""
    assert inject("some-future-extension/page") == PAGE
    assert inject("") == PAGE


# ---------------------------------------------------------------------------
# Which panel, which body
# ---------------------------------------------------------------------------


def test_no_other_panels_html_is_touched() -> None:
    """The path list is JupyterLab's; another panel's `lab` page is still its own."""
    assert inject("lab", panel="artifacts") == PAGE
    assert inject("lab", panel="ariel") == PAGE


def test_a_body_that_is_not_html_is_untouched() -> None:
    payload = '{"status": "ok"}'

    assert _inject_control_target_bar(payload, JUPYTER, "lab", "application/json", "") == payload


def test_a_document_without_a_head_is_relayed_untouched() -> None:
    """An HTML fragment is not a document, and offers no injection point."""
    fragment = "<div class='panel'>no head here</div>"

    assert inject("lab", text=fragment) == fragment


def test_the_tags_go_in_at_the_top_of_the_head() -> None:
    """Before the page's own head content, because the module graph reads them."""
    injected = inject("lab")
    head_open = injected.index("<head>") + len("<head>")

    assert injected[head_open:].startswith("<script>window.__OSPREY_PREFIX__")
    assert injected.index(MODULE) < injected.index("<meta charset")
    assert injected.count(MODULE) == 1


# ---------------------------------------------------------------------------
# The per-user mount
# ---------------------------------------------------------------------------


def test_every_tag_carries_the_per_user_mount_prefix() -> None:
    """The ``/u/<user>`` shape, which no integration fixture ever produces.

    A doubled slash, or a scope key that stops matching the module's own
    directory, would pass every empty-prefix test and break only on a deployed
    multi-user stack — where the bar's module graph 404s in silence.
    """
    markup = _control_target_bar_markup("/u/alice/panel/jupyter", "/u/alice")

    assert 'window.__OSPREY_PREFIX__ = "/u/alice";' in markup
    assert f'src="/u/alice/panel/jupyter/terminal-static/js/{MODULE}"' in markup

    import_map = json.loads(markup.split('<script type="importmap">')[1].split("</script>")[0])
    scope = "/u/alice/panel/jupyter/terminal-static/js/"
    assert import_map == {"scopes": {scope: {"/design-system/": "/u/alice/design-system/"}}}

    # The scope is the directory the module is served from: a scope that did
    # not match it would leave the root-absolute import unmapped.
    assert f'src="{scope}{MODULE}"' in markup


def test_an_empty_prefix_maps_the_design_system_onto_itself() -> None:
    """Single-origin deployments get the identity map, the hub's own behaviour."""
    markup = _control_target_bar_markup("/panel/jupyter", "")

    assert 'window.__OSPREY_PREFIX__ = "";' in markup
    import_map = json.loads(markup.split('<script type="importmap">')[1].split("</script>")[0])
    assert import_map["scopes"]["/panel/jupyter/terminal-static/js/"] == {
        "/design-system/": "/design-system/"
    }


def test_a_prefix_cannot_close_the_tags_it_is_written_into() -> None:
    """Both script tags are built from the same string, so both escape it.

    The prefix is deployment configuration rather than operator input, so this
    is depth — but the escape is ``\\u003c``, which is valid JSON, so the map
    still parses and the global still reads back as the literal prefix.
    """
    hostile = "/u/</script><script>alert(1)</script>"
    markup = _control_target_bar_markup(f"{hostile}/panel/jupyter", hostile)

    assert "</script><script>alert(1)" not in markup
    assert markup.count("</script>") == 3

    raw = markup.split('<script type="importmap">')[1].split("</script>")[0]
    scopes = json.loads(raw)["scopes"]
    assert next(iter(scopes.values())) == {"/design-system/": f"{hostile}/design-system/"}

    global_json = markup.split("window.__OSPREY_PREFIX__ = ")[1].split(";</script>")[0]
    assert json.loads(global_json) == hostile


# ---------------------------------------------------------------------------
# The Lab page's script policy (q-0001)
# ---------------------------------------------------------------------------
#
# JupyterLab renders kernel output inside its own application document, on the
# console's origin, with the operator's cookie. The proxy is the one place that
# can decide what that document may execute: a per-response nonce on the
# parser-inserted tags it controls, Lab's own static trees and the hub's
# design system by fully qualified path, and nothing else — no 'self' (the
# origin also serves agent-writable files), no 'unsafe-inline'. 'unsafe-eval'
# is in as the q-0001 condition-3 fallback (Lab's settings registry compiles
# schemas with new Function); no output renderer that evaluates remains on.
# Every other proxied response keeps the policy its upstream
# wrote, above all jupyter-server's ``sandbox`` on ``files/``.

#: JupyterLab's own application page, shaped like ``jupyterlab/static/index.html``:
#: a data block, the deferred main bundle, and the inline token-stripper.
LAB_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>JupyterLab</title>'
    '<script id="jupyter-config-data" type="application/json">{"baseUrl": "/panel/jupyter/"}</script>'
    '<script defer="defer" src="/panel/jupyter/static/lab/main.abc.js?v=abc"></script>'
    '</head><body class="jp-ThemedContainer"><script>/* Remove token from URL. */'
    "(function () {})();</script></body></html>"
)

#: Jupyter-server's own policy, as it arrives on every Lab page today.
JUPYTER_CSP = "frame-ancestors 'self'; report-uri /panel/jupyter/api/security/csp-report"

#: What jupyter-server adds for a user's own file: it must reach the browser unchanged.
FILES_CSP = f"{JUPYTER_CSP}; sandbox allow-scripts"

_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>]*>", re.IGNORECASE)


def _directives(policy: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in policy.split(";"):
        tokens = part.split()
        if tokens:
            assert tokens[0] not in out, f"directive {tokens[0]} declared twice"
            out[tokens[0]] = tokens[1:]
    return out


def _nonce_of(policy: str) -> str:
    sources = _directives(policy)["script-src"]
    nonces = [s for s in sources if s.startswith("'nonce-")]
    assert len(nonces) == 1, sources
    return nonces[0][len("'nonce-") : -1]


def test_the_policy_names_lab_and_the_design_system_by_fully_qualified_path() -> None:
    """Forwarded host plus a per-user prefix: every source carries both."""
    policy = _lab_page_csp("N0nce", "https://osprey.example.org", "/u/alice")
    d = _directives(policy)

    base = "https://osprey.example.org/u/alice"
    assert d["script-src"] == [
        "'nonce-N0nce'",
        "'unsafe-eval'",
        f"{base}/panel/jupyter/static/",
        f"{base}/panel/jupyter/lab/extensions/",
        f"{base}/panel/jupyter/terminal-static/",
        f"{base}/design-system/",
    ]
    assert d["object-src"] == ["'none'"]
    assert d["base-uri"] == ["'none'"]
    assert d["frame-ancestors"] == ["'self'"]
    assert d["form-action"] == ["'none'"]


def test_the_policy_never_trusts_the_origin_or_inline_script() -> None:
    """``'unsafe-eval'`` is the q-0001 fallback, and the only keyword besides the nonce.

    JupyterLab's settings registry compiles every plugin schema with Ajv
    (``new Function``); without eval Lab boots unthemed and unconfigured. It
    lets running script evaluate a string; it admits no new script.
    """
    policy = _lab_page_csp("n", "http://localhost:8087", "")
    sources = _directives(policy)["script-src"]
    for banned in ("'self'", "'unsafe-inline'", "'strict-dynamic'", "*"):
        assert banned not in sources, banned
    assert [s for s in sources if s.startswith("'")] == ["'nonce-n'", "'unsafe-eval'"]
    # Every other source is a path inside the origin, never the origin itself.
    for source in sources[2:]:
        assert source.startswith("http://localhost:8087/") and source.endswith("/"), source
        assert source != "http://localhost:8087/", source


def test_every_bar_tag_carries_the_nonce() -> None:
    markup = _control_target_bar_markup("/u/alice/panel/jupyter", "/u/alice", nonce="abc123")
    tags = _SCRIPT_OPEN_RE.findall(markup)

    assert len(tags) == 3
    assert all('nonce="abc123"' in tag for tag in tags), tags


def test_the_bar_markup_without_a_nonce_is_unchanged() -> None:
    """Callers that do not police the page see the same three tags as before."""
    assert "nonce" not in _control_target_bar_markup("/panel/jupyter", "")


# --- through the proxy -------------------------------------------------------


@pytest.fixture
def jupyter_proxy(tmp_path, monkeypatch):
    """The whole app, the notebook panel resolvable, its upstream faked.

    Yields ``(client, serve)``: ``serve(body, content_type, csp)`` sets what the
    fake sidecar answers next.
    """
    import httpx

    from osprey.interfaces.web_terminal.routes import proxy as proxy_mod

    from ._proxy_fakes import panel_app

    ws = tmp_path / "_agent_data"
    ws.mkdir()
    monkeypatch.delenv(EXTERNAL_ORIGIN_ENV, raising=False)
    monkeypatch.setattr(proxy_mod, "compute_url_prefix", lambda: "/u/alice")

    answer: dict = {}

    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        return httpx.Response(
            status_code=200,
            text=answer["body"],
            headers={
                "content-type": answer["type"],
                "content-security-policy": answer["csp"],
                **answer["extra"],
            },
        )

    def serve(
        body: str,
        content_type: str = "text/html; charset=UTF-8",
        csp: str = JUPYTER_CSP,
        extra: dict[str, str] | None = None,
    ):
        answer.update(body=body, type=content_type, csp=csp, extra=extra or {})

    for app, client in panel_app(ws, []):
        setattr(app.state, _PANEL_STATE_MAP[JUPYTER], "http://127.0.0.1:8899/u/alice/panel/jupyter")
        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
        yield client, serve


def test_a_lab_page_through_the_proxy_carries_the_policy_and_every_script_its_nonce(
    jupyter_proxy,
) -> None:
    client, serve = jupyter_proxy
    serve(LAB_PAGE)

    resp = client.get("/panel/jupyter/lab", headers={"host": "osprey.example.org"})

    assert resp.status_code == 200
    policies = resp.headers.get_list("content-security-policy")
    assert len(policies) == 1, policies  # upstream's replaced, not appended to
    policy = policies[0]
    assert "report-uri" not in policy

    d = _directives(policy)
    assert "http://osprey.example.org/u/alice/panel/jupyter/static/" in d["script-src"]
    assert "http://osprey.example.org/u/alice/design-system/" in d["script-src"]

    nonce = _nonce_of(policy)
    tags = _SCRIPT_OPEN_RE.findall(resp.text)
    # Lab's data block, main bundle and token-stripper, plus the bar's three.
    assert len(tags) == 6, tags
    for tag in tags:
        assert f'nonce="{nonce}"' in tag, tag
        assert tag.count("nonce=") == 1, tag
    assert MODULE in resp.text


def test_every_response_gets_its_own_nonce(jupyter_proxy) -> None:
    client, serve = jupyter_proxy
    serve(LAB_PAGE)

    first = _nonce_of(client.get("/panel/jupyter/lab").headers["content-security-policy"])
    second = _nonce_of(client.get("/panel/jupyter/lab").headers["content-security-policy"])

    assert first != second
    assert len(first) >= 16


def test_the_configured_external_origin_wins_over_the_host_header(
    jupyter_proxy, monkeypatch
) -> None:
    """The origin the Origin check expects, which behind nginx is not ``Host``."""
    client, serve = jupyter_proxy
    serve(LAB_PAGE)
    monkeypatch.setenv(EXTERNAL_ORIGIN_ENV, " https://Osprey.Example.org:443 ")

    policy = client.get("/panel/jupyter/lab", headers={"host": "terminal-alice:8087"}).headers[
        "content-security-policy"
    ]

    sources = _directives(policy)["script-src"]
    assert "https://osprey.example.org/u/alice/panel/jupyter/static/" in sources
    assert not any("terminal-alice" in s for s in sources)


@pytest.mark.parametrize(
    "path",
    [
        "files/report.html",
        "view/files/report.html",
        "login",
        "lab/extensions/x/static/a.html",
        "lab/api/themes/x/index.html",
    ],
)
def test_a_page_that_is_not_labs_keeps_its_upstream_policy_and_gets_no_nonce(
    jupyter_proxy, path: str
) -> None:
    """Above all a user's own HTML file: jupyter-server sandboxes it, and that reaches the browser."""
    client, serve = jupyter_proxy
    serve(LAB_PAGE, csp=FILES_CSP)

    resp = client.get(f"/panel/jupyter/{path}")

    assert resp.headers.get_list("content-security-policy") == [FILES_CSP]
    assert "nonce=" not in resp.text


def test_a_lab_page_is_never_cached_or_revalidated(jupyter_proxy) -> None:
    """A per-response nonce holds only if the page is never served from a cache.

    The upstream's validators describe the un-nonced body, and a caching
    decision of its own would otherwise survive the ``setdefault``.
    """
    client, serve = jupyter_proxy
    serve(
        LAB_PAGE,
        extra={
            "etag": '"abc"',
            "last-modified": "Sat, 24 Oct 2026 12:00:00 GMT",
            "cache-control": "max-age=3600",
        },
    )

    resp = client.get("/panel/jupyter/lab")

    assert resp.headers.get_list("cache-control") == ["no-store"]
    assert "etag" not in resp.headers
    assert "last-modified" not in resp.headers


def test_a_non_html_lab_answer_keeps_its_upstream_policy(jupyter_proxy) -> None:
    client, serve = jupyter_proxy
    serve('{"ok": true}', content_type="application/json")

    resp = client.get("/panel/jupyter/lab/api/settings")

    assert resp.headers["content-security-policy"] == JUPYTER_CSP


def test_labs_javascript_output_renderer_is_switched_off() -> None:
    """Output carrying ``application/javascript`` has no renderer to run it."""
    from osprey.interfaces.web_terminal.jupyter_sidecar import (
        LAB_DISABLED_EXTENSIONS,
        lab_page_config,
    )

    assert "@jupyterlab/javascript-extension" in LAB_DISABLED_EXTENSIONS
    assert lab_page_config()["disabledExtensions"]["@jupyterlab/javascript-extension"] is True


def test_labs_vega_output_renderer_is_switched_off() -> None:
    """Vega compiles spec expressions with ``new Function``; under the policy's
    ``'unsafe-eval'`` a kernel-emitted Vega-Lite output would be an eval sink."""
    from osprey.interfaces.web_terminal.jupyter_sidecar import (
        LAB_DISABLED_EXTENSIONS,
        lab_page_config,
    )

    assert "@jupyterlab/vega5-extension" in LAB_DISABLED_EXTENSIONS
    assert lab_page_config()["disabledExtensions"]["@jupyterlab/vega5-extension"] is True


def test_labs_app_is_published_for_the_jupyter_tiles_theme_follower() -> None:
    """``window.jupyterapp`` is the handle ``lab-theme-follow.js`` drives.

    A boolean is enough: Lab reads the option through ``PageConfig.getOption``,
    which hands every non-string value back JSON-encoded, and compares it with
    ``"true"``.
    """
    from osprey.interfaces.web_terminal.jupyter_sidecar import lab_page_config

    assert lab_page_config()["exposeAppInBrowser"] is True


# --- where Lab loads prebuilt extensions from --------------------------------
#
# ``lab/extensions/`` is in the policy's path list, and jupyter-server serves it
# from ``labextensions_path``, whose default includes ``$JUPYTER_DATA_DIR`` —
# on the shared root, which kernels and agent tooling write. Lab also loads
# every extension it finds there on its own. The sidecar pins the path to the
# image's install tree.


def _drop_extension(data_dir, name: str = "agent-ext") -> None:
    ext = data_dir / "labextensions" / name
    (ext / "static").mkdir(parents=True)
    (ext / "package.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "0.0.1",
                "jupyterlab": {"_build": {"load": "static/remoteEntry.js", "extension": "./ext"}},
            }
        ),
        encoding="utf-8",
    )
    (ext / "static" / "remoteEntry.js").write_text("/* agent */", encoding="utf-8")


def _lab_server_app(argv: list[str]):
    from jupyterlab_server.app import LabServerApp
    from traitlets.config.loader import KVArgParseConfigLoader

    # Parsed the way the sidecar's ServerApp parses it: before the Lab
    # extension loads, so without the class known and the value deferred.
    loader = KVArgParseConfigLoader([a for a in argv if a.startswith("--LabServerApp.")])
    config = loader.load_config()
    return LabServerApp(config=config)


def test_the_sidecar_keeps_the_shared_root_off_the_labextensions_path(tmp_path) -> None:
    import sys
    from pathlib import Path

    from osprey.interfaces.web_terminal.jupyter_sidecar import JupyterSidecar

    shared = tmp_path / "shared"
    argv = JupyterSidecar(shared, "/u/alice", None)._argv()

    app = _lab_server_app(argv)
    searched = [Path(p) for p in app.extra_labextensions_path + app.labextensions_path]
    assert searched == [Path(sys.prefix) / "share" / "jupyter" / "labextensions"]
    assert not any(p.is_relative_to(shared) for p in searched)


def test_an_extension_dropped_on_the_shared_root_is_not_loaded_by_lab(
    tmp_path, monkeypatch
) -> None:
    from jupyterlab_server.config import get_federated_extensions

    from osprey.interfaces.web_terminal.jupyter_sidecar import JupyterSidecar

    sidecar = JupyterSidecar(tmp_path / "shared", "", None)
    _drop_extension(sidecar.data_dir)
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(sidecar.data_dir))
    monkeypatch.delenv("JUPYTER_PATH", raising=False)

    # The vector is real without the pin: Lab's default path lists the drop.
    default = _lab_server_app([])
    assert "agent-ext" in get_federated_extensions(
        default.extra_labextensions_path + default.labextensions_path
    )

    app = _lab_server_app(sidecar._argv())
    assert "agent-ext" not in get_federated_extensions(
        app.extra_labextensions_path + app.labextensions_path
    )
