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

import pytest

from osprey.interfaces.web_terminal.routes.proxy import (
    _control_target_bar_markup,
    _inject_control_target_bar,
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
