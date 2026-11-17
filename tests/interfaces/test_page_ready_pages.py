"""Every page OSPREY serves reaches the page-ready mark by a declared route.

The mark is ``data-page-ready="true"`` on the page's ``<html>``, written by one
module, ``design-system/js/page-ready.js``. A page reaches it by one of three
routes, read here from its source without a browser:

* **the shared start-up** -- the page says nothing about readiness, and a script
  it loads calls ``initTheme()``, which marks it once the document's
  ``DOMContentLoaded`` handlers have run;
* **its own call** -- a script it loads calls ``markPageReady()``. When the
  page also runs the shared start-up, its ``<html>`` declares
  ``data-page-ready="pending"``, or the shared start-up would mark it first. A
  shell page with no script of its own (the window the hub pops a pane into)
  is marked by the script that opens and dresses it, which names its file;
* **the literal attribute** -- a page with no script at all carries
  ``data-page-ready="true"`` in its markup.

A page is every ``.html`` / ``.html.j2`` file under ``src/osprey`` except a
Jinja child (``{% extends %}``: its base answers for it) and a partial another
template ``{% include %}``-s, plus the pages Python builds: the artifact
gallery's injected snippets and its markdown page, and the refusal page a
navigating browser gets. A new page that takes none of the routes fails here.
The rule lives in :mod:`osprey.interfaces.design_system.panels.page_ready`.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from osprey.interfaces import common_middleware
from osprey.interfaces.artifacts import app as artifacts_app
from osprey.interfaces.design_system.panels.page_ready import route_problem

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"
WEB_TERMINAL_STATIC = SRC / "interfaces" / "web_terminal" / "static"

_EXTENDS = re.compile(r"""\{%-?\s*extends\s+["']([^"']+)["']""")
_INCLUDE = re.compile(r"""\{%-?\s*(?:include|import|from)\s+["']([^"']+)["']""")


def _template_roles() -> tuple[set[Path], set[Path]]:
    """The Jinja children and the included partials among the tree's HTML files."""
    children: set[Path] = set()
    partials: set[Path] = set()
    for path in _html_files():
        text = path.read_text(encoding="utf-8")
        if _EXTENDS.search(text):
            children.add(path)
        for name in _INCLUDE.findall(text):
            partials.add((path.parent / name).resolve())
    return children, partials


@cache
def _html_files() -> tuple[Path, ...]:
    files = [*SRC.rglob("*.html"), *SRC.rglob("*.html.j2")]
    return tuple(sorted(p.resolve() for p in files if "vendor" not in p.parts))


def served_pages() -> list[Path]:
    children, partials = _template_roles()
    return [p for p in _html_files() if p not in children and p not in partials]


def _python_pages() -> dict[str, tuple[str, Path]]:
    """The pages Python builds, each with the file its relative addresses resolve from."""
    artifacts = Path(artifacts_app.__file__)
    markdown = artifacts_app._MARKDOWN_PAGE_TEMPLATE.replace("{{", "{").replace("}}", "}")
    return {
        "artifact plot page": (artifacts_app._RESPONSIVE_PLOTLY, artifacts),
        "artifact table / html page": (artifacts_app._RESPONSIVE_TABLE_HTML, artifacts),
        "artifact notebook page": (artifacts_app._NOTEBOOK_SNIPPET, artifacts),
        "artifact markdown page": (markdown, artifacts),
        "refusal page": (common_middleware._refusal_page(401, "no session"), Path(__file__)),
    }


@pytest.mark.parametrize(
    "page", served_pages(), ids=lambda p: p.relative_to(SRC.resolve()).as_posix()
)
def test_every_served_html_page_reaches_the_mark(page: Path) -> None:
    problem = route_problem(page.read_text(encoding="utf-8"), page)
    assert problem is None, f"{page.relative_to(SRC.resolve())}: {problem}"


@pytest.mark.parametrize("name", sorted(_python_pages()))
def test_every_page_python_builds_reaches_the_mark(name: str) -> None:
    page, where = _python_pages()[name]
    problem = route_problem(page, where)
    assert problem is None, f"{name}: {problem}"


def test_the_walk_finds_the_pages_it_must() -> None:
    found = {p.relative_to(SRC.resolve()).as_posix() for p in served_pages()}
    for page in (
        "interfaces/web_terminal/static/index.html",
        "interfaces/web_terminal/static/attached-window.html",
        "history/templates/base.html",
        "services/auth_sidecar/templates/_frame.html",
        "templates/modules/web_terminals/landing.html.j2",
        "apps/scaffold-app/panel/index.html",
    ):
        assert page in found
    assert "history/templates/sessions.html" not in found
    assert "history/templates/_rows.html" not in found


def test_a_page_beside_a_served_page_that_takes_no_route_fails(tmp_path: Path) -> None:
    where = tmp_path / "static" / "x.html"
    assert route_problem("", where) is not None
    assert route_problem("<!DOCTYPE html><html><body></body></html>", where) is not None


def test_each_route_is_read_from_the_markup_and_the_scripts_it_loads() -> None:
    where = WEB_TERMINAL_STATIC / "x.html"
    themed = '<script type="module" src="/design-system/js/theme-manager.js"></script>'
    inline_theme = (
        '<script type="module">import { initTheme } from '
        "'/design-system/js/theme-manager.js'; initTheme();</script>"
    )
    inline_mark = (
        '<script type="module">import { markPageReady } from '
        "'/design-system/js/page-ready.js'; markPageReady();</script>"
    )
    # Loading the module is not running the start-up.
    assert route_problem(f"<html>{themed}</html>", where) is not None
    assert route_problem(f"<html>{inline_theme}</html>", where) is None
    assert route_problem(f"<html>{inline_mark}</html>", where) is None
    assert route_problem('<html data-page-ready="true"></html>', where) is None
    assert route_problem(f'<html data-page-ready="pending">{inline_theme}</html>', where)
    assert route_problem(f'<html data-page-ready="pending">{inline_mark}</html>', where) is None
    assert route_problem(f"<html>{inline_theme}{inline_mark}</html>", where) is not None
    assert route_problem('<html data-page-ready="yes"></html>', where) is not None
    assert route_problem('<html data-page-ready="failed"></html>', where) is not None
