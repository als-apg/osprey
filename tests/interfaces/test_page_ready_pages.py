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
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from osprey.interfaces import common_middleware
from osprey.interfaces.artifacts import app as artifacts_app

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"
DESIGN_SYSTEM_STATIC = SRC / "interfaces" / "design_system" / "static"
ARTIFACTS_STATIC = SRC / "interfaces" / "artifacts" / "static"
WEB_TERMINAL_STATIC = SRC / "interfaces" / "web_terminal" / "static"

_HTML_TAG = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_READY_VALUE = re.compile(r"""\bdata-page-ready\s*=\s*["']?([\w-]+)""", re.IGNORECASE)
_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_SRC = re.compile(r"""\bsrc\s*=\s*(["'])(.*?)\1""", re.IGNORECASE)
_IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(?\s*|new\s+URL\(\s*)(["'])([^"'\n]+?\.m?js)\1""")
_JINJA_EXPR = re.compile(r"\{\{.*?\}\}")
_EXTENDS = re.compile(r"""\{%-?\s*extends\s+["']([^"']+)["']""")
_INCLUDE = re.compile(r"""\{%-?\s*(?:include|import|from)\s+["']([^"']+)["']""")
_QUOTED = re.compile(r"""["']([^"']+)["']""")
#: The sign-in frame serves no static files, so its inline script writes the
#: attribute itself: the literal route, set by script.
_LITERAL_BY_SCRIPT = re.compile(
    r"""documentElement\.setAttribute\(\s*["']data-page-ready["']\s*,\s*["']true["']\s*\)"""
)
#: The writer and the shared start-up define these calls; a page runs them only when
#: a script of its own calls them.
_DEFINERS = frozenset(
    (DESIGN_SYSTEM_STATIC / "js" / name).resolve() for name in ("page-ready.js", "theme-manager.js")
)
_JS_COMMENT = re.compile(r"/\*.*?\*/|(?<![:'\"])//[^\n]*", re.DOTALL)
_INIT_THEME_CALL = re.compile(r"(?<!function )\binitTheme\s*\(")
_MARK_CALL = re.compile(r"(?<!function )\bmarkPageReady\s*\(")


def _strip_comments(source: str) -> str:
    return _JS_COMMENT.sub("", source)


def _resolve(spec: str, base: Path) -> Path | None:
    """The file a script ``src`` or an import specifier names, seen from *base*'s directory."""
    if "{{" in spec:
        # `{{ prefixed('/static/js/app.js') }}`: the address is the expression's last string.
        inner = [q for e in _JINJA_EXPR.findall(spec) for q in _QUOTED.findall(e)]
        spec = inner[-1] if inner else _JINJA_EXPR.sub("", spec)
    spec = spec.split("?", 1)[0].split("#", 1)[0]
    if not spec or spec.startswith(("http:", "https:", "data:", "//")):
        return None
    candidates: list[Path] = []
    if spec.startswith("/design-system/"):
        candidates.append(DESIGN_SYSTEM_STATIC / spec.removeprefix("/design-system/"))
    elif spec.startswith("/static/"):
        rest = spec.removeprefix("/static/")
        for ancestor in [base.parent, *base.parent.parents]:
            if ancestor.name == "static":
                candidates.append(ancestor / rest)
            candidates.append(ancestor / "static" / rest)
            if ancestor == SRC:
                break
    elif not spec.startswith("/"):
        candidates.append(base.parent / spec)
        # A panel's files are served from static/ beside its index.html.
        if spec.startswith("static/"):
            candidates.append(base.parent / spec.removeprefix("static/"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


@cache
def _module_graph(path: Path) -> frozenset[Path]:
    """*path* and every first-party module it imports, statically or dynamically."""
    seen: set[Path] = set()
    todo = [path]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        source = _strip_comments(current.read_text(encoding="utf-8"))
        for match in _IMPORT.finditer(source):
            target = _resolve(match.group(2), current)
            if target is not None:
                todo.append(target)
    return frozenset(seen)


def _graph_sources(target: Path | None) -> list[str]:
    if target is None:
        return []
    return [p.read_text(encoding="utf-8") for p in _module_graph(target) if p not in _DEFINERS]


def _reached_sources(page: str, where: Path) -> list[str]:
    """The source of every script *page* loads or holds inline, and of what those import.

    The modules that define ``initTheme`` and ``markPageReady`` are left out: loading
    them is not calling them.
    """
    sources: list[str] = []
    for attrs, body in _SCRIPT.findall(page):
        src = _SRC.search(attrs)
        if src is not None:
            sources += _graph_sources(_resolve(src.group(2), where))
            continue
        sources.append(body)
        for match in _IMPORT.finditer(_strip_comments(body)):
            sources += _graph_sources(_resolve(match.group(2), where))
    return [_strip_comments(source) for source in sources]


def _dressers(page_file: Path) -> list[str]:
    """The scripts that open *page_file* as a shell and dress it: they name its file."""
    named = re.compile(rf"""["'][^"']*/{re.escape(page_file.name)}["']""")
    return [
        _strip_comments(path.read_text(encoding="utf-8"))
        for path in sorted(SRC.rglob("*.js"))
        if "vendor" not in path.parts and named.search(path.read_text(encoding="utf-8"))
    ]


def route_problem(page: str, where: Path) -> str | None:
    """Why *page* (its markup, located at *where*) never reaches the mark, or None."""
    tag = _HTML_TAG.search(page)
    value = None
    if tag is not None and (found := _READY_VALUE.search(tag.group(0))) is not None:
        value = found.group(1).lower()
    if value == "true" or _LITERAL_BY_SCRIPT.search(page):
        return None
    sources = _reached_sources(page, where)
    if not _SCRIPT.search(page) and value == "pending":
        sources = _dressers(where)
    marks = any(_MARK_CALL.search(source) for source in sources)
    themed = any(_INIT_THEME_CALL.search(source) for source in sources)
    if value == "pending":
        return None if marks else "declares the mark pending but no script it loads sets it"
    if value is not None:
        return f"data-page-ready={value!r} is not a declared value (true or pending)"
    if marks and themed:
        return (
            "calls markPageReady() and runs the shared start-up; declare "
            'data-page-ready="pending" on <html>, or the start-up marks it first'
        )
    if marks or themed:
        return None
    return (
        "takes no route to the page-ready mark: run the shared start-up (initTheme), "
        "call markPageReady() from a script it loads, or carry the literal attribute"
    )


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
