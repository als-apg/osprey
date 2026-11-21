"""The Python reader of the page-ready contract ``design-system/js/page-ready.js`` states.

The mark is ``data-page-ready="true"`` on a page's ``<html>``, written by one module,
``page-ready.js``. The writer also sets ``failed`` on a pending page that threw before it
marked itself, a value a page never declares. A page reaches it by one of three routes, read here from its source
without a browser:

* **the shared start-up** -- the page says nothing about readiness, and a script it loads
  calls ``initTheme()``, which marks it once the document's ``DOMContentLoaded`` handlers
  have run;
* **late dressing** -- a script it loads calls ``markPageReady()``. When the page also runs
  the shared start-up, its ``<html>`` declares ``data-page-ready="pending"``, or the shared
  start-up would mark it first. A shell page with no script of its own (the window the hub
  pops a pane into) is marked by the script that opens and dresses it, which names its file;
* **the literal attribute** -- a page with no script at all carries
  ``data-page-ready="true"`` in its markup.

A panel takes the late route only: ``pending`` in its markup, and a script its entry page
loads calling ``markPageReady()`` once its first state is drawn, error or not. This holds for
a bundle the web terminal discovers and for an application's panel alike.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["FAILED", "MARKUP_VALUES", "READY_IMPORT", "route_problem"]

#: The line a panel's script adds to set the page-ready mark.
READY_IMPORT = "import { markPageReady } from '/design-system/js/page-ready.js';"

#: The values a page's markup may declare.
MARKUP_VALUES = ("true", "pending")

#: The writer's value for a pending page that threw; never markup.
FAILED = "failed"

#: The osprey package: the one tree a ``/static/`` or ``/design-system/`` address names.
_PACKAGE = Path(__file__).resolve().parents[3]
DESIGN_SYSTEM_STATIC = Path(__file__).resolve().parents[1] / "static"

_HTML_TAG = re.compile(r"<html\b[^>]*>", re.IGNORECASE)
_READY_VALUE = re.compile(r"""\bdata-page-ready\s*=\s*["']?([\w-]+)""", re.IGNORECASE)
_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_SRC = re.compile(r"""\bsrc\s*=\s*(["'])(.*?)\1""", re.IGNORECASE)
_IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(?\s*|new\s+URL\(\s*)(["'])([^"'\n]+?\.m?js)\1""")
_JINJA_EXPR = re.compile(r"\{\{.*?\}\}")
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

_Graphs = dict[Path, frozenset[Path]]


def _strip_comments(source: str) -> str:
    return _JS_COMMENT.sub("", source)


def _resolve(spec: str, base: Path) -> Path | None:
    """The file a script ``src`` or an import specifier names, seen from *base*'s directory.

    A ``/static/`` address is looked up in the ``static`` directories between *base* and the
    osprey package directory; seen from a page outside the package it names nothing.
    """
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
        directory = base.resolve().parent
        if directory == _PACKAGE or _PACKAGE in directory.parents:
            for ancestor in [directory, *directory.parents]:
                if ancestor.name == "static":
                    candidates.append(ancestor / rest)
                candidates.append(ancestor / "static" / rest)
                if ancestor == _PACKAGE:
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


def _module_graph(path: Path, graphs: _Graphs) -> frozenset[Path]:
    """*path* and every first-party module it imports, statically or dynamically.

    *graphs* memoizes within one :func:`route_problem` call only, so a file edited between
    two calls is read afresh.
    """
    if path in graphs:
        return graphs[path]
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
    graphs[path] = frozenset(seen)
    return graphs[path]


def _graph_sources(target: Path | None, graphs: _Graphs) -> list[str]:
    if target is None:
        return []
    return [
        p.read_text(encoding="utf-8") for p in _module_graph(target, graphs) if p not in _DEFINERS
    ]


def _reached_sources(page: str, where: Path) -> list[str]:
    """The source of every script *page* loads or holds inline, and of what those import.

    The modules that define ``initTheme`` and ``markPageReady`` are left out: loading
    them is not calling them.
    """
    graphs: _Graphs = {}
    sources: list[str] = []
    for attrs, body in _SCRIPT.findall(page):
        src = _SRC.search(attrs)
        if src is not None:
            sources += _graph_sources(_resolve(src.group(2), where), graphs)
            continue
        sources.append(body)
        for match in _IMPORT.finditer(_strip_comments(body)):
            sources += _graph_sources(_resolve(match.group(2), where), graphs)
    return [_strip_comments(source) for source in sources]


def _dressers(page_file: Path) -> list[str]:
    """The scripts that open *page_file* as a shell and dress it: they name its file.

    They are looked for in the osprey package's own scripts, third-party builds excluded.
    """
    named = re.compile(rf"""["'][^"']*/{re.escape(page_file.name)}["']""")
    return [
        _strip_comments(path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE.rglob("*.js"))
        if "vendor" not in path.parts and named.search(path.read_text(encoding="utf-8"))
    ]


def _ready_value(page: str) -> str | None:
    tag = _HTML_TAG.search(page)
    if tag is not None and (found := _READY_VALUE.search(tag.group(0))) is not None:
        return found.group(1).lower()
    return None


def _undeclared(value: str) -> str:
    if value == FAILED:
        return (
            f'data-page-ready="{FAILED}" is the writer\'s value for a page that threw while '
            "pending; markup declares true or pending"
        )
    return f"data-page-ready={value!r} is not a declared value (true or pending)"


def _panel_problem(page: str, where: Path, value: str | None) -> str | None:
    if value == "true":
        return (
            'carries data-page-ready="true", which marks it before its first state is drawn; '
            'a panel declares "pending" and its script calls markPageReady()'
        )
    if not any(_MARK_CALL.search(source) for source in _reached_sources(page, where)):
        return (
            "never sets the page-ready mark; the panel's script adds "
            f"`{READY_IMPORT}` and calls markPageReady() once its first state is drawn, "
            "error or not"
        )
    if value != "pending":
        return (
            'calls markPageReady() without data-page-ready="pending" on <html>, so the page '
            "can be marked before that call; a panel declares it"
        )
    return None


def route_problem(page: str, where: Path, *, panel: bool = False) -> str | None:
    """Why *page* never reaches the page-ready mark by a declared route, or None.

    Args:
        page: The page's markup.
        where: The file the page is served as; its relative script addresses resolve
            from here.
        panel: Judge *page* as a panel, which takes the late route only:
            ``data-page-ready="pending"`` on ``<html>`` and a script the page loads (or a
            module that script imports) calling ``markPageReady()``. Neither a script
            that sets the literal attribute nor a script that dresses the page as a shell
            excuses a panel.

    Returns:
        One sentence naming what the page lacks, or None when it reaches the mark.
    """
    value = _ready_value(page)
    if panel:
        if value is not None and value not in MARKUP_VALUES:
            return _undeclared(value)
        return _panel_problem(page, where, value)
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
        return _undeclared(value)
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
