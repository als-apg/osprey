"""A browser test drives a page only after the page says it is ready.

Every page OSPREY serves sets ``data-page-ready="true"`` on its ``<html>`` once
it can be driven (``design-system/js/page-ready.js``), and a browser test waits
for that mark through one helper, ``tests.interfaces._browser.wait_for_page_ready``.
This guard holds every test file to it, with one rule: after a navigation, the
next use of that page is ``wait_for_page_ready`` or a helper that calls it.

* **A navigation** is ``goto``, ``reload``, ``wait_for_url``, ``go_back`` or
  ``go_forward`` on a name bound to a page, and a page a test takes from
  ``expect_popup`` / ``expect_page`` (the ``.value`` of the context). The guard
  follows names bound to a page -- a parameter annotated or named as one, a
  name bound from ``new_page()`` and its kin, the receiver of ``goto`` -- so
  ``importlib.reload`` or a container's ``reload()`` is not one.
* **A helper that calls it** is any function under ``tests/`` whose body calls
  ``wait_for_page_ready`` or another such helper; the set is derived, not
  listed. A page bound from ``open_panel_host`` navigates through
  ``PanelHost.goto`` / ``reload``, which wait themselves.
* **The one way out** is a marker on the navigation's own line,
  ``# page-ready: <reason>``, for a document OSPREY does not author or a test
  whose subject is the page before it is ready. A marker without a reason is
  refused. There is no list of exempt files.

The files are every ``.py`` under ``tests/`` that navigates a page or waits for one,
so the set is derived too.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]

#: The one wait.
WAIT = "wait_for_page_ready"
#: Calls on a page that load another document into it.
NAVIGATIONS = frozenset({"goto", "reload", "wait_for_url", "go_back", "go_forward"})
#: Of those, the ones only a page (or frame) has: their receiver is a page.
PAGE_ONLY = NAVIGATIONS - {"reload"}
#: Context managers whose ``.value`` is a new page another one opened.
POPUP_CONTEXTS = frozenset({"expect_popup", "expect_page"})
#: Calls that hand back a page (or frame) the test then drives.
PAGE_PRODUCERS = frozenset(
    {"new_page", "open_hub_page", "content_frame", "frame", "frame_locator", "PanelHost"}
)
#: Calls that hand back a page whose own navigations wait (``PanelHost``).
WAITING_PRODUCERS = frozenset({"open_panel_host", "PanelHost"})
#: Annotations naming a page.
PAGE_TYPES = frozenset({"Page", "Frame", "FrameLocator", "PanelHost"})
#: Parameter names a page arrives under (pytest-playwright's ``page`` and its kin).
PAGE_PARAM = re.compile(r"^(?:page|popup|window|frame|.+_page|.+_popup|.+_window|.+_frame)$")

MARKER = re.compile(r"#\s*page-ready:(.*)$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _call_name(node: ast.AST) -> str | None:
    """``f`` for ``f(...)`` and ``x.y.f(...)``."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _key(node: ast.AST) -> str | None:
    """A stable spelling of a page expression: ``page``, ``self.page``, ``host.frame``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _key(node.value)
        return f"{inner}.{node.attr}" if inner else None
    return None


def _annotation_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return set(re.findall(r"\w+", node.value))
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def waiting_helpers(trees: Iterable[ast.Module]) -> frozenset[str]:
    """Every function name whose body calls the wait or another such function."""
    bodies: dict[str, set[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                calls = {name for n in ast.walk(node) if (name := _call_name(n)) is not None}
                bodies.setdefault(node.name, set()).update(calls)
    found = {WAIT}
    while True:
        more = {name for name, calls in bodies.items() if calls & found} - found
        if not more:
            return frozenset(found)
        found |= more


def _own_nodes(func: ast.AST) -> Iterator[ast.AST]:
    """The nodes of *func*, nested function bodies left out (they are their own scope)."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _pos(node: ast.AST) -> tuple[int, int]:
    return (node.lineno, node.col_offset)  # type: ignore[attr-defined]


def _end(node: ast.AST) -> tuple[int, int]:
    return (node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]


class _Scope:
    """One function: which names hold pages, and where each one is used."""

    def __init__(self, func: ast.FunctionDef | ast.AsyncFunctionDef, fixtures: set[str]) -> None:
        self.func = func
        self.nodes = list(_own_nodes(func))
        self.pages: set[str] = set()
        self.waiting: set[str] = set()
        self.popup_infos: set[str] = set()
        self._reads: dict[str, list[ast.AST]] | None = None
        args = func.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            annotated = _annotation_names(arg.annotation)
            if annotated & PAGE_TYPES or PAGE_PARAM.match(arg.arg):
                self.pages.add(arg.arg)
            if "PanelHost" in annotated or arg.arg in fixtures:
                self.waiting.add(arg.arg)
        for node in self.nodes:
            if isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                if _call_name(node.context_expr) in POPUP_CONTEXTS:
                    self.popup_infos.add(node.optional_vars.id)
            if isinstance(node, ast.Call) and _call_name(node) in PAGE_ONLY:
                if isinstance(node.func, ast.Attribute) and (k := _key(node.func.value)):
                    self.pages.add(k)
            if isinstance(node, ast.Assign | ast.AnnAssign):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [k for t in targets if (k := _key(t)) is not None]
                if value is not None and _call_name(value) in PAGE_PRODUCERS | WAITING_PRODUCERS:
                    self.pages.update(names)
                if value is not None and _call_name(value) in WAITING_PRODUCERS:
                    self.waiting.update(names)
                if isinstance(node, ast.AnnAssign) and _annotation_names(node.annotation) & (
                    PAGE_TYPES
                ):
                    self.pages.update(names)

    def is_page(self, key: str) -> bool:
        return key in self.pages or key.rsplit(".", 1)[-1] in {"page", "frame", "popup"}

    def first_use_after(self, key: str, after: tuple[int, int]) -> ast.AST | None:
        """The first expression that reads *key* after position *after*."""
        if self._reads is None:
            reads: dict[str, list[ast.AST]] = {}
            for node in self.nodes:
                if isinstance(node, ast.Name | ast.Attribute) and isinstance(
                    getattr(node, "ctx", None), ast.Load
                ):
                    if (k := _key(node)) is not None:
                        reads.setdefault(k, []).append(node)
            self._reads = {k: sorted(v, key=_pos) for k, v in reads.items()}
        return next((n for n in self._reads.get(key, ()) if _pos(n) >= after), None)


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _fixtures_returning_waiting_pages(tree: ast.Module) -> set[str]:
    """Fixtures (by name) whose value is a page bound from ``open_panel_host``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any("fixture" in ast.unparse(d) for d in node.decorator_list):
            continue
        bound = {
            k
            for n in ast.walk(node)
            if isinstance(n, ast.Assign) and _call_name(n.value) in WAITING_PRODUCERS
            for t in n.targets
            if (k := _key(t))
        }
        for n in ast.walk(node):
            if isinstance(n, ast.Return | ast.Yield) and n.value is not None:
                if _call_name(n.value) in WAITING_PRODUCERS or _key(n.value) in bound:
                    names.add(node.name)
    return names


def check_source(
    source: str, path: str, helpers: frozenset[str], fixtures: frozenset[str] = frozenset()
) -> list[Finding]:
    """Every navigation in *source* whose page is driven before it is waited for."""
    tree = ast.parse(source)
    lines = source.splitlines()
    parents = _parents(tree)
    waiting_fixtures = set(fixtures) | _fixtures_returning_waiting_pages(tree)
    findings: list[Finding] = []

    def marker(node: ast.AST) -> str | None:
        for line in {node.lineno, node.end_lineno}:  # type: ignore[attr-defined]
            if (found := MARKER.search(lines[line - 1])) is not None:
                return found.group(1).strip()
        return None

    def waited(use: ast.AST) -> bool:
        """Whether *use* is an argument of the wait, or of a helper that calls it."""
        node = use
        while node in parents:
            parent = parents[node]
            if isinstance(parent, ast.Call) and node is not parent.func:
                return _call_name(parent) in helpers
            if isinstance(parent, ast.Attribute | ast.Subscript | ast.Starred | ast.keyword):
                node = parent
                continue
            return False
        return False

    def check(scope: _Scope, key: str, navigation: ast.AST, what: str) -> None:
        reason = marker(navigation)
        if reason is not None:
            if not reason:
                findings.append(
                    Finding(path, navigation.lineno, "a page-ready marker needs a reason")  # type: ignore[attr-defined]
                )
            return
        use = scope.first_use_after(key, _end(navigation))
        if use is not None and not waited(use):
            findings.append(
                Finding(
                    path,
                    navigation.lineno,  # type: ignore[attr-defined]
                    f"{what} drives `{key}` (line {use.lineno}) before "  # type: ignore[attr-defined]
                    f"{WAIT}({key})",
                )
            )

    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        scope = _Scope(func, waiting_fixtures)
        for node in scope.nodes:
            name = _call_name(node)
            if name in NAVIGATIONS and isinstance(node.func, ast.Attribute):  # type: ignore[attr-defined]
                key = _key(node.func.value)  # type: ignore[attr-defined]
                if key is None or not scope.is_page(key) or key in scope.waiting:
                    continue
                check(scope, key, node, f"`{key}.{name}()`")
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "value"
                and isinstance(node.value, ast.Name)
                and node.value.id in scope.popup_infos
            ):
                parent = parents.get(node)
                if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
                    key = _key(parent.targets[0])
                    if key is not None:
                        check(scope, key, parent, "a new page")
                elif not waited(node) and marker(node) is None:
                    findings.append(
                        Finding(path, node.lineno, f"a new page is driven before {WAIT}()")
                    )
    return findings


#: Text a file must hold to navigate a page or to wait for one; the rest of the tree is
#: read no further (the AST pass below is the judge, this only skips files fast).
_NAVIGATING_TEXT = re.compile(
    r"\.(?:goto|reload|wait_for_url|go_back|go_forward|expect_popup|expect_page)\(|" + WAIT
)


def _test_sources() -> dict[str, str]:
    """Every test module that navigates or waits, by repo-relative path."""
    sources: dict[str, str] = {}
    for path in sorted(TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if _NAVIGATING_TEXT.search(text):
            sources[path.relative_to(TESTS.parent).as_posix()] = text
    return sources


def _tree_findings() -> list[Finding]:
    sources = _test_sources()
    trees = [ast.parse(text) for text in sources.values()]
    helpers = waiting_helpers(trees)
    fixtures: set[str] = set()
    for tree in trees:
        fixtures |= _fixtures_returning_waiting_pages(tree)
    findings: list[Finding] = []
    for rel, text in sources.items():
        findings += check_source(text, rel, helpers, frozenset(fixtures))
    return findings


def test_every_browser_test_waits_for_the_page_ready_mark_before_driving_a_page() -> None:
    findings = _tree_findings()
    assert not findings, "\n".join(str(f) for f in findings)


# --- the guard's own fixtures ---------------------------------------------------

HELPERS = frozenset({WAIT})


def _check(source: str) -> list[str]:
    helpers = waiting_helpers([ast.parse(source)]) | HELPERS
    return [str(f) for f in check_source(source, "t.py", helpers)]


def test_a_navigation_followed_by_a_drive_is_refused_naming_file_and_line() -> None:
    found = _check(
        "def test_x(page):\n    page.goto('http://h/')\n    page.locator('#x').click()\n"
    )
    assert found == [
        "t.py:2: `page.goto()` drives `page` (line 3) before wait_for_page_ready(page)"
    ]


def test_a_navigation_followed_by_the_wait_complies() -> None:
    assert not _check(
        "def test_x(page):\n"
        "    page.goto('http://h/')\n"
        "    wait_for_page_ready(page)\n"
        "    page.locator('#x').click()\n"
    )


def test_a_navigation_followed_by_a_helper_that_waits_complies() -> None:
    assert not _check(
        "def settle(p):\n"
        "    wait_for_page_ready(p)\n"
        "\n"
        "def test_x(page):\n"
        "    page.reload()\n"
        "    settle(page)\n"
        "    page.locator('#x').click()\n"
    )


def test_a_marked_navigation_is_left_alone_and_a_marker_needs_a_reason() -> None:
    assert not _check(
        "def test_x(page):\n"
        "    page.goto('about:blank')  # page-ready: not an OSPREY page\n"
        "    page.locator('#x').click()\n"
    )
    assert _check(
        "def test_x(page):\n"
        "    page.goto('about:blank')  # page-ready:\n"
        "    page.locator('#x').click()\n"
    ) == ["t.py:2: a page-ready marker needs a reason"]


def test_a_popup_is_a_navigation() -> None:
    found = _check(
        "def test_x(page):\n"
        "    with page.expect_popup() as info:\n"
        "        page.click('a')\n"
        "    popup = info.value\n"
        "    popup.locator('#x').click()\n"
    )
    assert found == ["t.py:4: a new page drives `popup` (line 5) before wait_for_page_ready(popup)"]


def test_calls_that_are_not_page_navigations_are_not_matched() -> None:
    assert not _check(
        "import importlib\n"
        "def test_x(container, module):\n"
        "    importlib.reload(module)\n"
        "    module.run()\n"
        "    container.reload()\n"
        "    container.status()\n"
    )


def test_a_page_from_open_panel_host_navigates_through_its_waiting_verbs() -> None:
    assert not _check(
        "def test_x(page):\n"
        "    host = open_panel_host(page, 'http://h/')\n"
        "    host.reload()\n"
        "    host.locator('#x').click()\n"
    )


@pytest.mark.parametrize("call", ["wait_for_url('**/x')", "go_back()"])
def test_every_navigation_verb_is_matched(call: str) -> None:
    found = _check(f"def test_x(page):\n    page.{call}\n    page.locator('#x').click()\n")
    assert len(found) == 1
