"""The page-ready judge: the general routes, and the late route a panel takes."""

from __future__ import annotations

from pathlib import Path

import pytest

import osprey
from osprey.interfaces.design_system.panels.page_ready import (
    DESIGN_SYSTEM_STATIC,
    FAILED,
    MARKUP_VALUES,
    READY_IMPORT,
    WRITER_TAG,
    route_problem,
)

PACKAGE = Path(osprey.__file__).resolve().parent
WEB_TERMINAL_STATIC = PACKAGE / "interfaces" / "web_terminal" / "static"

INLINE_MARK = (
    '<script type="module">import { markPageReady } from '
    "'/design-system/js/page-ready.js'; markPageReady();</script>"
)
INLINE_THEME = (
    '<script type="module">import { initTheme } from '
    "'/design-system/js/theme-manager.js'; initTheme();</script>"
)
PENDING = '<html data-page-ready="pending">'
WRITER = WRITER_TAG
WRITER_PROBLEM = (
    f"does not load the page-ready writer first; add `{WRITER_TAG}` ahead of every other "
    "module or deferred script, so a script that fails to load marks the page failed"
)
ASYNC_PROBLEM = "loads an own module `async`; drop `async` so it runs after the page-ready writer"

TRUE_PROBLEM = (
    'carries data-page-ready="true", which marks it before its first state is drawn; '
    'a panel declares "pending" and its script calls markPageReady()'
)
NEVER_PROBLEM = (
    f"never sets the page-ready mark; the panel's script adds `{READY_IMPORT}` and calls "
    "markPageReady() once its first state is drawn, error or not"
)
FAILED_PROBLEM = (
    'data-page-ready="failed" is the writer\'s value for a page that threw while pending; '
    "markup declares true or pending"
)
PENDING_PROBLEM = (
    'calls markPageReady() without data-page-ready="pending" on <html>, so the page '
    "can be marked before that call; a panel declares it"
)


def _panel_index(root: Path) -> Path:
    return root / "apps" / "knob" / "panel" / "index.html"


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (f"{PENDING}{WRITER}{INLINE_MARK}</html>", None),
        (f"{PENDING}{WRITER}{INLINE_THEME}{INLINE_MARK}</html>", None),
        (f"<html>{INLINE_MARK}</html>", PENDING_PROBLEM),
        (f"<html>{INLINE_THEME}{INLINE_MARK}</html>", PENDING_PROBLEM),
        ('<html data-page-ready="true"></html>', TRUE_PROBLEM),
        (f"{PENDING}{INLINE_THEME}</html>", NEVER_PROBLEM),
        (
            '<html data-page-ready="yes"></html>',
            "data-page-ready='yes' is not a declared value (true or pending)",
        ),
        (f'<html data-page-ready="failed">{INLINE_MARK}</html>', FAILED_PROBLEM),
        (f"{PENDING}{INLINE_MARK}</html>", WRITER_PROBLEM),
        (f"{PENDING}{INLINE_MARK}{WRITER}</html>", WRITER_PROBLEM),
        (
            f"{PENDING}{WRITER.replace('<script ', '<script async ')}{INLINE_MARK}</html>",
            WRITER_PROBLEM,
        ),
        (
            f'{PENDING}<script src="/design-system/js/theme-boot.js"></script>{WRITER}'
            f"{INLINE_MARK}</html>",
            None,
        ),
        (
            f'{PENDING}<script defer src="x.js"></script>{WRITER}{INLINE_MARK}</html>',
            WRITER_PROBLEM,
        ),
        (
            f'{PENDING}{WRITER}{INLINE_MARK}<script type="module" async src="x.js"></script></html>',
            ASYNC_PROBLEM,
        ),
        (
            f"{PENDING}{WRITER}{INLINE_MARK}"
            '<script async src="https://cdn.example/x.js"></script></html>',
            None,
        ),
    ],
    ids=[
        "pending-writer-and-mark",
        "pending-start-up-and-mark",
        "mark-without-pending",
        "start-up-and-mark-without-pending",
        "literal-true",
        "pending-and-start-up-only",
        "undeclared-value",
        "failed-in-markup",
        "pending-mark-no-writer-tag",
        "writer-tag-after-a-module",
        "writer-tag-async",
        "writer-tag-after-theme-boot",
        "writer-tag-after-a-deferred-script",
        "own-async-module-after-the-writer",
        "vendor-async-script",
    ],
)
def test_a_panel_takes_the_late_route_only(page: str, expected: str | None, tmp_path: Path) -> None:
    assert route_problem(page, _panel_index(tmp_path), panel=True) == expected


def test_a_panel_is_never_excused_by_a_dresser_or_a_literal_set_by_script() -> None:
    # A file a package script opens and dresses as a shell, and the sign-in frame's
    # inline write of the literal attribute: both reach the mark as served pages.
    shell = WEB_TERMINAL_STATIC / "attached-window.html"
    shell_page = shell.read_text(encoding="utf-8")
    literal = (
        "<html><script>document.documentElement.setAttribute("
        "'data-page-ready', 'true');</script></html>"
    )
    assert route_problem(shell_page, shell) is None
    assert route_problem(literal, WEB_TERMINAL_STATIC / "x.html") is None

    assert route_problem(shell_page, shell, panel=True) == NEVER_PROBLEM
    assert route_problem(literal, WEB_TERMINAL_STATIC / "x.html", panel=True) == (NEVER_PROBLEM)


def test_markup_that_declares_failed_is_refused_on_every_route(tmp_path: Path) -> None:
    for page in (
        '<html data-page-ready="failed"></html>',
        f'<html data-page-ready="failed">{INLINE_MARK}</html>',
        f'<html data-page-ready="failed">{INLINE_THEME}</html>',
    ):
        for panel in (False, True):
            assert route_problem(page, _panel_index(tmp_path), panel=panel) == (FAILED_PROBLEM)


def test_the_judge_and_the_writer_name_the_same_values() -> None:
    writer = (DESIGN_SYSTEM_STATIC / "js" / "page-ready.js").read_text(encoding="utf-8")
    assert MARKUP_VALUES[1] == "pending"
    assert f"const PENDING = '{MARKUP_VALUES[1]}';" in writer
    assert f"const FAILED = '{FAILED}';" in writer


def test_a_pending_page_on_the_general_route_loads_the_writer_first(tmp_path: Path) -> None:
    where = tmp_path / "x.html"
    assert route_problem(f"{PENDING}{INLINE_MARK}</html>", where) == WRITER_PROBLEM
    assert route_problem(f"{PENDING}{WRITER}{INLINE_MARK}</html>", where) is None
    # A shell with no script of its own is dressed by the page that opens it.
    shell = WEB_TERMINAL_STATIC / "attached-window.html"
    assert route_problem(shell.read_text(encoding="utf-8"), shell) is None


def test_the_writer_tag_is_recognised_however_its_address_is_spelled(tmp_path: Path) -> None:
    for tag, where in (
        (
            '<script type="module" src="{{ prefixed(\'/design-system/js/page-ready.js\') }}">'
            "</script>",
            tmp_path / "x.html",
        ),
        (
            '<script type="module" src="{{ root }}/design-system/js/page-ready.js"></script>',
            tmp_path / "x.html",
        ),
        (
            '<script type="module" src="js/page-ready.js"></script>',
            DESIGN_SYSTEM_STATIC / "theme-lab.html",
        ),
    ):
        assert route_problem(f"{PENDING}{tag}{INLINE_MARK}</html>", where) is None, tag
        assert route_problem(f"{PENDING}{tag}{INLINE_MARK}</html>", where, panel=True) is None


def test_the_writer_judges_an_own_script_by_origin_and_vendor_path() -> None:
    shell_boot = WEB_TERMINAL_STATIC / "js" / "shell-boot.js"
    for judge in (DESIGN_SYSTEM_STATIC / "js" / "page-ready.js", shell_boot):
        source = judge.read_text(encoding="utf-8")
        assert "file.origin === window.location.origin" in source, judge.name
        assert "file.pathname.includes('/vendor/')" in source, judge.name


def test_the_general_route_still_accepts_the_three_routes(tmp_path: Path) -> None:
    where = tmp_path / "x.html"
    assert route_problem(f"<html>{INLINE_THEME}</html>", where) is None
    assert route_problem(f"<html>{INLINE_MARK}</html>", where) is None
    assert route_problem('<html data-page-ready="true"></html>', where) is None


def _shipped_app_panels() -> list[Path]:
    return sorted(
        [
            *PACKAGE.glob("apps/*/panel/index.html"),
            *PACKAGE.glob("profiles/presets/apps/*/panel/index.html"),
        ]
    )


def test_the_shipped_app_panels_are_found() -> None:
    names = {index.parent.parent.name for index in _shipped_app_panels()}
    assert {"scaffold-app", "orbit-bump"} <= names


@pytest.mark.parametrize(
    "index", _shipped_app_panels(), ids=lambda p: p.relative_to(PACKAGE).as_posix()
)
def test_every_shipped_app_panel_takes_the_late_route(index: Path) -> None:
    assert route_problem(index.read_text(encoding="utf-8"), index, panel=True) is None


def test_a_static_address_outside_the_package_resolves_to_nothing(tmp_path: Path) -> None:
    index = tmp_path / "app" / "panel" / "index.html"
    index.parent.mkdir(parents=True)
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "x.js").write_text(READY_IMPORT + "\nmarkPageReady();\n")
    page = f'{PENDING}<script type="module" src="/static/x.js"></script></html>'

    assert route_problem(page, index, panel=True) == NEVER_PROBLEM
