"""The Channels panel has one severity vocabulary: severity.js and severity.css.

Rendering code takes a severity from the data and hands it to ``severity.js``
(``severityOf``, ``severityMark``); a module that spells a severity name is the
sign of a second vocabulary, and a stylesheet that styles the mark is a second
look. The scan is narrow on purpose: a string literal equal to a name, and a
selector naming ``cf-sev``. Prose and comments are not read.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).parents[3] / "src/osprey/interfaces/channel_finder/static"
_JS = _STATIC / "js"
_CSS = _STATIC / "css"

_WORDS = ("MAJOR", "MINOR", "INVALID", "UNDEFINED", "NO_ALARM")
SEVERITY_NAMES = frozenset(_WORDS) | {f"{word}_ACK" for word in _WORDS}

#: A selector naming the mark: ``.cf-sev`` or a ``.cf-sev--<severity>`` modifier.
_MARK_SELECTOR = re.compile(r"\.cf-sev(?:--[\w-]+)?(?![\w-])")


def string_literals(source: str) -> list[tuple[int, str]]:
    """Every string literal of a JS source with its line, comments skipped.

    A template literal yields each static run between its interpolations.
    """
    found: list[tuple[int, str]] = []
    i, line, n = 0, 1, len(source)
    while i < n:
        ch = source[i]
        if ch == "\n":
            line += 1
            i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end < 0 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += source.count("\n", i, end)
            i = end
        elif ch in "'\"`":
            start_line, i = line, i + 1
            run: list[str] = []
            while i < n and source[i] != ch:
                if source[i] == "\\" and i + 1 < n:
                    run.append(source[i : i + 2])
                    i += 2
                    continue
                if ch == "`" and source.startswith("${", i):
                    found.append((start_line, "".join(run)))
                    run, depth = [], 0
                    i += 1  # past the `$`; the braces are counted from `{`
                    while i < n:
                        depth += {"{": 1, "}": -1}.get(source[i], 0)
                        i += 1
                        if depth == 0:
                            break
                    continue
                if source[i] == "\n":
                    line += 1
                run.append(source[i])
                i += 1
            found.append((start_line, "".join(run)))
            i += 1
        else:
            i += 1
    return found


def severity_literals(source: str) -> list[tuple[int, str]]:
    """The string literals of a JS source that equal a severity name."""
    return [(line, text) for line, text in string_literals(source) if text in SEVERITY_NAMES]


def mark_selectors(stylesheet: str) -> list[tuple[int, str]]:
    """Every rule prelude of a stylesheet that names the severity mark."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), stylesheet, flags=re.S)
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"[^{};]+(?=\{)", text):
        prelude = match.group().strip()
        if _MARK_SELECTOR.search(prelude):
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            found.append((text.count("\n", 0, start) + 1, prelude))
    return found


def test_no_module_but_severity_js_spells_a_severity_name() -> None:
    offenders = [
        f"{path.name}:{line}: {text!r}"
        for path in sorted(_JS.glob("*.js"))
        if path.name != "severity.js"
        for line, text in severity_literals(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "a second severity vocabulary; use severity.js:\n" + "\n".join(
        offenders
    )


def test_no_stylesheet_but_severity_css_styles_the_mark() -> None:
    offenders = [
        f"{path.name}:{line}: {prelude}"
        for path in sorted(_CSS.glob("*.css"))
        if path.name != "severity.css"
        for line, prelude in mark_selectors(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "the severity mark styled outside severity.css:\n" + "\n".join(
        offenders
    )


def test_the_scan_refuses_a_spelled_severity(tmp_path: Path) -> None:
    module = tmp_path / "other.js"
    module.write_text(
        "// MAJOR in a comment is prose\n"
        "/** @typedef {'MAJOR'|'MINOR'} T */\n"
        "const label = 'Major alarm';\n"
        "const worst = 'MAJOR';\n"
        'const acked = "MINOR_ACK";\n'
        "const tpl = `NO_ALARM`;\n"
        "const mixed = `${a}INVALID${b}`;\n",
        encoding="utf-8",
    )
    assert severity_literals(module.read_text(encoding="utf-8")) == [
        (4, "MAJOR"),
        (5, "MINOR_ACK"),
        (6, "NO_ALARM"),
        (7, "INVALID"),
    ]


@pytest.mark.parametrize(
    ("stylesheet", "expected"),
    [
        (".row .cf-sev { color: red; }", [(1, ".row .cf-sev")]),
        ("/* .cf-sev */\n.a,\n.cf-sev--major > i { x: y; }", [(2, ".a,\n.cf-sev--major > i")]),
        (".pv { color: var(--cf-severity-invalid); }", []),
        ("@media (x) { .cf-sev { a: b; } }", [(1, ".cf-sev")]),
    ],
)
def test_the_scan_finds_a_mark_selector(stylesheet: str, expected: list[tuple[int, str]]) -> None:
    assert mark_selectors(stylesheet) == expected
