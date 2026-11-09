"""ARIEL's static surface carries nothing the one-screen client no longer has.

ARIEL moved from a set of page views (search, dashboard, create entry, an
answer box, an entry modal) to one mail-client screen. Nothing fails at
runtime when the old surface lingers: a dead stylesheet rule simply matches
nothing, and a leftover global such as ``window.app`` keeps working until the
next refactor relies on it being gone. These checks catch the drift statically:

* no ARIEL static file reaches for the retired globals and views
  (``window.app``, the answer-box renderer, "Vision Search", ``view-``
  sections);
* no ARIEL stylesheet restyles a design-system ``.osprey-*`` primitive — those
  visuals belong to ``primitives.css``, and a host override forks them; and
* every class or id selector in ARIEL CSS names something ARIEL's HTML or
  JavaScript can actually produce. A name built from a template literal prefix
  (``diagnostics-msg-${level}``) counts as produced by that prefix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[3] / "src/osprey/interfaces/ariel/static"
_CSS_FILES = sorted((_STATIC / "css").glob("*.css"))
_JS_FILES = sorted((_STATIC / "js").glob("*.js"))
_HTML_FILES = sorted(_STATIC.glob("*.html"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
    return path.relative_to(_STATIC).as_posix()


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _css_preludes(css: str) -> list[str]:
    """Every selector prelude in ``css``, at-rule preludes and keyframe steps excluded."""
    preludes = []
    for match in re.finditer(r"([^{};]+)\{", _strip_css_comments(css)):
        prelude = match.group(1).strip()
        if not prelude or prelude.startswith("@"):
            continue
        if re.fullmatch(r"(from|to|[\d.]+%)(\s*,\s*(from|to|[\d.]+%))*", prelude):
            continue
        preludes.append(prelude)
    return preludes


def _selector_names(prelude: str) -> set[str]:
    """The ``.class`` and ``#id`` names a selector prelude targets."""
    without_attrs = re.sub(r"\[[^\]]*\]", "", prelude)
    return {
        f"{kind}{name}" for kind, name in re.findall(r"([.#])(-?[A-Za-z_][\w-]*)", without_attrs)
    }


def _js_literals(source: str) -> list[str]:
    """The bodies of every string and template literal, comments skipped."""
    literals: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = n if end == -1 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif ch in "'\"`":
            j = i + 1
            while j < n and source[j] != ch:
                if source[j] == "\\":
                    j += 1
                elif ch != "`" and source[j] == "\n":
                    break
                j += 1
            literals.append(source[i + 1 : j])
            i = j + 1
        else:
            i += 1
    return literals


_IDENT = re.compile(r"[A-Za-z_][\w-]*")
_TEMPLATE_PREFIX = re.compile(r"([A-Za-z][\w-]*-)\$\{")


@pytest.fixture(scope="module")
def produced() -> tuple[frozenset[str], tuple[str, ...]]:
    """Names ARIEL markup and scripts can put on an element, and template prefixes."""
    names: set[str] = set()
    prefixes: set[str] = set()
    for path in _HTML_FILES:
        html = re.sub(r"<!--.*?-->", "", _read(path), flags=re.DOTALL)
        for value in re.findall(r"\b(?:class|id|for)\s*=\s*\"([^\"]*)\"", html):
            names.update(value.split())
    for path in _JS_FILES:
        for literal in _js_literals(_read(path)):
            names.update(_IDENT.findall(literal))
            prefixes.update(_TEMPLATE_PREFIX.findall(literal))
    return frozenset(names), tuple(sorted(prefixes))


@pytest.mark.parametrize("path", _HTML_FILES + _JS_FILES + _CSS_FILES, ids=_rel)
def test_no_retired_global_or_view_survives(path):
    text = _read(path)
    retired = {
        "window.app": "window.app" in text,
        "answer-box renderer": bool(re.search(r"renderAnswerBox|answer-box", text)),
        "Vision Search": "Vision Search" in text,
        "view- section": bool(re.search(r"(?<![\w-])view-[A-Za-z]", text)),
    }
    found = sorted(name for name, hit in retired.items() if hit)
    assert not found, f"{_rel(path)} still carries the retired surface: {found}"


@pytest.mark.parametrize("path", _CSS_FILES, ids=_rel)
def test_no_rule_restyles_a_design_system_primitive(path):
    offenders = sorted(p for p in _css_preludes(_read(path)) if ".osprey-" in p)
    assert not offenders, (
        f"{_rel(path)} restyles .osprey-* primitives owned by primitives.css: {offenders}"
    )


@pytest.mark.parametrize("path", _CSS_FILES, ids=_rel)
def test_every_selector_names_something_ariel_produces(path, produced):
    names, prefixes = produced
    dead = set()
    for prelude in _css_preludes(_read(path)):
        for selector in _selector_names(prelude):
            name = selector[1:]
            if name in names or name.startswith(prefixes):
                continue
            dead.add(selector)
    assert not dead, f"{_rel(path)} styles names no ARIEL HTML or JS produces: {sorted(dead)}"


def test_the_usage_scan_sees_template_prefixes(produced):
    """A selector reached only through ``diagnostics-msg-${level}`` must count as used."""
    _, prefixes = produced
    assert "diagnostics-msg-" in prefixes
