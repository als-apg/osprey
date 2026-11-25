"""No served page keeps "only the newest applies" by hand; ``latest-request.js`` does.

Wherever a page lets only the newest of one kind apply (a request whose reply
must not overwrite a newer one, a view lifetime that a remount retires, a timer
that a newer one supersedes), it holds one ``createLatestRequest`` instance from
``design_system/static/js/latest-request.js`` per kind and asks its token
whether it is still current. A hand-written copy bumps a counter, captures its
value, and compares the capture back to the counter later.

This guard walks every script and page template under ``src/osprey`` for that
shape: a counter that is bumped (``++y``, ``y++`` or ``y += 1``), a ``const`` or
``let`` that captures it, and an ``===`` or ``!==`` between the two. Comments
are blanked first, so prose that describes the shape never hits. The pattern is
a net, not a proof: a counter captured by destructuring or compared through a
helper function slips past it, and review owns those forms.

A guard of another rule is not this one: a panel that shows a state answer
only when it is no older than the one shown (``<=``) lets a newer answer show
while a still-newer poll is out, and the pattern leaves it alone.
"""

from __future__ import annotations

import re

import pytest

from tests.interfaces.design_system._served_sources import SRC, served_sources

#: An identifier or a dotted member path, optionally on ``this``.
_NAME = r"(?:this\.)?[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*"

#: Every bump of a counter by one.
_BUMP = re.compile(rf"\+\+\s*(?P<a>{_NAME})|(?P<b>{_NAME})\s*\+\+|(?P<c>{_NAME})\s*\+=\s*1\b")

#: A capture of a counter's value that ends the statement.
_CAPTURE = re.compile(
    rf"\b(?:const|let)\s+(?P<x>[A-Za-z_$][\w$]*)\s*=\s*(?:\+\+\s*)?(?P<y>{_NAME})\s*[;,\n]"
)

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"(?<![:\\'\"`])//[^\n]*")

#: Shapes the pattern matches that are not newest-wins, with the reason. The
#: table only shrinks: an entry whose code no longer matches fails until it is
#: removed.
_NOT_A_GUARD = {
    ("interfaces/design_system/static/js/latest-request.js", "generation"): "the helper itself",
    ("interfaces/web_terminal/static/js/sidebar-artifact-listing.js", "ownedMoves"): (
        "a stamp of the frames applied here since a summary read began; overlapping "
        "summarize reads each compare against it, so a read starting must not retire another"
    ),
    ("interfaces/web_terminal/static/js/tour.js", "this.i"): (
        "the tour's step index, moved back as well as forward"
    ),
}


def _strip_comments(text: str) -> str:
    """*text* with block comments blanked (lines kept) and line comments removed."""
    text = _BLOCK_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    return _LINE_COMMENT.sub("", text)


def _guards(text: str) -> list[tuple[int, str]]:
    """The 1-based line and counter of every hand-written newest-wins capture."""
    code = _strip_comments(text)
    bumped = {m.group("a") or m.group("b") or m.group("c") for m in _BUMP.finditer(code)}
    found = []
    for m in _CAPTURE.finditer(code):
        x, y = m.group("x"), m.group("y")
        if y not in bumped:
            continue
        ex, ey = re.escape(x), re.escape(y)
        compare = re.compile(
            rf"(?<![\w$.]){ex}\s*[!=]==\s*{ey}(?![\w$.])|(?<![\w$.]){ey}\s*[!=]==\s*{ex}(?![\w$])"
        )
        if compare.search(code):
            found.append((code.count("\n", 0, m.start()) + 1, y))
    return found


def test_no_served_page_keeps_newest_wins_by_hand():
    found = []
    for path in served_sources():
        rel = path.relative_to(SRC).as_posix()
        for line, counter in _guards(path.read_text(encoding="utf-8")):
            if (rel, counter) not in _NOT_A_GUARD:
                found.append(f"{rel}:{line} {counter}")
    assert not found, (
        "These pages keep only the newest of a kind by hand; use createLatestRequest from "
        "/design-system/js/latest-request.js instead:\n  " + "\n  ".join(found)
    )


@pytest.mark.parametrize(("rel", "counter"), sorted(_NOT_A_GUARD))
def test_each_exemption_still_matches(rel, counter):
    path = SRC / rel
    assert path.is_file(), f"{rel} is exempt but does not exist"
    counters = {c for _, c in _guards(path.read_text(encoding="utf-8"))}
    assert counter in counters, f"{rel} is exempt for {counter} but no longer matches"


@pytest.mark.parametrize(
    "snippet",
    [
        "const seq = ++current.pollSeq;\nawait read();\nif (seq !== current.pollSeq) return;",
        "let generation = 0;\ngeneration += 1;\nconst mine = generation;\nif (mine !== generation) {}",
        "this.generation += 1;\nconst g = this.generation;\ng === this.generation;",
        "x++;\nconst t = x;\nt !== x;",
    ],
)
def test_the_pattern_sees_every_form_it_forbids(snippet):
    assert _guards(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "const id = `id-${++idSeq}`;",
        "if (++polls >= MAX) stop();",
        "for (let i = 0; i < n; i++) {}",
        "/**\n * const seq = ++x;\n * seq !== x\n */\nfunction f() {}",
        "const token = latest.begin();\nif (!token.isCurrent()) return;",
    ],
)
def test_the_pattern_leaves_other_counters_alone(snippet):
    assert not _guards(snippet)
