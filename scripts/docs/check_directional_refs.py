#!/usr/bin/env python3
"""Guard: no "see below" next to a link that leaves the page.

The IA restructure moves whole sections onto pages of their own. Prose that
said "as described below" was correct while the section it pointed at sat a few
paragraphs down; once that section becomes its own page, the same sentence
sends the reader looking for something that is no longer there. Sphinx cannot
see this — the cross-reference still resolves, so the build stays green and the
sentence is quietly wrong.

This script pairs every ``:ref:`` and ``:doc:`` role with the words around it.
When the role points at a *different* page and a directional word sits in the
same paragraph, that is a hit. Same-page targets are fine: "below" next to a
link to a section of this very page is exactly right.

Targets are resolved rather than guessed. ``:doc:`` targets are docnames —
absolute (``/a/b``) from ``docs/source`` or relative to the page using them.
``:ref:`` targets are labels, so every ``.. _label:`` line in the tree is read
into a label-to-page map first; a label that map cannot explain is reported
separately as unresolved (Sphinx's ``-W`` build would fail on it too, so there
should be none).

The window is the role's own paragraph — out to the nearest blank line or
non-prose line in each direction — so a directional word cannot bleed across a
paragraph break into a link that has nothing to do with it.

Four deliberate exclusions keep the noise down:

* Code blocks, literal blocks and RST comments are skipped, so "above" in a
  YAML comment or a shell transcript does not fire. A literal block opened
  inside a list item is measured from the item's body, not from its bullet, so
  the rest of the item stays visible.
* ``inline literals`` are blanked before the words are counted: ``above`` is a
  value being quoted, not a direction.
* "above 30" and "below 0.5" are comparisons, so a directional word followed
  directly by a number is ignored. "options below, 2 of them" still counts.
* Only the word list in this module counts. Vaguer phrasing ("further on", "as
  we saw") is left to review; this guard is meant to be trusted, not exhaustive.

One class of false positive remains by design: a sentence whose directional word
points at something genuinely on this page ("both steps above") while a link to
another page sits in the same paragraph. The reader cannot tell which the word
belongs to either, so the fix is to rewrite the sentence — name the section, or
say "earlier steps" — never to drop the fact.

Scope is reStructuredText. ``myst_parser`` is enabled, so a Markdown page would
be legal but would need MyST's ``(target)=`` anchors and fenced code taught to
:func:`collect_labels` and :func:`prose_lines`. Skipping such a page silently
would report every ``:ref:`` pointing into it as a false "unresolved", so
:func:`check` refuses to run at all while one exists.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Words that promise the reader the answer is somewhere on this page.
DIRECTIONAL_WORDS = (
    "below",
    "above",
    "earlier on this page",
    "earlier in this page",
    "the section that follows",
    "later in this guide",
    "later on this page",
    "following section",
    "previous section",
    "next section",
)
DIRECTIONAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in DIRECTIONAL_WORDS) + r")\b",
    re.IGNORECASE,
)
#: "a value above 30" compares, it does not point at a section. The number has
#: to follow directly: "options below, 2 of them" is still a direction.
COMPARISON_RE = re.compile(r"^\s+-?\d")
#: ``some code`` — the reader sees a literal, not a direction.
INLINE_LITERAL_RE = re.compile(r"``.+?``", re.DOTALL)
#: A bullet or enumerator, up to the column its body starts in.
LIST_MARKER_RE = re.compile(r"[ \t]*(?:[-*+]|\d+[.)]|#\.)[ \t]+")

#: ``:ref:`Text <label>``` and ``:doc:`/a/b``` anywhere in the source.
ROLE_RE = re.compile(r":(?P<role>ref|doc):`(?P<body>[^`]*)`", re.DOTALL)
#: ``Text <target>`` inside a role body.
EXPLICIT_TARGET_RE = re.compile(r"^.*<\s*(?P<target>[^<>]+?)\s*>$", re.DOTALL)
#: ``.. _some-label:`` — a page or section anchor.
LABEL_RE = re.compile(r"^\.\.\s+_([^:]+):\s*$")
#: ``.. directive-name::`` with its options and body.
DIRECTIVE_RE = re.compile(r"^\.\.\s+([\w+-]+)::")

#: Directives whose body is verbatim text, not prose.
LITERAL_DIRECTIVES = frozenset(
    {
        "code",
        "code-block",
        "doctest",
        "graphviz",
        "literalinclude",
        "math",
        "mermaid",
        "parsed-literal",
        "program-output",
        "raw",
        "testcode",
        "testoutput",
    }
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index just past the indented block that follows line *start*."""
    index = start
    while index < len(lines) and (not lines[index].strip() or _indent(lines[index]) > indent):
        index += 1
    return index


def prose_lines(text: str) -> list[bool]:
    """One flag per line: is this line prose the reader is meant to read?

    Literal blocks (``.. code-block::`` bodies and anything indented under a
    line ending in ``::``) and RST comments come back ``False``.
    """
    lines = text.splitlines()
    flags = [True] * len(lines)
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        indent = _indent(line)
        directive = DIRECTIVE_RE.match(stripped)
        if directive:
            if directive.group(1) in LITERAL_DIRECTIVES:
                end = _block_end(lines, index + 1, indent)
                for number in range(index, end):
                    flags[number] = False
                index = end
                continue
            index += 1
            continue
        if stripped.startswith("..") and not LABEL_RE.match(stripped):
            # A plain comment: the marker line plus whatever hangs under it.
            end = _block_end(lines, index + 1, indent)
            for number in range(index, end):
                flags[number] = False
            index = end
            continue
        if stripped.endswith("::"):
            # The introducing line is prose; the block it opens is not. Inside a
            # list item the block is indented past the *marker*, not past the
            # bullet, so measure from the item body — otherwise the rest of the
            # item disappears along with the block.
            marker = LIST_MARKER_RE.match(line)
            block_indent = marker.end() if marker else indent
            end = _block_end(lines, index + 1, block_indent)
            for number in range(index + 1, end):
                flags[number] = False
            index = end
            continue
        index += 1
    return flags


def collect_labels(pages: dict[str, str]) -> dict[str, str]:
    """Map every ``.. _label:`` in *pages* (docname to source) to its docname."""
    labels: dict[str, str] = {}
    for docname, text in pages.items():
        for line in text.splitlines():
            match = LABEL_RE.match(line.strip())
            if match:
                # Labels with spaces are written ``.. _`Two Words`:`` — the
                # backticks are quoting, not part of the name.
                labels.setdefault(match.group(1).strip().strip("`"), docname)
    return labels


def _role_target(body: str) -> str:
    """The target of a role body, with any ``Text <...>`` wrapper removed."""
    explicit = EXPLICIT_TARGET_RE.match(body.strip())
    return (explicit.group("target") if explicit else body).strip().replace("\n", " ")


def resolve_doc(target: str, docname: str) -> str:
    """The docname a ``:doc:`` target names, from the page *docname* using it."""
    if target.startswith("/"):
        parts = target[1:].split("/")
    else:
        parts = docname.split("/")[:-1] + target.split("/")
    resolved: list[str] = []
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            if resolved:
                resolved.pop()
            continue
        resolved.append(part)
    return "/".join(resolved)


def paragraph_span(lines: list[str], is_prose: list[bool], number: int) -> tuple[int, int]:
    """The 1-based first and last line of the paragraph holding line *number*.

    A paragraph runs until a blank line or a line the reader does not read as
    prose, which is what keeps a directional word from bleeding across the gap
    into a neighbouring paragraph.
    """
    low = high = number
    while low > 1 and lines[low - 2].strip() and is_prose[low - 2]:
        low -= 1
    while high < len(lines) and lines[high].strip() and is_prose[high]:
        high += 1
    return low, high


def check(docs_root: Path) -> tuple[list[str], str]:
    """Directional-word hits in *docs_root*, plus a one-line summary."""
    docs_root = docs_root.resolve()
    markdown = sorted(path.relative_to(docs_root).as_posix() for path in docs_root.rglob("*.md"))
    if markdown:
        raise SystemExit(
            "check_directional_refs.py reads reStructuredText only, and this tree now has "
            f"{len(markdown)} Markdown page(s): {', '.join(markdown[:5])}"
            f"{' …' if len(markdown) > 5 else ''}.\n"
            "MyST spells anchors `(target)=` and code as fenced blocks, so those pages would "
            "be skipped silently and any :ref: into them misreported as unresolved. Teach "
            "collect_labels() and prose_lines() the MyST shapes, or keep the docs in RST."
        )
    pages = {
        path.relative_to(docs_root).with_suffix("").as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(docs_root.rglob("*.rst"))
    }
    labels = collect_labels(pages)

    findings: list[str] = []
    roles_checked = 0
    for docname, text in sorted(pages.items()):
        lines = text.splitlines()
        is_prose = prose_lines(text)
        shown = f"{docs_root.name}/{docname}.rst"
        try:
            shown = (docs_root / f"{docname}.rst").relative_to(REPO_ROOT).as_posix()
        except ValueError:
            pass
        seen: set[str] = set()
        for match in ROLE_RE.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            if not is_prose[number - 1]:
                continue
            roles_checked += 1
            role = match.group("role")
            target = _role_target(match.group("body"))
            if role == "doc":
                points_at = resolve_doc(target, docname)
                if points_at not in pages:
                    continue  # a broken :doc: link is the -W build's job
            else:
                points_at = labels.get(target)
                if points_at is None:
                    findings.append(f"{shown}:{number}: unresolved :ref:`{target}`")
                    continue
            if points_at == docname:
                continue
            low, high = paragraph_span(lines, is_prose, number)
            for around in range(low, high + 1):
                readable = INLINE_LITERAL_RE.sub(
                    lambda literal: " " * len(literal.group(0)), lines[around - 1]
                )
                for word in DIRECTIONAL_RE.finditer(readable):
                    if COMPARISON_RE.match(readable[word.end() :]):
                        continue
                    line = (
                        f"{shown}:{around}: {word.group(1).lower()} near "
                        f":{role}:`{target}` (target on {points_at})"
                    )
                    if line not in seen:
                        seen.add(line)
                        findings.append(line)
    summary = f"directional refs: OK ({roles_checked} roles checked across {len(pages)} pages)"
    return findings, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check no directional wording sits next to an off-page cross-reference."
    )
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs" / "source")
    args = parser.parse_args(argv)

    findings, summary = check(args.docs_root)
    if findings:
        for finding in findings:
            print(finding)
        print(f"\ndirectional refs FAILED: {len(findings)} finding(s)")
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
