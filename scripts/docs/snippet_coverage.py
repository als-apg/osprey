#!/usr/bin/env python3
"""Count the code blocks on every docs page, and which of them a test reads.

The goal for the documentation is that every code block a reader copies is
read by a test. This script measures how far off that is, page by page, and
keeps the number from growing.

What counts as a block
----------------------
Every ``*.rst`` file under ``docs/source`` is scanned for:

* ``.. code-block::``, ``.. code::`` and ``.. sourcecode::`` -- the language is
  the directive argument, ``text`` when there is none;
* ``.. literalinclude::`` -- language ``literalinclude``; the included path is
  resolved against the page;
* a paragraph ending in ``::`` -- language ``literal``.

Directive options (``:caption:``, ``:linenos:``) are not content. A block ends at
the first non-blank line indented no deeper than its opener, so a block inside a
``.. note::`` or a ``.. tab-item::`` closes where that directive's body does.

What counts as tested
---------------------
``snippet_coverage.yml`` (beside this script) is the ledger: one claim per test
that reads doc blocks, naming the pages (globs allowed), the block languages it
reads, how it reads them (``kind``) and, optionally, a text the block must
contain. A block is tested when a claim matches its page and language. A
``literalinclude`` block is tested when a ``literalinclude`` claim matches its
page or the file it includes. Claims under ``pending:`` belong to tests on open
pull requests. Until the test is on disk the report shows what it would cover;
once it is (the file exists, or holds the claim's ``marker``) it counts as
tested, so the PRs can merge in any order. A tidy-up commit after they merge
moves the claims to ``tested:`` and regenerates the baseline.

The ratchet
-----------
``snippet_coverage_baseline.json`` holds each page's allowed number of untested
blocks. ``--check`` fails when a page has more untested blocks than its entry
(or any, for a page not in the file), and when an entry is larger than the
page's count today, so the allowance can only shrink -- unless a landed pending
claim reads that page, which is only noted. It also fails on a claim whose test
is missing, never names the page it claims, or reads no block. ``--update-baseline``
rewrites the file from today's counts.

Usage::

    uv run python scripts/docs/snippet_coverage.py                 # Markdown report
    uv run python scripts/docs/snippet_coverage.py --page how-to/build-profiles.rst
    uv run python scripts/docs/snippet_coverage.py --json
    uv run python scripts/docs/snippet_coverage.py --check
    uv run python scripts/docs/snippet_coverage.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).resolve().with_name("snippet_coverage.yml")
BASELINE = Path(__file__).resolve().with_name("snippet_coverage_baseline.json")
DOCS_SOURCE = Path("docs") / "source"
SCRIPT = "scripts/docs/snippet_coverage.py"

#: Strongest first. A page's ``kind`` column shows the weakest claim on it.
KINDS = ("runs", "checks", "pins")

_CODE_OPENER = re.compile(r"^(\s*)\.\. (?:code-block|code|sourcecode)::[ \t]*(\S*)")
_INCLUDE_OPENER = re.compile(r"^(\s*)\.\. literalinclude::[ \t]*(\S+)")
#: A line ending in ``::`` that is not a directive opens a literal block.
_LITERAL_OPENER = re.compile(r"^(\s*)(?!\s)(?!\.\.(?:\s|$)).*::\s*$")


@dataclass(frozen=True)
class Block:
    """One code block on one page."""

    page: str
    """Path relative to ``docs/source``, with forward slashes."""
    line: int
    """First content line (the directive line for a ``literalinclude``)."""
    language: str
    include: str | None = None
    """For a ``literalinclude``: the included file, relative to ``docs/source``."""
    text: str = ""
    """The content, dedented; for a ``literalinclude``, the included file's text."""


@dataclass(frozen=True)
class Claim:
    """One ledger entry: a test that reads some blocks."""

    test: str
    pages: tuple[str, ...]
    languages: tuple[str, ...]
    kind: str
    what: str
    contains: str | None = None
    pr: int | None = None
    marker: str | None = None

    def matches(self, block: Block) -> bool:
        if block.language not in self.languages:
            return False
        if self.contains is not None and self.contains not in block.text:
            return False
        targets = [block.page] if block.include is None else [block.page, block.include]
        return any(glob_match(pattern, target) for pattern in self.pages for target in targets)


@dataclass
class PageCoverage:
    """One page's blocks, split by whether a claim reads them."""

    page: str
    blocks: list[Block] = field(default_factory=list)
    tested: list[tuple[Block, list[Claim]]] = field(default_factory=list)
    untested: list[Block] = field(default_factory=list)
    pending: list[Block] = field(default_factory=list)
    """Untested blocks a pending claim would cover once its PR merges."""

    @property
    def weakest_kind(self) -> str:
        kinds = {claim.kind for _, claims in self.tested for claim in claims}
        return next((kind for kind in reversed(KINDS) if kind in kinds), "-")


# -- scanning ------------------------------------------------------------------


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _dedent(lines: list[str]) -> str:
    width = min((_indent(line) for line in lines if line.strip()), default=0)
    return "\n".join(line[width:] for line in lines)


def scan_page(text: str, page: str, docs_source: Path | None = None) -> list[Block]:
    """Every block on one page, in page order."""
    lines = text.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        code = _CODE_OPENER.match(line)
        include = None if code else _INCLUDE_OPENER.match(line)
        literal = None if code or include else _LITERAL_OPENER.match(line)
        opener = code or include or literal
        index += 1
        if opener is None:
            continue
        opener_line = index
        base = len(opener.group(1))
        if not literal:
            # Directive options, right under the opener, are not content.
            while (
                index < len(lines)
                and _indent(lines[index]) > base
                and lines[index].strip().startswith(":")
            ):
                index += 1
        content: list[tuple[int, str]] = []
        while index < len(lines):
            current = lines[index]
            if current.strip() and _indent(current) <= base:
                break
            content.append((index + 1, current))
            index += 1
        while content and not content[-1][1].strip():
            content.pop()
        while content and not content[0][1].strip():
            content.pop(0)
        if include:
            target = posixpath.normpath(posixpath.join(posixpath.dirname(page), include.group(2)))
            body = ""
            if docs_source is not None and (docs_source / target).is_file():
                body = (docs_source / target).read_text(encoding="utf-8", errors="replace")
            blocks.append(Block(page, opener_line, "literalinclude", target, body))
            continue
        if not content:
            continue
        first = content[0][0]
        language = (code.group(2).lower() or "text") if code else "literal"
        blocks.append(Block(page, first, language, None, _dedent([t for _, t in content])))
    return blocks


def scan_tree(docs_source: Path) -> dict[str, list[Block]]:
    """Every page under ``docs_source`` mapped to its blocks (an empty list if none)."""
    found: dict[str, list[Block]] = {}
    for path in sorted(docs_source.rglob("*.rst")):
        page = path.relative_to(docs_source).as_posix()
        found[page] = scan_page(path.read_text(encoding="utf-8"), page, docs_source)
    return found


# -- the ledger ----------------------------------------------------------------


def glob_match(pattern: str, path: str) -> bool:
    """``fnmatch`` with ``/``-aware wildcards: ``*`` stays in one directory, ``**`` spans any."""
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            regex += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return re.fullmatch(regex, path) is not None


def _claims(entries: list[dict] | None, *, pending: bool) -> list[Claim]:
    return [
        Claim(
            test=entry["test"],
            pages=tuple(entry["pages"]),
            languages=tuple(entry["languages"]),
            kind=entry["kind"],
            what=entry["what"],
            contains=entry.get("contains"),
            pr=entry.get("pr") if pending else None,
            marker=entry.get("marker") if pending else None,
        )
        for entry in entries or []
    ]


def load_ledger(path: Path = LEDGER) -> tuple[list[Claim], list[Claim]]:
    """``(live claims, pending claims)`` from the ledger file."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _claims(document.get("tested"), pending=False), _claims(
        document.get("pending"), pending=True
    )


# -- coverage ------------------------------------------------------------------


def coverage(
    tree: dict[str, list[Block]], live: list[Claim], pending: list[Claim]
) -> list[PageCoverage]:
    """Per-page coverage, worst first (most untested, then most blocks, then name)."""
    pages = []
    for page, blocks in tree.items():
        row = PageCoverage(page, list(blocks))
        for block in blocks:
            claims = [claim for claim in live if claim.matches(block)]
            if claims:
                row.tested.append((block, claims))
                continue
            row.untested.append(block)
            if any(claim.matches(block) for claim in pending):
                row.pending.append(block)
        pages.append(row)
    pages.sort(key=lambda row: (-len(row.untested), -len(row.blocks), row.page))
    return pages


def _lines_by_language(blocks: list[Block]) -> str:
    grouped: dict[str, list[int]] = {}
    for block in blocks:
        grouped.setdefault(block.language, []).append(block.line)
    return "; ".join(
        f"{language} {', '.join(map(str, numbers))}"
        for language, numbers in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    )


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def totals_line(rows: list[PageCoverage]) -> str:
    blocks = sum(len(row.blocks) for row in rows)
    tested = sum(len(row.tested) for row in rows)
    untested = sum(len(row.untested) for row in rows)
    pending = sum(len(row.pending) for row in rows)
    with_blocks = sum(1 for row in rows if row.blocks)
    covered = sum(1 for row in rows if row.blocks and not row.untested)
    return (
        f"{blocks} blocks on {with_blocks} pages: {tested} tested ({_percent(tested, blocks)}), "
        f"{untested} untested ({pending} of those read by a test on an open PR); "
        f"{covered} of {with_blocks} pages fully tested."
    )


def render_markdown(rows: list[PageCoverage], *, narrowed: bool = False) -> str:
    """The human report: totals, a per-language table and a per-page table, worst first."""
    shown = [row for row in rows if row.blocks]
    out = ["# Docs snippet coverage", "", totals_line(rows), ""]
    if not narrowed:
        empty = len(rows) - len(shown)
        out += [
            f"Pages with no code blocks: {empty} (not listed).",
            "",
            "A block is *tested* when a claim in `scripts/docs/snippet_coverage.yml` names its "
            "page and language. *Kind* is the weakest claim on the page: `runs` (executed, or "
            "loaded by the code that owns it) > `checks` (parsed, names resolved) > `pins` "
            "(text compared with the code). *Pending* counts untested blocks a test on an open "
            "pull request reads. *Untested blocks* gives the first content line of each, by "
            "language.",
            "",
            "## By language",
            "",
            "| Language | Blocks | Tested | Untested |",
            "|---|---:|---:|---:|",
        ]
        languages: dict[str, list[int]] = {}
        for row in shown:
            for block, _ in row.tested:
                languages.setdefault(block.language, [0, 0])[0] += 1
            for block in row.untested:
                languages.setdefault(block.language, [0, 0])[1] += 1
        for language, (tested, untested) in sorted(
            languages.items(), key=lambda item: (-sum(item[1]), item[0])
        ):
            out.append(f"| {language} | {tested + untested} | {tested} | {untested} |")
        out += ["", "## By page, most untested first", ""]
    out += [
        "| Page | Blocks | Tested | Untested | Pending | Kind | Untested blocks (language: lines) |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in shown:
        out.append(
            f"| {row.page} | {len(row.blocks)} | {len(row.tested)} | {len(row.untested)} "
            f"| {len(row.pending) or ''} | {row.weakest_kind} | {_lines_by_language(row.untested)} |"
        )
    if narrowed:
        for row in shown:
            if row.tested:
                out += ["", f"Tested on {row.page}:", ""]
                for block, claims in row.tested:
                    names = ", ".join(f"{claim.test} ({claim.kind})" for claim in claims)
                    out.append(f"- line {block.line} {block.language}: {names}")
    return "\n".join(out) + "\n"


def render_json(rows: list[PageCoverage]) -> str:
    return json.dumps(
        {
            "totals": {
                "blocks": sum(len(row.blocks) for row in rows),
                "tested": sum(len(row.tested) for row in rows),
                "untested": sum(len(row.untested) for row in rows),
                "pending": sum(len(row.pending) for row in rows),
            },
            "pages": [
                {
                    "page": row.page,
                    "blocks": len(row.blocks),
                    "tested": len(row.tested),
                    "untested": len(row.untested),
                    "pending": len(row.pending),
                    "kind": row.weakest_kind,
                    "untested_blocks": [
                        {"line": block.line, "language": block.language, "include": block.include}
                        for block in row.untested
                    ],
                }
                for row in rows
                if row.blocks
            ],
        },
        indent=2,
    )


# -- the ratchet ---------------------------------------------------------------


def load_baseline(path: Path = BASELINE) -> dict[str, int]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["untested"])


def write_baseline(rows: list[PageCoverage], path: Path = BASELINE) -> None:
    untested = {row.page: len(row.untested) for row in rows if row.untested}
    document = {
        "_what": (
            "Untested code blocks each docs page may have. Only shrinks: "
            f"see {SCRIPT} --check. Regenerate with --update-baseline."
        ),
        "untested": dict(sorted(untested.items())),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _count(n: int) -> str:
    return f"{n} untested block{'' if n == 1 else 's'}"


def ratchet_problems(
    rows: list[PageCoverage], baseline: dict[str, int], explained: frozenset[str] = frozenset()
) -> tuple[list[str], list[str]]:
    """``(failures, notes)``: pages over their allowance, and allowances above need.

    An allowance above need fails (STALE) unless a pending claim whose test has
    landed reads a block on that page: then the drop is that PR's, and the
    tidy-up commit that promotes the claim shrinks the allowance.
    """
    failures: list[str] = []
    notes: list[str] = []
    current = {row.page: row for row in rows}
    for row in rows:
        allowed = baseline.get(row.page, 0)
        if len(row.untested) > allowed:
            allowance = (
                f"the baseline allows {allowed}"
                if row.page in baseline
                else "the page is not in the baseline, so it may have none"
            )
            failures.append(
                f"GREW   {row.page}: {_count(len(row.untested))}, {allowance}. "
                f"Untested now: {_lines_by_language(row.untested)}."
            )
    for page, allowed in sorted(baseline.items()):
        row = current.get(page)
        now = len(row.untested) if row else 0
        if allowed > now:
            gone = "" if row else " (the page no longer exists)"
            line = f"the baseline allows {_count(allowed)}, the page has {now}{gone}."
            if page in explained:
                notes.append(f"SHRINK {page}: {line} A landed pending claim explains it.")
            else:
                failures.append(f"STALE  {page}: {line}")
    return failures, notes


FIX_GREW = f"""\
A page with a GREW line has more code blocks that no test reads than before.
Either:
  1. add or extend a test under tests/docs that reads the new block (load it,
     run it, or compare it with the code), and add a claim for it to
     scripts/docs/snippet_coverage.yml (test, pages, languages, kind, what).
     If your PR already has a claim under `pending:` that should cover the
     block, check its pages, languages and `contains`, and that its test file
     (or `marker:`) is on disk; or
  2. if the block cannot be tested (it needs Docker, a network install, an LLM),
     run `uv run python {SCRIPT} --update-baseline` and say why in the PR, so a
     reviewer sees the allowance grow in the diff."""

FIX_STALE = f"""\
A page with a STALE line has fewer untested blocks than the baseline allows --
good news. Shrink the allowance so it stays shrunk:
  uv run python {SCRIPT} --update-baseline
and commit scripts/docs/snippet_coverage_baseline.json."""


def ledger_problems(live: list[Claim], tree: dict[str, list[Block]], root: Path) -> list[str]:
    """A claim must name a test that exists, names the pages it claims, and reads a block.

    "Names the pages" is a cheap honesty guard: each page pattern's stem (the
    last path part without glob characters or suffix; ``docs/source`` for a
    ``**`` pattern) must appear in the test file's text.
    """
    problems = []
    for claim in live:
        path = root / claim.test
        if not path.is_file():
            problems.append(f"LEDGER {claim.test}: no such test file (moved or deleted?).")
        else:
            text = path.read_text(encoding="utf-8")
            for pattern in claim.pages:
                needle = page_stem(pattern)
                if needle not in text:
                    problems.append(
                        f"LEDGER {claim.test}: claims {pattern}, but the test never names "
                        f"{needle!r}."
                    )
        if claim.kind not in KINDS:
            problems.append(f"LEDGER {claim.test}: kind {claim.kind!r} is not one of {KINDS}.")
        if not any(claim.matches(block) for blocks in tree.values() for block in blocks):
            problems.append(
                f"LEDGER {claim.test}: the claim ({', '.join(claim.pages)}; "
                f"{', '.join(claim.languages)}) matches no block on any page."
            )
    return problems


def page_stem(pattern: str) -> str:
    """What a test reading ``pattern`` must spell: see :func:`ledger_problems`."""
    if "**" in pattern:
        return "docs/source"
    for part in reversed(pattern.split("/")):
        if not any(char in part for char in "*?["):
            return posixpath.splitext(part)[0]
    return "docs/source"


def landed(claim: Claim, root: Path) -> bool:
    """Whether a pending claim's test is on disk under ``root`` (with its ``marker``, if any)."""
    path = root / claim.test
    if not path.is_file():
        return False
    return claim.marker is None or claim.marker in path.read_text(encoding="utf-8")


def pending_problems(pending: list[Claim], root: Path) -> list[str]:
    """Notes, not failures: a pending claim whose test has landed can move up to ``tested:``.

    A claim for a new test file has landed when the file exists; a claim for a
    file the PR widens, when the file contains the claim's ``marker``. A landed
    claim already counts as tested, so nothing fails; the tidy-up commit after
    the PRs merge moves the claims and regenerates the baseline.
    """
    notes = []
    for claim in pending:
        if not landed(claim, root):
            continue
        how = "the file exists" if claim.marker is None else f"the file contains {claim.marker!r}"
        notes.append(
            f"PROMOTE {claim.test} (PR #{claim.pr}): {how}; it counts as tested. Move the claim "
            f"to `tested:` (drop `pr:` and `marker:`) and run --update-baseline in the tidy-up."
        )
    return notes


@dataclass
class Result:
    """Everything one run measures."""

    tree: dict[str, list[Block]]
    rows: list[PageCoverage]
    failures: list[str]
    notes: list[str]


def evaluate(root: Path, ledger: Path = LEDGER, baseline: Path = BASELINE) -> Result:
    """Scan, apply the ledger (landed pending claims count) and judge the ratchet.

    The command line and the test both call this, so they agree.
    """
    tree = scan_tree(root / DOCS_SOURCE)
    live, pending = load_ledger(ledger)
    arrived = [claim for claim in pending if landed(claim, root)]
    waiting = [claim for claim in pending if claim not in arrived]
    rows = coverage(tree, live + arrived, waiting)
    explained = frozenset(
        page
        for page, blocks in tree.items()
        if any(claim.matches(block) for claim in arrived for block in blocks)
    )
    failures, notes = ratchet_problems(rows, load_baseline(baseline), explained)
    failures = ledger_problems(live + arrived, tree, root) + failures
    notes = pending_problems(pending, root) + notes
    return Result(tree, rows, failures, notes)


# -- command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--page", help="report one page (path relative to docs/source)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--check", action="store_true", help="fail if any page's count grew")
    parser.add_argument("--update-baseline", action="store_true", help="rewrite the baseline")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository to scan")
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)

    result = evaluate(args.root, args.ledger, args.baseline)
    rows = result.rows

    if args.update_baseline:
        write_baseline(rows, args.baseline)
        print(f"wrote {args.baseline}: {totals_line(rows)}")
        return 0
    if args.check:
        for note in result.notes:
            print(f"note: {note}")
        if not result.failures:
            print(f"snippet coverage: ok. {totals_line(rows)}")
            return 0
        print(f"snippet coverage: {len(result.failures)} problem(s)\n")
        print("\n".join(result.failures))
        if any(p.startswith("GREW") for p in result.failures):
            print("\n" + FIX_GREW)
        if any(p.startswith("STALE") for p in result.failures):
            print("\n" + FIX_STALE)
        return 1
    if args.page:
        rows = [row for row in rows if row.page == args.page]
        if not rows:
            print(f"no page {args.page!r} under {args.root / DOCS_SOURCE}", file=sys.stderr)
            return 2
    print(render_json(rows) if args.json else render_markdown(rows, narrowed=bool(args.page)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
