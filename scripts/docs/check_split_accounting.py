#!/usr/bin/env python3
"""Guard: nothing is silently dropped when a docs page is split up.

The IA restructure cuts several large how-to pages into pieces and scatters
them across new reference, architecture and contributing pages. Two things are
easy to lose in that kind of move, and neither shows up in a Sphinx build:
a paragraph that belonged to no destination, and a section heading that ends up
in no page at all. This script reads ``split_table.yml`` and checks both against
the baseline commit the branch started from.

1. Coverage — the ``ranges`` plus ``stays`` of each source must together cover
   every non-blank line of that file as it stood at the baseline.
2. Headings — every H1/H2 the file had at the baseline must appear verbatim as a
   heading in some page under ``docs/source`` today, or be listed in
   ``retired_headings`` as a deliberate drop.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RULE_CHARS = set("=-~^*#")


def _is_rule(line: str) -> bool:
    text = line.strip()
    return len(text) >= 2 and len(set(text)) == 1 and text[0] in RULE_CHARS


def parse_headings(text: str) -> list[tuple[int, str, int]]:
    """Every RST section title in *text* as ``(level, title, line number)``.

    Levels follow RST's own rule: the order in which underline styles first
    appear in the file decides the depth, so H1/H2 are the first two styles
    seen rather than any fixed character. Both underlined and overlined+
    underlined titles are recognised.
    """
    lines = text.splitlines()
    styles: list[tuple[str, bool]] = []
    found: list[tuple[int, str, int]] = []
    index = 0
    while index < len(lines):
        first = lines[index]
        second = lines[index + 1] if index + 1 < len(lines) else ""
        third = lines[index + 2] if index + 2 < len(lines) else ""
        if (
            _is_rule(first)
            and second.strip()
            and _is_rule(third)
            and first.strip()[0] == third.strip()[0]
            and len(first.strip()) >= len(second.rstrip())
        ):
            title, style, number, step = second, (first.strip()[0], True), index + 2, 3
        elif first.strip() and _is_rule(second) and len(second.strip()) >= len(first.rstrip()):
            title, style, number, step = first, (second.strip()[0], False), index + 1, 2
        else:
            index += 1
            continue
        if style not in styles:
            styles.append(style)
        found.append((styles.index(style) + 1, title.strip(), number))
        index += step
    return found


def _span(text: str) -> set[int]:
    start, _, end = str(text).partition("-")
    return set(range(int(start), int(end or start) + 1))


def _summarize(numbers: list[int]) -> str:
    spans: list[list[int]] = []
    for number in numbers:
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def check(
    table_path: Path,
    docs_root: Path,
    ref: str | None = None,
    only: list[str] | None = None,
) -> tuple[dict[str, list[str]], str]:
    table = yaml.safe_load(table_path.read_text(encoding="utf-8")) or {}
    ref = ref or table["baseline_ref"]
    docs_root = docs_root.resolve()
    repo_root = Path(_git(docs_root, "rev-parse", "--show-toplevel").stdout.strip() or docs_root)
    prefix = docs_root.relative_to(repo_root.resolve()).as_posix()

    present = set()
    for page in sorted(docs_root.rglob("*.rst")):
        present.update(title for _, title, _ in parse_headings(page.read_text(encoding="utf-8")))
    retired = set(table.get("retired_headings") or [])

    problems: dict[str, list[str]] = {}
    sources = table.get("sources") or {}
    if only:
        unknown = sorted(set(only) - set(sources))
        if unknown:
            raise SystemExit(f"--only names sources not in the table: {unknown}")
        sources = {relative: sources[relative] for relative in only}
    checked = 0
    for relative, spec in sources.items():
        shown = _git(repo_root, "show", f"{ref}:{prefix}/{relative}")
        if shown.returncode != 0:
            problems[relative] = [f"cannot read {ref}:{prefix}/{relative}: {shown.stderr.strip()}"]
            continue
        lines = shown.stdout.splitlines()
        covered: set[int] = set()
        for item in spec.get("ranges") or []:
            covered |= _span(item["lines"])
        for stay in spec.get("stays") or []:
            covered |= _span(stay)
        messages = []
        gaps = [n for n, line in enumerate(lines, 1) if line.strip() and n not in covered]
        if gaps:
            messages.append(
                f"{len(gaps)} non-blank line(s) assigned to nothing: {_summarize(gaps)}"
            )
        past_end = sorted(n for n in covered if n > len(lines))
        if past_end:
            messages.append(
                f"range runs past the end of the file ({len(lines)} lines): {_summarize(past_end)}"
            )
        for level, title, number in parse_headings(shown.stdout):
            checked += 1 if level <= 2 else 0
            if level <= 2 and title not in present and title not in retired:
                messages.append(
                    f"line {number}: H{level} {title!r} survives in no page and is not retired"
                )
        if messages:
            problems[relative] = messages
    summary = f"split accounting: OK ({len(sources)} sources, {checked} headings checked, {len(retired)} retired)"
    return problems, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the docs split table accounts for every line and heading."
    )
    parser.add_argument(
        "--table", type=Path, default=REPO_ROOT / "scripts" / "docs" / "split_table.yml"
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="baseline commit to read old pages from (overrides baseline_ref)",
    )
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs" / "source")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SOURCE",
        help="check only these table sources (old paths relative to docs/source)",
    )
    args = parser.parse_args(argv)

    problems, summary = check(args.table, args.docs_root, args.ref, args.only)
    if problems:
        for relative, messages in problems.items():
            print(f"{relative}:")
            for message in messages:
                print(f"  {message}")
        print(f"\nsplit accounting FAILED for {len(problems)} source file(s)")
        return 1
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
