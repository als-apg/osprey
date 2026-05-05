#!/usr/bin/env python3
"""Extract bash code blocks from a Sphinx RST file as a runnable script.

Filters:
  - Code blocks inside a ``.. tab-item::`` whose label does NOT contain
    "linux" (case-insensitive) are skipped. Blocks outside any tab-item
    are always included.
  - Code blocks whose first non-blank body line is exactly ``# skip-ci``
    are skipped (per-block opt-out).

Output: a bash script on stdout, with a ``set -euxo pipefail`` preamble
and ``# === <basename>:<line> ===`` headers per emitted block.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TAB_ITEM_RE = re.compile(r"^(?P<indent>\s*)\.\. tab-item::\s*(?P<label>.+?)\s*$")
CODE_BLOCK_RE = re.compile(r"^(?P<indent>\s*)\.\. code(?:-block)?::\s*bash\s*$")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def extract(path: Path) -> str:
    lines = path.read_text().splitlines()
    out: list[str] = ["#!/usr/bin/env bash", "set -euxo pipefail", ""]
    tab_stack: list[tuple[int, str]] = []  # (indent, label)

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped:
            cur = indent_of(line)
            while tab_stack and cur <= tab_stack[-1][0]:
                tab_stack.pop()

        m = TAB_ITEM_RE.match(line)
        if m:
            tab_stack.append((len(m.group("indent")), m.group("label")))
            i += 1
            continue

        m = CODE_BLOCK_RE.match(line)
        if m:
            directive_indent = len(m.group("indent"))
            block_line_no = i + 1
            i += 1
            while i < n:
                cur_line = lines[i]
                cs = cur_line.strip()
                if not cs:
                    i += 1
                    continue
                if cs.startswith(":") and indent_of(cur_line) > directive_indent:
                    i += 1
                    continue
                break
            if i >= n:
                break
            body_indent = indent_of(lines[i])
            if body_indent <= directive_indent:
                continue
            body: list[str] = []
            while i < n:
                cur_line = lines[i]
                if not cur_line.strip():
                    body.append("")
                    i += 1
                    continue
                if indent_of(cur_line) < body_indent:
                    break
                body.append(cur_line[body_indent:])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            if not body:
                continue
            in_non_linux_tab = bool(tab_stack) and "linux" not in tab_stack[-1][1].lower()
            if in_non_linux_tab:
                continue
            first = next((b for b in body if b.strip()), "")
            if first.strip() == "# skip-ci":
                continue
            out.append(f"# === {path.name}:{block_line_no} ===")
            out.extend(body)
            out.append("")
            continue

        i += 1

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract bash blocks from RST docs.")
    parser.add_argument("path", type=Path, help="Path to an RST file")
    args = parser.parse_args()
    sys.stdout.write(extract(args.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
