#!/usr/bin/env python3
"""Print one release's section of a Keep-a-Changelog file, for a GitHub Release body.

    python3 scripts/ci/release_notes.py <CHANGELOG.md> <version> > release_notes.md

The section is the ``## [<version>]`` heading, matched literally (a version's
dots are not wildcards), and every line up to the next ``## [`` heading or the
end of the file. Trailing blank lines are dropped and nothing else is: a
section that is the last one in the file keeps its last bullet. A version with
no section prints ``See <CHANGELOG.md> for details.`` instead, so the release
still gets a body.

Shared by release.yml (the framework's CHANGELOG.md) and release-connectors.yml
(packages/osprey-connectors/CHANGELOG.md). Stdlib-only.
"""

from __future__ import annotations

import sys
from pathlib import Path


def section(text: str, version: str) -> list[str]:
    """The lines of *version*'s section, heading included; empty when absent."""
    heading = f"## [{version}]"
    found: list[str] = []
    for line in text.splitlines():
        if found:
            if line.startswith("## ["):
                break
            found.append(line)
        elif line.startswith(heading):
            found.append(line)
    while found and not found[-1].strip():
        found.pop()
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: release_notes.py <CHANGELOG.md> <version>", file=sys.stderr)
        return 2
    changelog, version = args
    lines = section(Path(changelog).read_text(encoding="utf-8"), version)
    print("\n".join(lines) if lines else f"See {changelog} for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
