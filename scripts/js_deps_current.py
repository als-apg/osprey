#!/usr/bin/env python3
"""Exit 0 when ``node_modules`` holds exactly what ``package-lock.json`` pins.

npm records the installed tree in ``node_modules/.package-lock.json``. The tree
is current when every entry there equals the same entry of the lockfile, and
every lockfile entry that is not ``optional`` is installed: npm leaves the
optional platform packages of other hosts out. Otherwise exit 1 with one line
naming the first package that differs. Stdlib only, so it runs before any venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = "package-lock.json"
HIDDEN = "node_modules/.package-lock.json"


def first_difference(root: Path) -> str | None:
    """Return a line naming the first mismatch, or ``None`` when current."""
    hidden_path = root / HIDDEN
    if not hidden_path.is_file():
        return f"{HIDDEN} is absent"
    locked = json.loads((root / LOCKFILE).read_text(encoding="utf-8")).get("packages", {})
    installed = json.loads(hidden_path.read_text(encoding="utf-8")).get("packages", {})
    for key in sorted(installed):
        if installed[key] != locked.get(key):
            return f"{key} differs from {LOCKFILE}"
    for key in sorted(locked):
        if key and key not in installed and not locked[key].get("optional"):
            return f"{key} is in {LOCKFILE} but not installed"
    return None


def main() -> int:
    difference = first_difference(ROOT)
    if difference is None:
        return 0
    print(f"node_modules is stale: {difference}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
