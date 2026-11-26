#!/usr/bin/env python3
"""Record the application contract's surface under the number this OSPREY implements.

``tests/apps/contract/surface.json`` holds, for each contract number, the names
an application may rely on (``osprey.apps.contract.surface()``). The guard test
``tests/apps/test_contract_surface.py`` compares the live surface with the entry
for ``APP_CONTRACT``. This script is the only writer of that file and the one
statement of its rule: a number's record only grows. It refuses to drop a name
from an existing number's record, or to change a recorded manifest type: either
breaks apps written for that number, so raise ``APP_CONTRACT``, add the history
row, and record again. Entries of other numbers are kept as they are.

Run as ``uv run python scripts/record_app_contract.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The record, relative to the repository root.
RECORD = Path("tests/apps/contract/surface.json")


class RecordRefused(Exception):
    """Recording would take a name away from an existing number."""


def entries(section: Any) -> dict[str, str | None]:
    """A section as ``{name: type}``; a list section's names carry no type."""
    if isinstance(section, dict):
        return dict(section)
    return dict.fromkeys(section)


def breaks(live: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """What ``live`` takes away from ``recorded``: one line per gone or retyped name."""
    lines = []
    for section, names in recorded.items():
        have = entries(live.get(section, []))
        for name, kind in entries(names).items():
            if name not in have:
                lines.append(f"{section}: {name} is gone")
            elif have[name] != kind:
                lines.append(f"{section}: {name} changed type {kind} -> {have[name]}")
    return lines


def record(existing: dict[str, Any], number: int, live: dict[str, Any]) -> dict[str, Any]:
    """The record with ``live`` stored under ``number``.

    Args:
        existing: The record as read, ``{"<N>": surface}``.
        number: The contract number to record ``live`` under.
        live: The surface to record.

    Returns:
        A new record, numbers in ascending order.

    Raises:
        RecordRefused: If ``live`` drops or retypes a name ``existing`` holds
            for ``number``.
    """
    key = str(number)
    lost = breaks(live, existing.get(key, {}))
    if lost:
        raise RecordRefused(
            f"contract {number} would lose names apps rely on:\n  "
            + "\n  ".join(lost)
            + "\nRaise APP_CONTRACT, add the history row, re-record."
        )
    merged = {**existing, key: live}
    return {k: merged[k] for k in sorted(merged, key=int)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--record", type=Path, default=REPO_ROOT / RECORD, help="the record file to write"
    )
    args = parser.parse_args(argv)

    from osprey.apps.contract import APP_CONTRACT, surface

    path: Path = args.record
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    try:
        updated = record(existing, APP_CONTRACT, surface())
    except RecordRefused as refused:
        print(refused, file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"recorded contract {APP_CONTRACT} in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
