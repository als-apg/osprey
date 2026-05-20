"""Translate family-prefixed device segments in unified benchmark queries to bare-numeric.

One-shot importer for cross_paradigm/queries/tier{1,2,3}_queries.json — applies the
schema-cleanup convention from commit 30a4af8e (device segment digits-only) to the
paper-repo-style queries that still carry family prefixes.

PV format: RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD (6 segments, ':'-delimited).
Only segment index 3 (DEVICE) is translated; other segments are left untouched.

Translation rules for DEVICE:
  1. Explicit override map (handles non-regex cases: MAIN, sector-side gauges).
  2. Otherwise, match ^[A-Za-z]+(\\d+)$ and replace with f"{int(digits):02d}".
     This covers BPM01, B24, H18, V05, QF03, QD08, SF03, SD12, GAMM01, NEUT01,
     C1, C2, K1, K2, etc.
  3. If neither matches and the segment is already bare-numeric (^\\d+$), leave
     it. This makes the script idempotent.

Idempotent: running on already-translated input is a no-op.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

QUERIES_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/osprey/templates/apps/control_assistant"
    / "data/benchmarks/cross_paradigm/queries"
)

OVERRIDES: dict[str, str] = {
    "MAIN": "01",
    "SR01A": "01",
    "SR01B": "02",
    "SR02A": "03",
    "SR02B": "04",
    "SR03A": "05",
    "SR03B": "06",
}

FAMILY_NUM_RE = re.compile(r"^[A-Za-z]+(\d+)$")
BARE_NUM_RE = re.compile(r"^\d+$")


def translate_device(segment: str) -> str:
    if segment in OVERRIDES:
        return OVERRIDES[segment]
    m = FAMILY_NUM_RE.match(segment)
    if m:
        return f"{int(m.group(1)):02d}"
    if BARE_NUM_RE.match(segment):
        return segment
    raise ValueError(f"Cannot translate device segment: {segment!r}")


def translate_pv(pv: str) -> str:
    parts = pv.split(":")
    if len(parts) != 6:
        raise ValueError(f"Expected 6-segment PV, got {len(parts)}: {pv!r}")
    parts[3] = translate_device(parts[3])
    return ":".join(parts)


def translate_file(path: Path) -> tuple[int, int]:
    queries = json.loads(path.read_text())
    changed = 0
    total = 0
    for q in queries:
        new_pvs = []
        for pv in q["targeted_pv"]:
            total += 1
            new = translate_pv(pv)
            if new != pv:
                changed += 1
            new_pvs.append(new)
        q["targeted_pv"] = new_pvs
    path.write_text(json.dumps(queries, indent=2) + "\n")
    return changed, total


def main() -> int:
    for tier in (1, 2, 3):
        path = QUERIES_DIR / f"tier{tier}_queries.json"
        changed, total = translate_file(path)
        print(f"tier{tier}: translated {changed}/{total} PVs in {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
