"""The demo's ``limits.yaml``: three records, each teaching one limits shape.

The demo limits three setpoints and leaves every other channel to the
deployment's limits mode. Each record is written under a one-line comment
naming what it teaches:

* a corrector's symmetric range;
* the wired RF cavity's range plus its largest single step, in MHz;
* a setpoint blocked outright.

Run as a script, this module writes the limits golden: each record resolved as
the schema resolves an absent slot (``writable`` true on a setpoint carrying
both bounds, false otherwise; ``confirm`` true).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

#: The golden this module writes, relative to the repository root.
GOLDEN = "tests/facility/golden/limits.json"

#: This module, as the golden's ``_sources`` names it.
SOURCE = "scripts/facility_demo/_limits.py"

#: ``(comment, record)`` per record, in file order.
RECORDS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "Symmetric range: the corrector current stays within [-12, 12].",
        {"address": "SR:MAG:HCM:01:CURRENT:SP", "min_value": -12.0, "max_value": 12.0},
    ),
    (
        "Range plus largest single step: the cavity frequency in MHz.",
        {
            "address": "SR:RF:CAVITY:01:FREQUENCY:SP",
            "min_value": 500.0,
            "max_value": 500.8,
            "max_step": 0.01,
        },
    ),
    (
        "Blocked outright: no write reaches this setpoint.",
        {"address": "SR:VAC:ION-PUMP:01:VOLTAGE:SP", "writable": False},
    ),
)


def text() -> str:
    """``limits.yaml``'s text: the records, each under its comment."""
    lines = ["records:"]
    for comment, record in RECORDS:
        body = yaml.safe_dump([record], sort_keys=False, default_flow_style=False)
        lines.append(f"# {comment}")
        lines.extend(body.rstrip("\n").splitlines())
    return "\n".join(lines) + "\n"


def resolve(record: dict[str, Any], role: str) -> dict[str, Any]:
    """One record's resolved slots, for a channel of ``role``.

    Args:
        record: A ``limits.yaml`` record.
        role: The channel's role, ``setpoint`` or ``readback``.

    Returns:
        ``min_value``, ``max_value``, ``max_step``, ``writable`` and
        ``confirm``, each absent slot filled as the schema defines it.
    """
    bounded = record.get("min_value") is not None and record.get("max_value") is not None
    return {
        "min_value": record.get("min_value"),
        "max_value": record.get("max_value"),
        "max_step": record.get("max_step"),
        "writable": record.get("writable", role == "setpoint" and bounded),
        "confirm": record.get("confirm", True),
    }


def golden() -> bytes:
    """The limits golden: every record resolved, keyed by address."""
    channels = {
        record["address"]: resolve(record, "setpoint")
        for _comment, record in sorted(RECORDS, key=lambda item: item[1]["address"])
    }
    document = {
        "_reproduce": f"uv run python {SOURCE} --write {GOLDEN}",
        "_sources": [SOURCE],
        "count": len(channels),
        "channels": channels,
    }
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the demo's limits golden.")
    parser.add_argument("--write", type=Path, required=True, metavar="FILE", help="the golden")
    args = parser.parse_args(argv)
    args.write.write_bytes(golden())
    return 0


if __name__ == "__main__":
    sys.exit(main())
