#!/usr/bin/env python3
"""Generate the example facility's demo sources.

The ``injector`` mode merges the injector's records, built by ``_injector.py``
from the element table, into a facility tree's records and seeds.

The ``peripherals`` mode merges the storage ring's peripheral devices, built by
``_peripherals.py`` (ion pumps, gauges, valves, insertion devices, shutters,
absorber thermocouples, scrapers, tunnel environment, harmonic cavities), into
a facility tree's records, classes and seeds. Run ``scripts/demo_seeds/reseed.py``
after it to stamp their noise and tolerances.

Usage::

    uv run python scripts/facility_demo/generate.py injector src/osprey/templates/facilities/example
    uv run python scripts/facility_demo/generate.py peripherals src/osprey/templates/facilities/example
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _injector
import _peripherals


def main(argv: list[str] | None = None) -> int:
    """Run one generator mode and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    injector = modes.add_parser("injector", help="Write the injector records into a facility tree.")
    injector.add_argument("tree", type=Path, help="The facility tree to write into.")
    peripherals = modes.add_parser(
        "peripherals", help="Write the storage ring's peripheral records into a facility tree."
    )
    peripherals.add_argument("tree", type=Path, help="The facility tree to write into.")
    args = parser.parse_args(argv)

    if args.mode == "injector":
        _injector.write_sources(args.tree)
    elif args.mode == "peripherals":
        _peripherals.write_sources(args.tree)
    return 0


if __name__ == "__main__":
    sys.exit(main())
