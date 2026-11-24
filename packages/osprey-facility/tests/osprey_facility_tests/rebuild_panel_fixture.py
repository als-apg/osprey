"""Rebuild the panel tests' checked-in facility document from its fixture tree.

The panel tests read
``packages/osprey-panels/tests/osprey_panels_tests/fixture/facility.json``
instead of building the fixture tree every session. The tree is a description
and its build needs neither the panels nor the framework, so it lives here, in
the package suite, beside ``panel_fixture/``. Run this after a change to the
tree or to the build, so the checked-in document is again what the build
writes::

    uv run python packages/osprey-facility/tests/osprey_facility_tests/rebuild_panel_fixture.py
    uv run python packages/osprey-facility/tests/osprey_facility_tests/rebuild_panel_fixture.py --check

The second form only compares, and exits 1 when the document is stale.

Run as a file, the script imports nothing but ``osprey_facility``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: The command that regenerates the checked-in document.
COMMAND = (
    "uv run python packages/osprey-facility/tests/osprey_facility_tests/rebuild_panel_fixture.py"
)

#: The fixture's ``data/facility`` tree.
FIXTURE_TREE = Path(__file__).resolve().with_name("panel_fixture")
#: The repository checkout the package lives in.
REPO_ROOT = Path(__file__).resolve().parents[4]
#: The fixture tree built into its facility file, checked in beside the panel tests.
FIXTURE_DOCUMENT = (
    REPO_ROOT / "packages" / "osprey-panels" / "tests" / "osprey_panels_tests" / "fixture"
) / "facility.json"
#: The project name the fixture tree is built under.
PROJECT_NAME = "xr"


def rebuilt() -> bytes:
    """The fixture tree built with the description's own checks, as the build writes it."""
    from osprey_facility.build import build_facility
    from osprey_facility.render import facility_bytes

    return facility_bytes(build_facility(FIXTURE_TREE, project_name=PROJECT_NAME, extra_checks=()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="compare only; exit 1 when the document is stale"
    )
    parser.add_argument(
        "--path", type=Path, default=FIXTURE_DOCUMENT, help="the document (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    data = rebuilt()
    path: Path = args.path
    if args.check:
        if not path.is_file():
            print(f"{path} is missing; run {COMMAND}")
            return 1
        if path.read_bytes() != data:
            print(f"{path} is stale; run {COMMAND}")
            return 1
        return 0
    path.write_bytes(data)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
