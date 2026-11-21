"""Rebuild the panel tests' checked-in facility document from its fixture tree.

The panel tests read ``tests/facility/panels/fixture/facility.json`` instead of
building the fixture tree every session. Run this after a change to the tree or
to the build, so the checked-in document is again what the build writes::

    uv run python tests/facility/rebuild_panel_fixture.py          # write it
    uv run python tests/facility/rebuild_panel_fixture.py --check  # exit 1 when stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: The command that regenerates the checked-in document.
COMMAND = "uv run python tests/facility/rebuild_panel_fixture.py"


def rebuilt() -> bytes:
    """The fixture tree built, as the build writes its facility file."""
    from osprey.facility.render import facility_bytes
    from tests.facility.panels._fixture import build_fixture_document

    return facility_bytes(build_fixture_document())


def main(argv: list[str] | None = None) -> int:
    from tests.facility.panels._fixture import FIXTURE_DOCUMENT

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
    # Run as a file, the script's own directory is on sys.path, not the repo root.
    REPO_ROOT = str(Path(__file__).resolve().parents[2])
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    sys.exit(main())
