# ruff: noqa: T201 -- the image checks report on the build log
"""The CANOPY panel's image checks: what an image build runs before the image exists.

The panel names two commands in the registry
(:attr:`~osprey.profiles.sidecar_panels.SidecarPanel.image_checks`), and
``Dockerfile.j2`` runs each from the deployment directory, on the tree the image
has just copied in::

    python -I -m osprey.integrations.canopy.image_check library
    python -I -m osprey.integrations.canopy.image_check collections

``library`` holds the screen library the image serves to its write-binding
report (``canopy_mcp.report --check``). ``collections`` holds every facility
collection the image serves to CANOPY's own rules, its library screens judged
against that same library (``canopy_mcp.collections_cli``); a deployment
without a collections directory has none to check and says so.

Both run with the CANOPY the image carries. An image without CANOPY serves
neither and has nothing to check: each check says so and passes. Any other
failure, of the probe or of a check (a refused collection included), fails the
build with the check's own exit status. Every skip says so in its own words, so
a build log tells them apart.

``python -I`` keeps the copied deployment tree, which is the working directory,
from shadowing this module or CANOPY; each check runs CANOPY the same way. This
module imports nothing outside the standard library and its package's
stdlib-only names, so it runs in an image whatever else that image lacks.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from osprey.integrations.canopy import FACILITY_COLLECTIONS_SUBPATH, LIBRARY_SUBPATH
from osprey.integrations.canopy.library import DEPLOYMENT_IGNORED_NAMES, IGNORED_NAMES_ENV

#: What each check says when the image carries no CANOPY.
LIBRARY_NOT_INSTALLED = "canopy not installed, library not served"
COLLECTIONS_NOT_INSTALLED = "canopy not installed, collections not served"
#: What the collections check says when the deployment has no collections directory.
NO_COLLECTIONS = "no facility collections, none checked"

#: The CANOPY package both checks run.
CANOPY_PACKAGE = "canopy_mcp"


def canopy_installed() -> bool:
    """Whether this interpreter sees CANOPY's ``canopy_mcp`` package."""
    return importlib.util.find_spec(CANOPY_PACKAGE) is not None


def _run(module: str, *args: str) -> int:
    """Run ``python -I -m <module> <args>`` under this interpreter; its exit status."""
    sys.stdout.flush()
    env = {**os.environ, IGNORED_NAMES_ENV: " ".join(DEPLOYMENT_IGNORED_NAMES)}
    return subprocess.call([sys.executable, "-I", "-m", module, *args], env=env)


def check_library() -> int:
    """The screen library against its write-binding report."""
    if not canopy_installed():
        print(LIBRARY_NOT_INSTALLED)
        return 0
    library = LIBRARY_SUBPATH.as_posix()
    return _run(f"{CANOPY_PACKAGE}.report", library, "--served-at", "library", "--check")


def check_collections() -> int:
    """Every facility collection against CANOPY's rules and the library."""
    if not canopy_installed():
        print(COLLECTIONS_NOT_INSTALLED)
        return 0
    if not Path(FACILITY_COLLECTIONS_SUBPATH).is_dir():
        print(NO_COLLECTIONS)
        return 0
    return _run(
        f"{CANOPY_PACKAGE}.collections_cli",
        FACILITY_COLLECTIONS_SUBPATH.as_posix(),
        "--library",
        LIBRARY_SUBPATH.as_posix(),
    )


CHECKS = {"library": check_library, "collections": check_collections}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the one check *argv* names; its exit status, 2 for a bad call."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in CHECKS:
        sys.stderr.write(
            f"usage: python -I -m osprey.integrations.canopy.image_check {{{'|'.join(CHECKS)}}}\n"
        )
        return 2
    return CHECKS[args[0]]()


if __name__ == "__main__":
    sys.exit(main())
